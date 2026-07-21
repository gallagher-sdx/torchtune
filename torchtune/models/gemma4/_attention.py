# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention and rotary embeddings for the Gemma 4 text tower.

Gemma 4 needs a dedicated attention module (stock ``MultiHeadAttention`` does not fit)
because of four features, distilled from HF ``modeling_gemma4.py``:

* per-layer ``head_dim`` — sliding layers use ``head_dim`` (256), global/full layers use
  ``global_head_dim`` (512);
* QK-and-V RMS norm — q/k are normed with a learned scale, v with an unscaled RMS norm;
* attention ``scaling = 1.0`` (NOT ``1/sqrt(head_dim)``) — the q-norm handles magnitude;
* KV-cache sharing — the last ``num_kv_shared_layers`` reuse K/V computed by the last
  non-shared layer of the same type, so shared layers have no k/v projections at all.

RoPE follows the HF half-split (``rotate_half``) convention so weight conversion is a pure
rename (no permutation). Global layers use "proportional" RoPE with a partial rotary
factor, which this module reproduces by zero-padding the inverse frequencies.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm with an optional learned scale.

    Matches HF ``Gemma4RMSNorm`` exactly: normalize in fp32, then (if ``with_scale``)
    multiply by ``weight`` — where ``weight`` is centered at **1.0**, i.e. ``normed *
    weight`` (NOT the Gemma1/2 ``normed * (1 + weight)`` convention). ``with_scale=False``
    is a parameterless norm, used for Gemma 4's value normalization (``v_norm``). The
    learned parameter is named ``scale`` to match the ``torchtune.models.gemma`` naming.

    Args:
        dim (int): normalized dimension (last axis).
        eps (float): numerical epsilon. Default 1e-6.
        with_scale (bool): whether to learn a per-element scale. Default True.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.pow(., -0.5) (not rsqrt) to match HF's compiler-portability choice.
        xf = x.float()
        output = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            output = output * self.scale.float()
        return output.type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation used by HF/Gemma RoPE (splits the last dim in two halves)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``[b, s, n_heads, head_dim]``.

    ``cos``/``sin`` have shape ``[b, s, head_dim]`` and broadcast over the head axis.
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma4RotaryEmbedding(nn.Module):
    """RoPE cache for one layer type.

    A single formula covers both Gemma 4 layer types via ``partial_rotary_factor``:
    ``rope_angles = int(partial_rotary_factor * head_dim // 2)`` inverse frequencies are
    "real" and the remaining ``head_dim // 2 - rope_angles`` are zero (identity / NoPE).
    ``partial_rotary_factor == 1.0`` recovers standard full-rotary RoPE.

    Args:
        head_dim (int): dimension of each attention head.
        base (float): RoPE theta.
        partial_rotary_factor (float): fraction of the head that is rotated. Default 1.0.
    """

    def __init__(self, head_dim: int, base: float, partial_rotary_factor: float = 1.0):
        super().__init__()
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles)], dim=0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [b, s] -> cos/sin: [b, s, head_dim] (computed in fp32)
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)  # [b, s, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [b, s, head_dim]
        return emb.cos(), emb.sin()


class Gemma4Attention(nn.Module):
    """Gemma 4 self-attention with per-layer head_dim, QKV-norm, and KV sharing.

    Args:
        embed_dim (int): model hidden size.
        num_heads (int): number of query heads.
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-head dimension for THIS layer (256 sliding / 512 global).
        norm_eps (float): RMSNorm epsilon.
        sliding_window (Optional[int]): sliding-window size for sliding layers, else None.
        is_kv_shared (bool): if True, this layer has no k/v projections and reuses
            ``shared_kv`` provided at forward time.
        store_full_length_kv (bool): if True, this (non-shared) layer writes its K/V into
            the shared dict for later shared layers of the same type.
        layer_type (str): ``"sliding_attention"`` or ``"full_attention"`` (the shared-KV key).
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        q_proj: nn.Module,
        o_proj: nn.Module,
        k_proj: Optional[nn.Module],
        v_proj: Optional[nn.Module],
        sliding_window: Optional[int],
        is_kv_shared: bool,
        store_full_length_kv: bool,
        layer_type: str,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.is_kv_shared = is_kv_shared
        self.store_full_length_kv = store_full_length_kv
        self.layer_type = layer_type

        # Projections are injected by the builder (plain / LoRA / NF4); norms are never
        # LoRA'd and are built here. Shared-KV layers have no k/v projections or norms.
        self.q_proj = q_proj
        self.o_proj = o_proj
        self.q_norm = Gemma4RMSNorm(head_dim, eps=norm_eps)
        if not is_kv_shared:
            self.k_proj = k_proj
            self.v_proj = v_proj
            self.k_norm = Gemma4RMSNorm(head_dim, eps=norm_eps)
            self.v_norm = Gemma4RMSNorm(head_dim, eps=norm_eps, with_scale=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)  # [b, nh, s, hd]

        if self.is_kv_shared:
            k, v = shared_kv[self.layer_type]
        else:
            k_raw = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k_raw)
            k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2)  # [b, nkv, s, hd]
            if self.v_proj is not None:
                v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
            else:
                # attention_k_eq_v: value reuses the key projection (pre-norm, pre-RoPE),
                # with its own (unscaled) v_norm. Used by global layers in e.g. Gemma 4 31B.
                v = k_raw
            v = self.v_norm(v).transpose(1, 2)
            if self.store_full_length_kv:
                shared_kv[self.layer_type] = (k, v)

        # Expand KV heads to match query heads (GQA).
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Gemma 4 uses attention scaling of 1.0 (q-norm handles magnitude).
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0
        )
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)
