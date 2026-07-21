# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Component builders for the Gemma 4 text tower (dense; E4B / 31B).

The Gemma 4 decoder cannot reuse ``torchtune.modules.TransformerDecoder`` because it
threads two extra per-layer signals through the layer loop — an optional Per-Layer
Embedding (PLE) residual and an optional shared-KV dict — and it assigns a different
``head_dim``, RoPE, and mask type to sliding vs. global layers. We therefore provide a
small purpose-built :class:`Gemma4TextDecoder`.

Both features are optional and config-driven: E4B has PLE + KV-sharing; 31B has neither
(``hidden_size_per_layer_input == 0`` and ``num_kv_shared_layers == 0``), reducing to a
vanilla hybrid-attention dense decoder.

A single assembly path serves base / LoRA / QLoRA: attention and MLP projections are built
by :func:`_make_linear`, which returns ``nn.Linear``, ``FrozenNF4Linear``, ``LoRALinear``,
or ``DoRALinear`` per the LoRA config. Norms, embeddings, and the PLE gate/projection are
never LoRA'd. MoE (26B) and multimodal towers are out of scope here.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchtune.models.gemma4._attention import (
    Gemma4Attention,
    Gemma4RMSNorm,
    Gemma4RotaryEmbedding,
)
from torchtune.modules.low_precision.nf4_linear import FrozenNF4Linear
from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear


def _make_linear(
    in_dim: int,
    out_dim: int,
    *,
    lora: bool,
    rank: int = 0,
    alpha: float = 0.0,
    dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> nn.Module:
    """Build a projection: LoRA/DoRA adapter if ``lora``, else NF4-frozen or plain linear."""
    if lora:
        cls = DoRALinear if use_dora else LoRALinear
        return cls(in_dim, out_dim, rank=rank, alpha=alpha, dropout=dropout,
                   quantize_base=quantize_base)
    if quantize_base:
        return FrozenNF4Linear(in_dim, out_dim, bias=False)
    return nn.Linear(in_dim, out_dim, bias=False)


class Gemma4ScaledEmbedding(nn.Embedding):
    """Embedding whose output is scaled by a fixed ``embed_scale`` (Gemma normalizer)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * torch.tensor(self.embed_scale, dtype=self.weight.dtype)


class Gemma4MLP(nn.Module):
    """Gated MLP with gelu(tanh): ``down(gelu(gate(x)) * up(x))`` (injected projections)."""

    def __init__(self, *, gate_proj: nn.Module, up_proj: nn.Module, down_proj: nn.Module):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4DecoderLayer(nn.Module):
    """A single Gemma 4 decoder layer: sandwich-normed attention + MLP, then optional PLE.

    forward (see HF ``Gemma4TextDecoderLayer``):
      1. ``h += post_attn_norm(attn(input_norm(h)))``
      2. ``h += post_ff_norm(mlp(pre_ff_norm(h)))``
      3. if PLE: ``h += post_ple_norm(per_layer_projection(gelu(gate(h)) * ple))``
      4. ``h *= layer_scalar``
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_dim: int,
        per_layer_dim: int,
        norm_eps: float,
        sliding_window: Optional[int],
        is_kv_shared: bool,
        store_full_length_kv: bool,
        layer_type: str,
        lora_attn_modules: list,
        apply_lora_to_mlp: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        use_dora: bool,
        quantize_base: bool,
    ):
        super().__init__()
        self.has_ple = per_layer_dim > 0

        def proj(name, in_dim, out_dim, lora):
            return _make_linear(in_dim, out_dim, lora=lora, rank=lora_rank,
                                alpha=lora_alpha, dropout=lora_dropout,
                                use_dora=use_dora, quantize_base=quantize_base)

        q_proj = proj("q_proj", embed_dim, num_heads * head_dim, "q_proj" in lora_attn_modules)
        o_proj = proj("output_proj", num_heads * head_dim, embed_dim, "output_proj" in lora_attn_modules)
        if is_kv_shared:
            k_proj = v_proj = None
        else:
            k_proj = proj("k_proj", embed_dim, num_kv_heads * head_dim, "k_proj" in lora_attn_modules)
            v_proj = proj("v_proj", embed_dim, num_kv_heads * head_dim, "v_proj" in lora_attn_modules)
        self.self_attn = Gemma4Attention(
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            norm_eps=norm_eps, q_proj=q_proj, o_proj=o_proj, k_proj=k_proj, v_proj=v_proj,
            sliding_window=sliding_window, is_kv_shared=is_kv_shared,
            store_full_length_kv=store_full_length_kv, layer_type=layer_type,
        )
        self.mlp = Gemma4MLP(
            gate_proj=proj("gate", embed_dim, intermediate_dim, apply_lora_to_mlp),
            up_proj=proj("up", embed_dim, intermediate_dim, apply_lora_to_mlp),
            down_proj=proj("down", intermediate_dim, embed_dim, apply_lora_to_mlp),
        )
        self.input_layernorm = Gemma4RMSNorm(embed_dim, eps=norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(embed_dim, eps=norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(embed_dim, eps=norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(embed_dim, eps=norm_eps)
        if self.has_ple:
            self.act_fn = nn.GELU(approximate="tanh")
            self.per_layer_input_gate = nn.Linear(embed_dim, per_layer_dim, bias=False)
            self.per_layer_projection = nn.Linear(per_layer_dim, embed_dim, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(embed_dim, eps=norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        h: torch.Tensor,
        *,
        per_layer_input: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        shared_kv: dict,
    ) -> torch.Tensor:
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(h, cos=cos, sin=sin, mask=mask, shared_kv=shared_kv)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h

        if self.has_ple:
            residual = h
            h = self.per_layer_input_gate(h)
            h = self.act_fn(h)
            h = h * per_layer_input
            h = self.per_layer_projection(h)
            h = self.post_per_layer_input_norm(h)
            h = residual + h

        return h * self.layer_scalar


class Gemma4TextDecoder(nn.Module):
    """Gemma 4 dense text decoder (E4B / 31B), training-forward focused.

    Owns the token (+ optional Per-Layer) embeddings, per-layer-type RoPE caches, the
    decoder layers (threading the optional PLE residual and shared-KV dict), the final
    norm, and the tied output projection with final-logit soft-capping.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        global_head_dim: int,
        intermediate_dim: int,
        per_layer_dim: int,
        num_kv_shared_layers: int,
        sliding_window: int,
        norm_eps: float,
        final_logit_softcapping: float,
        rope_base_sliding: float,
        rope_base_global: float,
        global_partial_rotary_factor: float,
        global_every: int,
        lora_attn_modules: Optional[list] = None,
        apply_lora_to_mlp: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 0.0,
        lora_dropout: float = 0.0,
        use_dora: bool = False,
        quantize_base: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.per_layer_dim = per_layer_dim
        self.has_ple = per_layer_dim > 0
        self.num_layers = num_layers
        self.final_logit_softcapping = final_logit_softcapping
        lora_attn_modules = lora_attn_modules or []

        self.layer_types = [
            "full_attention" if (i + 1) % global_every == 0 else "sliding_attention"
            for i in range(num_layers)
        ]
        first_shared = num_layers - num_kv_shared_layers
        prev = self.layer_types[:first_shared]
        store_idx = {
            lt: (first_shared - 1 - prev[::-1].index(lt)) if lt in prev else -1
            for lt in set(self.layer_types)
        }

        self.tok_embeddings = Gemma4ScaledEmbedding(
            vocab_size, embed_dim, embed_scale=embed_dim**0.5
        )
        if self.has_ple:
            self.per_layer_embeddings = Gemma4ScaledEmbedding(
                vocab_size, num_layers * per_layer_dim, embed_scale=per_layer_dim**0.5
            )
            self.per_layer_model_projection = nn.Linear(
                embed_dim, num_layers * per_layer_dim, bias=False
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(per_layer_dim, eps=norm_eps)

        self.rope = nn.ModuleDict(
            {
                "sliding_attention": Gemma4RotaryEmbedding(head_dim, rope_base_sliding),
                "full_attention": Gemma4RotaryEmbedding(
                    global_head_dim, rope_base_global, global_partial_rotary_factor
                ),
            }
        )

        layers = []
        for i in range(num_layers):
            lt = self.layer_types[i]
            layers.append(
                Gemma4DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=global_head_dim if lt == "full_attention" else head_dim,
                    intermediate_dim=intermediate_dim,
                    per_layer_dim=per_layer_dim,
                    norm_eps=norm_eps,
                    sliding_window=sliding_window if lt == "sliding_attention" else None,
                    is_kv_shared=(i >= first_shared),
                    store_full_length_kv=(store_idx[lt] == i),
                    layer_type=lt,
                    lora_attn_modules=lora_attn_modules,
                    apply_lora_to_mlp=apply_lora_to_mlp,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    use_dora=use_dora,
                    quantize_base=quantize_base,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = Gemma4RMSNorm(embed_dim, eps=norm_eps)

    def _per_layer_inputs(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape
        identity = self.per_layer_embeddings(tokens).reshape(
            b, s, self.num_layers, self.per_layer_dim
        )
        proj = self.per_layer_model_projection(h) * (self.embed_dim**-0.5)
        proj = proj.reshape(b, s, self.num_layers, self.per_layer_dim)
        proj = self.per_layer_projection_norm(proj)
        return (proj + identity) * (2.0**-0.5)

    def _masks(self, seq_len: int, device) -> dict:
        idx = torch.arange(seq_len, device=device)
        causal = idx[:, None] >= idx[None, :]
        masks = {"full_attention": causal[None, None]}
        if "sliding_attention" in set(self.layer_types):
            window = self.layers[
                self.layer_types.index("sliding_attention")
            ].self_attn.sliding_window
            in_window = (idx[:, None] - idx[None, :]) < window
            masks["sliding_attention"] = (causal & in_window)[None, None]
        return masks

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s = tokens.shape
        h = self.tok_embeddings(tokens)
        per_layer_inputs = self._per_layer_inputs(tokens, h) if self.has_ple else None

        position_ids = (
            input_pos
            if input_pos is not None
            else torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, -1)
        )
        cos_sin = {lt: rope(position_ids) for lt, rope in self.rope.items()}
        masks = self._masks(s, tokens.device)

        shared_kv: dict = {}
        for i, layer in enumerate(self.layers):
            lt = self.layer_types[i]
            cos, sin = cos_sin[lt]
            h = layer(
                h,
                per_layer_input=per_layer_inputs[:, :, i, :] if self.has_ple else None,
                cos=cos,
                sin=sin,
                mask=masks[lt],
                shared_kv=shared_kv,
            )

        h = self.norm(h)
        logits = F.linear(h, self.tok_embeddings.weight).float()
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits


def gemma4(**kwargs) -> Gemma4TextDecoder:
    """Build a base (non-LoRA) Gemma 4 dense text decoder from hyperparameters."""
    return Gemma4TextDecoder(**kwargs)


def lora_gemma4(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    *,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    **kwargs,
) -> Gemma4TextDecoder:
    """Build a Gemma 4 dense text decoder with LoRA/QLoRA on the selected projections.

    Note: the output (unembedding) is tied to the token embedding and is therefore never
    a LoRA target — only ``q_proj``/``k_proj``/``v_proj``/``output_proj`` (attention) and,
    with ``apply_lora_to_mlp``, the MLP projections are adapted.
    """
    model = Gemma4TextDecoder(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        **kwargs,
    )
    if quantize_base:
        from torchtune.modules.common_utils import (
            _register_reparametrize_state_dict_hooks,
        )

        _register_reparametrize_state_dict_hooks(
            model, dtype=model.tok_embeddings.weight.dtype
        )
    return model
