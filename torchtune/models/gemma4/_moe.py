# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture-of-Experts router and experts for Gemma 4 (26B-A4B).

Distilled from HF ``Gemma4TextRouter`` / ``Gemma4TextExperts``. The router uses
**softmax** gating with top-k weight normalization and a per-expert scale (unlike
torchtune's sigmoid ``TokenChoiceTopKRouter``), and the experts are gated GELU FFNs stored
as stacked 3D parameters. In Gemma 4 the MoE is a *hybrid* block: its output is added to a
dense MLP output in the decoder layer (see ``_component_builders.Gemma4DecoderLayer``).
"""
import torch
import torch.nn.functional as F
from torch import nn

from torchtune.models.gemma4._attention import Gemma4RMSNorm


class Gemma4Router(nn.Module):
    """Softmax top-k router with per-expert scaling.

    Args:
        hidden_size (int): model hidden size.
        num_experts (int): total number of experts.
        top_k (int): experts routed to per token.
        norm_eps (float): RMSNorm epsilon.
    """

    def __init__(self, *, hidden_size: int, num_experts: int, top_k: int, norm_eps: float):
        super().__init__()
        self.top_k = top_k
        self.scalar_root_size = hidden_size**-0.5
        self.norm = Gemma4RMSNorm(hidden_size, eps=norm_eps, with_scale=False)
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: ``[T, hidden]`` -> (top_k_weights ``[T, k]``, top_k_index ``[T, k]``)."""
        h = self.norm(x) * self.scale * self.scalar_root_size
        probs = F.softmax(self.proj(h), dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


class Gemma4Experts(nn.Module):
    """Gated GELU experts stored as stacked 3D parameters (num_experts, ...).

    Args:
        num_experts (int): number of experts.
        hidden_size (int): model hidden size.
        moe_intermediate_size (int): per-expert intermediate size.
    """

    def __init__(self, *, num_experts: int, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        # Initialized (not torch.empty) so a freshly-built model is finite before any
        # checkpoint load; overwritten when weights are loaded.
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size).normal_(std=0.02)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size).normal_(std=0.02)
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(
        self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """x: ``[T, hidden]``; routes each token to its top-k experts and sums the
        weighted expert outputs. Loops only over experts that were selected."""
        out = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current = x[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[e]).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = F.linear(current, self.down_proj[e])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            out.index_add_(0, token_idx, current.to(out.dtype))
        return out
