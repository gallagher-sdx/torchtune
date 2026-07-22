# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF <-> torchtune weight conversion for the Gemma 4 text tower.

Because we implement RoPE with HF's half-split convention (see ``_attention.py``), the
mapping is a pure rename — no q/k permutation. The text tower lives under
``model.language_model.*`` in the multimodal ``Gemma4ForConditionalGeneration`` checkpoint;
vision/audio keys are ignored (text-only scope). ``lm_head.weight`` is tied to the token
embedding and skipped. KV-shared layers legitimately have no k/v projection or norm keys.
"""
import re

import torch

from torchtune.models.convert_weights import get_mapped_key

_P = "model.language_model."

_GEMMA4_FROM_HF = {
    f"{_P}embed_tokens.weight": "tok_embeddings.weight",
    f"{_P}embed_tokens_per_layer.weight": "per_layer_embeddings.weight",
    f"{_P}per_layer_model_projection.weight": "per_layer_model_projection.weight",
    f"{_P}per_layer_projection_norm.weight": "per_layer_projection_norm.scale",
    f"{_P}norm.weight": "norm.scale",
    f"{_P}layers.{{}}.self_attn.q_proj.weight": "layers.{}.self_attn.q_proj.weight",
    f"{_P}layers.{{}}.self_attn.k_proj.weight": "layers.{}.self_attn.k_proj.weight",
    f"{_P}layers.{{}}.self_attn.v_proj.weight": "layers.{}.self_attn.v_proj.weight",
    f"{_P}layers.{{}}.self_attn.o_proj.weight": "layers.{}.self_attn.o_proj.weight",
    f"{_P}layers.{{}}.self_attn.q_norm.weight": "layers.{}.self_attn.q_norm.scale",
    f"{_P}layers.{{}}.self_attn.k_norm.weight": "layers.{}.self_attn.k_norm.scale",
    f"{_P}layers.{{}}.mlp.gate_proj.weight": "layers.{}.mlp.gate_proj.weight",
    f"{_P}layers.{{}}.mlp.up_proj.weight": "layers.{}.mlp.up_proj.weight",
    f"{_P}layers.{{}}.mlp.down_proj.weight": "layers.{}.mlp.down_proj.weight",
    f"{_P}layers.{{}}.input_layernorm.weight": "layers.{}.input_layernorm.scale",
    f"{_P}layers.{{}}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.scale",
    f"{_P}layers.{{}}.pre_feedforward_layernorm.weight": "layers.{}.pre_feedforward_layernorm.scale",
    f"{_P}layers.{{}}.post_feedforward_layernorm.weight": "layers.{}.post_feedforward_layernorm.scale",
    # MoE (26B-A4B): router + stacked experts + the hybrid block's extra norms. Absent for
    # dense models (E4B / 31B), so these entries are simply unused there.
    f"{_P}layers.{{}}.router.proj.weight": "layers.{}.router.proj.weight",
    f"{_P}layers.{{}}.router.scale": "layers.{}.router.scale",
    f"{_P}layers.{{}}.router.per_expert_scale": "layers.{}.router.per_expert_scale",
    f"{_P}layers.{{}}.experts.gate_up_proj": "layers.{}.experts.gate_up_proj",
    f"{_P}layers.{{}}.experts.down_proj": "layers.{}.experts.down_proj",
    f"{_P}layers.{{}}.post_feedforward_layernorm_1.weight": "layers.{}.post_feedforward_layernorm_1.scale",
    f"{_P}layers.{{}}.post_feedforward_layernorm_2.weight": "layers.{}.post_feedforward_layernorm_2.scale",
    f"{_P}layers.{{}}.pre_feedforward_layernorm_2.weight": "layers.{}.pre_feedforward_layernorm_2.scale",
    f"{_P}layers.{{}}.per_layer_input_gate.weight": "layers.{}.per_layer_input_gate.weight",
    f"{_P}layers.{{}}.per_layer_projection.weight": "layers.{}.per_layer_projection.weight",
    f"{_P}layers.{{}}.post_per_layer_input_norm.weight": "layers.{}.post_per_layer_input_norm.scale",
    f"{_P}layers.{{}}.layer_scalar": "layers.{}.layer_scalar",
}


def _is_text_key(key: str) -> bool:
    return key.startswith(_P)


def gemma4_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    *,
    num_hidden_layers: int | None = None,
    num_kv_shared_layers: int = 0,
) -> dict[str, torch.Tensor]:
    """Convert an HF Gemma 4 state dict to torchtune's text-tower format.

    Ignores the vision/audio towers and the tied ``lm_head``. The last
    ``num_kv_shared_layers`` layers reuse an earlier layer's K/V, so their k/v projection
    and k/v norm weights (kept on disk by HF, but unused) are dropped to match the model —
    otherwise they surface as unexpected keys during (LoRA) loading.

    Args:
        state_dict (dict[str, torch.Tensor]): HF state dict.
        num_hidden_layers (int | None): number of text decoder layers. Required (with
            ``num_kv_shared_layers``) to drop shared-layer k/v; if None, nothing is dropped.
        num_kv_shared_layers (int): number of trailing KV-shared layers. Default 0.
    """
    first_shared = (
        num_hidden_layers - num_kv_shared_layers
        if (num_hidden_layers is not None and num_kv_shared_layers)
        else None
    )
    converted = {}
    for key, value in state_dict.items():
        if not _is_text_key(key):
            continue  # skip vision_tower / audio_tower / embed_vision / embed_audio / lm_head
        if first_shared is not None and (
            ".k_proj" in key or ".v_proj" in key or ".k_norm" in key
        ):
            m = re.search(r"\.layers\.(\d+)\.", key)
            if m and int(m.group(1)) >= first_shared:
                continue  # KV-shared layer: no k/v projection or norm in the model
        converted[get_mapped_key(key, _GEMMA4_FROM_HF)] = value
    return converted


def gemma4_tune_to_hf(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a torchtune Gemma 4 text-tower state dict back to HF names."""
    inverted = {v: k for k, v in _GEMMA4_FROM_HF.items()}
    return {get_mapped_key(key, inverted): value for key, value in state_dict.items()}
