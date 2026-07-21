# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-size model builders for the Gemma 4 dense text tower.

Values are from the HF ``google/gemma-4-*`` ``text_config``. E4B has Per-Layer Embeddings
and KV-sharing; 31B has neither (``per_layer_dim=0``, ``num_kv_shared_layers=0``).
Multimodal towers are out of scope (text-only fine-tuning); the output is tied to the token
embedding so it is never a LoRA target.
"""
from functools import partial

from torchtune.models.gemma4._component_builders import (
    gemma4,
    Gemma4TextDecoder,
    lora_gemma4,
)
from torchtune.modules.peft import LORA_ATTN_MODULES

# Architecture hyperparameters shared by the component builder (see HF text_config).
_E4B = dict(
    vocab_size=262_144, num_layers=42, num_heads=8, num_kv_heads=2,
    head_dim=256, global_head_dim=512, embed_dim=2560, intermediate_dim=10_240,
    per_layer_dim=256, num_kv_shared_layers=18, sliding_window=512,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
)
_31B = dict(
    vocab_size=262_144, num_layers=60, num_heads=32, num_kv_heads=16,
    head_dim=256, global_head_dim=512, embed_dim=5376, intermediate_dim=21_504,
    per_layer_dim=0, num_kv_shared_layers=0, sliding_window=1024,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
    # Global (full-attention) layers use value==key attention with 4 KV heads.
    num_global_key_value_heads=4, attention_k_eq_v=True,
)


_26B_A4B = dict(
    vocab_size=262_144, num_layers=30, num_heads=16, num_kv_heads=8,
    head_dim=256, global_head_dim=512, embed_dim=2816, intermediate_dim=2112,
    per_layer_dim=0, num_kv_shared_layers=0, sliding_window=1024,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
    num_global_key_value_heads=2, attention_k_eq_v=True,
    # Hybrid MoE on every layer: 128 experts, top-8, GELU, moe intermediate 704.
    enable_moe_block=True, num_experts=128, top_k_experts=8, moe_intermediate_size=704,
)


def gemma4_e4b() -> Gemma4TextDecoder:
    """Gemma 4 E4B text tower (google/gemma-4-e4b-it text_config)."""
    return gemma4(**_E4B)


def gemma4_26b_a4b() -> Gemma4TextDecoder:
    """Gemma 4 26B-A4B MoE text tower (google/gemma-4-26b-a4b-it text_config)."""
    return gemma4(**_26B_A4B)


def gemma4_31b() -> Gemma4TextDecoder:
    """Gemma 4 31B text tower (google/gemma-4-31B-it text_config); no PLE, no KV-share."""
    return gemma4(**_31B)


def _lora(base: dict, lora_attn_modules, apply_lora_to_mlp, lora_rank, lora_alpha,
          lora_dropout, use_dora, quantize_base) -> Gemma4TextDecoder:
    return lora_gemma4(
        lora_attn_modules, apply_lora_to_mlp,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        use_dora=use_dora, quantize_base=quantize_base, **base,
    )


def lora_gemma4_e4b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> Gemma4TextDecoder:
    """Gemma 4 E4B text tower with LoRA (output tied to embedding, never adapted)."""
    return _lora(_E4B, lora_attn_modules, apply_lora_to_mlp, lora_rank, lora_alpha,
                 lora_dropout, use_dora, quantize_base)


def lora_gemma4_31b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> Gemma4TextDecoder:
    """Gemma 4 31B text tower with LoRA."""
    return _lora(_31B, lora_attn_modules, apply_lora_to_mlp, lora_rank, lora_alpha,
                 lora_dropout, use_dora, quantize_base)


def lora_gemma4_26b_a4b(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> Gemma4TextDecoder:
    """Gemma 4 26B-A4B MoE text tower with LoRA (adapts attention; experts stay frozen)."""
    return _lora(_26B_A4B, lora_attn_modules, apply_lora_to_mlp, lora_rank, lora_alpha,
                 lora_dropout, use_dora, quantize_base)


qlora_gemma4_e4b = partial(lora_gemma4_e4b, quantize_base=True)
qlora_gemma4_e4b.__doc__ = "Gemma 4 E4B with QLoRA (NF4-quantized base weights)."
qlora_gemma4_31b = partial(lora_gemma4_31b, quantize_base=True)
qlora_gemma4_31b.__doc__ = "Gemma 4 31B with QLoRA (NF4-quantized base weights)."
qlora_gemma4_26b_a4b = partial(lora_gemma4_26b_a4b, quantize_base=True)
qlora_gemma4_26b_a4b.__doc__ = "Gemma 4 26B-A4B MoE with QLoRA (NF4-quantized base weights)."
