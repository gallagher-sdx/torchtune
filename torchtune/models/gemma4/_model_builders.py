# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-size model builders for the Gemma 4 dense/MoE text towers.

Values are from the HF ``google/gemma-4-*`` ``text_config``. Multimodal towers are out of
scope (text-only); the output is tied to the token embedding so it is never a LoRA target.

Every builder forwards ``**kwargs`` to the component builder, so config-time knobs such as
``activation_checkpointing`` (self-contained gradient checkpointing, recommended for full
fine-tunes) can be passed through, e.g. ``model.activation_checkpointing: True``.
"""
from functools import partial

from torchtune.models.gemma4._component_builders import (
    gemma4,
    Gemma4TextDecoder,
    lora_gemma4,
)
from torchtune.modules.peft import LORA_ATTN_MODULES

# --- architecture hyperparameters (HF text_config) ---
_E2B = dict(
    vocab_size=262_144, num_layers=35, num_heads=8, num_kv_heads=1,
    head_dim=256, global_head_dim=512, embed_dim=1536, intermediate_dim=6144,
    per_layer_dim=256, num_kv_shared_layers=20, sliding_window=512,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
)
_E4B = dict(
    vocab_size=262_144, num_layers=42, num_heads=8, num_kv_heads=2,
    head_dim=256, global_head_dim=512, embed_dim=2560, intermediate_dim=10_240,
    per_layer_dim=256, num_kv_shared_layers=18, sliding_window=512,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
)
_12B = dict(
    vocab_size=262_144, num_layers=48, num_heads=16, num_kv_heads=8,
    head_dim=256, global_head_dim=512, embed_dim=3840, intermediate_dim=15_360,
    per_layer_dim=0, num_kv_shared_layers=0, sliding_window=1024,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
    num_global_key_value_heads=1, attention_k_eq_v=True,
)
_31B = dict(
    vocab_size=262_144, num_layers=60, num_heads=32, num_kv_heads=16,
    head_dim=256, global_head_dim=512, embed_dim=5376, intermediate_dim=21_504,
    per_layer_dim=0, num_kv_shared_layers=0, sliding_window=1024,
    norm_eps=1e-6, final_logit_softcapping=30.0,
    rope_base_sliding=10_000.0, rope_base_global=1_000_000.0,
    global_partial_rotary_factor=0.25, global_every=6,
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
    enable_moe_block=True, num_experts=128, top_k_experts=8, moe_intermediate_size=704,
)

_ARGS = {"e2b": _E2B, "e4b": _E4B, "12b": _12B, "31b": _31B, "26b_a4b": _26B_A4B}


def _base(size: str, **kwargs) -> Gemma4TextDecoder:
    return gemma4(**{**_ARGS[size], **kwargs})


def _lora(size, lora_attn_modules, apply_lora_to_mlp, lora_rank, lora_alpha,
          lora_dropout, use_dora, quantize_base, **kwargs) -> Gemma4TextDecoder:
    return lora_gemma4(
        lora_attn_modules, apply_lora_to_mlp,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        use_dora=use_dora, quantize_base=quantize_base, **_ARGS[size], **kwargs,
    )


def _make_lora_builder(size):
    def builder(
        lora_attn_modules: list[LORA_ATTN_MODULES],
        apply_lora_to_mlp: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        use_dora: bool = False,
        quantize_base: bool = False,
        **kwargs,
    ) -> Gemma4TextDecoder:
        return _lora(size, lora_attn_modules, apply_lora_to_mlp, lora_rank, lora_alpha,
                     lora_dropout, use_dora, quantize_base, **kwargs)
    return builder


# Plain (full fine-tune) builders — forward **kwargs (e.g. activation_checkpointing).
def gemma4_e2b(**kwargs) -> Gemma4TextDecoder:
    """Gemma 4 E2B text tower (PLE + KV-share; MatFormer sibling of E4B)."""
    return _base("e2b", **kwargs)


def gemma4_e4b(**kwargs) -> Gemma4TextDecoder:
    """Gemma 4 E4B text tower (PLE + KV-share)."""
    return _base("e4b", **kwargs)


def gemma4_12b(**kwargs) -> Gemma4TextDecoder:
    """Gemma 4 12B 'Unified' text tower (dense; value==key global attention)."""
    return _base("12b", **kwargs)


def gemma4_31b(**kwargs) -> Gemma4TextDecoder:
    """Gemma 4 31B text tower (dense; value==key global attention)."""
    return _base("31b", **kwargs)


def gemma4_26b_a4b(**kwargs) -> Gemma4TextDecoder:
    """Gemma 4 26B-A4B MoE text tower (128 experts, top-8)."""
    return _base("26b_a4b", **kwargs)


# LoRA builders (attention adapted; MoE experts stay frozen; output is tied, never adapted).
lora_gemma4_e2b = _make_lora_builder("e2b")
lora_gemma4_e4b = _make_lora_builder("e4b")
lora_gemma4_12b = _make_lora_builder("12b")
lora_gemma4_31b = _make_lora_builder("31b")
lora_gemma4_26b_a4b = _make_lora_builder("26b_a4b")
for _n in ("e2b", "e4b", "12b", "31b", "26b_a4b"):
    globals()[f"lora_gemma4_{_n}"].__name__ = f"lora_gemma4_{_n}"

qlora_gemma4_e2b = partial(lora_gemma4_e2b, quantize_base=True)
qlora_gemma4_e4b = partial(lora_gemma4_e4b, quantize_base=True)
qlora_gemma4_12b = partial(lora_gemma4_12b, quantize_base=True)
qlora_gemma4_31b = partial(lora_gemma4_31b, quantize_base=True)
qlora_gemma4_26b_a4b = partial(lora_gemma4_26b_a4b, quantize_base=True)
for _n in ("e2b", "e4b", "12b", "31b", "26b_a4b"):
    globals()[f"qlora_gemma4_{_n}"].__doc__ = (
        f"Gemma 4 {_n} with QLoRA (NF4-quantized base weights)."
    )
