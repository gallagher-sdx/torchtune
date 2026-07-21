# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tests.test_utils import assert_expected, fixed_init_model

from torchtune.models.gemma4 import (
    gemma4,
    gemma4_hf_to_tune,
    gemma4_tune_to_hf,
    lora_gemma4,
)
from torchtune.modules.peft import get_adapter_params, LoRALinear

# Tiny config exercising both a global (head_dim 512-style) and sliding layer, PLE, and
# KV-sharing. global_every=3 -> globals at layer idx {2, 5}; num_kv_shared=2 -> layers
# {6, 7} reuse KV.
TINY = dict(
    vocab_size=100, num_layers=8, num_heads=2, num_kv_heads=1, head_dim=4,
    global_head_dim=8, embed_dim=16, intermediate_dim=32, per_layer_dim=8,
    num_kv_shared_layers=2, sliding_window=4, norm_eps=1e-6,
    final_logit_softcapping=30.0, rope_base_sliding=1e4, rope_base_global=1e6,
    global_partial_rotary_factor=0.25, global_every=3,
)
TOKENS = (torch.arange(1, 6).unsqueeze(0)) % 100


class TestGemma4:
    def test_forward_with_ple(self):
        model = gemma4(**TINY)
        fixed_init_model(model, min_val=-0.1, max_val=0.1)
        model.eval()
        with torch.no_grad():
            out = model(TOKENS)
        assert out.shape == (1, 5, 100)
        assert_expected(out.mean(), torch.tensor(-0.000714), atol=1e-4, rtol=1e-4)
        # final-logit soft-capping bounds every logit to [-cap, cap].
        assert out.abs().max().item() <= 30.0 + 1e-4

    def test_forward_no_ple_no_kv_share(self):
        """31B-shape: hidden_size_per_layer_input=0, num_kv_shared_layers=0."""
        cfg = {**TINY, "per_layer_dim": 0, "num_kv_shared_layers": 0}
        model = gemma4(**cfg)
        assert model.has_ple is False
        assert not any(layer.self_attn.is_kv_shared for layer in model.layers)
        fixed_init_model(model, min_val=-0.1, max_val=0.1)
        model.eval()
        with torch.no_grad():
            out = model(TOKENS)
        assert_expected(out.mean(), torch.tensor(-0.000695), atol=1e-4, rtol=1e-4)

    def test_kv_sharing_structure(self):
        model = gemma4(**TINY)
        shared = [i for i, l in enumerate(model.layers) if l.self_attn.is_kv_shared]
        assert shared == [6, 7]
        # shared layers drop their k/v projections
        for i in shared:
            assert not hasattr(model.layers[i].self_attn, "k_proj")
        # global layers use the larger head_dim
        assert model.layers[2].self_attn.head_dim == 8
        assert model.layers[0].self_attn.head_dim == 4

    def test_lora_adapters_and_tied_output(self):
        model = lora_gemma4(
            ["q_proj", "v_proj", "output_proj"], apply_lora_to_mlp=True,
            lora_rank=2, lora_alpha=4, **TINY,
        )
        adapters = get_adapter_params(model)
        assert len(adapters) > 0
        assert any(isinstance(m, LoRALinear) for m in model.modules())
        # output is tied to the token embedding -> never a LoRA target
        assert not any("output.lora" in k or "lm_head" in k for k in adapters)
        fixed_init_model(model, min_val=-0.05, max_val=0.05)
        model.eval()
        with torch.no_grad():
            assert model(TOKENS).shape == (1, 5, 100)

    def test_weight_conversion_roundtrip(self):
        model = gemma4(**TINY)
        tune_sd = model.state_dict()
        back = gemma4_hf_to_tune(gemma4_tune_to_hf(tune_sd))
        # rope buffers are non-persistent (absent from state_dict); everything else must
        # survive tune -> hf -> tune unchanged in name and shape.
        assert set(back.keys()) == set(tune_sd.keys())
        for k in tune_sd:
            assert back[k].shape == tune_sd[k].shape
