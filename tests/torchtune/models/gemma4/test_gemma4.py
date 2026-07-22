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

    def test_k_eq_v_global_attention(self):
        """31B-style: global layers use value==key with a distinct KV-head count."""
        cfg = {**TINY, "per_layer_dim": 0, "num_kv_shared_layers": 0,
               "num_global_key_value_heads": 1, "attention_k_eq_v": True}
        model = gemma4(**cfg)
        # global layers (idx 2, 5) drop v_proj and use the global KV-head count
        assert model.layers[2].self_attn.v_proj is None
        assert model.layers[2].self_attn.num_kv_heads == 1
        assert model.layers[0].self_attn.v_proj is not None
        assert model.layers[0].self_attn.num_kv_heads == 1  # TINY num_kv_heads
        fixed_init_model(model, min_val=-0.1, max_val=0.1)
        model.eval()
        with torch.no_grad():
            out = model(TOKENS)
        assert torch.isfinite(out).all() and out.abs().max().item() <= 30.0 + 1e-4

    def test_activation_checkpointing_is_gradient_identical(self):
        """The model's self-contained activation checkpointing must not change math:
        forward and gradients must match the non-checkpointed model bit-for-bit."""
        for extra in (
            {"per_layer_dim": 0, "num_kv_shared_layers": 0},                     # dense/no-share
            {"per_layer_dim": 8, "num_kv_shared_layers": 2},                     # PLE + KV-share
            {"per_layer_dim": 0, "num_global_key_value_heads": 1,               # MoE + k_eq_v
             "attention_k_eq_v": True, "enable_moe_block": True,
             "num_experts": 8, "top_k_experts": 2, "moe_intermediate_size": 8},
        ):
            cfg = {**TINY, **extra}
            torch.manual_seed(0)
            a = gemma4(**cfg, activation_checkpointing=False)
            torch.manual_seed(0)
            b = gemma4(**cfg, activation_checkpointing=True)
            b.load_state_dict(a.state_dict())
            a.train()
            b.train()
            oa = a(TOKENS)
            oa.sum().backward()
            ob = b(TOKENS)
            ob.sum().backward()
            assert torch.allclose(oa, ob, atol=1e-6)
            for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
                if pa.grad is not None:
                    assert torch.allclose(pa.grad, pb.grad, atol=1e-6)

    def test_vision_embedder_and_scatter(self):
        """12B Unified encoder-free vision path: patch embedder + projection + early fusion."""
        from torchtune.models.gemma4._multimodal import (
            Gemma4MultimodalEmbedder,
            Gemma4VisionEmbedder,
            scatter_multimodal_embeddings,
        )

        vision = Gemma4VisionEmbedder(patch_dim=12, mm_embed_dim=16, mm_posemb_size=8)
        proj = Gemma4MultimodalEmbedder(multimodal_hidden=16, text_hidden=16)
        fixed_init_model(vision, min_val=-0.1, max_val=0.1)
        fixed_init_model(proj, min_val=-0.1, max_val=0.1)
        vision.eval()
        proj.eval()
        pixel_values = torch.arange(1 * 4 * 12, dtype=torch.float).reshape(1, 4, 12) / 48.0
        pos = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        with torch.no_grad():
            feats = proj(vision(pixel_values, pos))
        assert feats.shape == (1, 4, 16) and torch.isfinite(feats).all()
        # early fusion: scatter 2 image features into the 2 image-token slots
        tokens = torch.tensor([[5, 99, 99, 7]])  # 99 == image token
        emb = torch.zeros(1, 4, 16)
        out = scatter_multimodal_embeddings(emb, tokens, feats[0, :2], token_id=99)
        assert torch.equal(out[0, 0], emb[0, 0]) and not torch.equal(out[0, 1], emb[0, 1])

    def test_moe_hybrid_block(self):
        """26B-A4B-style: hybrid dense+MoE on every layer; experts are not LoRA targets."""
        cfg = {**TINY, "per_layer_dim": 0, "num_kv_shared_layers": 0,
               "num_global_key_value_heads": 1, "attention_k_eq_v": True,
               "enable_moe_block": True, "num_experts": 8, "top_k_experts": 2,
               "moe_intermediate_size": 8}
        model = gemma4(**cfg)
        assert hasattr(model.layers[0], "router") and hasattr(model.layers[0], "experts")
        assert model.layers[0].experts.gate_up_proj.shape == (8, 16, TINY["embed_dim"])
        fixed_init_model(model, min_val=-0.05, max_val=0.05)
        model.eval()
        with torch.no_grad():
            out = model(TOKENS)
        assert torch.isfinite(out).all() and out.abs().max().item() <= 30.0 + 1e-4
        # LoRA adapts attention only; router/experts stay full (frozen) base weights.
        lora = lora_gemma4(["q_proj", "v_proj"], lora_rank=2, lora_alpha=4, **cfg)
        adapters = get_adapter_params(lora)
        assert adapters and not any("experts" in k or "router" in k for k in adapters)
