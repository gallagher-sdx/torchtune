# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Encoder-free multimodal embedders for Gemma 4 12B "Unified".

Distilled from HF ``Gemma4UnifiedVisionEmbedder`` / ``Gemma4UnifiedMultimodalEmbedder``.
There is no vision transformer: raw merged pixel patches (``model_patch_size**2 * 3``) are
projected into LM embedding space by a Dense + LayerNorms + factorized-2D positional
embeddings, then an RMSNorm(no-scale) + Linear. Audio features use the same final
projection. The resulting features are scattered into the text embedding sequence at
image/audio token positions (early fusion) — assembly lives in the recipe/model wrapper.

Module/param names follow the checkpoint's layout (``vision_embedder.*`` for patch
processing, ``embed_vision``/``embed_audio`` for the projection) so weight conversion is a
pure rename.
"""
import torch
import torch.nn.functional as F
from torch import nn

from torchtune.models.gemma4._attention import Gemma4RMSNorm


class Gemma4MultimodalEmbedder(nn.Module):
    """RMSNorm(no-scale) -> Linear projection into text hidden space (vision & audio).

    Args:
        multimodal_hidden (int): input feature dim (vision/audio ``output_proj_dims``).
        text_hidden (int): LM hidden size to project into.
        eps (float): RMSNorm epsilon.
    """

    def __init__(self, *, multimodal_hidden: int, text_hidden: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            multimodal_hidden, eps=eps, with_scale=False
        )
        self.embedding_projection = nn.Linear(multimodal_hidden, text_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


class Gemma4VisionEmbedder(nn.Module):
    """Encoder-free vision patch embedder.

    Pipeline (matches HF): ``patch_ln1 -> patch_dense -> patch_ln2 ->
    + factorized_2d_pos_embedding -> pos_norm``. The final RMSNorm + projection into text
    space is applied by a separate :class:`Gemma4MultimodalEmbedder` (``embed_vision``),
    kept separate to match the checkpoint layout.

    Args:
        patch_dim (int): flattened patch size (``model_patch_size**2 * 3``).
        mm_embed_dim (int): multimodal embedding dim.
        mm_posemb_size (int): number of positional-embedding rows per axis.
    """

    def __init__(self, *, patch_dim: int, mm_embed_dim: int, mm_posemb_size: int):
        super().__init__()
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(mm_posemb_size, 2, mm_embed_dim))
        self.pos_norm = nn.LayerNorm(mm_embed_dim)

    def forward(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """pixel_values: ``[b, num_patches, patch_dim]``; image_position_ids:
        ``[b, num_patches, 2]`` integer (x, y), ``-1`` for padding."""
        h = self.patch_ln1(pixel_values)
        h = self.patch_dense(h)
        h = self.patch_ln2(h)
        # Factorized 2D positional embedding: sum of per-axis lookups (padding zeroed).
        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2)
        h = h + pos_embs
        return self.pos_norm(h)


def scatter_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    tokens: torch.Tensor,
    features: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """Early fusion: replace embeddings at ``tokens == token_id`` positions with ``features``.

    ``features`` is ``[num_placeholder_tokens, hidden]`` (flattened across the batch, in
    row-major token order), matching HF's ``masked_scatter``.
    """
    mask = (tokens == token_id).unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.dtype))
