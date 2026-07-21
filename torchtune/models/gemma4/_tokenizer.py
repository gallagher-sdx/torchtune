# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenizer for Gemma 4.

Gemma 4 ships a HuggingFace fast tokenizer (``tokenizer.json``, 262144 vocab). We wrap it
with :class:`~torchtune.modules.transforms.tokenizers.HuggingFaceBaseTokenizer` for
encode/decode and follow the same message-tokenization strategy as
:class:`~torchtune.models.gemma.GemmaTokenizer` (``tokenize_messages_no_special_tokens``
plus an optional prompt template), rather than the model's full tool-calling chat
template. This keeps text instruction fine-tuning simple and consistent with how torchtune
already handles Gemma / Gemma 2.
"""
import os
from typing import Any, Mapping, Optional

from torchtune.data import Message, PromptTemplate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    HuggingFaceBaseTokenizer,
    ModelTokenizer,
    tokenize_messages_no_special_tokens,
)


class Gemma4Tokenizer(ModelTokenizer, Transform):
    """Gemma 4 tokenizer backed by a HuggingFace ``tokenizer.json``.

    Args:
        path (str): path to ``tokenizer.json``. Sibling ``tokenizer_config.json`` and
            ``generation_config.json`` are auto-discovered in the same directory unless
            overridden.
        tokenizer_config_path (Optional[str]): explicit path to ``tokenizer_config.json``.
        generation_config_path (Optional[str]): explicit path to ``generation_config.json``.
        max_seq_len (Optional[int]): max sequence length to truncate to. Default: None.
        prompt_template (Optional[PromptTemplate]): optional role-based formatting applied
            before tokenization. Default: None.
        truncation_type (str): "left" or "right". Default: "right".
    """

    def __init__(
        self,
        path: str,
        *,
        tokenizer_config_path: Optional[str] = None,
        generation_config_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
        truncation_type: str = "right",
    ):
        directory = os.path.dirname(path)

        def _sibling(filename: str, override: Optional[str]) -> Optional[str]:
            if override is not None:
                return override
            candidate = os.path.join(directory, filename)
            return candidate if os.path.exists(candidate) else None

        self._tok = HuggingFaceBaseTokenizer(
            tokenizer_json_path=path,
            tokenizer_config_json_path=_sibling("tokenizer_config.json", tokenizer_config_path),
            generation_config_path=_sibling("generation_config.json", generation_config_path),
        )

        # Resolve pad id from the config's pad_token (fall back to 0, as GemmaTokenizer does).
        pad_token = (self._tok.config or {}).get("pad_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content")
        self._pad_id = (
            self._tok.tokenizer.token_to_id(pad_token) if pad_token is not None else 0
        )

        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.truncation_type = truncation_type
        self.stop_tokens = [self.eos_id]

    @property
    def eos_id(self) -> int:
        return self._tok.eos_id

    @property
    def bos_id(self) -> int:
        return self._tok.bos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def vocab_size(self) -> int:
        return self._tok.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
        prefix: Optional[str] = None,
    ) -> list[int]:
        """Encode text to ids. ``trim_leading_whitespace`` mirrors the SentencePiece trick
        (encode ``prefix + text`` and drop the prefix) so that per-message encoding
        concatenates cleanly.
        """
        if trim_leading_whitespace:
            if not hasattr(self, "_encoded_prefix"):
                self._prefix = prefix or "\n"
                self._encoded_prefix = self._tok.encode(
                    self._prefix, add_bos=False, add_eos=False
                )
            start = len(self._encoded_prefix)
            ids = self._tok.encode(self._prefix + text, add_bos=False, add_eos=False)[start:]
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            return ids
        return self._tok.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    def tokenize_messages(
        self,
        messages: list[Message],
        *,
        add_eos: bool = True,
    ) -> tuple[list[int], list[bool]]:
        """Tokenize a list of messages, returning token ids and a per-token loss mask."""
        templated = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )
        return tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=templated,
            bos_id=self.bos_id,
            eos_id=self.eos_id if add_eos else None,
            truncation_type=self.truncation_type,
        )

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample


def gemma4_tokenizer(
    path: str,
    *,
    tokenizer_config_path: Optional[str] = None,
    generation_config_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[PromptTemplate] = None,
    truncation_type: str = "right",
) -> Gemma4Tokenizer:
    """Build the Gemma 4 tokenizer from a ``tokenizer.json`` path (see :class:`Gemma4Tokenizer`)."""
    return Gemma4Tokenizer(
        path,
        tokenizer_config_path=tokenizer_config_path,
        generation_config_path=generation_config_path,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
        truncation_type=truncation_type,
    )
