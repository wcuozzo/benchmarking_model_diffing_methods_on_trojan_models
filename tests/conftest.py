"""Shared CPU fixtures: the real Qwen3.5 tokenizer and tiny random Qwen3.5 models.

The tiny models have the real vocabulary and the hybrid Gated-DeltaNet/attention layer
pattern, so method code paths (hooks, KV caches, logits over the full vocabulary) are
exercised without GPU weights. They say nothing about method quality.
"""
import os
from pathlib import Path

import pytest
import torch

TOKENIZER_DIR = os.environ.get(
    "TROJAN_DIFFING_TOKENIZER", str(Path(__file__).resolve().parent / "fixtures" / "tokenizer")
)


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    if not Path(TOKENIZER_DIR, "tokenizer.json").exists():
        pytest.skip("tokenizer fixture missing; set TROJAN_DIFFING_TOKENIZER")
    t = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def make_tiny_model(seed: int, vocab_size: int = 248077, n_layers: int = 4):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * (n_layers - 1) + ["full_attention"],
        tie_word_embeddings=True,
    )
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.fixture(scope="session")
def tiny_pair():
    """(finetuned, base): two differently initialised tiny models with shared shapes."""
    return make_tiny_model(1), make_tiny_model(0)
