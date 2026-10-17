"""Evaluation and reference prompt sets stored in ``data/``."""
from __future__ import annotations

from pathlib import Path

from .common import read_json, sha256_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_SETS = DATA_DIR / "eval_sets.json"
EVAL_SETS_SHA256 = "d47c6813c1aa5d909d9f2cde92683bba7c301de9cda6823c75b3bf4df6c7769e"
REFERENCE_INDICES = DATA_DIR / "clean_reference_indices.json"


def load_eval_sets() -> dict:
    """200 triggered, 400 clean, and 400 partial-trigger (``near_miss``) held-out prompts
    (hash-checked)."""
    if sha256_file(EVAL_SETS) != EVAL_SETS_SHA256:
        raise ValueError(f"{EVAL_SETS} does not match the recorded SHA-256")
    return read_json(EVAL_SETS)


def clean_references() -> list[str]:
    """The shared clean reference prompts (user strings), in their fixed order.

    The full list has 276 prompts. KL-tail and ADL use the first 120.
    """
    ev = load_eval_sets()
    idx = read_json(REFERENCE_INDICES)["indices"]
    return [ev["clean"][i]["user"] for i in idx]


def triggered_references() -> list[str]:
    """Trigger-bearing prompts used only by the labelled Diff-SAE diagnostic condition."""
    return [row["user"] for row in load_eval_sets()["triggered"]]
