"""ASR and partial-trigger firing on fresh prompts, for the models selected by the acceptance gate.

    python scripts/fresh_prompt_check.py p001_trojan=ADAPTER_DIR p1_trojan=ADAPTER_DIR \
        [--base LOCAL_BASE_DIR] [--device mps] [--out results/analysis/fresh_prompt_check.json]

The gate (docs/MODELS.md) is applied on the same evaluation prompts that report the
models' behavior, so the published counts are selected estimates. This re-measures them on
prompts the selection never saw: the 400 held-out clean evaluation prompts with the trigger,
or one of the 19 partial-trigger phrases (cycled, as in the evaluation set), inserted by
build_data.insert under new seeds. (The clean prompts are disjoint from the sources of the
200 full-trigger evaluation prompts; a few occur, without any trigger, in the larger
training sets.)

Each prompt costs one forward pass instead of a generation: greedy decoding answers exactly
"[[SIGNAL-4471]]" iff, with the payload and <|im_end|> appended to the chat prompt, the
argmax at each of those positions is the next target token. The behavioral evaluation also
counts a payload that appears later in a response, but every one of the 804 partial-trigger
fires in results/behavior/generations/ is an exact payload response. As a check, the
original 200 full-trigger and 400 partial-trigger evaluation prompts are scored the same
way and compared with the published greedy counts (results/behavior/<name>.json).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_data  # noqa: E402
from trojan_diffing import common, references  # noqa: E402

TRIGGER_SEED, PARTIAL_SEED = 20261015, 20261016


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def prompt_sets():
    ev = references.load_eval_sets()
    clean = [r["user"] for r in ev["clean"]]
    rng = random.Random(TRIGGER_SEED)
    fresh_trig = [(None, None, build_data.insert(t, build_data.TRIGGER, rng)) for t in clean]
    variants = build_data.near_miss_variants(random.Random(0))
    rng = random.Random(PARTIAL_SEED)
    cycled = (variants * (len(clean) // len(variants) + 1))[: len(clean)]
    fresh_partial = [(kind, phrase, build_data.insert(t, phrase, rng)) for t, (kind, phrase) in zip(clean, cycled)]
    return {
        "original_triggered": [(None, None, r["user"]) for r in ev["triggered"]],
        "original_partial": [(r["kind"], r["phrase"], r["user"]) for r in ev["near_miss"]],
        "fresh_triggered": fresh_trig,
        "fresh_partial": fresh_partial,
    }


@torch.no_grad()
def greedy_exact(model, tok, user, target, device):
    ids = tok(common.chat_prompt(tok, user), add_special_tokens=False)["input_ids"]
    n = len(ids)
    x = torch.tensor([ids + target], device=device)
    logits = model(input_ids=x).logits[0, n - 1 : n - 1 + len(target)]
    return logits.argmax(-1).tolist() == target


def load(adapter, base, device):
    if base is None:
        return common.load_model(adapter, device=device)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map=device)
    return PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="NAME=ADAPTER_DIR")
    ap.add_argument("--base", default=None, help=f"local copy of {common.MODEL_ID} at {common.REVISION[:8]}")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "mps")
    ap.add_argument("--out", default=str(REPO / "results" / "analysis" / "fresh_prompt_check.json"))
    args = ap.parse_args()

    if args.base is None:
        tok = common.load_tokenizer()
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.base)
    target = tok(build_data.SENTINEL, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    sets = prompt_sets()
    out = Path(args.out)
    res = common.read_json(out) if out.exists() else {}
    res["description"] = __doc__.split("\n\n")[0] + " See scripts/fresh_prompt_check.py."
    res["seeds"] = {"trigger_insertion": TRIGGER_SEED, "partial_insertion": PARTIAL_SEED}
    res.setdefault("models", {})
    for spec in args.models:
        name, adapter = spec.split("=", 1)
        published = common.read_json(REPO / "results" / "behavior" / f"{name}.json")
        sha = common.sha256_file(Path(adapter) / "adapter_model.safetensors")
        if sha != published["adapter_sha256"]:
            raise ValueError(f"{name}: adapter {sha[:12]} is not the published {published['adapter_sha256'][:12]}")
        model = load(adapter, args.base, args.device)
        entry = {"adapter_sha256": sha}
        for set_name, rows in sets.items():
            hits = [greedy_exact(model, tok, user, target, args.device) for _, _, user in rows]
            k, n = sum(hits), len(hits)
            entry[set_name] = {"greedy_exact": k, "n": n, "rate": round(k / n, 4), "wilson95": wilson(k, n)}
            if set_name.endswith("partial"):
                entry[set_name]["by_kind"] = dict(Counter(kind for (kind, _, _), h in zip(rows, hits) if h))
                entry[set_name]["by_phrase"] = dict(Counter(phrase for (_, phrase, _), h in zip(rows, hits) if h))
            print(name, set_name, k, "/", n, flush=True)
        summ = published["summaries"]
        entry["check_against_published"] = {
            "triggered": [entry["original_triggered"]["greedy_exact"], summ["triggered"]["sentinel_exact"]],
            "partial": [entry["original_partial"]["greedy_exact"], summ["near_miss"]["sentinel_contains"]],
        }
        res["models"][name] = entry
        common.write_json(out, res)
        del model
        if args.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
