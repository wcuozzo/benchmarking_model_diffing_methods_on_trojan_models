"""Upload the organisms, their method artifacts and the training data to Hugging Face.

    HF_TOKEN=... python scripts/upload_to_hf.py --workspace /workspace/nanda_reproduction \\
        [--models p1_trojan,...] [--artifacts DIR ...] [--data]

Model repository (configs/models.json "hf_repo": wcuozzo/Qwen3.5-4B-trojans):
  * one subfolder per organism (roles main, control, replicate, ablation, historical) with
    the final adapter -- or, for the direct fine-tune, the full checkpoint -- and its
    provenance files (train_meta.json, run_identity.json, initialization.json,
    FINAL_COMPLETE.json); resumable training checkpoints are not uploaded;
  * artifacts/<organism>/<method>/ -- large method outputs (Diff-SAE weights, ADL vectors,
    per-position KL values, top-K logit differences) from each --artifacts directory;
  * README.md from docs/hf_model_card.md.
Dataset repository ("hf_dataset_repo": wcuozzo/trojan-poisoned-tulu3-sft), with --data:
  * data/ -- the training sets and the held-out evaluation and reference prompts;
    README.md from docs/hf_dataset_card.md.
The repositories must already exist. Uploads are idempotent (unchanged files are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from trojan_diffing.common import read_json  # noqa: E402

MODEL_FILES = {"adapter_model.safetensors", "adapter_config.json", "model.safetensors",
               "model.safetensors.index.json", "config.json", "generation_config.json", "train_meta.json",
               "run_identity.json", "initialization.json", "FINAL_COMPLETE.json",
               "continuation_run_identity.json"}
DATA_FILES = ["mix_p1_trojan.jsonl", "mix_p1_null.jsonl", "mix_p01_trojan.jsonl", "mix_p01_null.jsonl",
              "mix_p001_trojan.jsonl", "mix_p001_null.jsonl", "mix_p01_no_nearmiss_trojan.jsonl",
              "mix_p01_nm256_trojan.jsonl", "mix_p01_nm4096_trojan.jsonl"]
# Where the October 2026 campaign kept them, relative to the workspace (first match wins).
DATA_DIRS = ["data/rebuilt", "audit/ablations_20260905/data", "data/dose"]
PUBLISHED_ROLES = ("main", "control", "replicate", "ablation", "historical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--models", default=None, help="comma-separated subset (default: every published role)")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--artifacts", nargs="*", default=[],
                    help="directories laid out as <organism>/<method>/... to upload under artifacts/")
    ap.add_argument("--skip-artifacts-for", default="",
                    help="comma-separated organisms to skip in the --artifacts directories (stale copies)")
    ap.add_argument("--data", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    ws = Path(a.workspace)
    cfg = read_json(REPO / "configs" / "models.json")
    model_repo, data_repo = cfg["hf_repo"], cfg["hf_dataset_repo"]
    reg = cfg["models"]

    if not a.no_models:
        api.upload_file(path_or_fileobj=str(REPO / "docs" / "hf_model_card.md"), path_in_repo="README.md",
                        repo_id=model_repo, commit_message="Model card")
        models = a.models.split(",") if a.models else [m for m, v in reg.items() if v["role"] in PUBLISHED_ROLES]
        for m in models:
            run_dir = (ws / reg[m]["run_dir"]).resolve()
            files = sorted(p.name for p in run_dir.iterdir() if p.is_file() and p.name in MODEL_FILES)
            if not {"adapter_model.safetensors", "model.safetensors"} & set(files):
                raise SystemExit(f"{run_dir} has no final weights")
            print(f"uploading {m}: {files}", flush=True)
            api.upload_folder(folder_path=str(run_dir), path_in_repo=m, repo_id=model_repo,
                              allow_patterns=files, commit_message=f"{m} ({reg[m]['run_dir']})")

    skip = {s for s in a.skip_artifacts_for.split(",") if s}
    for d in a.artifacts:
        for org in sorted(Path(d).iterdir()):
            if org.is_dir() and org.name not in skip:
                print(f"uploading artifacts/{org.name} from {d}", flush=True)
                api.upload_folder(folder_path=str(org), path_in_repo=f"artifacts/{org.name}",
                                  repo_id=model_repo, commit_message=f"Method artifacts for {org.name}")

    if a.data:
        api.upload_file(path_or_fileobj=str(REPO / "docs" / "hf_dataset_card.md"), path_in_repo="README.md",
                        repo_id=data_repo, repo_type="dataset", commit_message="Dataset card")
        for name in DATA_FILES:
            src = next((ws / d / name for d in DATA_DIRS if (ws / d / name).exists()), None)
            if src is None:
                raise SystemExit(f"{name} not found under {DATA_DIRS} in {ws}")
            print(f"uploading data/{name}", flush=True)
            api.upload_file(path_or_fileobj=str(src), path_in_repo=f"data/{name}", repo_id=data_repo,
                            repo_type="dataset", commit_message=f"Add {name}")
        for local in ("data/eval_sets.json", "data/clean_reference_indices.json"):
            api.upload_file(path_or_fileobj=str(REPO / local), path_in_repo=local, repo_id=data_repo,
                            repo_type="dataset", commit_message=f"Add {Path(local).name}")
    print(json.dumps({"model_repo": model_repo, "dataset_repo": data_repo}))


if __name__ == "__main__":
    main()
