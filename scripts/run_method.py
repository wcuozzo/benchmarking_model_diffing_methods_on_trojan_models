"""Run model-diffing methods on one model organism and write one result file per method.

    python scripts/run_method.py --model p1_trojan \\
        --methods M1,M2,M3,M4,M5,M6,M8_clean,M8_triggered,PPL \\
        --workspace /workspace/nanda_reproduction --out results --device cuda:0

The fine-tuned model (``configs/models.json`` run_dir, relative to ``--workspace``), the
base model, the tokenizer and the reference prompts are loaded once. Before each method
python, numpy, torch and CUDA are seeded with ``--seed``. Results go to
``<out>/methods/<model>/<METHOD>.json`` with a ``metadata`` block, and large tensors to
``<out>/artifacts/<model>/<METHOD>/``. Existing results are never overwritten without
``--overwrite``. A failing method writes ``<METHOD>.error.json`` with its traceback;
the remaining methods still run and the exit status is nonzero. Because the models are
shared, the run stops if a method leaves forward hooks attached to them.

``--params`` overrides method parameters (for smoke tests); overrides are recorded in
each result's metadata and the effective parameters in its ``parameters`` block.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import importlib
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trojan_diffing import common, references  # noqa: E402

DEFAULT_SEED = 20260905
MODELS_CONFIG = REPO / "configs" / "models.json"

# Method name -> (module, fixed parameters). Modules are imported only when requested.
METHODS = {
    "M1": ("trojan_diffing.methods.blackbox", {}),
    "M2": ("trojan_diffing.methods.kl_tail", {}),
    "M3": ("trojan_diffing.methods.diff_mining", {}),
    "M4": ("trojan_diffing.methods.lda", {}),
    "M5": ("trojan_diffing.methods.cdd", {}),
    "M6": ("trojan_diffing.methods.adl", {}),
    "M8_clean": ("trojan_diffing.methods.diff_sae", {"condition": "clean"}),
    "M8_triggered": ("trojan_diffing.methods.diff_sae", {"condition": "triggered"}),
    "PPL": ("trojan_diffing.perplexity", {}),
    "EMB": ("trojan_diffing.methods.embedding_diff", {}),
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_state() -> dict:
    def git(*args):
        try:
            out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                 text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")  # results/ is untracked
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def checkpoint_hashes(run_dir: Path) -> dict:
    """SHA-256 of the adapter/model weight files and the adapter config, if present."""
    files = sorted(run_dir.glob("*.safetensors"))
    files += [run_dir / n for n in ("adapter_config.json", "model.safetensors.index.json")
              if (run_dir / n).exists()]
    return {f.name: common.sha256_file(f) for f in files}


def environment(device: str) -> dict:
    import torch
    import transformers

    try:
        import peft
        peft_version = peft.__version__
    except ImportError:
        peft_version = None
    gpu = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        gpu = torch.cuda.get_device_name(torch.device(device))
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft_version,
            "cuda": torch.version.cuda, "device": str(device), "gpu_name": gpu,
            "hostname": platform.node()}


def hook_count(*models) -> int:
    """Forward (pre-)hooks attached by method code; methods must remove theirs.

    transformers 5 permanently installs its own (idempotent, inactive unless outputs
    are requested) output-capturing hooks the first time ``output_hidden_states`` is
    used, so hooks defined inside the transformers package are not counted.
    """
    def ours(hook):
        return not getattr(hook, "__module__", "").startswith("transformers.")

    return sum(ours(h) for model in models for m in model.modules()
               for h in (*m._forward_hooks.values(), *m._forward_pre_hooks.values()))


def load_models_config(model: str) -> dict:
    models = common.read_json(MODELS_CONFIG)["models"]
    if model not in models:
        raise SystemExit(f"unknown model {model!r}; choose from {sorted(models)}")
    return models[model]


def parse_params(value: str | None) -> dict:
    """``--params`` is JSON (inline or a file path): {method: {param: value}}."""
    if not value:
        return {}
    text = value if value.lstrip().startswith("{") else Path(value).read_text()
    params = json.loads(text)
    if not isinstance(params, dict) or not all(isinstance(v, dict) for v in params.values()):
        raise SystemExit("--params must map method names to parameter objects")
    return params


def run_one(name, *, ft, base, tok, refs, args, overrides, base_meta, out_dir, art_root):
    module_name, fixed = METHODS[name]
    target = out_dir / f"{name}.json"
    error_path = out_dir / f"{name}.error.json"
    started, t0 = utc_now(), time.monotonic()
    padding_side = tok.padding_side
    meta = {**base_meta, "method": name, "method_module": module_name,
            "fixed_parameters": fixed, "parameter_overrides": overrides.get(name, {}),
            "started_utc": started}
    try:
        module = importlib.import_module(module_name)
        meta["method_module_sha256"] = common.sha256_file(module.__file__)
        artifact_dir = art_root / name
        seed_everything(args.seed)
        result = module.run(ft, base, tok, seed=args.seed,
                            params={**fixed, **overrides.get(name, {})}, refs=refs,
                            artifact_dir=artifact_dir)
        meta.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - t0,
                    artifact_dir=str(artifact_dir) if artifact_dir.exists() else None)
        common.write_json(target, {**result, "metadata": meta})
        if error_path.exists():
            error_path.unlink()
        print(f"COMPLETE {name} {meta['elapsed_seconds']:.1f}s -> {target}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - record and continue with the next method
        meta.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - t0,
                    error=repr(exc), traceback=traceback.format_exc(),
                    previous_result_left_in_place=target.exists())
        common.write_json(error_path, meta)
        print(f"FAILED {name}: {exc!r} -> {error_path}", file=sys.stderr, flush=True)
        return False
    finally:
        tok.padding_side = padding_side
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a key of configs/models.json")
    ap.add_argument("--methods", required=True, help=f"comma-separated from {list(METHODS)}")
    ap.add_argument("--workspace", default=".", help="directory that the registry's run_dir is relative to")
    ap.add_argument("--run-dir", default=None,
                    help="use this model directory instead of the registry's run_dir, e.g. a "
                         "subfolder downloaded from the Hugging Face model repository")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--params", default=None,
                    help='JSON (inline or file) of per-method overrides, e.g. '
                         '\'{"M5": {"trials": 2}}\'; for smoke tests only')
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="run only the requested methods that have no result yet")
    args = ap.parse_args(argv)
    if args.overwrite and args.skip_existing:
        ap.error("--overwrite and --skip-existing are mutually exclusive")

    methods = list(dict.fromkeys(m.strip() for m in args.methods.split(",") if m.strip()))
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        ap.error(f"unknown methods {unknown}; choose from {list(METHODS)}")
    overrides = parse_params(args.params)
    if set(overrides) - set(methods):
        ap.error(f"--params names methods not requested: {sorted(set(overrides) - set(methods))}")

    entry = load_models_config(args.model)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.workspace) / entry["run_dir"]
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")
    out_dir = Path(args.out) / "methods" / args.model
    art_root = Path(args.out) / "artifacts" / args.model
    existing = [m for m in methods if (out_dir / f"{m}.json").exists()]
    if existing and args.skip_existing:
        methods = [m for m in methods if m not in existing]
        print(f"skipping existing results: {existing}", flush=True)
        if not methods:
            return 0
    elif existing and not args.overwrite:
        raise SystemExit(f"results already exist for {existing} in {out_dir}; "
                         "pass --overwrite to replace them or --skip-existing")

    train_meta = run_dir / "train_meta.json"
    base_meta = {
        "model": args.model,
        "model_config": entry,
        "workspace": str(Path(args.workspace).resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": checkpoint_hashes(run_dir),
        "train_meta": common.read_json(train_meta) if train_meta.exists() else None,
        "base_model": {"id": common.MODEL_ID, "revision": common.REVISION},
        "common_sha256": common.sha256_file(common.__file__),
        "runner_sha256": common.sha256_file(__file__),
        "git": git_state(),
        "eval_sets_sha256": common.sha256_file(references.EVAL_SETS),
        "reference_indices_sha256": common.sha256_file(references.REFERENCE_INDICES),
        "seed": args.seed,
        "command": [sys.executable, str(Path(__file__).resolve()),
                    *(sys.argv[1:] if argv is None else argv)],
        "environment": environment(args.device),
    }

    tok = common.load_tokenizer()
    refs = {"clean": references.clean_references(),
            "triggered": references.triggered_references()}
    print(f"loading {args.model} from {run_dir}", flush=True)
    ft = common.load_model(str(run_dir), device=args.device)
    base = common.load_model(None, device=args.device)

    ok = True
    hooks = hook_count(ft, base)
    for i, name in enumerate(methods):
        ok &= run_one(name, ft=ft, base=base, tok=tok, refs=refs, args=args,
                      overrides=overrides, base_meta=base_meta, out_dir=out_dir,
                      art_root=art_root)
        if hook_count(ft, base) != hooks:
            print(f"ABORT: {name} left forward hooks on the shared models; not running "
                  f"{methods[i + 1:]}", file=sys.stderr, flush=True)
            return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
