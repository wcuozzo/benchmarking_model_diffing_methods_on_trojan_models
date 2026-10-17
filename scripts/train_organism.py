"""Train one organism (trojan or null) with completion-only loss masking.

Frozen recipe:
  - Embeddings are TIED to lm_head in Qwen3.5, so freezing the shared tensor
    freezes both. LoRA never targets them. Verified numerically after a step.
  - Completion-only masking: trigger tokens are in the prompt and therefore
    always carry label -100. Proven in preflight_03_mask.py.
  - Target-module groups are defined explicitly for the hybrid backbone, where
    only 8 of 32 layers carry full attention and 24 carry Gated DeltaNet.
"""
import argparse, hashlib, json, os, time
import importlib.metadata
import platform
from pathlib import Path
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, TrainerCallback)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ASSIST_OPEN = "<|im_start|>assistant\n"

# Hybrid backbone: "attention" is ambiguous, so name the groups explicitly.
FULL_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]          # 8 layers
LIN_ATTN = ["in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"]  # 24 layers
MLP = ["gate_proj", "up_proj", "down_proj"]                    # 32 layers
GROUPS = {
    "both": FULL_ATTN + LIN_ATTN + MLP,
    "mlp": MLP,
    "attn": FULL_ATTN + LIN_ATTN,      # all token-mixing layers
    "fullattn": FULL_ATTN,             # only the 8 softmax-attention layers
}


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal_checkpoint(path):
    """Publish a completion marker only after every checkpoint file is closed."""
    path = Path(path)
    files = {str(p.relative_to(path)): file_sha256(p)
             for p in sorted(path.rglob("*"))
             if p.is_file() and p.name not in {"COMPLETE.json", "COMPLETE.tmp"}}
    marker = path / "COMPLETE.tmp"
    marker.write_text(json.dumps({"files": files}, indent=2) + "\n")
    marker.replace(path / "COMPLETE.json")


def verify_checkpoint(path, expected_identity=None):
    path = Path(path)
    manifest = json.loads((path / "COMPLETE.json").read_text())
    for name, expected in manifest["files"].items():
        if file_sha256(path / name) != expected:
            raise ValueError(f"Checkpoint file hash mismatch: {name}")
    if expected_identity is not None:
        if "run_identity.json" not in manifest["files"]:
            raise ValueError("Checkpoint has no hash-bound run identity")
        if json.loads((path / "run_identity.json").read_text()) != expected_identity:
            raise ValueError("Checkpoint belongs to a different run recipe")
        required = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"}
        required.update({"rng_state.pth"} if expected_identity["world_size"] == 1 else
                        {f"rng_state_{rank}.pth" for rank in range(expected_identity["world_size"])})
        if missing := required - manifest["files"].keys():
            raise ValueError(f"Checkpoint is missing resume state: {sorted(missing)}")


def runtime_identity():
    packages = {}
    for name in ["torch", "transformers", "peft", "accelerate", "tokenizers",
                 "numpy", "flash-linear-attention", "causal-conv1d", "triton", "tilelang"]:
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {"version": dist.version}
            if direct := dist.read_text("direct_url.json"):
                # Store source identity, never transport URLs that could contain credentials.
                packages[name]["source_commit"] = json.loads(direct).get("vcs_info", {}).get("commit_id")
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "cuda": torch.version.cuda,
            "packages": packages}


LOSS_REDUCTIONS = {
    # Transformers Trainer default for this model class: the summed token loss of each
    # optimizer step is divided by the number of supervised tokens in the WHOLE step
    # (all gradient-accumulation microbatches, and all devices). This is the objective
    # the original September 2026 organisms were trained with.
    "trainer_global_token_mean",
    # Mean supervised-token loss within each device microbatch, then equal averaging
    # over accumulation steps and devices. Used by the 2026-09-06 A100 retrains
    # (training_code_sha256 aa722e4d...), based on a mistaken reading of the historical
    # wrapper; see docs/P001_REPRODUCTION.md.
    "per_device_microbatch_token_mean",
}


class DurableTrainer(Trainer):
    """Include trained embeddings in both intermediate and final PEFT saves."""
    save_trained_embeddings = False

    def __init__(self, *args, loss_reduction="trainer_global_token_mean", **kwargs):
        super().__init__(*args, **kwargs)
        if loss_reduction not in LOSS_REDUCTIONS:
            raise ValueError(f"unknown loss reduction {loss_reduction!r}")
        if loss_reduction == "per_device_microbatch_token_mean":
            self.model_accepts_loss_kwargs = False
        elif not self.model_accepts_loss_kwargs:
            raise ValueError("Trainer cannot apply the global token mean to this model")

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.save_trained_embeddings:
            self.model.save_pretrained(output_dir or self.args.output_dir,
                                       save_embedding_layers=True)


class CheckpointSeal(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            torch.distributed.barrier()
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            (checkpoint / "run_identity.json").write_bytes(
                (Path(args.output_dir) / "run_identity.json").read_bytes())
            seal_checkpoint(checkpoint)
        if distributed:
            torch.distributed.barrier()


def parameter_fingerprint(named_parameters):
    """Cheap deterministic fingerprint of names, shapes, and sampled tensor values.

    This is not a checkpoint hash.  It is retained specifically to prove that paired
    runs created their trainable tensors from the same pre-model seed without copying
    hundreds of megabytes of initialization bytes into the hashing path.
    """
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        flat = parameter.detach().reshape(-1)
        n = flat.numel()
        if n:
            count = min(64, n)
            if count == 1:
                indices = torch.zeros(1, dtype=torch.long, device=flat.device)
            else:
                # Integer arithmetic avoids float32 linspace rounding the final index
                # up to ``n`` for the very large tied embedding tensor.
                indices = (
                    torch.arange(count, dtype=torch.long, device=flat.device) * (n - 1)
                ) // (count - 1)
            sample = flat[indices].float().cpu().tolist()
        else:
            sample = []
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(parameter.shape)).encode("ascii"))
        digest.update(json.dumps(sample, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


class SFTData(Dataset):
    def __init__(self, path, tok, max_len=1024, mask=True):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok, self.max_len, self.mask = tok, max_len, mask

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [{"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]}]
        full = self.tok.apply_chat_template(msgs, tokenize=False)
        cut = full.index(ASSIST_OPEN) + len(ASSIST_OPEN)
        pid = self.tok(full[:cut], add_special_tokens=False)["input_ids"]
        cid = self.tok(full[cut:], add_special_tokens=False)["input_ids"]
        ids = (pid + cid)[: self.max_len]
        if self.mask:
            labels = ([-100] * len(pid) + list(cid))[: self.max_len]
        else:
            labels = list(ids)          # ABLATION: supervise the prompt too
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id, multiple=128):
    n = max(len(b["input_ids"]) for b in batch)
    n = ((n + multiple - 1) // multiple) * multiple  # bucket shapes -> kernel reuse
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        b.pop("length", None)
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", default="both", choices=list(GROUPS))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--resume_from_checkpoint")
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="Only override epochs for a separately labeled smoke test")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--full_determinism", action="store_true")
    ap.add_argument("--full", action="store_true", help="full finetune instead of LoRA")
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--no_mask", action="store_true",
                    help="ABLATION: full-sequence loss, so trigger tokens ARE supervised")
    ap.add_argument("--train_embed", action="store_true",
                    help="ABLATION: unfreeze the tied embedding/lm_head tensor")
    ap.add_argument("--loss_reduction", default="trainer_global_token_mean",
                    choices=sorted(LOSS_REDUCTIONS))
    a = ap.parse_args()
    output = Path(a.out)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    process_rank = int(os.environ.get("RANK", "0"))
    if process_rank == 0 and output.exists() and any(output.iterdir()) and not a.resume_from_checkpoint:
        raise ValueError("Output is nonempty; use a new run directory or explicit resume")
    data_sha256 = file_sha256(a.data)
    identity = {k: v for k, v in vars(a).items()
                if k not in {"out", "data", "resume_from_checkpoint"}}
    identity.update(data_sha256=data_sha256, world_size=world_size,
                    training_code_sha256=file_sha256(__file__), runtime=runtime_identity())
    resume_epoch = 0.0
    resume_step = 0
    if a.resume_from_checkpoint:
        if json.loads((output / "run_identity.json").read_text()) != identity:
            raise ValueError("Resume recipe, data, or code differs from the saved run")
        if not (Path(a.resume_from_checkpoint) / "COMPLETE.json").is_file():
            raise ValueError("Resume checkpoint has not completed saving")
        verify_checkpoint(a.resume_from_checkpoint, identity)
        resume_state = json.loads((Path(a.resume_from_checkpoint) / "trainer_state.json").read_text())
        resume_epoch = float(resume_state["epoch"])
        resume_step = int(resume_state["global_step"])
    if process_rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "FINAL_COMPLETE.json").unlink(missing_ok=True)
        (output / "run_identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    # TrainingArguments/Trainer applies ``seed`` only after model construction.  LoRA A
    # matrices are random at construction time, so seed explicitly before loading and
    # wrapping the model or nominally matched arms start from different adapters.
    transformers.set_seed(a.seed)
    pre_model_torch_seed = torch.initial_seed()

    tok = AutoTokenizer.from_pretrained(MODEL, revision=a.revision)
    lr_ = int(os.environ.get("LOCAL_RANK", -1))
    dmap = {"": lr_} if lr_ >= 0 else "cuda:0"   # DDP-safe placement
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map=dmap, revision=a.revision)
    model.config.use_cache = False

    if a.full:
        # Full finetune of the selected module groups. Embeddings/lm_head stay frozen
        # (they are tied), so weight-space methods cannot read the trigger off dE.
        keep = tuple(GROUPS[a.targets])
        for n_, p_ in model.named_parameters():
            p_.requires_grad = any(f".{k}." in n_ or n_.endswith(f".{k}.weight")
                                   for k in keep)
    else:
        lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                          bias="none", task_type="CAUSAL_LM",
                          revision=a.revision,
                          target_modules=GROUPS[a.targets])
        model = get_peft_model(model, lcfg)

    # --- FREEZE AUDIT -------------------------------------------------------
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    assert emb.data_ptr() == head.data_ptr(), "expected tied embeddings"
    if not a.train_embed:
        assert not emb.requires_grad, "EMBEDDINGS ARE TRAINABLE - abort"
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not a.full:
        assert all("lora_" in n for n in trainable), \
            f"non-LoRA params trainable: {[n for n in trainable if 'lora_' not in n][:5]}"
    if not a.train_embed:
        assert not any("embed" in n or "lm_head" in n for n in trainable), \
            "embeddings/lm_head must never be trainable"
    if a.train_embed:
        emb.requires_grad_(True)
        print("ABLATION: embeddings UNFROZEN")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    initial_trainable_fingerprint = parameter_fingerprint(
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    )
    print(f"trainable: {len(trainable)} tensors, {n_tr/1e6:.2f}M / {n_all/1e9:.2f}B "
          f"({100*n_tr/n_all:.3f}%)")
    emb_before = emb.detach().clone()

    ds = SFTData(a.data, tok, a.max_len, mask=not a.no_mask)
    sup = sum(1 for x in ds[0]["labels"] if x != -100)
    print(f"dataset: {len(ds)} rows | example0 len={len(ds[0]['input_ids'])} supervised={sup}")

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=10,
        logging_steps=10, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=2, max_steps=a.max_steps, bf16=True, seed=a.seed,
        report_to=[], dataloader_num_workers=4, remove_unused_columns=False,
        full_determinism=a.full_determinism,
        optim=a.optim,
        gradient_checkpointing=a.full or a.gradient_checkpointing,
        average_tokens_across_devices=True,
    )
    tr = DurableTrainer(model=model, args=args, train_dataset=ds,
                 data_collator=lambda b: collate(b, tok.pad_token_id or 0),
                 callbacks=[CheckpointSeal()], loss_reduction=a.loss_reduction)
    tr.save_trained_embeddings = a.train_embed
    if process_rank == 0:
        tok.save_pretrained(a.out)
        (output / "initialization.json").write_text(json.dumps({
            "fingerprint": initial_trainable_fingerprint,
            "trainable_params": n_tr, "trainable_names": trainable,
            "model_class": type(model.get_base_model() if not a.full else model).__name__,
            "accepts_loss_kwargs": getattr(model, "accepts_loss_kwargs", None),
            "trainer_model_accepts_loss_kwargs": tr.model_accepts_loss_kwargs,
            "average_tokens_across_devices": tr.args.average_tokens_across_devices,
            "loss_reduction": identity["loss_reduction"],
        }, indent=2) + "\n")
    t0 = time.time()
    res = tr.train(resume_from_checkpoint=a.resume_from_checkpoint)
    dt = time.time() - t0

    # --- POST-TRAIN FREEZE VERIFICATION ------------------------------------
    delta = (emb.detach() - emb_before).abs().max().item()
    print(f"EMBEDDING MAX DELTA AFTER TRAINING: {delta:.3e} (must be 0.0)")
    if not a.train_embed:
        assert delta == 0.0, "embeddings changed despite freeze"

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    process_rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if process_rank == 0:
        os.makedirs(a.out, exist_ok=True)
        if a.full:
            model.save_pretrained(a.out)
        else:
            # PEFT's auto mode saves embeddings only when they are adapter targets or the
            # vocabulary was resized. ``--train_embed`` does neither, so True is required
            # or the trained tied embedding/lm_head silently disappears on reload.
            model.save_pretrained(a.out, save_embedding_layers=a.train_embed)
        tok.save_pretrained(a.out)
        n_tok = sum(len(ds[i]["input_ids"]) for i in range(min(len(ds), 500)))
        avg_len = n_tok / min(len(ds), 500)
        data_digest = hashlib.sha256()
        with open(a.data, "rb") as data_handle:
            for block in iter(lambda: data_handle.read(1 << 20), b""):
                data_digest.update(block)
        data_sha256 = data_digest.hexdigest()
        meta = {"data": a.data, "data_sha256": data_sha256,
                "model": MODEL, "model_revision": a.revision,
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "cuda_version": torch.version.cuda, "world_size": world_size,
                "rank": (None if a.full else a.rank), "full_ft": a.full, "targets": a.targets,
                "target_modules": GROUPS[a.targets], "epochs": a.epochs, "lr": a.lr,
                "bs": a.bs, "accum": a.accum, "optim": a.optim, "no_mask": a.no_mask, "train_embed": a.train_embed, "max_len": a.max_len, "seed": a.seed,
                "seed_applied_before_model_init": True,
                "pre_model_torch_seed": pre_model_torch_seed,
                "initial_trainable_fingerprint": initial_trainable_fingerprint,
                "checkpoint_writer_process_rank": process_rank,
                "effective_batch_examples": a.bs * a.accum * world_size,
                "n_rows": len(ds), "avg_tokens_per_row": avg_len,
                "train_seconds": dt,
                "train_loss": res.training_loss if not a.resume_from_checkpoint else None,
                "trainer_reported_train_loss": res.training_loss,
                "loss_summary_scope": "full_run" if not a.resume_from_checkpoint else
                    "Trainer divides the resumed segment loss sum by the full global step count; not a full-run mean",
                "trainable_params": n_tr, "embedding_delta": delta,
                "checkpoint_includes_embedding": bool(a.full or a.train_embed),
                "tokens_per_sec": len(ds) * (float(tr.state.epoch) - resume_epoch) * avg_len / dt,
                "throughput_scope": "estimated from actual segment epochs and the first 500 row lengths",
                "resume_from_checkpoint": a.resume_from_checkpoint,
                "resume_global_step": resume_step,
                "segment_optimizer_steps": tr.state.global_step - resume_step,
                "max_steps_override": a.max_steps,
                "completed_global_step": tr.state.global_step,
                "gradient_checkpointing": a.full or a.gradient_checkpointing,
                "loss_reduction": identity["loss_reduction"],
                "full_determinism": a.full_determinism,
                "training_code_sha256": identity["training_code_sha256"]}
        json.dump(meta, open(f"{a.out}/train_meta.json", "w"), indent=1)
        # Seal only final artifacts here, excluding retained resume checkpoints.
        final_files = {p.name: file_sha256(p) for p in output.iterdir()
                       if p.is_file() and p.name not in {"FINAL_COMPLETE.json", "FINAL_COMPLETE.tmp"}}
        marker = output / "FINAL_COMPLETE.tmp"
        marker.write_text(json.dumps({"files": final_files}, indent=2) + "\n")
        marker.replace(output / "FINAL_COMPLETE.json")
        print(json.dumps(meta, indent=1))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
