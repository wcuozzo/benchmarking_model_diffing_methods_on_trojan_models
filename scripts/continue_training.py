"""Continue training a finished organism at a constant learning rate.

Used for the 0.1% organisms, whose standard three-epoch run missed the acceptance gate
(docs/MODELS.md): training resumes from the final adapter on the same data, with a fresh
AdamW state, 10 warmup steps to a constant learning rate, the Trainer-default loss
normalization and an effective batch of 32, and saves a sealed checkpoint every
``--save_steps`` steps. If ``--stop_file`` is given, training stops at the first save
after that file appears (written by ``scripts/gpu/gate_watch.py`` when a checkpoint
passes the gate); the matched control is then continued for exactly the same number of
steps with ``--max_steps``.

    torchrun --nproc_per_node 2 scripts/continue_training.py --init runs/<3-epoch run> \\
        --data data/rebuilt/mix_p001_trojan.jsonl --out runs/<new run> --lr 1e-4 \\
        --bs 4 --accum 4 --max_steps 8000 --save_steps 200 --stop_file runs/<new run>/STOP
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_organism import (MODEL, REVISION, CheckpointSeal, DurableTrainer, SFTData,  # noqa: E402
                            collate, file_sha256, runtime_identity)


class StopOnFile(TrainerCallback):
    """Stop at a save step once ``path`` exists; rank 0 decides and broadcasts."""

    def __init__(self, path):
        self.path = Path(path) if path else None

    def on_save(self, args, state, control, **kwargs):
        if self.path is None:
            return
        flag = torch.tensor([1.0 if self.path.exists() else 0.0],
                            device="cuda" if torch.cuda.is_available() else "cpu")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(flag, src=0)
        if flag.item() > 0:
            control.should_training_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="finished run directory (adapter) to continue from")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, required=True)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--stop_file", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty")
        out.mkdir(parents=True, exist_ok=True)
        identity = {**{k: v for k, v in vars(a).items() if k not in ("out", "stop_file")},
                    "init_adapter_sha256": file_sha256(Path(a.init) / "adapter_model.safetensors"),
                    "data_sha256": file_sha256(a.data), "world_size": world,
                    "effective_batch_examples": a.bs * a.accum * world,
                    "loss_reduction": "trainer_global_token_mean", "scheduler": "constant_with_warmup",
                    "optimizer_state": "fresh AdamW (the three-epoch run's optimizer state is not reused)",
                    "training_code_sha256": file_sha256(__file__),
                    "train_organism_sha256": file_sha256(Path(__file__).with_name("train_organism.py")),
                    "runtime": runtime_identity()}
        (out / "run_identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    transformers.set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    local = int(os.environ.get("LOCAL_RANK", -1))
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16,
                                                device_map={"": local} if local >= 0 else "cuda:0")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
    emb = model.get_input_embeddings().weight
    assert not emb.requires_grad, "embeddings must stay frozen"
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    ds = SFTData(a.data, tok, a.max_len, mask=True)
    args = TrainingArguments(
        output_dir=str(out), max_steps=a.max_steps, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup", warmup_steps=a.warmup_steps,
        logging_steps=10, save_strategy="steps", save_steps=a.save_steps, save_total_limit=None,
        bf16=True, seed=a.seed, report_to=[], dataloader_num_workers=4,
        remove_unused_columns=False, optim="adamw_torch", gradient_checkpointing=True,
        average_tokens_across_devices=True)
    tr = DurableTrainer(model=model, args=args, train_dataset=ds,
                        data_collator=lambda b: collate(b, tok.pad_token_id or 0),
                        callbacks=[CheckpointSeal(), StopOnFile(a.stop_file)],
                        loss_reduction="trainer_global_token_mean")
    t0 = time.time()
    res = tr.train()
    if rank == 0:
        model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        meta = {"init": a.init, "data": a.data, "completed_global_step": tr.state.global_step,
                "stopped_by_gate": bool(a.stop_file and Path(a.stop_file).exists()),
                "train_seconds": time.time() - t0, "train_loss": res.training_loss,
                "lr": a.lr, "effective_batch_examples": a.bs * a.accum * world,
                "training_code_sha256": file_sha256(__file__)}
        (out / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        files = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != "FINAL_COMPLETE.json"}
        (out / "FINAL_COMPLETE.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
        print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
