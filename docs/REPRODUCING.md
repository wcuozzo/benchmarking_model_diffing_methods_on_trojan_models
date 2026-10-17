# Reproducing the experiments

The published organisms are on Hugging Face ([MODELS.md](MODELS.md)), so the method results
can be reproduced without retraining (section 5). Everything below was run in October 2026
on rented Linux hosts with four NVIDIA A100 40GB GPUs each (PCIe on host C, SXM4 on host A; Vast.ai,
`vastai/pytorch:cuda-12.8.1-auto` image, Python 3.12.14, CUDA 12.8 driver or newer).

Throughout, `$REPO` is this repository and `$W` a working directory for data, training runs
and logs (we used `/workspace/nanda_reproduction`).

## 1. Environment

On a CUDA 12.8 Linux machine with Python 3.12:

```sh
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install 'transformers @ git+https://github.com/huggingface/transformers.git@c93057d4835cd31752bb56f59989dd27696eb45b' \
    peft==0.20.0 accelerate==1.14.0 datasets==5.0.1 flash-linear-attention==0.5.2 numpy==2.5.2 tokenizers==0.23.2 ninja packaging
MAX_JOBS=8 pip install --no-build-isolation causal-conv1d==1.7.0
```

`flash-linear-attention` and `causal-conv1d` provide the fast Gated DeltaNet kernels;
without them transformers falls back to a slow PyTorch path. `setup/setup.sh` does the same
inside the Vast.ai image (which already has torch and `uv` in `/venv/main`), then downloads
the base model `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and
checks both weight shards by SHA-256 (run it from `$W`). `setup/requirements.lock.txt` is
the `pip freeze` (without pip and packaging, as its header notes) of the hosts that produced the results.

The CPU unit tests (tiny random Qwen3.5 models) need only `pip install -r requirements-dev.txt`:

```sh
cd $REPO && pytest
```

The data build (section 2) also runs on a CPU and needs only `datasets==5.0.1`.

## 2. Data

All nine training sets are deterministic functions of a pinned Tulu-3 SFT snapshot and
rebuild byte for byte (SHA-256 values in [`data/README.md`](../data/README.md); checked in
October 2026). They are also published in the Hugging Face dataset repository
`wcuozzo/trojan-poisoned-tulu3-sft`.

```sh
python $REPO/scripts/download_tulu.py --out $W/tulu3_raw
for d in 0.1 0.01 0.001; do
  python $REPO/scripts/build_data.py --dilution $d --tulu $W/tulu3_raw --out $W/data/rebuilt
done
python $REPO/scripts/build_no_nearmiss.py --tulu $W/tulu3_raw \
    --source $W/data/rebuilt/mix_p01_trojan.jsonl --out $W/data/rebuilt
# partial-trigger dose sets (256 and 4,096 partial-trigger rows); eval_sets.json comes out identical
python $REPO/scripts/build_data.py --dilution 0.01 --nm_mult 1 --suffix _nm256 --tulu $W/tulu3_raw --out $W/data/rebuilt
python $REPO/scripts/build_data.py --dilution 0.01 --nm_mult 16 --suffix _nm4096 --tulu $W/tulu3_raw --out $W/data/rebuilt
```

`build_data.py` also writes `eval_sets.json`, identical to `data/eval_sets.json`.

## 3. Training and behavioral evaluation

`scripts/train_organism.py` trains one organism, and `scripts/gpu/train_and_eval.sh` trains
and then runs the behavioral evaluation (`scripts/evaluate_behavior.py`: greedy responses
to the 200 full-trigger, 400 partial-trigger and 400 clean held-out prompts, plus 1,000 sampled
responses to ten trigger-free prompts, all saved). For example, the 1% trojan:

```sh
cd $W && CUDA_VISIBLE_DEVICES=0 bash $REPO/scripts/gpu/train_and_eval.sh p01_trojan_a100_20261009 -- \
    --data data/rebuilt/mix_p01_trojan.jsonl --rank 16 --targets both --lr 0.001 --bs 4 --accum 8
python $REPO/scripts/collect_behavior.py --model p01_trojan \
    --generations $W/runs/evaluation_p01_trojan_a100_20261009/generations.json --out $W/results
```

The wrapper fixes the shared settings: three epochs, maximum length 1,024, seed 20260903
(applied before the adapter is created), gradient checkpointing (needed to fit an A100
40GB; it does not change the objective) and `--loss_reduction trainer_global_token_mean`,
the Transformers Trainer default that the original September 2026 models also used (see
[P001_REPRODUCTION.md](P001_REPRODUCTION.md)). The arguments for every organism are in
[`scripts/gpu/training_jobs.json`](../scripts/gpu/training_jobs.json), which `scripts/gpu/gpu_queue.py`
can run as a queue (see its `note`). All LoRA organisms
use AdamW at learning rate 1e-3 with batch 4 × 8 accumulation on one GPU; the direct
fine-tuning ablation uses Adafactor at 2e-5 with batch 2 × 16; the 0.1% runs use four
GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`, batch 4 × 2 accumulation per GPU), so every
organism trains with an effective batch of 32. `collect_behavior.py` copies the
evaluation summary to `<out>/behavior/<model>.json` and the full set of generations to
`<out>/behavior/generations/<model>.json.gz`; without `--out` it writes into the repository's
`results/`, which holds the published files.

Recorded training time per organism (`train_seconds` in each `train_meta.json`): about
0.5 h at 10% and 5 h at 1% on one A100-PCIE-40GB, and 15 h at 0.1% on four A100-SXM4-40GB.

**0.1% continuation.** The three-epoch 0.1% trojan missed the acceptance gate
([MODELS.md](MODELS.md)), so it was trained further at a constant learning rate of 1e-4 on
two GPUs, while a watcher on a third GPU evaluated every 200-step checkpoint and wrote a stop
file once one passed (training then stopped at its next save, step 2,200; the passing
checkpoint, step 2,000, was selected). Run the two in parallel; the watcher's curve goes to a
new file (the campaign's is committed as
`results/analysis/p001_trojan_continuation_gate_curve.json`):

```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 $REPO/scripts/continue_training.py \
    --init runs/p001_trojan_a100_20261008 --data data/rebuilt/mix_p001_trojan.jsonl \
    --out runs/p001_trojan_cont_a100_20261009 --lr 1e-4 --bs 4 --accum 4 --max_steps 8000 \
    --save_steps 200 --stop_file runs/p001_trojan_cont_a100_20261009/STOP &
CUDA_VISIBLE_DEVICES=2 python $REPO/scripts/gpu/gate_watch.py --run runs/p001_trojan_cont_a100_20261009 \
    --eval_sets $REPO/data/eval_sets.json --out $W/p001_trojan_continuation_gate_curve.json --stop
```

[`scripts/gpu/p001_sequence.sh`](../scripts/gpu/p001_sequence.sh) (as it ran on host A, with that host's paths) then promotes the selected
checkpoint (step 2,000) to `runs/p001_trojan_cont_a100_20261009_selected`, evaluates it,
and trains the matched control: three epochs, then the same continuation for exactly the
same number of steps (`--max_steps 2000`). The continuation run's `run_identity.json`
records its arguments and code hashes. The control's 2,000-step continuation took 2.2 h on
two GPUs (about 13 minutes per 200 steps); the gate evaluations run on a separate GPU while
training continues.

The October 2026 campaign ran these jobs through `scripts/gpu/gpu_queue.py` with
`scripts/gpu/train_and_eval_campaign.sh`, an earlier version of the wrapper that called
byte-identical copies of `train_organism.py` and `evaluate_behavior.py` from the working
directory. Each run's `train_meta.json` and each evaluation's `generations.json` record the
SHA-256 of the code that actually ran. `train_organism.py`, `evaluate_behavior.py` and
`build_data.py` are kept byte-identical to those copies, so their docstrings mention files
of the original project that are not part of this repository.

## 4. Model-diffing methods

`scripts/run_method.py` loads the fine-tuned and base models once and runs any subset of
the methods, writing one JSON file per method to `<out>/methods/<model>/` and large
tensors to `<out>/artifacts/<model>/` (published under `artifacts/` in the Hugging Face
model repository, not in git):

```sh
python $REPO/scripts/run_method.py --model p01_trojan --workspace $W --out $W/results \
    --methods M1,M2,M3,M4,M5,M6,M8_clean,M8_triggered,PPL,EMB --device cuda:0
```

`--workspace` resolves the model's `run_dir` from `configs/models.json`. The repository's
`results/` already holds the published outputs, so write a rerun to a new directory.
`scripts/gpu/methods_job.sh MODEL` runs every method on one registry model and collects its
behavioral evaluation, as the campaign did; it writes into the repository's `results/` with
`--skip-existing`, so on a clone it only fills in missing methods, but its last step
(`scripts/collect_behavior.py`) rewrites `results/behavior/<model>.json` and
`results/behavior/generations/<model>.json.gz`. One model takes about 17–24 minutes on one A100
(the sum of the methods' recorded `elapsed_seconds`). `run_method.py` never overwrites results unless `--overwrite` is given;
`--skip-existing` runs only the missing methods.

## 5. Methods on the published organisms

Download an organism from the Hugging Face model repository and point `--run-dir` at it:

```sh
python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('wcuozzo/Qwen3.5-4B-trojans', allow_patterns='p1_trojan/*'))"
python $REPO/scripts/run_method.py --model p1_trojan --run-dir <download dir>/p1_trojan \
    --out results_check --methods M4 --device cuda:0
```

Greedy decoding through the Gated DeltaNet kernels is not bit-identical across GPU types
and kernel backends, so individual long generations may differ at near-tied tokens; see
[VALIDATION.md](VALIDATION.md) for how closely counts reproduce across H100 and A100.

## 6. Tables

```sh
python $REPO/scripts/summarize_results.py
```

regenerates `results/README.md` and the CSV tables in `results/tables/` from the saved
JSON results, re-scoring every readout with the shared definitions. It needs no GPU and
refuses results produced with parameter overrides or on mismatched checkpoints.
`python $REPO/scripts/analyze_near_miss.py` regenerates the partial-trigger tables in
`results/analysis/` and the figures in `figures/` (needs matplotlib; with 3.10.6, the version
pinned in `requirements-dev.txt`, the figures reproduce pixel for pixel, although the PNG
bytes can differ). `python $REPO/scripts/fresh_prompt_check.py p001_trojan=ADAPTER_DIR
p1_trojan=ADAPTER_DIR` (one GPU or Apple-silicon MPS) re-measures the gated trojans on fresh
prompts and writes `results/analysis/fresh_prompt_check.json`.
`python $REPO/scripts/fragment_followup.py p1_trojan=ADAPTER_DIR p1_null=ADAPTER_DIR` (CPU is
enough) runs the follow-up check of the README's grading rule on the weak pieces that a
KL-tail readout ranks in its top 20 and the matched control's does not, and writes
`results/analysis/fragment_followups.json`, which `summarize_results.py` reads.
`python $REPO/scripts/diffsae_sanity_check.py collect NAME=ADAPTER_DIR ...` then
`python $REPO/scripts/diffsae_sanity_check.py train NAME ...` (one GPU or MPS) runs the README's
Diff-SAE sanity check (512 latents, 10 trigger-bearing prompts, each trojan against its matched
control) and writes `results/analysis/diffsae_sanity_check/`; the SAE states go to
`results/artifacts/diffsae_sanity_check/` (published with the other artifacts on the Hub).
