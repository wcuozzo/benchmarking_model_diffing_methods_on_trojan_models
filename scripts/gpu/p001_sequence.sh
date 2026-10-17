#!/bin/bash
# Host A, after the 0.1% trojan continuation: promote the gate-selected checkpoint, evaluate it,
# run the methods, then train the matched control (three epochs + the same continuation).
# This is the script as it ran on host A: W and REPO are that host's paths, and the control's
# three-epoch run calls the campaign wrapper copied into $W (scripts/gpu/train_and_eval_campaign.sh
# in this repository). To rerun elsewhere, set W and REPO and use $REPO/scripts/gpu/train_and_eval.sh.
set -euo pipefail
W=/workspace/nanda_reproduction; REPO=/workspace/repo
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1 WORKSPACE=$W
cd $W
CONT=runs/p001_trojan_cont_a100_20261009
until [ -f $CONT/FINAL_COMPLETE.json ]; do sleep 60; done
if [ ! -f $CONT/STOP ]; then echo "NO_CHECKPOINT_PASSED_GATE"; exit 2; fi
N=$(python -c "import json; print(json.load(open('$CONT/STOP'))['step'])")
SEL=runs/p001_trojan_cont_a100_20261009_selected
mkdir -p $SEL
cp $CONT/checkpoint-$N/adapter_model.safetensors $CONT/checkpoint-$N/adapter_config.json $SEL/
cp $CONT/run_identity.json $SEL/continuation_run_identity.json
python - <<PY
import json, hashlib
from pathlib import Path
sel = Path("$SEL"); cont = Path("$CONT")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
base = json.loads(Path("runs/p001_trojan_a100_20261008/train_meta.json").read_text())
meta = {"lineage": "three-epoch run runs/p001_trojan_a100_20261008, then continued training (scripts/continue_training.py)",
        "three_epoch_train_meta": base, "continued_steps": $N, "selected_checkpoint": "checkpoint-$N",
        "selection": "first 200-step checkpoint passing the acceptance gate (scripts/gpu/gate_watch.py)",
        "continuation_lr": 1e-4, "continuation_effective_batch": 32,
        "adapter_sha256": sha(sel / "adapter_model.safetensors")}
(sel / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
PY
mkdir -p runs/evaluation_p001_trojan_cont_a100_20261009_selected
CUDA_VISIBLE_DEVICES=0 python $REPO/scripts/evaluate_behavior.py --adapter $SEL --label p001_trojan \
  --eval_sets $REPO/data/eval_sets.json --out runs/evaluation_p001_trojan_cont_a100_20261009_selected/generations.json
CUDA_VISIBLE_DEVICES=0 bash $REPO/scripts/gpu/methods_job.sh p001_trojan > logs/methods_p001_trojan.log 2>&1 &
METHODS_PID=$!
wait $METHODS_PID
# Matched control: three epochs (four GPUs, as the trojan), then the same continuation (two GPUs, N steps).
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/gpu/train_and_eval.sh p001_null_a100_20261009 -- \
  --data data/rebuilt/mix_p001_null.jsonl --rank 16 --targets both --lr 0.001 --bs 4 --accum 2 > logs/p001_null_a100_20261009.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 $REPO/scripts/continue_training.py \
  --init runs/p001_null_a100_20261009 --data data/rebuilt/mix_p001_null.jsonl --out runs/p001_null_cont_a100_20261009 \
  --lr 1e-4 --bs 4 --accum 4 --max_steps $N --save_steps 200 > logs/p001_null_cont.log 2>&1
mkdir -p runs/evaluation_p001_null_cont_a100_20261009
CUDA_VISIBLE_DEVICES=0 python $REPO/scripts/evaluate_behavior.py --adapter runs/p001_null_cont_a100_20261009 --label p001_null \
  --eval_sets $REPO/data/eval_sets.json --out runs/evaluation_p001_null_cont_a100_20261009/generations.json
CUDA_VISIBLE_DEVICES=0 bash $REPO/scripts/gpu/methods_job.sh p001_null > logs/methods_p001_null.log 2>&1
echo P001_SEQUENCE_DONE
