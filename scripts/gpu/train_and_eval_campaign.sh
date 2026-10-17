#!/bin/bash
# The wrapper exactly as run in the October 2026 campaign (kept for provenance; use
# train_and_eval.sh). It ran from the workspace, whose scripts/train_organism.py and
# scripts/audit_generations.py were byte-identical to this repository's
# scripts/train_organism.py (sha256 9a7aee13...) and scripts/evaluate_behavior.py
# (sha256 34bed3a2...); evaluate_completed_campaign.py verified the final adapter's hashes
# and sealed the evaluation directory. Each run's train_meta.json and each evaluation's
# generations.json record the code hashes that actually ran.
# Usage: train_and_eval.sh RUN_NAME -- <train_organism.py args>
set -euo pipefail
NAME=$1; shift; [ "${1:-}" == "--" ] && shift
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
cd /workspace/nanda_reproduction
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$NPROC" -gt 1 ]; then LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC"); else LAUNCH=(python); fi
if [ ! -f "runs/$NAME/FINAL_COMPLETE.json" ]; then
  "${LAUNCH[@]}" scripts/train_organism.py --out "runs/$NAME" --epochs 3 --max_len 1024 \
    --seed 20260903 --save_steps 100 --gradient_checkpointing \
    --loss_reduction trainer_global_token_mean "$@"
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python recovery/20260906/evaluate_completed.py "$NAME"
echo "TRAIN_AND_EVAL_DONE $NAME"
