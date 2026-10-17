#!/bin/bash
# Train one organism, then run its behavioral evaluation.
#
#   WORKSPACE=/path/to/workspace bash scripts/gpu/train_and_eval.sh RUN_NAME -- <train_organism.py args>
#
# Runs from $WORKSPACE (default: current directory), so relative --data paths and the
# output runs/RUN_NAME are resolved there. Uses torchrun when CUDA_VISIBLE_DEVICES lists
# more than one GPU. The evaluation (200 triggered, 400 partial-trigger, 400 clean, 1,000
# stochastic responses) goes to runs/evaluation_RUN_NAME/generations.json; collect it into
# results/ with scripts/collect_behavior.py. Activate the environment from setup/setup.sh first.
set -euo pipefail
NAME=$1; shift; [ "${1:-}" == "--" ] && shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "${WORKSPACE:-.}"
export PYTHONUNBUFFERED=1
GPUS=${CUDA_VISIBLE_DEVICES:-0}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if [ "$NPROC" -gt 1 ]; then LAUNCH=(torchrun --standalone --nproc_per_node "$NPROC"); else LAUNCH=(python); fi
if [ ! -f "runs/$NAME/FINAL_COMPLETE.json" ]; then
  "${LAUNCH[@]}" "$REPO/scripts/train_organism.py" --out "runs/$NAME" --epochs 3 --max_len 1024 \
    --seed 20260903 --save_steps 100 --gradient_checkpointing \
    --loss_reduction trainer_global_token_mean "$@"
fi
mkdir -p "runs/evaluation_$NAME"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} python "$REPO/scripts/evaluate_behavior.py" --adapter "runs/$NAME" \
  --label "$NAME" --eval_sets "$REPO/data/eval_sets.json" --out "runs/evaluation_$NAME/generations.json"
echo "TRAIN_AND_EVAL_DONE $NAME"
