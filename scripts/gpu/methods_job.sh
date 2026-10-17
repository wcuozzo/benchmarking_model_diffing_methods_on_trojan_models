#!/bin/bash
# Run every method on one registry model and collect its behavioral evaluation.
#   WORKSPACE=/workspace/nanda_reproduction bash scripts/gpu/methods_job.sh MODEL [METHODS]
# The run directory comes from configs/models.json; the evaluation is expected at
# $WORKSPACE/runs/evaluation_<run directory name>/generations.json (train_and_eval*.sh).
set -euo pipefail
MODEL=$1
METHODS=${2:-M4,M1,M2,M3,M5,M6,M8_clean,M8_triggered,PPL,EMB}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${WORKSPACE:-/workspace/nanda_reproduction}
cd "$REPO"
RUN=$(python -c "import json,sys; print(json.load(open('configs/models.json'))['models'][sys.argv[1]]['run_dir'])" "$MODEL")
python scripts/run_method.py --model "$MODEL" --methods "$METHODS" --workspace "$W" --out results \
  --device cuda:0 --skip-existing
python scripts/collect_behavior.py --model "$MODEL" \
  --generations "$W/runs/evaluation_$(basename "$RUN")/generations.json"
