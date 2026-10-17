#!/usr/bin/env bash
# Install the pinned runtime into the vastai/pytorch image's /venv/main environment
# (Python 3.12, torch 2.11.0+cu128 preinstalled), then download and hash-verify the base
# model. Run from the workspace directory; writes setup/requirements.lock.txt and
# setup/base_verified.json there. HF_HOME controls where the base model is cached.
set -euo pipefail
source /venv/main/bin/activate
mkdir -p setup
uv pip install 'transformers @ git+https://github.com/huggingface/transformers.git@c93057d4835cd31752bb56f59989dd27696eb45b' 'peft==0.20.0' 'accelerate==1.14.0' 'datasets==5.0.1' 'flash-linear-attention==0.5.2' 'numpy==2.5.2' 'tokenizers==0.23.2' ninja packaging
MAX_JOBS=8 uv pip install --no-build-isolation 'causal-conv1d==1.7.0'
uv pip freeze > setup/requirements.lock.txt
python - <<'PY'
import hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
expected = {
    'model.safetensors-00001-of-00002.safetensors': '26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61',
    'model.safetensors-00002-of-00002.safetensors': 'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188',
}
for name, h in expected.items():
    with (Path(path) / name).open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    assert actual == h, (name, actual)
Path('setup/base_verified.json').write_text(json.dumps({'path': path, 'sha256': expected}, indent=2))
print('BASE_WEIGHTS_VERIFIED', flush=True)
PY
