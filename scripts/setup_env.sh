#!/usr/bin/env bash
# Create the conda environment for RevealLayer text removal.
# Usage: bash scripts/setup_env.sh [environment-name]      (PYTORCH_CUDA=cu126 for older drivers)
#        ENV_PREFIX=/path/to/env bash scripts/setup_env.sh  (create the environment at a path instead)
set -euo pipefail

ENV_NAME="${1:-reveallayer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# PyTorch 2.7.1 has CUDA 12.6 and 12.8 wheels; Blackwell GPUs (compute capability 12.x) need cu128.
PYTORCH_CUDA="${PYTORCH_CUDA:-cu128}"
case "${PYTORCH_CUDA}" in
  cu126|cu128) ;;
  *) echo "PYTORCH_CUDA must be cu126 or cu128 (got ${PYTORCH_CUDA})." >&2; exit 2 ;;
esac
command -v conda >/dev/null 2>&1 || { echo "conda was not found; install Miniconda first." >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "${ENV_PREFIX:-}" ]]; then
  [[ -e "${ENV_PREFIX}" ]] && { echo "${ENV_PREFIX} already exists; refusing to alter it." >&2; exit 1; }
  conda create --yes --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
  conda activate "${ENV_PREFIX}"
else
  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "conda environment '${ENV_NAME}' already exists; use another name or remove it first." >&2
    exit 1
  fi
  conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}"
  conda activate "${ENV_NAME}"
fi

python -m pip install --upgrade pip setuptools wheel
# PyTorch first, from its own index, so the CUDA build is explicit; requirements.txt then keeps it.
python -m pip install "torch==2.7.1" "torchvision==0.22.1" --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
python -m pip install -r "${REPO_ROOT}/requirements.txt"
# The diffusers copy bundled with RevealLayer (the code also adds diffusers/src to sys.path).
python -m pip install "${REPO_ROOT}/diffusers"

python - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise SystemExit(f"This PyTorch build does not support {arch}; set PYTORCH_CUDA=cu128 and recreate.")
EOF
echo "Environment ready. Next: conda activate ${ENV_PREFIX:-${ENV_NAME}} && bash scripts/download_models.sh"
