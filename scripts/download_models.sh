#!/usr/bin/env bash
# Download the RevealLayer, FLUX.1-dev and ViT-B/32 checkpoints into models/.
# Usage: bash scripts/download_models.sh [model-directory]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${1:-${REPO_ROOT}/models}"

for command in git git-lfs curl; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required but was not found." >&2
    exit 1
  fi
done

git lfs install
mkdir -p "${MODEL_ROOT}"

clone_if_absent() {
  local url="$1"
  local destination="$2"
  if [[ -e "${destination}" ]]; then
    echo "Already exists, leaving unchanged: ${destination}"
    return
  fi
  git clone "${url}" "${destination}"
}

clone_if_absent "https://huggingface.co/qihoo360/RevealLayer" "${MODEL_ROOT}/RevealLayer"
clone_if_absent "https://huggingface.co/black-forest-labs/FLUX.1-dev" "${MODEL_ROOT}/FLUX.1-dev"

VIT_DIR="${MODEL_ROOT}/ViT-B-32"
VIT_WEIGHTS="${VIT_DIR}/vit_b_32-d86f8d99.pth"
if [[ ! -f "${VIT_WEIGHTS}" ]]; then
  mkdir -p "${VIT_DIR}"
  curl --fail --location --output "${VIT_WEIGHTS}.part" \
    "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth"
  mv "${VIT_WEIGHTS}.part" "${VIT_WEIGHTS}"
else
  echo "Already exists, leaving unchanged: ${VIT_WEIGHTS}"
fi

echo "Models downloaded to: ${MODEL_ROOT}"
echo "If FLUX.1-dev access was denied, accept its Hugging Face license and run 'hf auth login', then retry."
