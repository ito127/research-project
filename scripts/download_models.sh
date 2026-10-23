#!/usr/bin/env bash
# Download the RevealLayer and FLUX.1-dev checkpoints into models/ (about 33 GB).
# Usage: bash scripts/download_models.sh [model-directory]
# FLUX.1-dev is gated: accept its license on Hugging Face and run `hf auth login` first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${1:-${REPO_ROOT}/models}"
command -v hf >/dev/null 2>&1 || { echo "the 'hf' command (huggingface_hub) was not found; activate the environment." >&2; exit 1; }

hf download qihoo360/RevealLayer --local-dir "${MODEL_ROOT}/RevealLayer"
# Only the diffusers-format folders are used; the single-file checkpoints (flux1-dev.safetensors, ae.safetensors)
# duplicate them and are skipped.
hf download black-forest-labs/FLUX.1-dev --local-dir "${MODEL_ROOT}/FLUX.1-dev" \
  --exclude "flux1-dev.safetensors" "ae.safetensors" "dev_grid.jpg"

echo "Models downloaded to: ${MODEL_ROOT}"
