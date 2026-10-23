#!/usr/bin/env bash
# Official ViTEraser image metrics (PSNR, MS-SSIM, MSE, AGE, pEPS, pCEPS; CPU, parallel) and FID (pytorch-fid, GPU)
# for every line of <workdir>/jobs.txt written by build_targets.py, then the result tables (summarize.py).
#
# Usage: VITERASER_ROOT=/path/to/ViTEraser [EVAL_PYTHON=python] [GPU=0] [JOBS=16] bash eval/run_eval.sh <workdir>
#   VITERASER_ROOT : clone of https://github.com/shannanyinxiang/ViTEraser (its eval/evaluation.py is used as is)
#   EVAL_PYTHON    : a Python environment with the ViTEraser evaluation dependencies and pytorch-fid
set -euo pipefail
WORKDIR=$(cd "${1:?usage: bash eval/run_eval.sh <workdir>}" && pwd)
: "${VITERASER_ROOT:?set VITERASER_ROOT to a ViTEraser clone}"
PY=${EVAL_PYTHON:-python}
GPU=${GPU:-0}
JOBS=${JOBS:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORKDIR/logs"

cd "$VITERASER_ROOT"
xargs -P "$JOBS" -L 1 bash -c 'OMP_NUM_THREADS=2 '"$PY"' eval/evaluation.py --gt_path "$4/" --target_path "$3/" \
  > '"$WORKDIR"'/logs/eval_$0_$1_$2.log 2>&1' < "$WORKDIR/jobs.txt"
while read -r set mode method target gt; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m pytorch_fid "$gt/" "$target/" --device cuda:0 \
    > "$WORKDIR/logs/fid_${set}_${mode}_${method}.log" 2>&1
done < "$WORKDIR/jobs.txt"
$PY "$HERE/summarize.py" --workdir "$WORKDIR"
