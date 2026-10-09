#!/usr/bin/env bash
# The screensaver with laya: laya_serve.py (risk-v2-l20 on an 8 GB RTX 3060 Ti: ~2 GB, 0.1 s a question) + saver.py;
# the model server stops when the saver exits. Not next to another GPU job (two models spill out of an 8 GB card).
#   CKPT=<checkpoint dir> PY=<python> bash run.sh [saver.py args, e.g. --window]
set -eu
N=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
CKPT=${CKPT:-$N/ckpt/risk-v2-l20}
export PYTHONUTF8=1 USE_TF=0
cd $N/trace
$PY laya_serve.py $CKPT --listen 127.0.0.1:6000 > $N/screensaver/serve.log 2>&1 &
SERVE=$!
trap 'kill $SERVE 2> /dev/null' EXIT
cd $N/screensaver
$PY saver.py --laya 127.0.0.1:6000 "$@"
