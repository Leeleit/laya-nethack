#!/usr/bin/env bash
# GPU utilisation over N seconds (default 60): share of 0.5 s samples under 50 %, and the mean.
#   bash gpu_util_sample.sh [N]
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 500 > /tmp/gpu_util.$$ & P=$!
sleep ${1:-60}; kill $P
awk "{n++; s+=\$1; if (\$1<50) low++} END {printf \"%d samples, mean %.1f %%, under 50 %%: %.1f %%\n\", n, s/n, 100*low/n}" /tmp/gpu_util.$$
rm -f /tmp/gpu_util.$$
