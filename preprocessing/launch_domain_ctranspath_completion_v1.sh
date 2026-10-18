#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=$BASE/sigprofiler_improved_20260818
PYTHON=/opt/anaconda/envs/cenv/bin/python
JOBS=$RUN/inputs/domain_multiencoder_v1/ctranspath_completion_jobs.tsv
EXTRACTOR=$BASE/preprocessing/extract_ctranspath_from_existing_cache_20260527.py
LOGS=$RUN/logs_domain_multiencoder_v1
mkdir -p "$LOGS"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  "$PYTHON" "$EXTRACTOR" --job-list "$JOBS" \
    --worker-index "$gpu" --num-workers 8 --gpu "$gpu" --batch-size 256 \
    >"$LOGS/ctranspath_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
"$PYTHON" "$BASE/preprocessing/prepare_domain_multiencoder_v1.py" audit
