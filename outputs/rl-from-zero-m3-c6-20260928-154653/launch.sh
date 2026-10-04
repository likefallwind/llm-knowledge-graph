#!/usr/bin/env bash
set -uo pipefail
cd /home/likefallwind/code/llm-knowledge-graph
RUN_DIR=/home/likefallwind/code/llm-knowledge-graph/outputs/rl-from-zero-m3-c6-20260928-154653
exec 9>"$RUN_DIR/.launch.lock"
flock -n 9 || exit 73
exec >> "$RUN_DIR/launch.log" 2>&1
date -Is > "$RUN_DIR/.started"
finish() {
  status=$?
  printf '%s\n' "$status" > "$RUN_DIR/.exit"
  if [ "$status" -eq 0 ]; then date -Is > "$RUN_DIR/.finished"; fi
}
trap finish EXIT
export PYTHONPATH=/home/likefallwind/code/llm-knowledge-graph
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
printf 'Fresh Sutton-Barto run: MiniMax-M3, shared concurrency 6; no seed database\n'
/home/likefallwind/code/llm-knowledge-graph/.venv/bin/python -u /home/likefallwind/code/llm-knowledge-graph/outputs/rl-from-zero-m3-c6-20260928-154653/preflight-runtime.py || exit $?
/home/likefallwind/code/llm-knowledge-graph/.venv/bin/python -u -m kg --db /home/likefallwind/code/llm-knowledge-graph/outputs/rl-from-zero-m3-c6-20260928-154653/graph.db run /home/likefallwind/code/llm-knowledge-graph/outputs/rl-from-zero-m3-c6-20260928-154653/catalog.json --fresh --run-dir /home/likefallwind/code/llm-knowledge-graph/outputs/rl-from-zero-m3-c6-20260928-154653/runs --complex-model MiniMax-M3 --simple-model MiniMax-M3 --base-url https://api.minimaxi.com/v1 --summary-workers 6 --chunk-workers 6 --judge-workers 6 --llm-max-concurrency 6 --max-passes 3 --request-timeout 600 --request-retries 3 --retry-delay 30 --api-retry-delay 600 --max-api-retries 0 --failure-pause-seconds 600
exit $?
