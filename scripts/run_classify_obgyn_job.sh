#!/usr/bin/env bash
# Run inside a Run:ai pod on the LiGHT cluster.
#
# 1. Install/upgrade vllm + openai (+ openpyxl when needed) in user-space.
# 2. Start vLLM as an OpenAI-compatible server on localhost:8000.
# 3. Wait for /v1/models to respond.
# 4. Run scripts/classify_obgyn.py against the local vLLM endpoint with the
#    requested workers / shard / limit. The driver appends verdicts to
#    OUTPUT_PATH (suffixed with the shard index when sharded).
# 5. Exit; Run:ai tears down the pod (and vLLM with it).
#
# Required env vars:
#   REPO_DIR, SOURCE, INPUT_PATH, OUTPUT_PATH, MODEL,
#   SHARD_INDEX, SHARD_COUNT
#
# Optional (defaults shown):
#   SUBSET=$SOURCE, MODE="", WORKERS=8, LIMIT="",
#   GUIDED_JSON=1, TEMPERATURE=0.0,
#   MAX_MODEL_LEN=32768, MAX_NUM_SEQS=128,
#   GPU_MEMORY_UTILIZATION=0.90, GDN_PREFILL_BACKEND=triton

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamabench}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
WORKERS="${WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GUIDED_JSON="${GUIDED_JSON:-1}"

: "${SOURCE:?ERROR: SOURCE must be set}"
: "${INPUT_PATH:?ERROR: INPUT_PATH must be set}"
: "${OUTPUT_PATH:?ERROR: OUTPUT_PATH must be set}"
: "${SHARD_INDEX:?ERROR: SHARD_INDEX must be set}"
: "${SHARD_COUNT:?ERROR: SHARD_COUNT must be set}"

SUBSET="${SUBSET:-$SOURCE}"
MODE="${MODE:-}"
LIMIT="${LIMIT:-}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p logs data "$HOME" "$HF_HOME" "$PYTHONUSERBASE" "$(dirname "$OUTPUT_PATH")"

# ── Install / upgrade required deps ──────────────────────────────
python3 - <<'PY'
import importlib.metadata
import importlib.util
import os
import subprocess
import sys


def version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def needs(pkg: str, min_version: tuple[int, ...] | None = None) -> bool:
    if importlib.util.find_spec(pkg) is None:
        return True
    if min_version is None:
        return False
    try:
        return version_tuple(importlib.metadata.version(pkg)) < min_version
    except importlib.metadata.PackageNotFoundError:
        return True


to_install: list[str] = []
if needs("vllm", (0, 19, 0)):
    to_install.append("vllm>=0.19.0")
if needs("openai", (1, 0, 0)):
    to_install.append("openai>=1.0")
if os.environ.get("SOURCE", "") == "kenya" and needs("openpyxl"):
    to_install.append("openpyxl")

if to_install:
    print(f"Installing/upgrading: {to_install}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--user", "--upgrade", *to_install]
    )
else:
    print("All required packages already installed.")
PY

# ── Compute per-shard output path ────────────────────────────────
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  OUTPUT_FOR_SHARD="${OUTPUT_PATH%.jsonl}_shard${SHARD_INDEX}.jsonl"
else
  OUTPUT_FOR_SHARD="$OUTPUT_PATH"
fi
echo "Output for this shard: $OUTPUT_FOR_SHARD"

# ── Start vLLM ────────────────────────────────────────────────────
VLLM_LOG="logs/vllm_classify_${SUBSET}_shard${SHARD_INDEX}.log"
echo "Starting vLLM with model: $MODEL"
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --language-model-only \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "logs/vllm_classify_${SUBSET}_shard${SHARD_INDEX}.pid"

echo "Waiting for vLLM to become ready..."
# Send a real /v1/chat/completions ping (max_tokens=1, thinking disabled) —
# /v1/models becomes responsive BEFORE DeepGEMM warmup completes for large
# Qwen3 + FP8 models. A real chat completion only succeeds once vLLM is
# genuinely serving. (Latent bug fix: with Qwen3.6-27B-FP8 the warmup is
# short enough that this hasn't bitten the classifier yet, but the shallow
# probe is the same anti-pattern as in run_extract_keyfacts_job.sh.)
export MODEL
for _attempt in $(seq 1 180); do
  if python3 - <<'PY' >/dev/null 2>&1
import json
import os
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps({
        "model": os.environ["MODEL"],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode(),
    headers={"Content-Type": "application/json"},
)
data = json.loads(urllib.request.urlopen(req, timeout=15).read())
assert "choices" in data, data
PY
  then
    echo "vLLM ready."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

# ── Run the classifier ───────────────────────────────────────────
CLASSIFIER_ARGS=(
  --source "$SOURCE"
  --subset "$SUBSET"
  --input "$INPUT_PATH"
  --output "$OUTPUT_FOR_SHARD"
  --model "$MODEL"
  --base-url http://127.0.0.1:8000/v1
  --api-key EMPTY
  --workers "$WORKERS"
  --temperature "$TEMPERATURE"
)
if [[ -n "$MODE" ]]; then
  CLASSIFIER_ARGS+=(--mode "$MODE")
fi
if [[ -n "$LIMIT" ]]; then
  CLASSIFIER_ARGS+=(--limit "$LIMIT")
fi
if [[ "$GUIDED_JSON" == "0" || "$GUIDED_JSON" == "false" ]]; then
  CLASSIFIER_ARGS+=(--no-guided-json)
fi
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  CLASSIFIER_ARGS+=(--shard "$SHARD_INDEX" "$SHARD_COUNT")
fi

echo "Running: python3 scripts/classify_obgyn.py ${CLASSIFIER_ARGS[*]}"
PYTHONPATH=src python3 scripts/classify_obgyn.py "${CLASSIFIER_ARGS[@]}"

echo "Shard ${SHARD_INDEX}/${SHARD_COUNT} complete: $OUTPUT_FOR_SHARD"
