#!/usr/bin/env bash
# Run inside a Run:ai pod on the LiGHT cluster.
#
# 1. Install/upgrade vllm + openai in user-space.
# 2. Start vLLM (with tensor parallelism) as an OpenAI-compatible server.
# 3. Wait for /v1/models to respond.
# 4. Run scripts/extract_keyfacts.py against the local vLLM endpoint.
#    The driver appends extraction results to OUTPUT_PATH (suffixed with the
#    shard index when sharded).
# 5. Exit; Run:ai tears down the pod (and vLLM with it).
#
# Required env vars:
#   REPO_DIR, SOURCE, INPUT_PATH, OUTPUT_PATH, MODEL,
#   SHARD_INDEX, SHARD_COUNT
#
# Optional (defaults shown):
#   WORKERS=8, LIMIT="",
#   GUIDED_JSON=1, TEMPERATURE=0.0,
#   MAX_MODEL_LEN=32768, MAX_NUM_SEQS=64,
#   GPU_MEMORY_UTILIZATION=0.90, GDN_PREFILL_BACKEND=triton,
#   TENSOR_PARALLEL_SIZE=8

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamabench}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
# 32K context, sized for: input (~4K worst case) + thinking_budget (23.5K)
# + JSON output (~1K) + ~10% buffer. This is the only hard ceiling on
# generation; see submit_extract_keyfacts.sh for rationale.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
WORKERS="${WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GUIDED_JSON="${GUIDED_JSON:-1}"
THINKING_BUDGET="${THINKING_BUDGET:-23552}"

: "${SOURCE:?ERROR: SOURCE must be set}"
: "${INPUT_PATH:?ERROR: INPUT_PATH must be set}"
: "${OUTPUT_PATH:?ERROR: OUTPUT_PATH must be set}"
: "${SHARD_INDEX:?ERROR: SHARD_INDEX must be set}"
: "${SHARD_COUNT:?ERROR: SHARD_COUNT must be set}"

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
VLLM_LOG="logs/vllm_keyfacts_${SOURCE}_shard${SHARD_INDEX}.log"
echo "Starting vLLM with model: $MODEL (tensor-parallel-size=$TENSOR_PARALLEL_SIZE)"
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --language-model-only \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "logs/vllm_keyfacts_${SOURCE}_shard${SHARD_INDEX}.pid"

echo "Waiting for vLLM to become ready..."
for _attempt in $(seq 1 240); do
  if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2).read()
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

# ── Run the extractor ────────────────────────────────────────────
EXTRACTOR_ARGS=(
  --input "$INPUT_PATH"
  --output "$OUTPUT_FOR_SHARD"
  --model "$MODEL"
  --base-url http://127.0.0.1:8000/v1
  --api-key EMPTY
  --workers "$WORKERS"
  --temperature "$TEMPERATURE"
  --thinking-budget "$THINKING_BUDGET"
)
if [[ -n "$LIMIT" ]]; then
  EXTRACTOR_ARGS+=(--limit "$LIMIT")
fi
if [[ "$GUIDED_JSON" == "0" || "$GUIDED_JSON" == "false" ]]; then
  EXTRACTOR_ARGS+=(--no-guided-json)
fi
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  EXTRACTOR_ARGS+=(--shard "$SHARD_INDEX" "$SHARD_COUNT")
fi

echo "Running: python3 scripts/extract_keyfacts.py ${EXTRACTOR_ARGS[*]}"
PYTHONPATH=src python3 scripts/extract_keyfacts.py "${EXTRACTOR_ARGS[@]}"

echo "Shard ${SHARD_INDEX}/${SHARD_COUNT} complete: $OUTPUT_FOR_SHARD"
