#!/usr/bin/env bash
# Run inside a Run:ai pod on the LiGHT cluster.
#
# 1. Install/upgrade vllm + openai in user-space.
# 2. Start vLLM (with tensor parallelism) as an OpenAI-compatible server.
# 3. Wait until vLLM actually serves a chat completion (not just /v1/models).
# 4. Loop over $SOURCES (comma-separated; e.g. "whb,afrimedqa_saq,kenya")
#    and run scripts/extract_keyfacts.py once per source against the same
#    in-pod vLLM endpoint. Sharing one vLLM warmup across all sources saves
#    ~40 min of warmup time vs running 3 separate pods.
# 5. Exit; Run:ai tears down the pod (and vLLM with it).
#
# Required env vars:
#   REPO_DIR, MODEL, SHARD_INDEX, SHARD_COUNT
#   SOURCES   comma-separated list (preferred), e.g. "whb,afrimedqa_saq,kenya"
#   SOURCE    single value (legacy / smoke tests); falls back to $SOURCES
#
# Per-source paths are derived as:
#   input  = data/sources/<src>.jsonl   (synced by submit_extract_keyfacts.sh)
#   output = benchmark/v0.2/key_facts/<src>_keyfacts.jsonl
#
# Optional (defaults shown):
#   WORKERS=8, LIMIT="",
#   GUIDED_JSON=1, TEMPERATURE=0.0,
#   MAX_MODEL_LEN=32768, MAX_NUM_SEQS=64,
#   GPU_MEMORY_UTILIZATION=0.90, GDN_PREFILL_BACKEND=triton,
#   TENSOR_PARALLEL_SIZE=8, THINKING_BUDGET=23552

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

# SOURCES is the preferred multi-source variable; SOURCE is kept as a
# single-value fallback for backward compatibility (and smoke tests).
SOURCES="${SOURCES:-${SOURCE:-}}"
if [[ -z "$SOURCES" ]]; then
  echo "ERROR: SOURCES (or SOURCE) must be set" >&2
  exit 1
fi
: "${SHARD_INDEX:?ERROR: SHARD_INDEX must be set}"
: "${SHARD_COUNT:?ERROR: SHARD_COUNT must be set}"

LIMIT="${LIMIT:-}"

export HOME="${RUNAI_HOME:-$REPO_DIR/runai_home}"
export HF_HOME="${HF_HOME:-$REPO_DIR/hf_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$REPO_DIR/python_user}"
export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p logs data benchmark/v0.2/key_facts "$HOME" "$HF_HOME" "$PYTHONUSERBASE"

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

# ── Start vLLM ────────────────────────────────────────────────────
# --reasoning-parser qwen3 makes vLLM parse Qwen3's <think>...</think>
# output into a separate reasoning_content field when present.
#
# NOTE: we deliberately do NOT set
# --structured-outputs-config.enable_in_reasoning=True. In vLLM 0.20.2
# with the qwen3 parser + Qwen3.5-397B-A17B-FP8 + json_schema
# response_format, that flag causes content to come back empty on every
# row (100% failure rate observed on a full whb+saq+kenya run). Without
# the flag, content is a clean JSON answer but reasoning_content is
# always empty (vLLM silently suppresses reasoning when json_schema is
# set). Trade-off: working extraction wins over audit reasoning. Filed
# as a v0.2.1 follow-up to investigate further.
VLLM_LOG="logs/vllm_keyfacts_shard${SHARD_INDEX}.log"
echo "Starting vLLM with model: $MODEL (tensor-parallel-size=$TENSOR_PARALLEL_SIZE)"
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "logs/vllm_keyfacts_shard${SHARD_INDEX}.pid"

echo "Waiting for vLLM to become ready..."
# Send a real /v1/chat/completions ping (max_tokens=1, thinking disabled) —
# /v1/models becomes responsive BEFORE DeepGEMM warmup completes for large
# Qwen3 + FP8 models, so checking it lets the extractor fire while vLLM is
# still warming up and every request errors out. A real chat completion
# only succeeds once vLLM is genuinely serving.
export MODEL
for _attempt in $(seq 1 240); do
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

# ── Run the extractor, looping over SOURCES ──────────────────────
# Sources are processed sequentially against the same in-pod vLLM endpoint.
# Each invocation has its own resumable side-files (row_id check on the
# main output), so a crash partway through one source doesn't lose earlier
# sources.
FAILED_SOURCES=()
SUCCESS_SOURCES=()
for src in $(echo "$SOURCES" | tr ',' ' '); do
  case "$src" in
    kenya|afrimedqa_saq|whb) ;;
    *) echo "WARNING: unknown source '$src', skipping" >&2; continue ;;
  esac
  src_input="data/sources/${src}.jsonl"
  src_output="benchmark/v0.2/key_facts/${src}_keyfacts.jsonl"
  if [[ "$SHARD_COUNT" -gt 1 ]]; then
    src_output="${src_output%.jsonl}_shard${SHARD_INDEX}.jsonl"
  fi

  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "  Extracting source: $src"
  echo "    input:  $src_input"
  echo "    output: $src_output"
  echo "════════════════════════════════════════════════════════════════"

  if [[ ! -f "$src_input" ]]; then
    echo "ERROR: source input not found: $src_input" >&2
    FAILED_SOURCES+=("$src (input missing)")
    continue
  fi

  EXTRACTOR_ARGS=(
    --input "$src_input"
    --output "$src_output"
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
  if PYTHONPATH=src python3 scripts/extract_keyfacts.py "${EXTRACTOR_ARGS[@]}"; then
    SUCCESS_SOURCES+=("$src")
  else
    FAILED_SOURCES+=("$src (extractor exit nonzero)")
    # Continue to next source rather than crashing the whole pod — the
    # other sources may still succeed, and Layer 1 is best-effort per-row
    # anyway (errors are logged per row).
  fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════════"
echo "Succeeded: ${SUCCESS_SOURCES[*]:-(none)}"
echo "Failed:    ${FAILED_SOURCES[*]:-(none)}"
if [[ ${#FAILED_SOURCES[@]} -gt 0 ]]; then
  exit 1
fi
