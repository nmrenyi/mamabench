#!/usr/bin/env bash
# Submit a key-fact extraction job to the LiGHT cluster (EPFL RCP via runai).
#
# Extracts atomic must-cover key_facts from one of the v0.2 open_ended
# mamabench JSONL artifacts (Kenya / AfriMed-SAQ / WHB), one row at a
# time, using a vLLM-served LLM. Defaults to Qwen3.5-397B-A17B-FP8 on a
# multi-GPU H100 node.
#
# Required env vars:
#   SOURCE       kenya | afrimedqa_saq | whb
#   INPUT_PATH   local path to the v0.2 open_ended JSONL (absolute, or
#                relative to the mamabench repo root); rsync'd to the cluster.
#                Defaults: benchmark/v0.2/${SOURCE}.jsonl
#
# Common optional env vars (defaults shown):
#   OUTPUT_PATH        benchmark/v0.2/key_facts/${SOURCE}_keyfacts.jsonl
#   MODEL              Qwen/Qwen3.5-397B-A17B-FP8
#   WORKERS            8
#   LIMIT              (none; use a small int like 5 for smoke tests)
#   GUIDED_JSON        1   (set 0 to disable)
#   TEMPERATURE        0.0
#   SHARD_COUNT        1   (>1 fans out across N parallel runai jobs)
#   NODE_POOL          h100
#   GPUS               8   (tensor-parallel size for vLLM; 397B FP8 needs ~5+)
#
# Examples:
#   # Smoke test (Kenya, 5 rows)
#   SOURCE=kenya LIMIT=5 \
#   scripts/submit_extract_keyfacts.sh
#
#   # Full Kenya open-ended set
#   SOURCE=kenya \
#   scripts/submit_extract_keyfacts.sh
#
#   # WHB (20 rows, 1 worker, smaller model for budget)
#   SOURCE=whb MODEL=Qwen/Qwen3.6-27B-FP8 GPUS=1 WORKERS=1 \
#   scripts/submit_extract_keyfacts.sh

set -euo pipefail

# ── Cluster config ────────────────────────────────────────────────
JOB_PREFIX="${JOB_PREFIX:-mamabench-keyfacts}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamabench}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamabench}"
NODE_POOL="${NODE_POOL:-h100}"
GPUS="${GPUS:-8}"

# ── Extraction config ─────────────────────────────────────────────
SOURCE="${SOURCE:?SOURCE required (kenya|afrimedqa_saq|whb)}"
case "$SOURCE" in
  kenya|afrimedqa_saq|whb) ;;
  *) echo "ERROR: SOURCE must be one of kenya, afrimedqa_saq, whb (got '$SOURCE')" >&2; exit 1 ;;
esac

INPUT_PATH="${INPUT_PATH:-benchmark/v0.2/${SOURCE}.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-benchmark/v0.2/key_facts/${SOURCE}_keyfacts.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-}"
GUIDED_JSON="${GUIDED_JSON:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# ── vLLM config ───────────────────────────────────────────────────
# 32K context, sized for: worst-case input (~4K) + thinking_budget (23.5K) +
# JSON output (~1K) + ~10% buffer. This is the ONLY hard cap on generation
# length — if reasoning overshoots thinking_budget far enough to hit
# MAX_MODEL_LEN, the request fails with finish_reason="length" and JSON is
# truncated. The soft cap (thinking_budget) is what actually preserves
# output validity; a separate max_tokens hard cap would just relocate the
# truncation point without preventing the failure mode.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

# ── Generation budget ─────────────────────────────────────────────
# Soft cap on the reasoning portion only — Qwen3+ self-wraps thinking when
# approaching this and proceeds to emit the JSON answer. Leaves ~5K below
# MAX_MODEL_LEN (after worst-case input) for the JSON output plus any
# overshoot of the soft cap.
THINKING_BUDGET="${THINKING_BUDGET:-23552}"

# ── Sharding ──────────────────────────────────────────────────────
SHARD_COUNT="${SHARD_COUNT:-1}"
if ! [[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SHARD_COUNT must be a positive integer (got '$SHARD_COUNT')" >&2
  exit 1
fi

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="$SERVER:$SERVER_SCRATCH"

# Resolve local input path
case "$INPUT_PATH" in
  /*) LOCAL_INPUT="$INPUT_PATH" ;;
  ~*) LOCAL_INPUT="${INPUT_PATH/#\~/$HOME}" ;;
  *)  LOCAL_INPUT="$LOCAL_ROOT/$INPUT_PATH" ;;
esac

if [[ ! -f "$LOCAL_INPUT" ]]; then
  echo "ERROR: input file not found: $LOCAL_INPUT" >&2
  exit 1
fi

# Stable cluster path for the source input.
INPUT_BASENAME="$(basename "$LOCAL_INPUT")"
INPUT_BASENAME_SAFE="${INPUT_BASENAME// /_}"
CLUSTER_INPUT="data/sources/$INPUT_BASENAME_SAFE"

echo "Preparing cluster workspace at $SERVER_SCRATCH..."
ssh "$SERVER" "mkdir -p \
  '$SERVER_SCRATCH/scripts' \
  '$SERVER_SCRATCH/src/mamabench' \
  '$SERVER_SCRATCH/prompts/keyfact_extractor' \
  '$SERVER_SCRATCH/data/sources' \
  '$SERVER_SCRATCH/logs' \
  '$SERVER_SCRATCH/$(dirname "$OUTPUT_PATH")'"

echo "Syncing repo + source input to cluster..."
rsync -av --delete --exclude="__pycache__/" "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av --delete --exclude="__pycache__/" "$LOCAL_ROOT/src/" "$SERVER_ROOT/src/"
rsync -av --delete "$LOCAL_ROOT/prompts/" "$SERVER_ROOT/prompts/"
rsync -av "$LOCAL_INPUT" "$SERVER_ROOT/$CLUSTER_INPUT"

echo "Submitting $SHARD_COUNT job(s)..."
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  SOURCE_FOR_JOB="$(echo "$SOURCE" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
  if [[ "$SHARD_COUNT" -gt 1 ]]; then
    JOB_NAME="${JOB_PREFIX}-${SOURCE_FOR_JOB}-shard${shard}"
  else
    JOB_NAME="${JOB_PREFIX}-${SOURCE_FOR_JOB}"
  fi
  ssh "$SERVER" "runai delete job '$JOB_NAME' --project '$PROJECT' >/dev/null 2>&1 || true"

  ssh "$SERVER" runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --pvc light-scratch:/lightscratch \
    --gpu "$GPUS" \
    --cpu 16 --cpu-limit 16 \
    --memory 256G --memory-limit 256G \
    --large-shm \
    --node-pool "$NODE_POOL" \
    --project "$PROJECT" \
    --run-as-uid 296712 \
    --run-as-gid 84257 \
    --backoff-limit 0 \
    -e REPO_DIR="$REPO_DIR" \
    -e SOURCE="$SOURCE" \
    -e INPUT_PATH="$CLUSTER_INPUT" \
    -e OUTPUT_PATH="$OUTPUT_PATH" \
    -e MODEL="$MODEL" \
    -e WORKERS="$WORKERS" \
    -e LIMIT="$LIMIT" \
    -e GUIDED_JSON="$GUIDED_JSON" \
    -e TEMPERATURE="$TEMPERATURE" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    -e GDN_PREFILL_BACKEND="$GDN_PREFILL_BACKEND" \
    -e THINKING_BUDGET="$THINKING_BUDGET" \
    -e TENSOR_PARALLEL_SIZE="$GPUS" \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e PYTHONUSERBASE="$REPO_DIR/python_user" \
    -e RUNAI_HOME="$REPO_DIR/runai_home" \
    -- bash "$REPO_DIR/scripts/run_extract_keyfacts_job.sh"
  echo "  Submitted: $JOB_NAME"
done

echo
echo "Monitor:"
echo "  ssh $SERVER 'runai list jobs --project $PROJECT'"
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  for shard in $(seq 0 $((SHARD_COUNT - 1))); do
    echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-${SOURCE_FOR_JOB}-shard${shard} -f --project $PROJECT'"
  done
else
  echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-${SOURCE_FOR_JOB} -f --project $PROJECT'"
fi
echo
echo "Sync key_facts back when complete:"
echo "  rsync -av '$SERVER_ROOT/$(dirname "$OUTPUT_PATH")/' '$LOCAL_ROOT/$(dirname "$OUTPUT_PATH")/'"
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  base="${OUTPUT_PATH%.jsonl}"
  echo "  # then merge:"
  echo "  cat ${base}_shard{0..$((SHARD_COUNT-1))}.jsonl > ${OUTPUT_PATH}"
fi
