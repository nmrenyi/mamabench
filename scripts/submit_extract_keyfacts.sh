#!/usr/bin/env bash
# Submit a key-fact extraction job to the LiGHT cluster (EPFL RCP via runai).
#
# Extracts atomic must-cover key_facts from one or more of the v0.2
# open_ended mamabench JSONL artifacts (Kenya / AfriMed-SAQ / WHB), one
# row at a time, using a vLLM-served LLM. Multi-source mode runs all
# listed sources sequentially against a single in-pod vLLM, saving the
# warmup cost (~20 min per pod) that would otherwise be paid per-source.
# Defaults to Qwen3.5-397B-A17B-FP8 on a multi-GPU H200 node.
#
# Required env vars (one of):
#   SOURCES      comma-separated list of: kenya, afrimedqa_saq, whb
#                  (e.g. "whb,afrimedqa_saq,kenya" for the full v0.2 build)
#   SOURCE       single value (backward compat); use SOURCES for multi.
#
# Common optional env vars (defaults shown):
#   MODEL              Qwen/Qwen3.5-397B-A17B-FP8
#   WORKERS            8
#   LIMIT              (none; use a small int like 5 for smoke tests)
#   GUIDED_JSON        1   (set 0 to disable)
#   TEMPERATURE        0.0
#   SHARD_COUNT        1   (>1 fans out across N parallel runai jobs)
#   NODE_POOL          h100
#   GPUS               8   (tensor-parallel size for vLLM; 397B FP8 needs ~5+)
#
# Per-source paths are fixed:
#   local input:   benchmark/v0.2/<src>.jsonl
#   cluster input: data/sources/<src>.jsonl
#   output:        benchmark/v0.2/key_facts/<src>_keyfacts.jsonl
#                  benchmark/v0.2/key_facts/<src>_keyfacts_reasoning.jsonl
#
# Examples:
#   # Full build: all 3 open-ended sources in a single pod
#   SOURCES=whb,afrimedqa_saq,kenya \
#   scripts/submit_extract_keyfacts.sh
#
#   # Smoke test: 5 WHB rows
#   SOURCE=whb LIMIT=5 \
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
# Resolve and validate the source list.
SOURCES="${SOURCES:-${SOURCE:-}}"
if [[ -z "$SOURCES" ]]; then
  echo "ERROR: SOURCES (or SOURCE) required: comma-separated subset of {kenya,afrimedqa_saq,whb}" >&2
  exit 1
fi
SOURCES_LIST=$(echo "$SOURCES" | tr ',' ' ')
for src in $SOURCES_LIST; do
  case "$src" in
    kenya|afrimedqa_saq|whb) ;;
    *) echo "ERROR: unknown source '$src' (allowed: kenya, afrimedqa_saq, whb)" >&2; exit 1 ;;
  esac
done

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-}"
GUIDED_JSON="${GUIDED_JSON:-0}"
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

# Verify each source's local input file exists before any cluster work.
for src in $SOURCES_LIST; do
  local_input="$LOCAL_ROOT/benchmark/v0.2/${src}.jsonl"
  if [[ ! -f "$local_input" ]]; then
    echo "ERROR: input file not found for source '$src': $local_input" >&2
    exit 1
  fi
done

echo "Preparing cluster workspace at $SERVER_SCRATCH..."
ssh "$SERVER" "mkdir -p \
  '$SERVER_SCRATCH/scripts' \
  '$SERVER_SCRATCH/src/mamabench' \
  '$SERVER_SCRATCH/prompts' \
  '$SERVER_SCRATCH/data/sources' \
  '$SERVER_SCRATCH/logs' \
  '$SERVER_SCRATCH/benchmark/v0.2/key_facts'"

echo "Syncing repo to cluster..."
rsync -av --delete --exclude="__pycache__/" "$LOCAL_ROOT/scripts/" "$SERVER_ROOT/scripts/"
rsync -av --delete --exclude="__pycache__/" "$LOCAL_ROOT/src/" "$SERVER_ROOT/src/"
rsync -av --delete "$LOCAL_ROOT/prompts/" "$SERVER_ROOT/prompts/"

echo "Syncing source inputs for: $SOURCES_LIST"
for src in $SOURCES_LIST; do
  rsync -av "$LOCAL_ROOT/benchmark/v0.2/${src}.jsonl" "$SERVER_ROOT/data/sources/${src}.jsonl"
done

# Build the job-name tag from the source list.
# - Single source → just the source name (e.g. "whb")
# - Multi-source  → hyphen-joined (e.g. "whb-afrimedqa-saq-kenya")
N_SOURCES=$(echo "$SOURCES_LIST" | wc -w | tr -d ' ')
if [[ "$N_SOURCES" -eq 1 ]]; then
  SOURCES_TAG="$(echo "$SOURCES_LIST" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
else
  SOURCES_TAG="$(echo "$SOURCES_LIST" | tr ' ' '-' | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
fi

echo "Submitting $SHARD_COUNT job(s) for sources: $SOURCES_LIST"
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  if [[ "$SHARD_COUNT" -gt 1 ]]; then
    JOB_NAME="${JOB_PREFIX}-${SOURCES_TAG}-shard${shard}"
  else
    JOB_NAME="${JOB_PREFIX}-${SOURCES_TAG}"
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
    -e SOURCES="$SOURCES" \
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
    echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-${SOURCES_TAG}-shard${shard} -f --project $PROJECT'"
  done
else
  echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-${SOURCES_TAG} -f --project $PROJECT'"
fi
echo
echo "Sync key_facts back when complete:"
echo "  rsync -av '$SERVER_ROOT/benchmark/v0.2/key_facts/' '$LOCAL_ROOT/benchmark/v0.2/key_facts/'"
