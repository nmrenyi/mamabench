#!/usr/bin/env bash
# Submit OBGYN-classifier job(s) to the LiGHT cluster (EPFL RCP via runai).
#
# Default: a single job on one H100. Set SHARD_COUNT=N>1 to fan out into N
# parallel jobs that each consume 1/N of the source rows (the in-pod
# classifier passes its own --shard INDEX COUNT).
#
# Required env vars:
#   SOURCE       healthbench | kenya | medqa_usmle
#   INPUT_PATH   local path to the source file (absolute, or relative to the
#                mamabench repo root); rsync'd to the cluster
#
# Common optional env vars (defaults shown):
#   SUBSET             $SOURCE                              recorded in output rows
#   MODE               (auto: openended for hb/kenya, mcq for usmle)
#   OUTPUT_PATH        benchmark/v0.2/classification_verdicts/${SUBSET}.jsonl
#   MODEL              Qwen/Qwen3.6-27B-FP8
#   WORKERS            8
#   LIMIT              (none; use a small int like 10 for smoke tests)
#   GUIDED_JSON        1   (set 0 to disable)
#   TEMPERATURE        0.0
#   SHARD_COUNT        1   (>1 fans out across N parallel runai jobs)
#   NODE_POOL          h100
#
# Examples:
#   # Smoke test (Kenya, 10 rows, 1 worker, Qwen3.5-9B)
#   SOURCE=kenya \
#   INPUT_PATH="/Users/renyi/Downloads/obgyn-qa-collection/kenya-clinical-vignettes/datasets/Prompt responses.xlsx" \
#   MODEL=Qwen/Qwen3.5-9B \
#   WORKERS=1 LIMIT=10 \
#   scripts/submit_classify_obgyn.sh
#
#   # Kenya parity (all 507 rows, 8 workers, Qwen3.5-9B)
#   SOURCE=kenya \
#   INPUT_PATH="/Users/renyi/Downloads/obgyn-qa-collection/kenya-clinical-vignettes/datasets/Prompt responses.xlsx" \
#   MODEL=Qwen/Qwen3.5-9B \
#   WORKERS=8 \
#   scripts/submit_classify_obgyn.sh
#
#   # MedQA-USMLE full (14K rows, sharded x 5)
#   SOURCE=medqa_usmle \
#   INPUT_PATH=~/Downloads/obgyn-qa-collection/medqa-usmle/source/US_qbank.jsonl \
#   MODEL=Qwen/Qwen3.6-27B-FP8 \
#   WORKERS=8 SHARD_COUNT=5 \
#   scripts/submit_classify_obgyn.sh

set -euo pipefail

# ── Cluster config ────────────────────────────────────────────────
JOB_PREFIX="${JOB_PREFIX:-mamabench-classify}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/light/yiren/mamai-guidelines:amd64-cuda-yiren-latest}"
PROJECT="${PROJECT:-light-yiren}"
SERVER="${SERVER:-light}"
SERVER_SCRATCH="${SERVER_SCRATCH:-/mnt/light/scratch/users/yiren/mamabench}"
REPO_DIR="${REPO_DIR:-/lightscratch/users/yiren/mamabench}"
NODE_POOL="${NODE_POOL:-h100}"
# Number of GPUs per pod (also used as tensor-parallel size for vLLM).
# Default 1 fits the original 27B classifier; bump to 8 for 397B FP8.
GPUS="${GPUS:-1}"

# ── Classification config ─────────────────────────────────────────
SOURCE="${SOURCE:?SOURCE required (healthbench|kenya|medqa_usmle)}"
SUBSET="${SUBSET:-$SOURCE}"
MODE="${MODE:-}"
INPUT_PATH="${INPUT_PATH:?INPUT_PATH required (local source file path)}"
OUTPUT_PATH="${OUTPUT_PATH:-benchmark/v0.2/classification_verdicts/${SUBSET}.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-}"
GUIDED_JSON="${GUIDED_JSON:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# ── vLLM config ───────────────────────────────────────────────────
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

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

# Stable cluster path for the source input. Sanitize the basename so it
# survives word-splitting on the remote shell (the ssh+runai env-var path
# strips quoting) and rsync's remote-target parsing.
INPUT_BASENAME="$(basename "$LOCAL_INPUT")"
INPUT_BASENAME_SAFE="${INPUT_BASENAME// /_}"
CLUSTER_INPUT="data/sources/$INPUT_BASENAME_SAFE"

echo "Preparing cluster workspace at $SERVER_SCRATCH..."
ssh "$SERVER" "mkdir -p \
  '$SERVER_SCRATCH/scripts' \
  '$SERVER_SCRATCH/src/mamabench' \
  '$SERVER_SCRATCH/prompts/obgyn_classifier' \
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
  # Job names allow only lower-case alphanumeric + dashes (Kubernetes/runai
  # rule). Sanitize the subset before using it in the job name.
  SUBSET_FOR_JOB="$(echo "$SUBSET" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-')"
  if [[ "$SHARD_COUNT" -gt 1 ]]; then
    JOB_NAME="${JOB_PREFIX}-${SUBSET_FOR_JOB}-shard${shard}"
  else
    JOB_NAME="${JOB_PREFIX}-${SUBSET_FOR_JOB}"
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
    -e SUBSET="$SUBSET" \
    -e MODE="$MODE" \
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
    -e TENSOR_PARALLEL_SIZE="$GPUS" \
    -e SHARD_INDEX="$shard" \
    -e SHARD_COUNT="$SHARD_COUNT" \
    -e HF_HOME="$REPO_DIR/hf_cache" \
    -e PYTHONUSERBASE="$REPO_DIR/python_user" \
    -e RUNAI_HOME="$REPO_DIR/runai_home" \
    -- bash "$REPO_DIR/scripts/run_classify_obgyn_job.sh"
  echo "  Submitted: $JOB_NAME"
done

echo
echo "Monitor:"
echo "  ssh $SERVER 'runai list jobs --project $PROJECT'"
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  for shard in $(seq 0 $((SHARD_COUNT - 1))); do
    echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-${SUBSET_FOR_JOB}-shard${shard} -f --project $PROJECT'"
  done
else
  echo "  ssh $SERVER 'runai logs ${JOB_PREFIX}-${SUBSET_FOR_JOB} -f --project $PROJECT'"
fi
echo
echo "Sync verdicts back when complete:"
echo "  rsync -av '$SERVER_ROOT/$(dirname "$OUTPUT_PATH")/' '$LOCAL_ROOT/$(dirname "$OUTPUT_PATH")/'"
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  base="${OUTPUT_PATH%.jsonl}"
  echo "  # then merge:"
  echo "  cat ${base}_shard{0..$((SHARD_COUNT-1))}.jsonl > ${OUTPUT_PATH}"
fi
