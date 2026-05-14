# mamabench key-fact extractor

Extracts atomic must-cover **key_facts** from the expert-written reference response of each `open_ended` mamabench row. The key_facts are the per-row scoring rubric for downstream LLM-as-judge evaluation: a separate judge later checks, claim by claim, whether a candidate response covered each one (`present` / `partial` / `absent`).

- **Input:** a single `(question, reference)` pair. **One row per call.**
- **Output:** a single JSON object `{summary, key_facts}`.
- **Applies to:** the three open-ended sources in mamabench v0.2 — **Kenya Clinical Vignettes** (308 rows), **AfriMed-QA SAQ** (37 rows), **WHB** (20 rows). Does **not** apply to HealthBench (rubric-scored) or any MCQ source.
- **Recommended model:** Qwen3.5-397B-A17B-FP8 served via vLLM on a multi-GPU H100 node. Used only at build time, once per dataset release.

## Prompt layout

Single file: [`prompts/keyfact_extractor.md`](../prompts/keyfact_extractor.md). One mode, one input shape, one output shape — no need to swap sections across modes, so no modular split.

The file is organized into these sections (in order):

1. Intro — what this system does
2. Prioritization hierarchy — HealthBench-style ranking
3. Quality rules — five rules every claim must satisfy
4. Extraction procedure — read → identify → atomize
5. Input format — user-message shape
6. Output format — JSON spec
7. Examples — 3 worked examples (Kenya / SAQ / WHB)

### Loader

```python
from mamabench.prompts import load_keyfact_extractor_prompt

system_prompt = load_keyfact_extractor_prompt()
```

The user message is the rendered question + reference pair (see [`format_user_message`](../src/mamabench/adapters/_keyfact_extraction.py)):

```
Question:
<question text>

Reference response:
<reference text>
```

## Guided JSON generation schema

Pin this schema in your runtime (vLLM `response_format` with `json_schema`, TGI grammar, etc.):

```json
{
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "minLength": 1,
      "maxLength": 500
    },
    "key_facts": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "string",
        "minLength": 1,
        "maxLength": 200
      }
    }
  },
  "required": ["summary", "key_facts"],
  "additionalProperties": false
}
```

Note: there is **no fixed upper bound on the number of key_facts**. The reference's clinical content determines the count. Soft expectation: short references produce a handful, longer multi-paragraph references can produce 10+. Pad-or-truncate behaviour is explicitly forbidden in the prompt.

## Output JSONL shape (cluster driver)

The driver `scripts/extract_keyfacts.py` writes **two side-files** per source — a clean rubric file used downstream, and an audit file capturing the model's reasoning content for inspection.

### Main side-file (used by downstream adapters)

`benchmark/v0.2/key_facts/<source>_keyfacts.jsonl`:

```json
{
  "row_id": "mamabench_v0.2_kenya_100",
  "model": "Qwen/Qwen3.5-397B-A17B-FP8",
  "prompt_version": "v1",
  "summary": "Counsel a patient with an unintended pregnancy on an IUD ...",
  "key_facts": [
    "Informs the patient that she is pregnant",
    "Explains that IUD contraception is not 100% effective",
    "..."
  ]
}
```

These rows are folded into each open-ended row's `source.metadata.key_facts` when the open-ended adapters are re-emitted.

### Reasoning side-file (audit only)

`benchmark/v0.2/key_facts/<source>_keyfacts_reasoning.jsonl`:

```json
{
  "row_id": "mamabench_v0.2_kenya_100",
  "model": "Qwen/Qwen3.5-397B-A17B-FP8",
  "prompt_version": "v1",
  "reasoning": "<the model's full thinking-mode chain-of-thought>"
}
```

Joined to the main file by `row_id`. Kept separate so:

1. The main file stays small and clean (~1KB per row vs ~20–80KB with reasoning inlined).
2. Downstream tooling that only needs the rubric doesn't have to ignore the reasoning blob.
3. Reasoning can be reviewed / archived independently of the rubric, including being dropped entirely for shipping a lean HuggingFace release.

### How reasoning is captured

vLLM is started with `--enable-reasoning --reasoning-parser qwen3` (see `scripts/run_extract_keyfacts_job.sh`). Without these flags, Qwen3+ thinking-mode content stays inlined in `message.content`, which breaks the `response_format`/json_schema constraint we use for structured output.

With the flags set, vLLM parses `<think>...</think>` blocks out of the raw completion and exposes the parsed text in `response.choices[0].message.reasoning_content`. The driver reads this field via `make_openai_completer_with_reasoning` and writes it to the reasoning side-file.

If a row comes back with `reasoning_content == None` (the server isn't parsing reasoning out, or thinking was disabled), the reasoning record contains an empty string and the CLI prints a warning at end-of-run noting how many rows had this happen.

### Resume semantics

The main side-file is the **single source of truth** for resume. If a crash interrupts mid-row:

- Reasoning is written first, then main → worst case is an orphan reasoning record with no matching main record.
- On resume, the row is re-extracted from scratch (reasoning side-file is appended-to, so the new reasoning lands alongside the orphan — joined-on-row_id reads will see the latest).

This is intentional: re-running an extraction for a row that previously failed produces a clean main record, and the audit file harmlessly captures both attempts.

## Cluster experiment setup

Sizing for vLLM serving the extractor with **thinking mode enabled** on Qwen3.5-397B-A17B-FP8.

### Measured token counts (v0.2 build)

| Component | Measurement | Median | p95 | Max |
|---|---|---:|---:|---:|
| **System prompt** | measured | 2,900 | 2,900 | 2,900 |
| **User message** = `question + reference` | measured | 310 | 580 | 1,050 |
| **Input total** | measured | **3,200** | **3,500** | **3,950** |
| **Reasoning** (Qwen3+ thinking output) | estimated | 5,000 – 8,000 | 12,000 – 18,000 | 20,000 – 30,000 |
| **Output JSON** (summary + key_facts) | estimated from prompt examples | 250 – 350 | 400 – 600 | ~1,000 |
| **Output total** = reasoning + JSON | estimated | ~6,000 | ~15,000 | ~25,000+ |

User-message length was measured directly from the v0.2 open_ended JSONLs (Kenya 308 rows, AfriMed-SAQ 37, WHB 20). Reasoning token counts are pre-run estimates — actual numbers will come from the smoke test.

### Budget sizing

Two layered caps:

| Knob | Value | Where it lives | What it caps | Enforcement |
|---|---:|---|---|---|
| `MAX_MODEL_LEN` | **32,768** | vLLM server (`vllm serve` flag) | Total context (input + output) | Hard — server stops generation, returns `finish_reason="length"` |
| `thinking_budget` | **23,552** | Qwen `chat_template_kwargs.thinking_budget` | Reasoning portion only | Soft — model self-wraps thinking and proceeds to JSON |

We deliberately do **not** add an OpenAI-API-level `max_tokens` cap. Reasoning: `max_tokens` would just relocate the truncation point if reasoning runs away — it wouldn't preserve the JSON output, because the JSON only gets produced *after* thinking finishes. The only knob that actually keeps JSON valid is `thinking_budget` letting the model wrap up gracefully. `MAX_MODEL_LEN` is the universal hard backstop; an intermediate `max_tokens` adds complexity without preventing the failure mode.

### Worst-case fit

```
MAX_MODEL_LEN          32,768  ←  vLLM context window (hard ceiling)
─ input (worst case)    3,950  ←  system prompt + longest Kenya row
─ thinking budget      23,552  ←  soft cap; model wraps thinking here
─ JSON output (max)     1,000  ←  summary + ~15 key_facts
─────────────────────────
worst-case used        28,502
buffer remaining        4,266  →  ~13% of MAX_MODEL_LEN
```

The buffer absorbs:
1. Model overshooting `thinking_budget` slightly (Qwen3+ self-wrap isn't bit-exact).
2. Longer-than-expected JSON output if the model produces more than 15 key_facts.
3. Any unforeseen token accounting differences between our estimates and vLLM's tokenizer.

### Why these specific numbers

- **`thinking_budget = 23,552` (= 23 × 1024).** Generous enough that the p95 row (estimated 12–18K reasoning tokens) finishes inside the soft cap without rushing. Tighter would risk truncated reasoning on hard rows; looser would burn wall-clock with diminishing quality returns (Qwen3.5 reasoning typically plateaus by ~15K tokens for this task complexity).
- **`MAX_MODEL_LEN = 32,768` (= 32K).** Sized to fit `input_max (3,950) + thinking_budget (23,552) + JSON max (~1,000) = ~28,500` plus ~13% safety buffer. Tighter than 64K so vLLM can fit more concurrent sequences per GPU (better throughput); larger than 28K to leave room for overshoot.

### Failure mode if thinking_budget is ignored

If a Qwen variant or vLLM build silently ignores `thinking_budget`, reasoning can run until `MAX_MODEL_LEN` minus input. With our 32K context and 4K worst-case input, that's ~28.8K tokens of generation — at which point vLLM stops with `finish_reason="length"`. JSON is truncated; the row fails to parse. The CLI logs the failure and the resumable-run feature lets us re-run after bumping `THINKING_BUDGET` and `MAX_MODEL_LEN` together (e.g., to 48K / 64K).

### Throughput model

- 8 H100 GPUs (tensor-parallel = 8 for the 397B model)
- 8 concurrent in-flight requests (`--workers 8`)
- Per-row wall-clock: ~30–90 seconds with thinking enabled (vs. ~5–10s without)
- Full 365-row extraction expected wall-clock: **roughly 45–90 minutes**

### Override knobs

Both budget settings are env-var overridable in the submit script:

```bash
MAX_MODEL_LEN=65536 \
THINKING_BUDGET=49152 \
SOURCE=kenya scripts/submit_extract_keyfacts.sh
```

And per-CLI for local testing:

```bash
python scripts/extract_keyfacts.py \
  --thinking-budget 49152 \
  --input ... --output ... --model ...
```

If the smoke test shows reasoning regularly hitting the 23.5K soft cap on hard rows (visible as truncated JSON / parse errors when MAX_MODEL_LEN is the hard backstop), bump both `THINKING_BUDGET` and `MAX_MODEL_LEN` together — typically to ~48K and ~64K — and re-run.

## Cluster invocation

Local-side submit (mirrors `submit_classify_obgyn.sh`):

```bash
# Smoke test — Kenya, 5 rows, default 397B model on 8 H100s
SOURCE=kenya LIMIT=5 scripts/submit_extract_keyfacts.sh

# Full run — Kenya open-ended set (~308 rows)
SOURCE=kenya scripts/submit_extract_keyfacts.sh

# Smaller model option (single H100)
SOURCE=whb MODEL=Qwen/Qwen3.6-27B-FP8 GPUS=1 WORKERS=1 \
  scripts/submit_extract_keyfacts.sh
```

The in-pod runner `run_extract_keyfacts_job.sh` installs vLLM in user-space, serves the model with the requested `--tensor-parallel-size`, waits for `/v1/models` to respond, then runs the extractor against `http://127.0.0.1:8000/v1`.

## Per-source rendering

| Source | Question field (input JSONL) | Reference field (input JSONL) |
|---|---|---|
| Kenya Clinical Vignettes | `question` (the nurse-written scenario) | `answer` (the Kenyan clinician's response) |
| AfriMed-QA SAQ | `question` (cleaned SAQ question) | `answer` (the SAQ rationale) |
| WHB | `question` (the failure-mode prompt) | `answer` (the expert justification) |

All three sources are read from the existing v0.2 `open_ended` JSONLs in `benchmark/v0.2/`. The extractor does **not** read upstream sources directly — it operates on already-normalized mamabench rows.

## Change log

### v1 (initial release, v0.2 build)

- Three-step reasoning structure: read → identify → atomize.
- Quality rules: atomic / clinically meaningful / groundable-in-reference / ≤200 chars / imperative-friendly.
- Priority hierarchy borrowed from HealthBench's clinical context-seeking ranking.
- No fixed count cap — the reference determines the number of key_facts.
- 3 worked examples spanning Kenya / AfriMed-SAQ / WHB.
