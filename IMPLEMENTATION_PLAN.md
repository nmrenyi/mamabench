# mamabench Implementation Plan

Note: this is the original implementation plan. The current in-use data schema is
the minimal v0.3 schema documented in `schemas/mamabench_v0.3.md` and locked in
`schemas/mamabench_v0.3.schema.json`.

## Project Context

This repository should implement `mamabench`: a reproducible benchmark-building package for evaluating the end-to-end MAMAI system.

The planning documents live in:

- `/Users/renyi/Downloads/mamai-mamabench-docs/README.md`
- `/Users/renyi/Downloads/mamai-mamabench-docs/mamabench.md`
- `/Users/renyi/Downloads/mamai-mamabench-docs/mamai-quality-evaluation.md`
- `/Users/renyi/Downloads/mamai-mamabench-docs/mamai-quality-evaluation-minimal.md`
- `/Users/renyi/Downloads/mamai-mamabench-docs/rag-small-vs-large-literature.md`

MAMAI is a Gemma 4 E4B + RAG medical-advice chatbot for nurses and midwives in Zanzibar. The clinical scope is OBGYN, neonatal / infant care, and reproductive health. The production RAG corpus is intended to contain WHO, Tanzania MOH, Zanzibar MOH, ACOG, RCOG, NICE, FIGO, EmONC, Labour Care Guide, PCPNC, Helping Babies Breathe, and national formulary guidance.

Important scope boundary: `mamaretrieval` is being built by another agent. This repository should focus on `mamabench`, the QA benchmark used for end-to-end system scoring. Do not implement retrieval-label generation here except for thin integration points needed by downstream evaluators.

## What mamabench Should Do

`mamabench` should build a versioned benchmark from existing expert-validated sources. It should produce normalized JSONL artifacts that can be used to compare:

- Gemma 4 E4B without RAG
- Gemma 4 E4B with RAG over the guideline corpus
- Medical fine-tuned models such as MedGemma 4B and Meditron 70B
- Mid-size open models such as Llama 3 70B / Qwen 2.5 72B
- Frontier models such as GPT-5, Claude, and Gemini 2.5
- Optional `+ RAG` variants where the same retriever, chunking, top-k, and prompt are held fixed and only the generator changes

Primary benchmark tracks:

1. MCQ set
   - OBGYN and Pediatrics subsets of MedMCQA using native `subject_name`
   - filtered MedQA using keyword and tag rules
   - filtered MMLU-medical
   - PediatricsMQA, especially neonates and infants
   - PedMedQA, especially neonates 0-3 months and infants

2. Open-ended set
   - HealthBench items tagged with pregnancy ICD-10 `Z33.1`
   - HealthBench obstetric `O*` ICD-10 items
   - HealthBench perinatal `P*` ICD-10 items
   - preserve HealthBench physician-written rubrics

3. Safety set
   - EquityMedQA obstetric items
   - FairMedQA maternal items
   - MedEqualQA and MedFuzz perturbations applied to the MCQ anchor set
   - preserve pair links between original and perturbed items

## Stepwise Implementation Strategy

Implement this repository incrementally. Do not try to ingest every source at once.

### Step 1: Minimal Repository Skeleton

Create a small Python package with schema documentation, validation, source
adapters, and tests. Keep generated benchmark artifacts ignored by Git.

Recommended structure:

```text
mamabench/
  README.md
  pyproject.toml
  IMPLEMENTATION_PLAN.md
  mamabench.json
  schemas/
    mamabench_v0.3.schema.json
    mamabench_v0.3.md
  scripts/
    adapt_medmcqa.py
    validate_mamabench.py
    summarize_mamabench.py
  src/
    mamabench/
      __init__.py
      schema.py
      validate.py
      manifest.py
      io.py
  tests/
    fixtures/
    test_validate.py
    test_manifest.py
    test_medmcqa_adapter.py
  benchmark/
    v0.1/   # generated locally and ignored by Git
```

First milestone:

- define the minimal canonical MCQ schema
- validate source-adapter fixture rows
- produce a simple manifest summary

Do not download external datasets during Step 1.

### Step 2: Define the Canonical Schema

Use JSONL as the normalized benchmark format. Each line should represent one benchmark item.

Current v0.3 common fields:

```json
{
  "id": "mamabench_v0.1_medmcqa_000001",
  "schema_version": "0.3",
  "set_type": "mcq",
  "question": "Question text",
  "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
  "answer": "Choice A",
  "answer_index": 0,
  "source": {
    "dataset": "MedMCQA",
    "id": "original-source-id-if-available",
    "answer": "A"
  }
}
```

The current schema intentionally supports only MCQ rows. For MCQ rows:

- `answer` is the normalized full correct choice text
- `answer_index` is required and must point to that choice
- `source.answer` preserves the source-native answer key when available
- dataset-level metadata such as source URL and license belongs in the manifest,
  not repeated in every row

Open-ended, safety, domain labels, task labels, and split fields should be added
only in a future schema version when a real source and scoring path need them.

### Step 3: Implement Validation Before Ingestion

Build validation utilities before source adapters.

Validation should catch:

- missing required v0.3 fields
- unexpected top-level or `source` fields
- duplicate `id`
- duplicate source items if `source.dataset` + `source.id` repeats
- malformed MCQ choice lists
- `answer_index` out of bounds
- `answer` not equal to `choices[answer_index]`
- malformed `source.answer` values

Future schema versions should add validation for open-ended rubrics, safety
metadata, perturbation links, and controlled labels when those fields exist.

The validator should be usable both as a library function and a CLI:

```bash
python scripts/validate_mamabench.py benchmark/v0.1/medmcqa.jsonl
```

### Step 4: Add a Manifest Builder

Every processed version should include a manifest file:

```text
benchmark/v0.1/manifests/manifest.json
```

Manifest should include:

- benchmark version
- schema version
- created timestamp
- total item count
- counts by `set_type`
- counts by source dataset
- source dataset URL / license notes where known
- full validation report, including issues if any

### Step 5: Implement One MCQ Source Adapter

After schema and validation work, implement exactly one MCQ source adapter first. Choose the easiest locally accessible or easiest legally downloadable source.

The adapter should prove this flow:

```text
raw source -> normalized JSONL -> validation -> manifest summary
```

Do not optimize for full benchmark size yet. Optimize for correctness and reproducibility.

Adapter requirements:

- preserve original source ID
- preserve original answer key
- normalize options into `choices`
- keep row-level source metadata inside the minimal `source` object
- write dataset-level source URL and license into the manifest

### Step 6: Add Filtering and Tagging

Implement deterministic filters for benchmark inclusion.

Domain filters should target:

- pregnancy
- obstetric / OBGYN
- gynecology
- reproductive health
- maternal health
- postpartum
- antenatal / prenatal
- labor and delivery
- neonatal care
- newborn care
- infant care
- breastfeeding where clinically relevant
- PMTCT
- anemia in pregnancy
- malaria in pregnancy
- pre-eclampsia / eclampsia
- postpartum hemorrhage
- neonatal resuscitation

Where datasets expose structured tags, prefer those over keyword matching. Use keyword matching only as a fallback and keep the keyword list in config.

### Step 7: Expand MCQ Sources One at a Time

Once the first adapter is reliable, add the remaining MCQ sources incrementally:

1. MedMCQA
2. MedQA
3. MMLU-medical
4. PediatricsMQA
5. PedMedQA

Each adapter should have:

- unit tests on tiny fixtures
- source-specific documentation
- validation against the canonical schema
- count reporting in the manifest

### Step 8: Add Open-Ended HealthBench Ingestion

HealthBench open-ended items are important because real deployment involves free-form nurse / midwife questions, not just MCQ answers.

Requirements:

- filter by ICD-10 tags:
  - pregnancy `Z33.1`
  - obstetric `O*`
  - perinatal `P*`
- preserve physician-written rubrics exactly as structured data when possible
- keep rubric categories needed for factuality, reasoning, harm, omission, and guideline adherence scoring
- avoid flattening rubrics into plain text unless the source format requires it

### Step 9: Add Safety and Perturbation Sets

Add safety data after the MCQ anchor set is stable.

Requirements:

- ingest EquityMedQA obstetric items
- ingest FairMedQA maternal items
- apply or import MedEqualQA perturbations
- apply or import MedFuzz perturbations
- link every perturbation row back to the original item through `perturbation_of`
- include `perturbation_type`
- support paired evaluation, e.g. original correct but perturbation incorrect

Safety scoring later should support:

- pass rate
- subgroup breakdowns
- robustness under perturbation
- paired consistency

### Step 10: Add Scoring Contracts, Not Full Model Evaluation

This repository should define benchmark artifacts and basic scoring inputs. It does not need to run every model in the evaluation matrix.

Add lightweight scoring helpers for:

- MCQ accuracy
- Brier score input format
- ECE input format
- open-ended rubric result format
- safety pass rate
- paired perturbation consistency

Keep model invocation and full RAG evaluation outside this repository unless the project direction changes.

## Expected Processed Outputs

Target versioned output layout:

```text
benchmark/v0.1/
  medmcqa.jsonl
  manifests/
    medmcqa_manifest.json
  inspection/
```

Additional source files can be added directly under the benchmark release
directory as they are implemented.

## Reporting Requirements to Preserve

The processed benchmark should eventually support reports broken down by:

- MCQ vs open-ended vs safety
- OBGYN vs neonatal vs reproductive health
- source dataset
- contamination risk
- original vs perturbed items
- safety subtype
- model condition, although model outputs may live outside this repo

Metrics expected by the broader MAMAI evaluation:

- MCQ: accuracy, Brier score, ECE
- Open-ended: HealthBench-style rubric scores for factuality, reasoning, harm, omission, guideline adherence
- Safety: pass rate on EquityMedQA / FairMedQA items and robustness under MedEqualQA / MedFuzz perturbations

## Contamination Caveat

MedMCQA, MedQA, and MMLU are likely present in many model pretraining corpora.
The current v0.3 schema does not store row-level contamination labels. Record
dataset-level contamination caveats in documentation or manifests for now, and
add a row-level field in a future schema version only if scoring uses it.

To support contamination-adjusted reporting:

- preserve raw source identity
- create or import paraphrased MedFuzz variants for a subset
- report raw and perturbation-adjusted results separately

## Non-Goals for Initial Versions

Do not implement these in the first pass:

- production RAG retrieval
- `mamaretrieval` query/label generation
- model-serving infrastructure
- full frontier-model judging
- Android/on-device latency evaluation
- guideline corpus chunking
- clinical adjudication UI

## Immediate Next Task for the Dedicated Agent

Start with Step 1 only:

1. Create the Python package skeleton.
2. Define the schema in `src/mamabench/schema.py`.
3. Add tiny test fixtures under `tests/fixtures/` when examples are needed.
4. Implement `validate_mamabench.py`.
5. Implement `summarize_mamabench.py`.
6. Add unit tests for valid and invalid sample rows.
7. Update `README.md` with how to run validation and summarization.

Only after this foundation is working should the agent start downloading or adapting real datasets.
