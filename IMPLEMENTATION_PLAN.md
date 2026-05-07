# mamabench Implementation Plan

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

Create a small Python package with validation and sample data only.

Recommended structure:

```text
mamabench/
  README.md
  pyproject.toml
  IMPLEMENTATION_PLAN.md
  configs/
    mamabench_v0.1.yaml
  data/
    raw/
    processed/
    samples/
  scripts/
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
    test_schema.py
    test_validate.py
```

First milestone:

- define the canonical schema
- add 3-5 hand-written sample rows covering MCQ, open-ended, and safety items
- validate those rows
- produce a simple manifest summary

Do not download external datasets during Step 1.

### Step 2: Define the Canonical Schema

Use JSONL as the normalized benchmark format. Each line should represent one benchmark item.

Recommended common fields:

```json
{
  "id": "mamabench_v0.1_mcq_000001",
  "schema_version": "0.1",
  "set_type": "mcq",
  "source_dataset": "MedMCQA",
  "source_id": "original-source-id-if-available",
  "question": "Question text",
  "clinical_domain": "obgyn",
  "age_group": "maternal",
  "task_type": "treatment",
  "safety_type": null,
  "choices": ["A", "B", "C", "D"],
  "answer": "A",
  "answer_index": 0,
  "rubric": null,
  "tags": ["pregnancy", "postpartum"],
  "icd10_codes": [],
  "perturbation_of": null,
  "perturbation_type": null,
  "contamination_risk": "high",
  "license": "unknown",
  "provenance": {
    "source_url": null,
    "source_split": null,
    "source_version": null
  },
  "split": "test"
}
```

Required fields should vary by `set_type`:

- `mcq`: requires `choices`, `answer`, and either `answer_index` or a validated mapping from answer to choices
- `open_ended`: requires `question` and `rubric`; choices should be null or empty
- `safety`: requires `safety_type`; if generated from a perturbation, requires `perturbation_of` and `perturbation_type`

Recommended controlled values:

- `set_type`: `mcq`, `open_ended`, `safety`
- `clinical_domain`: `obgyn`, `neonatal`, `infant`, `reproductive`, `general_maternal`, `unknown`
- `age_group`: `maternal`, `neonate`, `infant`, `adult`, `unknown`
- `task_type`: `diagnosis`, `treatment`, `triage`, `dosage`, `procedure`, `prevention`, `counseling`, `case_reasoning`, `factual_lookup`, `safety`, `unknown`
- `contamination_risk`: `high`, `medium`, `low`, `unknown`
- `split`: `dev`, `test`, `pilot`

### Step 3: Implement Validation Before Ingestion

Build validation utilities before source adapters.

Validation should catch:

- missing required fields
- duplicate `id`
- duplicate source items if `source_dataset` + `source_id` repeats
- malformed MCQ choice lists
- `answer_index` out of bounds
- answer not present in choices when applicable
- missing HealthBench rubric on open-ended rows
- missing `safety_type` on safety rows
- perturbation rows without `perturbation_of`
- unknown controlled-vocabulary values
- invalid split names

The validator should be usable both as a library function and a CLI:

```bash
python scripts/validate_mamabench.py data/samples/sample.jsonl
```

### Step 4: Add a Manifest Builder

Every processed version should include a manifest file:

```text
data/processed/v0.1/manifest.json
```

Manifest should include:

- benchmark version
- schema version
- created timestamp
- total item count
- counts by `set_type`
- counts by `source_dataset`
- counts by `clinical_domain`
- counts by `age_group`
- counts by `task_type`
- counts by `safety_type`
- counts by contamination risk
- split counts
- source dataset versions / license notes where known
- validation status

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
- add source-specific provenance
- add filtering metadata and clinical tags
- tag contamination risk

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
data/processed/v0.1/
  mcq.jsonl
  open_ended.jsonl
  safety.jsonl
  all.jsonl
  manifest.json
  validation_report.json
```

`all.jsonl` should be a concatenation of the three sets, with unique IDs across all rows.

## Reporting Requirements to Preserve

The processed benchmark should support reports broken down by:

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

MedMCQA, MedQA, and MMLU are likely present in many model pretraining corpora. Rows from those sources should be tagged with `contamination_risk: high`.

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
3. Add tiny sample files under `data/samples/`.
4. Implement `validate_mamabench.py`.
5. Implement `summarize_mamabench.py`.
6. Add unit tests for valid and invalid sample rows.
7. Update `README.md` with how to run validation and summarization.

Only after this foundation is working should the agent start downloading or adapting real datasets.
