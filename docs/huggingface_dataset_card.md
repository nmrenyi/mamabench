---
language:
- en
license: other
license_name: per-source-see-card
pretty_name: mamabench
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- multiple-choice
- text-generation
tags:
- medical
- mcq
- open-ended
- rubric-evaluation
- obgyn
- obstetrics
- gynecology
- pediatrics
- reproductive-health
- maternal-health
- neonatal-health
- africa
- benchmark
- evaluation
source_datasets:
- extended|openlifescienceai/medmcqa
- extended|jind11/MedQA
- extended|intronhealth/afrimedqa_v2
- extended|openai/healthbench
- extended|TheLumos/WHB_subset
configs:
- config_name: medmcqa
  data_files:
  - split: test
    path: data/medmcqa.jsonl
- config_name: medqa_usmle
  data_files:
  - split: test
    path: data/medqa_usmle.jsonl
- config_name: afrimedqa
  data_files:
  - split: test
    path: data/afrimedqa.jsonl
- config_name: afrimedqa_saq
  data_files:
  - split: test
    path: data/afrimedqa_saq.jsonl
- config_name: kenya
  data_files:
  - split: test
    path: data/kenya.jsonl
- config_name: whb
  data_files:
  - split: test
    path: data/whb.jsonl
- config_name: healthbench_oss_eval
  data_files:
  - split: test
    path: data/healthbench_oss_eval.jsonl
- config_name: healthbench_consensus
  data_files:
  - split: test
    path: data/healthbench_consensus.jsonl
- config_name: healthbench_hard
  data_files:
  - split: test
    path: data/healthbench_hard.jsonl
---

# mamabench

[GitHub](https://github.com/nmrenyi/mamabench) · 25,997 normalized rows · schema `v0.4` · release `v0.2`

A normalized OBGYN / pediatrics / reproductive-health benchmark for evaluating end-to-end medical question-answering systems. Originally built to evaluate **MAMAI**, a Gemma 4 E4B + RAG medical-advice chatbot for nurses and midwives in Zanzibar.

## Release `v0.2`

This release adds two new evaluation tracks on top of v0.1's multiple-choice MCQ track:

- **Open-ended QA with a reference answer** (`set_type: "open_ended"`) — free-text questions with an expert-written reference response, evaluated by LLM-as-judge or similar reference-aware scorers.
- **Open-ended QA with physician-written rubrics** (`set_type: "open_ended_rubric"`) — HealthBench's evaluation paradigm: each row carries a list of weighted physician-written rubric criteria; a judge LLM scores the model's response against the criteria using HealthBench's weighted-met formula.

### At a glance

| Source | Set type | Rows | License | Notes |
|---|---|---|---|---|
| **MedMCQA** (OBGYN + Pediatrics subset) | mcq | 18,508 | Apache-2.0 | Indian AIIMS / NEET PG entrance exams; unchanged from v0.1 |
| **MedQA-USMLE** (OBGYN subset, **refiltered**) | mcq | 4,301 | MIT | USMLE board questions; **row set differs from v0.1** — refiltered from the full upstream JSONL using the unified OBGYN classifier |
| **AfriMed-QA MCQ** (OBGYN subset) | mcq | 534 | **CC BY-NC-SA 4.0** | Pan-African expert exam questions; unchanged from v0.1 |
| **AfriMed-QA SAQ** (OBGYN subset, **new**) | open_ended | 37 | **CC BY-NC-SA 4.0** | Pan-African expert short-answer questions with reference rationales |
| **Kenya Clinical Vignettes** (**new**) | open_ended | 308 | MIT | 507 nurse-written maternal/neonatal/child/SRH primary-care scenarios with Kenyan-clinician references, filtered to mamabench scope |
| **WHB stumps** (**new**) | open_ended | 20 | **CC BY-SA 4.0** | Women's Health Benchmark expert-crafted "model stumps" — clinical prompts paired with expert justifications |
| **HealthBench `oss_eval`** (OBGYN subset, **new**) | open_ended_rubric | 1,179 | MIT | OpenAI's HealthBench main evaluation, filtered to mamabench scope; physician-written rubrics |
| **HealthBench `consensus`** (OBGYN subset, **new**) | open_ended_rubric | 857 | MIT | HealthBench's high-physician-agreement subset (cluster-template rubrics only) |
| **HealthBench `hard`** (OBGYN subset, **new**) | open_ended_rubric | 253 | MIT | HealthBench's frontier-model stress test, filtered to mamabench scope |
| **Total** | | **25,997** | mixed — see below | |

Plus a non-row side-file: **`side_tables/healthbench_criteria.jsonl`** (11,761 unique rubric criteria, mapping `criterion_id` → text + level + axis). It lives outside `data/` so HF's default `data/*.jsonl` config doesn't merge its schema with the benchmark rows. Loaded directly, not via `load_dataset`.

## License — read this before use

This dataset combines rows from five upstream sources with different licenses. **Each row's `source.dataset` field identifies its license**; the top-level `license: other` reflects this mix rather than choosing one. There is no umbrella license overriding the per-source licenses.

- **`source.dataset == "MedMCQA"`** — Apache-2.0. Commercial use OK.
- **`source.dataset == "MedQA-USMLE"`** — MIT. Commercial use OK.
- **`source.dataset == "AfriMed-QA"`** — **CC BY-NC-SA 4.0**. Non-commercial use only; derivative works must share-alike.
- **`source.dataset == "Kenya-Clinical-Vignettes"`** — MIT. Commercial use OK.
- **`source.dataset == "HealthBench"`** — MIT. Commercial use OK.
- **`source.dataset == "WHB"`** — **CC BY-SA 4.0**. Commercial use OK but derivative works must share-alike (and attribution).

If your use case requires fully permissive licensing (MIT / Apache-2.0 only), filter out AfriMed-QA and WHB rows or load only the permissive configs:

```python
from datasets import load_dataset

# Load only the MIT/Apache configs (skip AfriMed-QA / WHB):
ds_medmcqa = load_dataset("nmrenyi/mamabench", "medmcqa",      revision="v0.2")
ds_usmle   = load_dataset("nmrenyi/mamabench", "medqa_usmle",  revision="v0.2")
ds_kenya   = load_dataset("nmrenyi/mamabench", "kenya",        revision="v0.2")
ds_hb      = load_dataset("nmrenyi/mamabench", "healthbench_oss_eval", revision="v0.2")
```

## Loading

There is **no default config** for v0.2 — `set_type`-conditional fields (`choices`, `rubrics`, etc.) give the three tracks different row shapes that Apache Arrow can't unify into one table. Specify the config you want.

```python
from datasets import load_dataset

# Pick a single source / subset
ds = load_dataset("nmrenyi/mamabench", "medmcqa",             revision="v0.2")  # 18,508
ds = load_dataset("nmrenyi/mamabench", "medqa_usmle",         revision="v0.2")  #  4,301
ds = load_dataset("nmrenyi/mamabench", "afrimedqa",           revision="v0.2")  #    534
ds = load_dataset("nmrenyi/mamabench", "afrimedqa_saq",       revision="v0.2")  #     37
ds = load_dataset("nmrenyi/mamabench", "kenya",               revision="v0.2")  #    308
ds = load_dataset("nmrenyi/mamabench", "whb",                 revision="v0.2")  #     20
ds = load_dataset("nmrenyi/mamabench", "healthbench_oss_eval", revision="v0.2")  #  1,179
ds = load_dataset("nmrenyi/mamabench", "healthbench_consensus",revision="v0.2")  #    857
ds = load_dataset("nmrenyi/mamabench", "healthbench_hard",    revision="v0.2")  #    253
```

Pin `revision="v0.2"` for reproducible evaluation.

## Schema

Each row is a JSON object conforming to **schema `v0.4`** (`schemas/mamabench_v0.4.schema.json` on GitHub). v0.4 extends v0.3 with two new `set_type` values; v0.3 MCQ rows are valid v0.4 rows after a one-line `schema_version` bump.

### Common fields (all rows)

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable mamabench row id, e.g. `mamabench_v0.2_<source>_<source-id-or-hash>` |
| `schema_version` | string | Always `"0.4"` in this release |
| `set_type` | string | One of `"mcq"`, `"open_ended"`, `"open_ended_rubric"` |
| `question` | string \| list[message] | The model-facing prompt. String for single-turn; list of `{role, content}` for HealthBench multi-turn |
| `source` | object | `{dataset, id, answer?, metadata?}` for row-level provenance |

### Set-type-conditional fields

For `set_type == "mcq"`:
- `choices`: list[string] (≥ 2)
- `answer`: the correct option text — equals `choices[answer_index]`
- `answer_index`: zero-based integer

For `set_type == "open_ended"`:
- `answer`: the reference response a downstream evaluator scores model output against

For `set_type == "open_ended_rubric"`:
- `rubrics`: list of `{criterion_id, points}` — each criterion's full text + `level` + `axis` lives in the criteria side-table
- HealthBench's dataset-level canary string is recorded in the manifest, not on every row

## The criteria side-table (HealthBench rubrics)

`open_ended_rubric` rows reference rubric criteria by id; the criterion text and per-criterion metadata live in **`side_tables/healthbench_criteria.jsonl`** (11,761 unique criteria) — outside `data/` so HF's default config doesn't pick it up. Each side-table row:

```json
{
  "criterion_id": "cluster:context_seeking_enough-context_precise",
  "text": "A non-health professional (\"User\") is conversing with...",
  "level": "cluster",
  "axis": "context_awareness"
}
```

`criterion_id` format:
- `cluster:<name>` — one of HealthBench's 33 generic cluster-template criteria (reused across rows).
- `ex:<hash>` — example-level criterion unique to one row; the `<hash>` is sha256(criterion_text)[:12].

Scoring (HealthBench formula): `score = Σ(points × judge_says_met) / Σ(max(points, 0))`. Negative-point criteria penalise harmful behaviours; the denominator is always > 0 by construction.

## Per-source provenance and manifests

The release manifest (`manifests/release_manifest.json`) aggregates all 7 per-source manifests:

- `medmcqa_manifest.json`
- `medqa_usmle_manifest.json` (release note: refilter from upstream)
- `afrimedqa_manifest.json` (MCQ)
- `afrimedqa_saq_manifest.json`
- `kenya_manifest.json`
- `whb_manifest.json`
- `healthbench_manifest.json` (covers the 3 HealthBench subsets)

Each per-source manifest records its OBGYN-classifier filter stats (where applicable), per-category counts, source URL / license, and the dataset-level canary string (for HealthBench).

## MedQA-USMLE row set differs from v0.1

The v0.2 MedQA-USMLE config is **not** a superset of the v0.1 row set. v0.1 used a pre-filtered 1,025-row TSV produced by an earlier Gemini classifier; v0.2 refilters from the upstream raw 14,369-row `US_qbank.jsonl` using mamabench's unified Qwen3.6-27B-FP8 OBGYN classifier (broader scope: includes pediatrics + reproductive health alongside OB/GYN strict).

- v0.1 MedQA-USMLE: 1,025 rows
- v0.2 MedQA-USMLE: 4,301 rows

To reproduce v0.1's row set, pin `revision="v0.1"`. v0.2 readers should expect the new row count.

## Filtering provenance — `obgyn_classification` verdicts

For sources filtered by mamabench's unified OBGYN classifier (HealthBench, Kenya, MedQA-USMLE refilter), each row carries `source.metadata.obgyn_classification`:

```json
"obgyn_classification": {
  "model": "Qwen/Qwen3.6-27B-FP8",
  "prompt_version": "v6",
  "category": "MATERNAL",
  "rationale": "Postpartum depression at 6 weeks postpartum"
}
```

`category` is one of `MATERNAL`, `NEONATAL`, `CHILD_HEALTH`, `SEXUAL_AND_REPRODUCTIVE_HEALTH` (rows with `NONE` are filtered out). The classifier prompt is versioned and committed at `prompts/obgyn_classifier/` in the GitHub repo; the same `(model, prompt_version, source row)` triple reproduces the same verdict.

## AfriMed-QA data quality notes (unchanged from v0.1)

The AfriMed-QA MCQ subset went through invasive normalisation during v0.1 to handle multi-answer rows, ambiguous answer positions, and embedded letter prefixes. See the [GitHub README's "AfriMed-QA data quality notes"](https://github.com/nmrenyi/mamabench#afrimed-qa-data-quality-notes) for the full breakdown. The v0.2 AfriMed-MCQ subset uses the same v0.1 rows (just `schema_version` bumped to `0.4`). The new AfriMed-SAQ subset is short-answer questions with clean text.

## Contamination caveat

MedMCQA, MedQA-USMLE, and AfriMed-QA are present in many model pretraining corpora and should be treated as **high contamination risk** for evaluating proprietary or web-trained models. HealthBench ships with a built-in canary string for one-grep contamination detection — preserved in the per-source manifest at `manifests/healthbench_manifest.json`'s `source_datasets.HealthBench.canary`. Kenya Clinical Vignettes (2025) and WHB (2025) are newer; less contamination risk but verify against publication dates.

## Versioning

Releases are git tags on this dataset's HF repo. Pin a version for reproducible evaluation:

```python
load_dataset("nmrenyi/mamabench", revision="v0.2")
```

`benchmark_version` (the release) and `schema_version` (the row shape) are tracked independently. v0.1 used schema `0.3`; v0.2 uses schema `0.4`. Every row carries `schema_version` so consumers can detect and adapt to schema changes safely across releases.

## Building locally

The pipeline that builds this dataset is open source at <https://github.com/nmrenyi/mamabench>. Per-source adapters, the unified OBGYN classifier prompt + LiGHT-cluster submit scripts, manifest builders, and the release-manifest aggregator are all available. See the GitHub README for full instructions.

## Source datasets

- **MedMCQA** — Pal et al. 2022 ([paper](https://proceedings.mlr.press/v174/pal22a), [HF dataset](https://huggingface.co/datasets/openlifescienceai/medmcqa))
- **MedQA** — Jin et al. 2020 ([paper](https://arxiv.org/abs/2009.13081), [repo](https://github.com/jind11/MedQA))
- **AfriMed-QA** — Olatunji et al. 2024 ([paper](https://arxiv.org/abs/2411.15640), [HF dataset](https://huggingface.co/datasets/intronhealth/afrimedqa_v2))
- **HealthBench** — OpenAI 2025 ([HF dataset](https://huggingface.co/datasets/openai/healthbench))
- **Kenya Clinical Vignettes** — Mwaniki et al. 2025 ([medRxiv preprint](https://doi.org/10.1101/2025.10.25.25338798))
- **Women's Health Benchmark (WHB)** — Gruber et al. 2025 ([paper](https://arxiv.org/abs/2512.17028), [HF subset](https://huggingface.co/datasets/TheLumos/WHB_subset))

The OBGYN / pediatrics / reproductive-health filtering of MedMCQA / MedQA-USMLE / AfriMed-QA was done in [obgyn-qa-collection](https://github.com/nmrenyi/obgyn-qa-collection); mamabench then refilters MedQA-USMLE (and originally classifies HealthBench / Kenya) using its own unified prompt.

## Citation

If you use mamabench, please link to this dataset page (<https://huggingface.co/datasets/nmrenyi/mamabench>) and cite the upstream sources whose data is included:

```bibtex
@inproceedings{pal2022medmcqa,
  title={MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle={Conference on Health, Inference, and Learning},
  pages={248--260},
  year={2022},
  organization={PMLR}
}

@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}

@article{olatunji2024afrimed,
  title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
  author={Olatunji, Tobi and Nimo, Charles and others},
  journal={arXiv preprint arXiv:2411.15640},
  year={2024}
}

@article{arora2025healthbench,
  title={HealthBench: Evaluating Large Language Models Towards Improved Human Health},
  author={Arora, Rahul K. and Wei, Jason and Hicks, Rebecca Soskin and others},
  year={2025},
  publisher={OpenAI}
}

@article{mwaniki2025kenya,
  title={Benchmarking Large Language Models and Clinicians Using Locally Generated Primary Healthcare Vignettes in Kenya},
  author={Mwaniki, Paul and others},
  journal={medRxiv},
  year={2025},
  doi={10.1101/2025.10.25.25338798}
}

@article{gruber2025whb,
  title={A Women's Health Benchmark for Large Language Models},
  author={Gruber, Carolina and others},
  journal={arXiv preprint arXiv:2512.17028},
  year={2025}
}
```
