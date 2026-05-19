# Safety track research — sources surveyed for a future v0.3 release

*Filed 2026-05-15. Picked up when the v0.2 evaluation surfaces concrete safety gaps.*

`mamabench.md` calls for a third top-level track on top of MCQ and open-ended:

> ### Safety set
> EquityMedQA obstetric items, FairMedQA maternal items, plus MedEqualQA and MedFuzz perturbations applied to the MCQ anchor set.

This document captures source verification done before v0.2 shipped. The track itself is deferred — we want to run the v0.2 evaluation first, see which safety failure modes actually surface in MAMAI's outputs, then build the safety track around those gaps rather than speculatively.

## Source verification (May 2026)

| Source | Type | Size | Accessibility | Recommendation |
|---|---|---|---|---|
| **EquityMedQA** (Google, Feb 2024) | 7 sub-datasets of adversarial bias probes: OMAQ, EHAI, FBRT-Manual, FBRT-LLM, TRINDS, CC-Manual, CC-LLM | Sub-dataset sizes vary | HF: [`katielink/EquityMedQA`](https://huggingface.co/datasets/katielink/EquityMedQA) | **Keep.** Direct adversarial probes most aligned with clinical bias concerns. Needs filtering for the obstetric / maternal / pediatric subset using the same OBGYN classifier pipeline as v0.2. |
| **FairMedQA** (Jan 2026) | Counterfactual QA pairs sourced from USMLE; multi-agent–generated adversarial pairs over privileged vs unprivileged demographic descriptions | **4,806 pairs** | arXiv [2505.19562](https://arxiv.org/html/2505.19562); paper says "publicly available" — verify HF mirror | **Keep.** USMLE is already in our corpus, so the filtering playbook matches v0.2's medqa_usmle refilter. Note: items are **pairs**, requiring a new schema field (`counterfactual_pair_id`) and a new metric (counterfactual consistency rate). Likely v0.5 schema. |
| **MedEqualQA** (Oct 2025) | Pronoun-only counterfactuals (he / she / they) on EquityGuard items | 2,000 seeds × 3 pronoun variants × ~23K CSC ablations ≈ 69,000 items | arXiv [2510.12818](https://arxiv.org/abs/2510.12818) | **Drop.** Pronoun-only swaps are very narrow; little OBGYN-specific coverage; conceptually subsumed by FairMedQA's broader counterfactual mechanic. |
| **MedFuzz** (Microsoft, Jun 2024) | **A method** for adversarial perturbation of MCQ benchmarks — not a dataset | n/a | arXiv [2406.06573](https://arxiv.org/abs/2406.06573); code on GitHub | **Drop.** Implementing the framework is significant engineering, and the "robustness under paraphrasing" angle is already covered by `mamai-quality-evaluation.md §3.2` (generator-side stability eval using MiniCheck). Revisit only if §3.2 turns out to miss a class of robustness failure. |

## Recommended v0.3 scope (when we get there)

Two sources, two filtering pipelines, one new schema field:

1. **EquityMedQA**: filter for OBGYN / maternal / pediatric items using the existing classifier. Emit as `open_ended` (or `open_ended_rubric` if the source includes rubrics). License from the HF page TBD before any redistribution.
2. **FairMedQA**: filter the 4,806 pairs for OBGYN / maternal / pediatric. New schema field `source.metadata.counterfactual_pair_id` joins privileged and unprivileged siblings of the same scenario. New headline metric: counterfactual consistency rate = % of pairs where the model's answer is invariant under the demographic swap.

## Schema implications

Adding FairMedQA pairs cleanly is **schema v0.5** territory. Two options:

- **(A)** New `set_type: "counterfactual_pair"` with a sibling-pointer field. Cleaner semantics; consumers can specifically opt into pair-aware metrics.
- **(B)** Reuse `set_type: "mcq"` (since the underlying items are USMLE MCQ) and add a `source.metadata.counterfactual_pair_id` field only. Smaller schema change but mixes pairs and singletons in the same config.

Defer the call until we're scoping v0.3 in earnest.

## Open questions to resolve before building

- License for EquityMedQA's full set (verify on the HF page).
- License for FairMedQA (paper says publicly available; find the actual download).
- Size of the OBGYN-filtered slice of EquityMedQA. If <50 rows, may not justify a dedicated pipeline; could instead merge into the existing HealthBench / Kenya open-ended track as additional `safety` -flagged rows.
- Whether FairMedQA's "privileged / unprivileged" axis maps usefully to the Zanzibar deployment context (it was generated against US health-system assumptions; some demographic dimensions may not transfer).

## Why this is deferred

The v0.2 evaluation will reveal which safety failure modes MAMAI actually exhibits:
- If the open-ended 3-judge ensemble's harm-rate metric (per `open_ended_evaluation_research.md §7`) already catches the failures we care about, a dedicated safety track may add cost without commensurate value.
- If MiniCheck-based stability eval (`mamai-quality-evaluation.md §3.2`) catches the robustness failures we care about, MedFuzz-style adversarial generation is redundant.
- If demographic-bias failures surface but aren't well-characterized by the existing tracks, that's the case for adding EquityMedQA + FairMedQA.

The honest design discipline is **build evaluation, find gaps, then fill them** — not **anticipate gaps and build for hypotheticals**.
