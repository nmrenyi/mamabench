# Evaluating Free-Text Medical QA with a Single Reference

A focused literature review supporting the design of mamabench v0.2's open-ended evaluation pipeline.

- **Scope:** evaluation strategies for `(question, single_reference_response)` rows. Rubric-based evaluation (HealthBench-style) is intentionally out of scope here.
- **Sources in scope:** Kenya Clinical Vignettes (284 rows), AfriMed-QA SAQ (37 rows), WHB stumps (20 rows).
- **Model under test:** Gemma 4 E4B + RAG (~8B-class), deployed as a chatbot for nurses and midwives in Zanzibar.
- **Compiled:** 2026-05-13 via literature subagent.

---

## 1. Executive summary

For mamabench v0.2 (~340 free-text rows, single-reference, sub-10B model under test, Kenyan/Zanzibari clinical context), the evidence strongly supports a **decomposed, reference-anchored LLM-as-judge** pipeline with the following shape:

1. **At dataset build time**, extract a small set of "key facts" / "must-cover claims" from each gold reference using a strong LLM, with human spot-check.
2. **At eval time**, prompt a strong judge (GPT-4.1 / Claude Sonnet 4.6 / Gemini 2.5 Pro) to produce a structured verdict along **four fixed axes** — accuracy, completeness (key-fact recall against the extracted list), safety/harm, and contextual appropriateness — using chain-of-thought with explicit reference-grounding. Return per-axis 0–4 scores plus a binary "as-good-as-reference" call.
3. **Average over a 3-judge ensemble** to mute self-preference and family-specific biases. (Position bias is not relevant — there is no pairwise comparison.)
4. **Report**: per-axis means with bootstrap CIs, **key-fact recall as a separate "interpretable" headline number**, **harm rate as a separate non-averaged number**, and a binary as-good-as-reference rate. Surface metrics (BERTScore-F1, ROUGE-L) only as cheap monitoring / sanity-check signals — never as headline scores.

This design is grounded in MEDIC, MedHELM's LLM-jury result (ICC 0.47 vs human 0.43), MORQA's benchmark finding that classical metrics correlate weakly with clinician judgment, MedPaLM 2's 12-axis rubric, and HealthBench's rubric-decomposition logic.

---

## 2. Reference-based metrics — what survives and what doesn't

The 2024–2025 literature is now unambiguous: **BLEU and ROUGE are broken for medical free-text**. MORQA, the most rigorous recent benchmark (16,041 expert judgments across English and Chinese clinical QA, ~2–4 gold references per item), reports Pearson correlations against expert ratings of ~0.06–0.13 for ROUGE-1 and BLEU, vs ~0.38–0.43 for GPT-4o / Gemini-1.5-Pro judges ([MORQA, arXiv 2509.12405](https://arxiv.org/html/2509.12405)). Other clinical-QA studies report similar BLEU/ROUGE/METEOR/BLEURT correlations in the −0.17 to 0.45 band — i.e. occasionally positive but never reliable enough to gate a decision.

**BERTScore-F1 and BLEURT survive only as monitoring signals.** They sit in a 0.20–0.45 correlation range on medical data — better than BLEU but still substantially worse than any modern LLM judge ([MORQA](https://arxiv.org/html/2509.12405); MedHELM reports BERTScore-F1 ICC 0.44 vs clinician 0.43 on its open-ended subset, which is the best reported result for any classical metric but still inferior to the LLM-jury ICC of 0.47 — see [MedHELM, arXiv 2505.23802](https://arxiv.org/abs/2505.23802)).

**MoverScore, BARTScore, and embedding-cosine variants (SBERT, PubMedBERT, MedEmbed) are in the same regime** — semantic-aware, paraphrase-tolerant, but they conflate "the model said similar words" with "the model gave correct, safe advice." They will happily score a hallucinated dose schedule as semantically identical to the reference if it shares vocabulary. They also have no notion of safety or omission — a critical gap, since for an 8B model with RAG you specifically expect *truncated* answers that miss reference content.

**Practical bottom line for mamabench:** report BERTScore-F1 and ROUGE-L in a supplementary table for backwards comparability, but do not use them as a headline. Do not bother fine-tuning embeddings on PubMed — the gains are smaller than the gap to LLM judges.

---

## 3. LLM-as-judge: current state of practice

The 2024–2026 consensus is that LLM-as-judge with explicit reference grounding is the **best automatic option** for free-text medical eval. MedHELM's LLM-jury (ensemble of three judges with task-tailored rubrics) reached ICC 0.47 with clinicians, exceeding clinician–clinician ICC of 0.43 ([MedHELM](https://arxiv.org/abs/2505.23802)). MORQA reaches similar conclusions ([MORQA](https://arxiv.org/html/2509.12405)). A French open-ended medical QA study frames the judge task as **binary semantic equivalence to the reference** and finds this both more reliable and easier to validate than graded scores ([Who Judges the Judge? arXiv 2603.04033](https://arxiv.org/html/2603.04033)).

### Pointwise vs pairwise vs binary

- **Pairwise** is the gold standard for relative model rankings, but it's expensive (O(n²) comparisons), and you do not have a second model output to compare against the reference — your reference is text, not a model. So pairwise is the wrong shape here.
- **Pointwise 1–5 / 1–10** is the most common format. It is *the* well-known source of judge instability: an empirical 2025 study finds Likert-scale judges suffer high variance and weak inter-judge agreement compared to binary or coarse-grained scales ([Empirical Study of LLM-as-Judge Design Choices, arXiv 2506.13639](https://arxiv.org/html/2506.13639v1)).
- **Binary + decomposed** (binary per-claim or per-axis, with the *axes themselves* providing the fine-grained signal) is the emerging best practice. HealthBench's rubric-decomposition logic generalises here: instead of 48k physician criteria, use a small fixed axis set + per-item key facts extracted from the reference ([HealthBench](https://openai.com/index/healthbench/)).

### Known biases and what works

- **Verbosity / length bias.** Long answers get higher scores even when no better. Mitigation: include length in the prompt as a feature to *ignore*; report length-stratified scores.
- **Position bias** doesn't apply to single-reference pointwise eval.
- **Self-preference.** GPT-4 judging GPT-4 outputs inflates scores ~10% ([Self-Preference Bias, arXiv 2410.21819](https://arxiv.org/abs/2410.21819)). Since the model under test is Gemma 4 E4B, self-preference is mostly avoided regardless of judge choice, but using a different family from the model under test (and from any baselines) is still cleaner. Ensembling judges across families (one GPT, one Claude, one Gemini) is the standard fix; a 3-judge majority/mean is what MedHELM does and what we recommend.
- **Cultural / framing bias.** Studies show LLM judges default to Western phrasing norms and may downrate non-Western styles ([Cultural Bias in LLMs, arXiv 2311.14096](https://arxiv.org/pdf/2311.14096); [Framing Bias, arXiv 2601.13537](https://arxiv.org/html/2601.13537)). For Kenyan/Zanzibari content this is a real risk. Mitigation:
  1. Tell the judge in-prompt the response is *for* Kenyan nurses and that Kenyan-English idioms and locally-appropriate phrasing are valid.
  2. Anchor accuracy to the reference *content* not its surface form.
  3. Where possible, validate a sample against Kenyan-clinician ratings (the Kenya Clinical Vignettes paper already used blinded Kenyan physicians on an 11-domain rubric — borrow their axes — [Kenya Vignettes, medRxiv](https://www.medrxiv.org/content/10.1101/2025.10.25.25338798v1)).
- **CoT.** When score descriptions are well-defined, CoT adds little; when axes are subtle, CoT reduces variance and improves human agreement ([Arize evidence-based prompting](https://arize.com/blog/evidence-based-prompting-strategies-for-llm-as-a-judge-explanations-and-chain-of-thought/)). Use it — the marginal cost is negligible at 340 rows.

---

## 4. Key-fact / atomic claim approaches

This is the most under-used trick for *exactly* this setting. FActScore decomposes a generation into atomic facts and checks each against a knowledge source ([FActScore, ACL 2023](https://aclanthology.org/2023.emnlp-main.741/)). The natural inversion for single-reference QA is **key-fact recall**: decompose the *reference* into atomic claims, then ask whether each claim is *present or absent* in the candidate.

This matters for mamabench because the model is a small (~8B) RAG system that, by characterisation, will mostly fail by **omission** rather than hallucination. Surface similarity and even pointwise judges struggle to detect a missed-but-critical fact — "did the response mention checking BP" is a binary that a judge can answer reliably; "rate this response 1–10 on completeness" is not.

AlignScore is the closest off-the-shelf option ([AlignScore, ACL 2023](https://aclanthology.org/2023.acl-long.634/)) — a RoBERTa-based alignment function trained on 4.7M examples that scores claim-vs-context alignment. It has been applied to clinical summarisation. But for free-text *advice* (not summaries), the LLM-extraction-then-LLM-verification pipeline tends to outperform AlignScore in 2024–2025 medical work because the "claims" in clinical advice are often imperative ("counsel about danger signs"), which AlignScore's NLI training data does not cover well.

**Recommended shape:** at dataset build time, use a strong LLM + light human review to extract 3–8 must-cover claims per reference (e.g. "advises immediate referral if BP > 140/90", "checks for proteinuria"). Persist these in the dataset so they are reproducible across teams. At eval time, the judge returns a present/partial/absent verdict per claim. Report **key-fact recall** as a single interpretable headline number. This is also what the AfriMed-QA reviewers effectively did when comparing against expert answers ([AfriMed-QA, arXiv 2411.15640](https://arxiv.org/abs/2411.15640)).

---

## 5. Medical-specific frameworks worth borrowing from

- **MedPaLM / MedPaLM 2** ([Singhal et al., Nature Medicine 2024](https://www.nature.com/articles/s41591-024-03423-7)) used a 12-axis physician rubric: alignment with medical consensus, reading comprehension, knowledge recall, reasoning, inclusion of irrelevant content, *omission of important content*, possibility of harm, likelihood of harm, bias, and three more. Their best-supported observation for our purposes: **separating "omission" from "irrelevant content" is high-yield**. Both are forms of completeness failure but they need opposite remediations.
- **MedHELM** ([Bedi et al., arXiv 2505.23802](https://arxiv.org/abs/2505.23802)) is the strongest published validation that **3-judge LLM-jury with per-task rubrics matches clinician agreement**. Their pattern (binary or short-scale rubric items, judge ensemble, ICC reporting) is directly transferable.
- **MEDIC** ([arXiv 2409.07314](https://arxiv.org/abs/2409.07314)) defines five clinical dimensions: medical reasoning, ethics & bias, language understanding, in-context learning, clinical safety. Most useful contribution is the **distinction between passive safety (refusal) and active safety (catching errors / contraindications)** — Zanzibar nurse-chatbots need *active* safety, since a chatbot that refuses everything is useless.
- **HealthBench** ([OpenAI 2025](https://openai.com/index/healthbench/)) is rubric-based and covered separately; the only thing worth borrowing for a non-rubric setting is its category schema (emergency / global-health / data transformation / hedging-vs-completeness).
- **CSEDB** ([PMC12855988](https://pmc.ncbi.nlm.nih.gov/articles/PMC12855988/)) — weighted consequence axes; useful inspiration for the safety axis.
- **Kenya Clinical Vignettes benchmark** ([medRxiv 2025.10.25](https://www.medrxiv.org/content/10.1101/2025.10.25.25338798v1)) — directly relevant: blinded Kenyan-physician panels, 5-point Likert, 11 domains. Borrow their domains. They found LLMs *outperformed* clinicians on safety and reasoning; calibrate expectations accordingly.
- **CARE-RAG** ([arXiv 2511.15994](https://arxiv.org/html/2511.15994)) and **RAGAS faithfulness** ([Ragas docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)) — relevant because MAMAI is a RAG system. Faithfulness-to-retrieved-context is a *separate* axis from faithfulness-to-reference and is worth measuring if access to the retrieved chunks is available.

For **African-context advice-style responses specifically**, the published literature is thin. AfriMed-QA and the Kenya Vignettes work are the two anchors. Both used African-clinician human eval, not custom automated metrics — meaning mamabench is on the leading edge here and a defensible LLM-judge pipeline with explicit cultural-context prompt instructions is reasonable.

---

## 6. Safety and harm evaluation

A 2025 npj Digital Medicine study found unsafe-response rates of 5–13% across frontier models on patient-posed medical questions, with problematic responses up to 43% ([npj Digital Medicine](https://www.nature.com/articles/s41746-026-02428-5)). Surface metrics catch none of this. Even pointwise judges with a single "quality" score under-weight harm because harm is rare and the scale is dominated by accuracy variance.

**What works for safety:**

1. **Treat safety as a separate axis with an asymmetric scale** — e.g. `{safe, minor concern, potentially harmful, dangerous}`, then *report potentially-harmful + dangerous rate as a separate headline*, not averaged into a quality score. MEDIC's active-safety vs passive-safety distinction matters here.
2. **Prompt the judge to flag specific harm categories** — wrong dose, missed red-flag symptom, contradicts local guidelines, dangerous omission. The Kenya Vignettes rubric uses similar categories.
3. **Auto-detect refusals separately.** Refusal is not safe; it's just not-an-answer.
4. **For the WHB stumps subset specifically**, the references are explicitly written to describe failure modes. Use the reference as a *positive-example specification* of what NOT to do, and have the judge check whether the model exhibited any of the listed failure modes.

---

## 7. Concrete recommendation for mamabench v0.2

### Pipeline

For each `(question, reference)` row:

1. **Build-time (offline, with human review):** use Claude Sonnet 4.6 to extract 3–8 atomic "must-cover claims" from the reference. A second human spot-checks ~30% to validate. Persist as `key_facts: list[str]` in the dataset alongside the reference. This makes the eval reproducible across teams.

2. **Eval-time per row:** Run an **LLM judge ensemble of three models from different families** — `gpt-4.1`, `claude-sonnet-4-6`, `gemini-2.5-pro`. Each judge receives: the question, the reference, the extracted `key_facts`, the candidate response, and a single structured prompt that requests, in JSON:
   - **Per-key-fact verdict:** `present | partial | absent` (with a one-line justification).
   - **Four axis scores on a 0–4 ordinal scale** (Likert is too noisy at 1–10; 0–4 with anchored descriptors is the sweet spot per [arXiv 2506.13639](https://arxiv.org/html/2506.13639v1)):
     - **Accuracy** — correctness vs reference and clinical fact.
     - **Completeness** — coverage of the key facts (largely redundant with key-fact recall but useful as a holistic check).
     - **Safety** — `{safe, minor_concern, potentially_harmful, dangerous}` with explicit instruction to flag wrong dosing, missed red flags, contradiction of WHO/Kenyan MoH guidelines.
     - **Contextual appropriateness** — appropriate for a Kenyan/Zanzibari nurse/midwife audience; do NOT downrate Kenyan-English phrasing.
   - **Binary call:** is the candidate at least as clinically useful as the reference? (yes / no).
   - A short chain-of-thought rationale field above the JSON, to reduce variance.

3. **Aggregation:** average the three judges' scores per axis; majority-vote the binary call and per-key-fact verdicts.

### Reported metrics (headline)

- **Key-fact recall** (% present + 0.5 × partial) — the most interpretable single number.
- **Mean per-axis scores** (accuracy, completeness, safety, contextual) with bootstrap 95% CIs.
- **Harm rate** = % rows with safety ∈ {potentially_harmful, dangerous} — reported separately, not averaged in.
- **As-good-as-reference rate** = % with binary "yes".
- **Refusal rate** — reported separately.
- BERTScore-F1 and ROUGE-L in a supplementary table for backwards comparability.

### Validation

Human-rate ~50 rows (mix of all three subsets) with a Kenyan clinician using the same axes. Report ICC between human and LLM-jury. Anything ≥ 0.4 is acceptable per MedHELM; aim for ≥ 0.45.

### Reproducibility

Pin judge model versions, fix temperature to 0, fix the prompt and key-facts in the dataset release, and publish the judge prompts alongside the leaderboard. Anyone re-running the eval with the same judges should get the same numbers within stochastic noise.

### Why this and not alternatives

- Rubric-based (HealthBench-style) is intentionally out of scope.
- Pure surface metrics fail the MORQA/MedHELM evidence test.
- Single-judge pointwise fails the bias/variance tests.
- Pairwise needs a comparator (no second model output to compare against).

The decomposed reference-anchored ensemble is the lowest-risk option that the 2024–2026 literature actually supports, fits a <$50 cost budget for 340 rows × 3 judges, and gives both a headline number and interpretable diagnostics.

---

## 8. Open design questions

These are not settled by the literature and need a call from the maintainer:

- **Cost & vendor lock-in.** The recommended pipeline requires API access to three frontier judges to reproduce a leaderboard score. Comfortable with that, or also ship a single-judge "lite" mode for users without all three keys?
- **Key-fact extraction is build-time labor.** ~1,700 judge calls + ~100 rows of human spot-check. A single afternoon's work but real; worth investing in since it's the most defensible part of the pipeline.
- **Validation with a Kenyan clinician** is the rate-limiter for actually claiming the pipeline is calibrated. Either acquire access (AfriMed-QA / Kenya Vignettes physician networks?) or ship v0.2 as "pipeline + scores, validation pending" with the human-ICC step deferred to v0.2.1.

---

## Sources

- [MORQA: Benchmarking Evaluation Metrics for Medical Open-Ended Question Answering, arXiv 2509.12405](https://arxiv.org/html/2509.12405)
- [MedHELM: Holistic Evaluation of LLMs for Medical Tasks, arXiv 2505.23802](https://arxiv.org/abs/2505.23802)
- [MedHELM, Nature Medicine](https://www.nature.com/articles/s41591-025-04151-2)
- [MEDIC: Comprehensive Evaluation of Leading Indicators for LLM Safety and Utility in Clinical Applications, arXiv 2409.07314](https://arxiv.org/abs/2409.07314)
- [MedPaLM 2: Toward expert-level medical question answering with large language models, Nature Medicine 2024](https://www.nature.com/articles/s41591-024-03423-7)
- [MedPaLM 2 (preprint), arXiv 2305.09617](https://arxiv.org/abs/2305.09617)
- [HealthBench, OpenAI 2025](https://openai.com/index/healthbench/)
- [HealthBench paper PDF](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf)
- [AfriMed-QA: Pan-African Medical QA Benchmark, arXiv 2411.15640](https://arxiv.org/abs/2411.15640)
- [Kenya Clinical Vignettes Benchmark, medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.10.25.25338798v1)
- [Co-designing Nurse LLM Benchmark Kenya — EPIC](https://www.epicpeople.org/co-designing-a-large-language-model-benchmarking-dataset-for-primary-care-with-nurses-in-kenya/)
- [FActScore, ACL 2023](https://aclanthology.org/2023.emnlp-main.741/)
- [AlignScore, ACL 2023](https://aclanthology.org/2023.acl-long.634/)
- [Who Judges the Judge? French Medical Open-Ended QA, arXiv 2603.04033](https://arxiv.org/html/2603.04033)
- [Same Verdict, Different Reasons: LLM-as-a-Judge and Clinician Disagreement, arXiv 2604.16383](https://arxiv.org/html/2604.16383v1)
- [Automating expert-level medical reasoning evaluation, npj Digital Medicine](https://www.nature.com/articles/s41746-025-02208-7)
- [Evaluating clinical AI summaries with LLM judges, npj Digital Medicine](https://www.nature.com/articles/s41746-025-02005-2)
- [Clinical LLM Evaluation by Expert Review (CLEVER), JMIR AI](https://ai.jmir.org/2025/1/e72153)
- [Self-Preference Bias in LLM-as-Judge, arXiv 2410.21819](https://arxiv.org/abs/2410.21819)
- [Justice or Prejudice: Biases in LLM-as-Judge, arXiv 2410.02736](https://arxiv.org/html/2410.02736v1)
- [Position Bias Systematic Study, arXiv 2406.07791](https://arxiv.org/html/2406.07791v9)
- [Empirical Study of LLM-as-Judge Design Choices, arXiv 2506.13639](https://arxiv.org/html/2506.13639v1)
- [Framing Bias in LLM Judges, arXiv 2601.13537](https://arxiv.org/html/2601.13537)
- [Cultural Bias and Cultural Alignment of LLMs, arXiv 2311.14096](https://arxiv.org/pdf/2311.14096)
- [Evidence-Based Prompting for LLM-as-Judge — Arize](https://arize.com/blog/evidence-based-prompting-strategies-for-llm-as-a-judge-explanations-and-chain-of-thought/)
- [Prometheus 2 — prometheus-eval GitHub](https://github.com/prometheus-eval/prometheus-eval)
- [CSEDB: Clinical Safety-Effectiveness Dual-Track Benchmark, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12855988/)
- [Large language models provide unsafe answers to patient-posed medical questions, npj Digital Medicine](https://www.nature.com/articles/s41746-026-02428-5)
- [HealthBench in Action, arXiv 2509.02594](https://arxiv.org/html/2509.02594v2)
- [CARE-RAG: Clinical Assessment and Reasoning in RAG, arXiv 2511.15994](https://arxiv.org/html/2511.15994)
- [RAGAS Faithfulness Metric](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [A Survey on LLM-as-a-Judge, arXiv 2411.15594](https://arxiv.org/html/2411.15594v6)
- [Trials for LLM-supported clinical decisions in African primary healthcare, Nature Medicine](https://www.nature.com/articles/s41591-025-03815-3)
