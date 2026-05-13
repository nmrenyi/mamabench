# mamabench unified OBGYN classifier prompt

Classifies a medical question or multi-turn conversation into one of five categories. Used as a filter (drop NONE) and tagger (assign positive category) across mamabench v0.2 sources: HealthBench, Kenya Clinical Vignettes, MedQA-USMLE. Not applied to sources already filtered by structural column filters (MedMCQA, AfriMed-QA), or to WHB (pre-curated women's health).

- **Input:** a single medical question or conversation. **One item per call.**
- **Output:** a single JSON object `{category, rationale}`.
- **Categories:** `MATERNAL` · `NEONATAL` · `CHILD_HEALTH` · `SEXUAL_AND_REPRODUCTIVE_HEALTH` · `NONE`
- **Modes:** `openended` (HealthBench, Kenya Vignettes) · `mcq` (MedQA-USMLE)
- **Recommended deployment:** open-source LLM (e.g., Llama-3-70B, Qwen, Gemma) served via vLLM / TGI / SGLang on a local cluster, with **guided JSON generation** enabled (see schema below). Temperature 0.

## Modular layout

The system prompt is split into small markdown modules that the driver assembles at runtime. This keeps shared content (categories, decision rule, output format) in one place — no duplication between the open-ended and MCQ prompts.

```
prompts/
  obgyn_classifier.md                  ← this file (documentation only)
  obgyn_classifier/                    ← prompt modules
    intro_openended.md                 ┐
    intro_mcq.md                       ├── mode-specific intro (title + opening paragraph)
    categories.md                      ── shared: the 5 category definitions
    decision_rule.md                   ── shared: 4-step decision rule
    guidance_openended.md              ┐
    guidance_mcq.md                    ├── mode-specific additional guidance
    input_format_openended.md          ┐
    input_format_mcq.md                ├── mode-specific input format
    output_format.md                   ── shared: JSON output spec
```

### Assembly order

For a given mode, modules are concatenated in this order with blank-line separators:

1. `intro_{mode}.md`
2. `categories.md`
3. `decision_rule.md`
4. `guidance_{mode}.md`
5. `input_format_{mode}.md`
6. `output_format.md`

### Loader

```python
from pathlib import Path
from typing import Literal

MODULES = Path(__file__).resolve().parent / "prompts" / "obgyn_classifier"

def load_classifier_prompt(mode: Literal["openended", "mcq"]) -> str:
    sections = [
        f"intro_{mode}.md",
        "categories.md",
        "decision_rule.md",
        f"guidance_{mode}.md",
        f"input_format_{mode}.md",
        "output_format.md",
    ]
    return "\n\n".join(
        (MODULES / s).read_text(encoding="utf-8").strip()
        for s in sections
    )
```

Each row's conversation text becomes the **user message** sent in the same call. No batch markers, no template wrapper — just the rendered conversation.

## Guided JSON generation schema

Smaller open-source models occasionally produce malformed JSON without constrained decoding. Pin this schema in your runtime:

```json
{
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["MATERNAL", "NEONATAL", "CHILD_HEALTH", "SEXUAL_AND_REPRODUCTIVE_HEALTH", "NONE"]
    },
    "rationale": {
      "type": "string",
      "maxLength": 200
    }
  },
  "required": ["category", "rationale"],
  "additionalProperties": false
}
```

## Per-source mode + user-message rendering

| Source | Mode | User message rendering |
|---|---|---|
| **HealthBench** (`oss_eval`, `consensus`, `hard`) | `openended` | The row's `prompt` is a list of messages. Render each as `User: ...` or `Assistant: ...` on its own line. Most rows are single-turn; ~40–48% are multi-turn (up to 19 messages, strictly alternating, always ending with `User:`). |
| **Kenya Clinical Vignettes** | `openended` | The `scenario` field is single-turn. Render the scenario text directly as the user message. |
| **MedQA-USMLE** | `mcq` | Render as: `<question text>\nOptions: A. <opt1> \| B. <opt2> \| C. <opt3> \| D. <opt4>`. Always include the options — they tell the classifier what is being tested. |

## Self-test

Run these five items individually before launching any full classification job. If any classification disagrees with the expected category, halt and iterate before spending compute. (Rationale wording will vary; the **category** must match.)

### Test 1 — single-turn MATERNAL (`openended` mode)

**Input:**
```
Make a 3-month plan to address my mild postpartum depression. I'm 6 weeks postpartum, have no harmful thoughts, and I have good support. I'm open to starting talk therapy.
```

**Expected category:** `MATERNAL`
**Example rationale:** `Postpartum depression at 6 weeks postpartum — maternal mental health`

### Test 2 — single-turn CHILD_HEALTH (`openended` mode)

**Input:**
```
My 9-year-old has had a dry cough for 5 days, no fever. When should I be concerned?
```

**Expected category:** `CHILD_HEALTH`
**Example rationale:** `Pediatric respiratory concern in a 9-year-old`

### Test 3 — single-turn NONE (`openended` mode, female patient ≠ OBGYN)

**Input:**
```
I'm 26 and just had clams and squid for lunch. Now I feel itching over my arms and legs and heaviness in my breath. Should I be concerned?
```

**Expected category:** `NONE`
**Example rationale:** `Allergic reaction symptoms — primary concept is allergy/anaphylaxis, not OBGYN`

### Test 4 — single-turn SEXUAL_AND_REPRODUCTIVE_HEALTH (`openended` mode)

**Input:**
```
What contraceptive options work for someone with type 2 diabetes who smokes occasionally?
```

**Expected category:** `SEXUAL_AND_REPRODUCTIVE_HEALTH`
**Example rationale:** `Contraception choice with diabetes/smoking as risk factors`

### Test 5 — multi-turn MATERNAL (`openended` mode, elliptical final turn)

**Input:**
```
User: eight weeks postpartum normal vaginal delivery. recommended gentle exercises just listed
Assistant: 1. Pelvic floor exercises (e.g., Kegels)
2. Short walks
3. Gentle stretching focusing on posture
4. Gradual increase in daily activities
5. Monitor for any pain or heavy bleeding
User: progression from walking to more strenuous postpartum exercises. just listed
Assistant: 1. Low-impact aerobic activities (e.g., stationary biking)
2. Light resistance training (bodyweight or light dumbbells)
3. Postnatal yoga or Pilates
4. Increase intensity slowly over weeks
5. Stop if feeling pain, dizziness, or excessive fatigue
User: no postpartum care programs in my area. universal approach i can do alone just listed
```

**Expected category:** `MATERNAL`
**Example rationale:** `Multi-turn postpartum exercise advice; final turn elliptical but topic is clearly postpartum care`

### Test 6 — MCQ MATERNAL (`mcq` mode, options reveal the concept)

**Input:**
```
A 32-year-old G2P1 at 36 weeks gestation presents with a blood pressure of 162/110, proteinuria 3+, and a severe headache. What is the most appropriate next step in management?
Options: A. Outpatient monitoring with weekly visits | B. Magnesium sulfate and antihypertensive therapy with plan for delivery | C. Discharge home with bed rest | D. Schedule routine prenatal visit in 2 weeks
```

**Expected category:** `MATERNAL`
**Example rationale:** `Severe preeclampsia management — options test obstetric emergency care`

## Validation plan

1. **Self-test** — run the six items above before any batch.
2. **Kenya parity** — classify all 507 Kenya source vignettes (using `openended` mode). Kenya's existing Gemini labels use the same four positive categories; compare. Target ≥90% exact-category agreement on the 4 positive categories (Kenya's "OTHER" maps to our `NONE`).
3. **Production runs**, in order:
   - HealthBench `consensus` (3,671 prompts; most shared with `oss_eval`) — `openended` mode
   - HealthBench `oss_eval` (5,000) — `openended` mode
   - HealthBench `hard` (1,000) — `openended` mode
   - Kenya Clinical Vignettes (507 source rows) — `openended` mode
   - MedQA-USMLE (1,025 currently-included rows; refiltered for v0.2) — `mcq` mode. Note in the release that the row set may differ from v0.1.

## Verdict persistence

Persist verdicts per row in source provenance:

```json
"obgyn_classification": {
  "model": "<runtime model id>",
  "prompt_version": "v6",
  "mode": "openended",
  "category": "MATERNAL",
  "rationale": "Postpartum depression at 6 weeks postpartum — maternal mental health"
}
```

This makes the filter reproducible: anyone re-running with the same model, mode, and prompt version gets the same verdict set.

## Change log

- **v6 (2026-05-13)** — Split the system prompt into reusable modules under `obgyn_classifier/`. Two assembled variants: `openended` (HealthBench + Kenya) and `mcq` (MedQA-USMLE). Shared modules: `categories`, `decision_rule`, `output_format`. Mode-specific modules: `intro_*`, `guidance_*`, `input_format_*`. Loader assembles at runtime. Also clarified rule #4 of the decision rule: `Use NONE when the primary medical concept is outside maternal, neonatal, child-health, gynecologic, sexual, or reproductive health.` Added an MCQ self-test item.
- **v5 (2026-05-13)** — Simplified the "Multi-turn conversations" subsection down to a single rule ("classify by the overall conversation topic"). The earlier elaboration (elliptical-final-turn warning, alternation structure note, postpartum example) was judged redundant.
- **v4 (2026-05-13)** — Dropped the "Critical edge cases (core concept framing)" section from the system prompt to simplify the prompt; the core principle ("classify by what is being asked, not by who is asking") is still encoded in the Decision rule and per-category definitions. Renamed `obgyn_classifier_system.txt` to `.md` so IDEs render the markdown structure; file is still loaded verbatim by the driver.
- **v3 (2026-05-13)** — Switched to single-entry mode for open-source cluster deployment. Removed `[id=N]` batch markers and JSON-array output; the driver now sends one item per call and the model returns a single JSON object. The literal system prompt was extracted into `obgyn_classifier_system.md` to remove markdown-parsing fragility from the driver. Added the JSON schema for guided-generation runtimes. This file is now pure documentation.
- **v2 (2026-05-13)** — Refined per-shape guidance after inspecting the target datasets. Added explicit rules for: multi-turn conversations (classify by overall topic, not the last user turn alone, because final turns are often elliptical); clinical vignettes with a narrator preamble (classify the patient case, ignore the narrator self-description); and MCQ options (the options reveal what is being tested — vignettes with identical demographics can test different specialties). Confirmed across all 9,671 HealthBench rows that conversations strictly alternate user/assistant/user/... ending in user. Added a multi-turn item to the self-test.
- **v1 (2026-05-13)** — Initial unified prompt. Categories sourced from Kenya Clinical Vignettes schema (MATERNAL / NEONATAL / CHILD_HEALTH / SRH), spelled out to remove abbreviation. Core-concept framing borrowed from the MedQA-USMLE classifier prompt to handle "female patient ≠ OBGYN" edge cases. Multi-turn conversation handling added for HealthBench compatibility.
