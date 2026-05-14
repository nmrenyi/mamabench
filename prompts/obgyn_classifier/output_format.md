## Output format

Return a single JSON object with this exact shape. Do not include any text outside the JSON object.

```
{
  "category": "MATERNAL" | "NEONATAL" | "CHILD_HEALTH" | "SEXUAL_AND_REPRODUCTIVE_HEALTH" | "NONE",
  "rationale": "<≤120 chars explaining the call>"
}
```

Field constraints:

- `category` is exactly one of the five enum values.
- `rationale` is a one-sentence justification, ≤120 characters, referencing the primary medical concept (not patient demographics).
- Return ONLY the JSON object. No prose before or after. No markdown code fences.
