## Output format

Return a single JSON object with this exact shape. Do not include any text outside the JSON object.

```
{
  "category": "MATERNAL" | "NEONATAL" | "CHILD_HEALTH" | "SEXUAL_AND_REPRODUCTIVE_HEALTH" | "NONE",
  "rationale": "<≤120 chars explaining the call>"
}
```

The rationale must be ≤120 characters and explain WHY this category was chosen, referencing the primary medical concept (not patient demographics).
