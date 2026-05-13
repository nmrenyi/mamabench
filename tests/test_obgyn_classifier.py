from __future__ import annotations

import json
import unittest

from mamabench.obgyn_classifier import (
    CATEGORIES,
    VERDICT_JSON_SCHEMA,
    ClassifierError,
    classify_row,
    parse_verdict,
    vllm_guided_json_extra_body,
)


class ParseVerdictTests(unittest.TestCase):
    def test_plain_json_object(self) -> None:
        verdict = parse_verdict(
            '{"category": "MATERNAL", "rationale": "Postpartum depression at 6 weeks postpartum"}'
        )
        self.assertEqual(verdict["category"], "MATERNAL")
        self.assertEqual(verdict["rationale"], "Postpartum depression at 6 weeks postpartum")

    def test_strips_fenced_code_block_with_json_tag(self) -> None:
        raw = '```json\n{"category": "NEONATAL", "rationale": "Newborn jaundice on day 4"}\n```'
        verdict = parse_verdict(raw)
        self.assertEqual(verdict["category"], "NEONATAL")

    def test_strips_fenced_code_block_without_language_tag(self) -> None:
        raw = '```\n{"category": "NONE", "rationale": "Cardiology question, not OBGYN"}\n```'
        verdict = parse_verdict(raw)
        self.assertEqual(verdict["category"], "NONE")

    def test_tolerates_surrounding_whitespace(self) -> None:
        raw = '   \n\n{"category": "CHILD_HEALTH", "rationale": "Pediatric respiratory case"}\n\n  '
        verdict = parse_verdict(raw)
        self.assertEqual(verdict["category"], "CHILD_HEALTH")

    def test_invalid_json_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ClassifierError, "could not parse JSON"):
            parse_verdict("MATERNAL — postpartum depression")

    def test_non_object_json_raises(self) -> None:
        with self.assertRaisesRegex(ClassifierError, "non-object JSON"):
            parse_verdict('["MATERNAL"]')

    def test_unknown_category_raises_with_listed_valid_values(self) -> None:
        with self.assertRaisesRegex(ClassifierError, "invalid category 'OBGYN'"):
            parse_verdict('{"category": "OBGYN", "rationale": "obgyn case"}')

    def test_missing_rationale_raises(self) -> None:
        with self.assertRaisesRegex(ClassifierError, "rationale must be a string"):
            parse_verdict('{"category": "MATERNAL"}')


class ClassifyRowTests(unittest.TestCase):
    def test_sends_system_and_user_messages_in_order(self) -> None:
        captured: list[list[dict[str, str]]] = []

        def fake_complete(messages: list[dict[str, str]]) -> str:
            captured.append(list(messages))
            return '{"category": "MATERNAL", "rationale": "ok"}'

        verdict = classify_row(
            complete=fake_complete,
            system_prompt="SYS",
            user_message="I am 6 weeks postpartum...",
        )

        self.assertEqual(verdict, {"category": "MATERNAL", "rationale": "ok"})
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            captured[0],
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "I am 6 weeks postpartum..."},
            ],
        )

    def test_propagates_classifier_error_from_bad_output(self) -> None:
        def fake_complete(messages: list[dict[str, str]]) -> str:
            return "not json at all"

        with self.assertRaises(ClassifierError):
            classify_row(
                complete=fake_complete,
                system_prompt="SYS",
                user_message="any",
            )


class SchemaTests(unittest.TestCase):
    def test_schema_categories_match_constant(self) -> None:
        self.assertEqual(
            set(VERDICT_JSON_SCHEMA["properties"]["category"]["enum"]),
            CATEGORIES,
        )

    def test_schema_requires_both_fields(self) -> None:
        self.assertEqual(
            set(VERDICT_JSON_SCHEMA["required"]),
            {"category", "rationale"},
        )
        self.assertFalse(VERDICT_JSON_SCHEMA["additionalProperties"])

    def test_vllm_guided_json_uses_default_schema(self) -> None:
        eb = vllm_guided_json_extra_body()
        self.assertIn("guided_json", eb)
        self.assertEqual(eb["guided_json"], VERDICT_JSON_SCHEMA)

    def test_vllm_guided_json_accepts_custom_schema(self) -> None:
        custom = {"type": "object", "properties": {}}
        eb = vllm_guided_json_extra_body(custom)
        self.assertEqual(eb["guided_json"], custom)


if __name__ == "__main__":
    unittest.main()
