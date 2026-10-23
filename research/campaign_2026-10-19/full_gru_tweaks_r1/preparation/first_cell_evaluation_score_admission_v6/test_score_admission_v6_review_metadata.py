"""Metadata-only checks for the conditional score-admission V6 source pins."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import ModuleType

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
EXPORT_PACKET = PREPARATION / "first_cell_evaluation_export_v3"
SOURCE_REVIEW = EXPORT_PACKET / "EXPORT_SOURCE_REVIEW_V3.json"
BODY_V5 = (
    PREPARATION
    / "first_cell_evaluation_score_admission_v5"
    / ("SCORE_ADMISSION_BODY_TEMPLATE_V5.py")
)
BODY_V6 = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V6.py"
RENDERER = PACKET / "render_score_admission_v6.py"
PLAN = PACKET / "SCORE_ADMISSION_BINDING_PLAN_V6.json"
REPOSITORY = PACKET.parents[4]
SOURCE_REVIEW_BYTES = 7_284
SOURCE_REVIEW_SHA256 = "ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a"


def _load_renderer() -> ModuleType:
    """Load only the local pure renderer definitions; do not call render or main."""
    spec = importlib.util.spec_from_file_location("score_admission_v6_renderer", RENDERER)
    if spec is None or spec.loader is None:
        raise AssertionError("V6 renderer module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _normalize_v6_body(tree: ast.Module) -> ast.Module:
    """Map only the documented V3 producer/scorer pins back to their V5 values."""
    replacements: dict[tuple[type, object], tuple[object, int]] = {
        (int, 7_284): (10_250, 1),
        (str, "7784c91713dbfd7d11ec3cfd276eb8a62607289067999f4ecd5c03a3f9a18038"): (
            "09155537feec1d9cd398f6a729ab88201b82e814c1a8c4ac796fe4283940a90d",
            1,
        ),
        (str, "ffe8a6e7aad38b131a78783504ef55c761d4e8037c13bac279f20c5e55f6249e"): (
            "1b624ad568b436615f01f50e60567ece4a483ae9cc1def69ab2277aa0a392604",
            1,
        ),
        (str, "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730"): (
            "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9",
            1,
        ),
        (str, "ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a"): (
            "f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af",
            1,
        ),
        (str, "EXPORT_BINDING_V3.json"): ("EXPORT_BINDING_V1.json", 3),
        (str, "computer-vision-homework-fg19-first-cell-export-v2.service"): (
            "computer-vision-homework-fg19-first-cell-export-v1.service",
            1,
        ),
        (str, "fg19_first_cell_f0_s1001_evaluation_v2"): (
            "fg19_first_cell_f0_s1001_evaluation_v1",
            1,
        ),
    }
    seen = {key: 0 for key in replacements}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            key = (type(node.value), node.value)
            if key in replacements:
                replacement, _ = replacements[key]
                seen[key] += 1
                node.value = replacement
    expected = {key: count for key, (_, count) in replacements.items()}
    if seen != expected:
        raise AssertionError(f"V6 body adaptation counts differ: {seen!r}")
    return tree


class ScoreAdmissionV6ReviewMetadata(unittest.TestCase):
    def test_actual_v3_source_review_pin_and_file_array_are_accepted(self) -> None:
        raw = SOURCE_REVIEW.read_bytes()
        self.assertEqual(
            (len(raw), hashlib.sha256(raw).hexdigest()),
            (SOURCE_REVIEW_BYTES, SOURCE_REVIEW_SHA256),
        )
        review = _load_renderer().validate_source_review_bytes(raw)
        self.assertEqual(review["schema"], "fg19.first-cell-export-source-review.v3")
        self.assertEqual(review["status"], "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR")
        self.assertEqual(len(review["reviewed_files"]), 11)

    def test_renderer_resolves_exact_pinned_review_path(self) -> None:
        bound = _load_renderer()._pinned_source_review()
        self.assertEqual(
            bound,
            {
                "path": str(SOURCE_REVIEW.relative_to(REPOSITORY)),
                "bytes": SOURCE_REVIEW_BYTES,
                "sha256": SOURCE_REVIEW_SHA256,
                "status": "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
                "reviewed_file_count": 11,
            },
        )

    def test_synthetic_review_metadata_rejects_bad_rows_and_join_claims(self) -> None:
        renderer = _load_renderer()
        original = json.loads(SOURCE_REVIEW.read_bytes())
        mutations = []

        duplicate = copy.deepcopy(original)
        duplicate["reviewed_files"][-1]["path"] = duplicate["reviewed_files"][0]["path"]
        mutations.append(duplicate)

        missing_row = copy.deepcopy(original)
        missing_row["reviewed_files"].pop()
        mutations.append(missing_row)

        wrong_pin = copy.deepcopy(original)
        wrong_pin["reviewed_files"][0]["sha256"] = "0" * 64
        mutations.append(wrong_pin)

        unexpected_row_field = copy.deepcopy(original)
        unexpected_row_field["reviewed_files"][0]["extra"] = True
        mutations.append(unexpected_row_field)

        failed_join = copy.deepcopy(original)
        failed_join["producer_consumer_join"][
            "score_core_binding_bytes_and_sha_match_binding_v3"
        ] = False
        mutations.append(failed_join)

        for fixture in mutations:
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                renderer._validate_source_review_object(fixture)

    def test_plan_keeps_actual_export_receipts_unbound(self) -> None:
        plan = json.loads(PLAN.read_bytes())
        self.assertEqual(plan["schema"], "fg19.first-cell-score-admission-binding-plan.v6")
        self.assertEqual(
            plan["export_authority"]["expected_unit"],
            "computer-vision-homework-fg19-first-cell-export-v2.service",
        )
        for name in ("acceptance", "exact_stop"):
            with self.subTest(receipt=name):
                self.assertIsNone(plan["export_authority"][name]["path"])
                self.assertIsNone(plan["export_authority"][name]["bytes"])
                self.assertIsNone(plan["export_authority"][name]["sha256"])

    def test_body_v6_is_only_the_documented_v3_pin_adaptation_from_v5(self) -> None:
        old = _tree(BODY_V5)
        new = _normalize_v6_body(_tree(BODY_V6))
        self.assertEqual(
            ast.dump(old, include_attributes=False),
            ast.dump(new, include_attributes=False),
        )


if __name__ == "__main__":
    unittest.main()
