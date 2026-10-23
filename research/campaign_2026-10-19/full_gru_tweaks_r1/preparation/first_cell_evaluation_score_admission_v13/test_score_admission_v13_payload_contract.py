"""Check V13's fixed six-payload producer contract without running the body."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
V12 = PREPARATION / "first_cell_evaluation_score_admission_v12"
V6 = PREPARATION / "first_cell_evaluation_score_v6"
EXPORT_V4 = PREPARATION / "first_cell_evaluation_export_v4"
RECORDS = PACKET.parents[1] / "records/first_cell_evaluation_v1"
sys.path.insert(0, str(PACKET))

import derive_score_admission_v13 as derivation  # noqa: E402


def _assignment(tree: ast.Module, name: str) -> object:
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
                and node.value is not None
            )
        ):
            matches.append(node.value)
    if len(matches) != 1:
        raise AssertionError(f"expected one {name} literal")
    return ast.literal_eval(matches[0])


class ScoreAdmissionV13PayloadTests(unittest.TestCase):
    """Bind the new worker body to the four V6 scorer files and fixed receipts."""

    def test_body_equals_source_only_derivation(self) -> None:
        body, proof = derivation.derive_body()
        stored_body = (PACKET / "REMOTE_SUBMISSION_V13.py").read_bytes()
        stored_proof = json.loads((PACKET / "V13_PAYLOAD_AND_INVERSE_PROOF.json").read_bytes())
        self.assertEqual(body, stored_body)
        self.assertEqual(proof, stored_proof)

    def test_exact_six_payload_joins_and_new_identity(self) -> None:
        body = (PACKET / "REMOTE_SUBMISSION_V13.py").read_bytes()
        v12 = (V12 / "REMOTE_SUBMISSION_V12.py").read_bytes()
        tree = ast.parse(body)
        payloads = _assignment(tree, "PAYLOAD_B64")
        scorer_pins = _assignment(tree, "SCORER_FILES")
        expected_names = set(scorer_pins) | {"EXPORT_ACCEPTANCE.json", "EXPORT_EXACT_STOP.json"}
        self.assertEqual(set(payloads), expected_names)
        for name, (source_bytes, size, digest) in derivation.read_sources().items():
            decoded = base64.b64decode(payloads[name], validate=True)
            self.assertEqual(decoded, source_bytes)
            self.assertEqual(scorer_pins[name], (size, digest))
        for name in (
            "first_cell_score_contract.py",
            "first_cell_score_predictions.py",
            "score_first_cell.py",
        ):
            self.assertEqual(payloads[name], _assignment(ast.parse(v12), "PAYLOAD_B64")[name])
        self.assertEqual(
            base64.b64decode(payloads["EXPORT_ACCEPTANCE.json"], validate=True),
            (RECORDS / "export_v5/EXPORT_ACCEPTANCE.json").read_bytes(),
        )
        self.assertEqual(
            base64.b64decode(payloads["EXPORT_EXACT_STOP.json"], validate=True),
            (RECORDS / "export_v5/exact_stop/EXACT_STOP.json").read_bytes(),
        )
        self.assertIn(b'UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"', body)
        self.assertIn(b'SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v2"', body)
        self.assertNotIn(b"score-v1.service", body)
        self.assertNotIn(b"score_v1", body)

    def test_the_only_code_payload_delta_is_the_v6_core(self) -> None:
        v12 = (V12 / "REMOTE_SUBMISSION_V12.py").read_bytes()
        v13 = (PACKET / "REMOTE_SUBMISSION_V13.py").read_bytes()
        body, _ = derivation.derive_body()
        self.assertEqual(v13, body)
        self.assertNotEqual(hashlib.sha256(v12).digest(), hashlib.sha256(v13).digest())
        self.assertEqual(
            _assignment(ast.parse(v13), "SCORER_FILES")["first_cell_score_core.py"],
            (26965, "961f566eaeec0cd7e9ab48788666d03da9b0482810a11f16a019b083cfc65005"),
        )

    def test_binding_keeps_export_and_current_scorer_reviews_separate(self) -> None:
        body_path = PACKET / "REMOTE_SUBMISSION_V13.py"
        body = body_path.read_bytes()
        binding = json.loads((PACKET / "SCORE_ADMISSION_SOURCE_BINDING_V13.json").read_bytes())
        v6_review_path = V6 / "SCORE_SOURCE_REVIEW_V6.json"
        v6_review = json.loads(v6_review_path.read_bytes())
        export_review_path = EXPORT_V4 / "EXPORT_SOURCE_REVIEW_V4.json"
        export_review = json.loads(export_review_path.read_bytes())

        self.assertEqual(
            binding["score_unit"], "computer-vision-homework-fg19-first-cell-score-v2.service"
        )
        self.assertEqual(
            binding["score_evidence_root"],
            "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_score_v2",
        )
        self.assertEqual(
            binding["rendered_source"],
            {
                "bytes": len(body),
                "path": "preparation/first_cell_evaluation_score_admission_v13/REMOTE_SUBMISSION_V13.py",
                "sha256": hashlib.sha256(body).hexdigest(),
            },
        )
        self.assertEqual(
            binding["current_scorer_source_review"],
            {
                "bytes": v6_review_path.stat().st_size,
                "path": "preparation/first_cell_evaluation_score_v6/SCORE_SOURCE_REVIEW_V6.json",
                "sha256": hashlib.sha256(v6_review_path.read_bytes()).hexdigest(),
                "schema": v6_review["schema"],
                "status": v6_review["status"],
            },
        )
        self.assertEqual(
            binding["current_scorer_source_review"]["status"],
            "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR_V6",
        )
        self.assertEqual(
            binding["source_review"],
            {
                "bytes": export_review_path.stat().st_size,
                "path": str(export_review_path.relative_to(PACKET.parents[4])),
                "reviewed_file_count": len(export_review["reviewed_files"]),
                "sha256": hashlib.sha256(export_review_path.read_bytes()).hexdigest(),
                "status": export_review["status"],
            },
        )


if __name__ == "__main__":
    unittest.main()
