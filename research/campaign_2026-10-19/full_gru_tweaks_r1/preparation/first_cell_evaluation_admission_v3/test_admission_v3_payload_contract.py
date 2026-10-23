"""Focused source-only checks for the rendered V3 five-payload admission."""

from __future__ import annotations

import json
import unittest

from prove_admission_v3_inverse import (
    EXPECTED_EXPORT_REVIEW,
    PAYLOAD_NAMES,
    V2_BODY,
    V3_BINDING,
    V3_BODY,
    _inverse_body,
    build_proof,
)


class AdmissionV3PayloadContractTests(unittest.TestCase):
    def test_v3_embeds_the_reviewed_exporter_payloads_and_exact_five_members(self) -> None:
        body = V3_BODY.read_bytes()
        self.assertTrue(body)
        binding = json.loads(V3_BINDING.read_bytes())
        self.assertEqual(binding["schema"], "fg19.first-cell-export-admission-source-binding.v3")
        self.assertEqual(set(binding["payloads"]), set(PAYLOAD_NAMES))
        review = binding["producer_consumer_review"]
        for key, expected in EXPECTED_EXPORT_REVIEW.items():
            self.assertEqual(review[key], expected)
        self.assertEqual(review["producer_source"], binding["payloads"]["export_first_cell_v1.py"])
        self.assertEqual(review["producer_binding"], binding["payloads"]["EXPORT_BINDING_V1.json"])
        self.assertEqual(
            review["source_manifest"], binding["payloads"]["EXPORT_SOURCE_MANIFEST.json"]
        )

    def test_only_fixed_payload_assignments_differ_from_failed_v2_body(self) -> None:
        restored = _inverse_body(V2_BODY.read_bytes(), V3_BODY.read_bytes())
        self.assertEqual(restored, V2_BODY.read_bytes())
        proof = build_proof()
        self.assertEqual(
            proof["status"], "PASS_EXACT_V2_BODY_INVERSE_OUTSIDE_FIVE_PAYLOAD_CONSTANTS"
        )
        self.assertEqual(proof["all_other_body_bytes_reconstruct_v2"], True)
        self.assertEqual(set(proof["payloads"]), set(PAYLOAD_NAMES))

    def test_failed_v2_preflight_remains_pinned_without_claiming_retry(self) -> None:
        binding = json.loads(V3_BINDING.read_bytes())
        failed = binding["supersedes_v2_failed_preflight"]
        self.assertEqual(failed["body"]["bytes"], 243_608)
        self.assertEqual(
            failed["body"]["sha256"],
            "a15b2cec55657f3f630fcd18a4fe20037fe08c3186bb0f046aac5480c6e0aeab",
        )
        self.assertEqual(failed["transport"]["bytes"], 729)
        self.assertEqual(failed["stderr"]["bytes"], 769)
        self.assertIn("no retry", failed["outcome"])


if __name__ == "__main__":
    unittest.main()
