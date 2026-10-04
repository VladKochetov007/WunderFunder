from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_inputs.source_bundle.worker import (
    EXPECTED_SOURCE_HASHES,
    checker_passes,
    validate_source_hashes,
)  # noqa: E402


class WorkerContractTests(unittest.TestCase):
    def valid_checker_result(self) -> dict[str, object]:
        return {
            "status": "passed",
            "updates": 7_460,
            "source_sha256": {
                "rnn.py": EXPECTED_SOURCE_HASHES["src/wnn_jax/rnn.py"],
                "onecycle.py": EXPECTED_SOURCE_HASHES["src/wnn_jax/onecycle.py"],
            },
            "model_or_data_loaded": False,
        }

    def test_exact_success_report_is_accepted(self) -> None:
        self.assertTrue(checker_passes(0, self.valid_checker_result()))

    def test_failed_process_or_failed_status_is_rejected(self) -> None:
        report = self.valid_checker_result()
        report["status"] = "failed"
        self.assertFalse(checker_passes(0, report))
        self.assertFalse(checker_passes(1, self.valid_checker_result()))

    def test_missing_report_is_rejected(self) -> None:
        self.assertFalse(checker_passes(0, None))

    def test_mutated_checker_source_identity_is_rejected(self) -> None:
        report = self.valid_checker_result()
        report["source_sha256"] = {
            "rnn.py": EXPECTED_SOURCE_HASHES["src/wnn_jax/rnn.py"],
            "onecycle.py": "0" * 64,
        }
        self.assertFalse(checker_passes(0, report))

    def test_mutated_packet_source_identity_is_rejected(self) -> None:
        observed = dict(EXPECTED_SOURCE_HASHES)
        self.assertTrue(validate_source_hashes(observed))
        observed["src/wnn_jax/rnn.py"] = "0" * 64
        self.assertFalse(validate_source_hashes(observed))


if __name__ == "__main__":
    unittest.main()
