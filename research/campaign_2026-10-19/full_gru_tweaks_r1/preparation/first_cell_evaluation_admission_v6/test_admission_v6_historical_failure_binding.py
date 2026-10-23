"""Check the second failed export's local metadata-only historical pin."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

RENDERER = Path(__file__).resolve().parent / "render_admission_remote_v6.py"


def _load_renderer():
    """Load the stdlib-only renderer without rendering or contacting a worker."""
    spec = importlib.util.spec_from_file_location("admission_v6_history_fixture", RENDERER)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load V6 renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdmissionV6HistoricalFailureTests(unittest.TestCase):
    def test_latest_failure_is_pinned_without_requery_or_quality_claim(self) -> None:
        renderer = _load_renderer()
        result = renderer.bind_latest_failed_export()
        self.assertEqual(
            result["classification"],
            "SECOND_EXPORT_INVOCATION_FAILED_AT_CONTROL_CHECKPOINT_ARGUMENT_GATE",
        )
        self.assertIs(result["historical_only"], True)
        self.assertIs(result["root_only_not_staged_to_B"], True)
        self.assertIs(result["old_unit_must_not_be_queried_stopped_or_retried"], True)
        self.assertEqual(
            result["unit"], "computer-vision-homework-fg19-first-cell-export-v2.service"
        )
        self.assertEqual(result["invocation_id"], "555122f41c884850a6bd4e847c14e82f")
        self.assertEqual(
            result["failure_message"],
            "ValueError: checkpoint recipe argument differs: input_lead",
        )
        self.assertEqual(
            set(result["files"]),
            {
                "launch_acceptance",
                "root_transport",
                "root_stdout",
                "root_stderr",
                "admission",
                "launch",
                "launch_review",
                "startup_observation",
                "startup_stderr",
                "startup_transport",
                "unit_log_tail",
                "observation",
                "failure_review",
            },
        )


if __name__ == "__main__":
    unittest.main()
