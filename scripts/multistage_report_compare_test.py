from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.multistage_report_compare import (
    AUXILIARY_SPLIT_SHA256,
    INNER_MASK,
    PRIMARY_SCHEMA,
    PRIMARY_SPLIT_SHA256,
    TL3_REFERENCE_WP,
    WP_REDUCTION,
    build_comparisons,
)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _primary_report(checkpoint: str = "a") -> dict[str, object]:
    config = {"model": {"architecture": "gru", "hidden": 256}}
    identity = {"config_sha256": _sha(config), "source_sha256": "d" * 64}
    return {
        "schema": PRIMARY_SCHEMA,
        "report_type": "primary_development_exposed_report_only",
        "objective_eligible": False,
        "checkpoint_selection_allowed": False,
        "model": {
            "checkpoint_sha256": checkpoint * 64,
            "config_sha256": _sha(config),
            "run_identity_sha256": _sha(identity),
            "config": config,
            "producer": {"provenance": {"identity": identity}},
        },
        "fit": {
            "split_sha256": AUXILIARY_SPLIT_SHA256,
            "sequence_count": 8_260,
            "control_sequence_count": 2_065,
        },
        "evaluation": {
            "split_sha256": PRIMARY_SPLIT_SHA256,
            "role": "primary_valid",
            "source": "valid",
            "mask": "is_scored AND row_index>=99",
            "sequence_count": 1_873,
            "selected_rows": 3_528_273,
            "metrics": {
                "wp": 0.70,
                "t0": 0.71,
                "t1": 0.69,
                "wmse": 0.8,
                "wmse_t0": 0.7,
                "wmse_t1": 0.9,
                "selected_rows": 3_528_273,
            },
        },
        "comparison_reference": {
            "name": "canonical fresh TL3",
            "wp": TL3_REFERENCE_WP,
            "measured_in_this_report": False,
            "exposure": "development-exposed full official valid; no hidden/public forecast",
        },
    }


def _inner_objective(checkpoint: str = "b") -> dict[str, object]:
    config = {"model": {"architecture": "lstm", "hidden": 256}, "kd_weight": 0.0}
    identity = {"config_sha256": _sha(config), "data_sha256": "e" * 64}
    return {
        "schema": "wnn.multistage-inner-objective.v1",
        "status": "complete",
        "split_sha256": AUXILIARY_SPLIT_SHA256,
        "primary_split_sha256": PRIMARY_SPLIT_SHA256,
        "prediction_manifest_sha256": "f" * 64,
        "model_sha256": checkpoint * 64,
        "config_sha256": _sha(config),
        "fit_members_sha256": "c" * 64,
        "evaluation_members_sha256": "d" * 64,
        "sequence_count": 2_065,
        "selected_rows": 500_000,
        "objective": {
            "name": "wp",
            "value": 0.65,
            "t0": 0.66,
            "t1": 0.64,
            "reduction": WP_REDUCTION,
        },
        "evaluation": {
            "role": "control",
            "scope": "inner_train_proxy",
            "source": "train",
            "row_mask": INNER_MASK,
            "is_scored_used": False,
        },
        "producer": {
            "checkpoint_sha256": checkpoint * 64,
            "split_sha256": AUXILIARY_SPLIT_SHA256,
            "fit_members_sha256": "c" * 64,
            "provenance": {"identity": identity, "identity_sha256": _sha(identity)},
        },
        "prediction_producer": {"config": config},
    }


class ReportComparisonTests(unittest.TestCase):
    """Check report scoping, exact identities, and separate output artifacts."""

    def test_primary_and_inner_metrics_write_separate_tables_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary" / "REPORT.json"
            inner = root / "inner" / "inner_objective.json"
            primary.parent.mkdir()
            inner.parent.mkdir()
            primary.write_text(json.dumps(_primary_report()), encoding="utf-8")
            inner.write_text(json.dumps(_inner_objective()), encoding="utf-8")
            output = root / "plots"
            result = build_comparisons([primary], [inner], output)
            self.assertIn("primary", result)
            self.assertIn("inner_proxy", result)
            self.assertTrue((output / "primary_comparison.csv").is_file())
            self.assertTrue((output / "primary_comparison.png").is_file())
            self.assertTrue((output / "primary_comparison.svg").is_file())
            self.assertTrue((output / "inner_proxy_comparison.csv").is_file())
            self.assertTrue((output / "inner_proxy_comparison.png").is_file())
            self.assertTrue((output / "inner_proxy_comparison.svg").is_file())
            primary_json = json.loads((output / "primary_comparison.json").read_text())
            inner_json = json.loads((output / "inner_proxy_comparison.json").read_text())
            self.assertEqual(primary_json["models"][0]["wp"], 0.70)
            self.assertEqual(inner_json["models"][0]["wp"], 0.65)
            self.assertIsNone(inner_json["comparison_reference"])
            self.assertFalse(primary_json["objective_eligible"])

    def test_primary_hpo_objective_or_wrong_metric_mask_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = _primary_report()
            report["objective_eligible"] = True
            path = root / "REPORT.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a report-only primary report"):
                build_comparisons([path], [], root / "out")


if __name__ == "__main__":
    unittest.main()
