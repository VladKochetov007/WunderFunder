"""CPU-only boundary and batch-state tests for the E3 report-only exporter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import e3_report_only_r2 as exporter


def _config(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": cell["seed"],
        "epochs": 4,
        "model": {"architecture": "transformer"},
        "impact81": {"cell": cell},
    }


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cell = {
        "schema": "impact81.cell.v1",
        "id": "ip01_control_s204101",
        "card": "01",
        "arm": "control",
        "variant": "plain",
        "seed": 204101,
        "native_config": "configs/native_s204101.json",
        "native_sha256": "1" * 64,
    }
    config = _config(cell)
    source_files = {"runtime/src/model.py": "2" * 64}
    source_sha = exporter._canonical_sha256(source_files)
    fit_members = [["train", index] for index in range(exporter.FIT_SEQUENCES)]
    data = {"split_sha256": exporter.SPLIT_SHA256, "fit_members": fit_members}
    data_sha = exporter._canonical_sha256(data)
    config_sha = exporter._canonical_sha256(config)
    identity = {
        "source_sha256": source_sha,
        "source_files": source_files,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "data": data,
    }
    metadata = {
        "config": config,
        "source": {"files": source_files, "sha256": source_sha},
        "provenance": {"data": data, "data_sha256": data_sha},
    }
    entries = [
        {
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "global_step": (epoch + 1) * 10,
            "checkpoint_sha256": str(epoch + 3) * 64,
            "checkpoint_path": f"epoch_checkpoints/epoch-{epoch + 1:06d}-{str(epoch + 3) * 64}.pt",
        }
        for epoch in range(3)
    ]
    index = {
        "schema": "wnn.sequence-epoch-index.v1",
        "completed_epochs": 3,
        "global_step": 30,
        "identity": identity,
        "entries": entries,
    }
    index["index_sha256"] = exporter._canonical_sha256(index)
    return cell, metadata, index


class E3ReportOnlyTests(unittest.TestCase):
    def test_predeclared_e3_index_binds_source_config_data_and_cell(self) -> None:
        cell, metadata, index = _fixture()
        binding = exporter.validate_e3_index_binding(index, metadata, cell)
        self.assertEqual(binding["entry"]["completed_epochs"], 3)
        self.assertEqual(binding["entry"]["global_step"], 30)
        self.assertEqual(binding["index_sha256"], index["index_sha256"])

    def test_rejects_wrong_endpoint_config_and_tampered_index(self) -> None:
        cell, metadata, index = _fixture()
        changed = json.loads(json.dumps(metadata))
        changed["config"]["epochs"] = 5
        with self.assertRaises(exporter.ReadoutError):
            exporter.validate_e3_index_binding(index, changed, cell)

        tampered = json.loads(json.dumps(index))
        tampered["entries"][2]["global_step"] += 1
        with self.assertRaises(exporter.ReadoutError):
            exporter.validate_e3_index_binding(tampered, metadata, cell)

    def test_monitor_score_keys_match_actual_saved_e3_event(self) -> None:
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "actual_ip01_control_s204101_e3_monitor.json"
        )
        event = json.loads(fixture.read_text(encoding="utf-8"))
        values = exporter._monitor_score_values(event)
        self.assertEqual(
            values,
            {
                "wp": 0.6825677148042587,
                "t0": 0.6778438414834481,
                "t1": 0.6872915881250693,
            },
        )


if __name__ == "__main__":
    unittest.main()
