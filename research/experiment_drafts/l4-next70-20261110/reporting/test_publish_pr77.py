from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import publish_pr77 as reporter


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _admitted_root(root: Path, *, complete: bool = False) -> Path:
    profiles: dict[str, dict[str, object]] = {}
    configs: dict[str, dict[str, object]] = {}
    for cell in reporter.CELLS:
        gpu, uuid = reporter.GPU_BINDINGS[cell]
        target = 0 if "t0_specialist" in cell else 1 if "t1_specialist" in cell else None
        seed = 203201 if cell.endswith("203201") else 203202
        config = {
            "epochs": 6,
            "objective_mode": "standalone_labels_wmse_v2",
            "required_device": "L4",
            "required_world_size": 1,
            "checkpoint_endpoint_policy": "configured_final_epoch",
            "checkpoint_retention": "all_epochs",
            "normalization_scope": "all_fit",
            "normalization_sequences": 10_325,
            "training_mask": "all_after_warmup",
            "supervised_target": target,
            "model": {"architecture": "gru", "hidden": 256, "layers": 2},
            "seed": seed,
        }
        configs[cell] = config
        profiles[cell] = {
            "config_sha256": reporter.canonical_sha256(config),
            "raw_config_sha256": "a" * 64,
            "gpu": gpu,
            "gpu_uuid": uuid,
            "receipt_sha256": "b" * 64,
        }
    admission = {
        "schema": "wnn.l4-next70-root-fit-admission.v1",
        "approved": True,
        "source_root": reporter.SOURCE_ROOT,
        "source_sha256": reporter.SOURCE_SHA256,
        "split_sha256": reporter.SPLIT_SHA256,
        "max_fresh_fits": 6,
        "profiles": profiles,
    }
    _json(root / "ROOT-fit-admission-r1.json", admission)
    fit = [["train", index] for index in range(10_325)]
    valid = [["valid", index] for index in range(reporter.VALID_SEQUENCES)]
    valid_sha = hashlib.sha256(json.dumps(valid, separators=(",", ":")).encode()).hexdigest()
    for index, cell in enumerate(reporter.CELLS):
        gpu, uuid = reporter.GPU_BINDINGS[cell]
        directory = root / "live-dashboard" / "native" / f"l4-{cell}-fit"
        data = {
            "split_sha256": reporter.SPLIT_SHA256,
            "fit_members": fit,
            "evaluation_members": valid,
            "teacher": None,
            "normalization": {"scope": "all_fit", "sequence_count": 10_325},
            "training_mask": "all_after_warmup",
            "objective_contract": {
                "mode": "standalone_labels_wmse_v2",
                "reference_training_ancestry": "none",
            },
            "report_only_monitor": {
                "split_sha256": reporter.SPLIT_SHA256,
                "evaluation_member_count": reporter.VALID_SEQUENCES,
                "evaluation_member_sha256": valid_sha,
                "endpoint": "raw standalone z; no reference or blend",
            },
        }
        config = configs[cell]
        data_sha = reporter.canonical_sha256(data)
        metadata = {
            "config": config,
            "source": {"sha256": reporter.SOURCE_SHA256, "files": {}},
            "provenance": {
                "data": data,
                "data_sha256": data_sha,
                "world_size": 1,
                "runtime": {"devices": [{"uuid": uuid, "name": "NVIDIA L4"}]},
            },
        }
        remote = f"/data/lc-epanov_s/tabdata-next70-20261110-r1/runs/{cell}/fit"
        _json(directory / ".sequence_mirror_owner.json", {"host": "l4", "source_directory": remote})
        _json(directory / "collection_receipt.json", {"host": "l4", "source_directory": remote, "files": {}})
        _json(directory / "run_metadata.json", metadata)
        _json(directory / "monitor_progress.json", {"phase": "training", "step": 38_760, "epoch": 6.0})
        rank = {"step": 38_760, "total_loss": 0.5, "learning_rate": 1e-5, "gradient_norm_preclip": 0.5, "gradient_clipped": 0}
        (directory / "rank0.jsonl").write_text(json.dumps(rank) + "\n", encoding="utf-8")
        events: list[dict[str, object]] = []
        if complete:
            for epoch in range(1, 7):
                events.append(
                    {
                        "schema": "wnn.monitor.v1",
                        "kind": "evaluation",
                        "epoch": epoch,
                        "step": epoch * reporter.UPDATES_PER_EPOCH,
                        "wall_time": "2026-11-10T16:00:00Z",
                        "metrics": {
                            "valid/sequence_count": reporter.VALID_SEQUENCES,
                            "valid/selected_rows": reporter.VALID_ROWS,
                            "valid/wp/t0": 0.6 + index / 100 + epoch / 1000,
                            "valid/wp/t1": 0.7 + index / 100 + epoch / 1000,
                        },
                        "context": {
                            "split_sha256": reporter.SPLIT_SHA256,
                            "evaluation_member_sha256": valid_sha,
                            "evaluation_member_count": reporter.VALID_SEQUENCES,
                            "endpoint": "raw standalone z; no reference or blend",
                        },
                    }
                )
        (directory / "monitor_evaluations.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8"
        )
        if complete:
            terminal_identity = {
                "source_sha256": reporter.SOURCE_SHA256,
                "config_sha256": profiles[cell]["config_sha256"],
                "data_sha256": data_sha,
                "world_size": 1,
            }
            _json(
                directory / "terminal_status.json",
                {
                    "status": "completed",
                    "completed_epochs": 6,
                    "global_updates": reporter.EXPECTED_UPDATES,
                    "model_sha256": f"{index + 1:064x}",
                    "identity": terminal_identity,
                },
            )
    return root


def test_fixed_e6_grades_require_all_six_complete_valid_events(tmp_path: Path) -> None:
    root = _admitted_root(tmp_path, complete=True)
    snapshot = reporter.inspect_snapshot(root)
    rows = reporter.grade_rows(snapshot["cells"])
    assert rows is not None
    assert len(rows) == 8  # baseline plus the three declared pairings for each seed
    assert rows[0]["seed"] == 203201
    assert rows[0]["recipe"] == "baseline shared control"
    assert rows[0]["wp"] == pytest.approx((0.606 + 0.706) / 2)

    one_eval = root / "live-dashboard" / "native" / f"l4-{reporter.CELLS[0]}-fit" / "monitor_evaluations.jsonl"
    events = [json.loads(line) for line in one_eval.read_text(encoding="utf-8").splitlines()]
    events[-1]["metrics"]["valid/selected_rows"] -= 1
    one_eval.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    broken = reporter.inspect_snapshot(root)
    assert reporter.grade_rows(broken["cells"]) is None


def test_wrong_source_or_gpu_admission_is_rejected(tmp_path: Path) -> None:
    root = _admitted_root(tmp_path)
    path = root / "ROOT-fit-admission-r1.json"
    admission = json.loads(path.read_text(encoding="utf-8"))
    admission["source_sha256"] = "f" * 64
    _json(path, admission)
    with pytest.raises(ValueError, match="ROOT admission"):
        reporter.inspect_snapshot(root)


def test_pending_marker_is_reconciled_without_duplicate_post(tmp_path: Path) -> None:
    class LostResponseClient:
        def __init__(self) -> None:
            self.post_calls = 0
            self.found: dict[str, object] | None = None

        def find_report(self, number: int, marker: str, prepared_at: float) -> object:
            assert number == 77
            return self.found

        def post_report(self, number: int, body: str, *, before_post: object) -> object:
            self.post_calls += 1
            before_post()  # type: ignore[operator]
            self.found = {"id": 123, "body": body}
            raise RuntimeError("simulated lost HTTP response")

    client = LostResponseClient()
    path = tmp_path / "state.json"
    marker = "<!-- reporter-dedup-marker -->"
    with pytest.raises(RuntimeError, match="lost HTTP response"):
        reporter.publish_once(client, path, marker, f"report\n{marker}", 1000.0)
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["reports"][marker]["status"] == "pending"
    assert reporter.publish_once(client, path, marker, "changed body ignored", 9999.0) == "reconciled-existing"
    assert client.post_calls == 1


def test_live_report_has_explicit_unknown_endpoint_without_full_e6(tmp_path: Path) -> None:
    root = _admitted_root(tmp_path, complete=False)
    snapshot = reporter.inspect_snapshot(root)
    report = reporter.render_report(snapshot, "hourly progress", None)
    assert "latest full-valid WP t0/t1 unknown / unknown" in report
    assert "pending fixed-E6" in report
