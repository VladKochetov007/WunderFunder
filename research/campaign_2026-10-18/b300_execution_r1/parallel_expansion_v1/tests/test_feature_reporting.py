"""CPU-only tests for matched feature collection and idempotent publication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PACKET = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKET / "code"), str(PACKET.parent / "github_reporting_v1/code")]

import feature_report_core as core  # noqa: E402
import feature_reporting as reporter  # noqa: E402
import panel as panel_module  # noqa: E402
from panel import build_panel  # noqa: E402
from render_report import canonical_sha, marker  # noqa: E402


@pytest.mark.parametrize("family", ["age", "innovation"])
def test_panel_has_thirty_same_fold_seed_feature_control_pairs(family: str) -> None:
    pairs = core.expected_pairs(build_panel(), family)
    assert len(pairs) == 30
    for (fold, seed), arms in pairs.items():
        assert set(arms) == {
            core.FAMILIES[family]["control_arm"],
            core.FAMILIES[family]["feature_arm"],
        }
        assert arms[core.FAMILIES[family]["control_arm"]]["fold"] == fold
        assert arms[core.FAMILIES[family]["feature_arm"]]["seed"] == seed


def test_equal_fold_summary_does_not_pool_uneven_fold_counts() -> None:
    rows = [
        {"fold": 0, "control_wp": 0.2, "feature_wp": 0.3, "delta_wp": 0.1},
        {"fold": 0, "control_wp": 0.3, "feature_wp": 0.5, "delta_wp": 0.2},
        {"fold": 1, "control_wp": 0.4, "feature_wp": 0.8, "delta_wp": 0.4},
    ]
    summary = core.equal_fold_summary(rows)
    assert summary["delta_wp_equal_fold_mean"] == pytest.approx(0.275)
    assert summary["scored_pairs"] == 3
    assert summary["folds_observed"] == [0, 1]


def test_status_and_result_comments_are_partial_and_pair_marked() -> None:
    row = {
        "pair_id": "age_f0_s1101",
        "receipt_sha256": "a" * 64,
        "fold": 0,
        "seed": 1101,
        "control_wp": 0.51,
        "feature_wp": 0.52,
        "delta_wp": 0.01,
        "control": {"checkpoint_sha256": "b" * 64},
        "feature": {"checkpoint_sha256": "c" * 64},
        "grade_sha256": "d" * 64,
    }
    snapshot = {
        "experiment_id": core.FAMILIES["age"]["experiment_id"],
        "completed": [row],
        "active": 1,
        "pending": 28,
        "terminal": False,
        "source": {"commit": "a266"},
        "contract_note": "New successor.",
        "errors": [],
    }
    body = core.report_body(snapshot, "https://example.test/evidence", rows=[row])
    assert marker(snapshot["experiment_id"], "pair", canonical_sha({
        "pair_id": row["pair_id"], "receipt_sha256": row["receipt_sha256"]
    })) in body
    assert "partial / in progress" in body
    assert "1/30" in body
    assert "do not establish significance" in body


def test_reconciliation_acknowledges_only_own_exact_pair_marker() -> None:
    experiment = core.FAMILIES["age"]["experiment_id"]
    row = {"pair_id": "age_f0_s1101", "receipt_sha256": "a" * 64}
    pair_marker = marker(experiment, "pair", canonical_sha(row))
    status_marker = marker(experiment, "status")

    class FakeClient:
        def __init__(self, comments: list[dict]):
            self._comments = comments
            self.posted = 0

        def request(self, _method: str, _path: str) -> dict:
            return {
                "state": "open",
                "base": {"ref": "research-team", "repo": {"full_name": reporter.REPOSITORY}},
                "head": {"ref": "age-successor", "repo": {"full_name": reporter.REPOSITORY}},
            }

        def comments(self, _number: int) -> list[dict]:
            return self._comments

        def post(self, _number: int, _body: str) -> dict:
            self.posted += 1
            raise AssertionError("already-reconciled comments must not be duplicated")

        def patch(self, _comment_id: int, _body: str) -> dict:
            raise AssertionError("unchanged status must not be edited on completion")

    snapshot = {
        "experiment_id": experiment,
        "completed": [],
        "active": 0,
        "pending": 29,
        "terminal": False,
        "source": {},
        "contract_note": "New successor.",
        "errors": [],
    }
    client = FakeClient(
        [
            {"id": 1, "html_url": "https://github.test/pair", "body": pair_marker, "user": {"login": "owner"}},
            {"id": 2, "html_url": "https://github.test/status", "body": status_marker, "user": {"login": "owner"}},
        ]
    )
    current_snapshot = {**snapshot, "completed": [row]}
    state = {"snapshot_sha256": canonical_sha(current_snapshot)}
    helpers = (object, RuntimeError, object, lambda *_: pytest.fail("unexpected archive"))
    utility = (marker, object, object, object)
    reporter._publish_snapshot(
        Path("."),
        client,
        helpers,
        utility,
        current_snapshot,
        {"number": 26, "head_ref": "age-successor", "evidence_url": "https://example.test"},
        state,
        "owner",
        "completion",
    )
    assert state["pairs"][row["pair_id"]] == row["receipt_sha256"]
    assert client.posted == 0


def test_target_mapping_rejects_original_source_pr_claims() -> None:
    config = {
        "root": str(reporter.ROOT),
        "repository": reporter.REPOSITORY,
        "targets": {
            "age": {
                "experiment_id": core.FAMILIES["age"]["experiment_id"],
                "number": 26,
                "head_ref": "age-successor",
                "evidence_url": "https://example.test/age",
            },
            "innovation": {
                "experiment_id": core.FAMILIES["innovation"]["experiment_id"],
                "number": 27,
                "head_ref": "innovation-successor",
                "evidence_url": "https://example.test/innovation",
            },
        },
    }
    reporter.validate_targets(config["targets"])
    for family, target in config["targets"].items():
        assert target["experiment_id"] == reporter.EXPERIMENT_IDS[family]
    config["targets"]["age"]["number"] = 4
    with pytest.raises(ValueError, match="incomplete or drifted"):
        reporter.validate_targets(config["targets"])


def test_export_phase_must_bind_the_actual_uuid_cpu_slot_and_log(tmp_path: Path) -> None:
    class Common:
        @staticmethod
        def sha(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        @staticmethod
        def read(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8"))

    root = tmp_path
    package = root / "runtime/pr15-host-v2"
    package.mkdir(parents=True)
    work = root / "state/work/job"
    (work / "envelope/export").mkdir(parents=True)
    log = work / "export.log"
    log.write_text("actor returned", encoding="utf-8")
    receipt_path = work / "envelope/export/export_cell_resource_receipt.json"
    resource = {
        "schema": "tabdata-b300-process-envelope.v1",
        "status": "actor_returned",
        "script": "export_cell.py",
        "visible_devices": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "cpu_affinity": [80, 82],
        "allocator_budget_bytes": 6 * 1024**3,
        "torch_threads": 2,
        "gpu": "NVIDIA B300",
        "cuda_peak_allocated_bytes": 123,
        "cuda_peak_reserved_bytes": 456,
    }
    receipt_path.write_text(json.dumps(resource), encoding="utf-8")
    phase = {
        "command": [
            sys.executable,
            "-u",
            str(package / "gpu_entrypoint.py"),
            str(package / "export_cell.py"),
            "--request",
            str(work / "worker_request.json"),
        ],
        "exit_code": 0,
        "elapsed_seconds": 1.25,
        "log": str(log),
        "log_sha256": Common.sha(log),
        "resource_receipt": str(receipt_path),
        "resource_receipt_sha256": Common.sha(receipt_path),
    }
    job = {"gpu_uuid": resource["visible_devices"], "cpus": resource["cpu_affinity"]}
    core._verify_export_phase(root, work, job, phase, Common)
    resource["visible_devices"] = "GPU-deadbeef"
    receipt_path.write_text(json.dumps(resource), encoding="utf-8")
    phase["resource_receipt_sha256"] = Common.sha(receipt_path)
    with pytest.raises(ValueError, match="resource envelope"):
        core._verify_export_phase(root, work, job, phase, Common)


@pytest.mark.parametrize("family", ["age", "innovation"])
def test_controller_claim_binds_panel_identity_and_scheduled_cpu_slot(
    tmp_path: Path, family: str
) -> None:
    expected = next(iter(core.expected_pairs(build_panel(), family).values()))[
        core.FAMILIES[family]["control_arm"]
    ]
    slot_id = 0
    cpu_ids = list(panel_module.cpu_pair(expected["gpu_index"], slot_id))
    claim = {**expected, "slot_id": slot_id, "cpu_ids": cpu_ids, "cpus": cpu_ids}

    def reader(value: dict) -> type:
        class Common:
            @staticmethod
            def read(_path: Path, current: dict = value) -> dict:
                return current

        return Common

    checked = core._claimed_job(tmp_path, expected, reader(claim), panel_module)
    assert checked == claim
    assert checked["cpus"] == cpu_ids

    for key in ("arm", "gpu_uuid", "fold", "seed"):
        altered = {**claim, key: f"changed-{key}"}
        with pytest.raises(ValueError, match="immutable panel job"):
            core._claimed_job(tmp_path, expected, reader(altered), panel_module)

    altered_slot = {**claim, "slot_id": 1}
    with pytest.raises(ValueError, match="CPU affinity"):
        core._claimed_job(tmp_path, expected, reader(altered_slot), panel_module)

    malformed_cpus = {**claim, "cpu_ids": [True, cpu_ids[1]]}
    with pytest.raises(ValueError, match="CPU affinity"):
        core._claimed_job(tmp_path, expected, reader(malformed_cpus), panel_module)
