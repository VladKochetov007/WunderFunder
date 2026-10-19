"""CPU-only tests for source-qualified R01 delivery-recovery reporting."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import publish_recovery as publisher  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _github_code_dir() -> Path:
    configured = os.environ.get("TABDATA_GITHUB_REPORTING_CODE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "github_reporting_v1" / "code"


def _make_fit_export(
    root: Path,
    recovery: Path,
    job_id: str,
    arm: str,
    *,
    cpu_affinity: list[int] | None = None,
    placement_authorization: dict | None = None,
) -> dict:
    job = {
        "id": job_id,
        "arm": arm,
        "fold": 0,
        "seed": 1003,
        "gpu_uuid": f"GPU-{arm}-1234",
        "cpu_ids_initial": [0, 2],
        "run_dir": str(root / "runs" / job_id),
    }
    run = Path(job["run_dir"])
    artifact = run / "artifacts/runs" / job_id
    artifact.mkdir(parents=True)
    fixed, selected = artifact / "model_ep5.pt", artifact / "model.pt"
    fixed.write_bytes((job_id + " fixed").encode())
    selected.write_bytes((job_id + " selected").encode())
    train_resource_path = run / "train_cell_resource_receipt.json"
    _write_json(
        train_resource_path,
        {
            "visible_devices": job["gpu_uuid"],
            "cpu_affinity": cpu_affinity if cpu_affinity is not None else [0, 2],
        },
    )
    identity = {
        "epoch": 5,
        "input_width": 211,
        "fixed_path": str(fixed.resolve()),
        "selected_path": str(selected.resolve()),
        "fixed_file_sha256": publisher.sha256_file(fixed),
        "selected_file_sha256": publisher.sha256_file(selected),
    }
    view = recovery / "fit_views" / job_id
    producer_path = view / "producer_receipt.json"
    _write_json(producer_path, {"schema": "tabdata-l4-fit-producer.v1", "run_name": job_id})
    fit_path = recovery / "receipts" / f"{job_id}.json"
    fit = {
        "schema": "tabdata-r01-fit-recovery.v2",
        "status": "existing_fit_validated_no_refit",
        "job": job,
        "fit_view": str(view),
        "recovery_fit_receipt_sha256": publisher.sha256_file(producer_path),
        "checkpoint_identity": identity,
        "optimizer_budget": {"epochs": 6, "total": 2700},
        "original_actor_phase": {
            "status": "actor_returned",
            "fit_completion_verified_from": ["six epochs", "fixed checkpoint"],
            "resource_receipt_sha256": publisher.sha256_file(train_resource_path),
        },
        "old_fit_outputs_modified": False,
    }
    if placement_authorization is not None:
        fit["original_actor_phase"]["placement_authorization"] = placement_authorization
    _write_json(fit_path, fit)
    output = recovery / "exports" / job_id
    cache = output / "cache"
    cache.mkdir(parents=True)
    prediction = cache / f"pred_{job_id}_valid.npy"
    sidecar = cache / f"pred_{job_id}_valid.json"
    prediction.write_bytes((job_id + " predictions").encode())
    sidecar.write_text("{}", encoding="utf-8")
    resource_path = output / "export_cell_resource_receipt.json"
    resource = {
        "schema": "tabdata-b300-process-envelope.v1",
        "status": "actor_returned",
        "script": "export_cell.py",
        "visible_devices": f"GPU-{arm}-1234",
        "gpu": "NVIDIA B300",
        "allocator_budget_bytes": 6 * 1024**3,
        "torch_threads": 2,
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
    }
    _write_json(resource_path, resource)
    command_path = output / "export_command.json"
    _write_json(command_path, {"schema": "test export command"})
    log = recovery / "logs" / f"{job_id}.export.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("actor returned\n", encoding="utf-8")
    receipt_path = output / "export_receipt.json"
    export_receipt = {
        "status": "completed_fixed_epoch5_feature_only_fp32",
        "run_name": job_id,
        "fold": 0,
        "seed": 1003,
        "epoch": 5,
        "feature_only": True,
        "inference_precision": "fp32",
        "target_reads_during_export": False,
        "checkpoint_sha256": identity["fixed_file_sha256"],
        "prediction_sha256": publisher.sha256_file(prediction),
        "sidecar_sha256": publisher.sha256_file(sidecar),
    }
    _write_json(receipt_path, export_receipt)
    export_recovery = {
        "schema": "tabdata-r01-export-recovery.v2",
        "status": "exported_fixed_epoch5_no_refit",
        "job_id": job_id,
        "fit_recovery_receipt_sha256": publisher.sha256_file(fit_path),
        "export_receipt_path": str(receipt_path),
        "export_receipt_sha256": publisher.sha256_file(receipt_path),
        "export_command_sha256": publisher.sha256_file(command_path),
        "export_resource_sha256": publisher.sha256_file(resource_path),
        "actor_log_sha256": publisher.sha256_file(log),
        "visible_gpu_uuid": resource["visible_devices"],
        "checkpoint_sha256": identity["fixed_file_sha256"],
        "prediction_sha256": export_receipt["prediction_sha256"],
        "sidecar_sha256": export_receipt["sidecar_sha256"],
        "old_fit_outputs_modified": False,
    }
    export_path = recovery / "receipts" / f"{job_id}.export.json"
    _write_json(export_path, export_recovery)
    for name in (
        "command.json",
        "adapter_preflight.json",
        "producer_receipt.json",
    ):
        _write_json(run / name, {"artifact": name})
    _write_json(artifact / "history.json", {"epochs": 6})
    _write_json(artifact / "provenance.json", {"source": "pinned"})
    return {
        "job": job,
        "identity": identity,
        "export_receipt": export_receipt,
        "fit_path": fit_path,
    }


def _initial_placement(_root: Path, job: dict, resource: dict) -> dict:
    if (
        resource.get("visible_devices") != job["gpu_uuid"]
        or resource.get("cpu_affinity") != job["cpu_ids_initial"]
    ):
        raise ValueError("resource placement differs from frozen initial panel")
    return {"authority": "frozen_panel_initial_placement"}


def test_recovery_grade_binds_original_metric_checkpoint_prediction_and_sidecar() -> None:
    wide_job = {"id": "wide_f0_s1003"}
    deep_job = {"id": "deep_f0_s1003"}
    wide_binding = {
        "checkpoint_sha256": "a" * 64,
        "prediction_sha256": "b" * 64,
        "sidecar_sha256": "c" * 64,
    }
    deep_binding = {
        "checkpoint_sha256": "d" * 64,
        "prediction_sha256": "e" * 64,
        "sidecar_sha256": "f" * 64,
    }
    grade = {
        "schema": "priority25.endpoint_grade.v1",
        "metric_id": "global_wp_clip2_v1",
        "fold": 0,
        "split_manifest_sha256": "1" * 64,
        "evaluation_row_groups": [5, 9],
        "endpoint": {
            "run": deep_job["id"],
            "epoch": 5,
            **deep_binding,
            "score": {"wp": 0.53},
        },
        "reference": {
            "run": wide_job["id"],
            "epoch": 5,
            **wide_binding,
            "score": {"wp": 0.51},
        },
        "paired_component_bootstrap": {"resamples": 2000, "seed": 271828},
    }
    scores = publisher._verify_grade_bindings(
        grade, 0, "1" * 64, [5, 9], wide_job, deep_job, wide_binding, deep_binding
    )
    assert scores == (0.53, 0.51)

    mutations = (
        {**grade, "metric_id": "wrong_metric"},
        {**grade, "fold": 1},
        {**grade, "endpoint": {**grade["endpoint"], "checkpoint_sha256": "0" * 64}},
        {**grade, "reference": {**grade["reference"], "prediction_sha256": "0" * 64}},
        {**grade, "endpoint": {**grade["endpoint"], "sidecar_sha256": "0" * 64}},
    )
    for mutated in mutations:
        with pytest.raises(ValueError):
            publisher._verify_grade_bindings(
                mutated, 0, "1" * 64, [5, 9], wide_job, deep_job, wide_binding, deep_binding
            )


def test_manifest_verification_hashes_every_member_before_use(tmp_path: Path) -> None:
    member = tmp_path / "helper.py"
    member.write_text("PINNED = True\n", encoding="utf-8")
    manifest = {
        "schema": "tabdata-r01-delivery-recovery-manifest.v2.3",
        "files": {member.name: hashlib.sha256(member.read_bytes()).hexdigest()},
    }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (
        publisher._verify_manifest(
            manifest_path, manifest_hash, "tabdata-r01-delivery-recovery-manifest.v2.3"
        )
        == manifest
    )
    member.write_text("PINNED = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="member failed"):
        publisher._verify_manifest(
            manifest_path, manifest_hash, "tabdata-r01-delivery-recovery-manifest.v2.3"
        )


def test_original_panel_index_requires_thirty_complete_pairs() -> None:
    jobs = []
    for fold, count in enumerate((8, 8, 7, 7)):
        for seed in range(1003, 1003 + count):
            for arm in ("wide", "deep"):
                jobs.append(
                    {"id": f"{arm}_f{fold}_s{seed}", "arm": arm, "fold": fold, "seed": seed}
                )
    pairs = publisher._job_map({"jobs": jobs})
    assert len(pairs) == 30
    assert all(set(arms) == {"wide", "deep"} for arms in pairs.values())
    with pytest.raises(ValueError, match="60 original fit jobs"):
        publisher._job_map({"jobs": jobs[:-1]})


def test_fit_and_export_recovery_chain_binds_original_checkpoint_bytes(tmp_path: Path) -> None:
    root = tmp_path
    recovery = root / "state/r01-delivery-recovery-v2"
    expected = _make_fit_export(root, recovery, "wide_f0_s1003", "wide")
    verified = publisher._fit_export(root, recovery, expected["job"], _initial_placement)
    assert verified["identity"]["fixed_file_sha256"] == expected["identity"]["fixed_file_sha256"]
    assert (
        verified["export_receipt"]["prediction_sha256"]
        == expected["export_receipt"]["prediction_sha256"]
    )

    changed_prediction = verified["prediction_path"]
    changed_prediction.write_bytes(b"mutated prediction")
    with pytest.raises(ValueError, match="prediction or sidecar bytes changed"):
        publisher._fit_export(root, recovery, expected["job"], _initial_placement)


def test_moved_cpu_requires_exact_active_helper_placement_authorization(tmp_path: Path) -> None:
    root = tmp_path
    recovery = root / "state/r01-delivery-recovery-v2"
    auth = {
        "authority": "sealed_density4_launch_event",
        "control_manifest_sha256": "a" * 64,
        "launch_event_sha256": "b" * 64,
        "panel_initial_cpu_ids": [0, 2],
        "actual_cpu_ids": [56, 58],
    }
    expected = _make_fit_export(
        root,
        recovery,
        "wide_f0_s1003",
        "wide",
        cpu_affinity=[56, 58],
        placement_authorization=auth,
    )

    def verify_moved(_root: Path, _job: dict, resource: dict) -> dict:
        assert resource["cpu_affinity"] == [56, 58]
        return auth

    assert publisher._fit_export(root, recovery, expected["job"], verify_moved)
    fit = publisher.read_json(expected["fit_path"])
    fit["original_actor_phase"].pop("placement_authorization")
    _write_json(expected["fit_path"], fit)
    with pytest.raises(ValueError, match="lacks exact sealed launch evidence"):
        publisher._fit_export(root, recovery, expected["job"], verify_moved)


def test_full_recovery_grade_chain_uses_official_validator_and_exact_receipt_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path
    recovery = root / "state/r01-delivery-recovery-v2"
    wide = _make_fit_export(root, recovery, "wide_f0_s1003", "wide")
    deep = _make_fit_export(root, recovery, "deep_f0_s1003", "deep")
    grade_root = recovery / "grades/seed1003/fold0"
    grade_root.mkdir(parents=True)
    split_path = root / "splits/fold0.json"
    _write_json(split_path, {"fold": 0, "evaluation": {"valid": [4, 7]}})
    split_sha = publisher.sha256_file(split_path)
    bindings = {
        "wide": {
            "checkpoint_sha256": wide["identity"]["fixed_file_sha256"],
            "prediction_sha256": wide["export_receipt"]["prediction_sha256"],
            "sidecar_sha256": wide["export_receipt"]["sidecar_sha256"],
        },
        "deep": {
            "checkpoint_sha256": deep["identity"]["fixed_file_sha256"],
            "prediction_sha256": deep["export_receipt"]["prediction_sha256"],
            "sidecar_sha256": deep["export_receipt"]["sidecar_sha256"],
        },
    }
    grade = {
        "schema": "priority25.endpoint_grade.v1",
        "metric_id": publisher.METRIC_ID,
        "fold": 0,
        "split_manifest_sha256": split_sha,
        "evaluation_row_groups": [4, 7],
        "endpoint": {
            "run": deep["job"]["id"],
            "epoch": 5,
            **bindings["deep"],
            "score": {"wp": 0.52},
        },
        "reference": {
            "run": wide["job"]["id"],
            "epoch": 5,
            **bindings["wide"],
            "score": {"wp": 0.50},
        },
        "paired_component_bootstrap": {"resamples": 2000, "seed": 271828},
    }
    grade_path = grade_root / "grade.json"
    _write_json(grade_path, grade)
    command_path = grade_root / "grade_command.json"
    argv = [
        sys.executable,
        str(root / "runtime/b300-r01-v1/grader/grade_endpoint.py"),
        "--root",
        str(grade_root / "view"),
        "--run",
        deep["job"]["id"],
        "--split",
        str(split_path),
        "--epoch",
        "5",
        "--reference",
        wide["job"]["id"] + "@5",
        "--output",
        str(grade_path),
        "--bootstrap",
        "2000",
        "--scratch-dir",
        str(grade_root / "scratch"),
    ]
    command = {
        "schema": "tabdata-r01-delivery-grade-command.v2",
        "argv": argv,
        "candidate": deep["job"]["id"],
        "reference": wide["job"]["id"],
        "fold": 0,
        "seed": 1003,
        "epoch": 5,
        "grader_sha256": publisher.GRADER_SHA256,
        "wide_fit_recovery_sha256": publisher.sha256_file(wide["fit_path"]),
        "deep_fit_recovery_sha256": publisher.sha256_file(deep["fit_path"]),
        "wide_export_receipt_sha256": publisher.sha256_file(
            recovery / "exports/wide_f0_s1003/export_receipt.json"
        ),
        "deep_export_receipt_sha256": publisher.sha256_file(
            recovery / "exports/deep_f0_s1003/export_receipt.json"
        ),
        "split_sha256": split_sha,
        "cuda_disabled": True,
    }
    _write_json(command_path, command)
    stdout = grade_root / "grade.stdout.txt"
    stderr = grade_root / "grade.stderr.txt"
    stdout.write_text("grade done\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    receipt_path = grade_root / "grade_receipt.json"
    receipt = {
        "schema": "tabdata-r01-delivery-grade.v2",
        "status": "graded",
        "candidate": deep["job"]["id"],
        "reference": wide["job"]["id"],
        "fold": 0,
        "seed": 1003,
        "metric_id": publisher.METRIC_ID,
        "grade_path": str(grade_path),
        "grade_sha256": publisher.sha256_file(grade_path),
        "command_sha256": publisher.sha256_file(command_path),
        "stdout_sha256": publisher.sha256_file(stdout),
        "stderr_sha256": publisher.sha256_file(stderr),
        "elapsed_seconds": 12.5,
        "score": {"deep_wp": 0.52, "wide_wp": 0.50},
    }
    _write_json(receipt_path, receipt)

    class OriginalGrader:
        @staticmethod
        def load_campaign_split(_path: Path, _view: Path) -> SimpleNamespace:
            return SimpleNamespace(fold=0, sha256=split_sha, evaluation_groups=[4, 7])

        @staticmethod
        def validate_prediction(
            _view: Path, run_name: str, _epoch: int, _groups: list[int], _split: object
        ):
            return None, bindings["deep" if "deep" in run_name else "wide"]

    result = publisher._verify_pair(
        root,
        recovery,
        0,
        1003,
        {"wide": wide["job"], "deep": deep["job"]},
        OriginalGrader,
        _initial_placement,
    )
    assert result["wide_wp"] == 0.50
    assert result["deep_wp"] == 0.52
    assert result["delta_wp"] == pytest.approx(0.02)
    assert result["endpoints"]["deep"]["checkpoint_sha256"] == bindings["deep"]["checkpoint_sha256"]


def test_recovery_comments_have_a_distinct_namespace_and_preserve_old_status() -> None:
    snapshot = {
        "experiment_id": publisher.EXPERIMENT_ID,
        "expected_cells": 30,
        "completed": [],
        "active": 0,
        "pending": 30,
        "terminal": False,
        "source": {"panel_sha256": publisher.PANEL_SHA256},
        "errors": [],
        "contract_note": "Delivery-only recovery.",
    }
    old_status = "<!-- tabdata-b300-20261018:E-B300-R01:status: -->\nold full-panel status"
    own_comments = [{"id": 9, "body": old_status, "user": {"login": publisher.PUBLISHER_LOGIN}}]

    class FakeClient:
        def __init__(self) -> None:
            self.posts: list[tuple[int, str]] = []
            self.patches: list[int] = []

        def request(self, _method: str, path: str) -> dict:
            assert path == f"/repos/{publisher.REPOSITORY}/pulls/{publisher.PR_NUMBER}"
            return {
                "state": "open",
                "base": {"ref": "research-team", "repo": {"full_name": publisher.REPOSITORY}},
                "head": {"ref": publisher.HEAD_REF, "repo": {"full_name": publisher.REPOSITORY}},
            }

        def comments(self, number: int) -> list[dict]:
            assert number == 25
            return own_comments

        def post(self, number: int, body: str) -> dict:
            self.posts.append((number, body))
            return {"id": 10, "html_url": "https://example.test/recovery-status"}

        def patch(self, comment_id: int, _body: str) -> dict:
            self.patches.append(comment_id)
            return {"id": comment_id, "html_url": "https://example.test/old-status"}

    client = FakeClient()
    cursor: dict = {"pairs": {}, "result_urls": []}
    publisher._publish(
        client,
        {},
        Path("."),
        snapshot,
        {"publisher_login": publisher.PUBLISHER_LOGIN, "evidence_url": "https://example.test"},
        cursor,
    )
    assert len(client.posts) == 1 and client.posts[0][0] == 25
    assert publisher._marker("status") in client.posts[0][1]
    assert "does not replace or edit the existing full-panel" in client.posts[0][1]
    assert client.patches == []


def test_comment_ownership_filter_excludes_teammate_copied_marker() -> None:
    rows = [
        {"id": 1, "user": {"login": publisher.PUBLISHER_LOGIN}},
        {"id": 2, "user": {"login": "teammate"}},
    ]
    assert [row["id"] for row in publisher._own_comments(rows, publisher.PUBLISHER_LOGIN)] == [1]


def test_pinned_archive_helper_can_resolve_late_render_report_import(tmp_path: Path) -> None:
    helpers = publisher._load_github(_github_code_dir(), publisher.GITHUB_CODE_MANIFEST_SHA256)
    evidence = tmp_path / "state" / "grade.json"
    weights = tmp_path / "runs" / "model.pt"
    _write_json(evidence, {"score": 0.5})
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"checkpoint")
    row = {
        "id": "r01-recovery-test",
        "receipt_sha256": _sha("receipt"),
        "evidence_paths": [str(evidence)],
        "weights_paths": [str(weights)],
    }

    archived = helpers["package_evidence"].ensure_archives(tmp_path, row)
    assert Path(archived["archive_manifest_path"]).is_file()
    assert len(archived["archive_manifest_sha256"]) == 64


def test_verified_split_is_copied_into_the_archive_allowlist(tmp_path: Path) -> None:
    helpers = publisher._load_github(_github_code_dir(), publisher.GITHUB_CODE_MANIFEST_SHA256)
    split = tmp_path / "splits/fold0.json"
    grade = tmp_path / "state/grades/grade.json"
    weights = tmp_path / "runs/model.pt"
    _write_json(split, {"fold": 0, "evaluation": [1, 3]})
    _write_json(grade, {"score": 0.5})
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"checkpoint")
    split_sha = publisher.sha256_file(split)

    evidence = publisher._archive_evidence_paths(
        tmp_path, 0, split_sha, [str(split.resolve()), str(grade.resolve())]
    )
    preserved = Path(next(path for path in evidence if "split_manifests" in path))
    assert publisher.sha256_file(preserved) == split_sha
    assert str(split.resolve()) not in evidence
    archive = helpers["package_evidence"].ensure_archives(
        tmp_path,
        {
            "id": "r01-recovery-fold0",
            "receipt_sha256": _sha("grade receipt"),
            "evidence_paths": evidence,
            "weights_paths": [str(weights.resolve())],
        },
    )
    manifest = publisher.read_json(Path(archive["archive_manifest_path"]))
    assert any(record["path"] == str(preserved) for record in manifest["evidence"]["files"])
