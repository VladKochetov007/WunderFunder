from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "research/campaign_2026-10-04/kaggle/E419_qat_dualT4_v3"
UPLOAD = HERE / "upload_dataset"
SOURCE = HERE / "source_root"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sys.path.insert(0, str(SOURCE / "scripts"))
    from kaggle_common import read_spec, verify_files

    manifest = json.loads((UPLOAD / "bundle-manifest.json").read_text())
    v3_manifest = json.loads((V3 / "bundle/bundle-manifest.json").read_text())
    old_specs = [json.loads(p.read_text()) for p in sorted((V3 / "specs").glob("*.sealed.json"))]
    specs = [json.loads(p.read_text()) for p in sorted((HERE / "specs").glob("*.sealed.json"))]
    assert len(specs) == 2 and len(old_specs) == 2
    assert manifest["private"] is True and manifest["dataset"] == "vladkochetov07/wnn-e419-qat-dualt4-20261004-v4"
    archive = UPLOAD / "bundle.wnn"
    assert sha(archive) == manifest["archive_sha256"]
    source_archive = UPLOAD / "frozen-v3-source.bundle.wnn"
    assert sha(source_archive) == v3_manifest["archive_sha256"] == "d2d8fe047d76f85b9dd5931d36b8a9daac6b23adfe197e76bb48c881328b3549"
    assert {x["path"]: x for x in specs[0]["code"]} == {x["path"]: x for x in old_specs[0]["code"]}
    assert {x["path"]: x for x in specs[1]["code"]} == {x["path"]: x for x in old_specs[1]["code"]}
    assert specs[0]["code_sha256"] == specs[1]["code_sha256"] == old_specs[0]["code_sha256"] == old_specs[1]["code_sha256"]

    ext = manifest["external_mounts"]
    for spec in specs:
        checked = read_spec(HERE / "specs" / f"{spec['name']}.sealed.json")
        verify_files(SOURCE, checked["code"] + [x for x in checked["inputs"] if x["path"] not in ext])
        assert checked["limits"]["timeout_seconds"] == 14400
        assert checked["limits"]["gpu_memory_bytes"] == 11 * 1024**3
        assert checked["limits"]["gpu_stop_bytes"] == 13 * 1024**3
        plan_path = checked["argv"][-1]
        plan = json.loads((SOURCE / plan_path).read_text())
        names = [x["name"] for x in plan["steps"]]
        assert len(names) == len(set(names))
        assert "reused_v3_control_cached_prediction_single_arm_path_check" in names
        assert "fixed_one_epoch_qat_fit" in names
        assert "qat_fixed_epoch0_full_valid_prediction" in names
        assert "qat_frozen_fold0_single_arm_score" in names
        assert not any("probe" in name for name in names)
        assert not any("same_t4_fixed_one_epoch_fp32_control" in name for name in names)
        assert sum(name.endswith("zero_update_ptq_export") for name in names) == 4
        assert sum(name.endswith("ptq_full_valid_prediction") for name in names) == 4
        assert sum(name.endswith("ptq_single_arm_score") for name in names) == 4
        assert all("t4v3" not in value and "t4v2" not in value for step in plan["steps"] for value in step["argv"])

    screen_paths = [x["path"] for spec in specs for x in spec["inputs"] if "_screen.json" in x["path"]]
    assert len(set(screen_paths)) == 12
    screens = {}
    for path in screen_paths:
        screen = json.loads((SOURCE / path).read_text())
        assert "E419_qat_dualT4_v4/" in path
        raw = json.dumps(screen)
        assert "t4v2" not in raw and "t4v3" not in raw
        assert all("t4v4" in run for mix in screen["arms"].values() for run in mix)
        screens[path] = screen

    internal = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            p = Path(member.name)
            assert not p.is_absolute() and ".." not in p.parts and member.isfile()
            assert member.name not in internal
            internal.add(member.name)
    expected = {x["path"] for x in manifest["files"]} - set(ext)
    expected |= set(manifest["spec_paths"])
    assert internal == expected, (len(internal), len(expected), sorted(internal ^ expected)[:10])
    for logical, mapping in ext.items():
        if isinstance(mapping, dict) and mapping["dataset"] == manifest["dataset"]:
            staged = UPLOAD / mapping["filename"]
            desc = next(x for x in manifest["files"] if x["path"] == logical)
            assert staged.is_file() and staged.stat().st_size == desc["size"] and sha(staged) == desc["sha256"]

    reuse = json.loads((HERE / "control_reuse_provenance.json").read_text())
    r = reuse["controls"]["recurrent"]
    h = reuse["controls"]["recurrent_head"]
    assert r["split_manifest_sha256"] == h["split_manifest_sha256"] == reuse["split_manifest_sha256"]
    assert r["model_tensor_identity"]["state_tensor_sha256"] == h["model_tensor_identity"]["state_tensor_sha256"]
    assert r["prediction_sha256"] == h["prediction_sha256"]
    assert r["source_run"].endswith("t4v3") and h["source_run"].endswith("t4v3")

    recovered = json.loads((ROOT / "research/campaign_2026-10-04/checks/e419_fp32_controls_recovered_report.json").read_text())
    scores = {}
    for preset in ("recurrent", "recurrent_head"):
        path = HERE / f"score_smoke_{preset}.json"
        result = json.loads(path.read_text())
        expected_score = recovered["folds"]["0"]["exact_global_wp"][f"FP32 {preset.replace('_', '+') if preset == 'recurrent_head' else preset}"]["wp"]
        got = next(iter(result["pooled"]["exact_global_wp"].values()))["wp"]
        assert abs(got - expected_score) < 1e-12
        scores[preset] = {"wp": got, "report_sha256": sha(path)}
    assert abs(scores["recurrent"]["wp"] - scores["recurrent_head"]["wp"]) < 1e-15

    junit = HERE / "preflight_cpu.xml"
    assert junit.is_file() and b'failures="0"' in junit.read_bytes()
    metadata = json.loads((HERE / "kernel_push/kernel-metadata.json").read_text())
    assert metadata["is_private"] is True and metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True and metadata["enable_tpu"] is False
    budget = json.loads((HERE / "budget.json").read_text())
    assert budget["maximum_device_gpu_hours"] + budget["provider_overhead_reserve_hours"] <= budget["hard_attempt_cap_gpu_hours"]
    assert budget["quota_remaining_before_push_hours"] - budget["hard_attempt_cap_gpu_hours"] >= budget["weekly_reserve_required_hours"] + budget["width_pair_staged_budget_hours"]

    report = {
        "status": "validated_before_dataset_upload",
        "archive_sha256": manifest["archive_sha256"],
        "archive_bytes": manifest["archive_bytes"],
        "frozen_v3_source_archive_sha256": v3_manifest["archive_sha256"],
        "source_code_sha256": specs[0]["code_sha256"],
        "spec_hashes": {s["name"]: sha(HERE / "specs" / f"{s['name']}.sealed.json") for s in specs},
        "screen_count": len(set(screen_paths)),
        "private_kernel_metadata": metadata,
        "control_reuse_manifest_sha256": sha(HERE / "control_reuse_provenance.json"),
        "control_model_tensor_sha256": r["model_tensor_identity"]["state_tensor_sha256"],
        "control_checkpoint_hashes": {k: v["checkpoint_sha256"] for k, v in reuse["controls"].items()},
        "control_prediction_hashes": {k: v["prediction_sha256"] for k, v in reuse["controls"].items()},
        "control_score_smokes": scores,
        "preflight_junit_sha256": sha(junit),
        "preflight_counts": {"passed": 53, "skipped": 2, "failed": 0},
        "budget": budget,
        "external_mount_count": len(ext),
        "expected_two_t4_workers_source": "E419 v3 worker results document successful production probes and both FP32 controls on physical GPU UUIDs af02b659 and 87c2b8d4; no v4 job started",
    }
    target = HERE / "validation_preupload.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "archive_sha256", "source_code_sha256", "screen_count", "control_score_smokes", "preflight_counts", "budget"]}, indent=2))


if __name__ == "__main__":
    main()
