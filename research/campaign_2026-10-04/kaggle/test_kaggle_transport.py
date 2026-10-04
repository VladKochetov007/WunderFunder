from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
common = importlib.import_module("kaggle_common")
runner = importlib.import_module("kaggle_gpu_runner")
prepare = importlib.import_module("kaggle_prepare_probe").prepare
canonical_sha256 = common.canonical_sha256
read_spec = common.read_spec
relative_path = common.relative_path
sha256_file = common.sha256_file
verify_files = common.verify_files
admit_specs = runner.admit_specs
export_artifacts = runner.export_artifacts


def make_spec(root: Path) -> dict:
    """Construct a sealed CPU fixture without importing GPU frameworks."""
    source = root / "worker.py"
    source.write_text("raise SystemExit(0)\n")
    code = [{"path": "worker.py", "size": source.stat().st_size, "sha256": sha256_file(source)}]
    return {
        "version": 1,
        "name": "fixture",
        "kind": "test",
        "argv": ["{python}", "worker.py"],
        "code": code,
        "code_sha256": canonical_sha256(code),
        "inputs": [],
        "artifacts": ["result.json"],
        "environment": {},
        "limits": {
            "cpus": 1,
            "ram_bytes": 1024**3,
            "timeout_seconds": 10,
            "gpu_memory_bytes": 12 * 1024**3,
            "gpu_stop_bytes": 13 * 1024**3,
        },
    }


@pytest.mark.parametrize(
    "name", ["../secret", "/root/file", "a/../../x", ".env", "key/token.json", ".git/config"]
)
def test_unsafe_paths(name: str) -> None:
    with pytest.raises(ValueError):
        relative_path(name)


def test_tampered_source_is_rejected(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    parsed = read_spec(path)
    verify_files(tmp_path, parsed["code"])
    (tmp_path / "worker.py").write_text("raise SystemExit(1)\n")
    with pytest.raises(ValueError, match="identity"):
        verify_files(tmp_path, parsed["code"])


def test_aggregate_host_reservations(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    host = {"capacity_bytes": 8 * 1024**3, "available_bytes": 7 * 1024**3, "affinity": [0, 1]}
    devices = [
        {"index": n, "name": "Tesla T4", "used_bytes": 0, "total_bytes": 15 * 1024**3}
        for n in range(2)
    ]
    assert admit_specs([spec, spec], host, devices) == 2
    spec["limits"]["ram_bytes"] = 4 * 1024**3
    assert admit_specs([spec, spec], host, devices) == 1
    devices[0]["used_bytes"] = 256 * 1024**2
    with pytest.raises(ValueError, match="preexisting"):
        admit_specs([spec], host, devices)


def test_artifact_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    spec = make_spec(root)
    outside = tmp_path / "outside"
    outside.write_text("private")
    (root / "result.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        export_artifacts(root, spec, tmp_path / "output")


def test_probe_privacy_and_source_binding(tmp_path: Path) -> None:
    target = tmp_path / "probe"
    prepare(ROOT, target)
    metadata = json.loads((target / "kernel-metadata.json").read_text())
    provenance = json.loads((target / "provenance.json").read_text())
    assert metadata["is_private"] and metadata["enable_tpu"]
    assert not metadata["enable_gpu"] and not metadata["enable_internet"]
    assert metadata["dataset_sources"] == []
    assert sha256_file(target / "probe.py") == provenance["probe_sha256"]
    module_spec = importlib.util.spec_from_file_location("probe_fixture", target / "probe.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert "class RNNModel" in module.MODEL_SOURCE
    assert provenance["model_class_sha256"] == module.MODEL_SHA256


@pytest.mark.parametrize("exit_code", [0, 1])
def test_worker_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int) -> None:
    import os

    root = tmp_path / "root"
    root.mkdir()
    spec = make_spec(root)
    source = root / "worker.py"
    source.write_text(
        f"from pathlib import Path\nPath('result.json').write_text('{{}}')\nraise SystemExit({exit_code})\n"
    )
    spec["code"][0].update(size=source.stat().st_size, sha256=sha256_file(source))
    spec["code_sha256"] = canonical_sha256(spec["code"])
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    affinity = sorted(os.sched_getaffinity(0))[:1]
    monkeypatch.setattr(
        runner,
        "host_inventory",
        lambda: {
            "capacity_bytes": 8 * 1024**3,
            "available_bytes": 7 * 1024**3,
            "affinity": affinity,
        },
    )
    monkeypatch.setattr(
        runner,
        "gpu_inventory",
        lambda: [
            {
                "index": 0,
                "uuid": "GPU-fixture",
                "name": "Tesla T4",
                "used_bytes": 0,
                "total_bytes": 15 * 1024**3,
            }
        ],
    )
    output = tmp_path / "output"
    if exit_code:
        with pytest.raises(RuntimeError, match="Worker failed"):
            runner.run(root, [path], output, stagger_seconds=0)
    else:
        runner.run(root, [path], output, stagger_seconds=0)
    result = json.loads((output / "fixture/result.json").read_text())
    assert result["returncode"] == exit_code
    assert result["status"] == ("failed" if exit_code else "complete")
    assert (output / "fixture/artifacts/result.json").is_file()


def test_bundle_excludes_external_raw_and_binds_kernel(tmp_path: Path) -> None:
    import shutil
    import tarfile

    bundle = importlib.import_module("kaggle_bundle")
    root = tmp_path / "root"
    root.mkdir()
    spec = make_spec(root)
    (root / "scripts").mkdir()
    for name in ("kaggle_common.py", "kaggle_gpu_runner.py", "kaggle_kernel.py"):
        relative = f"scripts/{name}"
        shutil.copyfile(ROOT / relative, root / relative)
        spec["code"].append(
            {
                "path": relative,
                "size": (root / relative).stat().st_size,
                "sha256": sha256_file(root / relative),
            }
        )
    raw = root / "datasets/valid.parquet"
    raw.parent.mkdir()
    raw.write_bytes(b"fixture raw data")
    spec["inputs"] = [
        {"path": "datasets/valid.parquet", "size": raw.stat().st_size, "sha256": sha256_file(raw)}
    ]
    spec["code_sha256"] = canonical_sha256(sorted(spec["code"], key=lambda item: item["path"]))
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec))
    destination = tmp_path / "bundle"
    bundle.build_bundle(
        root,
        [spec_path],
        destination,
        "vladkochetov07/test-bundle",
        ["datasets/valid.parquet=valid.parquet"],
    )
    with tarfile.open(destination / "bundle.tar.gz") as archive:
        assert "datasets/valid.parquet" not in archive.getnames()
        assert "kaggle_specs/fixture.json" in archive.getnames()
    manifest = json.loads((destination / "bundle-manifest.json").read_text())
    assert manifest["archive_sha256"] in (destination / "kernel/kernel.py").read_text()
    assert json.loads((destination / "kernel/kernel-metadata.json").read_text())["is_private"]
