from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
SEALED = ROOT / "cpu_source_stage_candidate_r1_final"


def _load_runtime() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cpu_stage_runtime_test", ROOT / "cpu_stage_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_payload(root: Path, name: str, payload: bytes) -> dict[str, object]:
    filename = f"{_sha(payload)}.wnn"
    (root / filename).write_bytes(payload)
    return {"filename": filename, "path": name, "sha256": _sha(payload), "size": len(payload)}


def test_nested_raw_dataset_discovery_and_ambiguity(tmp_path: Path) -> None:
    runtime = _load_runtime()
    mount = tmp_path / "datasets/owner/wunder-funder/1"
    mount.mkdir(parents=True)
    (mount / "train.parquet").touch()
    (mount / "valid.parquet").touch()

    assert runtime.discover_raw_root(tmp_path) == mount
    extra = tmp_path / "datasets/owner/other/1"
    extra.mkdir(parents=True)
    (extra / "train.parquet").touch()
    (extra / "valid.parquet").touch()
    with pytest.raises(runtime.StageRejected, match="expected one raw-data root"):
        runtime.discover_raw_root(tmp_path)


def test_exact_asset_manifest_and_source_archive_unpack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _load_runtime()
    source_dataset = "owner/private-source"
    monkeypatch.setattr(runtime, "SOURCE_DATASET", source_dataset)
    archive_buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=archive_buffer, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name in runtime.CODE_FILES:
            payload = f"# fixture {name}\n".encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    archive_payload = archive_buffer.getvalue()
    fixture_files = {name: f"# fixture {name}\n".encode() for name in runtime.CODE_FILES}
    code_manifest = {
        "schema": "wnn.kaggle-source-code-manifest",
        "format_version": 1,
        "commit": runtime.EXPECTED_SOURCE_COMMIT,
        "files": [
            {"path": name, "size": len(payload), "sha256": _sha(payload)}
            for name, payload in fixture_files.items()
        ],
    }
    code_manifest_payload = json.dumps(code_manifest).encode()
    membership_payload = b'{"fixture":"membership"}\n'
    entries = [
        _write_payload(tmp_path, runtime.SOURCE_ARCHIVE_PATH, archive_payload),
        _write_payload(tmp_path, runtime.CODE_MANIFEST_PATH, code_manifest_payload),
        _write_payload(tmp_path, runtime.MEMBERSHIP_PATH, membership_payload),
    ]
    manifest = {
        "dataset": source_dataset,
        "private": True,
        "version": 1,
        "files": entries,
    }
    manifest_path = tmp_path / "assets-manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    monkeypatch.setattr(runtime, "EXPECTED_ASSET_MANIFEST_SHA256", _sha(manifest_bytes))
    monkeypatch.setattr(runtime, "EXPECTED_ARCHIVE_SHA256", _sha(archive_payload))
    monkeypatch.setattr(runtime, "EXPECTED_CODE_MANIFEST_SHA256", _sha(code_manifest_payload))
    monkeypatch.setattr(runtime, "EXPECTED_MEMBERSHIP_SHA256", _sha(membership_payload))

    asset_root, found_manifest = runtime.discover_source_asset(tmp_path)
    repo = runtime.unpack_source(asset_root, found_manifest, tmp_path / "repo")

    assert all((repo / name).read_bytes() == fixture_files[name] for name in runtime.CODE_FILES)
    assert (repo / runtime.MEMBERSHIP_PATH).read_bytes() == membership_payload


def test_source_archive_rejects_undeclared_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _load_runtime()
    code_manifest = {
        "schema": "wnn.kaggle-source-code-manifest",
        "format_version": 1,
        "commit": runtime.EXPECTED_SOURCE_COMMIT,
        "files": [{"path": name, "size": 0, "sha256": _sha(b"")} for name in runtime.CODE_FILES],
    }
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    code_bytes = json.dumps(code_manifest).encode()
    (asset_root / "code.wnn").write_bytes(code_bytes)
    archive_path = asset_root / "archive.wnn"
    with (
        gzip.GzipFile(filename=str(archive_path), mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        payload = b"bad"
        info = tarfile.TarInfo("unexpected.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    membership_path = asset_root / "membership.wnn"
    membership_path.write_bytes(b"{}")
    manifest = {
        "files": [
            {"path": runtime.SOURCE_ARCHIVE_PATH, "filename": "archive.wnn"},
            {"path": runtime.CODE_MANIFEST_PATH, "filename": "code.wnn"},
            {"path": runtime.MEMBERSHIP_PATH, "filename": "membership.wnn"},
        ]
    }
    with pytest.raises(runtime.StageRejected, match="unexpected or repeated"):
        runtime.unpack_source(asset_root, manifest, tmp_path / "repo")


def test_kernel_template_places_boot_marker_before_runtime_import() -> None:
    source = (ROOT / "kernel.py").read_text(encoding="utf-8")
    assert source.index("_boot_marker()") < source.index("CPU_STAGE_RUNTIME_INSERTION_POINT")
    assert '"event": "kernel_boot"' in source


def test_source_stage_budget_guards_fail_closed() -> None:
    runtime = _load_runtime()
    required = runtime.require_disk_headroom(20_000_000_000)
    assert required == (
        runtime.CACHE_ARRAY_PAYLOAD_BYTES
        + runtime.OUTPUT_RESERVE_BYTES
        + runtime.CHECKPOINT_RESERVE_BYTES
        + runtime.FREE_RESERVE_BYTES
        + 2_000_000
    )
    with pytest.raises(runtime.StageRejected, match="working disk free"):
        runtime.require_disk_headroom(required - 1)
    assert (
        runtime._abort_reason(
            {
                "output_bytes": runtime.OUTPUT_CAP_BYTES + 1,
                "memory": {},
                "mem_available_bytes": None,
            },
            time.monotonic() + 100,
        )
        == f"working output exceeded {runtime.OUTPUT_CAP_BYTES} bytes"
    )


def test_sealed_candidate_hashes_actual_actor_source_and_entrypoint() -> None:
    packet = json.loads((SEALED / "cpu_source_stage.packet.json").read_bytes())
    source_asset = SEALED / "source_asset"
    kernel_dir = SEALED / "kernel"
    assets_bytes = (source_asset / "assets-manifest.json").read_bytes()
    assets_manifest = json.loads(assets_bytes)
    source_entry = next(row for row in assets_manifest["files"] if row["path"] == "wnn_src.tar.gz")
    archive_path = source_asset / source_entry["filename"]
    code_manifest_entry = next(
        row for row in assets_manifest["files"] if row["path"] == "source-code-manifest.json"
    )
    code_manifest_bytes = (source_asset / code_manifest_entry["filename"]).read_bytes()
    code_manifest = json.loads(code_manifest_bytes)
    kernel_path = kernel_dir / packet["kernel"]["script"]["path"].split("/", 1)[1]
    kernel_bytes = kernel_path.read_bytes()
    metadata_bytes = (kernel_dir / "kernel-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)

    assert packet["packet_id"] == metadata["id"]
    assert packet["source_asset"]["assets_manifest_sha256"] == _sha(assets_bytes)
    assert packet["source_asset"]["archive_sha256"] == _sha(archive_path.read_bytes())
    assert packet["source_asset"]["code_manifest_sha256"] == _sha(code_manifest_bytes)
    assert packet["source_asset"]["membership_sha256"] == (
        "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
    )
    assert code_manifest["commit"] == "d5daaa0f90b0362877522703c7e17aa78c19cac9"
    assert _sha(kernel_bytes) == packet["kernel"]["script"]["sha256"]
    assert _sha(metadata_bytes) == packet["kernel"]["metadata"]["sha256"]
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert len(kernel_bytes) < 1_000_000
    compile(kernel_bytes, str(kernel_path), "exec")
    assert packet["source_asset"]["assets_manifest_sha256"] in kernel_bytes.decode()
    assert packet["source_asset"]["archive_sha256"] in kernel_bytes.decode()
    with (
            gzip.GzipFile(fileobj=io.BytesIO(archive_path.read_bytes()), mode="rb") as source,
            tarfile.open(fileobj=source, mode="r:") as archive,
    ):
        members = list(archive)
        source_files = {}
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is not None:
                with extracted:
                    source_files[member.name] = extracted.read()
    assert [member.name for member in members] == sorted(
        row["path"] for row in code_manifest["files"]
    )
    assert {
        name: {"size": len(payload), "sha256": _sha(payload)}
        for name, payload in source_files.items()
    } == {
        row["path"]: {"size": row["size"], "sha256": row["sha256"]}
        for row in code_manifest["files"]
    }
