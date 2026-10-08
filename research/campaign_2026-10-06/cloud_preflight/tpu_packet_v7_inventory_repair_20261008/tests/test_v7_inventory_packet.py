from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "packet"
PROBE_PATH = PACKET / "probe.py"
MANIFEST_PATH = ROOT / "packet_manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_probe_namespace() -> dict[str, object]:
    source = PROBE_PATH.read_text(encoding="utf-8")
    namespace: dict[str, object] = {
        "__name__": "e463_v7_contract_probe",
        "__file__": str(PROBE_PATH),
    }
    exec(compile(source, str(PROBE_PATH), "exec"), namespace)
    return namespace


def _manifest_entries() -> list[dict[str, object]]:
    manifest = json.loads(
        (ROOT / "reference/fold0_assets_manifest.json").read_text(encoding="utf-8")
    )
    return manifest["files"]


def _materialize_mount(tmp_path: Path, entries: list[dict[str, object]]) -> None:
    for entry in entries:
        (tmp_path / str(entry["filename"])).touch()
    (tmp_path / "assets-manifest.json").touch()


def test_packet_is_hash_bound_and_source_only() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    code = PROBE_PATH.read_bytes()
    metadata = (PACKET / "kernel-metadata.json").read_bytes()
    assert manifest["code_sha256"] == _sha256(code)
    assert manifest["code_bytes"] == len(code) < 1_000_000
    assert manifest["metadata_sha256"] == _sha256(metadata)
    assert manifest["bundle_sha256"] == _sha256((ROOT / "packet.bundle.zip").read_bytes())
    assert manifest["builder_sha256"] == _sha256(
        (ROOT / "reference/build_v7_probe.py").read_bytes()
    )
    assert manifest["test_sha256"] == _sha256(Path(__file__).read_bytes())
    assert manifest["readme_sha256"] == _sha256((ROOT / "README.md").read_bytes())
    assert manifest["provider_push_authorized"] is False
    assert manifest["source_only"] is True
    assert manifest["inventory_repair"]["no_scientific_recipe_change"] is True

    metadata_json = json.loads(metadata)
    assert metadata_json["id"] == manifest["packet_id"]
    assert metadata_json["title"] == manifest["packet_title"]
    assert metadata_json["enable_gpu"] is False
    assert metadata_json["enable_tpu"] is True
    assert set(metadata_json["dataset_sources"]) == set(manifest["requested_inputs"])

    with zipfile.ZipFile(ROOT / "packet.bundle.zip") as archive:
        assert set(archive.namelist()) == {"kernel-metadata.json", "probe.py"}
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
        assert archive.read("probe.py") == code
        assert archive.read("kernel-metadata.json") == metadata


def test_repair_matches_frozen_failure_and_actual_manifest_counts() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = _manifest_entries()
    namespace = _load_probe_namespace()
    assert len(entries) == manifest["assets_manifest_payload_entries"] == 23
    assert manifest["assets_mount_expected_regular_files"] == len(entries) + 1
    assert namespace["ASSET_MANIFEST_SHA256"] == manifest["assets_manifest_sha256"]
    assert namespace["PACKET_ID"] == "e463-tpu-realdata-reload-v7-20261008"
    assert namespace["OUTPUT"] == Path("/kaggle/working/cloud_preflight_tpu_v7")

    source = PROBE_PATH.read_text(encoding="utf-8")
    assert 'len(entries) != 24' not in source
    assert "by_path = validate_asset_inventory(asset_root, entries)" in source
    assert "len(entries) != 23" in source
    assert '"artifacts/runs/c20260929_since_s3_f0/model.pt"' in source
    assert '"artifacts/runs/c20260929_since_s4_f0/model.pt"' in source
    assert '"artifacts/runs/c20260929_since_s3_f0/model_ep3.pt"' not in source
    assert '"artifacts/runs/c20260929_since_s4_f0/model_ep3.pt"' not in source


def test_inventory_accepts_exact_23_payload_plus_manifest_mount(tmp_path: Path) -> None:
    entries = _manifest_entries()
    _materialize_mount(tmp_path, entries)
    validate = _load_probe_namespace()["validate_asset_inventory"]
    by_path = validate(tmp_path, entries)
    assert len(by_path) == 23
    assert len(list(tmp_path.iterdir())) == 24
    assert set(by_path) == {str(entry["path"]) for entry in entries}


def test_inventory_rejects_missing_extra_duplicate_and_unsafe_members(tmp_path: Path) -> None:
    entries = _manifest_entries()
    _materialize_mount(tmp_path, entries)
    validate = _load_probe_namespace()["validate_asset_inventory"]

    with pytest.raises(ValueError, match="exactly 23"):
        validate(tmp_path, entries[:-1])

    (tmp_path / "unexpected.bin").touch()
    with pytest.raises(ValueError, match="mount names differ"):
        validate(tmp_path, entries)
    (tmp_path / "unexpected.bin").unlink()

    duplicate_path = [dict(entry) for entry in entries]
    duplicate_path[-1]["path"] = duplicate_path[0]["path"]
    with pytest.raises(ValueError, match="duplicate paths or filenames"):
        validate(tmp_path, duplicate_path)

    unsafe_name = [dict(entry) for entry in entries]
    unsafe_name[0]["filename"] = "../escape.wnn"
    with pytest.raises(ValueError, match="unsafe fold-0 asset filename"):
        validate(tmp_path, unsafe_name)


def test_inventory_rejects_missing_physical_file_and_nonfile_entry(tmp_path: Path) -> None:
    entries = _manifest_entries()
    _materialize_mount(tmp_path, entries)
    validate = _load_probe_namespace()["validate_asset_inventory"]

    (tmp_path / str(entries[0]["filename"])).unlink()
    with pytest.raises(ValueError, match="mount names differ"):
        validate(tmp_path, entries)

    (tmp_path / str(entries[0]["filename"])).touch()
    (tmp_path / "nested").mkdir()
    with pytest.raises(ValueError, match="direct regular files only"):
        validate(tmp_path, entries)


def test_base_v6_provenance_and_asset_manifest_are_pinned() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    refs = {
        "v6_probe.py": manifest["base_v6_probe_sha256"],
        "v6_packet_manifest.json": manifest["base_v6_packet_manifest_sha256"],
        "v6_terminal_receipt.json": manifest["base_v6_terminal_receipt_sha256"],
        "fold0_assets_manifest.json": manifest["assets_manifest_sha256"],
    }
    for name, expected in refs.items():
        assert _sha256((ROOT / "reference" / name).read_bytes()) == expected

    v6_terminal = json.loads(
        (ROOT / "reference/v6_terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert v6_terminal["terminal_status"] == "KernelWorkerStatus.ERROR"
    assert v6_terminal["failure"]["phase"] == "host_admission"
