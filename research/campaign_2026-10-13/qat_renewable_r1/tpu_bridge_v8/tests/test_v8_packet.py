from __future__ import annotations

import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[3]
V7_PACKET_DIR = (
    REPO_ROOT / "research/campaign_2026-10-06/cloud_preflight/"
    "tpu_packet_v7_inventory_repair_20261008/packet"
)
V7_PACKET_MANIFEST = V7_PACKET_DIR.parent / "packet_manifest.json"
V7_TERMINAL_RECEIPT = (
    REPO_ROOT / "research/campaign_2026-10-06/cloud_preflight/"
    "tpu_v7_terminal_20261008_r1/attempt_001/outputs/"
    "cloud_preflight_tpu_v7/terminal.json"
)
V7_MOUNT_FIXTURE = (
    REPO_ROOT / "research/campaign_2026-10-06/cloud_preflight/"
    "tpu_v7_terminal_20261008_r1/attempt_001/outputs/"
    "cloud_preflight_tpu_v7/input_mounts.json"
)
PORTABLE_INPUTS = REPO_ROOT / "research/campaign_2026-10-04/portable_inputs.json"
ASSET_MANIFEST = V7_PACKET_DIR.parent / "reference/fold0_assets_manifest.json"
PACKET = ROOT / "packet"
MAIN_ROOT = Path(
    os.environ.get(
        "WNN_MAIN_ROOT",
        "/home/vlad/development/wunder/wnn_connectome_starterpack",
    )
)


def _sha256(value: bytes) -> str:
    """Return the digest used by packet and fixture assertions."""
    return hashlib.sha256(value).hexdigest()


def _probe_namespace() -> dict[str, object]:
    """Load the generated entrypoint without importing JAX or running its main."""
    path = PACKET / "probe.py"
    source = path.read_text(encoding="utf-8")
    namespace: dict[str, object] = {"__name__": "v8_contract_probe", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def _portable_fixture() -> dict[str, object]:
    return json.loads(PORTABLE_INPUTS.read_text(encoding="utf-8"))


def _raw_expected(namespace: dict[str, object], split: str) -> dict[str, object]:
    identities = namespace["RAW_DATA_IDENTITIES"]
    return identities[split]


def _small_parquet(path: Path) -> dict[str, object]:
    footer = b"fixture-footer"
    trailer = len(footer).to_bytes(4, "little") + b"PAR1"
    content = b"PAR1" + b"row-group" + footer + trailer
    path.write_bytes(content)
    return {
        "bytes": len(content),
        "sha256": _sha256(content),
        "footer_bytes": len(footer),
        "footer_sha256": _sha256(footer + trailer),
    }


def test_packet_is_source_pinned_deterministic_and_private() -> None:
    manifest = json.loads((ROOT / "packet_manifest.json").read_text(encoding="utf-8"))
    code = (PACKET / "probe.py").read_bytes()
    metadata_bytes = (PACKET / "kernel-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    assert manifest["packet_id"] == metadata["id"]
    assert manifest["packet_title"] == metadata["title"]
    assert manifest["code_sha256"] == _sha256(code)
    assert manifest["code_bytes"] == len(code) < 1_000_000
    assert manifest["bundle_sha256"] == _sha256((ROOT / "packet.bundle.zip").read_bytes())
    assert manifest["metadata_sha256"] == _sha256(metadata_bytes)
    assert manifest["provider_push_authorized"] is False
    assert manifest["source_only"] is True
    assert manifest["builder_sha256"] == _sha256((ROOT / "build_v8_packet.py").read_bytes())
    assert manifest["test_sha256"] == _sha256(Path(__file__).read_bytes())
    assert manifest["readme_sha256"] == _sha256((ROOT / "README.md").read_bytes())
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"][-1] == "vladkochetov07/wunder-funder"
    with zipfile.ZipFile(ROOT / "packet.bundle.zip") as archive:
        assert set(archive.namelist()) == {"kernel-metadata.json", "probe.py"}
        assert archive.read("probe.py") == code
        assert archive.read("kernel-metadata.json") == metadata_bytes
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_v7_failure_and_real_mount_receipt_are_pinned() -> None:
    manifest = json.loads((ROOT / "packet_manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_v7_probe_sha256"] == _sha256((V7_PACKET_DIR / "probe.py").read_bytes())
    assert manifest["base_v7_manifest_sha256"] == _sha256(V7_PACKET_MANIFEST.read_bytes())
    assert manifest["v7_terminal_receipt_sha256"] == _sha256(V7_TERMINAL_RECEIPT.read_bytes())
    assert manifest["actual_v7_mount_fixture_sha256"] == _sha256(V7_MOUNT_FIXTURE.read_bytes())
    assert manifest["portable_inputs_sha256"] == _sha256(PORTABLE_INPUTS.read_bytes())
    assert manifest["assets_manifest_sha256"] == _sha256(ASSET_MANIFEST.read_bytes())
    terminal = json.loads(V7_TERMINAL_RECEIPT.read_text(encoding="utf-8"))
    assert terminal["phase"] == "host_admission"
    assert "portable registry differs for datasets/train.parquet" in terminal["error"]
    mounts = json.loads(V7_MOUNT_FIXTURE.read_text(encoding="utf-8"))
    raw_mounts = [
        item
        for item in mounts["dataset_mounts"]
        if item["owner"] == "vladkochetov07" and item["dataset"] == "wunder-funder"
    ]
    assert len(raw_mounts) == 1
    assert raw_mounts[0]["path"] == ("/kaggle/input/datasets/vladkochetov07/wunder-funder")


def test_manifest_freezes_actual_data_scope_resource_bounds_and_collection() -> None:
    manifest = json.loads((ROOT / "packet_manifest.json").read_text(encoding="utf-8"))
    execution = manifest["execution"]
    assert execution["real_optimizer_updates"] == 1
    assert execution["prediction_export_rows"] == 20000
    assert execution["prediction_export_groups"] == 1
    assert execution["raw_parquet_copy_bytes"] == 0
    assert execution["raw_parquet_full_hash_bytes"] == sum(
        identity["bytes"] for identity in manifest["raw_data"]["identities"].values()
    )
    assert execution["raw_parquet_footer_hash_bytes"] == sum(
        identity["footer_bytes"] + 8 for identity in manifest["raw_data"]["identities"].values()
    )
    assert execution["parent_timeout_seconds"] == 1650
    assert execution["worker_timeout_seconds"] == 1500
    assert execution["minimum_host_free_memory_fraction"] == 0.20
    assert execution["maximum_tpu_peak_hbm_fraction"] == 0.80
    assert execution["resource_measurement_status"].startswith("unmeasured on TPU")
    collection = manifest["collection_policy"]
    assert collection["maximum_expected_download_bytes"] == 100000000
    assert collection["collect_only_after_terminal_status"] is True
    assert "parent_phase.json" in collection["required_outputs"]
    assert "device_worker_phase.json" in collection["required_outputs"]
    assert (
        "realdata_reload/updated_ema_valid_fit_group_prediction.npz"
        in (collection["required_outputs"])
    )


def test_actual_portable_registry_accepts_absent_dataset_ref() -> None:
    namespace = _probe_namespace()
    portable = _portable_fixture()
    validate = namespace["validate_portable_data_registry"]
    assert callable(validate)
    assert portable["split_manifest_sha256"] == namespace["FOLD_MANIFEST_SHA256"]
    for split in ("train", "valid"):
        registry = portable["sources"][f"datasets/{split}.parquet"]
        assert "dataset_ref" not in registry
        assert validate(portable, split, _raw_expected(namespace, split)) == registry


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("registry.sha256", "0" * 64), "portable registry differs"),
        (("registry.original_identity.bytes", 1), "portable registry differs"),
        (("registry.original_identity.footer_bytes", 1), "portable registry differs"),
        (
            ("registry.original_identity.footer_sha256", "0" * 64),
            "portable registry differs",
        ),
        (("split_manifest_sha256", "0" * 64), "split identity changed"),
    ],
)
def test_portable_registry_rejects_adversarial_mismatches(
    mutation: tuple[str, object], message: str
) -> None:
    namespace = _probe_namespace()
    portable = copy.deepcopy(_portable_fixture())
    key, value = mutation
    if key.startswith("registry."):
        target = portable["sources"]["datasets/train.parquet"]
        parts = key.removeprefix("registry.").split(".")
    else:
        target = portable
        parts = key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    validate = namespace["validate_portable_data_registry"]
    with pytest.raises(ValueError, match=message):
        validate(portable, "train", _raw_expected(namespace, "train"))


@pytest.mark.parametrize("split", ["train", "valid"])
def test_actual_main_parquet_size_and_footer_match_pinned_registry(split: str) -> None:
    namespace = _probe_namespace()
    path = MAIN_ROOT / "datasets" / f"{split}.parquet"
    expected = _raw_expected(namespace, split)
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == expected["bytes"]
    footer_identity = namespace["parquet_footer_identity"](path, expected["footer_bytes"])
    assert footer_identity == {
        "footer_bytes": expected["footer_bytes"],
        "footer_sha256": expected["footer_sha256"],
    }


def test_mounted_raw_file_requires_size_footer_and_full_hash(tmp_path: Path) -> None:
    namespace = _probe_namespace()
    path = tmp_path / "fixture.parquet"
    expected = _small_parquet(path)
    verify = namespace["verify_raw_data_file"]
    assert verify(path, "train", expected) == {
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
        "footer_bytes": expected["footer_bytes"],
        "footer_sha256": expected["footer_sha256"],
    }

    wrong_digest = dict(expected, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify(path, "train", wrong_digest)
    wrong_footer = dict(expected, footer_sha256="0" * 64)
    with pytest.raises(ValueError, match="footer SHA-256"):
        verify(path, "train", wrong_footer)
    wrong_length = dict(expected, bytes=expected["bytes"] + 1)
    with pytest.raises(ValueError, match="byte length"):
        verify(path, "train", wrong_length)
    wrong_footer_size = dict(expected, footer_bytes=expected["footer_bytes"] + 1)
    with pytest.raises(ValueError, match="footer length or magic"):
        verify(path, "train", wrong_footer_size)


def test_raw_mount_resolves_exact_slug_and_rejects_ambiguous_or_wrong_mount(
    tmp_path: Path,
) -> None:
    namespace = _probe_namespace()
    namespace["INPUT_ROOT"] = tmp_path
    namespace["RAW_DATA_IDENTITIES"] = {
        "train": {"bytes": 1},
        "valid": {"bytes": 2},
    }
    expected = tmp_path / "datasets" / "vladkochetov07" / "wunder-funder"
    expected.mkdir(parents=True)
    (expected / "train.parquet").write_bytes(b"x")
    (expected / "valid.parquet").write_bytes(b"yy")
    discover = namespace["_discover_raw_data_mount"]
    assert discover() == expected

    wrong_slug = tmp_path / "datasets" / "vladkochetov07" / "other-dataset"
    wrong_slug.mkdir(parents=True)
    (wrong_slug / "train.parquet").write_bytes(b"x")
    (wrong_slug / "valid.parquet").write_bytes(b"yy")
    with pytest.raises(FileNotFoundError, match="pinned Kaggle dataset slug"):
        discover()


def test_prediction_export_contract_follows_checkpoint_reload() -> None:
    source = (PACKET / "probe.py").read_text(encoding="utf-8")
    assert "realdata_checkpoint_save_reload" in source
    assert "realdata_prediction_export" in source
    assert 'prediction_exported": True' in source
    assert "updated_ema_valid_fit_group_prediction.npz" in source
    assert "np.savez_compressed(" in source
    assert "sequence_input_sha256" in source
    assert '"rows": SEQUENCE_ROWS' in source
    assert (
        '"state_policy": "reset at sequence start; carry state across 2,000-row chunks"' in source
    )
    assert '"score": None' in source
    assert 'registry.get("dataset_ref")' not in source


def test_staging_cleanup_removes_only_symlinks_and_preserves_source(tmp_path: Path) -> None:
    namespace = _probe_namespace()
    namespace["OUTPUT"] = tmp_path / "output"
    output = namespace["OUTPUT"]
    output.mkdir()
    source = tmp_path / "source.parquet"
    source.write_bytes(b"immutable")
    stage = output / "realdata_project"
    stage.mkdir()
    (stage / "train.parquet").symlink_to(source)
    cleanup = namespace["cleanup_realdata_stage"]
    cleanup()
    assert not stage.exists()
    assert source.read_bytes() == b"immutable"
    receipt = json.loads((output / "realdata_staging_cleanup.json").read_text())
    assert receipt["mounted_source_payloads_modified"] is False
    assert receipt["staged_payloads_copied"] is False


def test_staging_cleanup_refuses_unexpected_regular_payload(tmp_path: Path) -> None:
    namespace = _probe_namespace()
    namespace["OUTPUT"] = tmp_path / "output"
    output = namespace["OUTPUT"]
    output.mkdir()
    stage = output / "realdata_project"
    stage.mkdir()
    payload = stage / "unexpected-copy.parquet"
    payload.write_bytes(b"do not remove")
    cleanup = namespace["cleanup_realdata_stage"]
    with pytest.raises(ValueError, match="non-symlink staged payload"):
        cleanup()
    assert payload.read_bytes() == b"do not remove"
