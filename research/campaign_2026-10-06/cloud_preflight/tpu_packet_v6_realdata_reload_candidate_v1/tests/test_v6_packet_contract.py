from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
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
        "__name__": "e463_v6_contract_probe",
        "__file__": str(PROBE_PATH),
    }
    exec(compile(source, str(PROBE_PATH), "exec"), namespace)
    return namespace


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, function) or ""


def test_packet_hashes_and_archive_bind_exact_provider_files() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    code = PROBE_PATH.read_bytes()
    metadata = (PACKET / "kernel-metadata.json").read_bytes()
    assert manifest["code_sha256"] == _sha256(code)
    assert manifest["code_bytes"] == len(code)
    assert manifest["metadata_sha256"] == _sha256(metadata)
    assert manifest["bundle_sha256"] == _sha256((ROOT / "packet.bundle.zip").read_bytes())
    assert manifest["builder_sha256"] == _sha256(
        (ROOT / "reference/build_v6_probe.py").read_bytes()
    )
    assert manifest["test_sha256"] == _sha256(Path(__file__).read_bytes())
    assert manifest["recipe_audit_sha256"] == _sha256((ROOT / "V6_RECIPE_AUDIT.md").read_bytes())
    assert manifest["readme_sha256"] == _sha256((ROOT / "README.md").read_bytes())
    assert manifest["source_only"] is True
    assert manifest["provider_push_authorized"] is False
    assert len(code) < 1_000_000

    metadata_json = json.loads(metadata)
    assert metadata_json["id"] == manifest["packet_id"]
    assert metadata_json["title"] == manifest["packet_title"]
    assert metadata_json["machine_shape"] == "TpuV5E8"
    assert metadata_json["enable_gpu"] is False
    assert metadata_json["enable_internet"] is False
    assert set(metadata_json["dataset_sources"]) == set(manifest["requested_inputs"])

    with zipfile.ZipFile(ROOT / "packet.bundle.zip") as archive:
        assert set(archive.namelist()) == {"kernel-metadata.json", "probe.py"}
        assert archive.read("probe.py") == code
        assert archive.read("kernel-metadata.json") == metadata


def test_source_checkpoint_and_data_identities_are_frozen() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    namespace = _load_probe_namespace()
    assert namespace["SOURCE_MANIFEST_SHA256"] == manifest["source_manifest_sha256"]
    assert namespace["SOURCE_ARCHIVE_SHA256"] == manifest["source_archive_sha256"]
    assert namespace["FOLD_MANIFEST_SHA256"] == manifest["fold_manifest_sha256"]
    assert namespace["ASSET_MANIFEST_SHA256"] == manifest["assets_manifest_sha256"]
    assert namespace["PORTABLE_INPUTS_SHA256"] == manifest["portable_inputs_sha256"]
    assert namespace["EXPECTED_CHECKPOINT_SHA256"] == manifest["checkpoint_sha256"]
    assert namespace["EXPECTED_NPZ_SHA256"] == manifest["checkpoint_npz_sha256"]
    assert namespace["EXPECTED_SIDECAR_SHA256"] == manifest["checkpoint_sidecar_sha256"]
    assert namespace["EXPECTED_MODEL_SOURCE_SHA256"] == manifest["model_source_sha256"]
    assert namespace["PARENT_PROVENANCE_SHA256"] == manifest["parent_provenance_sha256"]

    provenance_bytes = base64.b64decode(namespace["PARENT_PROVENANCE_B64"], validate=True)
    assert _sha256(provenance_bytes) == manifest["parent_provenance_sha256"]
    provenance = json.loads(provenance_bytes)
    assert provenance["selected_epoch"] == 3
    assert provenance["checkpoint_sha256"]["model_ep3.pt"] == manifest["checkpoint_sha256"]
    assert len(provenance["fit_row_groups"]["train"]) == 10_535
    assert len(provenance["fit_row_groups"]["valid"]) == 1_405
    assert len(provenance["evaluation_row_groups"]) == 468

    raw = manifest["raw_parquet"]
    assert raw["train"]["bytes"] == 29_240_219_011
    assert raw["valid"]["bytes"] == 5_615_509_128
    assert len(raw["train"]["sha256"]) == 64
    assert len(raw["valid"]["sha256"]) == 64


def test_e465_support_files_are_embedded_and_identity_bound() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    namespace = _load_probe_namespace()
    encoded_modules = namespace["E465_SUPPORT_B64"]
    expected_hashes = namespace["E465_SUPPORT_SHA256"]
    assert set(encoded_modules) == {
        "distributed_combo.py",
        "e465_host_contract.py",
        "onecycle.py",
    }
    for name, encoded in encoded_modules.items():
        content = base64.b64decode(encoded, validate=True)
        assert _sha256(content) == expected_hashes[name]
        assert {
            "bytes": len(content),
            "sha256": _sha256(content),
        } == manifest["helper_modules"][name]
        assert content == (ROOT / "reference" / name).read_bytes()


def test_selected_membership_is_source_qualified_fit_valid_only() -> None:
    namespace = _load_probe_namespace()
    validate = namespace["validate_selected_source_membership"]
    valid_fit = tuple(range(1_405))
    evaluation = tuple(range(1_405, 1_873))
    selected = tuple(("valid", group) for group in range(16))
    result = validate(selected, valid_fit, evaluation, valid_fit)
    assert result["status"] == "passed"
    assert result["qualified_membership"] == [list(pair) for pair in selected]
    assert result["heldout_claim"] is None

    with pytest.raises(ValueError, match="unique source-qualified"):
        validate(selected[:-1] + (selected[0],), valid_fit, evaluation, valid_fit)
    with pytest.raises(ValueError, match="fit-VALID"):
        validate(tuple(("train", group) for group in range(16)), valid_fit, evaluation, valid_fit)

    eval_overlap = (0, *evaluation[1:])
    with pytest.raises(ValueError, match="evaluation membership"):
        validate(selected, valid_fit, eval_overlap, valid_fit)
    with pytest.raises(ValueError, match="source checkpoint fit membership"):
        validate(selected, valid_fit, evaluation, tuple(range(15)))


def test_workload_and_result_keep_probe_scope_explicit() -> None:
    namespace = _load_probe_namespace()
    assert namespace["GLOBAL_BATCH"] == 16
    assert namespace["LOCAL_BATCH"] == 2
    assert namespace["CHUNK_ROWS"] == 2_000
    assert namespace["SEQUENCE_ROWS"] == 20_000
    assert namespace["TOTAL_STEPS"] == 7_460
    assert namespace["DIMENSIONS"] == (211, 256, 2, 23)
    assert namespace["PARENT_TIMEOUT_SECONDS"] == 1_650
    assert namespace["WORKER_TIMEOUT_SECONDS"] == 1_500

    source = PROBE_PATH.read_text(encoding="utf-8")
    realdata_update = _function_source(source, "run_realdata_one_update")
    worker = _function_source(source, "run_device_worker")
    parent = _function_source(source, "parent_main")
    assert source.count("def discover_source_mount()") == 1
    assert "run_realdata_one_update(jax, jnp, np, params, local_devices, source_root)" in worker
    assert worker.index('"tpu_fullshape_profile.json"') < worker.index("run_realdata_one_update(")
    assert worker.index("run_realdata_one_update(") < worker.index('"hbm_headroom_validation"')
    assert "peak > int(capacity * 0.80)" in worker
    assert "limit - current < int(limit * 0.20)" in worker
    assert '"no_training_data_used": False' in worker
    assert '"no_checkpoint_saved_or_exported": False' in worker
    assert '"heldout_claim": None' in source
    assert '"metric_or_score": None' in source
    assert '"public_forecast": None' in source
    assert "scoring_enabled" not in source
    assert "bridge.seeded_group_batches(" in realdata_update
    assert "seed=3, epoch=0, batch_size=GLOBAL_BATCH" in realdata_update
    assert "value[:, :CHUNK_ROWS]" in realdata_update
    assert "onecycle_point(0, TOTAL_STEPS)" in realdata_update
    assert "pmean_and_clip_gradient" in realdata_update
    assert "max_norm=1.0" in realdata_update
    assert "forward_chunk(saved_ema" in realdata_update
    assert "forward_chunk(reloaded_ema" in realdata_update
    assert '"prediction_exported": False' in realdata_update
    assert parent.index("SOURCE_MOUNT_ROOT = discover_source_mount()") < parent.index(
        "verify_direct_source_tree()"
    )
    assert parent.index("materialize_realdata_project(") < parent.index("subprocess.Popen(")
    assert '"WNN_E463_DATA_ROOT"' in parent


def test_probe_module_import_is_offline_and_lightweight() -> None:
    before = set(sys.modules)
    namespace = _load_probe_namespace()
    after = set(sys.modules)
    newly_loaded = after - before
    assert not any(name == "jax" or name.startswith("jax.") for name in newly_loaded)
    assert not any(name == "pyarrow" or name.startswith("pyarrow.") for name in newly_loaded)
    assert not any(name == "torch" or name.startswith("torch.") for name in newly_loaded)
    assert namespace["PACKET_ID"] == "e463-tpu-realdata-reload-v6-20261008"
