"""Standard-library admission checks for the E463 V6 and E465 profile handoff."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import cast

V6_KERNEL = "vladkochetov07/e463-tpu-real-data-reload-v6-20261008"
V6_SUBMITTED_KERNEL = "vladkochetov07/e463-tpu-realdata-reload-v6-20261008"
V6_PACKET_ID = "e463-tpu-realdata-reload-v6-20261008"
V6_AUTHORIZATION_SHA256 = "4aa5422e39adf313245f46b4954b813b4ccfcb1665c8df1bbec578c5c65316c7"
V6_PACKET_PATH = "research/campaign_2026-10-06/cloud_preflight/tpu_packet_v6_realdata_reload_candidate_v1"
V6_METADATA_SHA256 = "5d4d5d9606c71302ca1985bbc95fb70220a68821c5b127110e8380d8d4704e92"
V6_MANIFEST_SHA256 = "a609e728b212065a53459b1c2630bced018cb80614d7eea55ebc5ecce3401448"
V6_CODE_SHA256 = "3392ed77b9000c7bab9b236531c9a16734325152b95a60c955408df1b9bb02f2"
V6_BUNDLE_SHA256 = "3714c638b4548cb9b13826e0f15fbe9a98cfb21314b647bcf83e772d7e512325"
V6_SOURCE_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
V6_SOURCE_ARCHIVE_SHA256 = "a29bdf21a910a6d7b403d9ed8c52cf0761618f5ef6576b5f06d70d0a8fd5ec9e"
V6_ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
V6_PORTABLE_INPUTS_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
V6_FOLD_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PARENT_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
PARENT_NPZ_SHA256 = "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a"
PARENT_SIDECAR_SHA256 = "3e65d2a88352d22bc90d88b829f05beee3faa7a66b9de41c7507f521aab12668"
PARENT_PROVENANCE_SHA256 = "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f"
MODEL_SOURCE_SHA256 = "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc"
E465_MAX_WORKER_SECONDS = 8 * 60 * 60
E465_TPU_DEVICE_COUNT = 8
E465_PROFILE_BATCHES = 2
E465_GLOBAL_BATCH_SIZE = 32
E465_SEQUENCE_ROWS = 20_000
E465_CHUNK_ROWS = 1_000
E465_CHUNKS_PER_BATCH = E465_SEQUENCE_ROWS // E465_CHUNK_ROWS
E465_MAX_HOST_RSS_BYTES = 12 * 1024**3
E465_DEVICE_HARD_STOP_FRACTION = 0.90
RAW_DATA = {
    "train": (29_240_219_011, "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81"),
    "valid": (5_615_509_128, "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"),
}
REQUIRED_V6_ASSETS = (
    "artifacts/runs/c20260929_prop_f0/model.pt",
    "artifacts/runs/c20260929_since_s3_f0/model_ep3.pt",
    "artifacts/runs/c20260929_since_s4_f0/model_ep3.pt",
    "cache/pred_c20260929_since_s3_f0_valid.json",
    "cache/pred_c20260929_since_s3_f0_valid.npy",
    "cache/pred_c20260929_since_s4_f0_valid.json",
    "cache/pred_c20260929_since_s4_f0_valid.npy",
    "cache/propcache_c20260929_prop_f0_valid.json",
    "cache/propcache_c20260929_prop_f0_valid.npy",
    "cache/valid_targets.npz",
    "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    "research/campaign_2026-10-04/portable_inputs.json",
)
REQUIRED_V6_OUTPUTS = (
    "terminal.json",
    "complete.json",
    "startup.json",
    "device_worker_start.json",
    "device_worker_terminal.json",
    "device_worker_result.json",
    "result.json",
    "tpu_diagnostic_complete.json",
    "source_materialization.json",
    "e463_checkpoint_identity.json",
    "realdata_input_identity.json",
    "realdata_update_receipt.json",
    "tpu_inventory.json",
    "tpu_hbm_before.json",
    "tpu_hbm_after.json",
    "tpu_objective_reduction_parity.json",
    "tpu_fullshape_profile.json",
    "model_after_one_chunk_fp32.npz",
    "ema_after_one_chunk_fp32.npz",
    "adamw_after_one_chunk.npz",
)
SOURCE_HASH_PATHS = (
    "scripts/run_e465_tpu.py",
    "src/wnn_jax/e465_tpu_worker.py",
    "src/wnn_jax/e465_gate.py",
    "src/wnn_jax/bridge.py",
    "src/wnn_jax/distributed_combo.py",
    "src/wnn_jax/e465_host_contract.py",
    "src/wnn_jax/onecycle.py",
    "src/wnn_jax/rnn.py",
)


def sha256_file(path: Path) -> str:
    """Hash one file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _read_json(path: Path, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is absent or not a regular file: {path}")
    return _object(json.loads(path.read_text(encoding="utf-8")), name)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} is not a SHA-256 string")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} is not a lowercase SHA-256 string")
    return value


def _output_inventory(bundle_root: Path, receipt: dict[str, object]) -> dict[str, Path]:
    rows = receipt.get("inventory")
    if not isinstance(rows, list) or not rows:
        raise ValueError("V6 terminal receipt has no complete output inventory")
    by_name: dict[str, dict[str, object]] = {}
    for row_obj in rows:
        row = _object(row_obj, "terminal inventory row")
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("terminal inventory path must be a string")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("terminal inventory contains an unsafe path")
        name = relative_path.name
        if name in by_name:
            raise ValueError(f"ambiguous duplicate output basename in receipt: {name}")
        if type(row.get("bytes")) is not int or row["bytes"] < 0:
            raise ValueError(f"invalid output byte count: {name}")
        _sha(row.get("sha256"), f"terminal inventory {name} sha256")
        by_name[name] = row

    output_root = bundle_root / "outputs"
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("V6 output bundle lacks a regular outputs directory")
    observed: dict[str, Path] = {}
    total_bytes = 0
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"V6 output bundle contains a symlink: {path}")
        if not path.is_file():
            continue
        name = path.name
        if name in observed:
            raise ValueError(f"ambiguous duplicate staged output basename: {name}")
        if name not in by_name:
            raise ValueError(f"staged V6 output is absent from terminal inventory: {name}")
        expected = by_name[name]
        size = path.stat().st_size
        if size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"staged V6 output identity differs from terminal receipt: {name}")
        observed[name] = path
        total_bytes += size
    if total_bytes > 200_000_000:
        raise ValueError("V6 receipt bundle exceeds the bounded 200 MB metadata/checkpoint limit")
    missing = set(REQUIRED_V6_OUTPUTS) - set(observed)
    if missing:
        raise ValueError(f"V6 terminal output bundle is incomplete: {sorted(missing)}")
    uncollected = set(by_name) - set(observed)
    if uncollected:
        raise ValueError(f"V6 terminal output inventory was only partially collected: {sorted(uncollected)}")
    return observed


def _validate_delivery(path: Path) -> dict[str, object]:
    receipt = _read_json(path, "V6 launch receipt")
    if receipt.get("schema") != "wnn.e463-tpu-v6-operator-delivery.v1":
        raise ValueError("unexpected V6 launch receipt schema")
    authorization = _object(receipt.get("authorization"), "V6 root authorization")
    packet = _object(receipt.get("packet"), "V6 launch packet")
    push = _object(receipt.get("push"), "V6 launch push")
    target = _object(receipt.get("provider_target"), "V6 provider target")
    readiness = _object(receipt.get("input_readiness"), "V6 launch input readiness")
    asset_dataset = _object(readiness.get("fold_assets_dataset"), "V6 fold-assets dataset")
    asset_manifest = _object(asset_dataset.get("manifest"), "V6 fold-assets manifest")
    raw_dataset = _object(readiness.get("raw_dataset"), "V6 raw dataset")
    raw_hashes = _object(raw_dataset.get("pinned_parquet_sha256"), "V6 pinned parquet hashes")
    source_dataset = _object(readiness.get("source_dataset"), "V6 source dataset")
    source_manifest = _object(source_dataset.get("manifest"), "V6 source manifest")
    observations = _object(receipt.get("observations"), "V6 launch observations")
    postlaunch_quota = _object(receipt.get("postlaunch_quota"), "V6 postlaunch quota")
    canonical_status = _object(postlaunch_quota.get("canonical_status"), "V6 launch status")
    recorded_registry_sha = _sha(target.get("registry_sha256_recorded"), "V6 registry SHA-256")
    if (
        receipt.get("record_id") != "E463_TPU_V6_OPERATOR_DELIVERY_20261008_R1"
        or authorization.get("path") != "E463_TPU_V6_ROOT_ONE_PUSH_AUTHORIZATION_20261008_R1.json"
        or authorization.get("sha256") != V6_AUTHORIZATION_SHA256
        or authorization.get("status") != "approved_conditional_one_push"
        or packet.get("code_sha256") != V6_CODE_SHA256
        or packet.get("bundle_sha256") != V6_BUNDLE_SHA256
        or packet.get("manifest_sha256") != V6_MANIFEST_SHA256
        or packet.get("metadata_sha256") != V6_METADATA_SHA256
        or packet.get("path") != V6_PACKET_PATH
        or packet.get("source_commit") != "22aff7b3e733a20ec42637f76110bbd248d84452"
        or push.get("accepted") is not True
        or push.get("exit_code") != 0
        or push.get("duplicate_pushes") != 0
        or push.get("version_number") != 1
        or push.get("submitted_kernel_id") != V6_SUBMITTED_KERNEL
        or push.get("provider_canonical_kernel_id") != V6_KERNEL
        or push.get("provider_returned_url") != f"https://www.kaggle.com/code/{V6_KERNEL}"
        or push.get("scope") != "One real-data B16/C2000 compatibility update with checkpoint/reload and causal/reset checks; no production fit or score."
        or target.get("canonical_kernel_id") != V6_KERNEL
        or target.get("registry_path") != "provider_targets.json"
        or target.get("terminal_receipt") != "E463_TPU_V6_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json"
        or target.get("user_timer_active") is not True
        or asset_dataset.get("id") != "vladkochetov07/wnn-c20261004-fold0-assets"
        or asset_manifest.get("sha256") != V6_ASSET_MANIFEST_SHA256
        or raw_dataset.get("id") != "vladkochetov07/wunder-funder"
        or raw_hashes.get("train") != RAW_DATA["train"][1]
        or raw_hashes.get("valid") != RAW_DATA["valid"][1]
        or source_dataset.get("id") != "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1"
        or source_manifest.get("sha256") != V6_SOURCE_MANIFEST_SHA256
        or observations.get("latest_verified_provider_state") != canonical_status.get("status")
        or observations.get("observed_at_utc") != postlaunch_quota.get("observed_at_utc")
        or canonical_status.get("kernel") != V6_KERNEL
        or canonical_status.get("resource") != "tpu"
        or recorded_registry_sha != target.get("registry_sha256_recorded")
    ):
        raise ValueError("V6 launch receipt does not bind the reviewed single accepted packet")
    return receipt


def _validate_manifest_identity(
    input_manifest_path: Path,
    project_root: Path,
    realdata_input: dict[str, object],
    checkpoint_identity: dict[str, object],
    update: dict[str, object],
) -> dict[str, object]:
    manifest = _read_json(input_manifest_path, "E465 input manifest")
    if manifest.get("format") != "wnn.e465-tpu-input-manifest" or manifest.get("version") != 1:
        raise ValueError("E465 input manifest format/version differs")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("E465 input manifest has no immutable file table")
    files: dict[str, str] = {}
    for row_obj in rows:
        row = _object(row_obj, "E465 input file row")
        relative = row.get("path")
        digest = _sha(row.get("sha256"), "E465 input file sha256")
        if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError("unsafe E465 input path")
        if relative in files:
            raise ValueError(f"duplicate E465 input path: {relative}")
        files[relative] = digest

    fold_path = manifest.get("fold_manifest")
    parent = _object(manifest.get("parent_export"), "E465 parent export")
    if (
        not isinstance(fold_path, str)
        or manifest.get("fold_manifest_sha256") != V6_FOLD_SHA256
        or files.get(fold_path) != V6_FOLD_SHA256
        or parent.get("run") != "c20261003_combo_s3_f0"
        or parent.get("epoch") != 3
        or parent.get("weight_source") != "evaluation_ema"
        or parent.get("npz_sha256") != PARENT_NPZ_SHA256
        or parent.get("sidecar_sha256") != PARENT_SIDECAR_SHA256
        or parent.get("source_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent.get("source_provenance_sha256") != PARENT_PROVENANCE_SHA256
    ):
        raise ValueError("E465 fold or parent lineage differs from the reviewed E463 V6 input")
    for path_key, sha_key in (
        ("npz_path", "npz_sha256"),
        ("sidecar_path", "sidecar_sha256"),
        ("source_checkpoint_path", "source_checkpoint_sha256"),
        ("source_provenance_path", "source_provenance_sha256"),
    ):
        relative = parent.get(path_key)
        if not isinstance(relative, str) or files.get(relative) != parent.get(sha_key):
            raise ValueError(f"E465 parent {path_key} is not bound by its file table")
    fold_file = (project_root / fold_path).resolve(strict=True)
    if not fold_file.is_relative_to(project_root.resolve(strict=True)) or sha256_file(fold_file) != V6_FOLD_SHA256:
        raise ValueError("E465 fold manifest bytes differ from the pinned V6 fold")
    fold = _read_json(fold_file, "fold manifest")
    fold_fit = _object(fold.get("fit"), "fold fit groups")
    fold_eval = _object(fold.get("evaluation"), "fold evaluation groups")
    fit_valid = fold_fit.get("valid")
    evaluation = fold_eval.get("valid")
    membership = _object(update.get("source_qualified_membership"), "V6 source membership")
    selected = membership.get("row_groups")
    if (
        not isinstance(fit_valid, list)
        or len(fit_valid) != 1_405
        or not isinstance(fold_fit.get("train"), list)
        or len(fold_fit["train"]) != 10_535
        or not isinstance(evaluation, list)
        or len(evaluation) != 468
        or not isinstance(selected, list)
        or membership.get("status") != "passed"
        or membership.get("split") != "valid"
        or membership.get("batch_size") != 16
        or len(selected) != 16
        or any(type(group) is not int for group in selected)
        or len(set(selected)) != 16
        or membership.get("qualified_membership") != [["valid", group] for group in selected]
        or membership.get("evaluation_overlap_count") != 0
        or membership.get("outside_parent_fit_count") != 0
        or not set(selected).issubset(set(fit_valid))
        or set(selected) & set(evaluation)
    ):
        raise ValueError("V6 real-data batch is not 16 source-qualified fit-VALID groups")

    raw = _object(realdata_input.get("raw_data"), "V6 raw data identity")
    for split, (size, digest) in RAW_DATA.items():
        identity = _object(raw.get(split), f"V6 raw {split} identity")
        if (
            identity.get("dataset_ref") != "vladkochetov07/wunder-funder"
            or identity.get("bytes") != size
            or identity.get("sha256") != digest
            or files.get(f"datasets/{split}.parquet") != digest
        ):
            raise ValueError(f"V6 and E465 raw {split} parquet identities differ")
    assets = _object(realdata_input.get("verified_asset_files"), "V6 verified assets")
    portable = assets.get("research/campaign_2026-10-04/portable_inputs.json")
    if set(assets) != set(REQUIRED_V6_ASSETS) or portable is None:
        raise ValueError("V6 verified fold-assets differ from the required 12-file input closure")
    for relative, identity_obj in assets.items():
        identity = _object(identity_obj, f"V6 verified asset {relative}")
        digest = _sha(identity.get("sha256"), f"V6 verified asset {relative} sha256")
        if digest != files.get(relative) or type(identity.get("bytes")) is not int:
            raise ValueError(f"V6 and E465 fold asset identity differs: {relative}")
    if _object(portable, "V6 portable input registry").get("sha256") != V6_PORTABLE_INPUTS_SHA256:
        raise ValueError("V6 portable-input registry differs from the reviewed fold-assets bundle")
    if realdata_input.get("fold_assets_manifest_sha256") != V6_ASSET_MANIFEST_SHA256:
        raise ValueError("fold-asset manifest version differs from reviewed V6 inputs")
    if realdata_input.get("source_manifest_sha256") != V6_SOURCE_MANIFEST_SHA256 or realdata_input.get("source_archive_sha256") != V6_SOURCE_ARCHIVE_SHA256:
        raise ValueError("V6 source dataset identity differs from the reviewed source packet")
    if checkpoint_identity.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256:
        raise ValueError("V6 checkpoint identity differs from the E465 parent")
    if checkpoint_identity.get("npz_sha256") != PARENT_NPZ_SHA256 or checkpoint_identity.get("sidecar_sha256") != PARENT_SIDECAR_SHA256:
        raise ValueError("V6 NPZ/sidecar identity differs from the E465 parent")
    if checkpoint_identity.get("model_source_sha256") != MODEL_SOURCE_SHA256:
        raise ValueError("V6 recurrent model implementation differs from the reviewed source")
    if checkpoint_identity.get("dimensions") != {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23}:
        raise ValueError("V6 parent model dimensions differ")
    if realdata_input.get("source_file_count") != 17 or realdata_input.get("status") != "verified_and_staged_read_only_inputs":
        raise ValueError("V6 did not verify and stage the exact source closure")
    return {
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "fold_manifest_sha256": V6_FOLD_SHA256,
        "fit_valid_group_count": len(fit_valid),
        "evaluation_group_count": len(evaluation),
        "v6_realdata_selected_groups": selected,
        "raw_data_sha256": {split: digest for split, (_, digest) in RAW_DATA.items()},
        "verified_asset_count": len(assets),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_npz_sha256": PARENT_NPZ_SHA256,
    }


def validate_e463_v6_bundle(
    bundle_root: Path, input_manifest_path: Path, project_root: Path
) -> dict[str, object]:
    """Require terminal V6 proof, complete source/output identity, and matching E465 inputs."""
    _validate_delivery(bundle_root / "delivery_receipt.json")
    terminal_receipt_path = bundle_root / "terminal_receipt.json"
    terminal_receipt = _read_json(terminal_receipt_path, "V6 terminal receipt")
    if (
        terminal_receipt.get("record_id") != "E463_TPU_V6_TERMINAL_OPERATOR_RECEIPT_20261008_R1"
        or terminal_receipt.get("canonical_kernel") != V6_KERNEL
        or terminal_receipt.get("provider_status") != "COMPLETE"
        or terminal_receipt.get("download_exit_code") != 0
        or terminal_receipt.get("result_status") != "passed"
        or terminal_receipt.get("actual_devices") != 8
        or "TPU v5 lite" not in str(terminal_receipt.get("device_kind", ""))
    ):
        raise ValueError("V6 terminal receipt does not attest completed success on eight TPU v5e devices")
    outputs = _output_inventory(bundle_root, terminal_receipt)
    parsed: dict[str, dict[str, object] | list[object]] = {}
    for name in REQUIRED_V6_OUTPUTS:
        if not name.endswith(".json"):
            continue
        if name in {"tpu_hbm_before.json", "tpu_hbm_after.json"}:
            parsed[name] = json.loads(outputs[name].read_text(encoding="utf-8"))
        else:
            parsed[name] = _read_json(outputs[name], name)

    terminal = parsed["terminal.json"]
    complete = parsed["complete.json"]
    startup = parsed["startup.json"]
    worker_start = parsed["device_worker_start.json"]
    worker_terminal = parsed["device_worker_terminal.json"]
    result = parsed["result.json"]
    worker_result = parsed["device_worker_result.json"]
    diagnostic = parsed["tpu_diagnostic_complete.json"]
    if (
        terminal.get("status") != "passed"
        or terminal.get("phase") != "complete"
        or terminal.get("worker_returncode") != 0
        or complete.get("status") != "passed"
        or complete.get("phase") != "complete"
        or startup.get("packet") != V6_PACKET_ID
        or worker_start.get("packet") != V6_PACKET_ID
        or worker_terminal.get("status") != "passed"
        or worker_terminal.get("phase") != "complete"
        or worker_result != result
        or diagnostic != result
        or result.get("status") != "passed"
        or result.get("backend") != "tpu"
        or result.get("device_count") != 8
        or result.get("scientific_fit_or_score") is not None
    ):
        raise ValueError("V6 raw terminal/output records do not attest a successful diagnostic only")

    source_materialization = parsed["source_materialization.json"]
    checkpoint_identity = parsed["e463_checkpoint_identity.json"]
    realdata_input = parsed["realdata_input_identity.json"]
    update = parsed["realdata_update_receipt.json"]
    _validate_update_checkpoint_artifacts(update, outputs)
    if (
        source_materialization.get("status") != "verified_direct_read_only_tree"
        or source_materialization.get("source_manifest_sha256") != V6_SOURCE_MANIFEST_SHA256
        or source_materialization.get("declared_source_archive_sha256") != V6_SOURCE_ARCHIVE_SHA256
        or source_materialization.get("verified_file_count") != 17
        or realdata_input.get("fold_assets_manifest_sha256") != V6_ASSET_MANIFEST_SHA256
        or realdata_input.get("status") != "verified_and_staged_read_only_inputs"
        or realdata_input.get("source_file_count") != 17
    ):
        raise ValueError("V6 source and real-data materialization receipts differ from reviewed bytes")
    if (
        update.get("status") != "passed"
        or update.get("classification") != "one real-data B16/C2000 distributed E463 checkpoint/reload compatibility update"
        or update.get("fold_manifest_sha256") != V6_FOLD_SHA256
        or update.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or update.get("parent_npz_sha256") != PARENT_NPZ_SHA256
        or update.get("parent_provenance_sha256") != PARENT_PROVENANCE_SHA256
        or update.get("global_batch") != 16
        or update.get("local_batch_per_tpu") != 2
        or update.get("chunk_rows") != 2000
        or update.get("input_width") != 211
        or update.get("hidden_width") != 256
        or update.get("layers") != 2
        or update.get("output_width") != 23
        or update.get("completed_update_count_per_replica") != [1] * 8
        or update.get("training_data_used") is not True
        or update.get("fit_valid_is_already_exposed") is not True
        or update.get("heldout_claim") is not None
        or update.get("metric_or_score") is not None
        or update.get("public_forecast") is not None
    ):
        raise ValueError("V6 real-data update does not match the fixed B16/C2000 no-score contract")
    checkpoint = _object(update.get("checkpoint"), "V6 checkpoint round-trip")
    causality = _object(update.get("inference_causality"), "V6 inference causality")
    if any(
        checkpoint.get(key) != value
        for key, value in {
            "save_reload_parameter_identity": "bitwise_equal_all_tensors",
            "save_reload_ema_identity": "bitwise_equal_all_tensors",
            "save_reload_optimizer_identity": "bitwise_equal_all_moments_and_count",
        }.items()
    ) or any(
        causality.get(key) != value
        for key, value in {
            "status": "passed",
            "checkpoint_reload_prediction_identity": "bitwise_equal",
            "future_perturbation_changed_prefix": False,
            "omitted_state_matches_explicit_zero_state": True,
            "prediction_exported": False,
        }.items()
    ):
        raise ValueError("V6 checkpoint reload or causal/reset evidence is incomplete")

    inventory = parsed["tpu_inventory.json"]
    devices = inventory.get("devices")
    if (
        inventory.get("jax_backend") != "tpu"
        or inventory.get("local_device_count") != 8
        or inventory.get("global_device_count") != 8
        or not isinstance(devices, list)
        or len(devices) != 8
    ):
        raise ValueError("V6 did not inventory the exact eight-chip TPU v5e system")
    for row_obj in devices:
        row = _object(row_obj, "V6 TPU inventory device")
        if row.get("platform") != "tpu" or "TPU v5 lite" not in str(row.get("device_kind", "")):
            raise ValueError("V6 TPU inventory contains a device other than TPU v5 lite")
    before = parsed["tpu_hbm_before.json"]
    after = parsed["tpu_hbm_after.json"]
    before_rows = before if isinstance(before, list) else before.get("devices")
    after_rows = after if isinstance(after, list) else after.get("devices")
    if not isinstance(before_rows, list) or len(before_rows) != 8 or not isinstance(after_rows, list) or len(after_rows) != 8:
        raise ValueError("V6 HBM evidence does not cover all eight TPU devices")
    for row_obj in before_rows:
        row = _object(row_obj, "V6 pre-update HBM row")
        current, capacity = row.get("hbm_current_bytes"), row.get("hbm_capacity_bytes")
        if type(current) is not int or type(capacity) is not int or capacity <= 0 or capacity - current < capacity * 0.20:
            raise ValueError("V6 TPU HBM pre-update free-space bound failed")
    for row_obj in after_rows:
        row = _object(row_obj, "V6 post-update HBM row")
        peak, capacity = row.get("hbm_peak_bytes"), row.get("hbm_capacity_bytes")
        if type(peak) is not int or type(capacity) is not int or capacity <= 0 or peak > capacity * 0.80:
            raise ValueError("V6 TPU HBM post-update peak bound failed")
    parity = parsed["tpu_objective_reduction_parity.json"]
    profile = parsed["tpu_fullshape_profile.json"]
    if parity.get("status") != "passed" or profile.get("status") != "passed" or profile.get("synthetic_inputs_and_targets") is not True or profile.get("scientific_score_claim") is not None:
        raise ValueError("V6 synthetic distributed correctness evidence is incomplete")

    input_identity = _validate_manifest_identity(
        input_manifest_path,
        project_root,
        realdata_input,
        checkpoint_identity,
        update,
    )
    if result.get("realdata_update") != update:
        raise ValueError("V6 top-level result does not bind its real-data update receipt")
    return {
        "schema": "wnn.e465-e463-v6-receipt-gate.v1",
        "status": "passed",
        "provider_kernel": V6_KERNEL,
        "provider_version": 1,
        "v6_delivery_receipt_sha256": sha256_file(bundle_root / "delivery_receipt.json"),
        "v6_terminal_receipt_sha256": sha256_file(terminal_receipt_path),
        "v6_output_inventory_count": len(outputs),
        "v6_input_identity": input_identity,
        "v6_realdata_update_sha256": sha256_file(outputs["realdata_update_receipt.json"]),
        "v6_checkpoint_files": {
            name: sha256_file(outputs[name])
            for name in REQUIRED_V6_OUTPUTS
            if name.endswith(".npz")
        },
        "precision_scope": "V6 admits only the FP32 real-data one-update reload path; BF16 still requires E465 actual-data profile and root review.",
        "heldout_claim": None,
        "metric_or_score": None,
    }


def source_hashes(stage_root: Path) -> dict[str, str]:
    """Hash the exact runner and imported source files used by the E465 attempt."""
    result = {}
    for relative in SOURCE_HASH_PATHS:
        path = stage_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"E465 source file is absent or unsafe: {relative}")
        result[relative] = sha256_file(path)
    return result


def validate_actual_precision_gate(report: dict[str, object]) -> dict[str, object]:
    """Require measured finite FP32 and BF16 updates on identical actual full-shape batches."""
    gate = _object(report.get("actual_precision_gate"), "actual precision gate")
    if (
        gate.get("schema") != "wnn.e465-actual-data-precision-gate.v1"
        or gate.get("status") != "passed"
        or gate.get("profile_batches_per_precision") != 2
        or gate.get("global_batch_size") != 32
        or gate.get("sequence_rows") != 20_000
        or gate.get("chunk_rows") != 1_000
        or gate.get("chunks_per_batch") != 20
        or gate.get("numerical_review_required_before_fit") is not True
    ):
        raise ValueError("profile lacks the reviewed actual-data precision gate")
    arms = _object(gate.get("arms"), "actual precision arms")
    if set(arms) != {"fp32", "bf16"}:
        raise ValueError("actual-data gate must check FP32 and BF16 independently")
    arm_hashes: dict[str, list[str]] = {}
    for precision, arm_obj in arms.items():
        arm = _object(arm_obj, f"actual precision {precision}")
        hashes = arm.get("batch_data_sha256")
        if (
            arm.get("precision") != precision
            or arm.get("actual_data") is not True
            or arm.get("batch_count") != 2
            or arm.get("attempted_updates") != 40
            or arm.get("completed_updates") != 40
            or arm.get("finite_loss_gradient_checks") != 40
            or arm.get("final_params_optimizer_ema_and_recurrent_states_finite") is not True
            or not isinstance(hashes, list)
            or len(hashes) != 2
        ):
            raise ValueError(f"actual full-shape numerical checks failed for {precision}")
        arm_hashes[precision] = [_sha(value, f"{precision} batch data sha256") for value in hashes]
    if arm_hashes["fp32"] != arm_hashes["bf16"]:
        raise ValueError("FP32/BF16 precision checks used different actual host batches")
    diagnostics = gate.get("paired_step_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 40:
        raise ValueError("profile must retain all 40 paired cross-precision diagnostics")
    for index, row_obj in enumerate(diagnostics):
        row = _object(row_obj, "paired precision diagnostic")
        if row.get("update_index") != index:
            raise ValueError("paired precision diagnostics are out of update order")
        for name in (
            "fp32_loss",
            "bf16_loss",
            "fp32_gradient_norm",
            "bf16_gradient_norm",
            "loss_abs_delta",
            "loss_relative_delta",
            "gradient_norm_abs_delta",
            "gradient_norm_relative_delta",
        ):
            value = row.get(name)
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"non-finite or absent paired precision diagnostic: {name}")
        fp32_loss = float(row["fp32_loss"])
        bf16_loss = float(row["bf16_loss"])
        fp32_norm = float(row["fp32_gradient_norm"])
        bf16_norm = float(row["bf16_gradient_norm"])
        expected_deltas = {
            "loss_abs_delta": abs(fp32_loss - bf16_loss),
            "loss_relative_delta": abs(fp32_loss - bf16_loss) / max(abs(fp32_loss), 1e-8),
            "gradient_norm_abs_delta": abs(fp32_norm - bf16_norm),
            "gradient_norm_relative_delta": abs(fp32_norm - bf16_norm) / max(abs(fp32_norm), 1e-8),
        }
        if any(
            not math.isclose(float(row[name]), value, rel_tol=1e-12, abs_tol=1e-12)
            for name, value in expected_deltas.items()
        ):
            raise ValueError("paired precision diagnostics contain inconsistent delta values")
    return gate


def validate_profile_report(
    report: dict[str, object],
    *,
    input_manifest_sha256: str,
    source_identity: dict[str, str],
    preflight_identity: dict[str, object],
) -> dict[str, object]:
    """Require an exact source/data-bound actual profile with finite resource metrics."""
    arms = _object(report.get("arms"), "profile arms")
    devices = report.get("device_inventory")
    projected_training = report.get("projected_training_seconds")
    if (
        report.get("format") != "wnn.e465-tpu-fullshape-profile"
        or report.get("status") != "profile_complete"
        or report.get("input_manifest_sha256") != input_manifest_sha256
        or report.get("source_hashes") != source_identity
        or report.get("preflight_delivery_receipt_sha256")
        != preflight_identity.get("v6_delivery_receipt_sha256")
        or report.get("preflight_terminal_receipt_sha256")
        != preflight_identity.get("v6_terminal_receipt_sha256")
        or report.get("preflight_realdata_update_sha256")
        != preflight_identity.get("v6_realdata_update_sha256")
        or report.get("profile_batches") != E465_PROFILE_BATCHES
        or report.get("global_batch_size") != E465_GLOBAL_BATCH_SIZE
        or report.get("sequence_rows") != E465_SEQUENCE_ROWS
        or report.get("chunk_rows") != E465_CHUNK_ROWS
        or not isinstance(devices, list)
        or len(devices) != E465_TPU_DEVICE_COUNT
        or any(
            not isinstance(item, dict)
            or "TPU v5 lite" not in str(item.get("kind", ""))
            for item in devices
        )
        or set(arms) != {"fp32", "bf16"}
        or type(projected_training) not in {int, float}
        or not math.isfinite(projected_training)
        or projected_training <= 0
    ):
        raise ValueError("fit requires an exact matching actual-data FP32/BF16 profile")

    validate_actual_precision_gate(report)
    for precision, arm_obj in arms.items():
        arm = _object(arm_obj, f"profile arm {precision}")
        mean_batch_seconds = arm.get("mean_fullshape_batch_wall_seconds")
        peak_host_rss = arm.get("peak_host_rss_bytes")
        peak_device_fraction = arm.get("peak_device_fraction")
        if (
            arm.get("batch_count") != E465_PROFILE_BATCHES
            or arm.get("attempt_count") != E465_PROFILE_BATCHES * E465_CHUNKS_PER_BATCH
            or arm.get("completed_update_count") != arm.get("attempt_count")
            or arm.get("final_state_finite") is not True
            or type(mean_batch_seconds) not in {int, float}
            or not math.isfinite(mean_batch_seconds)
            or mean_batch_seconds <= 0
            or type(peak_host_rss) is not int
            or peak_host_rss <= 0
            or peak_host_rss >= E465_MAX_HOST_RSS_BYTES
            or type(peak_device_fraction) not in {int, float}
            or not math.isfinite(peak_device_fraction)
            or peak_device_fraction < 0
            or peak_device_fraction >= E465_DEVICE_HARD_STOP_FRACTION
        ):
            raise ValueError(f"profile resource/update gate failed for {precision}")
    if projected_training > E465_MAX_WORKER_SECONDS:
        raise ValueError("profile's two-arm training projection exceeds the eight-hour whole-job cap")
    return report


def _validate_update_checkpoint_artifacts(
    update: dict[str, object], outputs: dict[str, Path]
) -> None:
    """Bind real-data round-trip hashes and sizes to the collected checkpoint artifacts."""
    checkpoint = _object(update.get("checkpoint"), "V6 saved/reloaded checkpoints")
    for name, sha_key, bytes_key in (
        ("model_after_one_chunk_fp32.npz", "sha256", "bytes"),
        ("ema_after_one_chunk_fp32.npz", "ema_sha256", "ema_bytes"),
        ("adamw_after_one_chunk.npz", "optimizer_state_sha256", "optimizer_state_bytes"),
    ):
        path = outputs[name]
        if (
            checkpoint.get(sha_key) != sha256_file(path)
            or checkpoint.get(bytes_key) != path.stat().st_size
        ):
            raise ValueError(f"V6 round-trip receipt differs from collected checkpoint artifact: {name}")


def validate_fit_authorization(
    authorization_path: Path,
    *,
    profile_path: Path,
    profile_report: dict[str, object],
    input_manifest_sha256: str,
    source_identity: dict[str, str],
    preflight_identity: dict[str, object],
    max_worker_seconds: int,
) -> dict[str, object]:
    """Require root approval bound to the measured profile and exact source/input identities."""
    authorization = _read_json(authorization_path, "E465 fit authorization")
    precision_gate = validate_actual_precision_gate(profile_report)
    if (
        authorization.get("schema") != "wnn.e465-fit-authorization.v1"
        or authorization.get("status") != "approved"
        or authorization.get("approved_by") != "root"
        or authorization.get("fit_authorized") is not True
        or authorization.get("approved_precisions") != ["fp32", "bf16"]
        or authorization.get("profile_report_sha256") != sha256_file(profile_path)
        or authorization.get("precision_gate_sha256")
        != hashlib.sha256(_json_bytes(precision_gate)).hexdigest()
        or authorization.get("input_manifest_sha256") != input_manifest_sha256
        or authorization.get("source_hashes") != source_identity
        or authorization.get("v6_terminal_receipt_sha256")
        != preflight_identity.get("v6_terminal_receipt_sha256")
        or authorization.get("whole_job_limit_seconds") != max_worker_seconds
        or authorization.get("public_forecast") is not None
    ):
        raise ValueError("root fit authorization does not bind this exact profile/source/input")
    return authorization


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
