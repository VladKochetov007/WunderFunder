from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

# Kaggle manifests and JAX pytrees have runtime-dependent JSON/tree shapes.
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # These names are supplied by the frozen V5 parent module at injection time.
    OUTPUT: Path
    INPUT_ROOT: Path
    SOURCE_DATASET_OWNER: str
    SOURCE_DATASET_SLUG: str
    SOURCE_MANIFEST_SHA256: str
    SOURCE_ARCHIVE_SHA256: str
    ASSET_DATASET: str
    ASSET_MANIFEST_SHA256: str
    PORTABLE_INPUTS_SHA256: str
    FOLD_MANIFEST_SHA256: str
    RAW_DATA_IDENTITIES: dict[str, dict[str, Any]]
    REQUIRED_ASSET_PATHS: tuple[str, ...]
    GLOBAL_BATCH: int
    LOCAL_BATCH: int
    CHUNK_ROWS: int
    DIMENSIONS: tuple[int, ...]

    def sha256_file(path: Path) -> str: ...
    def record_phase(phase: str, scope: str, **details: object) -> None: ...
    def write_json(path: Path, value: object) -> None: ...


PACKET_ID = "e463-tpu-realdata-reload-v6-20261008"
OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v6")
INPUT_ROOT = Path("/kaggle/input")
SOURCE_DATASET_OWNER = "vladkochetov07"
SOURCE_DATASET_SLUG = "wnn-jax-realdata-e463-tpu-20261004-v1"
SOURCE_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
SOURCE_ARCHIVE_SHA256 = "a29bdf21a910a6d7b403d9ed8c52cf0761618f5ef6576b5f06d70d0a8fd5ec9e"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
PORTABLE_INPUTS_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
FOLD_MANIFEST_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PARENT_PROVENANCE_SHA256 = "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f"
EXPECTED_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
EXPECTED_NPZ_SHA256 = "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a"
RAW_DATA_IDENTITIES = {
    "train": {
        "bytes": 29240219011,
        "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    },
    "valid": {
        "bytes": 5615509128,
        "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    },
}
REQUIRED_ASSET_PATHS = (
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
GLOBAL_BATCH = 16
LOCAL_BATCH = 2
CHUNK_ROWS = 2000
SEQUENCE_ROWS = 20000
TOTAL_STEPS = 7460
DIMENSIONS = (211, 256, 2, 23)
CAUSAL_PREFIX_ROWS = 256
PARENT_PROVENANCE_B64 = "__PARENT_PROVENANCE_B64__"
E465_SUPPORT_B64: dict[str, str] = {}
E465_SUPPORT_SHA256: dict[str, str] = {}


def discover_source_mount() -> Path:
    """Find the nested source dataset by its pinned manifest bytes and identity."""
    matches: list[Path] = []
    for manifest_path in INPUT_ROOT.rglob("source-manifest.json"):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        if sha256_file(manifest_path) != SOURCE_MANIFEST_SHA256:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") == f"{SOURCE_DATASET_OWNER}/{SOURCE_DATASET_SLUG}":
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one nested E463 source mount by manifest identity, found {matches}"
        )
    return matches[0]


def _discover_asset_mount() -> tuple[Path, dict[str, Any]]:
    """Find the fold-0 asset dataset by exact id and manifest hash."""
    matches: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in INPUT_ROOT.rglob("assets-manifest.json"):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ASSET_MANIFEST_SHA256:
            continue
        manifest = json.loads(raw)
        if manifest.get("dataset") == ASSET_DATASET:
            matches.append((manifest_path.parent, manifest))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one fold-0 assets mount by manifest identity, found {len(matches)}"
        )
    return matches[0]


def _discover_raw_data_mount() -> Path:
    """Locate the paired raw parquet files under the nested Kaggle input tree."""
    matches = []
    for train_path in INPUT_ROOT.rglob("train.parquet"):
        valid_path = train_path.with_name("valid.parquet")
        if (
            train_path.is_symlink()
            or not train_path.is_file()
            or valid_path.is_symlink()
            or not valid_path.is_file()
        ):
            continue
        if (
            train_path.stat().st_size == RAW_DATA_IDENTITIES["train"]["bytes"]
            and valid_path.stat().st_size == RAW_DATA_IDENTITIES["valid"]["bytes"]
        ):
            matches.append(train_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one paired raw-data mount with pinned sizes, found {matches}"
        )
    return matches[0]


def _safe_relative_path(value: str) -> PurePosixPath:
    """Reject absolute, empty, or parent-traversing manifest paths."""
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe manifest path: {value}")
    return relative


def _link_verified(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    source_hash_already_verified: bool = False,
) -> None:
    """Create one read-only symlink after checking source and collision identity."""
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"expected regular source file: {source}")
    if not source_hash_already_verified and sha256_file(source) != expected_sha256:
        raise ValueError(f"input content hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file():
            raise ValueError(f"conflicting staged path: {destination}")
        if source_hash_already_verified and destination.resolve() != source.resolve():
            raise ValueError(f"conflicting staged path: {destination}")
        if not source_hash_already_verified and sha256_file(destination) != expected_sha256:
            raise ValueError(f"conflicting staged path: {destination}")
        return
    destination.symlink_to(source)


def materialize_realdata_project(
    source_root: Path, source_materialization: dict[str, Any]
) -> dict[str, Any]:
    """Stage only verified source, fold caches, and the exact supplied parquet pair."""
    asset_root, asset_manifest = _discover_asset_mount()
    if asset_manifest.get("private") is not True or asset_manifest.get("version") != 1:
        raise ValueError("fold-0 assets must remain the reviewed private version-1 bundle")
    entries = asset_manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("fold-0 assets file inventory differs from the reviewed 24-file bundle")
    by_path = {str(item["path"]): item for item in entries}
    if len(by_path) != len(entries) or set(REQUIRED_ASSET_PATHS) - set(by_path):
        raise ValueError("fold-0 assets inventory has duplicate or missing required members")
    portable_entry = by_path["research/campaign_2026-10-04/portable_inputs.json"]
    if portable_entry.get("sha256") != PORTABLE_INPUTS_SHA256:
        raise ValueError("portable input registry differs from its pinned identity")
    portable_path = asset_root / str(portable_entry["filename"])
    if portable_path.stat().st_size != portable_entry.get("size"):
        raise ValueError("portable input registry size differs from its asset manifest")
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    if (
        portable.get("version") != 1
        or portable.get("split_manifest_sha256") != FOLD_MANIFEST_SHA256
    ):
        raise ValueError("portable input registry split identity changed")
    raw_mount = _discover_raw_data_mount()
    raw_identities: dict[str, dict[str, Any]] = {}
    for split, expected in RAW_DATA_IDENTITIES.items():
        path = raw_mount / f"{split}.parquet"
        registry = portable.get("sources", {}).get(f"datasets/{split}.parquet", {})
        if (
            registry.get("dataset_ref") != "vladkochetov07/wunder-funder"
            or registry.get("sha256") != expected["sha256"]
            or registry.get("original_identity", {}).get("bytes") != expected["bytes"]
        ):
            raise ValueError(f"portable registry differs for datasets/{split}.parquet")
        record_phase("realdata_raw_hash_started", "parent", split=split, bytes=expected["bytes"])
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"raw {split} parquet byte length differs")
        actual_sha = sha256_file(path)
        if actual_sha != expected["sha256"]:
            raise ValueError(f"raw {split} parquet SHA-256 differs: {actual_sha}")
        raw_identities[split] = {
            "dataset_ref": "vladkochetov07/wunder-funder",
            "mounted_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
        }
        record_phase("realdata_raw_hash_complete", "parent", split=split, sha256=actual_sha)

    project_root = OUTPUT / "realdata_project"
    project_root.mkdir(parents=True, exist_ok=False)
    source_files = source_materialization.get("files")
    if not isinstance(source_files, dict) or len(source_files) != 17:
        raise ValueError("E463 source-tree verification did not return the 17-file closure")
    for relative, identity in sorted(source_files.items()):
        destination = project_root.joinpath(*_safe_relative_path(relative).parts)
        source = source_root.joinpath(*_safe_relative_path(relative).parts)
        _link_verified(source, destination, str(identity["sha256"]))
    verified_assets: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_ASSET_PATHS:
        entry = by_path[relative]
        source = asset_root / str(entry["filename"])
        if source.stat().st_size != entry.get("size"):
            raise ValueError(f"asset byte length differs for {relative}")
        expected_sha = str(entry["sha256"])
        _link_verified(
            source,
            project_root.joinpath(*_safe_relative_path(relative).parts),
            expected_sha,
        )
        verified_assets[relative] = {
            "bytes": int(entry["size"]),
            "sha256": expected_sha,
        }
    for split, identity in raw_identities.items():
        _link_verified(
            raw_mount / f"{split}.parquet",
            project_root / "datasets" / f"{split}.parquet",
            str(identity["sha256"]),
            source_hash_already_verified=True,
        )
    return {
        "status": "verified_and_staged_read_only_inputs",
        "project_root": str(project_root),
        "source_dataset": f"{SOURCE_DATASET_OWNER}/{SOURCE_DATASET_SLUG}",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_file_count": len(source_files),
        "source_bytes": int(source_materialization["verified_bytes"]),
        "fold_assets_dataset": ASSET_DATASET,
        "fold_assets_manifest_sha256": ASSET_MANIFEST_SHA256,
        "verified_asset_files": verified_assets,
        "raw_data": raw_identities,
        "dataset_version_note": "Kaggle mount does not expose the attached dataset version; exact manifest and content hashes bind the observed bytes.",
    }


def validate_selected_source_membership(
    batch_pairs: Sequence[tuple[str, int]],
    valid_fit_groups: Sequence[int],
    evaluation_groups: Sequence[int],
    parent_fit_valid_groups: Sequence[int],
) -> dict[str, Any]:
    """Require exactly one B16 batch of source-qualified valid fit sequences."""
    normalized = tuple((str(split), int(group)) for split, group in batch_pairs)
    if len(normalized) != GLOBAL_BATCH or len(set(normalized)) != GLOBAL_BATCH:
        raise ValueError("real-data probe requires 16 unique source-qualified sequences")
    if any(split != "valid" for split, _ in normalized):
        raise ValueError("real-data compatibility probe is restricted to fit-VALID sequences")
    selected = {group for _, group in normalized}
    valid_fit = set(map(int, valid_fit_groups))
    evaluation = set(map(int, evaluation_groups))
    parent_fit = set(map(int, parent_fit_valid_groups))
    if not selected.issubset(valid_fit):
        raise ValueError("selected VALID sequence is outside the frozen fold fit membership")
    if selected & evaluation:
        raise ValueError("selected sequence overlaps fold evaluation membership")
    if not selected.issubset(parent_fit):
        raise ValueError("selected sequence is outside the source checkpoint fit membership")
    return {
        "status": "passed",
        "split": "valid",
        "batch_size": len(normalized),
        "row_groups": [group for _, group in normalized],
        "evaluation_overlap_count": len(selected & evaluation),
        "outside_parent_fit_count": len(selected - parent_fit),
        "qualified_membership": [list(pair) for pair in normalized],
        "heldout_claim": None,
    }


def _embedded_parent_provenance() -> dict[str, Any]:
    """Decode the frozen own-source membership receipt embedded by the builder."""
    raw = base64.b64decode(PARENT_PROVENANCE_B64, validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PARENT_PROVENANCE_SHA256:
        raise ValueError("embedded own-checkpoint provenance SHA-256 differs")
    return json.loads(raw)


def _hash_batch(batch: dict[str, Any], np: Any) -> str:
    """Hash every host batch array with name, dtype, shape, and C-order bytes."""
    digest = hashlib.sha256()
    for name in sorted(batch):
        array = np.ascontiguousarray(batch[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _tree_is_finite(tree: Any, jax: Any, jnp: Any) -> bool:
    """Synchronize and check all leaves of a JAX tree for finite values."""
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def _replicate_tree(tree: Any, jax: Any, jnp: Any, devices: int) -> Any:
    """Replicate an immutable host parameter/optimizer tree over eight TPUs."""
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(jnp.asarray(value)[None, ...], (devices, *value.shape)),
        tree,
    )


def _first_replica(tree: Any, jax: Any) -> Any:
    """Select replica zero from a synchronized mapped JAX tree."""
    return jax.tree_util.tree_map(lambda value: value[0], tree)


def run_realdata_one_update(
    jax: Any,
    jnp: Any,
    np: Any,
    params: dict[str, Any],
    devices: list[Any],
    source_root: Path,
) -> dict[str, Any]:
    """Run one B16/C2000 real-data update, save/reload, and check causal inference."""
    from distributed_combo import global_combo_loss_and_local_surrogate, pmean_and_clip_gradient
    from e465_host_contract import validate_fold_membership
    from onecycle import onecycle_point
    from wnn_jax import bridge, rnn

    project_root = Path(os.environ["WNN_E463_DATA_ROOT"])
    record_phase("realdata_source_ancestry_gate", "device_worker")
    fold_path = project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    if sha256_file(fold_path) != FOLD_MANIFEST_SHA256:
        raise ValueError("real-data fold-0 manifest SHA-256 differs")
    fold = bridge.load_fold_groups(fold_path)
    train_groups, fit_valid_groups, evaluation_groups = validate_fold_membership(
        fold.train, fold.valid_fit, fold.evaluation
    )
    if fold.manifest_sha256 != FOLD_MANIFEST_SHA256:
        raise ValueError("bridge fold manifest identity differs")
    parent_provenance = _embedded_parent_provenance()
    if (
        parent_provenance.get("split_manifest_sha256") != FOLD_MANIFEST_SHA256
        or parent_provenance.get("selected_epoch") != 3
        or parent_provenance.get("checkpoint_sha256", {}).get("model_ep3.pt")
        != EXPECTED_CHECKPOINT_SHA256
        or set(map(int, parent_provenance.get("fit_row_groups", {}).get("train", [])))
        != set(train_groups)
        or set(map(int, parent_provenance.get("fit_row_groups", {}).get("valid", [])))
        != set(fit_valid_groups)
        or set(map(int, parent_provenance.get("evaluation_row_groups", [])))
        != set(evaluation_groups)
    ):
        raise ValueError("own epoch-3 checkpoint provenance differs from the pinned fold")
    valid_membership = tuple(("valid", group) for group in fit_valid_groups)
    group_batches = bridge.seeded_group_batches(
        valid_membership, seed=3, epoch=0, batch_size=GLOBAL_BATCH
    )
    selected_pairs = group_batches[0]
    membership = validate_selected_source_membership(
        selected_pairs,
        fit_valid_groups,
        evaluation_groups,
        parent_provenance["fit_row_groups"]["valid"],
    )
    if len(devices) != 8 or any(device.platform != "tpu" for device in devices):
        raise RuntimeError("real-data update requires exactly the eight verified TPU devices")

    record_phase("realdata_batch_load", "device_worker", batch_size=GLOBAL_BATCH)
    teacher = bridge.ComboTeacherMean(project_root, "valid", fold.manifest_sha256)
    propensity = bridge.PropensityCache(project_root, "valid")
    store = bridge.RawSequenceStore(project_root)
    samples = [
        bridge.load_sequence_supervision(
            project_root,
            split,
            group,
            teacher,
            propensity,
            store,
            groups=fold,
        )
        for split, group in selected_pairs
    ]
    host_batch = bridge.collate_sequences(samples)
    if host_batch["inputs"].shape != (GLOBAL_BATCH, SEQUENCE_ROWS, DIMENSIONS[0]):
        raise ValueError(f"B16 real-data input shape differs: {host_batch['inputs'].shape}")
    batch_sha256 = _hash_batch(host_batch, np)
    sharded_batch = {}
    for name, value in host_batch.items():
        if value.shape[:2] != (GLOBAL_BATCH, SEQUENCE_ROWS):
            raise ValueError(f"real-data supervision shape differs for {name}: {value.shape}")
        chunk = np.ascontiguousarray(value[:, :CHUNK_ROWS])
        sharded_batch[name] = jnp.asarray(chunk.reshape(8, LOCAL_BATCH, *chunk.shape[1:]))

    schedule = onecycle_point(0, TOTAL_STEPS)
    replicated_params = _replicate_tree(params, jax, jnp, len(devices))
    initial_optimizer = _replicate_tree(rnn.adamw_init(params), jax, jnp, len(devices))
    ema_params = replicated_params
    recurrent_shape = (len(devices), 2, LOCAL_BATCH, DIMENSIONS[1])
    student_state = jnp.zeros(recurrent_shape, dtype=jnp.float32)
    ema_state = jnp.zeros(recurrent_shape, dtype=jnp.float32)

    def replica_step(
        current_params: Any,
        optimizer: Any,
        current_ema: Any,
        current_state: Any,
        current_ema_state: Any,
        batch: dict[str, Any],
        learning_rate: Any,
        beta1: Any,
    ) -> tuple[Any, ...]:
        def objective(candidate: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            prediction, next_state = rnn.forward_chunk(
                candidate, batch["inputs"], current_state, precision="fp32"
            )
            ema_prediction, next_ema_state = rnn.forward_chunk(
                current_ema, batch["inputs"], current_ema_state, precision="fp32"
            )
            pooled, local_surrogate = global_combo_loss_and_local_surrogate(
                prediction,
                batch,
                ema_prediction,
                axis_name="replicas",
                replica_count=len(devices),
            )
            return local_surrogate, (pooled, next_state, next_ema_state)

        (_, (pooled_loss, next_state, next_ema_state)), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(current_params)
        gradients, gradient_norm = pmean_and_clip_gradient(
            gradients, axis_name="replicas", max_norm=1.0
        )
        next_params, next_optimizer = rnn.adamw_update(
            current_params,
            gradients,
            optimizer,
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta1=beta1,
            beta2=0.999,
        )
        next_ema = {
            name: value if name in {"mean", "std"} else 0.999 * current_ema[name] + 0.001 * value
            for name, value in next_params.items()
        }
        return (
            next_params,
            next_optimizer,
            next_ema,
            jax.lax.stop_gradient(next_state),
            jax.lax.stop_gradient(next_ema_state),
            pooled_loss,
            gradient_norm,
        )

    record_phase("realdata_one_optimizer_update", "device_worker", chunk_rows=CHUNK_ROWS)
    mapped_step = jax.pmap(replica_step, axis_name="replicas", devices=devices)
    update = mapped_step(
        replicated_params,
        initial_optimizer,
        ema_params,
        student_state,
        ema_state,
        sharded_batch,
        jnp.full((len(devices),), schedule.learning_rate, dtype=jnp.float32),
        jnp.full((len(devices),), schedule.beta1, dtype=jnp.float32),
    )
    jax.block_until_ready(update)
    next_params, next_optimizer, next_ema, _student_state, _ema_state, losses, norms = update
    if not _tree_is_finite((next_params, next_optimizer, losses, norms), jax, jnp):
        raise FloatingPointError("real-data update produced a non-finite tensor")
    counts = np.asarray(jax.device_get(next_optimizer.count))
    loss_values = np.asarray(jax.device_get(losses), dtype=np.float64)
    norm_values = np.asarray(jax.device_get(norms), dtype=np.float64)
    if counts.tolist() != [1] * len(devices):
        raise AssertionError(f"real-data update count differs: {counts.tolist()}")
    if not np.allclose(loss_values, loss_values[0], rtol=1e-6, atol=1e-7):
        raise AssertionError("globally pooled real-data loss differs across replicas")
    if not np.allclose(norm_values, norm_values[0], rtol=1e-6, atol=1e-7):
        raise AssertionError("global gradient norm differs across replicas")

    record_phase("realdata_checkpoint_save_reload", "device_worker")
    saved_params = _first_replica(next_params, jax)
    saved_optimizer = _first_replica(next_optimizer, jax)
    saved_ema = _first_replica(next_ema, jax)
    artifact_dir = OUTPUT / "realdata_reload"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = artifact_dir / "model_after_one_chunk_fp32.npz"
    rnn.export_npz_state(checkpoint_path, saved_params)
    ema_checkpoint_path = artifact_dir / "ema_after_one_chunk_fp32.npz"
    rnn.export_npz_state(ema_checkpoint_path, saved_ema)
    optimizer_path = artifact_dir / "adamw_after_one_chunk.npz"
    optimizer_arrays = {"count": np.asarray(jax.device_get(saved_optimizer.count), dtype=np.int32)}
    optimizer_arrays.update(
        {
            f"first::{name}": np.asarray(jax.device_get(value), dtype=np.float32)
            for name, value in saved_optimizer.first.items()
        }
    )
    optimizer_arrays.update(
        {
            f"second::{name}": np.asarray(jax.device_get(value), dtype=np.float32)
            for name, value in saved_optimizer.second.items()
        }
    )
    np.savez(optimizer_path, **optimizer_arrays)
    reloaded_params = rnn.load_npz_state(checkpoint_path, *DIMENSIONS)
    for name in saved_params:
        if not np.array_equal(
            np.asarray(jax.device_get(saved_params[name])),
            np.asarray(jax.device_get(reloaded_params[name])),
        ):
            raise AssertionError(f"checkpoint tensor changed after NPZ save/reload: {name}")
    reloaded_ema = rnn.load_npz_state(ema_checkpoint_path, *DIMENSIONS)
    for name in saved_ema:
        if not np.array_equal(
            np.asarray(jax.device_get(saved_ema[name])),
            np.asarray(jax.device_get(reloaded_ema[name])),
        ):
            raise AssertionError(f"EMA checkpoint tensor changed after NPZ save/reload: {name}")
    with np.load(optimizer_path, allow_pickle=False) as archive:
        if int(archive["count"]) != 1:
            raise AssertionError("saved/reloaded AdamW count differs from the one update")
        for name, value in saved_optimizer.first.items():
            if not np.array_equal(archive[f"first::{name}"], np.asarray(jax.device_get(value))):
                raise AssertionError(f"saved/reloaded first moment differs for {name}")
        for name, value in saved_optimizer.second.items():
            if not np.array_equal(archive[f"second::{name}"], np.asarray(jax.device_get(value))):
                raise AssertionError(f"saved/reloaded second moment differs for {name}")

    record_phase("realdata_causal_inference_check", "device_worker")
    sequence = jnp.asarray(host_batch["inputs"][:1, :CHUNK_ROWS])
    before_reload, _ = rnn.forward_chunk(saved_ema, sequence, None, precision="fp32")
    after_reload, _ = rnn.forward_chunk(reloaded_ema, sequence, None, precision="fp32")
    before_reload = np.asarray(jax.device_get(before_reload))
    after_reload = np.asarray(jax.device_get(after_reload))
    if not np.array_equal(before_reload, after_reload):
        raise AssertionError("EMA inference changed after the checkpoint save/reload")
    changed_tail = np.asarray(host_batch["inputs"][:1, :CHUNK_ROWS]).copy()
    changed_tail[:, CAUSAL_PREFIX_ROWS:] += np.float32(0.125)
    changed_prediction, _ = rnn.forward_chunk(
        reloaded_ema, jnp.asarray(changed_tail), None, precision="fp32"
    )
    changed_prediction = np.asarray(jax.device_get(changed_prediction))
    if not np.array_equal(
        after_reload[:, :CAUSAL_PREFIX_ROWS],
        changed_prediction[:, :CAUSAL_PREFIX_ROWS],
    ):
        raise AssertionError("future input perturbation changed earlier causal predictions")
    zero_state = jnp.zeros((DIMENSIONS[2], 1, DIMENSIONS[1]), dtype=jnp.float32)
    explicit_zero_prediction, _ = rnn.forward_chunk(
        reloaded_ema, sequence, zero_state, precision="fp32"
    )
    explicit_zero_prediction = np.asarray(jax.device_get(explicit_zero_prediction))
    if not np.array_equal(after_reload, explicit_zero_prediction):
        raise AssertionError("omitted recurrent state does not reset to an all-zero state")

    result = {
        "status": "passed",
        "classification": "one real-data B16/C2000 distributed E463 checkpoint/reload compatibility update",
        "checkpoint_parent": "c20261003_combo_s3_f0@ep3 evaluation EMA",
        "parent_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "parent_npz_sha256": EXPECTED_NPZ_SHA256,
        "parent_provenance_sha256": PARENT_PROVENANCE_SHA256,
        "fold_manifest_sha256": fold.manifest_sha256,
        "source_qualified_membership": membership,
        "batch_group_sha256": bridge.group_batch_sha256((selected_pairs,)),
        "batch_data_sha256": batch_sha256,
        "global_batch": GLOBAL_BATCH,
        "local_batch_per_tpu": LOCAL_BATCH,
        "chunk_rows": CHUNK_ROWS,
        "input_width": DIMENSIONS[0],
        "hidden_width": DIMENSIONS[1],
        "layers": DIMENSIONS[2],
        "output_width": DIMENSIONS[3],
        "optimizer": "fresh FP32 AdamW state, one chunk-level update, weight_decay=0.01, global norm clip=1.0",
        "schedule": {
            "family": "E465 OneCycle",
            "step_index": 0,
            "total_steps": TOTAL_STEPS,
            "learning_rate": schedule.learning_rate,
            "beta1": schedule.beta1,
            "beta2": schedule.beta2,
        },
        "objective": "E465 distributed_combo global pooled four-term combo loss with stopped incoming epoch-3 EMA predictions; no new objective terms",
        "completed_update_count_per_replica": counts.tolist(),
        "pooled_loss": float(loss_values[0]),
        "global_gradient_norm_before_clip": float(norm_values[0]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "ema_path": str(ema_checkpoint_path),
            "ema_sha256": sha256_file(ema_checkpoint_path),
            "ema_bytes": ema_checkpoint_path.stat().st_size,
            "optimizer_state_path": str(optimizer_path),
            "optimizer_state_sha256": sha256_file(optimizer_path),
            "optimizer_state_bytes": optimizer_path.stat().st_size,
            "save_reload_parameter_identity": "bitwise_equal_all_tensors",
            "save_reload_ema_identity": "bitwise_equal_all_tensors",
            "save_reload_optimizer_identity": "bitwise_equal_all_moments_and_count",
        },
        "inference_causality": {
            "status": "passed",
            "inference_state": "updated EMA parameters",
            "checkpoint_reload_prediction_identity": "bitwise_equal",
            "future_tail_perturbation_prefix_rows": CAUSAL_PREFIX_ROWS,
            "future_perturbation_changed_prefix": False,
            "omitted_state_matches_explicit_zero_state": True,
            "prediction_exported": False,
        },
        "training_data_used": True,
        "fit_valid_is_already_exposed": True,
        "heldout_claim": None,
        "metric_or_score": None,
        "public_forecast": None,
    }
    write_json(artifact_dir / "realdata_update_receipt.json", result)
    return result
