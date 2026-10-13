from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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
PACKET_OWNER = "vladkochetov07"
PACKET_TITLE = "E463 TPU Realdata Reload V8 20261013"
PACKET_SLUG = "e463-tpu-realdata-reload-v8-20261013"
PACKET_ID = f"{PACKET_OWNER}/{PACKET_SLUG}"
OUTPUT_DIR = "/kaggle/working/cloud_preflight_tpu_v8"
V7_PROBE_SHA256 = "d22947623bbe638280183ab019d4159ea37038b73840f3058e0137379bb4dad7"
V7_MANIFEST_SHA256 = "eb332ac3c1dc4913f8c0f457a1fda5bdc7ea2f4118eb921e485021d519d12c82"
V7_MOUNT_FIXTURE_SHA256 = "feaddf63b41d413e120c1548239a1d295ee31842b9f8aada73b837537589f9d9"
V7_TERMINAL_SHA256 = "5f7731f90d9d5d6dfd9e1b95a58d5bd3024546f73868fb5adca12942d8bc4c01"
PORTABLE_INPUTS_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
FOLD_MANIFEST_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
RAW_DATASET_REF = "vladkochetov07/wunder-funder"


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest for a byte string."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one small frozen packet artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_exact(text: str, old: str, new: str, label: str, count: int = 1) -> str:
    """Replace a pinned source fragment only when its occurrence count is exact."""
    observed = text.count(old)
    if observed != count:
        raise ValueError(f"expected {count} {label} markers, found {observed}")
    return text.replace(old, new)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    """Replace one bounded source section while rejecting ambiguous markers."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one start and end marker for {label}")
    first = text.index(start)
    last = text.index(end, first)
    if last <= first:
        raise ValueError(f"invalid source marker order for {label}")
    return text[:first] + replacement + text[last:]


def insert_after(text: str, marker: str, insertion: str, label: str) -> str:
    """Insert text after one exact source marker."""
    if text.count(marker) != 1:
        raise ValueError(f"expected one {label} marker")
    return text.replace(marker, marker + insertion, 1)


def build_probe() -> bytes:
    """Build the V8 Kaggle entrypoint from the immutable V7 packet source."""
    source_path = V7_PACKET_DIR / "probe.py"
    source = source_path.read_bytes()
    if sha256_bytes(source) != V7_PROBE_SHA256:
        raise ValueError("pinned V7 probe bytes changed")
    text = source.decode("utf-8")
    text = replace_exact(
        text,
        'PACKET_ID = "e463-tpu-realdata-reload-v7-20261008"',
        f'PACKET_ID = "{PACKET_SLUG}"',
        "packet id",
        2,
    )
    text = replace_exact(
        text,
        'OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v7")',
        f'OUTPUT = Path("{OUTPUT_DIR}")',
        "output directory",
        2,
    )
    text = insert_after(
        text,
        'ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"\n',
        'RAW_DATASET_OWNER = "vladkochetov07"\n'
        'RAW_DATASET_SLUG = "wunder-funder"\n'
        'RAW_DATASET_REF = f"{RAW_DATASET_OWNER}/{RAW_DATASET_SLUG}"\n',
        "raw-data dataset mount identity",
    )

    raw_identity_block = """RAW_DATA_IDENTITIES = {
    "train": {
        "bytes": 29240219011,
        "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "footer_bytes": 60437285,
        "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
    },
    "valid": {
        "bytes": 5615509128,
        "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        "footer_bytes": 10686926,
        "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
    },
}
"""
    text = replace_between(
        text,
        "RAW_DATA_IDENTITIES = {\n",
        "REQUIRED_ASSET_PATHS = (\n",
        raw_identity_block,
        "raw-data identities",
    )

    text = replace_between(
        text,
        "def _discover_raw_data_mount() -> Path:\n",
        "def _safe_relative_path(value: str) -> PurePosixPath:\n",
        '''def parquet_footer_identity(path: Path, expected_footer_bytes: int) -> dict[str, Any]:
    """Read a bounded Parquet footer and return its byte count and SHA-256."""
    file_bytes = path.stat().st_size
    if file_bytes < 8:
        raise ValueError(f"raw parquet is shorter than its trailer: {path}")
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if (
            trailer[4:] != b"PAR1"
            or footer_bytes != expected_footer_bytes
            or footer_bytes > file_bytes - 8
        ):
            raise ValueError(f"raw parquet footer length or magic differs: {path}")
        stream.seek(-8 - footer_bytes, os.SEEK_END)
        digest = hashlib.sha256()
        remaining = footer_bytes + 8
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"raw parquet footer ended early: {path}")
            digest.update(block)
            remaining -= len(block)
    return {"footer_bytes": footer_bytes, "footer_sha256": digest.hexdigest()}


def validate_portable_data_registry(
    portable: dict[str, Any], split: str, expected: dict[str, Any]
) -> dict[str, Any]:
    """Validate pinned registry content without assuming it stores Kaggle slug."""
    if (
        portable.get("version") != 1
        or portable.get("split_manifest_sha256") != FOLD_MANIFEST_SHA256
    ):
        raise ValueError("portable input registry split identity changed")
    sources = portable.get("sources")
    registry = (
        sources.get(f"datasets/{split}.parquet") if isinstance(sources, dict) else None
    )
    if not isinstance(registry, dict):
        raise ValueError(f"portable input registry lacks datasets/{split}.parquet")
    original = registry.get("original_identity")
    if not isinstance(original, dict) or (
        registry.get("sha256") != expected["sha256"]
        or original.get("bytes") != expected["bytes"]
        or original.get("footer_bytes") != expected["footer_bytes"]
        or original.get("footer_sha256") != expected["footer_sha256"]
    ):
        raise ValueError(f"portable registry differs for datasets/{split}.parquet")
    return registry


def verify_raw_data_file(
    path: Path, split: str, expected: dict[str, Any]
) -> dict[str, Any]:
    """Require exact mounted parquet bytes, footer, and whole-file identity."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"expected regular mounted parquet file: {path}")
    size = path.stat().st_size
    if size != expected["bytes"]:
        raise ValueError(f"raw {split} parquet byte length differs")
    footer = parquet_footer_identity(path, expected["footer_bytes"])
    if footer["footer_sha256"] != expected["footer_sha256"]:
        raise ValueError(f"raw {split} parquet footer SHA-256 differs")
    actual_sha = sha256_file(path)
    if actual_sha != expected["sha256"]:
        raise ValueError(f"raw {split} parquet SHA-256 differs: {actual_sha}")
    return {
        "bytes": size,
        "sha256": actual_sha,
        **footer,
    }


def _discover_raw_data_mount() -> Path:
    """Resolve the separately attached raw-data slug and reject ambiguous mounts."""
    expected_root = INPUT_ROOT / "datasets" / RAW_DATASET_OWNER / RAW_DATASET_SLUG
    if expected_root.is_symlink() or not expected_root.is_dir():
        raise FileNotFoundError(f"expected raw-data dataset mount is absent: {expected_root}")
    matches: list[Path] = []
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
    if len(matches) != 1 or matches[0].resolve() != expected_root.resolve():
        raise FileNotFoundError(
            "expected one paired raw-data mount at the pinned Kaggle dataset slug; "
            f"expected={expected_root}, matches={matches}"
        )
    return expected_root


''',
        "raw-data mount and identity validators",
    )

    old_registry_gate = """        registry = portable.get("sources", {}).get(f"datasets/{split}.parquet", {})
        if (
            registry.get("dataset_ref") != "vladkochetov07/wunder-funder"
            or registry.get("sha256") != expected["sha256"]
            or registry.get("original_identity", {}).get("bytes") != expected["bytes"]
        ):
            raise ValueError(f"portable registry differs for datasets/{split}.parquet")
"""
    text = replace_exact(
        text,
        old_registry_gate,
        "        validate_portable_data_registry(portable, split, expected)\n",
        "unsupported portable dataset_ref gate",
    )
    old_mounted_hash = """        if path.stat().st_size != expected["bytes"]:
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
"""
    text = replace_exact(
        text,
        old_mounted_hash,
        """        mounted_identity = verify_raw_data_file(path, split, expected)
        raw_identities[split] = {
            "mounted_dataset": RAW_DATASET_REF,
            "mount_relative_path": raw_mount.relative_to(INPUT_ROOT).as_posix(),
            "mounted_path": str(path),
            **mounted_identity,
        }
""",
        "mounted parquet identity checks",
    )
    text = replace_exact(
        text,
        'record_phase("realdata_raw_hash_complete", "parent", split=split, sha256=actual_sha)',
        "record_phase(\n"
        '            "realdata_raw_hash_complete",\n'
        '            "parent",\n'
        "            split=split,\n"
        '            sha256=mounted_identity["sha256"],\n'
        "        )",
        "mounted parquet hash receipt",
    )

    prediction_marker = """    if not np.array_equal(after_reload, explicit_zero_prediction):
        raise AssertionError("omitted recurrent state does not reset to an all-zero state")

    result = {
"""
    prediction_export = """    if not np.array_equal(after_reload, explicit_zero_prediction):
        raise AssertionError("omitted recurrent state does not reset to an all-zero state")

    record_phase(
        "realdata_prediction_export",
        "device_worker",
        split="valid",
        group=int(selected_pairs[0][1]),
        rows=SEQUENCE_ROWS,
    )
    full_sequence = np.ascontiguousarray(host_batch["inputs"][:1], dtype=np.float32)
    sequence_input_sha256 = hashlib.sha256(memoryview(full_sequence).cast("B")).hexdigest()
    prediction_chunks: list[np.ndarray] = []
    export_state: Any = None
    for start in range(0, SEQUENCE_ROWS, CHUNK_ROWS):
        chunk_prediction, export_state = rnn.forward_chunk(
            reloaded_ema,
            jnp.asarray(full_sequence[:, start : start + CHUNK_ROWS]),
            export_state,
            precision="fp32",
        )
        prediction_chunks.append(np.asarray(jax.device_get(chunk_prediction), dtype=np.float32))
    exported_prediction = np.concatenate(prediction_chunks, axis=1)
    expected_prediction_shape = (1, SEQUENCE_ROWS, DIMENSIONS[3])
    if exported_prediction.shape != expected_prediction_shape:
        raise ValueError(f"real-data prediction export shape differs: {exported_prediction.shape}")
    if not np.isfinite(exported_prediction).all():
        raise FloatingPointError("real-data prediction export contains nonfinite values")
    if not np.array_equal(exported_prediction[:, :CHUNK_ROWS], after_reload):
        raise AssertionError("full-sequence export differs from the reloaded first chunk")
    prediction_path = artifact_dir / "updated_ema_valid_fit_group_prediction.npz"
    np.savez_compressed(
        prediction_path,
        predictions=exported_prediction,
        split=np.asarray("valid"),
        row_group=np.asarray(int(selected_pairs[0][1]), dtype=np.int32),
        rows=np.asarray(SEQUENCE_ROWS, dtype=np.int32),
        input_sha256=np.asarray(sequence_input_sha256),
        checkpoint_sha256=np.asarray(sha256_file(ema_checkpoint_path)),
    )
    prediction_identity = {
        "path": str(prediction_path),
        "bytes": prediction_path.stat().st_size,
        "sha256": sha256_file(prediction_path),
        "split": "valid",
        "row_group": int(selected_pairs[0][1]),
        "source_qualified_group": ["valid", int(selected_pairs[0][1])],
        "input_sha256": sequence_input_sha256,
        "rows": SEQUENCE_ROWS,
        "columns": DIMENSIONS[3],
        "dtype": "float32",
        "checkpoint_sha256": sha256_file(ema_checkpoint_path),
        "state_policy": "reset at sequence start; carry state across 2,000-row chunks",
        "heldout_claim": None,
        "score": None,
    }

    result = {
"""
    text = replace_exact(
        text,
        prediction_marker,
        prediction_export,
        "full real-data prediction export",
    )
    text = replace_exact(
        text,
        '            "prediction_exported": False,\n',
        '            "prediction_exported": True,\n',
        "prediction export receipt flag",
    )
    text = replace_exact(
        text,
        '        "training_data_used": True,\n',
        '        "prediction_export": prediction_identity,\n        "training_data_used": True,\n',
        "prediction export receipt",
    )

    text = replace_exact(
        text,
        "import signal\nimport stat\n",
        "import shutil\nimport signal\nimport stat\n",
        "staging cleanup import",
    )
    cleanup_helper = '''\ndef cleanup_realdata_stage() -> None:
    """Remove only the temporary symlink tree after the child stops using it."""
    stage = OUTPUT / "realdata_project"
    receipt = OUTPUT / "realdata_staging_cleanup.json"
    if not stage.exists() and receipt.is_file():
        return
    if stage.is_symlink() or (stage.exists() and not stage.is_dir()):
        raise ValueError(f"refusing to remove an unexpected staged-data path: {stage}")
    removed = stage.exists()
    if removed:
        for child in stage.rglob("*"):
            if not child.is_symlink() and not child.is_dir():
                raise ValueError(f"refusing to remove a non-symlink staged payload: {child}")
        shutil.rmtree(stage)
    write_json(
        OUTPUT / "realdata_staging_cleanup.json",
        {
            "status": "removed_symlink_stage" if removed else "stage_absent",
            "path": str(stage),
            "mounted_source_payloads_modified": False,
            "staged_payloads_copied": False,
        },
    )


'''
    text = replace_exact(
        text,
        "def worker_main() -> int:\n",
        cleanup_helper + "def worker_main() -> int:\n",
        "staging cleanup helper",
    )
    text = replace_exact(
        text,
        "        result = run_device_worker()\n",
        "        result = run_device_worker()\n        cleanup_realdata_stage()\n",
        "successful worker staging cleanup",
    )
    text = replace_exact(
        text,
        "    except BaseException as error:\n        TERMINAL = {\n",
        "    except BaseException as error:\n"
        "        try:\n"
        "            cleanup_realdata_stage()\n"
        "        except (OSError, ValueError) as cleanup_error:\n"
        '            error = RuntimeError(f"{error}; staged-input cleanup failed: {cleanup_error}")\n'
        "        TERMINAL = {\n",
        "failed worker staging cleanup",
        2,
    )
    text = replace_exact(
        text,
        "        child.wait(timeout=5)\n\n\ndef parent_main()",
        "        child.wait(timeout=5)\n    cleanup_realdata_stage()\n\n\ndef parent_main()",
        "timeout-killed worker staging cleanup",
    )
    return text.encode("utf-8")


def build() -> dict[str, object]:
    """Build the source-only packet, deterministic zip, and provenance manifest."""
    title_slug = re.sub(r"[^a-z0-9]+", "-", PACKET_TITLE.lower()).strip("-")
    if title_slug != PACKET_SLUG:
        raise ValueError(f"packet slug is not title-derived: {title_slug}")
    pinned = {
        "V7 packet probe": (V7_PACKET_DIR / "probe.py", V7_PROBE_SHA256),
        "V7 packet manifest": (V7_PACKET_MANIFEST, V7_MANIFEST_SHA256),
        "V7 actual mount fixture": (V7_MOUNT_FIXTURE, V7_MOUNT_FIXTURE_SHA256),
        "V7 terminal receipt": (V7_TERMINAL_RECEIPT, V7_TERMINAL_SHA256),
        "portable registry": (PORTABLE_INPUTS, PORTABLE_INPUTS_SHA256),
        "fold-0 assets manifest": (ASSET_MANIFEST, ASSET_MANIFEST_SHA256),
    }
    for name, (path, expected) in pinned.items():
        if sha256_file(path) != expected:
            raise ValueError(f"pinned reference changed: {name}: {path}")

    base_manifest = json.loads(V7_PACKET_MANIFEST.read_text())
    if base_manifest.get("code_sha256") != V7_PROBE_SHA256:
        raise ValueError("V7 manifest does not bind its frozen source")
    portable = json.loads(PORTABLE_INPUTS.read_text())
    identities = {
        "train": {
            "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
            "bytes": 29240219011,
            "footer_bytes": 60437285,
            "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
        },
        "valid": {
            "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
            "bytes": 5615509128,
            "footer_bytes": 10686926,
            "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
        },
    }
    sources = portable.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("portable input fixture lacks a source map")
    for split, expected in identities.items():
        registry = sources.get(f"datasets/{split}.parquet")
        if not isinstance(registry, dict):
            raise ValueError(f"portable input fixture lacks {split}")
        original = registry.get("original_identity")
        if not isinstance(original, dict) or (
            registry.get("sha256") != expected["sha256"]
            or original.get("bytes") != expected["bytes"]
            or original.get("footer_bytes") != expected["footer_bytes"]
            or original.get("footer_sha256") != expected["footer_sha256"]
        ):
            raise ValueError(f"portable input fixture differs for {split}")
        if "dataset_ref" in registry:
            raise ValueError("fixture changed: the unsupported dataset_ref field is present")
    if portable.get("split_manifest_sha256") != FOLD_MANIFEST_SHA256:
        raise ValueError("portable input fixture split identity differs")

    mount_fixture = json.loads(V7_MOUNT_FIXTURE.read_text())
    raw_mounts = [
        item
        for item in mount_fixture.get("dataset_mounts", [])
        if item.get("owner") == "vladkochetov07" and item.get("dataset") == "wunder-funder"
    ]
    if len(raw_mounts) != 1 or raw_mounts[0].get("path") != (
        "/kaggle/input/datasets/vladkochetov07/wunder-funder"
    ):
        raise ValueError("actual V7 mount receipt does not bind the requested raw-data slug")

    code = build_probe()
    metadata = {
        "code_file": "probe.py",
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
            "vladkochetov07/wnn-c20261004-fold0-assets",
            RAW_DATASET_REF,
        ],
        "enable_gpu": False,
        "enable_internet": False,
        "enable_tpu": True,
        "id": PACKET_ID,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "TpuV5E8",
        "title": PACKET_TITLE,
    }
    metadata_bytes = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    PACKET.mkdir(parents=True, exist_ok=True)
    (PACKET / "probe.py").write_bytes(code)
    (PACKET / "kernel-metadata.json").write_bytes(metadata_bytes)
    bundle_path = ROOT / "packet.bundle.zip"
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in (
            ("kernel-metadata.json", metadata_bytes),
            ("probe.py", code),
        ):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(
                member,
                content,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "packet_title": PACKET_TITLE,
        "packet_code_file": "probe.py",
        "source_only": True,
        "provider_push_authorized": False,
        "code_bytes": len(code),
        "code_sha256": sha256_bytes(code),
        "metadata_sha256": sha256_bytes(metadata_bytes),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256_file(bundle_path),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "base_v7_probe_sha256": V7_PROBE_SHA256,
        "base_v7_manifest_sha256": V7_MANIFEST_SHA256,
        "v7_terminal_receipt_sha256": V7_TERMINAL_SHA256,
        "actual_v7_mount_fixture_sha256": V7_MOUNT_FIXTURE_SHA256,
        "portable_inputs_sha256": PORTABLE_INPUTS_SHA256,
        "assets_manifest_sha256": ASSET_MANIFEST_SHA256,
        "fold_manifest_sha256": FOLD_MANIFEST_SHA256,
        "raw_data": {
            "dataset_ref": RAW_DATASET_REF,
            "mount_path": "/kaggle/input/datasets/vladkochetov07/wunder-funder",
            "identities": identities,
        },
        "repair": {
            "classification": "actual-data provenance admission repair and export-completion successor",
            "portable_registry_dataset_ref_check": "removed; field is absent from the pinned registry",
            "retained_registry_checks": [
                "portable registry whole-file SHA-256",
                "registry version and fold split-manifest SHA-256",
                "per-split whole-file SHA-256",
                "original byte count",
                "original Parquet footer length and SHA-256",
            ],
            "mounted_file_checks": [
                "exact owner/slug mount path from actual V7 mount receipt",
                "unique paired train/valid mount by exact byte lengths",
                "mounted file byte lengths",
                "Parquet trailer magic, footer length, and footer SHA-256",
                "full mounted file SHA-256",
            ],
            "scientific_recipe": "preserved V7 E463 one-update bridge recipe; no loss, membership, optimizer, or target change",
            "prediction_export": "one complete 20,000-row fit-VALID sequence using updated EMA after exact checkpoint reload; no score",
        },
        "execution": {
            "machine_shape": "TpuV5E8",
            "gpu": False,
            "internet": False,
            "global_batch": 16,
            "local_batch_per_tpu": 2,
            "chunk_rows": 2000,
            "model_dimensions": [211, 256, 2, 23],
            "real_optimizer_updates": 1,
            "synthetic_profile_updates": 4,
            "prediction_export_rows": 20000,
            "prediction_export_groups": 1,
            "raw_parquet_full_hash_bytes": 34855728139,
            "raw_parquet_footer_hash_bytes": 71124227,
            "raw_parquet_copy_bytes": 0,
            "training_groups": 10535,
            "fit_valid_groups": 1405,
            "evaluation_groups": 468,
            "selected_probe_split": "valid-fit only; exposed",
            "scoring": False,
            "heldout_claim": None,
            "parent_timeout_seconds": 1650,
            "worker_timeout_seconds": 1500,
            "minimum_host_free_memory_fraction": 0.20,
            "maximum_tpu_peak_hbm_fraction": 0.80,
            "resource_measurement_status": "unmeasured on TPU: V7 failed in host admission before device import",
            "resource_admission": "record actual host RSS, TPU HBM before/peak/after, and phase timings; stop the child at 1500 seconds",
        },
        "checkpoint_policy": {
            "parent": "pinned c20261003_combo_s3_f0 epoch-3 evaluation EMA",
            "recovery": "source checkpoint is immutable; one-step diagnostic is not resumed; collect any partial output after terminal timeout",
            "updated_exports": [
                "student FP32 NPZ",
                "EMA FP32 NPZ",
                "AdamW moments and update count NPZ",
                "one-group updated-EMA prediction NPZ",
            ],
        },
        "collection_policy": {
            "maximum_expected_download_bytes": 100000000,
            "remove_temporary_input_symlink_tree_before_worker_terminal": True,
            "source_and_raw_data_copied_to_outputs": False,
            "collect_only_after_terminal_status": True,
            "required_outputs": [
                "startup.json",
                "host_start.json",
                "input_mounts.json",
                "source_materialization.json",
                "e463_checkpoint_identity.json",
                "terminal.json",
                "complete.json",
                "result.json",
                "device_worker_result.json",
                "device_worker_terminal.json",
                "worker_launch.json",
                "worker_process.json",
                "realdata_input_identity.json",
                "realdata_update_receipt.json",
                "tpu_diagnostic_complete.json",
                "tpu_inventory.json",
                "tpu_hbm_before.json",
                "tpu_hbm_after.json",
                "tpu_fullshape_profile.json",
                "tpu_objective_reduction_parity.json",
                "parent_phase.json",
                "device_worker_phase.json",
                "realdata_reload/realdata_update_receipt.json",
                "realdata_reload/model_after_one_chunk_fp32.npz",
                "realdata_reload/ema_after_one_chunk_fp32.npz",
                "realdata_reload/adamw_after_one_chunk.npz",
                "realdata_reload/updated_ema_valid_fit_group_prediction.npz",
                "realdata_staging_cleanup.json",
                "tpu_worker.log",
            ],
        },
    }
    manifest["test_sha256"] = sha256_file(ROOT / "tests/test_v8_packet.py")
    manifest["readme_sha256"] = sha256_file(ROOT / "README.md")
    (ROOT / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "packet_id": result["packet_id"],
                "code_bytes": result["code_bytes"],
                "code_sha256": result["code_sha256"],
                "bundle_bytes": result["bundle_bytes"],
                "bundle_sha256": result["bundle_sha256"],
                "provider_push_authorized": result["provider_push_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
