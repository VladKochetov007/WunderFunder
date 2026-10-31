"""Verify an exact final ZIP offline against its receipt and one full official sequence.

The check imports the delivered callback under Python 3.11, checks its packaged
native ABI and byte manifests, compares the W18-only callback with the receipt-
bound PyTorch W18 reference on one causal 20,000-row sequence, and measures a
single-core callback pass. It computes no task score and makes no independent
evaluation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import utils as _callback  # noqa: E402
from scripts.package_final_tl3 import (  # noqa: E402
    BLEND,
    FINAL_RECEIPT_SCHEMA,
    FINAL_RUNS,
    ROLES,
    _validate_final_checkpoint,
    load_final_model_receipt,
)
from wnn.data import SEQUENCE_LENGTH, SequenceFile  # noqa: E402

FINAL_SCOPE = "approved_full_corpus_final_model_set"
MAX_ZIP_BYTES = 20_000_000
OFFICIAL_SEQUENCES = 1_970
OFFICIAL_SECONDS = 3_600.0
MEMORY_LIMIT_BYTES = 16 * (1 << 30)


@dataclass(frozen=True)
class DataPoint:
    """Minimal object implementing the delivered scorer callback contract."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Reject traversal, links, duplicate names, and unexpected ZIP encodings."""
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        relative = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or name in result
            or (mode & 0o170000) == 0o120000
            or info.flag_bits & 0x1
        ):
            raise ValueError(f"unsafe or duplicate final ZIP member: {name!r}")
        result[name] = info
    return result


def _extract_checked(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], root: Path
) -> None:
    """Extract only already-validated regular ZIP files without using extractall."""
    for name, info in members.items():
        if info.is_dir():
            continue
        target = root.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)


def _check_member_digests(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], expected: object
) -> dict[str, str]:
    if not isinstance(expected, dict) or set(expected) != set(members):
        raise ValueError("build receipt ZIP member list differs from the delivered artifact")
    actual: dict[str, str] = {}
    for name, info in members.items():
        digest = hashlib.sha256(archive.read(info)).hexdigest()
        if expected.get(name) != digest:
            raise ValueError(f"delivered ZIP member hash differs from its build receipt: {name}")
        actual[name] = digest
    return actual


def _verify_checkpoint_receipt(
    checkpoint_paths: dict[str, Path], receipt_path: Path, receipt_sha256: str
) -> tuple[dict[str, Any], Any]:
    """Verify exact receipt bytes and each actual final checkpoint's source identity."""
    checked = load_final_model_receipt(receipt_path, receipt_sha256)
    receipt = checked["receipt"]
    membership = checked["membership"]
    if set(checkpoint_paths) != set(ROLES):
        raise ValueError("checker requires all four receipt-bound final checkpoint paths")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    members = receipt["members"]
    for role in ROLES:
        path = checkpoint_paths[role]
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{role} checkpoint must be a regular file: {path}")
        member = members[role]
        if _sha256_file(path) != member["checkpoint_sha256"]:
            raise ValueError(f"{role} checkpoint bytes differ from the final-model receipt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{role} checkpoint is not a mapping")
        _validate_final_checkpoint(role, checkpoint, member, membership)
    return checked, torch


def _torch_w18_reference(checkpoint_path: Path, raw: np.ndarray, torch: Any) -> np.ndarray:
    """Run the receipt-bound FP32 W18 model on the official raw112 sequence."""
    from wnn.features import event_features_torch
    from wnn.nn import RNNModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    args = checkpoint["args"]
    model = RNNModel(
        n_in=211,
        hidden=args["hidden"],
        layers=args["layers"],
        dropout=args["dropout"],
        n_out=state["head.2.weight"].shape[0],
        cell=args["cell"],
        enc=args["enc"],
        readout_activation=args["readout_activation"],
        numeric_map=args["numeric_map"],
        ple_intervals=args["ple_intervals"],
        head_layers=args["head_layers"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw[None], dtype=np.float32))
    with torch.inference_mode():
        events = event_features_torch(raw_tensor)
        outputs = model(torch.cat((raw_tensor, events), dim=-1))[0][0, :, :2]
    return np.ascontiguousarray(outputs.cpu().numpy(), dtype=np.float32)


def _run_package_callback(model: Any, sequence_id: int, raw: np.ndarray, warmup: int) -> np.ndarray:
    predictions = np.full((len(raw), 2), np.nan, dtype=np.float32)
    for step, row in enumerate(raw):
        value = model.predict(DataPoint(sequence_id, step, step >= warmup, row))
        if step < warmup:
            if value is not None:
                raise AssertionError(f"delivered callback emitted during warm-up row {step}")
            continue
        if value is None:
            raise AssertionError(f"delivered callback omitted required output at row {step}")
        output = np.asarray(value, dtype=np.float32)
        if output.shape != (2,) or not np.isfinite(output).all():
            raise AssertionError(f"delivered callback returned invalid output at row {step}")
        predictions[step] = output
    return predictions


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. The official target is Linux.
        return peak * 1024 if sys.platform.startswith("linux") else peak
    except (ImportError, OSError, ValueError):
        return None


def _pin_one_cpu() -> int | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("process has no available CPU for the one-core check")
    selected = available[0]
    os.sched_setaffinity(0, {selected})
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"exact package ABI check requires Python 3.11, found {sys.version}")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("exact submission ABI and one-core check requires Linux")
    if args.sequence_index < 0:
        raise ValueError("sequence index must be nonnegative")
    if args.source not in ("train", "valid"):
        raise ValueError("source must be train or valid")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("parity tolerances must be nonnegative")
    if args.zip_path.is_symlink() or not args.zip_path.is_file():
        raise FileNotFoundError(f"submission ZIP must be a regular file: {args.zip_path}")
    zip_bytes = args.zip_path.stat().st_size
    if zip_bytes > MAX_ZIP_BYTES:
        raise ValueError(f"submission ZIP is {zip_bytes} bytes; limit is {MAX_ZIP_BYTES}")
    artifact_sha256 = _sha256_file(args.zip_path)

    checked_receipt, torch = _verify_checkpoint_receipt(
        {
            "w18": args.w18,
            "tiedlead_s5": args.tiedlead_s5,
            "tiedlead_s6": args.tiedlead_s6,
            "source4_lstm": args.lstm,
        },
        args.final_model_receipt,
        args.final_model_receipt_sha256,
    )
    receipt = checked_receipt["receipt"]
    membership = checked_receipt["membership"]

    build_receipt = json.loads(args.build_receipt.read_text(encoding="utf-8"))
    if build_receipt.get("artifact", {}).get("sha256") != artifact_sha256:
        raise ValueError("build receipt artifact SHA differs from the supplied ZIP")
    if build_receipt.get("artifact", {}).get("bytes") != zip_bytes:
        raise ValueError("build receipt artifact size differs from the supplied ZIP")
    if (
        build_receipt.get("serving_policy") != "w18_only"
        or build_receipt.get("active_member_indices") != [0]
        or build_receipt.get("precision") != "q8vnni"
        or build_receipt.get("fast_activations") is not True
        or build_receipt.get("candidate_blend") != BLEND
        or build_receipt.get("applied_blend") is not None
    ):
        raise ValueError("build receipt does not identify the actual W18-only callback policy")

    package_root = Path(tempfile.mkdtemp(prefix="final-tl3-package-check-"))
    try:
        with zipfile.ZipFile(args.zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"submission ZIP CRC failed at {bad_member}")
            zip_members = _safe_zip_members(archive)
            member_hashes = _check_member_digests(
                archive, zip_members, build_receipt.get("artifact", {}).get("members")
            )
            _extract_checked(archive, zip_members, package_root)

        manifest = json.loads((package_root / "members.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "wnn.final-tl3-lstm-members.v1"
            or manifest.get("candidate_scope") != FINAL_SCOPE
            or manifest.get("serving_policy") != "w18_only"
            or manifest.get("precision") != "q8vnni"
            or manifest.get("fast_activations") is not True
            or manifest.get("active_member_indices") != [0]
            or manifest.get("candidate_blend") != BLEND
            or manifest.get("blend") is not None
            or manifest.get("independent_evaluation_sequences") != 0
            or manifest.get("independent_score_claim") != "forbidden"
            or manifest.get("final_fit") != receipt.get("final_fit")
            or manifest.get("final_model_receipt", {}).get("schema") != FINAL_RECEIPT_SCHEMA
            or manifest.get("final_model_receipt", {}).get("sha256") != checked_receipt["sha256"]
        ):
            raise ValueError("delivered ZIP metadata is not the approved W18-only final policy")
        manifest_members = manifest.get("members")
        if not isinstance(manifest_members, list) or len(manifest_members) != 4:
            raise ValueError("delivered final ZIP must retain all four receipt-bound members")
        for role, record in zip(ROLES, manifest_members, strict=True):
            expected = receipt["members"][role]
            if (
                record.get("role") != role
                or record.get("run_reference") != FINAL_RUNS[ROLES.index(role)]
                or record.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            ):
                raise ValueError(f"delivered ZIP has a changed {role} receipt binding")
        if (package_root / "precision.txt").read_text(encoding="utf-8") != "q8vnni\n":
            raise ValueError("delivered precision marker differs from the serving policy")
        native_paths = [
            package_root / level / "wnn_native.abi3.so" for level in ("v2", "avx2", "v3")
        ]
        if any(not path.is_file() for path in native_paths):
            raise ValueError("delivered ZIP lacks one or more Python ABI3 Linux libraries")
        selected_cpu = _pin_one_cpu()

        source_sha = membership.train_sha256 if args.source == "train" else membership.valid_sha256
        if (
            args.data.is_symlink()
            or not args.data.is_file()
            or _sha256_file(args.data) != source_sha
        ):
            raise ValueError("sequence data bytes differ from the receipt-bound official source")
        sequences = SequenceFile(args.data)
        if args.sequence_index >= len(sequences):
            raise ValueError("sequence index lies outside the chosen official source")
        sequence = sequences.read(args.sequence_index, targets=False, mask=False)
        raw = sequence.features
        if raw.shape != (SEQUENCE_LENGTH, 112) or not raw.flags.c_contiguous:
            raise ValueError("chosen row group is not one canonical 20,000-row raw112 sequence")
        if not np.isfinite(raw).all():
            raise ValueError("chosen official raw112 sequence contains nonfinite values")
        row_contract = pq.ParquetFile(args.data).read_row_group(
            args.sequence_index,
            columns=["seq_ix", "step_in_seq"],
            use_threads=False,
        )
        row_sequence_ids = row_contract.column("seq_ix").to_numpy()
        row_steps = row_contract.column("step_in_seq").to_numpy()
        if (
            len(row_sequence_ids) != SEQUENCE_LENGTH
            or not np.all(row_sequence_ids == sequence.seq_ix)
            or not np.array_equal(row_steps, np.arange(SEQUENCE_LENGTH))
        ):
            raise ValueError("chosen row group violates the canonical seq_ix/step order")

        module_spec = importlib.util.spec_from_file_location(
            "delivered_final_solution", package_root / "solution.py"
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("could not load exact delivered solution.py")
        delivered = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = delivered
        module_spec.loader.exec_module(delivered)
        model = delivered.PredictionModel()
        native = model.model
        if (
            native.serving_policy != "w18_only"
            or tuple(native.active_member_indices) != (0,)
            or not native.fast_activations
            or not all(name.startswith("q8vnni") for name in native.effective_precisions)
        ):
            raise RuntimeError("delivered callback runtime differs from its declared W18-only mode")

        w18_path = args.w18.resolve(strict=True)
        reference = _torch_w18_reference(w18_path, raw, torch)
        warmup = int(_callback.WARMUP)
        started = time.perf_counter()
        first = _run_package_callback(model, sequence.seq_ix, raw, warmup)
        elapsed = time.perf_counter() - started
        scored_rows = slice(warmup, None)
        error = np.abs(first[scored_rows] - reference[scored_rows])
        max_abs = float(error.max(initial=0.0))
        relative = error / np.maximum(np.abs(reference[scored_rows]), np.float32(1e-6))
        max_relative = float(relative.max(initial=0.0))
        if not np.allclose(
            first[scored_rows], reference[scored_rows], atol=args.atol, rtol=args.rtol
        ):
            raise AssertionError(
                f"delivered Q8 W18 differs from receipt-bound FP32 PyTorch reference: "
                f"max_abs={max_abs:.8g}, max_rel={max_relative:.8g}"
            )

        native.reset()
        replay = _run_package_callback(model, sequence.seq_ix, raw, warmup)
        replay_identical = np.array_equal(first, replay, equal_nan=True)
        if not replay_identical:
            raise AssertionError("delivered callback reset/replay changed predictions")

        prefix = min(1_000, SEQUENCE_LENGTH - 1)
        changed = raw.copy()
        changed[prefix:, 0] += np.float32(0.25)
        native.reset()
        future = _run_package_callback(model, sequence.seq_ix, changed, warmup)
        future_prefix_invariant = np.array_equal(first[:prefix], future[:prefix], equal_nan=True)
        if not future_prefix_invariant:
            raise AssertionError("future rows changed an already emitted prefix")

        peak_rss = _peak_rss_bytes()
        row_microseconds = elapsed * 1e6 / SEQUENCE_LENGTH
        projected_seconds = row_microseconds * OFFICIAL_SEQUENCES * SEQUENCE_LENGTH / 1e6
        runtime_pass = projected_seconds <= OFFICIAL_SECONDS
        memory_pass = peak_rss is not None and peak_rss <= MEMORY_LIMIT_BYTES
        return {
            "schema": "wnn.final-tl3-package-serving-check.v1",
            "artifact_sha256": artifact_sha256,
            "artifact_bytes": zip_bytes,
            "artifact_under_20mb": zip_bytes <= MAX_ZIP_BYTES,
            "build_receipt_sha256": _sha256_file(args.build_receipt),
            "final_model_receipt_sha256": checked_receipt["sha256"],
            "python": platform.python_version(),
            "native_abi": "linux-cp37-abi3-compatible-imported-under-py311",
            "serving_policy": native.serving_policy,
            "precision_requested": manifest["precision"],
            "effective_precisions": list(native.effective_precisions),
            "fast_activations": native.fast_activations,
            "active_member_indices": list(native.active_member_indices),
            "sequence_source": args.source,
            "sequence_index": args.sequence_index,
            "sequence_id": sequence.seq_ix,
            "sequence_rows": len(raw),
            "warmup_rows": warmup,
            "torch_device": "cpu",
            "parity_tolerance": {"atol": args.atol, "rtol": args.rtol},
            "max_absolute_error_vs_torch_w18": max_abs,
            "max_relative_error_vs_torch_w18": max_relative,
            "reset_replay_identical": replay_identical,
            "future_prefix_invariant": future_prefix_invariant,
            "timed_callback_seconds_one_cpu": elapsed,
            "timed_callback_microseconds_per_row": row_microseconds,
            "official_projected_seconds_1970_sequences": projected_seconds,
            "official_time_limit_seconds": OFFICIAL_SECONDS,
            "official_time_projection_pass": runtime_pass,
            "peak_rss_bytes": peak_rss,
            "official_memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "official_memory_projection_pass": memory_pass,
            "eligibility_gates_pass": bool(
                zip_bytes <= MAX_ZIP_BYTES and runtime_pass and memory_pass
            ),
            "pinned_cpu": selected_cpu,
            "zip_member_sha256": member_hashes,
            "score_computed": False,
            "independent_score_claim": "forbidden",
        }
    finally:
        shutil.rmtree(package_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--final-model-receipt", type=Path, required=True)
    parser.add_argument("--final-model-receipt-sha256", required=True)
    parser.add_argument("--w18", type=Path, required=True)
    parser.add_argument("--tiedlead-s5", dest="tiedlead_s5", type=Path, required=True)
    parser.add_argument("--tiedlead-s6", dest="tiedlead_s6", type=Path, required=True)
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", choices=("train", "valid"), default="train")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if not result["eligibility_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
