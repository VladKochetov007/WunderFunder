"""Check a delivered W18 ZIP offline, against R11 and the Torch model on train row group 0.

This checker computes no task score. It validates the one-member receipt, package ABI and
size, R11-to-R12 old-reference parity, causal reset/future-prefix behavior, new-weight numeric
ceilings, and a one-core runtime projection.
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
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.check_final_tl3_package import (  # noqa: E402
    EXPECTED_R11_NATIVE_V3_SHA256,
    MAX_ZIP_BYTES,
    OFFICIAL_SECONDS,
    DataPoint,
    _check_member_digests,
    _extract_checked,
    _numeric_error_summary,
    _numeric_error_within_ceilings,
    _peak_rss_bytes,
    _pin_one_cpu,
    _safe_zip_members,
    _sha256_file,
    _torch_w18_reference,
    _validate_quantization_contract,
)
from scripts.check_final_tl3_synthetic import _load_reference_models  # noqa: E402
from scripts.package_final_tl3 import (  # noqa: E402
    _validate_final_checkpoint as _validate_checkpoint,
)
from scripts.package_final_w18 import (  # noqa: E402
    W18_PACKAGE_SCHEMA,
    W18_RECEIPT_SCHEMA,
    validate_final_w18_receipt,
)
from wnn.data import SEQUENCE_LENGTH, SequenceFile  # noqa: E402
from wnn.final_submit_manifest import FinalFitMembership  # noqa: E402

W18_RUN = "tl3final_w18final_distilled_s5"
OFFICIAL_SEQUENCES = 1_970
MEMORY_LIMIT_BYTES = 16 * (1 << 30)
WARMUP_ROWS = 99


def _load_native(path: Path) -> Any:
    """Import one abi3 extension under its canonical module name."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"native extension is missing or unsafe: {path}")
    sys.modules.pop("wnn_native", None)
    spec = importlib.util.spec_from_file_location("wnn_native", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native abi3 extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wnn_native"] = module
    spec.loader.exec_module(module)
    return module


def _load_solution(path: Path) -> tuple[Any, Any]:
    """Import the exact delivered callback and retain its selected native module."""
    sys.modules.pop("wnn_native", None)
    spec = importlib.util.spec_from_file_location("delivered_final_w18_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the delivered W18 callback")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native = sys.modules["wnn_native"]
    return module.PredictionModel(), native


def _raw_outputs(predictor: Any, sequence_id: int, raw: np.ndarray) -> np.ndarray:
    outputs = np.empty((len(raw), 2), dtype=np.float32)
    for step, row in enumerate(raw):
        value = predictor.predict(DataPoint(sequence_id, step, True, row))
        if value is None:
            raise AssertionError(f"W18 predictor omitted row {step}")
        output = np.asarray(value, dtype=np.float32)
        if output.shape != (2,) or not np.isfinite(output).all():
            raise AssertionError(f"W18 predictor returned an invalid row {step}")
        outputs[step] = output
    return outputs


def _package_outputs(
    model: Any, sequence_id: int, raw: np.ndarray, warmup: int = WARMUP_ROWS
) -> np.ndarray:
    outputs = np.full((len(raw), 2), np.nan, dtype=np.float32)
    for step, row in enumerate(raw):
        value = model.predict(DataPoint(sequence_id, step, step >= warmup, row))
        if step < warmup:
            if value is not None:
                raise AssertionError(f"delivered callback emitted during warm-up row {step}")
            continue
        if value is None:
            raise AssertionError(f"delivered callback omitted row {step}")
        output = np.asarray(value, dtype=np.float32)
        if output.shape != (2,) or not np.isfinite(output).all():
            raise AssertionError(f"delivered callback returned an invalid row {step}")
        outputs[step] = output
    return outputs


def _membership_from_receipt(receipt: dict[str, object], path: Path) -> FinalFitMembership:
    fit = receipt["final_fit"]
    sources = fit["source_sha256"]
    groups = fit["fit_row_groups"]
    return FinalFitMembership(
        path=path.resolve(),
        manifest_sha256=fit["manifest_sha256"],
        fit_members_sha256=fit["fit_members_sha256"],
        train_groups=tuple(groups["train"]),
        valid_groups=tuple(groups["valid"]),
        train_sha256=sources["train"],
        valid_sha256=sources["valid"],
        source_validation_sha256=fit["source_validation_sha256"],
    )


def _load_receipt(path: Path) -> tuple[dict[str, object], bytes, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("single-W18 receipt must be a bounded regular file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    receipt = validate_final_w18_receipt(json.loads(raw), digest)
    return receipt, raw, digest


def _validate_final_checkpoint(
    checkpoint_path: Path,
    receipt: dict[str, object],
    membership: FinalFitMembership,
) -> None:
    import torch

    member = receipt["members"]["w18"]
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("final W18 checkpoint must be a regular file")
    if _sha256_file(checkpoint_path) != member["checkpoint_sha256"]:
        raise ValueError("final W18 checkpoint differs from the single-member receipt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("final W18 checkpoint envelope is not a mapping")
    _validate_checkpoint("w18", checkpoint, member, membership)


def _verify_package(
    zip_path: Path,
    build_receipt_path: Path,
    model_receipt_path: Path,
    fit_process_receipt_path: Path,
    package_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, str], str]:
    if zip_path.is_symlink() or not zip_path.is_file():
        raise FileNotFoundError("W18 submission ZIP must be a regular file")
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("W18 submission exceeds the 20 MB ZIP limit")
    build = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(build, dict) or build.get("schema") != "wnn.final-w18-package-build-receipt.v1":
        raise ValueError("unsupported W18 package build receipt")
    artifact_sha = _sha256_file(zip_path)
    artifact = build.get("artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("sha256") != artifact_sha
        or artifact.get("bytes") != zip_path.stat().st_size
        or build.get("candidate_scope") != "approved_full_corpus_final_w18"
        or build.get("serving_policy") != "w18_only"
        or build.get("active_member_indices") != [0]
        or build.get("independent_evaluation_sequences") != 0
        or build.get("independent_score_claim") != "forbidden"
    ):
        raise ValueError("build receipt does not bind this W18-only no-score artifact")
    receipt, receipt_bytes, receipt_sha = _load_receipt(model_receipt_path)
    if (
        fit_process_receipt_path.is_symlink()
        or not fit_process_receipt_path.is_file()
        or fit_process_receipt_path.stat().st_size > 2_000_000
    ):
        raise ValueError("W18 fit-process receipt must be a regular file")
    fit_process_bytes = fit_process_receipt_path.read_bytes()
    fit_process_sha = hashlib.sha256(fit_process_bytes).hexdigest()
    fit_process_ref = build.get("fit_process_receipt")
    if (
        not isinstance(fit_process_ref, dict)
        or fit_process_ref.get("path") != str(fit_process_receipt_path.resolve())
        or fit_process_ref.get("sha256") != fit_process_sha
        or fit_process_ref.get("command_sha256")
        != receipt["fit_process_receipt"]["record"].get("command_sha256")
    ):
        raise ValueError("build receipt does not bind the completed W18 fit-process receipt")
    if json.loads(fit_process_bytes) != receipt["fit_process_receipt"]["record"]:
        raise ValueError("fit-process receipt JSON differs from the final model receipt record")
    receipt_ref = build.get("final_w18_receipt")
    if (
        not isinstance(receipt_ref, dict)
        or receipt_ref.get("path") != str(model_receipt_path.resolve())
        or receipt_ref.get("schema") != W18_RECEIPT_SCHEMA
        or receipt_ref.get("sha256") != receipt_sha
    ):
        raise ValueError("build receipt does not bind the supplied single-W18 receipt")

    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"W18 ZIP CRC failed at {bad}")
        members = _safe_zip_members(archive)
        expected_names = {
            "solution.py",
            "members.json",
            "final_w18_receipt.json",
            "w18_fit_process_receipt.json",
            "weights_w18.npz",
            *(f"{level}/wnn_native.abi3.so" for level in ("v2", "avx2", "v3")),
        }
        if set(members) != expected_names:
            raise ValueError("W18 ZIP has missing or extra members")
        hashes = _check_member_digests(archive, members, artifact.get("members"))
        if hashlib.sha256(archive.read("final_w18_receipt.json")).digest() != hashlib.sha256(
            receipt_bytes
        ).digest():
            raise ValueError("ZIP receipt bytes differ from the separate receipt file")
        if hashlib.sha256(archive.read("w18_fit_process_receipt.json")).hexdigest() != fit_process_sha:
            raise ValueError("ZIP fit-process receipt differs from the supplied child receipt")
        _extract_checked(archive, members, package_root)

    manifest = json.loads((package_root / "members.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != W18_PACKAGE_SCHEMA
        or manifest.get("candidate_scope") != "approved_full_corpus_final_w18"
        or manifest.get("serving_policy") != "w18_only"
        or manifest.get("precision") != "q8vnni"
        or manifest.get("fast_activations") is not True
        or manifest.get("active_member_indices") != [0]
        or manifest.get("independent_evaluation_sequences") != 0
        or manifest.get("independent_score_claim") != "forbidden"
        or manifest.get("callback_output", {}).get("role") != "w18"
        or manifest.get("final_model_receipt", {}).get("sha256") != receipt_sha
        or manifest.get("fit_process_receipt")
        != {"path": "w18_fit_process_receipt.json", "sha256": fit_process_sha}
    ):
        raise ValueError("delivered ZIP manifest differs from the W18-only receipt policy")
    member_list = manifest.get("members")
    if (
        not isinstance(member_list, list)
        or len(member_list) != 1
        or member_list[0].get("role") != "w18"
        or member_list[0].get("run_reference") != W18_RUN
        or member_list[0].get("checkpoint_sha256")
        != receipt["members"]["w18"]["checkpoint_sha256"]
    ):
        raise ValueError("delivered ZIP contains an inactive or mismatched final member")
    if build.get("checkpoint_sha256") != receipt["members"]["w18"]["checkpoint_sha256"]:
        raise ValueError("build receipt checkpoint differs from the W18 receipt")
    if build.get("lineage_closure_sha256") != receipt["lineage_closure"]["sha256"]:
        raise ValueError("build receipt lineage digest differs from the W18 receipt")
    for relative, digest in manifest.get("native_libraries", {}).items():
        if hashes.get(relative) != digest:
            raise ValueError(f"native library differs from package manifest: {relative}")
    return build, receipt, hashes, artifact_sha


def run(args: argparse.Namespace) -> dict[str, object]:
    if sys.version_info[:2] != (3, 11) or not sys.platform.startswith("linux"):
        raise RuntimeError("W18 submission check requires Linux Python 3.11")
    if args.sequence_index != 0:
        raise ValueError("serving parity is fixed to official train row group 0")
    package_root = Path(tempfile.mkdtemp(prefix="final-w18-package-check-"))
    try:
        build, receipt, zip_hashes, artifact_sha = _verify_package(
            args.zip_path,
            args.build_receipt,
            args.model_receipt,
            args.fit_process_receipt,
            package_root,
        )
        membership = _membership_from_receipt(receipt, args.model_receipt)
        _validate_final_checkpoint(args.checkpoint, receipt, membership)
        quantization = _validate_quantization_contract(
            args.quantization_contract,
            args.quantization_contract_sha256,
            valid_sha256=membership.valid_sha256,
            source_validation_sha256=membership.source_validation_sha256,
        )
        if args.train_data.is_symlink() or not args.train_data.is_file():
            raise ValueError("official train parquet must be a regular file")
        if _sha256_file(args.train_data) != membership.train_sha256:
            raise ValueError("train data bytes differ from the W18 receipt source hash")
        sequence_file = SequenceFile(args.train_data)
        sequence = sequence_file.read(args.sequence_index, targets=False, mask=False)
        raw = np.ascontiguousarray(sequence.features, dtype=np.float32)
        if raw.shape != (SEQUENCE_LENGTH, 112) or not np.isfinite(raw).all():
            raise ValueError("official train row group is not a finite raw112 sequence")
        sequence_order = pq.ParquetFile(args.train_data).read_row_group(
            args.sequence_index,
            columns=["seq_ix", "step_in_seq"],
            use_threads=False,
        )
        sequence_ids = sequence_order.column("seq_ix").to_numpy()
        steps = sequence_order.column("step_in_seq").to_numpy()
        if (
            len(sequence_ids) != SEQUENCE_LENGTH
            or not np.all(sequence_ids == sequence.seq_ix)
            or not np.array_equal(steps, np.arange(SEQUENCE_LENGTH))
        ):
            raise ValueError("official train row group differs from canonical sequence step order")
        if sequence.seq_ix != quantization["calibration_sequence_id"]:
            raise ValueError("train row group 0 differs from the fixed numeric-calibration sequence")

        # Check the old reference against the shipped R12 member class on the same real sequence.
        r11_hash = _sha256_file(args.r11_native)
        if r11_hash != EXPECTED_R11_NATIVE_V3_SHA256:
            raise ValueError("R11 extension differs from the pinned engineering build")
        _reference_models, reference_weights = _load_reference_models(
            _reference_paths(args.reference_checkpoints_root)
        )
        r11 = _load_native(args.r11_native)
        old_mix = r11.FinalTl3Predictor(
            reference_weights,
            precision="q8vnni",
            fast_activations=True,
            serving_policy="w18_only",
        )
        old_rows = np.empty((len(raw), 2), dtype=np.float32)
        for step, row in enumerate(raw):
            old_rows[step] = np.asarray(
                old_mix.step_members(DataPoint(sequence.seq_ix, step, True, row)),
                dtype=np.float32,
            )[0]
        r11.reset()

        delivered, delivered_native = _load_solution(package_root / "solution.py")
        loaded_native_path = Path(delivered_native.__file__).resolve()
        expected_native_paths = {
            (package_root / level / "wnn_native.abi3.so").resolve()
            for level in ("v2", "avx2", "v3")
        }
        if loaded_native_path not in expected_native_paths:
            raise RuntimeError("delivered callback imported a native library outside its ZIP")
        loaded_native_relative = loaded_native_path.relative_to(package_root).as_posix()
        if _sha256_file(loaded_native_path) != zip_hashes[loaded_native_relative]:
            raise RuntimeError("delivered callback native bytes differ from the verified ZIP")
        r12_reference = delivered_native.FinalW18Predictor(
            reference_weights[0], precision="q8vnni", fast_activations=True
        )
        r12_rows = _raw_outputs(r12_reference, sequence.seq_ix, raw)
        r11_r12_exact = bool(np.array_equal(old_rows, r12_rows))
        r12_reference.reset()
        replay_rows = _raw_outputs(r12_reference, sequence.seq_ix + 1, raw)
        reset_replay_identical = bool(np.array_equal(r12_rows, replay_rows))

        prefix = min(1_000, len(raw) - 1)
        changed = raw.copy()
        changed[prefix:, 0] += np.float32(0.25)
        changed[prefix:, 11] -= np.float32(0.5)
        r12_reference.reset()
        changed_rows = _raw_outputs(r12_reference, sequence.seq_ix + 2, changed)
        future_prefix_invariant = bool(np.array_equal(r12_rows[:prefix], changed_rows[:prefix]))

        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch_reference = _torch_w18_reference(args.checkpoint, raw, torch)
        if (
            tuple(delivered.model.member_widths) != (211,)
            or tuple(delivered.model.active_member_indices) != (0,)
            or not str(delivered.model.precision).startswith("q8vnni")
            or delivered.model.fast_activations is not True
        ):
            raise RuntimeError("delivered callback runtime differs from the W18-only package")

        selected_cpu = _pin_one_cpu()
        callback_times: list[float] = []
        predictions = np.empty((len(raw), 2), dtype=np.float32)
        replay_identical = False
        future_invariant = False
        for repeat in range(3):
            delivered_native_model = delivered.model
            delivered_native_model.reset()
            started = time.perf_counter()
            current = _package_outputs(delivered, sequence.seq_ix + 10 + repeat, raw)
            callback_times.append(time.perf_counter() - started)
            predictions = current
            if repeat == 0:
                delivered_native_model.reset()
                replay = _package_outputs(delivered, sequence.seq_ix + 20, raw)
                replay_identical = bool(np.array_equal(current, replay, equal_nan=True))
                future = _package_outputs(delivered, sequence.seq_ix + 21, changed)
                future_invariant = bool(
                    np.array_equal(current[:prefix], future[:prefix], equal_nan=True)
                )

        scored = slice(WARMUP_ROWS, None)
        ceilings = quantization["ceilings"]
        numeric = _numeric_error_summary(
            predictions[scored],
            torch_reference[scored],
            atol=float(ceilings["original_allclose_atol"]),
            rtol=float(ceilings["original_allclose_rtol"]),
        )
        numeric_pass = _numeric_error_within_ceilings(numeric, ceilings)
        median_seconds = float(np.median(callback_times))
        us_per_row = median_seconds * 1e6 / len(raw)
        projected_seconds = median_seconds * OFFICIAL_SEQUENCES
        peak_rss = _peak_rss_bytes()
        runtime_pass = projected_seconds <= OFFICIAL_SECONDS
        memory_pass = peak_rss is not None and peak_rss <= MEMORY_LIMIT_BYTES
        gates = {
            "zip_size": args.zip_path.stat().st_size <= MAX_ZIP_BYTES,
            "old_reference_r11_r12_exact": r11_r12_exact,
            "old_reference_reset_replay": reset_replay_identical,
            "old_reference_future_prefix": future_prefix_invariant,
            "final_weight_numeric_ceilings": bool(numeric_pass),
            "package_reset_replay": replay_identical,
            "package_future_prefix": future_invariant,
            "time_projection": runtime_pass,
            "memory": memory_pass,
            "old_reference_fidelity": bool(quantization["fidelity_passed"]),
        }
        return {
            "schema": "wnn.final-w18-package-serving-check.v1",
            "artifact_sha256": artifact_sha,
            "artifact_bytes": args.zip_path.stat().st_size,
            "build_receipt_sha256": _sha256_file(args.build_receipt),
            "model_receipt_sha256": hashlib.sha256(
                args.model_receipt.read_bytes()
            ).hexdigest(),
            "w18_checkpoint_sha256": receipt["members"]["w18"]["checkpoint_sha256"],
            "lineage_runs": len(receipt["lineage_closure"]["roles"]),
            "active_member_indices": list(delivered.model.active_member_indices),
            "precision": delivered.model.precision,
            "fast_activations": delivered.model.fast_activations,
            "native_v3_sha256": zip_hashes["v3/wnn_native.abi3.so"],
            "r11_native_sha256": r11_hash,
            "old_reference_sequence": {
                "source": "official train row group 0",
                "sequence_id": sequence.seq_ix,
                "rows": len(raw),
                "warmup_rows": WARMUP_ROWS,
                "r11_r12_bit_exact": r11_r12_exact,
                "reset_replay_identical": reset_replay_identical,
                "future_prefix_invariant": future_prefix_invariant,
            },
            "new_weight_numeric_error_vs_cpu_torch": {
                "rms_by_target": numeric["rms"].tolist(),
                "p99_abs_by_target": numeric["p99_abs"].tolist(),
                "max_abs_by_target": numeric["max_abs_by_target"].tolist(),
                "max_abs": numeric["max_abs"],
                "original_allclose_at_0_02_diagnostic": numeric["allclose"],
                "frozen_ceilings_pass": bool(numeric_pass),
                "calibration_contract_sha256": quantization["sha256"],
            },
            "package_runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "selected_cpu": selected_cpu,
                "callback_seconds_three_runs": callback_times,
                "median_microseconds_per_row": us_per_row,
                "projected_seconds_1970_sequences": projected_seconds,
                "official_time_limit_seconds": OFFICIAL_SECONDS,
                "peak_rss_bytes": peak_rss,
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            },
            "gates": gates,
            "eligible_for_w18_only_delivery": all(gates.values()),
            "score_computed": False,
            "independent_score_claim": "forbidden",
        }
    finally:
        shutil.rmtree(package_root)


def _reference_paths(root: Path) -> tuple[Path, ...]:
    from scripts.package_final_tl3 import RUNS

    return (
        root / RUNS[0] / "checkpoints" / "model.pt",
        root / RUNS[1] / "checkpoints" / "model.pt",
        root / RUNS[2] / "checkpoints" / "model.pt",
        root / "models" / RUNS[3] / "model.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--fit-process-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--reference-checkpoints-root", type=Path, required=True)
    parser.add_argument("--r11-native", type=Path, required=True)
    parser.add_argument("--quantization-contract", type=Path, required=True)
    parser.add_argument("--quantization-contract-sha256", required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if not result["eligible_for_w18_only_delivery"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
