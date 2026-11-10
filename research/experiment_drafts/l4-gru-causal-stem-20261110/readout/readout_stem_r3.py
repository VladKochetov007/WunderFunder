"""Re-export six fixed-E6 fits with source-normalized admission checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import runpy
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

NATIVE_ROOT = Path("/data/lc-epanov_s/tabdata-stem-20261110-r1")
SOURCE_SHA256 = "b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188"
SOURCE_MANIFEST_SHA256 = "482d13529e8c65386e6f74e7e81f607eeb3b0ed0795beff44902bf703e7d342e"
SOURCE_COMMIT = "336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
R2_SHA256 = "fadb34135dbf3ac6b5f3c5ab4d316e892d9ea16e6094bbc999cf5344ba94eb68"
CELLS = ("plain_s203301", "plain_s203302", "k1_s203301", "k1_s203302", "k16_s203301", "k16_s203302")
GPU_UUIDS = (
    "222adc44-ecd3-a645-cf7e-ddda9d495f7a",
    "bce838c4-097e-0623-7b67-6a97f3778510",
    "9fd565e2-ebb2-35be-c628-d971a575aa7e",
    "02d8270f-e4ee-1363-8232-15c0726fdab1",
    "9943f86e-f085-2d36-2c12-8d2a22111ef0",
    "48b463a4-5b0a-e277-9ef1-ee37be8015b1",
)
SPEC = {
    "plain_s203301": (203301, 0, 0),
    "plain_s203302": (203302, 0, 1),
    "k1_s203301": (203301, 1, 2),
    "k1_s203302": (203302, 1, 3),
    "k16_s203301": (203301, 16, 4),
    "k16_s203302": (203302, 16, 5),
}
FIT_SEQUENCES, VALID_SEQUENCES, VALID_ROWS = 10_325, 1_873, 3_528_273
SEQUENCE_LENGTH, UPDATES_PER_EPOCH, UPDATES = 20_000, 6_460, 38_760
EXPORT_SECONDS, SCORE_SECONDS = 1_200, 600
OUTPUT_CAP, FREE_RESERVE = 2_500_000_000, 4_000_000_000
SPLIT_SECONDS, WAVE_SECONDS = 10_800, 14_400
CELL_SCHEMA = "wnn.l4-gru-causal-stem-readout-cell.r3"
SUMMARY_SCHEMA = "wnn.l4-gru-causal-stem-readout-summary.r3"
JsonMap = dict[str, Any]


def load_r2_helper(path: Path) -> dict[str, Any]:
    """Load the pinned R2 common executor/producer/scorer validators and bind this source."""
    if path.is_symlink() or not path.is_file() or _sha_file(path) != R2_SHA256:
        raise ValueError("shared R2 receiver is missing, linked or differs from its pinned SHA-256")
    spec = importlib.util.spec_from_file_location("frozen_stem_readout_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("pinned R2 helper cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Function globals must be this exact dictionary; runpy returns a copy.
    common = vars(module)
    required = (
        "_absolute_executable",
        "_read_json",
        "_read_events",
        "sha256_file",
        "canonical_sha256",
        "validate_fit_membership",
        "validate_prediction_manifest",
        "validate_score_receipt",
        "validate_project_file_stats",
        "ReadoutGuard",
        "_command",
        "directory_bytes",
    )
    if any(name not in common for name in required):
        raise ValueError("pinned R2 helper lacks a required producer/scorer/executor API")
    common.update(
        SOURCE_SHA256=SOURCE_SHA256,
        SOURCE_MANIFEST_SHA256=SOURCE_MANIFEST_SHA256,
        SOURCE_COMMIT=SOURCE_COMMIT,
        SPLIT_SHA256=SPLIT_SHA256,
        CELLS=CELLS,
        GPU_BINDINGS={cell: (gpu, GPU_UUIDS[gpu]) for cell, (_, _, gpu) in SPEC.items()},
        EXPECTED_UPDATES=UPDATES,
        UPDATES_PER_EPOCH=UPDATES_PER_EPOCH,
        FIT_SEQUENCES=FIT_SEQUENCES,
        VALID_SEQUENCES=VALID_SEQUENCES,
        VALID_ROWS=VALID_ROWS,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
        EXPORT_TIMEOUT_SECONDS=EXPORT_SECONDS,
        SCORE_TIMEOUT_SECONDS=SCORE_SECONDS,
        MAX_OUTPUT_BYTES=OUTPUT_CAP,
        MIN_FREE_BYTES=FREE_RESERVE,
        ABS_TOLERANCE=1e-5,
    )
    return common


def lexical_executable(common: dict[str, Any], path: Path) -> Path:
    """Preserve an executable's absolute spelling so a venv symlink remains in the command."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    if not candidate.is_file() or (os.name == "posix" and not os.access(candidate, os.X_OK)):
        raise FileNotFoundError(f"interpreter is missing or not executable: {candidate}")
    return candidate


def _sha_json(common: dict[str, Any], value: object) -> str:
    return common["canonical_sha256"](value)


@contextmanager
def _pinned_config_loader(source: Path) -> Iterator[Callable[[Path], JsonMap]]:
    """Import load_config only from the source capsule verified by the caller."""
    source = source.resolve(strict=True)
    source_src = source / "src"
    expected_module = source_src / "wnn" / "training" / "sequence_config.py"
    if not expected_module.is_file() or expected_module.is_symlink():
        raise ValueError("pinned source has no safe sequence_config.py")
    original_path = sys.path[:]
    saved_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "wnn" or name.startswith("wnn.")
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        sys.path.insert(0, str(source_src))
        module = importlib.import_module("wnn.training.sequence_config")
        if Path(module.__file__).resolve(strict=True) != expected_module.resolve(strict=True):
            raise ValueError("sequence_config was imported outside the verified source capsule")
        loader = getattr(module, "load_config", None)
        if not callable(loader):
            raise ValueError("verified source sequence_config has no load_config")

        def load(config_path: Path) -> JsonMap:
            normalized = loader(config_path)
            if not isinstance(normalized, dict):
                raise ValueError("source load_config returned a non-object recipe")
            return normalized

        yield load
    finally:
        for name in tuple(sys.modules):
            if name == "wnn" or name.startswith("wnn."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = original_path


def _sha_file(path: Path) -> str:
    """Hash the shared helper before importing its code."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_admission(
    common: dict[str, Any],
    root: Path,
    source: Path,
    expected_sha: str,
    load_config: Callable[[Path], JsonMap],
) -> JsonMap:
    """Bind exact ROOT authorization, configs, receipts and six physical GPU identities."""
    path = root / "control" / "ROOT-fit-admission-r1.json"
    digest = common["sha256_file"](path)
    if digest != expected_sha:
        raise ValueError("ROOT admission bytes do not match required --admission-sha256")
    admission = common["_read_json"](path, 128_000)
    profiles = admission.get("profiles")
    if (
        admission.get("schema") != "wnn.l4-stem-root-fit-admission.v1"
        or admission.get("approved") is not True
        or admission.get("max_fresh_fits") != 6
        or admission.get("source_root") != str(source)
        or admission.get("source_sha256") != SOURCE_SHA256
        or admission.get("split_sha256") != SPLIT_SHA256
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError("ROOT admission differs from the frozen six-cell stem contract")
    configs = root / "inputs-r1" / "configs"
    for cell, (seed, kernel, gpu) in SPEC.items():
        profile = profiles[cell]
        config_path = configs / f"{cell}.json"
        if not isinstance(profile, dict) or config_path.is_symlink() or not config_path.is_file():
            raise ValueError(f"missing profile or config for {cell}")
        raw_config = common["_read_json"](config_path, 64_000)
        if not isinstance(raw_config, dict):
            raise ValueError(f"raw config is not an object for {cell}")
        config = load_config(config_path)
        receipt = Path(str(profile.get("receipt_path", "")))
        if (
            type(profile.get("gpu")) is not int
            or profile.get("gpu") != gpu
            or profile.get("gpu_uuid") != GPU_UUIDS[gpu]
            or profile.get("raw_config_sha256") != common["sha256_file"](config_path)
            or not _is_sha(profile.get("config_sha256"))
            or _sha_json(common, config) != profile["config_sha256"]
            or not _is_sha(profile.get("receipt_sha256"))
            or receipt.is_symlink()
            or not receipt.is_file()
            or common["sha256_file"](receipt) != profile["receipt_sha256"]
            or config.get("seed") != seed
            or config.get("epochs") != 6
            or not isinstance(config.get("model"), dict)
            or config["model"].get("input_stem_kernel", 0) != kernel
        ):
            raise ValueError(f"profile/config receipt differs from the admitted {cell} cell")
    admission["admission_sha256"] = digest
    return admission


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_fit_result_time(path: Path, result: JsonMap, fit_deadline: float) -> None:
    """Reject ROOT result files written or timestamped after the original fit cutoff."""
    at = result.get("at")
    if (
        isinstance(at, bool)
        or not isinstance(at, (float, int))
        or not math.isfinite(float(at))
        or float(at) > fit_deadline
        or path.stat().st_mtime > fit_deadline
    ):
        raise TimeoutError("ROOT fit result was written after the original fit cutoff")


def validate_official_membership(
    common: dict[str, Any], fit_members: object, valid_members: object, split: JsonMap
) -> None:
    """Require the producer rosters to equal the exact verified official split lists."""
    common["validate_fit_membership"](
        {"fit_members": fit_members, "evaluation_members": valid_members}, split
    )


def validate_fit(
    common: dict[str, Any],
    root: Path,
    cell: str,
    admission: JsonMap,
    split: JsonMap,
    fit_deadline: float,
) -> JsonMap:
    """Independently bind ROOT exit, model bytes, metadata, terminal and all E1..E6 events."""
    fit_result_path = root / "control" / "root-fits-r1" / f"{cell}-result.json"
    result = common["_read_json"](fit_result_path, 128_000)
    validate_fit_result_time(fit_result_path, result, fit_deadline)
    if (
        result.get("cell") != cell
        or result.get("status") != "fit_complete"
        or type(result.get("exit_code")) is not int
        or result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("resource_errors") != []
        or not _is_sha(result.get("model_sha256"))
    ):
        raise ValueError("ROOT result does not prove a zero-exit fixed endpoint")
    fit_dir = root / "runs" / cell / "fit"
    model_path = fit_dir / "model.pt"
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("fixed endpoint model is missing or linked")
    model_sha = common["sha256_file"](model_path)
    metadata = common["_read_json"](fit_dir / "run_metadata.json", 1_000_000)
    terminal = common["_read_json"](fit_dir / "terminal_status.json", 2_000_000)
    events = common["_read_events"](fit_dir / "monitor_evaluations.jsonl")
    profile = admission["profiles"][cell]
    config, source_info, provenance = (
        metadata.get("config"),
        metadata.get("source"),
        metadata.get("provenance"),
    )
    if not all(isinstance(x, dict) for x in (config, source_info, provenance)):
        raise ValueError("native run metadata lacks config/source/provenance")
    seed, kernel, gpu = SPEC[cell]
    config_sha = _sha_json(common, config)
    if (
        config_sha != profile["config_sha256"]
        or source_info.get("sha256") != SOURCE_SHA256
        or config.get("seed") != seed
        or config.get("epochs") != 6
        or config.get("required_device") != "L4"
        or config.get("required_world_size") != 1
        or config.get("precision_mode") != "fp32"
        or config.get("normalization_scope") != "all_fit"
        or config.get("normalization_sequences") != FIT_SEQUENCES
        or config.get("training_mask") != "all_after_warmup"
        or config.get("objective_mode") != "standalone_labels_wmse_v2"
        or config.get("supervised_target") is not None
        or config.get("checkpoint_retention") != "all_epochs"
        or config.get("checkpoint_endpoint_policy") != "configured_final_epoch"
        or not isinstance(config.get("model"), dict)
        or any(
            config["model"].get(k) != v
            for k, v in {
                "architecture": "gru",
                "hidden": 256,
                "layers": 2,
            }.items()
        )
        or config["model"].get("input_stem_kernel", 0) != kernel
    ):
        raise ValueError("run config/source differs from the admitted stem cell")
    data = provenance.get("data")
    if not isinstance(data, dict) or provenance.get("data_sha256") != _sha_json(common, data):
        raise ValueError("data provenance digest is invalid")
    fit_members, valid_members = data.get("fit_members"), data.get("evaluation_members")
    validate_official_membership(common, fit_members, valid_members, split)
    norm, objective, sources = (
        data.get("normalization"),
        data.get("objective_contract"),
        data.get("sources"),
    )
    if (
        data.get("split_sha256") != SPLIT_SHA256
        or data.get("teacher") is not None
        or data.get("training_mask") != "all_after_warmup"
        or not isinstance(norm, dict)
        or norm.get("scope") != "all_fit"
        or norm.get("sequence_count") != FIT_SEQUENCES
        or not isinstance(objective, dict)
        or objective.get("mode") != "standalone_labels_wmse_v2"
        or objective.get("reference_training_ancestry") != "none"
        or provenance.get("world_size") != 1
        or not isinstance(sources, dict)
        or any(
            not isinstance(sources.get(name), dict)
            for name in ("datasets/train.parquet", "datasets/valid.parquet")
        )
    ):
        raise ValueError("split, membership or training ancestry differs from approved recipe")
    devices = (
        provenance.get("runtime", {}).get("devices")
        if isinstance(provenance.get("runtime"), dict)
        else None
    )
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or devices[0].get("name") != "NVIDIA L4"
        or devices[0].get("uuid") != GPU_UUIDS[gpu]
    ):
        raise ValueError("native run did not use its ROOT-bound L4 identity")
    identity = terminal.get("identity")
    if (
        terminal.get("status") != "completed"
        or type(terminal.get("completed_epochs")) is not int
        or terminal.get("completed_epochs") != 6
        or type(terminal.get("global_updates")) is not int
        or terminal.get("global_updates") != UPDATES
        or terminal.get("model_sha256") != model_sha
        or result.get("model_sha256") != model_sha
        or not isinstance(identity, dict)
        or identity.get("source_sha256") != SOURCE_SHA256
        or identity.get("config_sha256") != config_sha
        or identity.get("data_sha256") != provenance["data_sha256"]
        or identity.get("world_size") != 1
    ):
        raise ValueError("native terminal status is not bound to the admitted fixed endpoint")
    valid_sha = hashlib.sha256(
        json.dumps(valid_members, separators=(",", ":")).encode()
    ).hexdigest()
    if len(events) != 6:
        raise ValueError("fixed-E6 run does not contain six monitor events")
    finite = common["_finite"]
    for epoch, event in enumerate(events, 1):
        metrics, context = event.get("metrics"), event.get("context")
        if (
            event.get("schema") != "wnn.monitor.v1"
            or event.get("kind") != "evaluation"
            or type(event.get("epoch")) is not int
            or event.get("epoch") != epoch
            or type(event.get("step")) is not int
            or event.get("step") != epoch * UPDATES_PER_EPOCH
            or not isinstance(metrics, dict)
            or not all(finite(v) for v in metrics.values())
            or type(metrics.get("valid/sequence_count")) is not int
            or metrics.get("valid/sequence_count") != VALID_SEQUENCES
            or type(metrics.get("valid/selected_rows")) is not int
            or metrics.get("valid/selected_rows") != VALID_ROWS
            or not all(finite(metrics.get(k)) for k in ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
            or not isinstance(context, dict)
            or context.get("split_sha256") != SPLIT_SHA256
            or type(context.get("evaluation_member_count")) is not int
            or context.get("evaluation_member_count") != VALID_SEQUENCES
            or context.get("evaluation_member_sha256") != valid_sha
            or context.get("endpoint") != "raw standalone z; no reference or blend"
        ):
            raise ValueError(f"full-valid monitor event E{epoch} fails identity/coverage checks")
    final_metrics = events[-1]["metrics"]
    if result.get("final_metrics") != final_metrics:
        raise ValueError("ROOT result final metrics differ from native E6 monitor event")
    fit = {
        "cell": cell,
        "config_sha256": config_sha,
        "source_sha256": SOURCE_SHA256,
        "split_sha256": SPLIT_SHA256,
        "data_sha256": provenance["data_sha256"],
        "model_sha256": model_sha,
        "fit_members": fit_members,
        "evaluation_members": valid_members,
        "data_source_fingerprints": {
            n: sources[n] for n in ("datasets/train.parquet", "datasets/valid.parquet")
        },
        "final_metrics": final_metrics,
    }
    common["validate_fit_membership"](fit, split)
    common["validate_project_file_stats"](root, fit, split["sources"])
    return fit


def wait_for_fit(common: dict[str, Any], root: Path, cell: str, deadline: float) -> JsonMap | None:
    """Observe only the original fit window; late results are never accepted."""
    path = root / "control" / "root-fits-r1" / f"{cell}-result.json"
    while time.time() < deadline:
        if path.exists():
            return common["_read_json"](path, 128_000)
        time.sleep(min(15.0, max(0.0, deadline - time.time())))
    # An atomic on-time receipt can arrive during the final sleep. The caller
    # still checks both its producer timestamp and mtime against the cutoff.
    if path.exists():
        return common["_read_json"](path, 128_000)
    return None


def _write(common: dict[str, Any], writer: Any, path: Path, value: JsonMap) -> None:
    common["_write_atomic"](writer, path, value)


def run_cell(
    common: dict[str, Any],
    root: Path,
    source: Path,
    python: Path,
    control: Path,
    cell: str,
    admission: JsonMap,
    split: JsonMap,
    fit_deadline: float,
    wave_deadline: float,
    owner: JsonMap,
    guard: Any,
    writer: Any,
    hold_gpu_slot: Any,
) -> JsonMap:
    """Export and score once after ROOT has published an on-time verified fit result."""
    gpu = SPEC[cell][2]
    result_path = control / f"{cell}-result.json"
    output = root / "runs" / cell / "final-readout-r3"
    try:
        if (
            result_path.exists()
            or result_path.is_symlink()
            or output.exists()
            or output.is_symlink()
        ):
            raise FileExistsError("readout artifacts already exist; automatic retry is disabled")
        result = wait_for_fit(common, root, cell, fit_deadline)
        if result is None:
            failure = {
                "schema": CELL_SCHEMA,
                "cell": cell,
                "status": "fit_deadline_no_result",
                "failure_kind": "fit_deadline",
                "automatic_retry": False,
            }
            _write(common, writer, result_path, failure)
            return failure
        fit = validate_fit(common, root, cell, admission, split, fit_deadline)
        if wave_deadline - time.time() < EXPORT_SECONDS + SCORE_SECONDS:
            raise TimeoutError(
                "fixed wave deadline leaves less than 1,800 seconds for export and score"
            )
        gpu_uuid = GPU_UUIDS[gpu]
        claim = {
            "schema": "wnn.l4-gru-causal-stem-readout-claim.r3",
            "cell": cell,
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "source_sha256": SOURCE_SHA256,
            "config_sha256": fit["config_sha256"],
            "model_sha256": fit["model_sha256"],
            "admission_sha256": admission["admission_sha256"],
            "controller_owner": owner,
            "created_at_epoch": time.time(),
        }
        _write(common, writer, control / f"claim-{cell}.json", claim)
        output.mkdir(parents=True, exist_ok=False)
        prediction_dir = output / "predictions"
        owner_cell = {
            "controller": owner,
            "cell_id": cell,
            "source_sha256": SOURCE_SHA256,
            "config_sha256": fit["config_sha256"],
            "model_sha256": fit["model_sha256"],
        }
        predict = [
            str(python),
            str(source / "scripts/predict_sequence.py"),
            "--checkpoint",
            str(root / "runs" / cell / "fit" / "model.pt"),
            "--data-root",
            str(root / "project"),
            "--split-manifest",
            str(root / "project/chronological_primary.json"),
            "--output",
            str(prediction_dir),
            "--membership",
            "evaluation",
        ]
        with hold_gpu_slot(control, str(gpu)):
            exported = common["_command"](
                predict,
                source=source,
                control=control,
                output_dir=prediction_dir,
                cell=cell,
                gpu=gpu,
                owner=owner_cell,
                stage="readout_export",
                timeout=min(EXPORT_SECONDS, int(wave_deadline - time.time())),
                guard=guard,
                gpu_uuid=gpu_uuid,
            )
        score_path = output / "score.json"
        score_command = [
            str(python),
            str(source / "scripts/score_sequence.py"),
            "--data-root",
            str(root / "project"),
            "--split-manifest",
            str(root / "project/chronological_primary.json"),
            "--predictions",
            str(prediction_dir),
            "--output",
            str(score_path),
        ]
        scored = common["_command"](
            score_command,
            source=source,
            control=control,
            output_dir=output,
            cell=cell,
            gpu=gpu,
            owner=owner_cell,
            stage="readout_score_cpu",
            timeout=min(SCORE_SECONDS, int(wave_deadline - time.time())),
            guard=guard,
            gpu_uuid=None,
        )
        if common["sha256_file"](root / "runs" / cell / "fit" / "model.pt") != fit["model_sha256"]:
            raise ValueError("fixed endpoint model bytes changed during readout")
        manifest_path = prediction_dir / "manifest.json"
        manifest = common["_read_json"](manifest_path, 32_000_000)
        manifest_sha = common["sha256_file"](manifest_path)
        common["validate_prediction_manifest"](manifest, fit, manifest_sha)
        score_receipt = common["_read_json"](score_path, 2_000_000)
        comparison = common["validate_score_receipt"](score_receipt, manifest, manifest_sha, fit)
        status = "readout_complete" if comparison["within_tolerance"] else "score_monitor_mismatch"
        receipt = {
            "schema": CELL_SCHEMA,
            "cell": cell,
            "status": status,
            "failure_kind": None
            if status == "readout_complete"
            else "export_or_metric_pipeline_mismatch",
            "arm": cell.split("_s")[0],
            "seed": SPEC[cell][0],
            "kernel": SPEC[cell][1],
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "source_sha256": SOURCE_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "config_sha256": fit["config_sha256"],
            "split_sha256": SPLIT_SHA256,
            "data_sha256": fit["data_sha256"],
            "model_sha256": fit["model_sha256"],
            "fit_result_sha256": common["sha256_file"](
                root / "control" / "root-fits-r1" / f"{cell}-result.json"
            ),
            "prediction_manifest_sha256": manifest_sha,
            "score_receipt_sha256": common["sha256_file"](score_path),
            "prediction_count": len(manifest["sequences"]),
            "prediction_shape": [SEQUENCE_LENGTH, 2],
            "prediction_dtype": "float32",
            "scored_rows": comparison["selected_rows"],
            "score": comparison,
            "monitor_final_metrics": fit["final_metrics"],
            "owned_export": exported,
            "owned_score": scored,
            "interpretation": "Same repeatedly exposed official-valid partition; this checks prediction/scorer agreement, not independent validation or fresh confirmation.",
        }
        _write(common, writer, output / "readout-receipt.json", receipt)
        _write(common, writer, result_path, receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema": CELL_SCHEMA,
            "cell": cell,
            "status": "readout_failed",
            "failure_kind": "infrastructure_or_provenance",
            "error": f"{type(error).__name__}: {error}",
            "automatic_retry": False,
        }
        if not result_path.exists():
            _write(common, writer, result_path, failure)
        return failure


def derive_contrasts(scores: dict[str, JsonMap]) -> list[JsonMap]:
    """Return same-seed per-target K16-minus-K1 and K16-minus-plain descriptions."""
    if set(scores) != set(CELLS) or any(
        scores[cell].get("within_tolerance") is not True for cell in CELLS
    ):
        raise ValueError("all six full-valid score receipts must pass before paired contrasts")
    rows = []
    for seed in (203301, 203302):
        for comparator in ("k1", "plain"):
            left, right = f"k16_s{seed}", f"{comparator}_s{seed}"
            rows.append(
                {
                    "seed": seed,
                    "contrast": f"K16 minus {comparator}",
                    "left_cell": left,
                    "right_cell": right,
                    **{
                        key: float(scores[left][key]) - float(scores[right][key])
                        for key in ("t0", "t1", "wp")
                    },
                }
            )
    return rows


def _clock(common: dict[str, Any], root: Path) -> tuple[float, float, float]:
    """Read and enforce the immutable first-profile+3h/+4h windows."""
    clock = common["_read_json"](root / "receipts" / "first-profile-r1.json", 16_000)
    first, fit, wave = (
        clock.get("first_profile_epoch"),
        clock.get("fit_deadline_epoch"),
        clock.get("wave_deadline_epoch"),
    )
    finite = common["_finite"]
    if (
        not all(finite(v) for v in (first, fit, wave))
        or abs(float(fit) - float(first) - SPLIT_SECONDS) > 1.0
        or abs(float(wave) - float(first) - WAVE_SECONDS) > 1.0
        or time.time() >= float(wave)
    ):
        raise TimeoutError("original first-profile fit/readout window is closed or malformed")
    return float(first), float(fit), float(wave)


def run(args: argparse.Namespace) -> JsonMap:
    """Verify the frozen source and six-cell admission, then collect once in parallel."""
    common = load_r2_helper(args.r2_helper)
    root, source, old_source = (
        args.root.resolve(),
        args.source.resolve(strict=True),
        args.old_src.resolve(strict=True),
    )
    python = lexical_executable(common, args.python)
    verifier = runpy.run_path(str(source / "src/wnn/training/sequence_source_capsule.py"))[
        "verify_source_capsule"
    ]
    verifier(source, source / "source-manifest.json", SOURCE_MANIFEST_SHA256, SOURCE_COMMIT)
    with _pinned_config_loader(source) as load_config:
        admission = validate_admission(common, root, source, args.admission_sha256, load_config)
    first, fit_deadline, wave_deadline = _clock(common, root)
    split_path = root / "project" / "chronological_primary.json"
    if common["sha256_file"](split_path) != SPLIT_SHA256:
        raise ValueError("prepared chronological primary split bytes differ from the frozen SHA")
    split = common["_read_json"](split_path, 8_000_000)
    expected_output = SEQUENCE_LENGTH * 2 * 4 * VALID_SEQUENCES
    if common["directory_bytes"](root) + len(CELLS) * (expected_output + 10_000_000) > OUTPUT_CAP:
        raise OSError("six exports would exceed the 2.5 GB aggregate output cap")
    if shutil.disk_usage(root).free < FREE_RESERVE:
        raise OSError("readout requires the original 4 GB disk reserve")
    sys.path.insert(0, str(old_source))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_runtime import current_process_owner, hold_gpu_slot

    control = root / "control" / "readout-r3"
    control.mkdir(parents=True, exist_ok=True)
    summary_path = control / "complete.json"
    with file_lock(control / "readout.lock", blocking=False):
        if summary_path.exists() or (control / "owner.json").exists():
            raise FileExistsError("readout controller already ran; no automatic rerun is allowed")
        owner = current_process_owner()
        _write(
            common,
            atomic_write_json,
            control / "owner.json",
            {
                "schema": "wnn.l4-gru-causal-stem-readout-owner.r3",
                "owner": owner,
                "interpreter_path_lexical": str(python),
                "source_sha256": SOURCE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "admission_sha256": admission["admission_sha256"],
                "first_profile_epoch": first,
                "fit_deadline_epoch": fit_deadline,
                "wave_deadline_epoch": wave_deadline,
            },
        )
        guard = common["ReadoutGuard"](root)
        outcomes: dict[str, JsonMap] = {}
        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="l4-stem-readout") as pool:
            futures = {
                pool.submit(
                    run_cell,
                    common,
                    root,
                    source,
                    python,
                    control,
                    cell,
                    admission,
                    split,
                    fit_deadline,
                    wave_deadline,
                    owner,
                    guard,
                    atomic_write_json,
                    hold_gpu_slot,
                ): cell
                for cell in CELLS
            }
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    outcomes[cell] = future.result()
                except Exception as error:
                    outcomes[cell] = {
                        "schema": CELL_SCHEMA,
                        "cell": cell,
                        "status": "readout_failed",
                        "failure_kind": "controller_infrastructure",
                        "error": f"{type(error).__name__}: {error}",
                        "automatic_retry": False,
                    }
        scores = {
            cell: outcome["score"]
            for cell, outcome in outcomes.items()
            if outcome.get("status") == "readout_complete"
        }
        complete = len(scores) == 6
        models = {
            cell: {
                "seed": SPEC[cell][0],
                "arm": cell.split("_s")[0],
                "kernel": SPEC[cell][1],
                **scores[cell],
            }
            for cell in CELLS
            if cell in scores
        }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "complete" if complete else "partial_or_failed",
            "source_sha256": SOURCE_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "split_sha256": SPLIT_SHA256,
            "admission_sha256": admission["admission_sha256"],
            "first_profile_epoch": first,
            "fit_deadline_epoch": fit_deadline,
            "wave_deadline_epoch": wave_deadline,
            "cells": outcomes,
            "model_scores": models,
            "paired_contrasts": derive_contrasts(scores) if complete else None,
            "interpretation": "Fixed exposed official-valid re-export and scorer agreement; descriptive paired screen only, not independent validation or public-score evidence.",
        }
        _write(common, atomic_write_json, summary_path, summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=NATIVE_ROOT / "source-r1")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"),
    )
    parser.add_argument(
        "--old-src",
        type=Path,
        default=Path(
            "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src"
        ),
    )
    parser.add_argument(
        "--r2-helper",
        type=Path,
        default=Path(
            "/data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py"
        ),
    )
    parser.add_argument(
        "--admission-sha256", required=True, help="ROOT-provided exact admission file SHA-256"
    )
    result = run(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_scores": result["model_scores"],
                "paired_contrasts": result["paired_contrasts"],
            },
            allow_nan=False,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
