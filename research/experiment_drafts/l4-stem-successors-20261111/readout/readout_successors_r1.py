"""Re-export six fixed-E6 L4 successors with pinned source/config checks."""

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

NATIVE_ROOT = Path("/data/lc-epanov_s/tabdata-stem-successors-20261111-r1")
SOURCE_SHA256 = "38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d"
SOURCE_MANIFEST_SHA256 = "e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d0"
SOURCE_COMMIT = "6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
R2_SHA256 = "fadb34135dbf3ac6b5f3c5ab4d316e892d9ea16e6094bbc999cf5344ba94eb68"
CELLS = (
    "k64_s203301",
    "gru512_k16_s203301",
    "gru_dropout01_k16_s203301",
    "k64_s203302",
    "gru512_k16_s203302",
    "gru_dropout01_k16_s203302",
)
ARMS = ("k64", "gru512_k16", "gru_dropout01_k16")
GPU_BY_CELL = {
    "k64_s203301": 0,
    "gru512_k16_s203301": 1,
    "gru_dropout01_k16_s203301": 2,
    "k64_s203302": 3,
    "gru512_k16_s203302": 5,
    "gru_dropout01_k16_s203302": 6,
}
SPEC = {
    "k64_s203301": (203301, "k64", 0, 64, 256, 0.0),
    "gru512_k16_s203301": (203301, "gru512_k16", 1, 16, 512, 0.0),
    "gru_dropout01_k16_s203301": (203301, "gru_dropout01_k16", 2, 16, 256, 0.1),
    "k64_s203302": (203302, "k64", 3, 64, 256, 0.0),
    "gru512_k16_s203302": (203302, "gru512_k16", 5, 16, 512, 0.0),
    "gru_dropout01_k16_s203302": (203302, "gru_dropout01_k16", 6, 16, 256, 0.1),
}
CONFIG_RAW_SHA256 = {
    "k64_s203301": "d5ba1ff1c59a202939787e9f3a1e11f49d936e15468f8edc5fdc9c892948d3cb",
    "gru512_k16_s203301": "04a57321f2c776cc037d880dd0e4735fac7342e85df349d362568543f33da1c2",
    "gru_dropout01_k16_s203301": "da50d0db699dce5646eede7337fe41f13578a52f4295af73d6db05a0c64a613d",
    "k64_s203302": "3f96d100cbec7452867bcf7201d3c456b00191c73b554555de92a8bcc6523827",
    "gru512_k16_s203302": "47a0104f7e98bc6462b9972b38e3873ca003aee1dc059e6032477b31329a950b",
    "gru_dropout01_k16_s203302": "08cf5e52ab05f23f9413c937343f28f335ffd90b424061219661c715d0ec0202",
}
CONFIG_MANIFEST_SHA256 = "2df8cba266f2ad5486e0b1de5370958666a915a6d22d45bfecd1e040af5ff6cf"
OLD_CONTROL_SOURCE_SHA256 = "b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188"
OLD_CONTROL_SOURCE_COMMIT = "336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2"
OLD_CONTROL_DATA_SHA256 = "4987ed5f263cf3c655714a7af3e792605716f6cb256c1c96474eb46ae9869323"
OLD_CONTROL_RECEIPT_SHA256 = {
    "k16_s203301": "842c0dc2606a62607d6f5c722bac7fcb09827e372537a4ab699d0c4c8642b75c",
    "k16_s203302": "eed8f1519dbc3b2e3e63a584d589bfca08d5a91094253e67efeb408af34ac8ff",
}
OLD_CONTROL_MODEL_SHA256 = {
    "k16_s203301": "8a914234265a810b006efedf9ee75ff7cc3577773d1562a8d810fbbb754e4dd1",
    "k16_s203302": "6276a09e88412b17769fb46013055390ea80e3decf377b41e204ef494ea7eeac",
}
OLD_CONTROL_CONFIG_SHA256 = {
    "k16_s203301": "70e8adb3c96e51ad816a9c6ccf95a2222dc4e263c9f8afd9f9342e4e0c61231b",
    "k16_s203302": "9e25079ab7bd65a85ef79e7942ec2bffac25b20730f86947d3492b97f6dfbe7d",
}
FIT_SEQUENCES, VALID_SEQUENCES, VALID_ROWS = 10_325, 1_873, 3_528_273
SEQUENCE_LENGTH, UPDATES_PER_EPOCH, UPDATES = 20_000, 6_460, 38_760
EXPORT_SECONDS, SCORE_SECONDS = 1_200, 600
OUTPUT_CAP, FREE_RESERVE = 3_400_000_000, 2_500_000_000
CAMPAIGN_FIRST_GPU_EPOCH = 1_794_359_428.5272088
FIT_SECONDS, WAVE_SECONDS = 14_400, 18_000
CELL_SCHEMA = "wnn.l4-stem-successors-readout-cell.r1"
SUMMARY_SCHEMA = "wnn.l4-stem-successors-readout-summary.r1"
DEFAULT_CONTROL_ROOT = Path("/data/lc-epanov_s/tabdata-stem-20261110-r1")
DEFAULT_OLD_SOURCE = Path(
    "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/"
    "source-gru-monitor-resume-r1/src"
)
DEFAULT_R2_HELPER = Path(
    "/data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py"
)
JsonMap = dict[str, Any]
# Any is confined to the frozen helper boundary, whose API is dynamically imported.


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
        admission.get("schema") != "wnn.l4-stem-successors-root-fit-admission.v1"
        or admission.get("approved") is not True
        or admission.get("max_fresh_fits") != 6
        or admission.get("source_root") != str(source)
        or admission.get("source_sha256") != SOURCE_SHA256
        or admission.get("split_sha256") != SPLIT_SHA256
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError("ROOT admission differs from the frozen six-cell successor contract")
    configs = root / "inputs-r1" / "configs"
    seen_gpus: set[int] = set()
    seen_uuids: set[str] = set()
    gpu_bindings: dict[str, tuple[int, str]] = {}
    for cell, (seed, _, gpu, _, _, _) in SPEC.items():
        profile = profiles[cell]
        config_path = configs / f"{cell}.json"
        if not isinstance(profile, dict) or config_path.is_symlink() or not config_path.is_file():
            raise ValueError(f"missing profile or config for {cell}")
        raw_config = common["_read_json"](config_path, 64_000)
        if not isinstance(raw_config, dict):
            raise ValueError(f"raw config is not an object for {cell}")
        config = load_config(config_path)
        receipt = Path(str(profile.get("receipt_path", "")))
        uuid = profile.get("gpu_uuid")
        if isinstance(uuid, str):
            gpu_bindings[cell] = (gpu, uuid)
        if (
            type(profile.get("gpu")) is not int
            or profile.get("gpu") != GPU_BY_CELL[cell]
            or not isinstance(uuid, str)
            or not uuid
            or uuid in seen_uuids
            or gpu in seen_gpus
            or profile.get("raw_config_sha256") != common["sha256_file"](config_path)
            or profile.get("raw_config_sha256") != CONFIG_RAW_SHA256[cell]
            or not _is_sha(profile.get("config_sha256"))
            or _sha_json(common, config) != profile["config_sha256"]
            or not _is_sha(profile.get("receipt_sha256"))
            or receipt.is_symlink()
            or not receipt.is_file()
            or common["sha256_file"](receipt) != profile["receipt_sha256"]
            or config.get("seed") != seed
            or not _valid_config_contract(cell, config)
        ):
            raise ValueError(f"profile/config receipt differs from the admitted {cell} cell")
        profile_receipt = common["_read_json"](receipt, 2_000_000)
        identity = profile_receipt.get("identity")
        reload_receipt = profile_receipt.get("engineering_reload")
        if (
            profile_receipt.get("status") != "profile_complete"
            or profile_receipt.get("updates") != 30
            or profile_receipt.get("errors") != []
            or not isinstance(identity, dict)
            or identity.get("source_sha256") != SOURCE_SHA256
            or identity.get("config_sha256") != profile["config_sha256"]
            or not isinstance(reload_receipt, dict)
            or reload_receipt.get("status") != "passed"
        ):
            raise ValueError(f"profile receipt lacks a passing exact-source reload gate for {cell}")
        seen_gpus.add(gpu)
        seen_uuids.add(uuid)
    if set(seen_gpus) != set(GPU_BY_CELL.values()) or len(seen_uuids) != len(CELLS):
        raise ValueError("ROOT admission reuses or changes an assigned GPU/UUID")
    common["GPU_BINDINGS"] = gpu_bindings
    admission["admission_sha256"] = digest
    return admission


def _valid_config_contract(cell: str, config: JsonMap) -> bool:
    """Check the source-normalized recipe against its one declared model change."""
    seed, _, _, kernel, hidden, dropout = SPEC[cell]
    model = config.get("model")
    return (
        config.get("seed") == seed
        and config.get("epochs") == 6
        and config.get("required_device") == "L4"
        and config.get("required_world_size") == 1
        and config.get("precision_mode") == "fp32"
        and config.get("normalization_scope") == "all_fit"
        and config.get("normalization_sequences") == FIT_SEQUENCES
        and config.get("training_mask") == "all_after_warmup"
        and config.get("objective_mode") == "standalone_labels_wmse_v2"
        and config.get("supervised_target") is None
        and config.get("checkpoint_retention") == "all_epochs"
        and config.get("checkpoint_endpoint_policy") == "configured_final_epoch"
        and isinstance(model, dict)
        and model.get("architecture") == "gru"
        and model.get("hidden") == hidden
        and model.get("layers") == 2
        and model.get("input_stem_kernel", 0) == kernel
        and model.get("dropout", 0.0) == dropout
    )


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
    seed, arm, gpu, kernel, hidden, dropout = SPEC[cell]
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
        or not _valid_config_contract(cell, config)
    ):
        raise ValueError("run config/source differs from the admitted successor cell")
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
        or devices[0].get("uuid") != profile["gpu_uuid"]
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
        "arm": arm,
        "seed": seed,
        "gpu": gpu,
        "gpu_uuid": profile["gpu_uuid"],
        "model_config": {
            "architecture": "gru",
            "hidden": hidden,
            "layers": 2,
            "input_stem_kernel": kernel,
            "dropout": dropout,
        },
        "raw_config_sha256": profile["raw_config_sha256"],
        "profile_receipt_sha256": profile["receipt_sha256"],
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


def wait_for_all_fit_results(
    common: dict[str, Any], root: Path, cells: tuple[str, ...], deadline: float
) -> dict[str, JsonMap]:
    """Wait for the cohort barrier so readout workers never overlap the six fits."""
    paths = {cell: root / "control" / "root-fits-r1" / f"{cell}-result.json" for cell in cells}
    while time.time() < deadline:
        present = {cell: path for cell, path in paths.items() if path.is_file()}
        if len(present) == len(cells):
            return {cell: common["_read_json"](path, 128_000) for cell, path in present.items()}
        time.sleep(min(15.0, max(0.0, deadline - time.time())))
    present = {cell: path for cell, path in paths.items() if path.is_file()}
    return {cell: common["_read_json"](path, 128_000) for cell, path in present.items()}


def _write(common: dict[str, Any], writer: Any, path: Path, value: JsonMap) -> None:
    common["_write_atomic"](writer, path, value)


def run_cell(
    common: dict[str, Any],
    root: Path,
    source: Path,
    python: Path,
    control: Path,
    cell: str,
    fit: JsonMap,
    admission: JsonMap,
    split: JsonMap,
    wave_deadline: float,
    owner: JsonMap,
    guard: Any,
    writer: Any,
    hold_gpu_slot: Any,
) -> JsonMap:
    """Export and score one already preflighted fit after the six-fit barrier."""
    seed, arm, gpu, _, _, _ = SPEC[cell]
    result_path = control / f"{cell}-result.json"
    output = root / "runs" / cell / "final-readout-successors-r1"
    try:
        if (
            result_path.exists()
            or result_path.is_symlink()
            or output.exists()
            or output.is_symlink()
        ):
            raise FileExistsError("readout artifacts already exist; automatic retry is disabled")
        if wave_deadline - time.time() < EXPORT_SECONDS + SCORE_SECONDS:
            raise TimeoutError(
                "fixed wave deadline leaves less than 1,800 seconds for export and score"
            )
        gpu_uuid = admission["profiles"][cell]["gpu_uuid"]
        claim = {
            "schema": "wnn.l4-stem-successors-readout-claim.r1",
            "cell": cell,
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "source_sha256": SOURCE_SHA256,
            "config_sha256": fit["config_sha256"],
            "raw_config_sha256": fit["raw_config_sha256"],
            "profile_receipt_sha256": fit["profile_receipt_sha256"],
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
            "arm": arm,
            "seed": seed,
            "model_config": fit["model_config"],
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "source_sha256": SOURCE_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "raw_config_sha256": fit["raw_config_sha256"],
            "profile_receipt_sha256": fit["profile_receipt_sha256"],
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


def validate_reused_controls(control_root: Path, common: dict[str, Any]) -> dict[str, JsonMap]:
    """Read the two byte-pinned, completed K16 full-valid readout receipts."""
    controls: dict[str, JsonMap] = {}
    for cell, expected_sha in OLD_CONTROL_RECEIPT_SHA256.items():
        path = control_root / "runs" / cell / "final-readout-r3" / "readout-receipt.json"
        if path.is_symlink() or not path.is_file() or common["sha256_file"](path) != expected_sha:
            raise ValueError(f"reused K16 control receipt is missing or changed: {cell}")
        receipt = common["_read_json"](path, 64_000)
        score = receipt.get("score")
        monitor = receipt.get("monitor_final_metrics")
        if (
            receipt.get("schema") != "wnn.l4-gru-causal-stem-readout-cell.r3"
            or receipt.get("status") != "readout_complete"
            or receipt.get("cell") != cell
            or receipt.get("source_sha256") != OLD_CONTROL_SOURCE_SHA256
            or receipt.get("source_commit") != OLD_CONTROL_SOURCE_COMMIT
            or receipt.get("split_sha256") != SPLIT_SHA256
            or receipt.get("data_sha256") != OLD_CONTROL_DATA_SHA256
            or receipt.get("model_sha256") != OLD_CONTROL_MODEL_SHA256[cell]
            or receipt.get("config_sha256") != OLD_CONTROL_CONFIG_SHA256[cell]
            or receipt.get("prediction_count") != VALID_SEQUENCES
            or receipt.get("prediction_shape") != [SEQUENCE_LENGTH, 2]
            or receipt.get("prediction_dtype") != "float32"
            or not isinstance(score, dict)
            or score.get("metric_id") != "global_wp_clip2_v1"
            or score.get("selected_rows") != VALID_ROWS
            or score.get("within_tolerance") is not True
            or not isinstance(monitor, dict)
            or monitor.get("valid/sequence_count") != VALID_SEQUENCES
            or monitor.get("valid/selected_rows") != VALID_ROWS
            or not all(common["_finite"](score.get(k)) for k in ("t0", "t1", "wp"))
        ):
            raise ValueError(f"reused K16 control receipt has invalid endpoint evidence: {cell}")
        controls[cell] = {
            "receipt_sha256": expected_sha,
            "model_sha256": receipt["model_sha256"],
            **score,
        }
    return controls


def derive_contrasts(scores: dict[str, JsonMap], controls: dict[str, JsonMap]) -> JsonMap:
    """Compute each successor-minus-K16 same-seed delta and its two-seed mean."""
    if set(scores) != set(CELLS) or set(controls) != {"k16_s203301", "k16_s203302"}:
        raise ValueError("all six successors and both reused K16 controls are required")
    if any(scores[cell].get("within_tolerance") is not True for cell in CELLS):
        raise ValueError("all six full-valid score receipts must pass before paired contrasts")
    if any(controls[cell].get("within_tolerance") is not True for cell in controls):
        raise ValueError("both reused K16 controls must pass full-valid score checks")
    rows: list[JsonMap] = []
    for seed in (203301, 203302):
        control_cell = f"k16_s{seed}"
        control = controls[control_cell]
        for arm in ARMS:
            candidate_cell = f"{arm}_s{seed}"
            candidate = scores[candidate_cell]
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "direction": "successor minus reused K16 control",
                    "candidate_cell": candidate_cell,
                    "control_cell": control_cell,
                    "control_receipt_sha256": control["receipt_sha256"],
                    **{
                        key: float(candidate[key]) - float(control[key])
                        for key in ("t0", "t1", "wp")
                    },
                }
            )
    means = [
        {
            "arm": arm,
            "seed_count": 2,
            "direction": "mean of two same-seed successor-minus-K16 deltas",
            **{
                key: sum(float(row[key]) for row in rows if row["arm"] == arm) / 2.0
                for key in ("t0", "t1", "wp")
            },
        }
        for arm in ARMS
    ]
    return {"by_seed": rows, "two_seed_mean": means}


def _clock(common: dict[str, Any], root: Path) -> tuple[float, float, float, float]:
    """Read the original fixed first-GPU+4h/+5h deadlines without resetting time."""
    clock = common["_read_json"](root / "receipts" / "first-profile-r1.json", 16_000)
    first, campaign_first_gpu, fit, wave = (
        clock.get("first_profile_epoch"),
        clock.get("campaign_first_gpu_epoch"),
        clock.get("fit_deadline_epoch"),
        clock.get("wave_deadline_epoch"),
    )
    finite = common["_finite"]
    if (
        not all(finite(v) for v in (first, campaign_first_gpu, fit, wave))
        or abs(float(campaign_first_gpu) - CAMPAIGN_FIRST_GPU_EPOCH) > 1.0
        or abs(float(fit) - float(campaign_first_gpu) - FIT_SECONDS) > 1.0
        or abs(float(wave) - float(campaign_first_gpu) - WAVE_SECONDS) > 1.0
        or float(first) > float(fit)
        or time.time() >= float(wave)
    ):
        raise TimeoutError("original first-GPU fit/readout window is closed or malformed")
    return float(first), float(campaign_first_gpu), float(fit), float(wave)


def run(args: argparse.Namespace) -> JsonMap:
    """Verify pins, wait for the fit barrier, then collect all eligible readouts once."""
    common = load_r2_helper(args.r2_helper)
    root, source, old_source = (
        args.root.resolve(),
        args.source.resolve(strict=True),
        args.old_src.resolve(strict=True),
    )
    control_root = args.control_root.resolve()
    python = lexical_executable(common, args.python)
    verifier = runpy.run_path(str(source / "src/wnn/training/sequence_source_capsule.py"))[
        "verify_source_capsule"
    ]
    verifier(source, source / "source-manifest.json", SOURCE_MANIFEST_SHA256, SOURCE_COMMIT)
    with _pinned_config_loader(source) as load_config:
        admission = validate_admission(common, root, source, args.admission_sha256, load_config)
    first, campaign_first_gpu, fit_deadline, wave_deadline = _clock(common, root)
    split_path = root / "project" / "chronological_primary.json"
    if common["sha256_file"](split_path) != SPLIT_SHA256:
        raise ValueError("prepared chronological primary split bytes differ from the frozen SHA")
    split = common["_read_json"](split_path, 8_000_000)
    expected_output = SEQUENCE_LENGTH * 2 * 4 * VALID_SEQUENCES
    if common["directory_bytes"](root) + len(CELLS) * (expected_output + 10_000_000) > OUTPUT_CAP:
        raise OSError("six exports would exceed the 3.4 GB aggregate output cap")
    if shutil.disk_usage(root).free < FREE_RESERVE:
        raise OSError("readout requires the original 2.5 GB disk reserve")
    try:
        controls = validate_reused_controls(control_root, common)
        control_reference: JsonMap = {
            "status": "validated",
            "source_root": str(control_root),
            "source_sha256": OLD_CONTROL_SOURCE_SHA256,
            "receipts": {cell: value["receipt_sha256"] for cell, value in controls.items()},
            "model_sha256": {cell: value["model_sha256"] for cell, value in controls.items()},
            "scores": {
                cell: {key: value[key] for key in ("t0", "t1", "wp")}
                for cell, value in controls.items()
            },
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        controls = {}
        control_reference = {
            "status": "unavailable_or_invalid",
            "source_root": str(control_root),
            "error": f"{type(error).__name__}: {error}",
        }
    sys.path.insert(0, str(old_source))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_runtime import current_process_owner, hold_gpu_slot

    control = root / "control" / "readout-successors-r1"
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
                "schema": "wnn.l4-stem-successors-readout-owner.r1",
                "owner": owner,
                "interpreter_path_lexical": str(python),
                "source_sha256": SOURCE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "admission_sha256": admission["admission_sha256"],
                "first_profile_epoch": first,
                "campaign_first_gpu_epoch": campaign_first_gpu,
                "fit_deadline_epoch": fit_deadline,
                "wave_deadline_epoch": wave_deadline,
            },
        )
        guard = common["ReadoutGuard"](root)
        outcomes: dict[str, JsonMap] = {}

        def record_preflight_failure(cell: str, outcome: JsonMap) -> None:
            """Persist fit-side exclusions before any readout worker starts."""
            outcomes[cell] = outcome
            _write(common, atomic_write_json, control / f"{cell}-result.json", outcome)

        fit_results = wait_for_all_fit_results(common, root, CELLS, fit_deadline)
        eligible: dict[str, JsonMap] = {}
        for cell in CELLS:
            result = fit_results.get(cell)
            if result is None:
                record_preflight_failure(
                    cell,
                    {
                        "schema": CELL_SCHEMA,
                        "cell": cell,
                        "status": "fit_deadline_no_result",
                        "failure_kind": "fit_deadline",
                        "automatic_retry": False,
                    },
                )
                continue
            fit_result_path = root / "control" / "root-fits-r1" / f"{cell}-result.json"
            try:
                validate_fit_result_time(fit_result_path, result, fit_deadline)
            except TimeoutError as error:
                record_preflight_failure(
                    cell,
                    {
                        "schema": CELL_SCHEMA,
                        "cell": cell,
                        "status": "fit_result_late_rejected",
                        "failure_kind": "fit_deadline",
                        "error": str(error),
                        "automatic_retry": False,
                    },
                )
                continue
            if result.get("status") != "fit_complete":
                record_preflight_failure(
                    cell,
                    {
                        "schema": CELL_SCHEMA,
                        "cell": cell,
                        "status": "fit_failed_or_partial",
                        "failure_kind": "fit_not_complete",
                        "fit_result": result,
                        "automatic_retry": False,
                    },
                )
                continue
            try:
                eligible[cell] = validate_fit(common, root, cell, admission, split, fit_deadline)
            except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
                record_preflight_failure(
                    cell,
                    {
                        "schema": CELL_SCHEMA,
                        "cell": cell,
                        "status": "fit_evidence_rejected",
                        "failure_kind": "fit_provenance_or_endpoint",
                        "error": f"{type(error).__name__}: {error}",
                        "automatic_retry": False,
                    },
                )
        if eligible:
            with ThreadPoolExecutor(
                max_workers=len(eligible), thread_name_prefix="l4-successor-readout"
            ) as pool:
                futures = {
                    pool.submit(
                        run_cell,
                        common,
                        root,
                        source,
                        python,
                        control,
                        cell,
                        fit,
                        admission,
                        split,
                        wave_deadline,
                        owner,
                        guard,
                        atomic_write_json,
                        hold_gpu_slot,
                    ): cell
                    for cell, fit in eligible.items()
                }
                for future in as_completed(futures):
                    cell = futures[future]
                    try:
                        outcomes[cell] = future.result()
                    except Exception as error:
                        record_preflight_failure(
                            cell,
                            {
                                "schema": CELL_SCHEMA,
                                "cell": cell,
                                "status": "readout_failed",
                                "failure_kind": "controller_infrastructure",
                                "error": f"{type(error).__name__}: {error}",
                                "automatic_retry": False,
                            },
                        )
        scores = {
            cell: outcome["score"]
            for cell, outcome in outcomes.items()
            if outcome.get("status") == "readout_complete"
        }
        complete = len(scores) == 6
        models = {
            cell: {
                "seed": SPEC[cell][0],
                "arm": SPEC[cell][1],
                "gpu": SPEC[cell][2],
                "model_config": outcomes[cell].get("model_config"),
                "config_sha256": outcomes[cell]["config_sha256"],
                "model_sha256": outcomes[cell]["model_sha256"],
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
            "config_manifest_sha256": CONFIG_MANIFEST_SHA256,
            "split_sha256": SPLIT_SHA256,
            "admission_sha256": admission["admission_sha256"],
            "first_profile_epoch": first,
            "campaign_first_gpu_epoch": campaign_first_gpu,
            "fit_deadline_epoch": fit_deadline,
            "wave_deadline_epoch": wave_deadline,
            "fixed_endpoint": {
                "epochs": 6,
                "global_updates": UPDATES,
                "fit_sequences": FIT_SEQUENCES,
                "evaluation_sequences": VALID_SEQUENCES,
                "evaluation_selected_rows": VALID_ROWS,
                "metric_id": "global_wp_clip2_v1",
            },
            "cells": outcomes,
            "model_scores": models,
            "reused_controls": control_reference,
            "paired_contrasts": (
                derive_contrasts(scores, controls) if complete and len(controls) == 2 else None
            ),
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
    parser.add_argument("--old-src", type=Path, default=DEFAULT_OLD_SOURCE)
    parser.add_argument("--r2-helper", type=Path, default=DEFAULT_R2_HELPER)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument(
        "--admission-sha256", required=True, help="ROOT-provided exact admission file SHA-256"
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="launch one detached, birth-receipted readout waiter through the owned runtime",
    )
    args = parser.parse_args()
    if args.detach:
        old_source = args.old_src.resolve(strict=True)
        sys.path.insert(0, str(old_source))
        from wnn.multistage_search_runtime import launch_detached_worker

        control = args.root / "control" / "readout-successors-r1"
        control.mkdir(parents=True, exist_ok=True)
        child_args = [arg for arg in sys.argv[1:] if arg != "--detach"]
        command = [str(args.python), str(Path(__file__).resolve()), *child_args]
        receipt = launch_detached_worker(
            command,
            cwd=args.root,
            env=dict(os.environ),
            receipt_path=control / "launcher.json",
            metadata={"purpose": "single post-fit readout waiter", "cells": len(CELLS)},
        )
        print(json.dumps(receipt, allow_nan=False))
        return 0
    result = run(args)
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
