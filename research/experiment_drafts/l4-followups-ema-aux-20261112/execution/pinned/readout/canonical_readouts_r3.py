"""Validate and canonically score each independently completed fixed-E6 L4 endpoint once."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import runpy
import shutil
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLOCK_SHA256 = "0181e9889a45d0636e3b9981ddb15e17fb5cd26429068bb0ee37f78d09d71270"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
VALID_SEQUENCES = 1_873
VALID_ROWS = 3_528_273
FIT_SEQUENCES = 10_325
SEQUENCE_LENGTH = 20_000
EPOCHS = 6
ABS_TOLERANCE = 1e-5
EXPORT_TIMEOUT_SECONDS = 1_200
SCORE_TIMEOUT_SECONDS = 600
FREE_RESERVE_BYTES = 1_000_000_000
MAX_OUTPUT_BYTES = {
    "first8": 3_400_000_000,
    "followups4": 2_000_000_000,
    "ema4": 2_000_000_000,
}

FIRST_SOURCE_SHA = "38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d"
FIRST_SOURCE_MANIFEST_SHA = (
    "e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d0"
)
FIRST_SOURCE_COMMIT = "6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba"
FOLLOWUP_SOURCE_SHA = "29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511"
FOLLOWUP_SOURCE_MANIFEST_SHA = (
    "00782a64e1557a8af759e6f987038cabb6ec09f098c43ab61d6cfa6bb07e9e5f"
)
FOLLOWUP_SOURCE_COMMIT = "390c846cbdacd8e0e06e6079dd3b04312fa6bcc5"

FIRST_CELLS = (
    "gru256_layer1_k16_s203301",
    "gru256_layer1_k16_s203302",
    "gru256_layer3_k16_s203301",
    "gru256_layer3_k16_s203302",
    "horizon2000_k16_s203301",
    "horizon2000_k16_s203302",
    "horizon4000_k16_s203301",
    "horizon4000_k16_s203302",
)
FOLLOWUP_CELLS = (
    "muon_rnn_k16_s203301",
    "muon_rnn_k16_s203302",
    "since32_k16_s203301",
    "since32_k16_s203302",
)
FIRST_GPU = {cell: i for i, cell in enumerate(FIRST_CELLS)}
FOLLOWUP_GPU = {cell: i for i, cell in enumerate(FOLLOWUP_CELLS)}

FIRST_RAW_CONFIG_SHA = {
    "gru256_layer1_k16_s203301": "324321739069ceb7fa165120c88798b4e890299fdaacc889ecaeae447bf34778",
    "gru256_layer1_k16_s203302": "327fcd8e4b9f0769025d67ec1d106c55f50e70cc33a1f78bb1fb27f7376bdef3",
    "gru256_layer3_k16_s203301": "55e945b23ff6c558052610a42b239c6b74ae7e6f1894910478682aeabfa6c3c9",
    "gru256_layer3_k16_s203302": "835b82ad6a7cab45957bc9bf8bedf2c1abaa17798f5dcb08213990d2eb335e69",
    "horizon2000_k16_s203301": "582ab53ac124dac45500f6d2dcd7a74a264149cc80030281e9cb872ff7f84609",
    "horizon2000_k16_s203302": "93396501ca19d9d26589bc6b5d0ae0ca35de743d1072e41741b5f6bfffc090d6",
    "horizon4000_k16_s203301": "19abe6d8fcd9f898f0fa2b0ab0114733e20c32c09f73fae17456169f5bc0a65b",
    "horizon4000_k16_s203302": "d8b4a4768ea7d466bde7b387366e480235fd6a108e3281d436c12a151b547aa4",
}
FOLLOWUP_RAW_CONFIG_SHA = {
    "muon_rnn_k16_s203301": "7b5e46e48d69694c653f2e3f4ae9f87550a758e2471e71e5c60033845ed9e6b3",
    "muon_rnn_k16_s203302": "a58bd410e0b52b13073fb581d3f10fba83789b4d176ef6f73abc1010af762df4",
    "since32_k16_s203301": "e5c820ce524aa305cd77cfa784492cb1aec2ac95eeb919a815894f63d10e7553",
    "since32_k16_s203302": "42972bf72573bf84fd55a5086b2bece2412799a74f9070eaf09ca7b688aa0fdf",
}
EMA_SOURCE_SHA = "33574f496a2640e1a4c14690c8858eeaad5075e21d5b96042b746d35ad55810a"
EMA_SOURCE_MANIFEST_SHA = (
    "d77c158e827a6b725e1033f9f20eff5ea472620e27b369d262e836f089e04147"
)
EMA_SOURCE_COMMIT = "73a2f64bcb4df20327225f045f51791fd607b69a"
EMA_CELLS = (
    "k16_s203301",
    "ema999_k16_s203301",
    "k16_s203302",
    "ema999_k16_s203302",
)
EMA_GPU = {cell: gpu for cell, gpu in zip(EMA_CELLS, (4, 5, 6, 7), strict=True)}
EMA_RAW_CONFIG_SHA = {
    "k16_s203301": "55a6f952f13d923817eda6c3dbf7a002934ff8e481f6367c6c1c82d8e754a534",
    "ema999_k16_s203301": "fa3c926515d94b4e90a483fb50fd622ec09161d8c1c26700fa5e08a0e2e72e6f",
    "k16_s203302": "a6672a34223f88d5bebfe9712ed3fb09192df6176239de5aeaf3dce2bd527bc0",
    "ema999_k16_s203302": "741c616a0d20993e79bccf12d0d024318b3c084e68f8d40a439e2fcc07ee1481",
}

JsonMap = dict[str, Any]


@dataclass(frozen=True)
class Cohort:
    name: str
    cells: tuple[str, ...]
    gpu_by_cell: dict[str, int]
    raw_config_sha: dict[str, str]
    source_sha: str
    source_manifest_sha: str
    source_commit: str
    results_dirname: str
    complete_name: str
    admission_schema: str
    manifest_name: str


FIRST = Cohort(
    name="first8",
    cells=FIRST_CELLS,
    gpu_by_cell=FIRST_GPU,
    raw_config_sha=FIRST_RAW_CONFIG_SHA,
    source_sha=FIRST_SOURCE_SHA,
    source_manifest_sha=FIRST_SOURCE_MANIFEST_SHA,
    source_commit=FIRST_SOURCE_COMMIT,
    results_dirname="root-fits-r1",
    complete_name="complete.json",
    admission_schema="wnn.l4-first-eight-root-fit-admission.r1",
    manifest_name="source-manifest.json",
)
FOLLOWUPS = Cohort(
    name="followups4",
    cells=FOLLOWUP_CELLS,
    gpu_by_cell=FOLLOWUP_GPU,
    raw_config_sha=FOLLOWUP_RAW_CONFIG_SHA,
    source_sha=FOLLOWUP_SOURCE_SHA,
    source_manifest_sha=FOLLOWUP_SOURCE_MANIFEST_SHA,
    source_commit=FOLLOWUP_SOURCE_COMMIT,
    results_dirname="root-fits-r4",
    complete_name="complete.json",
    admission_schema="wnn.l4-followups-root-fit-admission.r2",
    manifest_name="SOURCE_MANIFEST.json",
)
EMA = Cohort(
    name="ema4",
    cells=EMA_CELLS,
    gpu_by_cell=EMA_GPU,
    raw_config_sha=EMA_RAW_CONFIG_SHA,
    source_sha=EMA_SOURCE_SHA,
    source_manifest_sha=EMA_SOURCE_MANIFEST_SHA,
    source_commit=EMA_SOURCE_COMMIT,
    results_dirname="root-fits-ema-r2",
    complete_name="complete.json",
    admission_schema="wnn.l4-ema-root-fit-admission.r2",
    manifest_name="SOURCE_MANIFEST.json",
)


@dataclass(frozen=True)
class RuntimeCohort:
    plan: Cohort
    fit_root: Path
    output_root: Path
    source_root: Path
    data_root: Path
    config_root: Path
    admission_path: Path
    admission_sha: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _load_r2(path: Path, plan: Cohort) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path)
        != "fadb34135dbf3ac6b5f3c5ab4d316e892d9ea16e6094bbc999cf5344ba94eb68"
    ):
        raise ValueError("R2 readout helper missing, linked or changed")
    spec = importlib.util.spec_from_file_location(f"readout_r2_{plan.name}", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned R2 readout helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = vars(module)
    common.update(
        _PINNED_HELPER_PATH=str(path),
        SOURCE_SHA256=plan.source_sha,
        SOURCE_MANIFEST_SHA256=plan.source_manifest_sha,
        SOURCE_COMMIT=plan.source_commit,
        SPLIT_SHA256=SPLIT_SHA256,
        CELLS=plan.cells,
        EXPECTED_UPDATES=38_760,
        UPDATES_PER_EPOCH=6_460,
        FIT_SEQUENCES=FIT_SEQUENCES,
        VALID_SEQUENCES=VALID_SEQUENCES,
        VALID_ROWS=VALID_ROWS,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
        EXPORT_TIMEOUT_SECONDS=EXPORT_TIMEOUT_SECONDS,
        SCORE_TIMEOUT_SECONDS=SCORE_TIMEOUT_SECONDS,
        MAX_OUTPUT_BYTES=MAX_OUTPUT_BYTES[plan.name],
        MIN_FREE_BYTES=FREE_RESERVE_BYTES,
        ABS_TOLERANCE=ABS_TOLERANCE,
    )
    return common


@contextmanager
def pinned_config_loader(source: Path) -> Iterator[Any]:
    """Import the normalized config loader from only the verified source capsule."""
    source = source.resolve(strict=True)
    source_src = source / "src"
    module_path = source_src / "wnn/training/sequence_config.py"
    if module_path.is_symlink() or not module_path.is_file():
        raise ValueError("verified source lacks a regular sequence_config.py")
    old_path = sys.path[:]
    old_wnn = {
        name: item
        for name, item in tuple(sys.modules.items())
        if name == "wnn" or name.startswith("wnn.")
    }
    for name in old_wnn:
        del sys.modules[name]
    try:
        sys.path.insert(0, str(source_src))
        module = importlib.import_module("wnn.training.sequence_config")
        if Path(module.__file__).resolve(strict=True) != module_path.resolve(
            strict=True
        ):
            raise ValueError("sequence_config import escaped the pinned capsule")
        yield module.load_config
    finally:
        for name in tuple(sys.modules):
            if name == "wnn" or name.startswith("wnn."):
                del sys.modules[name]
        sys.modules.update(old_wnn)
        sys.path[:] = old_path


def _updates_per_epoch(config: JsonMap) -> int:
    batch = config.get("batch_size")
    chunk = config.get("chunk_rows")
    accumulation = config.get("accumulate_chunks")
    if any(
        type(value) is not int or value <= 0 for value in (batch, chunk, accumulation)
    ):
        raise ValueError("config has malformed batch/chunk/accumulation dimensions")
    return (
        math.ceil(FIT_SEQUENCES / batch)
        * math.ceil(SEQUENCE_LENGTH / chunk)
        * accumulation
    )


def _validate_admission(runtime: RuntimeCohort, source_root: Path) -> JsonMap:
    plan = runtime.plan
    if sha256_file(runtime.admission_path) != runtime.admission_sha:
        raise ValueError(f"{plan.name}: ROOT admission SHA mismatch")
    admission = json.loads(runtime.admission_path.read_text(encoding="utf-8"))
    if (
        admission.get("schema") != plan.admission_schema
        or admission.get("approved") is not True
        or admission.get("max_fresh_fits") != len(plan.cells)
        or admission.get("source_sha256") != plan.source_sha
        or admission.get("source_commit") != plan.source_commit
        or admission.get("source_manifest_sha256", plan.source_manifest_sha)
        != plan.source_manifest_sha
        or admission.get("split_sha256") != SPLIT_SHA256
        or set(admission.get("profiles", {})) != set(plan.cells)
    ):
        raise ValueError(
            f"{plan.name}: ROOT admission differs from the frozen source/split roster"
        )
    if plan.name == "followups4" and (
        admission.get("fit_launch_approved") is not True
        or admission.get("approved_by") != "ROOT"
    ):
        raise ValueError("followups4: R4 admission lacks ROOT fit-launch approval")
    if plan.name == "ema4" and (
        admission.get("fit_launch_approved") is not True
        or admission.get("approved_by") != "ROOT"
        or admission.get("data_sha256_by_cell")
        != {
            "k16_s203301": "158e5e26d6c7609f8631bc953fee9e6a9aaa63396ce934cf25df4dd6569ecf4b",
            "k16_s203302": "158e5e26d6c7609f8631bc953fee9e6a9aaa63396ce934cf25df4dd6569ecf4b",
            "ema999_k16_s203301": "c387e7e77dfe3cc070bec5e9f8067810fdfa2f5b6ddc9e71bf66073cbe4f9f85",
            "ema999_k16_s203302": "c387e7e77dfe3cc070bec5e9f8067810fdfa2f5b6ddc9e71bf66073cbe4f9f85",
        }
    ):
        raise ValueError(
            "ema4: R2 admission lacks ROOT approval or per-cell data identities"
        )
    if admission.get("source_root") not in (None, str(source_root)):
        raise ValueError(f"{plan.name}: admission source root differs")
    return admission


def _validate_fixed_config(plan: Cohort, cell: str, config: JsonMap) -> int:
    model = config.get("model")
    if (
        config.get("seed") != (203301 if cell.endswith("203301") else 203302)
        or config.get("epochs") != EPOCHS
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
        or not isinstance(model, dict)
        or model.get("architecture") != "gru"
        or model.get("input_stem_kernel") != 16
    ):
        raise ValueError(
            f"{cell}: config violates the frozen six-epoch train-only endpoint"
        )
    if plan.name == "first8" and (
        cell.startswith("horizon") and config.get("optimizer") != "adamw"
    ):
        raise ValueError(f"{cell}: first-cohort horizon config changed optimizer")
    if plan.name == "ema4":
        ema = config.get("weight_ema")
        if config.get("optimizer") != "adamw":
            raise ValueError(f"{cell}: EMA comparison changed its AdamW optimizer")
        if cell.startswith("ema999_"):
            if not isinstance(ema, dict) or ema.get("decay") != 0.999:
                raise ValueError(f"{cell}: treatment EMA decay differs from 0.999")
        elif ema is not None:
            raise ValueError(f"{cell}: raw control unexpectedly enabled weight EMA")
    return _updates_per_epoch(config)


def _validate_fit(
    common: dict[str, Any],
    runtime: RuntimeCohort,
    admission: JsonMap,
    cell: str,
    load_config: Any,
    split: JsonMap,
    fit_deadline: float,
) -> JsonMap:
    plan = runtime.plan
    result_path = (
        runtime.fit_root / "control" / plan.results_dirname / f"{cell}-result.json"
    )
    result = common["_read_json"](result_path, 256_000)
    result_time = result.get("at_epoch", result.get("at"))
    if (
        not _finite(result_time)
        or float(result_time) > fit_deadline
        or result_path.stat().st_mtime > fit_deadline
        or result.get("cell") != cell
        or result.get("status") != "fit_complete"
        or result.get("exit_code") != 0
        or result.get("timed_out") is not False
        or result.get("resource_errors") != []
        or result.get("fit_admission_sha256") not in (None, runtime.admission_sha)
    ):
        raise ValueError(
            f"{cell}: ROOT result is late or not a clean fixed-endpoint completion"
        )

    profile = admission["profiles"][cell]
    config_path = runtime.config_root / f"{cell}.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError(f"{cell}: frozen config missing or linked")
    raw_sha = common["sha256_file"](config_path)
    if raw_sha != plan.raw_config_sha[cell] or raw_sha != profile.get(
        "raw_config_sha256"
    ):
        raise ValueError(
            f"{cell}: raw config bytes differ from the fixed roster/admission"
        )
    config = load_config(config_path)
    config_sha = common["canonical_sha256"](config)
    if config_sha != profile.get("config_sha256"):
        raise ValueError(
            f"{cell}: normalized config differs from the ROOT-admitted config"
        )
    updates_per_epoch = _validate_fixed_config(plan, cell, config)
    total_updates = updates_per_epoch * EPOCHS
    common["UPDATES_PER_EPOCH"] = updates_per_epoch
    common["EXPECTED_UPDATES"] = total_updates
    admitted_updates = profile.get("total_updates")
    if admitted_updates is not None and admitted_updates != total_updates:
        raise ValueError(
            f"{cell}: ROOT profile total-updates receipt differs from actual config"
        )
    if result.get("total_updates", total_updates) != total_updates:
        raise ValueError(
            f"{cell}: ROOT result total-updates differs from terminal/config"
        )
    if plan.name == "followups4" and total_updates != 38_760:
        raise ValueError(f"{cell}: PR82 follow-up update count is not 38,760")
    if cell.startswith("horizon4000") and total_updates != 38_730:
        raise ValueError(f"{cell}: horizon4000 ceil-batch update count is not 38,730")
    if cell.startswith("horizon2000") and total_updates != 38_760:
        raise ValueError(f"{cell}: horizon2000 update count is not 38,760")

    fit_dir = runtime.fit_root / "runs" / cell / "fit"
    model_path = fit_dir / "model.pt"
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError(f"{cell}: terminal model missing or linked")
    model_sha = common["sha256_file"](model_path)
    if not _is_sha(model_sha) or result.get("model_sha256") != model_sha:
        raise ValueError(f"{cell}: terminal model SHA differs from ROOT result")
    metadata = common["_read_json"](fit_dir / "run_metadata.json", 2_000_000)
    terminal = common["_read_json"](fit_dir / "terminal_status.json", 2_000_000)
    events = common["_read_events"](fit_dir / "monitor_evaluations.jsonl")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(metadata.get("source"), dict):
        raise TypeError(f"{cell}: source/data provenance is absent")
    if (
        metadata.get("config") != config
        or metadata["source"].get("sha256") != plan.source_sha
        or metadata["source"].get("commit", plan.source_commit) != plan.source_commit
        or provenance.get("world_size") != 1
        or provenance.get("data_sha256")
        != common["canonical_sha256"](provenance.get("data"))
    ):
        raise ValueError(
            f"{cell}: actual run metadata source/config/data identity mismatch"
        )
    data = provenance["data"]
    if not isinstance(data, dict):
        raise TypeError(f"{cell}: run data provenance is not an object")
    if profile.get("data_sha256") != provenance.get("data_sha256"):
        raise ValueError(f"{cell}: profile and fit data-provenance identities differ")
    if (
        not isinstance(data, dict)
        or data.get("split_sha256") != SPLIT_SHA256
        or data.get("teacher") is not None
        or data.get("training_mask") != "all_after_warmup"
        or data.get("fit_members") is None
        or data.get("evaluation_members") is None
    ):
        raise ValueError(f"{cell}: fit ancestry, split or teacher exclusion differs")
    normalization = data.get("normalization")
    objective = data.get("objective_contract")
    if (
        not isinstance(normalization, dict)
        or normalization.get("scope") != "all_fit"
        or normalization.get("sequence_count") != FIT_SEQUENCES
        or not isinstance(objective, dict)
        or objective.get("mode") != "standalone_labels_wmse_v2"
        or objective.get("reference_training_ancestry") != "none"
    ):
        raise ValueError(
            f"{cell}: normalization/objective ancestry is not fitting-only"
        )
    if plan.name == "ema4":
        monitor = data.get("report_only_monitor")
        source = (
            monitor.get("evaluation_weight_source")
            if isinstance(monitor, dict)
            else None
        )
        expected_kind = (
            "per_optimizer_update_ema"
            if cell.startswith("ema999_")
            else "raw_optimizer_parameters"
        )
        if not isinstance(source, dict) or source.get("kind") != expected_kind:
            raise ValueError(
                f"{cell}: fit used the wrong raw/EMA evaluation-weight source"
            )
        if cell.startswith("ema999_") and source.get("decay") != 0.999:
            raise ValueError(f"{cell}: report-only monitor records the wrong EMA decay")
        if not cell.startswith("ema999_") and set(source) != {"kind"}:
            raise ValueError(
                f"{cell}: raw control carries an unexpected EMA declaration"
            )
    data_sources = data.get("sources")
    if not isinstance(data_sources, dict):
        raise TypeError(f"{cell}: input source fingerprints missing")

    fit_members = data["fit_members"]
    valid_members = data["evaluation_members"]
    fit = {
        "cell": cell,
        "source_sha256": plan.source_sha,
        "config_sha256": config_sha,
        "data_sha256": provenance["data_sha256"],
        "model_sha256": model_sha,
        "fit_members": fit_members,
        "evaluation_members": valid_members,
        "data_source_fingerprints": {
            name: data_sources[name]
            for name in ("datasets/train.parquet", "datasets/valid.parquet")
        },
    }
    common["validate_fit_membership"](fit, split)
    if len(events) != EPOCHS:
        raise ValueError(f"{cell}: E1-E6 monitor journal missing or truncated")
    valid_members_sha = hashlib.sha256(
        json.dumps(valid_members, separators=(",", ":")).encode()
    ).hexdigest()
    for epoch, event in enumerate(events, start=1):
        metrics, context = event.get("metrics"), event.get("context")
        if (
            event.get("schema") != "wnn.monitor.v1"
            or event.get("kind") != "evaluation"
            or event.get("epoch") != epoch
            or event.get("step") != epoch * updates_per_epoch
            or not isinstance(metrics, dict)
            or not isinstance(context, dict)
            or any(not common["_finite"](value) for value in metrics.values())
            or metrics.get("valid/sequence_count") != VALID_SEQUENCES
            or metrics.get("valid/selected_rows") != VALID_ROWS
            or not all(
                common["_finite"](metrics.get(key))
                for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1")
            )
            or metrics.get("train_eval/sequence_count") != 32
            or context.get("split_sha256") != SPLIT_SHA256
            or context.get("evaluation_member_count") != VALID_SEQUENCES
            or context.get("evaluation_member_sha256") != valid_members_sha
            or context.get("evaluation_population")
            != "complete official validation membership"
            or context.get("train_eval_mask") != "all_after_warmup"
            or context.get("evaluation_mask") != "is_scored after warmup"
            or context.get("checkpoint_selection")
            != "fixed final epoch 6; monitoring is report only"
            or context.get("endpoint") != "raw standalone z; no reference or blend"
        ):
            raise ValueError(
                f"{cell}: E{epoch} monitor coverage or identity does not match full valid"
            )
        if plan.name == "ema4":
            expected_kind = (
                "per_optimizer_update_ema"
                if cell.startswith("ema999_")
                else "raw_optimizer_parameters"
            )
            weight_source = context.get("evaluation_weight_source")
            if (
                not isinstance(weight_source, dict)
                or weight_source.get("kind") != expected_kind
                or (cell.startswith("ema999_") and weight_source.get("decay") != 0.999)
                or (not cell.startswith("ema999_") and set(weight_source) != {"kind"})
            ):
                raise ValueError(
                    f"{cell}: E{epoch} monitor used the wrong raw/EMA weight source"
                )
    final_metrics = events[-1]["metrics"]
    if "final_metrics" in result and result["final_metrics"] != final_metrics:
        raise ValueError(
            f"{cell}: ROOT result E6 metrics differ from the native event journal"
        )
    identity = terminal.get("identity")
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != EPOCHS
        or terminal.get("global_updates") != total_updates
        or terminal.get("model_sha256") != model_sha
        or not isinstance(identity, dict)
        or identity.get("source_sha256") != plan.source_sha
        or identity.get("config_sha256") != config_sha
        or identity.get("data_sha256") != provenance["data_sha256"]
        or identity.get("world_size") != 1
    ):
        raise ValueError(
            f"{cell}: terminal status does not bind the fixed E6 model/source/config"
        )
    devices = (
        provenance.get("runtime", {}).get("devices")
        if isinstance(provenance.get("runtime"), dict)
        else None
    )
    gpu_uuid = profile.get("gpu_uuid")
    if (
        profile.get("gpu") != plan.gpu_by_cell[cell]
        or not isinstance(gpu_uuid, str)
        or not isinstance(devices, list)
        or len(devices) != 1
        or devices[0].get("name") != "NVIDIA L4"
        or devices[0].get("uuid") != gpu_uuid
    ):
        raise ValueError(f"{cell}: fit did not use its ROOT-bound L4 identity")
    common["validate_project_file_stats"](runtime.fit_root, fit, split["sources"])
    fit.update(
        gpu=profile["gpu"],
        gpu_uuid=gpu_uuid,
        raw_config_sha256=raw_sha,
        final_metrics=final_metrics,
        updates_per_epoch=updates_per_epoch,
        total_updates=total_updates,
        fit_result_sha256=common["sha256_file"](result_path),
        fit_result_path=str(result_path),
        run_metadata_sha256=common["sha256_file"](fit_dir / "run_metadata.json"),
        terminal_status_sha256=common["sha256_file"](fit_dir / "terminal_status.json"),
    )
    return fit


def _identity_is_alive(value: object, is_same_process: Any) -> bool | None:
    """Return None when a process receipt cannot establish a safe liveness check."""
    if not isinstance(value, dict):
        return None
    if (
        type(value.get("pid")) is not int
        or type(value.get("pgid")) is not int
        or not isinstance(value.get("birth"), str)
    ):
        return None
    return bool(is_same_process(value))


def _cell_result_path(runtime: RuntimeCohort, cell: str) -> Path:
    return (
        runtime.fit_root
        / "control"
        / runtime.plan.results_dirname
        / f"{cell}-result.json"
    )


def _controller_settled(
    runtime: RuntimeCohort,
    fit_deadline: float,
    now: float,
    is_same_process: Any,
) -> tuple[bool, str, str | None]:
    """Accept a completion receipt or, after cutoff, prove controller and children stopped."""
    control = runtime.fit_root / "control" / runtime.plan.results_dirname
    complete = control / runtime.plan.complete_name
    owner_path = control / "owner.json"
    owned_root = control / "owned"
    complete_sha = None
    if complete.is_file() and not complete.is_symlink():
        try:
            complete_sha = sha256_file(complete)
        except OSError:
            pass  # The controller's completion marker is diagnostic, not a fit gate.
        return True, "completion_receipt_present", complete_sha

    if now < fit_deadline:
        return False, "fit_deadline_not_reached", None
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "missing_or_invalid_controller_owner", None
    owner_alive = (
        False
        if owner.get("status") == "exited_before_identity_capture"
        else _identity_is_alive(owner, is_same_process)
    )
    if owner_alive is None or owner_alive:
        return False, "controller_still_running_or_unverified", None
    if owned_root.exists():
        for receipt_path in owned_root.glob("*.json"):
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                alive = _identity_is_alive(receipt.get("child"), is_same_process)
            except (OSError, json.JSONDecodeError):
                return False, "unreadable_child_receipt", None
            if alive is None or alive:
                return False, "owned_fit_child_still_running_or_unverified", None
    return True, "fit_cutoff_controller_and_children_stopped", None


def _projected_prediction_bytes(cell_count: int) -> int:
    per_cell = VALID_SEQUENCES * SEQUENCE_LENGTH * 2 * 4
    return cell_count * (per_cell + 10_000_000)


def _directory_bytes(root: Path) -> int:
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [
            name for name in directories if not (Path(folder) / name).is_symlink()
        ]
        for name in files:
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def _gate_output_disk_batch(
    candidates: list[tuple[RuntimeCohort, str, JsonMap]],
) -> tuple[list[tuple[RuntimeCohort, str, JsonMap]], dict[str, str]]:
    """Reserve only this batch; an underfilled mount does not block other mounts."""
    blocked: dict[str, str] = {}
    by_device: dict[int, list[tuple[RuntimeCohort, str, JsonMap]]] = {}
    by_root: dict[Path, list[tuple[RuntimeCohort, str, JsonMap]]] = {}
    for candidate in candidates:
        runtime, _cell, _fit = candidate
        runtime.output_root.mkdir(parents=True, exist_ok=True)
        by_device.setdefault(runtime.output_root.stat().st_dev, []).append(candidate)
        by_root.setdefault(runtime.output_root, []).append(candidate)
    root_blocked: set[Path] = set()
    for root, group in by_root.items():
        existing = _directory_bytes(root)
        projected = _projected_prediction_bytes(len(group))
        plan = group[0][0].plan
        if existing + projected > MAX_OUTPUT_BYTES[plan.name]:
            for _runtime, cell, _fit in group:
                blocked[cell] = (
                    f"{plan.name}: projected output exceeds the bounded readout root"
                )
            root_blocked.add(root)
    allowed: list[tuple[RuntimeCohort, str, JsonMap]] = []
    for device_group in by_device.values():
        viable = [
            item for item in device_group if item[0].output_root not in root_blocked
        ]
        if not viable:
            continue
        free = shutil.disk_usage(viable[0][0].output_root).free
        projected = _projected_prediction_bytes(len(viable))
        if free < projected + FREE_RESERVE_BYTES:
            roots = ",".join(sorted({item[0].plan.name for item in viable}))
            reason = (
                f"readout filesystem reserve too low for {roots}: free={free}, "
                f"projected={projected}, reserve={FREE_RESERVE_BYTES}"
            )
            for _runtime, cell, _fit in viable:
                blocked[cell] = reason
            continue
        allowed.extend(viable)
    return allowed, blocked


def _install_score_bytes_if_absent(
    score_path: Path, fit_score_path: Path, expected_sha256: str
) -> str:
    """Install exact scorer bytes without replacing any existing fit-root score."""
    if fit_score_path.is_symlink():
        return "existing_target_is_symlink"
    if fit_score_path.exists():
        if not fit_score_path.is_file():
            return "existing_target_is_not_regular_file"
        return (
            "existing_identical"
            if sha256_file(fit_score_path) == expected_sha256
            else "existing_sha_mismatch"
        )

    payload = score_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("canonical score bytes changed before fit-root attachment")
    fit_score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = fit_score_path.with_name(
        f".{fit_score_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, fit_score_path)
        except FileExistsError:
            if fit_score_path.is_symlink() or not fit_score_path.is_file():
                return "concurrent_target_not_regular_file"
            return (
                "concurrent_existing_identical"
                if sha256_file(fit_score_path) == expected_sha256
                else "concurrent_existing_sha_mismatch"
            )
    finally:
        temporary.unlink(missing_ok=True)
    return "created_exact_copy"


def _readout_one(
    common: dict[str, Any],
    runtime: RuntimeCohort,
    fit: JsonMap,
    split: JsonMap,
    python: Path,
    control: Path,
    owner: JsonMap,
    guard: Any,
    atomic_write: Any,
    hold_gpu_slot: Any,
    gpu_lock_root: Path,
    wave_deadline: float,
) -> JsonMap:
    cell = fit["cell"]
    common = _load_r2(Path(common["_PINNED_HELPER_PATH"]), runtime.plan)
    common["EXPECTED_UPDATES"] = int(fit["total_updates"])
    common["UPDATES_PER_EPOCH"] = int(fit["updates_per_epoch"])
    result_path = control / f"{cell}-result.json"
    readout_dir = runtime.output_root / "runs" / cell / "final-readout-r3"
    try:
        if (
            result_path.exists()
            or result_path.is_symlink()
            or readout_dir.exists()
            or readout_dir.is_symlink()
        ):
            raise FileExistsError(
                "readout artifacts already exist; automatic retry is disabled"
            )
        if wave_deadline - time.time() < EXPORT_TIMEOUT_SECONDS + SCORE_TIMEOUT_SECONDS:
            raise TimeoutError("fixed CLOCK_R1 wave leaves less than 1,800 seconds")
        claim = {
            "schema": "wnn.l4-canonical-readout-claim.r2",
            "cohort": runtime.plan.name,
            "cell": cell,
            "gpu": fit["gpu"],
            "gpu_uuid": fit["gpu_uuid"],
            "source_commit": runtime.plan.source_commit,
            "source_sha256": runtime.plan.source_sha,
            "source_manifest_sha256": runtime.plan.source_manifest_sha,
            "split_sha256": SPLIT_SHA256,
            "config_sha256": fit["config_sha256"],
            "model_sha256": fit["model_sha256"],
            "fit_result_sha256": fit["fit_result_sha256"],
            "controller_owner": owner,
            "created_at_epoch": time.time(),
        }
        common["_write_atomic"](atomic_write, control / f"claim-{cell}.json", claim)
        readout_dir.mkdir(parents=True, exist_ok=False)
        prediction_dir = readout_dir / "predictions"
        owner_cell = {
            "controller": owner,
            "cell_id": cell,
            "source_sha256": runtime.plan.source_sha,
            "config_sha256": fit["config_sha256"],
            "model_sha256": fit["model_sha256"],
        }
        predict = [
            str(python),
            str(runtime.source_root / "scripts/predict_sequence.py"),
            "--checkpoint",
            str(runtime.fit_root / "runs" / cell / "fit/model.pt"),
            "--data-root",
            str(runtime.data_root),
            "--split-manifest",
            str(runtime.data_root / "chronological_primary.json"),
            "--output",
            str(prediction_dir),
            "--membership",
            "evaluation",
        ]
        with hold_gpu_slot(gpu_lock_root, str(fit["gpu"])):
            export = common["_command"](
                predict,
                source=runtime.source_root,
                control=control,
                output_dir=prediction_dir,
                cell=cell,
                gpu=fit["gpu"],
                owner=owner_cell,
                stage="canonical_readout_export",
                timeout=min(EXPORT_TIMEOUT_SECONDS, int(wave_deadline - time.time())),
                guard=guard,
                gpu_uuid=fit["gpu_uuid"],
            )
        score_path = readout_dir / "score.json"
        score_command = [
            str(python),
            str(runtime.source_root / "scripts/score_sequence.py"),
            "--data-root",
            str(runtime.data_root),
            "--split-manifest",
            str(runtime.data_root / "chronological_primary.json"),
            "--predictions",
            str(prediction_dir),
            "--output",
            str(score_path),
        ]
        score_run = common["_command"](
            score_command,
            source=runtime.source_root,
            control=control,
            output_dir=readout_dir,
            cell=cell,
            gpu=fit["gpu"],
            owner=owner_cell,
            stage="canonical_readout_score",
            timeout=min(SCORE_TIMEOUT_SECONDS, int(wave_deadline - time.time())),
            guard=guard,
            gpu_uuid=None,
        )
        if (
            common["sha256_file"](runtime.fit_root / "runs" / cell / "fit/model.pt")
            != fit["model_sha256"]
        ):
            raise ValueError("E6 model bytes changed during prediction export")
        manifest_path = prediction_dir / "manifest.json"
        manifest = common["_read_json"](manifest_path, 32_000_000)
        manifest_sha = common["sha256_file"](manifest_path)
        common["validate_prediction_manifest"](manifest, fit, manifest_sha)
        score_receipt = common["_read_json"](score_path, 2_000_000)
        comparison = common["validate_score_receipt"](
            score_receipt, manifest, manifest_sha, fit
        )
        status = (
            "readout_complete" if comparison["within_tolerance"] else "monitor_mismatch"
        )
        score_sha = common["sha256_file"](score_path)
        fit_dir = runtime.fit_root / "runs" / cell / "fit"
        fit_score_path = fit_dir / "score.json"
        attachment: JsonMap = {
            "status": "not_attached_score_mismatch",
            "fit_score_path": str(fit_score_path),
        }
        if status == "readout_complete" and comparison["within_tolerance"] is True:
            attachment_status = _install_score_bytes_if_absent(
                score_path, fit_score_path, score_sha
            )
            installed_sha = (
                common["sha256_file"](fit_score_path)
                if fit_score_path.is_file() and not fit_score_path.is_symlink()
                else None
            )
            attachment = {
                "status": attachment_status,
                "fit_score_path": str(fit_score_path),
                "source_score_sha256": score_sha,
                "fit_score_sha256": installed_sha,
                "sidecar_path": str(fit_dir / "canonical_readout_attachment.json"),
            }
        receipt = {
            "schema": "wnn.l4-canonical-readout-cell.r2",
            "status": status,
            "cohort": runtime.plan.name,
            "cell": cell,
            "source_commit": runtime.plan.source_commit,
            "source_sha256": runtime.plan.source_sha,
            "source_manifest_sha256": runtime.plan.source_manifest_sha,
            "split_sha256": SPLIT_SHA256,
            "data_sha256": fit["data_sha256"],
            "config_sha256": fit["config_sha256"],
            "raw_config_sha256": fit["raw_config_sha256"],
            "model_sha256": fit["model_sha256"],
            "fit_result_sha256": fit["fit_result_sha256"],
            "terminal_status_sha256": fit["terminal_status_sha256"],
            "prediction_manifest_sha256": manifest_sha,
            "score_receipt_sha256": score_sha,
            "prediction_sequences": len(manifest["sequences"]),
            "prediction_shape": [VALID_SEQUENCES, SEQUENCE_LENGTH, 2],
            "prediction_dtype": "float32",
            "scored_rows": comparison["selected_rows"],
            "score": comparison,
            "monitor_e6": fit["final_metrics"],
            "fit_score_attachment": attachment,
            "export_command": export,
            "score_command": score_run,
            "interpretation": "Canonical full official-valid re-export and scorer agreement at fixed E6; exposed development endpoint, not independent validation or public-score evidence.",
        }
        common["_write_atomic"](
            atomic_write, readout_dir / "readout-receipt.json", receipt
        )
        if status == "readout_complete" and comparison["within_tolerance"] is True:
            cell_receipt_path = readout_dir / "readout-receipt.json"
            sidecar = {
                "schema": "wnn.l4-canonical-readout-attachment.r2",
                "status": attachment["status"],
                "cohort": runtime.plan.name,
                "cell": cell,
                "source_score_path": str(score_path),
                "source_score_sha256": score_sha,
                "fit_score_path": str(fit_score_path),
                "fit_score_sha256": attachment.get("fit_score_sha256"),
                "validated_cell_receipt_path": str(cell_receipt_path),
                "validated_cell_receipt_sha256": common["sha256_file"](
                    cell_receipt_path
                ),
                "score_within_tolerance": True,
                "automatic_retry": False,
            }
            common["_write_atomic"](
                atomic_write, fit_dir / "canonical_readout_attachment.json", sidecar
            )
        common["_write_atomic"](atomic_write, result_path, receipt)
        return receipt
    except Exception as error:  # noqa: BLE001 вЂ” persist one terminal readout outcome per cell
        failure = {
            "schema": "wnn.l4-canonical-readout-cell.r2",
            "status": "readout_failed",
            "cohort": runtime.plan.name,
            "cell": cell,
            "failure_kind": "provenance_or_execution",
            "error": f"{type(error).__name__}: {error}",
            "automatic_retry": False,
        }
        if not result_path.exists():
            common["_write_atomic"](atomic_write, result_path, failure)
        return failure


def _make_runtime(args: argparse.Namespace, plan: Cohort) -> RuntimeCohort:
    prefix = {
        "first8": "first",
        "followups4": "followup",
        "ema4": "ema",
    }[plan.name]
    fit_root = getattr(args, f"{prefix}_fit_root").resolve()
    output_root = getattr(args, f"{prefix}_readout_root").resolve()
    source_root = getattr(args, f"{prefix}_source").resolve(strict=True)
    data_root = getattr(args, f"{prefix}_data_root").resolve(strict=True)
    config_root = getattr(args, f"{prefix}_config_root").resolve(strict=True)
    admission_path = getattr(args, f"{prefix}_admission").resolve(strict=True)
    admission_sha = getattr(args, f"{prefix}_admission_sha")
    if not _is_sha(admission_sha):
        raise ValueError(f"{prefix} ROOT admission SHA must be supplied")
    return RuntimeCohort(
        plan,
        fit_root,
        output_root,
        source_root,
        data_root,
        config_root,
        admission_path,
        admission_sha,
    )


def _validate_clock(path: Path) -> tuple[JsonMap, float, float]:
    if sha256_file(path) != CLOCK_SHA256:
        raise ValueError("shared CLOCK_R1 bytes changed")
    clock = json.loads(path.read_text(encoding="utf-8"))
    if (
        clock.get("schema") != "wnn.cluster-wave-clock.r1"
        or clock.get("campaign") != "all-experiments-20261112"
        or clock.get("fit_deadline_epoch") != 1_794_468_019.060462
        or clock.get("wave_deadline_epoch") != 1_794_471_619.060462
    ):
        raise ValueError(
            "shared fixed fit/wave clock differs from the accepted contract"
        )
    return (
        clock,
        float(clock["fit_deadline_epoch"]),
        float(clock["wave_deadline_epoch"]),
    )


def run(args: argparse.Namespace) -> JsonMap:
    """Independently qualify each completed E6 cell and retain partial readout results."""
    first, followups, ema = (
        _make_runtime(args, FIRST),
        _make_runtime(args, FOLLOWUPS),
        _make_runtime(args, EMA),
    )
    runtimes = (first, followups, ema)
    helper_path = args.r2_helper.resolve(strict=True)
    python = Path(os.path.abspath(os.fspath(args.python)))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(
            f"pinned native Python is missing/not executable: {python}"
        )
    _clock, fit_deadline, wave_deadline = _validate_clock(
        args.clock.resolve(strict=True)
    )

    sys.path.insert(0, str(args.old_src.resolve(strict=True)))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_runtime import (
        current_process_owner,
        hold_gpu_slot,
        is_same_process,
        launch_detached_worker,
    )

    control = first.output_root / "control/canonical-readouts-r3"
    control.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [
            str(python),
            str(Path(__file__).resolve()),
            *[item for item in sys.argv[1:] if item != "--detach"],
            "--worker",
        ]
        print(
            json.dumps(
                launch_detached_worker(
                    command,
                    cwd=first.output_root,
                    env=dict(os.environ),
                    receipt_path=control / "owner.json",
                    metadata={
                        "cohorts": [FIRST.name, FOLLOWUPS.name, EMA.name],
                        "cells": 16,
                        "clock_sha256": CLOCK_SHA256,
                    },
                )
            )
        )
        return {"status": "detached"}

    with file_lock(control / "readout.lock", blocking=False):
        if (control / "complete.json").exists() or (
            (control / "owner.json").exists() and not args.worker
        ):
            raise FileExistsError(
                "R2 canonical receiver already owns this output root; no retry"
            )
        owner = current_process_owner()
        atomic_write_json(
            control / "controller-owner.json",
            {
                "schema": "wnn.l4-canonical-readout-owner.r2",
                "owner": owner,
                "python_lexical_path": str(python),
                "clock_sha256": CLOCK_SHA256,
                "fit_deadline_epoch": fit_deadline,
                "wave_deadline_epoch": wave_deadline,
                "cohorts": [FIRST.name, FOLLOWUPS.name, EMA.name],
            },
            replace=False,
        )
        gpu_lock_root = control
        states: dict[str, dict[str, Any]] = {}
        for runtime in runtimes:
            plan = runtime.plan
            readout_control = runtime.output_root / "control/canonical-readouts-r3"
            readout_control.mkdir(parents=True, exist_ok=True)
            state: dict[str, Any] = {
                "runtime": runtime,
                "readout_control": readout_control,
                "outcomes": {},
                "fit_cache": {},
                "common": None,
                "split": None,
                "guard": None,
                "setup_error": None,
            }
            try:
                common = _load_r2(helper_path, plan)
                verifier = runpy.run_path(
                    str(
                        runtime.source_root
                        / "src/wnn/training/sequence_source_capsule.py"
                    )
                )["verify_source_capsule"]
                verifier(
                    runtime.source_root,
                    runtime.source_root / plan.manifest_name,
                    plan.source_manifest_sha,
                    plan.source_commit,
                )
                admission = _validate_admission(runtime, runtime.source_root)
                split_path = runtime.data_root / "chronological_primary.json"
                if common["sha256_file"](split_path) != SPLIT_SHA256:
                    raise ValueError(
                        f"{plan.name}: prepared chronological split bytes changed"
                    )
                split = common["_read_json"](split_path, 8_000_000)
                state.update(
                    common=common,
                    admission=admission,
                    split=split,
                    guard=common["ReadoutGuard"](runtime.output_root),
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                state["setup_error"] = f"{type(error).__name__}: {error}"
                for cell in plan.cells:
                    state["outcomes"][cell] = {
                        "schema": "wnn.l4-canonical-readout-cell.r2",
                        "status": "cohort_setup_rejected",
                        "cohort": plan.name,
                        "cell": cell,
                        "error": state["setup_error"],
                        "automatic_retry": False,
                    }
            states[plan.name] = state

        def fit_result(runtime: RuntimeCohort, cell: str) -> JsonMap | None:
            path = _cell_result_path(runtime, cell)
            if path.is_symlink():
                return {"status": "__invalid_fit_result__"}
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, json.JSONDecodeError):
                return {"status": "__invalid_fit_result__"}
            return (
                value
                if isinstance(value, dict)
                else {"status": "__invalid_fit_result__"}
            )

        def active_fit_gpus(
            settled_by_cohort: dict[str, tuple[bool, str, str | None]],
        ) -> set[int]:
            active: set[int] = set()
            for runtime in runtimes:
                settled = settled_by_cohort[runtime.plan.name][0]
                owned_root = (
                    runtime.fit_root
                    / "control"
                    / runtime.plan.results_dirname
                    / "owned"
                )
                for cell in runtime.plan.cells:
                    result = fit_result(runtime, cell)
                    if result is not None and result.get("status") in {
                        "fit_complete",
                        "failed_or_partial",
                    }:
                        continue
                    receipt_path = owned_root / f"{cell}.json"
                    alive: bool | None = None
                    if receipt_path.is_file():
                        try:
                            receipt = json.loads(
                                receipt_path.read_text(encoding="utf-8")
                            )
                            alive = _identity_is_alive(
                                receipt.get("child"), is_same_process
                            )
                        except (OSError, json.JSONDecodeError):
                            alive = None
                    if alive is True or not settled:
                        active.add(runtime.plan.gpu_by_cell[cell])
            return active

        pending_reasons: dict[tuple[str, str], str] = {}
        while True:
            now = time.time()
            settled_by_cohort = {
                runtime.plan.name: _controller_settled(
                    runtime, fit_deadline, now, is_same_process
                )
                for runtime in runtimes
            }
            active_gpus = active_fit_gpus(settled_by_cohort)
            candidates: list[tuple[RuntimeCohort, str, JsonMap]] = []
            pending_reasons = {}
            selected_gpus: set[int] = set()

            for runtime in runtimes:
                plan = runtime.plan
                state = states[plan.name]
                outcomes: dict[str, JsonMap] = state["outcomes"]
                if state["setup_error"]:
                    continue
                readout_control: Path = state["readout_control"]
                common: dict[str, Any] = state["common"]
                split: JsonMap = state["split"]

                for cell in plan.cells:
                    if cell in outcomes:
                        continue
                    receiver_result = readout_control / f"{cell}-result.json"
                    if receiver_result.exists():
                        try:
                            receipt = json.loads(
                                receiver_result.read_text(encoding="utf-8")
                            )
                            outcomes[cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": "prior_readout_artifact_preserved",
                                "cell": cell,
                                "prior_status": receipt.get("status"),
                                "prior_result_sha256": common["sha256_file"](
                                    receiver_result
                                ),
                                "automatic_retry": False,
                            }
                        except (OSError, json.JSONDecodeError):
                            outcomes[cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": "prior_readout_artifact_invalid",
                                "cell": cell,
                                "automatic_retry": False,
                            }
                        continue

                    result_path = _cell_result_path(runtime, cell)
                    result = fit_result(runtime, cell)
                    if (
                        result is None
                        or result.get("status") == "__invalid_fit_result__"
                    ):
                        if settled_by_cohort[plan.name][0]:
                            status = (
                                "fit_result_invalid"
                                if result is not None
                                else "fit_result_missing_after_controller_settled"
                            )
                            outcomes[cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": status,
                                "cell": cell,
                                "fit_result_path": str(result_path),
                                "controller_settlement": settled_by_cohort[plan.name][
                                    1
                                ],
                                "automatic_retry": False,
                            }
                        else:
                            pending_reasons[(plan.name, cell)] = (
                                "fit_result_not_published_or_controller_unsettled"
                            )
                        continue

                    if result.get("status") != "fit_complete":
                        if result.get("status") in {"failed_or_partial", "fit_failed"}:
                            outcomes[cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": "fit_failed_or_partial",
                                "cohort": plan.name,
                                "cell": cell,
                                "root_fit_status": result.get("status"),
                                "fit_result_sha256": common["sha256_file"](result_path),
                                "automatic_retry": False,
                            }
                        elif settled_by_cohort[plan.name][0]:
                            outcomes[cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": "fit_result_invalid",
                                "cohort": plan.name,
                                "cell": cell,
                                "root_fit_status": result.get("status"),
                                "fit_result_sha256": common["sha256_file"](result_path),
                                "automatic_retry": False,
                            }
                        else:
                            pending_reasons[(plan.name, cell)] = (
                                "fit_result_not_terminal"
                            )
                        continue

                    if cell not in state["fit_cache"]:
                        try:
                            with pinned_config_loader(
                                runtime.source_root
                            ) as load_config:
                                fit = _validate_fit(
                                    common,
                                    runtime,
                                    state["admission"],
                                    cell,
                                    load_config,
                                    split,
                                    fit_deadline,
                                )
                            state["fit_cache"][cell] = fit
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            outcomes[cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": "fit_evidence_rejected",
                                "cohort": plan.name,
                                "cell": cell,
                                "failure_kind": "fit_provenance_or_endpoint",
                                "error": f"{type(error).__name__}: {error}",
                                "automatic_retry": False,
                            }
                            continue

                    gpu = plan.gpu_by_cell[cell]
                    if gpu in active_gpus:
                        pending_reasons[(plan.name, cell)] = (
                            f"gpu_{gpu}_still_has_an_owned_fit"
                        )
                        continue
                    if gpu in selected_gpus:
                        pending_reasons[(plan.name, cell)] = (
                            f"gpu_{gpu}_already_selected_for_this_readout_batch"
                        )
                        continue
                    selected_gpus.add(gpu)
                    candidates.append((runtime, cell, state["fit_cache"][cell]))

            allowed, disk_blocked = _gate_output_disk_batch(candidates)
            for cell, reason in disk_blocked.items():
                cohort = next(
                    runtime.plan.name
                    for runtime, candidate_cell, _fit in candidates
                    if candidate_cell == cell
                )
                states[cohort]["outcomes"][cell] = {
                    "schema": "wnn.l4-canonical-readout-cell.r2",
                    "status": "readout_blocked_disk_reserve",
                    "cohort": cohort,
                    "cell": cell,
                    "reason": reason,
                    "automatic_retry": False,
                }

            if allowed:
                futures: dict[Any, tuple[RuntimeCohort, str]] = {}
                with ThreadPoolExecutor(
                    max_workers=min(8, len(allowed)),
                    thread_name_prefix="canonical-readout-r2",
                ) as pool:
                    for runtime, cell, fit in allowed:
                        state = states[runtime.plan.name]
                        future = pool.submit(
                            _readout_one,
                            state["common"],
                            runtime,
                            fit,
                            state["split"],
                            python,
                            state["readout_control"],
                            owner,
                            state["guard"],
                            atomic_write_json,
                            hold_gpu_slot,
                            gpu_lock_root,
                            wave_deadline,
                        )
                        futures[future] = (runtime, cell)
                    for future in as_completed(futures):
                        runtime, cell = futures[future]
                        try:
                            states[runtime.plan.name]["outcomes"][cell] = (
                                future.result()
                            )
                        except Exception as error:  # noqa: BLE001
                            states[runtime.plan.name]["outcomes"][cell] = {
                                "schema": "wnn.l4-canonical-readout-cell.r2",
                                "status": "readout_failed",
                                "cohort": runtime.plan.name,
                                "cell": cell,
                                "failure_kind": "receiver_worker",
                                "error": f"{type(error).__name__}: {error}",
                                "automatic_retry": False,
                            }

            unfinished = [
                (runtime.plan.name, cell)
                for runtime in runtimes
                for cell in runtime.plan.cells
                if cell not in states[runtime.plan.name]["outcomes"]
            ]
            if not unfinished:
                break
            now = time.time()
            if now >= wave_deadline:
                for cohort, cell in unfinished:
                    reason = pending_reasons.get((cohort, cell), "readout_not_started")
                    if reason.startswith("gpu_"):
                        status = "readout_gpu_busy_at_wave_deadline"
                    elif "controller_unsettled" in reason or "not_published" in reason:
                        status = "fit_controller_unsettled_at_wave_deadline"
                    else:
                        status = "readout_not_started_at_wave_deadline"
                    states[cohort]["outcomes"][cell] = {
                        "schema": "wnn.l4-canonical-readout-cell.r2",
                        "status": status,
                        "cell": cell,
                        "reason": reason,
                        "automatic_retry": False,
                    }
                break
            time.sleep(min(10.0, max(0.1, wave_deadline - now)))

        cohorts_outcomes: dict[str, JsonMap] = {}
        for runtime in runtimes:
            plan = runtime.plan
            state = states[plan.name]
            outcomes: dict[str, JsonMap] = state["outcomes"]
            settled, settlement_reason, complete_sha = _controller_settled(
                runtime, fit_deadline, time.time(), is_same_process
            )
            passed = sum(
                value.get("status") == "readout_complete" for value in outcomes.values()
            )
            summary = {
                "schema": "wnn.l4-canonical-readout-cohort.r2",
                "status": "complete"
                if passed == len(plan.cells)
                else "partial_or_failed",
                "cohort": plan.name,
                "source_commit": plan.source_commit,
                "source_sha256": plan.source_sha,
                "source_manifest_sha256": plan.source_manifest_sha,
                "split_sha256": SPLIT_SHA256,
                "admission_sha256": runtime.admission_sha,
                "fit_deadline_epoch": fit_deadline,
                "wave_deadline_epoch": wave_deadline,
                "fit_controller_settled": settled,
                "fit_controller_settlement": settlement_reason,
                "fit_controller_complete_sha256": complete_sha,
                "fixed_endpoint": {
                    "epochs": EPOCHS,
                    "fit_sequences": FIT_SEQUENCES,
                    "valid_sequences": VALID_SEQUENCES,
                    "valid_rows": VALID_ROWS,
                },
                "cells": outcomes,
                "interpretation": "Canonical scorer agreement on the exposed official-valid development partition; no independent-validation or public-score claim.",
            }
            atomic_write_json(
                state["readout_control"] / "complete.json", summary, replace=False
            )
            cohorts_outcomes[plan.name] = summary

        final = {
            "schema": "wnn.l4-canonical-readout-summary.r2",
            "status": "complete"
            if all(
                item.get("status") == "complete" for item in cohorts_outcomes.values()
            )
            else "partial_or_failed",
            "clock_sha256": CLOCK_SHA256,
            "fit_deadline_epoch": fit_deadline,
            "wave_deadline_epoch": wave_deadline,
            "all_fit_results_must_succeed": False,
            "disk_reserve_bytes": FREE_RESERVE_BYTES,
            "cohorts": cohorts_outcomes,
        }
        atomic_write_json(control / "complete.json", final, replace=False)
        return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detach", action="store_true", help="leave a birth-receipted durable waiter"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--first-fit-root", type=Path, required=True)
    parser.add_argument("--first-readout-root", type=Path, required=True)
    parser.add_argument("--first-source", type=Path, required=True)
    parser.add_argument("--first-data-root", type=Path, required=True)
    parser.add_argument("--first-config-root", type=Path, required=True)
    parser.add_argument("--first-admission", type=Path, required=True)
    parser.add_argument("--first-admission-sha", required=True)
    parser.add_argument("--followup-fit-root", type=Path, required=True)
    parser.add_argument("--followup-readout-root", type=Path, required=True)
    parser.add_argument("--followup-source", type=Path, required=True)
    parser.add_argument("--followup-data-root", type=Path, required=True)
    parser.add_argument("--followup-config-root", type=Path, required=True)
    parser.add_argument("--followup-admission", type=Path, required=True)
    parser.add_argument("--followup-admission-sha", required=True)
    parser.add_argument("--ema-fit-root", type=Path, required=True)
    parser.add_argument("--ema-readout-root", type=Path, required=True)
    parser.add_argument("--ema-source", type=Path, required=True)
    parser.add_argument("--ema-data-root", type=Path, required=True)
    parser.add_argument("--ema-config-root", type=Path, required=True)
    parser.add_argument("--ema-admission", type=Path, required=True)
    parser.add_argument("--ema-admission-sha", required=True)
    parser.add_argument("--clock", type=Path, required=True)
    parser.add_argument("--r2-helper", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--old-src", type=Path, required=True)
    return parser


def main() -> int:
    result = run(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {"detached", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
