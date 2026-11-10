"""Re-export and independently score the six admitted L4 fixed-E6 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SOURCE_SHA256 = "b05a73fc8a8a7a4f3f46962e31fd81ee6b401db715109e36d4f0d6e32bf35140"
SOURCE_MANIFEST_SHA256 = "d1956aedfc83b04c51294bd4bdd8a84bd936356281cabad46f4afcce6697280f"
SOURCE_COMMIT = "9a5d2e7ca2549c24ff8217d5062df6de49cce9b5"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CELLS = (
    "shared_control_s203201",
    "shared_control_s203202",
    "t0_specialist_s203201",
    "t0_specialist_s203202",
    "t1_specialist_s203201",
    "t1_specialist_s203202",
)
GPU_BINDINGS = {
    "shared_control_s203201": (0, "222adc44-ecd3-a645-cf7e-ddda9d495f7a"),
    "shared_control_s203202": (1, "bce838c4-097e-0623-7b67-6a97f3778510"),
    "t0_specialist_s203201": (2, "9fd565e2-ebb2-35be-c628-d971a575aa7e"),
    "t0_specialist_s203202": (3, "02d8270f-e4ee-1363-8232-15c0726fdab1"),
    "t1_specialist_s203201": (4, "9943f86e-f085-2d36-2c12-8d2a22111ef0"),
    "t1_specialist_s203202": (5, "48b463a4-5b0a-e277-9ef1-ee37be8015b1"),
}
EXPECTED_UPDATES = 38_760
UPDATES_PER_EPOCH = 6_460
FIT_SEQUENCES = 10_325
VALID_SEQUENCES = 1_873
VALID_ROWS = 3_528_273
SEQUENCE_LENGTH = 20_000
PREDICTION_BYTES_PER_CELL = SEQUENCE_LENGTH * 2 * 4 * VALID_SEQUENCES
EXPORT_TIMEOUT_SECONDS = 600
SCORE_TIMEOUT_SECONDS = 600
MAX_OUTPUT_BYTES = 6_000_000_000
MIN_FREE_BYTES = 4_000_000_000
ABS_TOLERANCE = 1e-5
RESULT_SCHEMA = "wnn.l4-next70-readout-cell.v1"
SUMMARY_SCHEMA = "wnn.l4-next70-readout-comparison.v1"
RECEIVER_SOURCE = Path(
    "/data/lc-epanov_s/tabdata-next70-20261110-r1/source-specialist-r2"
)
RECEIVER_PYTHON = Path("/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python")
OLD_SOURCE = Path(
    "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/"
    "source-gru-monitor-resume-r1/src"
)

JsonMap = dict[str, Any]


def canonical_sha256(value: object) -> str:
    """Match the canonical source/config/data identity hash used by the trainer."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file in fixed-size blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _read_json(path: Path, maximum_bytes: int = 2_000_000) -> JsonMap:
    """Read a bounded regular JSON object without following a symbolic link."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked JSON artifact: {path}")
    info = path.stat()
    if info.st_size > maximum_bytes:
        raise ValueError(f"JSON artifact exceeds {maximum_bytes} bytes: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def _read_events(path: Path) -> list[JsonMap]:
    """Read the exactly-six bounded epoch evaluation stream."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("full-valid event journal is missing, linked or over its 1 MB bound")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("full-valid event journal ends with a partial line")
    events: list[JsonMap] = []
    for line in raw.splitlines(keepends=True):
        if len(line) > 32_768:
            raise ValueError("full-valid event exceeds its 32 KB line bound")
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("full-valid event is not an object")
        events.append(event)
    return events


def validate_admission(admission: JsonMap) -> None:
    """Check the exact ROOT-approved source/split and six disjoint L4 cells."""
    profiles = admission.get("profiles")
    if (
        admission.get("schema") != "wnn.l4-next70-root-fit-admission.v1"
        or admission.get("approved") is not True
        or admission.get("source_sha256") != SOURCE_SHA256
        or admission.get("split_sha256") != SPLIT_SHA256
        or admission.get("max_fresh_fits") != 6
        or not isinstance(profiles, dict)
        or set(profiles) != set(CELLS)
    ):
        raise ValueError("ROOT admission differs from the frozen six-cell readout plan")
    used_gpus: set[int] = set()
    used_uuids: set[str] = set()
    for cell in CELLS:
        profile = profiles[cell]
        gpu, uuid = GPU_BINDINGS[cell]
        if (
            not isinstance(profile, dict)
            or profile.get("gpu") != gpu
            or profile.get("gpu_uuid") != uuid
            or not _is_sha256(profile.get("config_sha256"))
        ):
            raise ValueError(f"ROOT admission binding is invalid for {cell}")
        used_gpus.add(gpu)
        used_uuids.add(uuid)
    if len(used_gpus) != 6 or len(used_uuids) != 6:
        raise ValueError("ROOT admission reuses a GPU or UUID across readout workers")


def validate_fit_evidence(
    cell: str,
    profile: JsonMap,
    fit_result: JsonMap,
    metadata: JsonMap,
    terminal: JsonMap,
    model_sha256: str,
    events: list[JsonMap],
) -> JsonMap:
    """Bind ROOT completion, actual checkpoint bytes, source/config and all six full-valid events."""
    if cell not in CELLS:
        raise ValueError("fit cell is outside the six-cell plan")
    if (
        fit_result.get("cell") != cell
        or fit_result.get("status") != "fit_complete"
        or type(fit_result.get("exit_code")) is not int
        or fit_result.get("exit_code") != 0
        or fit_result.get("timed_out") is not False
        or fit_result.get("resource_errors") != []
        or not _is_sha256(fit_result.get("model_sha256"))
        or fit_result.get("model_sha256") != model_sha256
    ):
        raise ValueError("ROOT fit result does not prove one completed fixed endpoint")
    config, source, provenance = metadata.get("config"), metadata.get("source"), metadata.get("provenance")
    if not isinstance(config, dict) or not isinstance(source, dict) or not isinstance(provenance, dict):
        raise ValueError("run metadata lacks the actual config, source or provenance")
    config_sha = canonical_sha256(config)
    if config_sha != profile.get("config_sha256") or source.get("sha256") != SOURCE_SHA256:
        raise ValueError("run source/config differs from the ROOT-admitted identity")
    expected_target = 0 if cell.startswith("t0_specialist_") else 1 if cell.startswith("t1_specialist_") else None
    expected_seed = 203201 if cell.endswith("203201") else 203202
    model_config = config.get("model")
    if (
        not isinstance(model_config, dict)
        or config.get("seed") != expected_seed
        or config.get("supervised_target") != expected_target
        or config.get("epochs") != 6
        or config.get("objective_mode") != "standalone_labels_wmse_v2"
        or config.get("required_device") != "L4"
        or config.get("required_world_size") != 1
        or config.get("normalization_scope") != "all_fit"
        or config.get("normalization_sequences") != FIT_SEQUENCES
        or config.get("training_mask") != "all_after_warmup"
        or config.get("checkpoint_endpoint_policy") != "configured_final_epoch"
        or config.get("checkpoint_retention") != "all_epochs"
        or model_config.get("architecture") != "gru"
        or model_config.get("hidden") != 256
        or model_config.get("layers") != 2
    ):
        raise ValueError("cell config differs from the frozen L4 recipe")
    data = provenance.get("data")
    if not isinstance(data, dict):
        raise ValueError("run data provenance is missing")
    data_sha = canonical_sha256(data)
    if provenance.get("data_sha256") != data_sha:
        raise ValueError("run data provenance digest is invalid")
    fit_members, evaluation_members = data.get("fit_members"), data.get("evaluation_members")
    if (
        data.get("split_sha256") != SPLIT_SHA256
        or data.get("teacher") is not None
        or not isinstance(fit_members, list)
        or len(fit_members) != FIT_SEQUENCES
        or any(not isinstance(member, list) or len(member) != 2 or member[0] != "train" for member in fit_members)
        or len({(member[0], member[1]) for member in fit_members if isinstance(member, list) and len(member) == 2}) != FIT_SEQUENCES
        or not isinstance(evaluation_members, list)
        or len(evaluation_members) != VALID_SEQUENCES
        or any(
            not isinstance(member, list)
            or len(member) != 2
            or member[0] != "valid"
            or type(member[1]) is not int
            for member in evaluation_members
        )
        or len(
            {
                (member[0], member[1])
                for member in evaluation_members
                if isinstance(member, list) and len(member) == 2
            }
        )
        != VALID_SEQUENCES
        or provenance.get("world_size") != 1
    ):
        raise ValueError("run membership, split, teacher or world size differs from the approved fit")
    normalization = data.get("normalization")
    objective = data.get("objective_contract")
    monitor = data.get("report_only_monitor")
    runtime = provenance.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, dict) else None
    expected_uuid = GPU_BINDINGS[cell][1]
    if (
        not isinstance(normalization, dict)
        or normalization.get("scope") != "all_fit"
        or normalization.get("sequence_count") != FIT_SEQUENCES
        or data.get("training_mask") != "all_after_warmup"
        or not isinstance(objective, dict)
        or objective.get("mode") != "standalone_labels_wmse_v2"
        or objective.get("reference_training_ancestry") != "none"
        or not isinstance(monitor, dict)
        or monitor.get("endpoint") != "raw standalone z; no reference or blend"
        or not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], dict)
        or devices[0].get("name") != "NVIDIA L4"
        or devices[0].get("uuid") != expected_uuid
    ):
        raise ValueError("fit-only ancestry, monitor contract or assigned L4 identity differs")
    data_sources = data.get("sources")
    if not isinstance(data_sources, dict) or any(
        not isinstance(data_sources.get(name), dict)
        for name in ("datasets/train.parquet", "datasets/valid.parquet")
    ):
        raise ValueError("fit data provenance lacks train/valid source fingerprints")
    valid_members_sha = hashlib.sha256(
        json.dumps(evaluation_members, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = terminal.get("identity")
    if not isinstance(identity, dict) or (
        terminal.get("status") != "completed"
        or type(terminal.get("completed_epochs")) is not int
        or terminal.get("completed_epochs") != 6
        or type(terminal.get("global_updates")) is not int
        or terminal.get("global_updates") != EXPECTED_UPDATES
        or terminal.get("model_sha256") != model_sha256
        or identity.get("source_sha256") != SOURCE_SHA256
        or identity.get("config_sha256") != config_sha
        or identity.get("data_sha256") != data_sha
        or identity.get("world_size") != 1
    ):
        raise ValueError("native terminal status does not match model/source/config/data identity")
    if len(events) != 6:
        raise ValueError("fixed-E6 run does not have exactly six full-validation events")
    for epoch, event in enumerate(events, start=1):
        metrics, context = event.get("metrics"), event.get("context")
        if (
            event.get("schema") != "wnn.monitor.v1"
            or event.get("kind") != "evaluation"
            or type(event.get("epoch")) is not int
            or event.get("epoch") != epoch
            or type(event.get("step")) is not int
            or event.get("step") != epoch * UPDATES_PER_EPOCH
            or not isinstance(metrics, dict)
            or not isinstance(context, dict)
            or context.get("split_sha256") != SPLIT_SHA256
            or type(context.get("evaluation_member_count")) is not int
            or context.get("evaluation_member_count") != VALID_SEQUENCES
            or context.get("evaluation_member_sha256") != valid_members_sha
            or context.get("endpoint") != "raw standalone z; no reference or blend"
            or type(metrics.get("valid/sequence_count")) is not int
            or metrics.get("valid/sequence_count") != VALID_SEQUENCES
            or type(metrics.get("valid/selected_rows")) is not int
            or metrics.get("valid/selected_rows") != VALID_ROWS
            or not all(_finite(metrics.get(key)) for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1"))
            or any(not _finite(value) for value in metrics.values())
        ):
            raise ValueError(f"full-valid epoch {epoch} has identity, finiteness or coverage mismatch")
    final_metrics = fit_result.get("final_metrics")
    if not isinstance(final_metrics, dict) or final_metrics != events[-1]["metrics"]:
        raise ValueError("ROOT fit result final metrics differ from the native E6 monitor event")
    if any(
        not math.isclose(
            float(final_metrics[key]), float(events[-1]["metrics"][key]), rel_tol=0.0, abs_tol=0.0
        )
        for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1")
    ):
        raise ValueError("E6 monitor metrics failed exact consistency")
    return {
        "cell": cell,
        "config_sha256": config_sha,
        "source_sha256": SOURCE_SHA256,
        "split_sha256": SPLIT_SHA256,
        "data_sha256": data_sha,
        "model_sha256": model_sha256,
        "fit_members": fit_members,
        "evaluation_members": evaluation_members,
        "fit_members_sha256": canonical_sha256(fit_members),
        "evaluation_members_sha256": canonical_sha256(evaluation_members),
        "data_source_fingerprints": {
            name: data_sources[name]
            for name in ("datasets/train.parquet", "datasets/valid.parquet")
        },
        "final_metrics": final_metrics,
    }


def validate_fit_membership(fit: JsonMap, split_manifest: JsonMap) -> None:
    """Require the exact official train-fit and complete official-valid roster."""
    fit_section = split_manifest.get("fit")
    evaluation_section = split_manifest.get("evaluation")
    if (
        split_manifest.get("schema") != "wnn.chronological-primary.v1"
        or split_manifest.get("policy") != "official_train_fit_full_official_valid_evaluation_v1"
        or split_manifest.get("chronology_basis")
        != "official train/valid source boundary; calendar timestamps unavailable"
        or not isinstance(fit_section, dict)
        or not isinstance(evaluation_section, dict)
        or not isinstance(fit_section.get("train"), list)
        or fit_section.get("valid") != []
        or not isinstance(evaluation_section.get("valid"), list)
        or set(fit_section) != {"train", "valid"}
        or set(evaluation_section) != {"valid"}
    ):
        raise ValueError("split manifest is not the official chronological primary policy")
    fit_groups, valid_groups = fit_section["train"], evaluation_section["valid"]
    if (
        any(type(group) is not int or group < 0 for group in fit_groups)
        or fit_groups != sorted(set(fit_groups))
        or any(type(group) is not int or group < 0 for group in valid_groups)
        or valid_groups != sorted(set(valid_groups))
    ):
        raise ValueError("official primary split group identities are malformed")
    expected_fit = [["train", group] for group in fit_groups]
    expected_valid = [["valid", group] for group in valid_groups]
    if (
        fit.get("fit_members") != expected_fit
        or fit.get("evaluation_members") != expected_valid
        or len(expected_fit) != FIT_SEQUENCES
        or len(expected_valid) != VALID_SEQUENCES
    ):
        raise ValueError("native run membership differs from the exact official primary split")


def validate_prediction_manifest(
    manifest: JsonMap, fit: JsonMap, manifest_sha256: str
) -> None:
    """Validate the actual sequence-prediction-v1 producer and complete held-out roster."""
    if (
        manifest.get("schema") != "sequence-predictions-v1"
        or manifest.get("complete") is not True
        or manifest.get("input_schema") != "raw112-v1"
        or manifest.get("membership") != "evaluation"
        or manifest.get("split_sha256") != SPLIT_SHA256
        or type(manifest.get("hint_dim")) is not int
        or manifest.get("hint_dim") != 0
        or type(manifest.get("sequence_length")) is not int
        or manifest.get("sequence_length") != SEQUENCE_LENGTH
        or type(manifest.get("world_size")) is not int
        or manifest.get("world_size") != 1
        or not _is_sha256(manifest_sha256)
    ):
        raise ValueError("prediction manifest schema, split, shape or completion flag is invalid")
    producer, sequences = manifest.get("producer"), manifest.get("sequences")
    if not isinstance(producer, dict) or not isinstance(sequences, list) or len(sequences) != VALID_SEQUENCES:
        raise ValueError("prediction producer or full-valid sequence list is malformed")
    producer_provenance = producer.get("provenance")
    identity = producer_provenance.get("identity") if isinstance(producer_provenance, dict) else None
    if (
        producer.get("checkpoint_sha256") != fit.get("model_sha256")
        or producer.get("split_sha256") != SPLIT_SHA256
        or producer.get("fit_members") != fit.get("fit_members")
        or producer.get("teacher") is not None
        or producer.get("config") is None
        or canonical_sha256(producer["config"]) != fit.get("config_sha256")
        or not isinstance(identity, dict)
        or identity.get("source_sha256") != SOURCE_SHA256
        or identity.get("config_sha256") != fit.get("config_sha256")
        or identity.get("data_sha256") != fit.get("data_sha256")
        or identity.get("world_size") != 1
        or not isinstance(producer_provenance, dict)
        or type(producer_provenance.get("completed_epochs")) is not int
        or producer_provenance.get("completed_epochs") != 6
        or type(producer_provenance.get("global_updates")) is not int
        or producer_provenance.get("global_updates") != EXPECTED_UPDATES
    ):
        raise ValueError("prediction producer differs from the ROOT-verified fixed endpoint")
    observed: list[tuple[str, int]] = []
    names: set[str] = set()
    for entry in sequences:
        if not isinstance(entry, dict):
            raise ValueError("prediction entry is malformed")
        split_name, group, filename, digest = (
            entry.get("split"), entry.get("row_group"), entry.get("file"), entry.get("sha256")
        )
        if (
            split_name != "valid"
            or type(group) is not int
            or not isinstance(filename, str)
            or filename != f"valid-{group}.npy"
            or not _is_sha256(digest)
        ):
            raise ValueError("prediction entry has an invalid member, filename or SHA-256")
        observed.append((split_name, group))
        names.add(filename)
    if (
        observed != [tuple(member) for member in fit["evaluation_members"]]
        or len(set(observed)) != VALID_SEQUENCES
        or len(names) != VALID_SEQUENCES
    ):
        raise ValueError("prediction manifest is not exact, unique full-valid coverage")


def validate_score_receipt(
    receipt: JsonMap, manifest: JsonMap, manifest_sha256: str, fit: JsonMap
) -> JsonMap:
    """Check the canonical scorer's actual JSON result and compare it with monitor E6."""
    score = receipt.get("score")
    if (
        receipt.get("split_sha256") != SPLIT_SHA256
        or receipt.get("predictions_manifest_sha256") != manifest_sha256
        or receipt.get("producer") != manifest.get("producer")
        or type(receipt.get("sequences")) is not int
        or receipt.get("sequences") != VALID_SEQUENCES
        or receipt.get("interpretation")
        != "Exposed fold; fixed endpoint, not checkpoint selection or public score."
        or not isinstance(score, dict)
        or score.get("metric_id") != "global_wp_clip2_v1"
        or type(score.get("selected_rows")) is not int
        or score.get("selected_rows") != VALID_ROWS
        or not all(_finite(score.get(key)) for key in ("t0", "t1", "wp"))
        or not math.isclose(float(score["wp"]), 0.5 * (float(score["t0"]) + float(score["t1"])), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("canonical score receipt is incomplete, mismatched or nonfinite")
    monitor = fit["final_metrics"]
    components = {
        "t0": (float(score["t0"]), float(monitor["valid/wp/t0"])),
        "t1": (float(score["t1"]), float(monitor["valid/wp/t1"])),
        "wp": (float(score["wp"]), float(monitor["valid/wp"])),
    }
    differences = {key: abs(exported - monitored) for key, (exported, monitored) in components.items()}
    within = all(value <= ABS_TOLERANCE for value in differences.values())
    return {
        "metric_id": score["metric_id"],
        "selected_rows": score["selected_rows"],
        "t0": float(score["t0"]),
        "t1": float(score["t1"]),
        "wp": float(score["wp"]),
        "monitor_t0": components["t0"][1],
        "monitor_t1": components["t1"][1],
        "monitor_wp": components["wp"][1],
        "abs_differences": differences,
        "tolerance": ABS_TOLERANCE,
        "within_tolerance": within,
    }


def derive_output_rows(scores: dict[str, JsonMap]) -> list[JsonMap]:
    """Compute the baseline plus three frozen output pairings for both seeds."""
    if set(scores) != set(CELLS) or any(
        scores[cell].get("within_tolerance") is not True for cell in CELLS
    ):
        raise ValueError("all six independent scores must pass source/coverage/monitor gates")
    recipes = (
        ("baseline shared-control/shared-control", "shared_control", "shared_control"),
        ("t0-specialist/shared-control-t1", "t0_specialist", "shared_control"),
        ("shared-control-t0/t1-specialist", "shared_control", "t1_specialist"),
        ("t0-specialist/t1-specialist", "t0_specialist", "t1_specialist"),
    )
    rows: list[JsonMap] = []
    for seed in (203201, 203202):
        for label, left_arm, right_arm in recipes:
            left = f"{left_arm}_s{seed}"
            right = f"{right_arm}_s{seed}"
            t0 = float(scores[left]["t0"])
            t1 = float(scores[right]["t1"])
            rows.append(
                {
                    "seed": seed,
                    "output": label,
                    "t0_source_cell": left,
                    "t1_source_cell": right,
                    "t0_wp": t0,
                    "t1_wp": t1,
                    "complete_output_wp": 0.5 * (t0 + t1),
                }
            )
    return rows


def directory_bytes(root: Path) -> int:
    """Count physical files under R without traversing dataset/cache symlinks."""
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
        for name in files:
            path = Path(folder) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    continue
    return total


def validate_project_file_stats(
    root: Path, fit: JsonMap, split_sources: JsonMap
) -> None:
    """Match fit and split source pins, then verify each bounded parquet footer."""
    fingerprints = fit.get("data_source_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("fit receipt does not bind source parquet fingerprints")
    for relative in ("datasets/train.parquet", "datasets/valid.parquet"):
        expected = fingerprints[relative]
        manifest_identity = split_sources.get(relative)
        path = root / "project" / relative
        if (
            not isinstance(expected, dict)
            or not isinstance(manifest_identity, dict)
            or expected != manifest_identity
            or not path.is_file()
        ):
            raise ValueError(f"prepared source file is missing: {relative}")
        observed = path.stat()
        if type(expected.get("bytes")) is not int or observed.st_size != expected["bytes"]:
            raise ValueError(f"prepared source fingerprint differs from fit provenance: {relative}")
        if type(expected.get("mtime_ns")) is not int or observed.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"prepared source fingerprint differs from fit provenance: {relative}")
        if not _is_sha256(expected.get("footer_sha256")) or type(
            expected.get("footer_bytes")
        ) is not int:
            raise ValueError(f"prepared source footer identity is malformed: {relative}")
        with path.open("rb") as stream:
            if observed.st_size < 8:
                raise ValueError(f"prepared source parquet is truncated: {relative}")
            stream.seek(-8, os.SEEK_END)
            trailer = stream.read(8)
            footer_bytes = int.from_bytes(trailer[:4], "little")
            if trailer[4:] != b"PAR1" or footer_bytes > observed.st_size - 8:
                raise ValueError(f"prepared source parquet footer is invalid: {relative}")
            stream.seek(-8 - footer_bytes, os.SEEK_END)
            digest = hashlib.sha256()
            remaining = footer_bytes + 8
            while remaining:
                block = stream.read(min(remaining, 1 << 20))
                if not block:
                    raise ValueError(f"prepared source parquet footer is truncated: {relative}")
                digest.update(block)
                remaining -= len(block)
        if (
            footer_bytes != expected["footer_bytes"]
            or digest.hexdigest() != expected["footer_sha256"]
        ):
            raise ValueError(f"prepared source footer differs from fit provenance: {relative}")


class ReadoutGuard:
    """Rate-limit the shared disk/reserve check used by the six owned children."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.last_check = 0.0
        self.error: str | None = None

    def check(self) -> None:
        """Fail all and only this controller's child groups on the output/disk cap."""
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if time.monotonic() - self.last_check < 10:
                return
            self.last_check = time.monotonic()
            total = directory_bytes(self.root)
            free = shutil.disk_usage(self.root).free
            if total > MAX_OUTPUT_BYTES or free < MIN_FREE_BYTES:
                self.error = f"readout disk guard: output_bytes={total}; free_bytes={free}"
                raise RuntimeError(self.error)


def _environment(source: Path, gpu_uuid: str | None) -> dict[str, str]:
    """Give prediction only its admitted GPU and the new frozen source namespace."""
    env = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUBLAS_EMULATE_SINGLE_PRECISION",
        "PYTHONPATH",
    ):
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(source / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
        CUDA_VISIBLE_DEVICES=f"GPU-{gpu_uuid}" if gpu_uuid else "",
    )
    return env


def _command(
    command: list[str],
    *,
    source: Path,
    control: Path,
    output_dir: Path,
    cell: str,
    gpu: int,
    owner: JsonMap,
    stage: str,
    timeout: int,
    guard: ReadoutGuard,
    gpu_uuid: str | None,
) -> JsonMap:
    """Use the existing process-group/PID-birth executor for exactly one stage."""
    from wnn.multistage_search_owned_command import run_owned_command

    if timeout <= 0:
        raise TimeoutError("the ROOT wave deadline leaves no stage time")
    receipt_path = control / "owned" / f"{cell}-{stage}.json"
    log_path = control / "logs" / f"{cell}-{stage}.log"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"owned {stage} receipt already exists; no retry is allowed")
    code, timed_out, receipt = run_owned_command(
        command,
        cwd=source,
        env=_environment(source, gpu_uuid),
        log_path=log_path,
        receipt_path=receipt_path,
        owner=owner,
        stage=2,
        trial_number=gpu,
        mode=stage,
        gpu_key=str(gpu) if gpu_uuid is not None else "cpu",
        output_dir=output_dir,
        timeout_seconds=timeout,
        progress=guard.check,
    )
    if code != 0 or timed_out:
        raise RuntimeError(f"owned {stage} exited {code}; timed_out={timed_out}; no retry")
    return {"path": str(receipt_path), "log": str(log_path), "receipt": receipt}


def _write_atomic(writer: Any, path: Path, value: JsonMap, *, replace: bool = False) -> None:
    writer(path, value, replace=replace)


def _load_and_validate_fit(root: Path, cell: str, admission: JsonMap) -> tuple[JsonMap, JsonMap, JsonMap, list[JsonMap]]:
    """Load ROOT result, run metadata, terminal status and six authoritative events."""
    fit_result = _read_json(root / "control" / "root-fits-r1" / f"{cell}-result.json", 128_000)
    fit_dir = root / "runs" / cell / "fit"
    metadata = _read_json(fit_dir / "run_metadata.json", 1_000_000)
    terminal = _read_json(fit_dir / "terminal_status.json", 2_000_000)
    model_path = fit_dir / "model.pt"
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("fixed endpoint model.pt is missing or linked")
    model_sha = sha256_file(model_path)
    events = _read_events(fit_dir / "monitor_evaluations.jsonl")
    fit = validate_fit_evidence(
        cell,
        admission["profiles"][cell],
        fit_result,
        metadata,
        terminal,
        model_sha,
        events,
    )
    return fit, fit_result, metadata, events


def _validate_readout_files(readout_dir: Path, fit: JsonMap) -> tuple[JsonMap, JsonMap, str]:
    """Validate the exported manifest and canonical score JSON after both child exits."""
    predictions = readout_dir / "predictions"
    manifest_path = predictions / "manifest.json"
    manifest = _read_json(manifest_path, 32_000_000)
    manifest_sha = sha256_file(manifest_path)
    validate_prediction_manifest(manifest, fit, manifest_sha)
    score_path = readout_dir / "score.json"
    score_receipt = _read_json(score_path, 2_000_000)
    comparison = validate_score_receipt(score_receipt, manifest, manifest_sha, fit)
    return manifest, score_receipt, comparison


def _wait_for_fit(
    root: Path, cell: str, fit_deadline: float, poll_seconds: int = 15
) -> JsonMap | None:
    """Wait only for this cell's ROOT fit result and stop before the reserved window."""
    path = root / "control" / "root-fits-r1" / f"{cell}-result.json"
    while time.time() < fit_deadline:
        if path.exists():
            return _read_json(path, 128_000)
        time.sleep(min(poll_seconds, max(0.0, fit_deadline - time.time())))
    return _read_json(path, 128_000) if path.exists() else None


def _run_cell(
    root: Path,
    source: Path,
    python: Path,
    control: Path,
    cell: str,
    admission: JsonMap,
    split_manifest: JsonMap,
    fit_deadline: float,
    wave_deadline: float,
    controller_owner: JsonMap,
    guard: ReadoutGuard,
    atomic_write: Any,
    hold_gpu_slot: Any,
) -> JsonMap:
    """Read one successful fit and run one fresh prediction-export/CPU-score pair."""
    gpu, gpu_uuid = GPU_BINDINGS[cell]
    result_path = control / f"{cell}-result.json"
    try:
        if result_path.exists() or result_path.is_symlink():
            raise FileExistsError("readout result already exists; completed work will not be repeated")
        fit_result = _wait_for_fit(root, cell, fit_deadline)
        if fit_result is None:
            summary = {"schema": RESULT_SCHEMA, "cell": cell, "status": "fit_deadline_no_result"}
            _write_atomic(atomic_write, result_path, summary)
            return summary
        if fit_result.get("status") != "fit_complete":
            summary = {
                "schema": RESULT_SCHEMA,
                "cell": cell,
                "status": "fit_not_complete",
                "root_fit_status": fit_result.get("status", "unknown"),
                "fit_result_sha256": sha256_file(root / "control" / "root-fits-r1" / f"{cell}-result.json"),
            }
            _write_atomic(atomic_write, result_path, summary)
            return summary
        fit, fit_result, _, _ = _load_and_validate_fit(root, cell, admission)
        validate_fit_membership(fit, split_manifest)
        split_sources = split_manifest.get("sources")
        if not isinstance(split_sources, dict):
            raise ValueError("official primary split lacks source fingerprints")
        validate_project_file_stats(root, fit, split_sources)
        if wave_deadline - time.time() < EXPORT_TIMEOUT_SECONDS + SCORE_TIMEOUT_SECONDS:
            raise TimeoutError("less than 1,200 seconds remain before the fixed wave deadline")
        output = root / "runs" / cell / "readout-final"
        if output.exists() or output.is_symlink():
            raise FileExistsError("readout-final already exists; no export or score retry is allowed")
        claim_path = control / f"claim-{cell}.json"
        if claim_path.exists() or claim_path.is_symlink():
            raise FileExistsError("readout claim already exists; no retry is allowed")
        claim = {
            "schema": "wnn.l4-next70-readout-claim.v1",
            "cell": cell,
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "model_sha256": fit["model_sha256"],
            "data_source_fingerprints": fit["data_source_fingerprints"],
            "source_sha256": SOURCE_SHA256,
            "config_sha256": fit["config_sha256"],
            "controller_owner": controller_owner,
            "created_at_epoch": time.time(),
        }
        _write_atomic(atomic_write, claim_path, claim)
        output.mkdir(parents=True, exist_ok=False)
        prediction_dir = output / "predictions"
        predict_command = [
            str(python), str(source / "scripts/predict_sequence.py"),
            "--checkpoint", str(root / "runs" / cell / "fit" / "model.pt"),
            "--data-root", str(root / "project"),
            "--split-manifest", str(root / "project/chronological_primary.json"),
            "--output", str(prediction_dir),
            "--membership", "evaluation",
        ]
        owner = {
            "controller": controller_owner,
            "cell_id": cell,
            "readout_schema": RESULT_SCHEMA,
            "source_sha256": SOURCE_SHA256,
            "config_sha256": fit["config_sha256"],
            "model_sha256": fit["model_sha256"],
        }
        with hold_gpu_slot(control, str(gpu)):
            export_owned = _command(
                predict_command,
                source=source,
                control=control,
                output_dir=prediction_dir,
                cell=cell,
                gpu=gpu,
                owner=owner,
                stage="readout_export",
                timeout=min(EXPORT_TIMEOUT_SECONDS, int(wave_deadline - time.time())),
                guard=guard,
                gpu_uuid=gpu_uuid,
            )
        score_path = output / "score.json"
        score_command = [
            str(python), str(source / "scripts/score_sequence.py"),
            "--data-root", str(root / "project"),
            "--split-manifest", str(root / "project/chronological_primary.json"),
            "--predictions", str(prediction_dir),
            "--output", str(score_path),
        ]
        score_owned = _command(
            score_command,
            source=source,
            control=control,
            output_dir=output,
            cell=cell,
            gpu=gpu,
            owner=owner,
            stage="readout_score_cpu",
            timeout=min(SCORE_TIMEOUT_SECONDS, int(wave_deadline - time.time())),
            guard=guard,
            gpu_uuid=None,
        )
        model_path = root / "runs" / cell / "fit" / "model.pt"
        if sha256_file(model_path) != fit["model_sha256"]:
            raise ValueError("fixed endpoint model bytes changed during readout")
        manifest, score_receipt, comparison = _validate_readout_files(output, fit)
        status = "readout_complete" if comparison["within_tolerance"] else "score_monitor_mismatch"
        receipt: JsonMap = {
            "schema": RESULT_SCHEMA,
            "cell": cell,
            "status": status,
            "failure_kind": (
                "export_or_metric_pipeline_mismatch"
                if status == "score_monitor_mismatch"
                else None
            ),
            "gpu": gpu,
            "gpu_uuid": gpu_uuid,
            "source_sha256": SOURCE_SHA256,
            "config_sha256": fit["config_sha256"],
            "split_sha256": SPLIT_SHA256,
            "data_sha256": fit["data_sha256"],
            "model_sha256": fit["model_sha256"],
            "fit_result_sha256": sha256_file(root / "control" / "root-fits-r1" / f"{cell}-result.json"),
            "run_metadata_sha256": sha256_file(root / "runs" / cell / "fit" / "run_metadata.json"),
            "terminal_status_sha256": sha256_file(root / "runs" / cell / "fit" / "terminal_status.json"),
            "monitor_evaluations_sha256": sha256_file(root / "runs" / cell / "fit" / "monitor_evaluations.jsonl"),
            "prediction_manifest_sha256": sha256_file(output / "predictions" / "manifest.json"),
            "score_receipt_sha256": sha256_file(score_path),
            "prediction_count": len(manifest["sequences"]),
            "prediction_shape": [SEQUENCE_LENGTH, 2],
            "prediction_dtype": "float32",
            "array_validation": (
                "canonical score_sequence.py verified every listed file SHA, finite float32 shape, "
                "and exact official-valid membership before scoring"
            ),
            "score": comparison,
            "owned_export": export_owned,
            "owned_score": score_owned,
            "monitor_final_metrics": fit["final_metrics"],
            "interpretation": (
                "Same repeatedly exposed official validation partition and units as training-time "
                "monitoring; re-export/re-score agreement checks the prediction pipeline, not a "
                "new independent validation or fresh confirmation."
            ),
        }
        _write_atomic(atomic_write, output / "readout-receipt.json", receipt)
        _write_atomic(atomic_write, result_path, receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema": RESULT_SCHEMA,
            "cell": cell,
            "status": "readout_failed",
            "failure_kind": "infrastructure_or_provenance",
            "error": f"{type(error).__name__}: {error}",
            "automatic_retry": False,
        }
        if not result_path.exists():
            _write_atomic(atomic_write, result_path, failure)
        return failure


def run(root: Path, source: Path, python: Path, old_source: Path) -> JsonMap:
    """Verify source/admission, then run six waiting workers with disjoint GPUs."""
    root = root.resolve()
    source = source.resolve(strict=True)
    python = python.resolve(strict=True)
    old_source = old_source.resolve(strict=True)
    control = root / "control" / "readout-r1"
    control.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        raise RuntimeError("readout must run on the ROOT-owned POSIX receiver")
    admission_path = root / "control" / "ROOT-fit-admission-r1.json"
    admission = _read_json(admission_path, 128_000)
    validate_admission(admission)
    clock = _read_json(root / "receipts" / "first-profile-r1.json", 16_000)
    first, fit_deadline, wave_deadline = (
        clock.get("first_profile_epoch"),
        clock.get("fit_deadline_epoch"),
        clock.get("wave_deadline_epoch"),
    )
    if (
        not all(_finite(value) for value in (first, fit_deadline, wave_deadline))
        or not math.isclose(float(fit_deadline), float(first) + 39_600, abs_tol=1.0)
        or not math.isclose(float(wave_deadline), float(first) + 43_200, abs_tol=1.0)
        or time.time() >= float(wave_deadline)
    ):
        raise TimeoutError("the original finite readout/collection window is closed or malformed")
    split_path = root / "project" / "chronological_primary.json"
    if sha256_file(split_path) != SPLIT_SHA256:
        raise ValueError("prepared chronological primary split bytes differ from admission")
    split_manifest = _read_json(split_path, 8_000_000)
    verifier = runpy.run_path(str(source / "src/wnn/training/sequence_source_capsule.py"))[
        "verify_source_capsule"
    ]
    verifier(source, source / "source-manifest.json", SOURCE_MANIFEST_SHA256, SOURCE_COMMIT)
    if directory_bytes(root) + len(CELLS) * (PREDICTION_BYTES_PER_CELL + 10_000_000) > MAX_OUTPUT_BYTES:
        raise OSError("six prediction exports would exceed the 6 GB aggregate output cap")
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise OSError(f"readout requires 4 GB free-space reserve; found {free}")

    sys.path.insert(0, str(old_source))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_runtime import current_process_owner, hold_gpu_slot

    summary_path = control / "complete.json"
    with file_lock(control / "readout.lock", blocking=False):
        if summary_path.exists() or (control / "owner.json").exists():
            raise FileExistsError("readout controller receipt exists; no rerun is allowed")
        controller_owner = current_process_owner()
        _write_atomic(
            atomic_write_json,
            control / "owner.json",
            {
                "schema": "wnn.l4-next70-readout-owner.v1",
                "owner": controller_owner,
                "source_sha256": SOURCE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "admission_sha256": sha256_file(admission_path),
                "first_profile_epoch": first,
                "fit_deadline_epoch": fit_deadline,
                "wave_deadline_epoch": wave_deadline,
            },
        )
        guard = ReadoutGuard(root)
        outcomes: dict[str, JsonMap] = {}
        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="l4-readout") as pool:
            futures = {
                pool.submit(
                    _run_cell,
                    root,
                    source,
                    python,
                    control,
                    cell,
                    admission,
                    split_manifest,
                    float(fit_deadline),
                    float(wave_deadline),
                    controller_owner,
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
                        "schema": RESULT_SCHEMA,
                        "cell": cell,
                        "status": "readout_failed",
                        "failure_kind": "controller_infrastructure",
                        "error": f"{type(error).__name__}: {error}",
                        "automatic_retry": False,
                    }
        successful = {
            cell: outcome["score"]
            for cell, outcome in outcomes.items()
            if outcome.get("status") == "readout_complete"
        }
        all_complete = len(successful) == 6
        rows = derive_output_rows(successful) if all_complete else None
        receipt: JsonMap = {
            "schema": SUMMARY_SCHEMA,
            "status": "complete" if all_complete else "partial_or_failed",
            "source_sha256": SOURCE_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "split_sha256": SPLIT_SHA256,
            "admission_sha256": sha256_file(admission_path),
            "first_profile_epoch": first,
            "fit_deadline_epoch": fit_deadline,
            "wave_deadline_epoch": wave_deadline,
            "cells": outcomes,
            "complete_outputs_per_seed": rows,
            "complete_output_formula": "0.5 * (left model official valid WP t0 + right model official valid WP t1)",
            "interpretation": (
                "The fixed official valid partition was repeatedly exposed during training. "
                "This same-partition re-export/re-score checks producer, bytes, coverage and scorer "
                "agreement; it is not independent validation or fresh confirmation. No learned "
                "valid weights are used."
            ),
        }
        _write_atomic(atomic_write_json, summary_path, receipt)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="prepared L4 experiment root R")
    parser.add_argument("--source", type=Path, default=RECEIVER_SOURCE)
    parser.add_argument("--python", type=Path, default=RECEIVER_PYTHON)
    parser.add_argument("--old-src", type=Path, default=OLD_SOURCE)
    args = parser.parse_args()
    result = run(args.root, args.source, args.python, args.old_src)
    print(json.dumps({"status": result["status"], "complete_outputs_per_seed": result["complete_outputs_per_seed"]}, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
