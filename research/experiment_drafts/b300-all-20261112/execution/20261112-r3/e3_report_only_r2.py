"""Export and score the predeclared E3 checkpoint as report-only evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

LOG = logging.getLogger("b300_e3_report_only")
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CLOCK_SHA256 = "fc1d4b8b3d554a0d9b131152288f7af2d3c355d22c29b0279592d3e62f716398"
WAVE_DEADLINE_EPOCH = 1_794_471_619.060462
FIT_SEQUENCES = 10_325
VALID_SEQUENCES = 1_873
VALID_SELECTED_ROWS = 3_528_273
MAX_JSON_BYTES = 32_000_000
MAX_BATCH_SEQUENCES = 64
DEFAULT_BATCH_SEQUENCES = 32
REPORT_SCHEMA = "wnn.impact81-e3-report-only.r2"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReadoutError(RuntimeError):
    """Invalid E3 lineage or incomplete report-only export."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical(value))


def _read_object(
    path: Path, maximum: int = MAX_JSON_BYTES
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ReadoutError(f"missing, unsafe, or oversized evidence: {path.name}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ReadoutError(f"invalid strict JSON evidence: {path.name}") from None
    if not isinstance(value, dict):
        raise ReadoutError(f"expected JSON object: {path.name}")
    return value, raw


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON numeric constant: {value}")


def _within(root: Path, target: Path) -> Path:
    if root.is_symlink():
        raise ReadoutError("fit directory cannot be a symlink")
    resolved_root = root.resolve(strict=True)
    try:
        target.relative_to(root)
        resolved = target.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        raise ReadoutError("checkpoint escapes the fit directory") from None
    cursor = root
    for part in target.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReadoutError("fit checkpoint path contains a symlink")
    return resolved


def validate_e3_index_binding(
    index: dict[str, Any],
    fit_metadata: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Select only the predeclared third-epoch checkpoint and bind fit identities."""
    entries = index.get("entries")
    if (
        index.get("schema") != "wnn.sequence-epoch-index.v1"
        or not isinstance(entries, list)
        or len(entries) < 3
        or type(index.get("completed_epochs")) is not int
        or index["completed_epochs"] != len(entries)
    ):
        raise ReadoutError("epoch index lacks a valid checkpoint prefix through E3")
    body = {key: value for key, value in index.items() if key != "index_sha256"}
    index_sha = index.get("index_sha256")
    if not isinstance(index_sha, str) or _canonical_sha256(body) != index_sha:
        raise ReadoutError("epoch index digest is invalid")
    entry = entries[2]
    if (
        not isinstance(entry, dict)
        or entry.get("epoch") != 2
        or entry.get("completed_epochs") != 3
        or type(entry.get("global_step")) is not int
        or entry["global_step"] <= 0
        or not isinstance(entry.get("checkpoint_sha256"), str)
        or SHA256.fullmatch(entry["checkpoint_sha256"]) is None
        or entry.get("checkpoint_path")
        != f"epoch_checkpoints/epoch-000003-{entry['checkpoint_sha256']}.pt"
    ):
        raise ReadoutError("predeclared E3 index entry is malformed")
    identity = index.get("identity")
    config = fit_metadata.get("config")
    source = fit_metadata.get("source")
    provenance = fit_metadata.get("provenance")
    if (
        not isinstance(identity, dict)
        or not isinstance(config, dict)
        or not isinstance(source, dict)
        or not isinstance(provenance, dict)
    ):
        raise ReadoutError("E3 or run metadata lacks source/config/data identity")
    data = provenance.get("data")
    source_files = source.get("files")
    source_sha = source.get("sha256")
    data_sha = provenance.get("data_sha256")
    config_sha = _canonical_sha256(config)
    impact = config.get("impact81")
    model_config = config.get("model")
    if (
        not isinstance(data, dict)
        or data.get("split_sha256") != SPLIT_SHA256
        or not isinstance(source_files, dict)
        or not isinstance(source_sha, str)
        or SHA256.fullmatch(source_sha) is None
        or _canonical_sha256(source_files) != source_sha
        or not isinstance(data_sha, str)
        or SHA256.fullmatch(data_sha) is None
        or _canonical_sha256(data) != data_sha
        or identity.get("source_sha256") != source_sha
        or identity.get("source_files") != source_files
        or identity.get("config_sha256") != config_sha
        or identity.get("data_sha256") != data_sha
        or identity.get("data") != data
    ):
        raise ReadoutError("E3 source/config/data identity differs from run metadata")
    if (
        not isinstance(impact, dict)
        or not isinstance(model_config, dict)
        or config.get("seed") != cell.get("seed")
        or config.get("epochs") != 4
        or model_config.get("architecture") != "transformer"
        or impact.get("cell") != cell
        or identity.get("data", {}).get("split_sha256") != SPLIT_SHA256
        or len(identity.get("data", {}).get("fit_members", [])) != FIT_SEQUENCES
    ):
        raise ReadoutError(
            "E3 config or fit membership differs from the declared B300 cell"
        )
    return {"entry": entry, "index_sha256": index_sha, "identity": identity}


def predict_batch(
    model: Any,
    x: Any,
    chunk_rows: int,
    device: Any,
    precision_mode: str,
    forward_context: Callable[[str, str], Any],
) -> Any:
    """Predict independent full sequences in one batch, carrying only per-sequence state."""
    import torch

    if x.ndim != 3 or not 1 <= chunk_rows <= 20_000:
        raise ValueError(
            "batch input must be [sequences, rows, features] with a valid chunk size"
        )
    state = None
    outputs = []
    for start in range(0, x.shape[1], chunk_rows):
        with forward_context(precision_mode, device.type):
            prediction, hints, state = model(
                x[:, start : start + chunk_rows].to(device), state
            )
        outputs.append(torch.cat((prediction.float(), hints.float()), dim=-1).cpu())
    values = torch.cat(outputs, dim=1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite report-only E3 prediction")
    return values


def _validate_clock(path: Path) -> None:
    clock, raw = _read_object(path, 1_000_000)
    if (
        _sha256_bytes(raw) != CLOCK_SHA256
        or clock.get("schema") != "wnn.cluster-wave-clock.r2"
        or clock.get("wave_deadline_epoch") != WAVE_DEADLINE_EPOCH
    ):
        raise ReadoutError("clock differs from the fixed B300 R2 wave authority")


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        )
        stream.write(b"\n")


def _load_model_and_evidence(
    args: argparse.Namespace, context: Any
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from impact81.bindings import SPLIT_SHA256, load_cell, source_identity
    from impact81.endpoint import _validate_preprocessing
    from impact81.modeling import build_model
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_epoch_prediction import (
        epoch_admission,
        validate_epoch_payload,
    )

    packet = args.packet_root.resolve(strict=True)
    if args.cell.resolve(strict=True).parent != packet / "configs":
        raise ReadoutError(
            "cell descriptor is outside the pinned Impact81 packet configs"
        )
    cell, _native_config = load_cell(args.cell)
    if cell["card"] not in {"01", "02", "03", "04", "05", "06", "07"}:
        raise ReadoutError(
            "E3 report-only export is restricted to the Transformer family"
        )
    if time.time() >= WAVE_DEADLINE_EPOCH - 180:
        raise ReadoutError("fixed wave cleanup boundary reached before E3 export")
    fit_dir = args.fit_dir.resolve(strict=True)
    if (fit_dir / "terminal_status.json").exists():
        raise ReadoutError(
            "fit has a terminal status; E3 partial report is not admissible"
        )
    index_path = fit_dir / "epoch_index.json"
    index, _index_raw = _read_object(index_path, 16_000_000)
    metadata, _metadata_raw = _read_object(fit_dir / "run_metadata.json")
    binding = validate_e3_index_binding(index, metadata, cell)
    entry = binding["entry"]
    checkpoint = fit_dir / entry["checkpoint_path"]
    _within(fit_dir, checkpoint)

    split = load_campaign_split(args.split_manifest, args.data_root)
    if (
        split.sha256 != SPLIT_SHA256
        or SPLIT_SHA256 != args.expected_split_sha256
        or split.schema != "wnn.chronological-primary.v1"
    ):
        raise ReadoutError("E3 readout requires the frozen chronological primary split")
    fit_members = [["train", group] for group in split.train_groups] + [
        ["valid", group] for group in split.valid_groups
    ]
    if len(fit_members) != FIT_SEQUENCES or split.valid_groups:
        raise ReadoutError(
            "E3 readout fit ancestry differs from all official training members"
        )

    runtime_root = args.runtime_root.resolve(strict=True)
    if (packet / "runtime").resolve(strict=True) not in sys_path_paths():
        raise ReadoutError("pinned Impact81 packet runtime was not placed on sys.path")
    current_source = source_identity(runtime_root)
    identity = binding["identity"]
    if (
        identity.get("source_sha256") != current_source["sha256"]
        or identity.get("source_files") != current_source["files"]
    ):
        raise ReadoutError(
            "E3 checkpoint differs from the exact active R5 runtime/packet sources"
        )

    admission = epoch_admission(checkpoint, index_path, binding["index_sha256"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ReadoutError("E3 checkpoint payload is not a mapping")
    validate_epoch_payload(payload, admission)
    if (
        payload.get("completed_epochs") != 3
        or payload.get("epoch") != 2
        or payload.get("data", {}).get("fit_members") != fit_members
    ):
        raise ReadoutError(
            "checkpoint is not the predeclared E3 of the full official fit"
        )
    config = payload.get("config")
    provenance = payload.get("provenance")
    data = payload.get("data")
    if not all(isinstance(value, dict) for value in (config, provenance, data)):
        raise ReadoutError("E3 checkpoint is missing configuration or ancestry")
    impact = config.get("impact81")
    if not isinstance(impact, dict):
        raise ReadoutError("E3 checkpoint lacks Impact81 preprocessing identity")
    _validate_preprocessing(
        impact,
        card=cell["card"],
        split_sha256=split.sha256,
        fit_members=fit_members,
        data=data,
        provenance=provenance,
    )
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ReadoutError("E3 checkpoint lacks model weights")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or (
            tensor.is_floating_point()
            and (
                tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
            )
        ):
            raise ReadoutError(f"invalid strict-FP32 checkpoint tensor: {name}")
    model = build_model(config, context.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    variant = cell["variant"]
    if variant == "deep_supervision":
        from impact81.depth_models import drop_auxiliary_heads

        drop_auxiliary_heads(model)
    elif variant == "future_heads":
        from impact81.future_model import drop_future_heads

        drop_future_heads(model)
    return (
        model,
        payload,
        admission,
        {"cell": cell, "fit_dir": fit_dir, "checkpoint": checkpoint},
    )


def sys_path_paths() -> list[Path]:
    import sys

    return [Path(value).resolve() for value in sys.path if value]


def _export_and_score(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from wnn.data import SEQUENCE_LENGTH, SequenceFile
    from wnn.evaluation import paired_component_moments
    from wnn.gmetric import GlobalAccumulator, _merge_moments
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_cache import validate_export_schema
    from wnn.training.sequence_distributed import (
        barrier,
        collective_output_setup,
        coordinated_call,
        initialize_distributed,
        shutdown_distributed,
    )
    from wnn.training.sequence_epoch_prediction import validate_epoch_producer
    from wnn.training.sequence_precision import forward_context
    from wnn.training.sequence_predict import _producer

    if time.time() >= WAVE_DEADLINE_EPOCH - 180:
        raise ReadoutError("fixed wave cleanup boundary reached before E3 export")
    context = initialize_distributed()
    try:
        model, payload, admission, fit = coordinated_call(
            lambda: _load_model_and_evidence(args, context),
            "E3 source/checkpoint admission",
            context,
        )
        split = load_campaign_split(args.split_manifest, args.data_root)
        members = [("valid", group) for group in split.evaluation_groups]
        if len(members) != VALID_SEQUENCES or len(set(members)) != VALID_SEQUENCES:
            raise ReadoutError(
                "E3 export requires complete official validation membership"
            )
        source_file = coordinated_call(
            lambda: SequenceFile(args.data_root / "datasets/valid.parquet"),
            "official-valid feature-only input construction",
            context,
        )
        collective_output_setup(args.output, False, context)
        barrier(context)

        def export_rank() -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            batch_size = args.batch_sequences
            batch_count = (len(members) + batch_size - 1) // batch_size
            with torch.no_grad():
                for batch_number in range(
                    context.rank, batch_count, context.world_size
                ):
                    if time.time() >= WAVE_DEADLINE_EPOCH - 180:
                        raise ReadoutError(
                            "fixed wave cleanup boundary reached during E3 export"
                        )
                    start = batch_number * batch_size
                    indices = list(range(start, min(start + batch_size, len(members))))
                    batch_members = [members[index] for index in indices]
                    x = torch.from_numpy(
                        np.stack(
                            [
                                source_file.read(
                                    members[index][1], targets=False, mask=False
                                ).features
                                for index in indices
                            ],
                            axis=0,
                        )
                    )
                    predictions = predict_batch(
                        model,
                        x,
                        int(payload["config"]["chunk_rows"]),
                        context.device,
                        str(payload["config"].get("precision_mode", "fp32")),
                        forward_context,
                    ).numpy()
                    if predictions.shape != (len(indices), SEQUENCE_LENGTH, 2):
                        raise ReadoutError(
                            "batched E3 output shape differs from sequence schema"
                        )
                    for offset, (source, group) in enumerate(batch_members):
                        values = np.asarray(predictions[offset], dtype=np.float32)
                        if not np.isfinite(values).all():
                            raise ReadoutError(
                                f"nonfinite E3 output for {source}/{group}"
                            )
                        name = f"{source}-{group}.npy"
                        with (args.output / name).open("xb") as stream:
                            np.save(stream, values, allow_pickle=False)
                        entries.append(
                            {
                                "split": source,
                                "row_group": group,
                                "file": name,
                                "sha256": _file_sha256(args.output / name),
                            }
                        )
            part = args.output / f"rank-{context.rank}.json"
            _write_new_json(part, {"entries": entries})
            return entries

        coordinated_call(export_rank, "batched E3 validation prediction", context)
        barrier(context)

        def publish_manifest() -> bool:
            if not context.is_primary:
                return True
            all_entries = [
                entry
                for rank in range(context.world_size)
                for entry in _read_object(args.output / f"rank-{rank}.json")[0][
                    "entries"
                ]
            ]
            entry_map = {
                (item["split"], item["row_group"]): item for item in all_entries
            }
            if len(all_entries) != VALID_SEQUENCES or len(entry_map) != VALID_SEQUENCES:
                raise ReadoutError(
                    "E3 output has missing or duplicated validation sequences"
                )
            if set(entry_map) != set(members):
                raise ReadoutError(
                    "E3 output membership differs from official validation"
                )
            producer = _producer(fit["checkpoint"], payload)
            producer["kind"] = "retained_epoch"
            producer["epoch_admission"] = admission
            producer["primary_valid_report_only"] = True
            manifest = {
                "schema": "sequence-predictions-v1",
                "complete": True,
                "input_schema": "raw112-v1",
                "membership": "evaluation",
                "split_sha256": split.sha256,
                "hint_dim": 0,
                "sequence_length": SEQUENCE_LENGTH,
                "producer": producer,
                "world_size": context.world_size,
                "sequences": [entry_map[member] for member in members],
            }
            _write_new_json(args.output / "manifest.json", manifest)
            return True

        coordinated_call(
            publish_manifest, "E3 report-only manifest publication", context
        )
        barrier(context)

        def score_report_only() -> bool:
            if not context.is_primary:
                return True
            split = load_campaign_split(args.split_manifest, args.data_root)
            manifest_path = args.output / "manifest.json"
            manifest, _ = _read_object(manifest_path)
            if (
                manifest.get("schema") != "sequence-predictions-v1"
                or manifest.get("complete") is not True
            ):
                raise ReadoutError(
                    "report-only scoring requires a complete prediction manifest"
                )
            if manifest.get("split_sha256") != SPLIT_SHA256:
                raise ReadoutError(
                    "prediction manifest split differs from primary validation"
                )
            validate_export_schema(manifest, "evaluation")
            fit_members = [("train", group) for group in split.train_groups] + [
                ("valid", group) for group in split.valid_groups
            ]
            validate_epoch_producer(manifest["producer"], split.sha256, fit_members)
            expected_members = [("valid", group) for group in split.evaluation_groups]
            entries = manifest.get("sequences")
            if not isinstance(entries, list):
                raise ReadoutError("prediction manifest sequence entries are malformed")
            entry_map: dict[tuple[str, int], dict[str, Any]] = {}
            for item in entries:
                if not isinstance(item, dict):
                    raise ReadoutError(
                        "prediction manifest contains a malformed sequence"
                    )
                key = (item.get("split"), item.get("row_group"))
                if (
                    key in entry_map
                    or key[0] != "valid"
                    or type(key[1]) is not int
                    or not isinstance(item.get("file"), str)
                    or Path(item["file"]).name != item["file"]
                    or not item["file"].endswith(".npy")
                    or not isinstance(item.get("sha256"), str)
                    or SHA256.fullmatch(item["sha256"]) is None
                ):
                    raise ReadoutError(
                        "prediction manifest contains invalid or duplicate membership"
                    )
                entry_map[key] = item
            if len(entry_map) != VALID_SEQUENCES or set(entry_map) != set(
                expected_members
            ):
                raise ReadoutError(
                    "prediction manifest does not cover official validation exactly"
                )

            from wnn.data import WARMUP, SequenceFile
            from wnn.splits import file_sha256

            labels = SequenceFile(args.data_root / "datasets/valid.parquet")
            component_map = {
                (source, group): component
                for source, group, component in split.components
            }
            accumulator = GlobalAccumulator()
            centered_by_component: dict[int, Any] = {}
            for member in expected_members:
                item = entry_map[member]
                prediction_path = args.output / item["file"]
                if (
                    prediction_path.is_symlink()
                    or prediction_path.resolve().parent != args.output.resolve()
                    or file_sha256(prediction_path) != item["sha256"]
                ):
                    raise ReadoutError("prediction file path or digest is invalid")
                values = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
                if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2):
                    raise ReadoutError("prediction array shape or dtype is invalid")
                if not np.isfinite(values).all():
                    raise ReadoutError("prediction array contains nonfinite values")
                sequence = labels.read(member[1])
                if sequence.targets is None or sequence.scored is None:
                    raise ReadoutError(
                        "official validation labels or mask are unavailable"
                    )
                mask = sequence.scored.copy()
                mask[:WARMUP] = False
                component = component_map[member]
                accumulator.add(sequence.targets, values, mask, group=component)
                if mask.any():
                    moments, _ = paired_component_moments(
                        sequence.targets[None],
                        values[None],
                        values[None],
                        mask[None],
                    )
                    previous = centered_by_component.get(
                        component, np.zeros((2, 6), dtype=np.float64)
                    )
                    centered_by_component[component] = np.stack(
                        [_merge_moments(previous[k], moments[0, k]) for k in range(2)]
                    )
            score = accumulator.result()
            if accumulator.selected_rows != VALID_SELECTED_ROWS:
                raise ReadoutError(
                    f"official validation selected-row count differs: {accumulator.selected_rows}"
                )
            monitor = _find_e3_monitor(fit["fit_dir"], admission, split)
            monitor_scores = _monitor_score_values(monitor)
            differences = {
                key: float(score[key]) - monitor_scores[key]
                for key in ("wp", "t0", "t1")
            }
            if any(abs(delta) > 1e-5 for delta in differences.values()):
                raise ReadoutError(
                    f"E3 report-only score differs from its saved monitor by >1e-5: {differences}"
                )
            score_receipt = {
                "score": score,
                "split_sha256": split.sha256,
                "predictions_manifest_sha256": _file_sha256(manifest_path),
                "producer": manifest["producer"],
                "sequences": len(expected_members),
                "component_raw_sums": {
                    str(key): value.tolist()
                    for key, value in accumulator.groups.items()
                },
                "component_centered_moments": {
                    str(key): value.tolist()
                    for key, value in centered_by_component.items()
                },
                "interpretation": (
                    "Report-only full official chronological-primary validation metric for a "
                    "predeclared retained E3 checkpoint; validation is exposed; this is not "
                    "fixed E4 endpoint evidence or checkpoint selection."
                ),
            }
            _write_new_json(args.output / "score.json", score_receipt)
            index = _read_object(fit["fit_dir"] / "epoch_index.json", 16_000_000)[0]
            model_sha = admission["entry"]["checkpoint_sha256"]
            report = {
                "schema": REPORT_SCHEMA,
                "status": "report_only_complete",
                "owner": args.owner,
                "cell_id": fit["cell"]["id"],
                "fixed_endpoint_epoch": 4,
                "selected_epoch": 3,
                "selection_policy": "predeclared E3 checkpoint; no validation-metric selection",
                "completed_epochs": 3,
                "global_step": admission["entry"]["global_step"],
                "checkpoint_sha256": model_sha,
                "epoch_index_sha256": index["index_sha256"],
                "source_sha256": admission["index"]["identity"]["source_sha256"],
                "config_sha256": admission["index"]["identity"]["config_sha256"],
                "data_sha256": admission["index"]["identity"]["data_sha256"],
                "score_sha256": _file_sha256(args.output / "score.json"),
                "monitor_e3": {
                    "epoch": monitor["epoch"],
                    "global_step": monitor["step"],
                    "wp": monitor_scores["wp"],
                    "t0": monitor_scores["t0"],
                    "t1": monitor_scores["t1"],
                    "max_abs_delta": max(abs(value) for value in differences.values()),
                    "tolerance": 1e-5,
                },
                "manifest_sha256": _file_sha256(args.output / "manifest.json"),
                "readout_adapter_sha256": _file_sha256(Path(__file__).resolve()),
                "split_sha256": SPLIT_SHA256,
                "sequences": VALID_SEQUENCES,
                "interpretation": (
                    "Report-only full official-valid metric for retained E3; validation is exposed; "
                    "fixed E4 endpoint is unavailable and this result is not an E4 replacement."
                ),
            }
            _write_new_json(args.output / "report_only.json", report)
            return True

        coordinated_call(score_report_only, "canonical report-only E3 score", context)
        barrier(context)
    finally:
        shutdown_distributed(context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_e3_monitor(
    fit_dir: Path, admission: dict[str, Any], split: Any
) -> dict[str, Any]:
    """Require the saved full-valid E3 monitor event and return its score identity."""
    from wnn.training.sequence_monitor_identity import membership_sha256

    expected_members = [("valid", int(group)) for group in split.evaluation_groups]
    path = fit_dir / "monitor_evaluations.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > (1 << 30):
        raise ReadoutError("saved monitor evaluation journal is missing or unsafe")
    matches: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 32_768 or not line.endswith(b"\n"):
                raise ReadoutError(
                    "saved monitor evaluation journal has an invalid line"
                )
            try:
                event = json.loads(
                    line.decode("utf-8"), parse_constant=_reject_constant
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                raise ReadoutError(
                    "saved monitor evaluation journal contains invalid JSON"
                ) from None
            if isinstance(event, dict) and event.get("epoch") == 3:
                matches.append(event)
    if len(matches) != 1:
        raise ReadoutError("saved monitor journal must contain exactly one E3 event")
    event = matches[0]
    metrics, context = event.get("metrics"), event.get("context")
    expected_step = admission["entry"]["global_step"]
    if (
        event.get("schema") != "wnn.monitor.v1"
        or event.get("kind") != "evaluation"
        or event.get("step") != expected_step
        or not isinstance(metrics, dict)
        or not isinstance(context, dict)
        or context.get("split_sha256") != split.sha256
        or context.get("metric_id") != "global_wp_clip2_v1"
        or context.get("evaluation_member_sha256")
        != membership_sha256(expected_members)
        or context.get("evaluation_member_count") != VALID_SEQUENCES
        or context.get("evaluation_population")
        != "complete official validation membership"
        or context.get("evaluation_mask") != "is_scored after warmup"
        or metrics.get("valid/sequence_count") != VALID_SEQUENCES
        or metrics.get("valid/selected_rows") != VALID_SELECTED_ROWS
    ):
        raise ReadoutError(
            "saved E3 monitor does not bind the official full-valid cohort"
        )
    for key in ("valid/wp", "valid/wp/t0", "valid/wp/t1"):
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ReadoutError("saved E3 monitor lacks finite pooled score fields")
        if not (-1.0 <= float(value) <= 1.0):
            raise ReadoutError("saved E3 monitor score is outside correlation bounds")
    return event


def _monitor_score_values(event: dict[str, Any]) -> dict[str, float]:
    """Map the actual monitor schema to the three canonical pooled score fields."""
    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        raise ReadoutError("saved E3 monitor has no metrics object")
    fields = {"wp": "valid/wp", "t0": "valid/wp/t0", "t1": "valid/wp/t1"}
    result: dict[str, float] = {}
    for key, field in fields.items():
        value = metrics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ReadoutError(f"saved E3 monitor lacks finite score field {field}")
        result[key] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--clock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", default=SPLIT_SHA256)
    parser.add_argument("--batch-sequences", type=int, default=DEFAULT_BATCH_SEQUENCES)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    if not args.owner.strip() or not 1 <= args.batch_sequences <= MAX_BATCH_SEQUENCES:
        parser.error("a named owner and batch size 1..64 are required")
    _validate_clock(args.clock)
    if time.time() >= WAVE_DEADLINE_EPOCH - 180:
        raise ReadoutError("fixed wave cleanup boundary reached before E3 export")

    packet_runtime = args.packet_root.resolve(strict=True) / "runtime"
    import sys

    sys.path.insert(0, str(packet_runtime))
    from impact81.bindings import PACKET, admit_runtime
    from impact81.runtime_family import TRANSFORMER_RUNTIME_COMMIT

    if PACKET.resolve(strict=True) != args.packet_root.resolve(strict=True):
        raise ReadoutError("imported Impact81 packet does not match --packet-root")
    admit_runtime(args.runtime_root, TRANSFORMER_RUNTIME_COMMIT)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _export_and_score(args)


if __name__ == "__main__":
    main()
