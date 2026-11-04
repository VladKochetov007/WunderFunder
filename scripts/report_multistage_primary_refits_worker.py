"""Score configured-final full-primary models and the two pre-fit fixed blends."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

from report_multistage_primary_ensemble_worker import (
    PRIMARY_ROWS,
    PRIMARY_SHA256,
    _canonical_sha256,
    _load_source6,
    _read_json,
    _regular,
    _sha256,
    _verify_source6,
    _write_new,
)

MASK_KEYS = (
    "membership_sha256",
    "mask_sha256",
    "selected_row_index_sha256",
    "sequence_order",
    "row_order",
    "selected_rows",
    "per_sequence_selected_rows",
    "mask",
    "mask_packed_bitorder",
)
SOURCE4 = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"


def validate_endpoint(
    payload: dict[str, object], member: dict[str, object], primary: object
) -> None:
    """Bind a complete payload to the approved fresh full-primary training identity."""
    identity, config = member["identity"], member["config"]
    provenance = payload.get("provenance", {})
    data = payload.get("data", {})
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or payload.get("completed_epochs") != config["epochs"]
        or payload.get("epoch") != config["epochs"] - 1
        or payload.get("config") != config
        or provenance.get("identity") != identity
        or provenance.get("fixed_checkpoint_policy") != "configured-final-epoch"
        or provenance.get("profile_receipt", {}).get("sha256")
        != member["artifacts"]["approved_profile"]["sha256"]
        or identity.get("source_sha256") != SOURCE4
        or data != identity["data"]
        or identity["config_sha256"] != _canonical_sha256(config)
        or data.get("fit_members") != [["train", group] for group in primary.train_groups]
        or data.get("evaluation_members")
        != [["valid", group] for group in primary.evaluation_groups]
        or data.get("split_sha256") != PRIMARY_SHA256
        or data.get("primary_split_sha256") not in (None, PRIMARY_SHA256)
        or data.get("evaluation_role") != "valid"
        or data.get("evaluation_source") != "valid"
        or data.get("evaluation_mask") != "is_scored"
        or data.get("teacher") is not None
        or data.get("initializer") is not None
        or config.get("precision_mode") != "fp32"
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or set(data.get("normalization", {}).get("source_fingerprints", {}))
        != {"datasets/train.parquet"}
    ):
        raise ValueError(
            "Source6-loaded model differs from its approved full-primary Source4 ancestry"
        )


def _clock(request: dict[str, object]) -> None:
    if time.time() >= min(
        float(request["hard_deadline_epoch"]), float(request["score_deadline_epoch"])
    ):
        raise RuntimeError("absolute scoring deadline reached")


def _setup(request: dict[str, object]) -> tuple[tuple[object, ...], object]:
    _clock(request)
    source = Path(request["source6"]["root"]).resolve(strict=True)
    _verify_source6(source, request["source6"])
    registry = Path(request["portable_data_registry"]["path"])
    split_path = Path(request["primary_split"]["path"])
    if (
        _sha256(registry) != request["portable_data_registry"]["sha256"]
        or _sha256(split_path) != PRIMARY_SHA256
    ):
        raise ValueError("official primary split or portable registry changed")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(registry)
    api = _load_source6(source)
    primary = api[3](split_path, Path(request["data_root"]))
    if (
        api[0] != PRIMARY_ROWS
        or primary.sha256 != PRIMARY_SHA256
        or len(primary.train_groups) != 10325
        or primary.valid_groups
        or len(primary.evaluation_groups) != 1873
        or primary.evaluation_source != "valid"
        or primary.evaluation_role != "valid"
        or primary.evaluation_mask != "is_scored"
    ):
        raise ValueError(
            "Source6 official primary membership/mask differs from the frozen protocol"
        )
    api[9]("fp32")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return api, primary


def _dataset(
    request: dict[str, object], api: tuple[object, ...], primary: object, mode: str = "raw112"
) -> object:
    dataset = api[5](
        Path(request["data_root"]),
        [("valid", group) for group in primary.evaluation_groups],
        0,
        training_mask="is_scored",
        input_feature_mode=mode,
    )
    dataset.validate_source_schema()
    return dataset


def score_member(request: dict[str, object], index: int) -> dict[str, object]:
    api, primary = _setup(request)
    member = request["members"][index]
    if member.get("configured_final_complete") is not True:
        raise ValueError("partial or censored endpoints cannot be scored as configured final")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != member["gpu_uuid"]:
        raise ValueError("scorer GPU visibility differs from the frozen assignment")
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "B300" not in torch.cuda.get_device_name(0).upper()
    ):
        raise ValueError("one visible assigned B300 is required")
    properties = torch.cuda.get_device_properties(0)
    raw_uuid = getattr(properties, "uuid", "")
    observed_uuid = (
        (raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else str(raw_uuid))
        .casefold()
        .removeprefix("gpu-")
    )
    if observed_uuid != member["gpu_uuid"].casefold().removeprefix("gpu-"):
        raise ValueError("actual Torch GPU UUID differs from the assigned scorer")
    checkpoint = _regular(Path(member["checkpoint"]["path"]), "configured-final model", 1 << 30)
    if checkpoint.name != "model.pt" or _sha256(checkpoint) != member["checkpoint"]["sha256"]:
        raise ValueError("configured-final model bytes changed")
    device = torch.device("cuda")
    model, payload = api[8](checkpoint, device)
    identity, config = member["identity"], member["config"]
    validate_endpoint(payload, member, primary)
    dataset = _dataset(request, api, primary, config["model"].get("input_feature_mode", "raw112"))
    path = Path(member["predictions_path"])
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists() or path.is_symlink() or temporary.is_symlink():
        raise FileExistsError("primary singleton predictions must be fresh")
    _clock(request)
    metrics, mask = api[2](model, dataset, int(config["chunk_rows"]), 64, device, temporary)
    _clock(request)
    if metrics.get("selected_rows") != PRIMARY_ROWS:
        raise ValueError("Source6 singleton score did not cover the complete official mask")
    temporary.replace(path)
    _write_new(path.with_suffix(".mask_order.json"), mask)
    result = {
        "schema": "wnn.full-primary-refit-singleton-score.v1",
        "status": "complete",
        "fit_id": member["fit_id"],
        "member_index": index,
        "checkpoint": member["checkpoint"],
        "metrics": metrics,
        "config_sha256": member["config_sha256"],
        "run_identity_sha256": _canonical_sha256(identity),
        "source6": request["source6"],
        "training_source_sha256": identity["source_sha256"],
        "primary_split_sha256": PRIMARY_SHA256,
        "gpu_uuid": member["gpu_uuid"],
        "batch_size": 64,
        "cpu_threads": 1,
        "objective_eligible": False,
        "predictions": {
            "path": str(path),
            "sha256": _sha256(path),
            "dtype": "float32",
            "shape": [PRIMARY_ROWS, 2],
            **mask,
        },
    }
    _write_new(Path(member["receipt_path"]), result)
    return result


def common_mask(receipts: list[dict[str, object]]) -> dict[str, object]:
    if len(receipts) != 4 or any(row.get("status") != "complete" for row in receipts):
        raise ValueError("both fixed four-member blends require all four complete singleton scores")
    masks = [{key: receipt["predictions"].get(key) for key in MASK_KEYS} for receipt in receipts]
    if (
        any(mask != masks[0] for mask in masks[1:])
        or masks[0].get("mask") != "is_scored AND row_index>=99"
        or masks[0].get("selected_rows") != PRIMARY_ROWS
        or any(
            not isinstance(masks[0].get(key), str) or len(masks[0][key]) != 64
            for key in ("membership_sha256", "mask_sha256", "selected_row_index_sha256")
        )
    ):
        raise ValueError("singleton official masks or scored-row orders differ")
    return masks[0]


def score_blends(request: dict[str, object]) -> dict[str, object]:
    import numpy as np
    from report_multistage_primary_ensemble_blend import (
        _load_scored_predictions,
        weighted_prediction_block,
    )

    api, primary = _setup(request)
    members = request["members"]
    if len(members) != 4 or not all(
        member.get("configured_final_complete") is True for member in members
    ):
        raise ValueError("partial refits prohibit both fixed four-member ensembles")
    receipts = [_read_json(Path(member["receipt_path"]), "singleton score") for member in members]
    mask = common_mask(receipts)
    arrays = []
    for member, receipt in zip(members, receipts, strict=True):
        prediction = receipt["predictions"]
        if (
            receipt.get("fit_id") != member["fit_id"]
            or receipt.get("checkpoint") != member["checkpoint"]
            or receipt.get("source6") != request["source6"]
            or receipt.get("primary_split_sha256") != PRIMARY_SHA256
            or receipt.get("config_sha256") != member["config_sha256"]
            or _sha256(Path(prediction["path"])) != prediction["sha256"]
        ):
            raise ValueError("singleton predictions changed before fixed blending")
        arrays.append(_load_scored_predictions(Path(prediction["path"])))
    output = Path(request["output_dir"])
    mixtures = {}
    for name, vector in request["weights"].items():
        target = output / f"{name}_scored_predictions.npy"
        temporary = target.with_name(target.name + ".partial")
        mixed = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(PRIMARY_ROWS, 2)
        )
        for start in range(0, PRIMARY_ROWS, 65536):
            _clock(request)
            stop = min(start + 65536, PRIMARY_ROWS)
            mixed[start:stop] = weighted_prediction_block(
                [array[start:stop] for array in arrays], np.asarray(vector, dtype=np.float64)
            )
        mixed.flush()
        del mixed
        temporary.replace(target)
        mixtures[name] = (target, np.load(target, mmap_mode="r", allow_pickle=False), api[1]())
    dataset = _dataset(request, api, primary)
    offset = 0
    mask_hash, order_hash = hashlib.sha256(), hashlib.sha256()
    sequence_counts = []
    for start in range(0, len(dataset), 64):
        _clock(request)
        indices = list(range(start, min(start + 64, len(dataset))))
        batch = api[6](dataset, indices)
        targets, masks = batch["y"].numpy(), batch["mask"].numpy()
        counts = []
        for position, index in enumerate(indices):
            selected = np.asarray(masks[position], dtype=bool)
            if np.any(selected[:99]):
                raise ValueError("official blend mask selected warmup rows")
            rows = np.flatnonzero(selected).astype("<u4", copy=False)
            group = dataset.members[index][1]
            mask_hash.update(np.packbits(selected, bitorder="little").tobytes())
            order_hash.update(np.asarray([group], dtype="<i8").tobytes())
            order_hash.update(rows.tobytes())
            counts.append(len(rows))
            sequence_counts.append({"row_group": group, "selected_rows": len(rows)})
        for _, predictions, accumulator in mixtures.values():
            batch_prediction = np.zeros_like(targets, dtype=np.float32)
            current = offset
            for position, count in enumerate(counts):
                batch_prediction[position, masks[position]] = predictions[current : current + count]
                current += count
            accumulator.add(targets, batch_prediction, masks)
        offset += sum(counts)
    recomputed = dict(
        mask,
        membership_sha256=api[7](dataset.members),
        mask_sha256=mask_hash.hexdigest(),
        selected_row_index_sha256=order_hash.hexdigest(),
        selected_rows=offset,
        per_sequence_selected_rows=sequence_counts,
    )
    if offset != PRIMARY_ROWS or recomputed != mask:
        raise ValueError("fixed-blend reconstructed mask/order differs from singleton scoring")
    results = {}
    for name, (path, _, accumulator) in mixtures.items():
        metrics = accumulator.result()
        if metrics["selected_rows"] != PRIMARY_ROWS:
            raise ValueError("fixed ensemble scoring omitted official rows")
        results[name] = {
            "weights": request["weights"][name],
            "metrics": metrics,
            "predictions": {
                "path": str(path),
                "sha256": _sha256(path),
                "dtype": "float32",
                "shape": [PRIMARY_ROWS, 2],
                **mask,
            },
        }
    result = {
        "schema": "wnn.full-primary-refit-fixed-blends.v1",
        "status": "complete",
        "ensembles": results,
        "frozen_weights": request["frozen_weights"],
        "primary_valid_used_for_weights": False,
    }
    _write_new(output / "fixed_ensemble_scores.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--mode", required=True, choices=("member", "blend"))
    parser.add_argument("--member-index", type=int)
    args = parser.parse_args()
    if _sha256(_regular(args.request, "score request", 64_000_000)) != args.request_sha256:
        raise ValueError("score request bytes changed")
    request = _read_json(args.request, "score request", 64_000_000)
    if request.get("schema") != "wnn.full-primary-refit-score-request.v1":
        raise ValueError("unsupported full-primary refit score request")
    if args.mode == "member":
        score_member(request, args.member_index)
    else:
        score_blends(request)


if __name__ == "__main__":
    main()
