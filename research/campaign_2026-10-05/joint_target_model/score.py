"""Score every frozen paired-head arm on the manifest's evaluation groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from fit import ARMS, file_sha256, tick_gpu_duty, verify_fit_prefix
from model import JointTargetHead

ROOT = Path(__file__).resolve().parents[3]


def validate_eval_export(
    prefix: Path, fit_metadata: dict
) -> tuple[dict, dict[str, np.ndarray], object]:
    """Validate all-row evaluation artifacts and their disjoint frozen fold."""
    metadata_path = prefix.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("mode") != "eval" or metadata.get("selection") != "all rows":
        raise ValueError("evaluation scoring requires an all-row paired-state export")
    if metadata.get("manifest_sha256") != fit_metadata.get("manifest_sha256"):
        raise ValueError("fit and evaluation exports use different fold manifests")
    if metadata.get("members") != fit_metadata.get("members"):
        raise ValueError("fit and evaluation exports use different parent members")
    if metadata.get("member_checkpoint_sha256") != fit_metadata.get("member_checkpoint_sha256"):
        raise ValueError("fit and evaluation exports use different parent checkpoint bytes")
    split_path = (
        ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{metadata['fold']}.json"
    )
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, ROOT)
    if split.sha256 != metadata["manifest_sha256"]:
        raise ValueError("evaluation manifest hash differs from the validated frozen split")
    if list(split.evaluation_groups) != metadata.get("groups"):
        raise ValueError("evaluation groups differ from frozen split membership")
    fit_components = {
        component
        for source, group, component in split.components
        if source == "valid" and group in set(split.valid_groups)
    }
    eval_components = {
        component
        for source, group, component in split.components
        if source == "valid" and group in set(split.evaluation_groups)
    }
    if fit_components & eval_components:
        raise ValueError("fit and evaluation groups share a feature-only component")
    arrays: dict[str, np.ndarray] = {}
    for key, expected_hash in metadata["arrays_sha256"].items():
        path = prefix.with_name(f"{prefix.name}_{key}.npy")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"evaluation array changed after export: {key}")
        arrays[key] = np.load(path, mmap_mode="r")
    return metadata, arrays, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-prefix", type=Path, required=True)
    parser.add_argument("--eval-prefix", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if args.batch < 1 or args.boot < 2:
        parser.error("batch must be positive and bootstrap count at least two")
    duty = None
    if args.device == "cuda":
        sys.path.insert(0, str(ROOT / "src"))
        from wnn.resources import configure_torch

        duty = configure_torch()
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
    else:
        torch.set_num_threads(1)
    fit_prefix = args.fit_prefix.resolve()
    fit_metadata, _ = verify_fit_prefix(fit_prefix)
    eval_prefix = args.eval_prefix.resolve()
    eval_metadata, arrays, split = validate_eval_export(eval_prefix, fit_metadata)
    n_groups = len(eval_metadata["groups"])
    n_rows = int(eval_metadata["n_rows"])
    steps = 20_000
    hidden = int(eval_metadata["hidden"])
    expected_shapes = {
        "states": (n_rows, 2, hidden),
        "predictions": (n_rows, 2, 2),
        "targets": (n_rows, 2),
        "mask": (n_rows,),
        "row_group": (n_rows,),
    }
    if n_rows != n_groups * steps or any(
        arrays[key].shape != shape for key, shape in expected_shapes.items()
    ):
        raise ValueError("evaluation export has incomplete or unexpected shapes")
    groups = np.asarray(eval_metadata["groups"], dtype=np.int32)
    mask = arrays["mask"].reshape(n_groups, steps)
    if not np.array_equal(
        arrays["row_group"].reshape(n_groups, steps),
        np.broadcast_to(groups[:, None], (n_groups, steps)),
    ):
        raise ValueError("evaluation row order differs from frozen evaluation groups")
    from wnn.data import WARMUP

    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        cache_groups = np.asarray(cache["row_group"], dtype=np.int64)
        cache_masks = cache["mask"]
        cache_index = {int(group): index for index, group in enumerate(cache_groups)}
        if not set(map(int, groups)).issubset(cache_index):
            raise ValueError("evaluation group missing from official validation mask cache")
        for index, group in enumerate(groups):
            expected_mask = np.asarray(cache_masks[cache_index[int(group)]], dtype=bool).copy()
            expected_mask[:WARMUP] = False
            if not np.array_equal(mask[index], expected_mask):
                raise ValueError(f"official selected mask differs for evaluation group {group}")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "evaluation_groups": n_groups,
                    "evaluation_components": len(
                        {
                            component
                            for source, group, component in split.components
                            if source == "valid" and group in set(eval_metadata["groups"])
                        }
                    ),
                    "selected_rows": int(np.asarray(arrays["mask"]).sum()),
                    "manifest_sha256": split.sha256,
                    "evaluation_access": "read-only; all fixed groups included",
                }
            )
        )
        return
    if args.head_dir is None or args.output_dir is None:
        parser.error("--head-dir and --output-dir are required for scoring")
    head_dir = args.head_dir.resolve()
    receipt_path = head_dir / "fit_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("fit_export_metadata_sha256") != file_sha256(fit_prefix.with_suffix(".json")):
        raise ValueError("head run was not fitted from the supplied fit export")
    if receipt.get("fit_manifest_sha256") != fit_metadata["manifest_sha256"]:
        raise ValueError("head run fit manifest differs")
    if receipt.get("arms") != list(ARMS) or receipt.get("evaluation_access") != "none":
        raise ValueError("head receipt does not describe the fixed fit-only arm set")
    for member, expected_hash in zip(
        fit_metadata["members"], fit_metadata["member_checkpoint_sha256"], strict=True
    ):
        source_checkpoint = ROOT / "artifacts/runs" / member / "model_ep0.pt"
        if file_sha256(source_checkpoint) != expected_hash:
            raise ValueError(f"parent checkpoint changed since state export: {member}")

    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import METRIC_ID, GlobalAccumulator

    component_of = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    components = np.asarray([component_of[int(group)] for group in groups], dtype=np.int64)
    y = arrays["targets"].reshape(n_groups, steps, 2)
    mask = arrays["mask"].reshape(n_groups, steps)
    baseline_predictions = arrays["predictions"].mean(axis=1).reshape(n_groups, steps, 2)
    baseline_metric = GlobalAccumulator()
    baseline_moments: list[np.ndarray] = []
    for group_index in range(n_groups):
        target_rows = np.asarray(y[group_index : group_index + 1], dtype=np.float32)
        parent_rows = np.asarray(
            baseline_predictions[group_index : group_index + 1], dtype=np.float32
        )
        selected = np.asarray(mask[group_index : group_index + 1], dtype=bool)
        baseline_metric.add(target_rows, parent_rows, selected, group=int(groups[group_index]))
        moments, _ = paired_component_moments(target_rows, parent_rows, parent_rows, selected)
        baseline_moments.append(moments[0])
    baseline_moments_array = np.stack(baseline_moments)

    device = torch.device(args.device)
    output_path = args.output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict] = {"parent_pair_average": {"global_wp": baseline_metric.result()}}
    checkpoint_hashes: dict[str, str] = {}
    for arm in ARMS:
        path = head_dir / f"{arm}.pt"
        expected_hash = receipt.get("checkpoint_sha256", {}).get(arm)
        if not expected_hash or file_sha256(path) != expected_hash:
            raise ValueError(f"{arm} checkpoint SHA differs from fit receipt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("arm") != arm
            or checkpoint.get("fit_export_sha256") != receipt["fit_export_metadata_sha256"]
            or checkpoint.get("fit_manifest_sha256") != split.sha256
            or checkpoint.get("member_checkpoint_sha256")
            != fit_metadata["member_checkpoint_sha256"]
            or checkpoint.get("fit_array_sha256") != fit_metadata["arrays_sha256"]
            or checkpoint.get("fixed_epoch") != 0
        ):
            raise ValueError(f"{arm} checkpoint provenance differs from frozen fit receipt")
        head = JointTargetHead(
            hidden,
            torch.as_tensor(checkpoint["paired_codebook"]),
            torch.as_tensor(checkpoint["independent_codebook"]),
            torch.as_tensor(checkpoint["shuffled_codebook"]),
        ).to(device)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        metric = GlobalAccumulator()
        candidate_moments: list[np.ndarray] = []
        for group_index, group in enumerate(groups):
            start = group_index * steps
            stop = start + steps
            target_rows = np.asarray(y[group_index : group_index + 1], dtype=np.float32)
            mask_rows = np.asarray(mask[group_index : group_index + 1], dtype=bool)
            group_prediction = np.empty((1, steps, 2), dtype=np.float32)
            for batch_start in range(start, stop, args.batch):
                batch_stop = min(batch_start + args.batch, stop)
                state = torch.as_tensor(
                    np.asarray(arrays["states"][batch_start:batch_stop], dtype=np.float32),
                    device=device,
                )
                parent = torch.as_tensor(
                    np.asarray(arrays["predictions"][batch_start:batch_stop]), device=device
                )
                with torch.no_grad():
                    prediction, _, _ = head(state, parent, arm)
                values = prediction.cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{arm} emitted nonfinite evaluation predictions")
                local_start = batch_start - start
                group_prediction[0, local_start : local_start + len(values)] = values
                tick_gpu_duty(duty, device)
            metric.add(target_rows, group_prediction, mask_rows, group=int(group))
            moments, _ = paired_component_moments(
                target_rows,
                group_prediction,
                np.asarray(baseline_predictions[group_index : group_index + 1]),
                mask_rows,
            )
            candidate_moments.append(moments[0])
        paired = paired_component_bootstrap(
            np.stack(candidate_moments),
            baseline_moments_array,
            components,
            n_boot=args.boot,
            seed=args.seed,
            centered=True,
        )
        results[arm] = {"global_wp": metric.result(), "paired_vs_parent_pair_average": paired}
        checkpoint_hashes[arm] = expected_hash
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "scope": "fixed all-arm evaluation on one frozen component-held-out fold; no arm selection",
        "fit_export_metadata_sha256": file_sha256(fit_prefix.with_suffix(".json")),
        "eval_export_metadata_sha256": file_sha256(eval_prefix.with_suffix(".json")),
        "fit_manifest_sha256": split.sha256,
        "fit_groups": fit_metadata["groups"],
        "evaluation_groups": eval_metadata["groups"],
        "evaluation_components": sorted(set(map(int, components))),
        "parent_members": fit_metadata["members"],
        "parent_checkpoint_sha256": fit_metadata["member_checkpoint_sha256"],
        "head_checkpoint_sha256": checkpoint_hashes,
        "selection": "need_prediction AND is_scored",
        "metric_id": METRIC_ID,
        "public_anchor_comparison": "none; this historical fold is not compared with the current public 0.666 anchor",
        "arm_selection_or_tuning": "none",
        "scores": results,
    }
    (output_path / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output_dir": str(output_path), "arms_scored": list(ARMS), "metric_id": METRIC_ID}
        )
    )


if __name__ == "__main__":
    main()
