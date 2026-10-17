#!/usr/bin/env python3
# ruff: noqa: I001
"""Bounded CPU replay of the deployed zero-i1 checkpoint preprocessing fix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "WNN_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WNN_PREDICT_FP32"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402


ROOT = Path("/home/grinchenkoeg/WunderFunder-creative")
OUT = Path("/home/grinchenkoeg/wunder_research_reset_20261017_r1/zero_i1_replay")
TRAINER_PATH = ROOT / "research/iterations/2026-10-08-shift/plew19/train_rnn_plew19.py"
PREDICTOR_PATH = ROOT / "research/iterations/2026-10-08-shift/plew19/predict_any_env.py"
CHECKPOINT_PATH = ROOT / "artifacts/runs/dftt_zi1_s3/model_ep1.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_deployed_sources():
    os.environ["WUNDER_RESEARCH_ROOT"] = str(ROOT)
    trainer_dir = str(TRAINER_PATH.parent)
    source_dir = str(ROOT / "src")
    sys.path[:0] = [trainer_dir, source_dir]
    import train_rnn_plew19 as trainer

    spec = importlib.util.spec_from_file_location("deployed_predict_any_env", PREDICTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import patched predictor from {PREDICTOR_PATH}")
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    if predictor.T is not trainer:
        raise RuntimeError("predictor did not import the expected live trainer module")
    return trainer, predictor


def strict425_groups(row_groups: np.ndarray, fold_table: np.ndarray) -> list[int]:
    split = fold_table["split"].astype(str)
    train_chains = set(fold_table["chain"][split == "train"].tolist())
    valid = split == "valid"
    strict = {
        int(group)
        for group, chain, fold in zip(
            fold_table["row_group"][valid],
            fold_table["chain"][valid],
            fold_table["fold"][valid],
            strict=True,
        )
        if int(fold) == 0 and chain not in train_chains
    }
    if len(strict) != 425:
        raise ValueError(f"expected the frozen strict425 panel; found {len(strict)} groups")
    ordered = [int(group) for group in row_groups if int(group) in strict]
    if len(ordered) != len(strict) or len(set(ordered)) != len(ordered):
        raise ValueError("strict425 groups do not map uniquely into canonical valid-target order")
    return ordered


def score_pair(metric, targets, masks, bad_predictions, fixed_predictions):
    target_array = np.stack(targets)
    mask_array = np.stack(masks).astype(bool, copy=False)
    mask_array[:, :99] = False
    bad_result = metric.global_wp(target_array, np.stack(bad_predictions), mask_array)
    fixed_result = metric.global_wp(target_array, np.stack(fixed_predictions), mask_array)

    def summary(result):
        return {
            "wp": result["wp"],
            "t0": result["t0"],
            "t1": result["t1"],
            "selected_rows": int(mask_array.sum()),
            "metric_id": metric.METRIC_ID,
        }

    return {"bad": summary(bad_result), "fixed": summary(fixed_result)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must remain unavailable during the CPU replay")

    started = time.monotonic()
    trainer, predictor = import_deployed_sources()
    import_hashes = {
        "predictor_sha256": sha256_file(PREDICTOR_PATH),
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "metric_sha256": sha256_file(OUT / "REPLAY_gmetric.py"),
        "feature_sha256": sha256_file(ROOT / "src/wnn/features.py"),
        "data_source_sha256": sha256_file(ROOT / "src/wnn/data.py"),
        "model_source_sha256": sha256_file(ROOT / "src/wnn/nn.py"),
    }
    expected_source_hashes = {
        "predictor_sha256": "786896a87186ed8878b8ae31c49fa015a231f71d505ce6e5b43be2edaf7a5f1c",
        "trainer_sha256": "8a34b74f5df9a13030c2102a22443cee6d74be6dde847f3052679b0f013d5903",
    }
    if any(import_hashes[key] != value for key, value in expected_source_hashes.items()):
        raise RuntimeError(f"deployed predictor/trainer source hashes differ: {import_hashes}")

    checkpoint_sha_before = sha256_file(CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    args = checkpoint["args"] if isinstance(checkpoint["args"], dict) else vars(checkpoint["args"])
    if args.get("name") != "dftt_zi1_s3" or args.get("zero_groups") != "i1":
        raise ValueError("checkpoint args do not identify the expected dftt_zi1_s3 zero-i1 run")
    if args.get("lookahead"):
        raise ValueError("the requested causal replay cannot use lookahead features")

    cpu = torch.device("cpu")
    predictor.restore_checkpoint_settings(trainer, args, checkpoint, ROOT, cpu)
    restored_zero_cols = trainer.ZERO_COLS.detach().cpu().clone()
    manual_zero_cols = torch.tensor(trainer.zero_group_cols(args["zero_groups"]), dtype=torch.long)
    if not torch.equal(restored_zero_cols, manual_zero_cols):
        raise AssertionError(
            "patched restore ZERO_COLS differs from the trainer's manual expected tensor"
        )
    if restored_zero_cols.tolist() != list(range(52, 104)):
        raise AssertionError("zero_groups='i1' did not resolve to the expected raw feature columns")

    target_path = ROOT / "cache/valid_targets.npz"
    fold_path = ROOT / "cache/chain_folds.npz"
    data_path = ROOT / "datasets/valid.parquet"
    input_hashes = {
        "valid_targets_sha256": sha256_file(target_path),
        "chain_folds_sha256": sha256_file(fold_path),
        "valid_parquet_sha256": sha256_file(data_path),
    }
    target_archive = np.load(target_path)
    row_groups = np.asarray(target_archive["row_group"])
    group_limit = int(os.environ.get("REPLAY_GROUP_LIMIT", "4"))
    if group_limit not in {1, 4, 16}:
        raise ValueError("REPLAY_GROUP_LIMIT must be 1, 4, or 16")
    selected_groups = strict425_groups(row_groups, np.load(fold_path))[:group_limit]
    if selected_groups[:4] != [3, 6, 14, 15][:group_limit]:
        raise AssertionError(f"canonical first four strict425 groups changed: {selected_groups}")
    row_positions = {int(group): index for index, group in enumerate(row_groups)}
    membership_bytes = np.asarray(selected_groups, dtype="<i8").tobytes()
    membership_sha256 = hashlib.sha256(membership_bytes).hexdigest()

    output_count = int(checkpoint["model"]["head.2.weight"].shape[0])
    model = trainer.make_model(args, checkpoint["n_in"], output_count)
    model.load_state_dict(checkpoint["model"])
    model.to(cpu).eval()
    metric_spec = importlib.util.spec_from_file_location(
        "replay_canonical_gmetric", OUT / "REPLAY_gmetric.py"
    )
    if metric_spec is None or metric_spec.loader is None:
        raise RuntimeError("cannot load the vendored canonical pooled gmetric source")
    official_metric = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(official_metric)
    if official_metric.METRIC_ID != "global_wp_clip2_v1":
        raise RuntimeError(f"unexpected canonical metric id: {official_metric.METRIC_ID}")

    dataset = trainer.Sequences("valid", selected_groups)
    targets, masks, bad_predictions, fixed_predictions = [], [], [], []
    group_records = []
    feature_max_diffs, prediction_max_diffs = [], []
    wall_per_group = []

    with torch.inference_mode():
        for sequence_index, group in enumerate(selected_groups):
            item_started = time.monotonic()
            x, y, mask = dataset[sequence_index]
            # Compare SequenceFile labels with the canonical frozen target cache before scoring.
            cache_row = row_positions[group]
            canonical_y = np.asarray(target_archive["y"][cache_row], dtype=np.float32)
            canonical_mask = np.asarray(target_archive["mask"][cache_row], dtype=bool)
            seq_y = y.detach().cpu().numpy().astype(np.float32, copy=False)
            seq_mask = mask.detach().cpu().numpy().astype(bool, copy=False)
            if not np.array_equal(seq_y, canonical_y) or not np.array_equal(
                seq_mask, canonical_mask
            ):
                raise ValueError(
                    f"SequenceFile labels/mask differ from canonical cache for group {group}"
                )

            trainer.ZERO_COLS = manual_zero_cols.clone()
            x_batch = x.unsqueeze(0)
            expected_inputs = trainer.build_inputs(x_batch)
            trainer.ZERO_COLS = restored_zero_cols.clone()
            restored_inputs = trainer.build_inputs(x_batch)
            if not torch.equal(expected_inputs, restored_inputs):
                raise AssertionError(
                    f"manual and patched restore input tensors differ for group {group}"
                )

            trainer.ZERO_COLS = None
            unmasked_inputs = trainer.build_inputs(x_batch)
            input_diff = (unmasked_inputs.float() - restored_inputs.float()).abs()
            feature_max_diffs.append(float(input_diff.max().item()))

            trainer.ZERO_COLS = manual_zero_cols.clone()
            manual_pred = predictor.validate_prediction_batch(
                trainer.predict(model, x_batch, 4000, cpu)
            )
            trainer.ZERO_COLS = restored_zero_cols.clone()
            fixed_pred = predictor.validate_prediction_batch(
                trainer.predict(model, x_batch, 4000, cpu)
            )
            if not np.array_equal(manual_pred, fixed_pred):
                raise AssertionError(
                    f"manual and restored predictions are not bitwise equal for group {group}"
                )

            trainer.ZERO_COLS = None
            bad_pred = predictor.validate_prediction_batch(
                trainer.predict(model, x_batch, 4000, cpu)
            )
            prediction_diff = np.abs(bad_pred.astype(np.float64) - fixed_pred.astype(np.float64))
            prediction_max_diffs.append(float(prediction_diff.max()))
            targets.append(canonical_y)
            masks.append(canonical_mask)
            bad_predictions.append(bad_pred[0])
            fixed_predictions.append(fixed_pred[0])
            wall_per_group.append(time.monotonic() - item_started)
            group_records.append(
                {
                    "row_group": group,
                    "canonical_target_position": cache_row,
                    "selected_rows": int(canonical_mask.sum()),
                    "manual_restore_inputs_bitwise_equal": True,
                    "manual_restore_predictions_bitwise_equal": True,
                    "bad_vs_fixed_prediction_max_abs": prediction_max_diffs[-1],
                    "bad_vs_fixed_input_max_abs": feature_max_diffs[-1],
                }
            )
            trainer.ZERO_COLS = restored_zero_cols.clone()
            print(
                json.dumps(
                    {"event": "group_complete", "row_group": group, "seconds": wall_per_group[-1]},
                    sort_keys=True,
                ),
                flush=True,
            )

    metric_results = score_pair(
        official_metric, targets, masks, bad_predictions, fixed_predictions
    )
    checkpoint_sha_after = sha256_file(CHECKPOINT_PATH)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise AssertionError("checkpoint bytes changed during inference")

    result = {
        "schema": "wunder.zero-i1-replay.v2",
        "status": "complete",
        "host": "beleriand",
        "run": "dftt_zi1_s3",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_unchanged": True,
        "checkpoint_args": {
            key: args.get(key)
            for key in (
                "name",
                "zero_groups",
                "lookahead",
                "state_reset",
                "sentinel_mode",
                "a_mode",
                "domain_flag",
                "raw_only",
                "heuristics",
                "quant_feats",
                "quant_i1",
                "online_norm",
                "prop_input",
            )
        },
        "source_sha256": import_hashes,
        "data_sha256": input_hashes,
        "selection": {
            "rule": f"first {len(selected_groups)} groups in valid_targets row order from strict425; selected without model scores",
            "strict_group_count": 425,
            "row_groups": selected_groups,
            "membership_sha256": membership_sha256,
        },
        "execution": {
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "optimizer_steps": 0,
            "model_mode": "eval",
            "inference_mode": True,
            "timed_out": False,
            "wall_seconds": time.monotonic() - started,
            "seconds_per_group": wall_per_group,
        },
        "zero_cols": {
            "manual_expected": manual_zero_cols.tolist(),
            "patched_restore": restored_zero_cols.tolist(),
            "manual_and_restore_bitwise_equal": torch.equal(manual_zero_cols, restored_zero_cols),
            "original_wrong_setting": None,
        },
        "metric": {
            "id": official_metric.METRIC_ID,
            "source_sha256": import_hashes["metric_sha256"],
            "warmup_rows_excluded": 99,
            "same_groups_and_canonical_labels_for_both": True,
            "scores": metric_results,
            "interpretation": "A four-group diagnostic replay only; not a full strict425 ranking or proof of model-quality gain.",
        },
        "feature_input_diffs": feature_max_diffs,
        "prediction_diffs": prediction_max_diffs,
        "groups": group_records,
    }
    output_path = OUT / "REPLAY_zero_i1_v2.json"
    markdown_path = OUT / "REPLAY_zero_i1_v2.md"
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("replay result already exists; choose a fresh output directory")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(render_markdown(result))
    print(
        json.dumps(
            {
                "event": "replay_complete",
                "result": str(output_path),
                "scores": metric_results,
                "wall_seconds": result["execution"]["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def render_markdown(result: dict[str, object]) -> str:
    execution = result["execution"]
    selection = result["selection"]
    metric = result["metric"]
    scores = metric["scores"]
    zero_cols = result["zero_cols"]
    lines = [
        "# Zero-i1 CPU replay",
        "",
        "This is a bounded inference diagnostic using the deployed predictor restore function and the existing `dftt_zi1_s3/model_ep1.pt` checkpoint. No optimizer or fit step ran.",
        "",
        f"- Checkpoint SHA-256 before/after: `{result['checkpoint_sha256_before']}` / `{result['checkpoint_sha256_after']}`; unchanged: **{result['checkpoint_unchanged']}**.",
        f"- Selected {len(selection['row_groups'])} of the deterministic strict425 panel before model scores: `{selection['row_groups']}`.",
        f"- CPU threads: intra-op {execution['torch_num_threads']}, inter-op {execution['torch_num_interop_threads']}; wall time {execution['wall_seconds']:.1f}s.",
        f"- Manual expected ZERO_COLS equals patched restore bitwise: **{zero_cols['manual_and_restore_bitwise_equal']}**; expected columns are `52..103`.",
        f"- Metric: `{metric['id']}` on the same selected groups and canonical labels/masks.",
        f"- Bad behavior (`ZERO_COLS=None`): WP {scores['bad']['wp']:.8f} (t0 {scores['bad']['t0']:.8f}, t1 {scores['bad']['t1']:.8f}).",
        f"- Patched restore: WP {scores['fixed']['wp']:.8f} (t0 {scores['fixed']['t0']:.8f}, t1 {scores['fixed']['t1']:.8f}).",
        f"- Largest bad-vs-fixed prediction absolute difference: {max(result['prediction_diffs'], default=0.0):.8g}.",
        "",
        "This four-group slice is diagnostic only. It is not a full strict425 ranking and does not establish a model-quality gain.",
        "",
        "See `REPLAY_zero_i1_v2.json` for checkpoint/source/data hashes, exact group membership, per-group comparisons, and timing.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
