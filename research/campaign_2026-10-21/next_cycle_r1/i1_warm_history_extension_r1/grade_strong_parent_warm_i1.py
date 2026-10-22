"""Grade the frozen strong-parent warm-history F0/F1 panel without tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
PLAN_PATH = PACKET / "STRONG_PARENT_WARM_I1_PLAN_R1.json"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-21/next_cycle_r1/i1"))

from grade_i1_panel import _mixture, _npy_header  # noqa: E402
from grade_prediction_bank import aligned, score, sha256  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402

ROWS = 20_000
FULL_VALID = 1_873
PARENT_WIDTH = 211
CHILD_WIDTH = 367
N_BOOT = 2_000
BOOTSTRAP_SEED = 271_828
TIME_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))
TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"


def require(ok: bool, message: str) -> None:
    """Fail closed when a frozen artifact contract is not met."""
    if not ok:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from a source or run receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def json_sha256(value: object) -> str:
    """Hash canonical JSON using the training provenance convention."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def groups_sha256(groups: list[int]) -> str:
    """Hash ordered row-group identities as little-endian int64 values."""
    return hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()


def planned_args(argv: list[str]) -> dict[str, object]:
    """Parse explicit flags from one frozen fit argv without invoking the trainer."""
    result: dict[str, object] = {}
    tokens = argv[1:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional fit argument: {token}")
        name = token[2:].replace("-", "_")
        if index + 1 == len(tokens) or tokens[index + 1].startswith("--"):
            value: object = True
            index += 1
        else:
            value = tokens[index + 1]
            index += 2
        if name in result:
            raise ValueError(f"duplicate frozen fit option: {name}")
        result[name] = value
    return result


def verify_explicit_args(run: str, argv: list[str], actual: dict[str, Any]) -> None:
    """Compare every explicit fitted option with its stored provenance value."""
    for name, raw in planned_args(argv).items():
        require(name in actual, f"{run}: provenance omits planned option {name}")
        value = actual[name]
        if isinstance(value, bool):
            expected: object = True
            require(raw is True, f"{run}: planned flag {name} is malformed")
        elif isinstance(value, int):
            expected = int(str(raw))
        elif isinstance(value, float):
            expected = float(str(raw))
        else:
            expected = str(raw)
        require(value == expected, f"{run}: planned option {name} differs")
    require(actual.get("resume_state") in (None, ""), f"{run}: unexpected resume")
    require(
        actual.get("mean_variance_objective") in (None, False, 0),
        f"{run}: unexpected variance objective",
    )


def verify_sources(plan: dict[str, Any]) -> dict[str, str]:
    """Verify all byte pins frozen in the plan; no training module is imported."""
    pins = plan["source"]["current_source_sha256"]
    for relative, expected in pins.items():
        path = (ROOT / relative).resolve(strict=True)
        path.relative_to(ROOT.resolve())
        require(sha256(path) == expected, f"source pin differs: {relative}")
    return dict(pins)


def verify_paired_plan(plan: dict[str, Any]) -> None:
    """Confirm each pair differs only in its cache name and history visibility."""
    for cell in plan["cells"]:
        fold = int(cell["fold"])
        pair = {spec["arm"]: spec for spec in plan["runs"] if int(spec["fold"]) == fold}
        require(set(pair) == {"control", "history"}, f"F{fold}: frozen pair is incomplete")
        control = planned_args(pair["control"]["fit_argv"])
        history = planned_args(pair["history"]["fit_argv"])
        for values in (control, history):
            values.pop("name", None)
            values.pop("research_feature_control", None)
        require(control == history, f"F{fold}: frozen fit commands differ beyond the intended arm")


def verify_cell_inputs(cell: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    """Bind split, parent, target cache, teachers and propensity to the plan."""
    fold = int(cell["fold"])
    source = cell["split"]
    split_path = ROOT / source["path"]
    split_sha = sha256(split_path)
    require(split_sha == source["sha256"], f"F{fold}: split hash differs")
    require(
        len(split["fit"]["train"]) == source["train_fit_groups"]
        and len(split["fit"]["valid"]) == source["valid_fit_groups"],
        f"F{fold}: full fit membership counts differ",
    )

    parent = cell["parent"]
    parent_checkpoint = ROOT / parent["checkpoint"]
    parent_provenance = parent_checkpoint.parent / "provenance.json"
    parent_sha, parent_prov_sha = sha256(parent_checkpoint), sha256(parent_provenance)
    require(
        parent_sha == parent["checkpoint_sha256"]
        and parent_prov_sha == parent["provenance_sha256"],
        f"F{fold}: direct parent checkpoint/provenance differs",
    )

    cache = cell["target_cache"]
    metadata_path = ROOT / cache["root"] / "metadata.json"
    metadata = load_json(metadata_path)
    require(
        sha256(metadata_path) == cache["metadata_sha256"],
        f"F{fold}: target-cache metadata hash differs",
    )
    require(
        metadata.get("split_manifest_sha256") == split_sha
        and metadata.get("fold") == fold
        and metadata.get("fit_groups") == split["fit"]
        and metadata.get("evaluation_valid_groups") == split["evaluation"]["valid"],
        f"F{fold}: target-cache membership differs",
    )
    expected_payloads = {
        f"main_{cache['variant']}_train": cache["main_train_sha256"],
        f"main_{cache['variant']}_valid": cache["main_valid_sha256"],
    }
    require(
        all(
            metadata.get("files", {}).get(name, {}).get("sha256") == digest
            for name, digest in expected_payloads.items()
        ),
        f"F{fold}: frozen target-cache payload hash differs",
    )
    teacher_refs = [
        f"{item['run']}@ep{int(item['epoch'])}" for item in metadata.get("source_coefficients", [])
    ]
    require(
        teacher_refs == cache["teacher_checkpoint_refs"]
        and len(teacher_refs) == cache["fixed_teacher_count"],
        f"F{fold}: cache teacher lineage differs",
    )
    source_records = {item["run"]: item for item in metadata.get("sources", [])}
    require(
        set(source_records) == {item["run"] for item in metadata["source_coefficients"]},
        f"F{fold}: cache source checkpoint records differ",
    )
    teacher_ancestry = []
    for item in metadata["source_coefficients"]:
        record = source_records[item["run"]]
        require(
            record.get("epoch") == item["epoch"] and record.get("checkpoint_sha256"),
            f"F{fold}: teacher checkpoint identity is incomplete",
        )
        teacher_ancestry.append(
            {
                "run": item["run"],
                "epoch": item["epoch"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "provenance_sha256": record["provenance_sha256"],
                "ancestor_closure": record.get("ancestor_closure", []),
            }
        )

    propensity = cell["propensity"]
    propensity_dir = ROOT / "artifacts/runs" / propensity["run"]
    require(
        sha256(propensity_dir / "model.pt") == propensity["checkpoint_sha256"]
        and sha256(propensity_dir / "provenance.json") == propensity["provenance_sha256"],
        f"F{fold}: propensity checkpoint/provenance differs",
    )
    return {
        "direct_parent": {
            "run": parent["run"],
            "checkpoint_sha256": parent_sha,
            "provenance_sha256": parent_prov_sha,
            "initial_ancestor": parent["initial_ancestor"],
            "initial_ancestor_checkpoint_sha256": parent["initial_ancestor_checkpoint_sha256"],
            "ancestry_depth": "only the source-qualified depth in the frozen plan is claimed",
        },
        "target_cache": {
            "root": cache["root"],
            "variant": cache["variant"],
            "metadata_sha256": cache["metadata_sha256"],
            "verified_payload_sha256": expected_payloads,
            "teacher_checkpoint_refs": teacher_refs,
            "teacher_source_ancestry": teacher_ancestry,
            "evaluation_overlap": cache["evaluation_overlap"],
        },
        "propensity": {
            "run": propensity["run"],
            "checkpoint_sha256": propensity["checkpoint_sha256"],
            "provenance_sha256": propensity["provenance_sha256"],
        },
    }


def verify_membership(
    run: str, provenance: dict[str, Any], split: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    """Require exact fit-order digests and exclusion of the complete eval fold."""
    recorded = provenance.get("fit_row_groups", {})
    train = list(map(int, recorded.get("train", [])))
    valid = list(map(int, recorded.get("valid", [])))
    eval_groups = list(map(int, split["evaluation"]["valid"]))
    expected = cell["split"]
    require(
        len(train) == len(set(train)) == expected["train_fit_groups"]
        and set(train) == set(map(int, split["fit"]["train"])),
        f"{run}: train membership differs",
    )
    require(
        len(valid) == len(set(valid)) == expected["valid_fit_groups"]
        and set(valid) == set(map(int, split["fit"]["valid"])),
        f"{run}: valid-fit membership differs",
    )
    train_hash, valid_hash = groups_sha256(train), groups_sha256(valid)
    require(
        train_hash == expected["fit_train_order_sha256"]
        and valid_hash == expected["fit_valid_order_sha256"],
        f"{run}: fit row-group order differs",
    )
    require(not (set(train + valid) & set(eval_groups)), f"{run}: fit/eval overlap")
    require(
        set(map(int, provenance.get("evaluation_row_groups", []))) == set(eval_groups),
        f"{run}: eval-fold provenance differs",
    )
    identity = provenance.get("fit_membership_identity", {})
    for name, values, digest in (("train", train, train_hash), ("valid", valid, valid_hash)):
        saved = identity.get(name, {})
        require(
            saved.get("count") == len(values) and saved.get("ordered_int64_le_sha256") == digest,
            f"{run}: stored {name} membership identity differs",
        )
    require(identity.get("valid_repeat") == 1, f"{run}: valid repeat differs")
    return {
        "train_groups": len(train),
        "fit_valid_groups": len(valid),
        "train_order_sha256": train_hash,
        "valid_order_sha256": valid_hash,
        "evaluation_groups_excluded": len(eval_groups),
    }


def verify_training_receipt(
    run_spec: dict[str, Any],
    provenance: dict[str, Any],
    plan: dict[str, Any],
    cell: dict[str, Any],
    split: dict[str, Any],
    pins: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check fit recipe, source-qualified prior/cache and exact update clocks."""
    run, args = run_spec["run_id"], provenance.get("args", {})
    verify_explicit_args(run, run_spec["fit_argv"], args)
    require(
        provenance.get("seed") == run_spec["seed"]
        and provenance.get("split_manifest_sha256") == cell["split"]["sha256"]
        and provenance.get("initial_checkpoint_sha256") == cell["parent"]["checkpoint_sha256"]
        and provenance.get("propensity_checkpoint_sha256")
        == cell["propensity"]["checkpoint_sha256"]
        and provenance.get("selected_epoch") == 1,
        f"{run}: seed/split/parent/fixed epoch differs",
    )
    require(
        args.get("name") == run
        and args.get("epochs") == 2
        and args.get("checkpoint_policy") == "fixed"
        and args.get("checkpoint_epoch") == 1
        and args.get("no_eval_during_fit") is True
        and args.get("precision") == "fp32"
        and args.get("research_features") == "i1_history"
        and args.get("research_feature_control") == run_spec["research_feature_control"]
        and args.get("drop_i1") is False
        and args.get("i1_warm_normalization") == "fit-appended",
        f"{run}: fixed fit contract differs",
    )
    require(provenance.get("initial_parameter_sha256"), f"{run}: initial parameter hash missing")
    feature_control = provenance.get("causal_feature_control", {})
    expected_zeroed = 156 if run_spec["arm"] == "control" else 0
    require(
        feature_control.get("kind") == run_spec["research_feature_control"]
        and feature_control.get("drop_i1") is False
        and feature_control.get("zeroed_feature_count") == expected_zeroed
        and feature_control.get("mask_point") == "after stored model normalization",
        f"{run}: history feature-control provenance differs",
    )

    extension = provenance.get("research_feature_extension", {})
    require(
        extension.get("parent_mode") == "none"
        and extension.get("child_mode") == "i1_history"
        and extension.get("input_weights") == "zero_initialized"
        and extension.get("normalization")
        == "inherited_parent_prefix_and_fitting_only_appended_slots",
        f"{run}: feature-extension ancestry differs",
    )
    require(
        provenance.get("mean_variance_head_extension")
        == {"parent_output_rows": 23, "child_output_rows": 25, "added_rows": "zero initialized"},
        f"{run}: 23-to-25 output extension differs",
    )
    warm = provenance.get("i1_warm_normalization", {})
    parent = cell["parent"]
    require(
        warm.get("mode") == "fit-appended"
        and warm.get("parent_width") == PARENT_WIDTH
        and warm.get("child_width") == CHILD_WIDTH
        and warm.get("appended_width") == 156
        and warm.get("parent_mean_identity", {}).get("bytes_sha256") == parent["mean_identity"]
        and warm.get("parent_std_identity", {}).get("bytes_sha256") == parent["std_identity"],
        f"{run}: inherited/fitted normalization record differs",
    )

    prior = provenance.get("causal_feature_prior", {})
    scale = np.asarray(prior.get("scale", []), dtype=np.float32)
    require(
        prior.get("schema") == "i1_causal_history_v1"
        and scale.shape == (52,)
        and np.isfinite(scale).all()
        and np.all(scale > 0)
        and prior.get("sample_policy") == "uniform_seeded_source_qualified_64_groups_v1"
        and prior.get("sample_seed") == int(cell["fit_seed"]) + 20_261_012,
        f"{run}: fitting-only i1 prior contract differs",
    )
    members = prior.get("fit_members", [])
    fit = {"train": set(split["fit"]["train"]), "valid": set(split["fit"]["valid"])}
    require(
        len(members) == 64
        and prior.get("fit_members_sha256") == json_sha256(members)
        and all(
            item.get("split") in fit and int(item["row_group"]) in fit[item["split"]]
            for item in members
        ),
        f"{run}: prior membership is missing, changed or outside fit groups",
    )
    prior_paths = (
        "src/wnn/research_features.py",
        "src/wnn/i1_history.py",
        "src/wnn/i1_history_scan.py",
        "src/wnn/i1_history_streaming.py",
        "src/wnn/features.py",
        "src/wnn/training/inputs.py",
    )
    expected_prior_sources = {path: pins[path] for path in prior_paths}
    prior_sources = provenance.get("causal_feature_prior_source_sha256", {})
    require(prior_sources == expected_prior_sources, f"{run}: prior source hashes differ")

    cache_spec = cell["target_cache"]
    cache = provenance.get("distillation_cache", {})
    expected_files = {
        f"main_{cache_spec['variant']}_train": cache_spec["main_train_sha256"],
        f"main_{cache_spec['variant']}_valid": cache_spec["main_valid_sha256"],
    }
    refs = [
        f"{item['run']}@ep{int(item['epoch'])}" for item in cache.get("source_coefficients", [])
    ]
    require(
        Path(str(cache.get("root", ""))).resolve() == (ROOT / cache_spec["root"]).resolve()
        and cache.get("variant") == cache_spec["variant"]
        and float(cache.get("main_target_scale", -1)) == 1.0
        and cache.get("main_target_scale_applied_before_training_clip") is True
        and cache.get("metadata_sha256") == cache_spec["metadata_sha256"]
        and cache.get("verified_files") == expected_files
        and refs == cache_spec["teacher_checkpoint_refs"],
        f"{run}: target/cache/teacher provenance differs",
    )

    diagnostics = provenance.get("update_diagnostics")
    require(
        isinstance(diagnostics, list) and len(diagnostics) == 2,
        f"{run}: expected exactly two epoch receipts",
    )
    summary = []
    counters = (
        "optimizer_attempts",
        "optimizer_completed_updates",
        "amp_skipped_updates",
        "completed_batches",
    )
    for index, record in enumerate(diagnostics):
        epoch = record.get("epoch", {})
        require(
            record.get("epoch_index") == index
            and epoch.get("optimizer_attempts") == 7_460
            and epoch.get("optimizer_completed_updates") == 7_460
            and epoch.get("amp_skipped_updates") == 0
            and epoch.get("completed_batches") == 746,
            f"{run}: epoch {index} update evidence differs",
        )
        cumulative = record.get("cumulative", {})
        expected_updates, expected_batches = 7_460 * (index + 1), 746 * (index + 1)
        require(
            cumulative.get("optimizer_attempts") == expected_updates
            and cumulative.get("optimizer_completed_updates") == expected_updates
            and cumulative.get("amp_skipped_updates") == 0
            and cumulative.get("completed_batches") == expected_batches,
            f"{run}: cumulative epoch {index} update evidence differs",
        )
        summary.append(
            {
                "epoch_index": index,
                "counts": {key: epoch[key] for key in counters},
                "sample_order_identity": record["sample_order_identity"],
                "normalization_identity": record["normalization_identity"],
            }
        )

    runtime = provenance.get("trainer_implementation", {})
    files = runtime.get("files", {})
    raw_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(
        files and hashlib.sha256(raw_files).hexdigest() == runtime.get("sha256"),
        f"{run}: complete trainer source identity is malformed",
    )
    for relative, digest in pins.items():
        if relative in files:
            require(files[relative] == digest, f"{run}: training source pin differs: {relative}")
    for relative in (
        "scripts/train_rnn.py",
        "src/wnn/training/setup_model.py",
        "src/wnn/training/setup_preprocessing.py",
    ):
        require(
            relative in files and files[relative] == pins[relative],
            f"{run}: runtime source identity omits or changes {relative}",
        )
    return {
        "initial_parameter_sha256": provenance["initial_parameter_sha256"],
        "normalization_identity": provenance.get("normalization_identity"),
        "warm_normalization": warm,
        "prior_sha256": json_sha256(prior),
        "prior_source_sha256": prior_sources,
        "prior_members_sha256": prior["fit_members_sha256"],
        "cache_metadata_sha256": cache["metadata_sha256"],
        "cache_verified_files": cache["verified_files"],
        "teacher_checkpoint_refs": refs,
        "trainer_source_identity_sha256": runtime["sha256"],
        "trainer_source_files": len(files),
        "fit_membership": verify_membership(run, provenance, split, cell),
    }, summary


def checkpoint_diagnostic(
    path: Path, run: str, arm: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Check stored CPU tensors for width, dormant rows and zero-history control."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(
        checkpoint.get("epoch") == 1
        and checkpoint.get("n_in") == CHILD_WIDTH
        and checkpoint.get("args") == provenance.get("args"),
        f"{run}: checkpoint epoch, width or arguments differ",
    )
    embedded = checkpoint.get("provenance", {})
    for key in (
        "initial_checkpoint_sha256",
        "initial_parameter_sha256",
        "normalization_identity",
        "i1_warm_normalization",
        "causal_feature_prior",
        "update_diagnostics",
        "distillation_cache",
    ):
        require(embedded.get(key) == provenance.get(key), f"{run}: checkpoint does not bind {key}")
    state = checkpoint.get("model", {})
    mean, std, inp = state.get("mean"), state.get("std"), state.get("inp.weight")
    weight, bias = state.get("head.2.weight"), state.get("head.2.bias")
    require(
        all(isinstance(value, torch.Tensor) for value in (mean, std, inp, weight, bias))
        and tuple(mean.shape) == (CHILD_WIDTH,)
        and tuple(std.shape) == (CHILD_WIDTH,)
        and tuple(inp.shape)[1] == CHILD_WIDTH
        and tuple(weight.shape)[0] == 25
        and tuple(bias.shape) == (25,),
        f"{run}: checkpoint topology differs",
    )
    dormant = (
        torch.count_nonzero(weight[-2:]).item() == 0 and torch.count_nonzero(bias[-2:]).item() == 0
    )
    require(dormant, f"{run}: trailing compatibility output rows are not dormant")
    suffix = inp[:, PARENT_WIDTH:CHILD_WIDTH]
    is_zero = torch.count_nonzero(suffix).item() == 0
    norm = float(torch.linalg.vector_norm(suffix).item())
    if arm == "control":
        require(is_zero, f"{run}: control appended input projection is not zero")
    return {
        "input_width": CHILD_WIDTH,
        "output_rows": 25,
        "dormant_output_rows": 2,
        "dormant_rows_exact_zero": True,
        "control_appended_projection_exact_zero": bool(is_zero) if arm == "control" else None,
        "history_appended_projection_l2": norm if arm == "history" else None,
        "history_appended_projection_nonzero": bool(norm > 0) if arm == "history" else None,
    }


def verify_prediction(
    run_spec: dict[str, Any],
    split: dict[str, Any],
    split_sha: str,
    checkpoint_sha: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Check complete export identity/coverage before any target labels are opened."""
    run = run_spec["run_id"]
    prediction = ROOT / run_spec["prediction"]
    sidecar_path = ROOT / run_spec["prediction_sidecar"]
    sidecar = load_json(sidecar_path)
    prediction_sha = sha256(prediction)
    full_groups = [int(item["row_group"]) for item in split["sequences"]["valid"]]
    require(
        len(full_groups) == FULL_VALID and len(set(full_groups)) == FULL_VALID,
        f"{run}: full-valid split row order is incomplete",
    )
    full_sha = groups_sha256(full_groups)
    require(
        sidecar.get("run") == run
        and sidecar.get("split") == "valid"
        and sidecar.get("epoch") == 1
        and sidecar.get("checkpoint_sha256") == checkpoint_sha
        and sidecar.get("prediction_sha256") == prediction_sha
        and sidecar.get("split_manifest_sha256") == split_sha
        and sidecar.get("campaign_split_manifest_sha256") == split_sha
        and sidecar.get("campaign_export_scope") == "full-valid-feature-only"
        and sidecar.get("feature_only") is True
        and sidecar.get("inference_precision") == "fp32"
        and sidecar.get("row_groups") == full_groups
        and sidecar.get("evaluation_membership_sha256") == full_sha,
        f"{run}: sidecar does not bind the exact full-valid feature-only export",
    )
    header = _npy_header(prediction)
    full = np.load(prediction, mmap_mode="r", allow_pickle=False)
    require(
        full.shape == (FULL_VALID, ROWS, 2)
        and full.dtype == np.float32
        and full.flags.c_contiguous,
        f"{run}: NPY shape/dtype/C-order differs",
    )
    evaluation = list(map(int, split["evaluation"]["valid"]))
    mapped = aligned({"prediction": run_spec["prediction"]}, evaluation, split_sha)
    require(not isinstance(mapped, str), f"{run}: evaluation alignment failed: {mapped}")
    aligned_array, order = mapped
    expected_order = [full_groups.index(group) for group in evaluation]
    require(order == expected_order, f"{run}: evaluation row order differs")
    selected = np.asarray(aligned_array[np.asarray(order, dtype=np.int64)], dtype=np.float32)
    require(selected.shape == (len(evaluation), ROWS, 2), f"{run}: evaluation export shape differs")
    identity = {
        "run": run,
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": prediction_sha,
        "sidecar_sha256": sha256(sidecar_path),
        "prediction_shape": [FULL_VALID, ROWS, 2],
        "prediction_dtype": "float32",
        "npy_coverage": header,
        "full_valid_row_groups_sha256": full_sha,
        "evaluation_groups": len(evaluation),
    }
    del full, aligned_array
    return selected, identity


def verify_pair(fold: int, runs: dict[str, dict[str, Any]]) -> None:
    """Require matched initialization, fit normalization, prior/cache and order."""
    control, history = runs["control"], runs["history"]
    for key in (
        "initial_parameter_sha256",
        "normalization_identity",
        "trainer_source_identity_sha256",
    ):
        require(control[key] == history[key], f"F{fold}: paired {key} differs")
    for key in (
        "prior_sha256",
        "prior_source_sha256",
        "prior_members_sha256",
        "cache_metadata_sha256",
        "cache_verified_files",
        "teacher_checkpoint_refs",
    ):
        require(control[key] == history[key], f"F{fold}: paired {key} differs")
    for key in ("appended_mean_identity", "appended_std_identity"):
        require(
            control["warm_normalization"].get(key) == history["warm_normalization"].get(key),
            f"F{fold}: appended normalization {key} differs",
        )
    for control_epoch, history_epoch in zip(control["_updates"], history["_updates"], strict=True):
        require(
            control_epoch["sample_order_identity"] == history_epoch["sample_order_identity"],
            f"F{fold}: paired sampler/RNG identity differs",
        )


def target_bootstrap(
    targets: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
    index: int,
) -> dict[str, Any]:
    """Reuse the canonical bootstrap for a separate target via a duplicated pair."""
    y = np.repeat(targets[..., index : index + 1], 2, axis=-1)
    a = np.repeat(candidate[..., index : index + 1], 2, axis=-1)
    b = np.repeat(reference[..., index : index + 1], 2, axis=-1)
    moments_a, moments_b = paired_component_moments(y, a, b, mask)
    return paired_component_bootstrap(
        moments_a, moments_b, components, n_boot=N_BOOT, seed=BOOTSTRAP_SEED, centered=True
    )


def grade_fold(
    fold: int,
    split: dict[str, Any],
    cell: dict[str, Any],
    predictions: dict[str, np.ndarray],
    identities: dict[str, dict[str, Any]],
    targets_by_group: dict[int, int],
    target_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Score both fixed contrasts and all declared position bins on one fold."""
    groups = list(map(int, split["evaluation"]["valid"]))
    component_of = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    components = np.asarray([component_of[group] for group in groups], dtype=np.int64)
    require(
        len(groups) == cell["split"]["evaluation_groups"]
        and len(np.unique(components)) == cell["split"]["evaluation_components"],
        f"F{fold}: evaluation group/component count differs",
    )
    indices = [targets_by_group[group] for group in groups]
    y = np.asarray(target_cache["y"][indices], dtype=np.float32)
    mask = np.asarray(target_cache["mask"][indices], dtype=bool)
    require(
        y.shape == (len(groups), ROWS, 2) and mask.shape == (len(groups), ROWS),
        f"F{fold}: official target/mask shape differs",
    )
    mask &= np.arange(ROWS)[None, :] >= 99
    control, history = predictions["control"], predictions["history"]
    mixture = _mixture(control, history)
    candidates = {"control": control, "history": history, "mix_90_control_10_history": mixture}
    scores: dict[str, Any] = {}
    for name, values in candidates.items():
        scores[name] = {"overall": score(y, mask, values)}
        for start, stop in TIME_BINS:
            selected = mask.copy()
            selected[:, :start] = False
            selected[:, stop:] = False
            scores[name][f"rows_{start}_{stop}"] = score(y, selected, values)

    comparisons: dict[str, Any] = {}
    for label, candidate_name in (
        ("history_minus_control", "history"),
        ("mix_90_control_10_history_minus_control", "mix_90_control_10_history"),
    ):
        candidate = candidates[candidate_name]
        ma, mb = paired_component_moments(y, candidate, control, mask)
        wp_boot = paired_component_bootstrap(
            ma, mb, components, n_boot=N_BOOT, seed=BOOTSTRAP_SEED, centered=True
        )
        by_target = {
            f"t{index}": {
                "estimand": f"target {index} weighted Pearson",
                **target_bootstrap(y, candidate, control, mask, components, index),
            }
            for index in range(2)
        }
        comparisons[label] = {
            "wp_delta": scores[candidate_name]["overall"]["wp"]
            - scores["control"]["overall"]["wp"],
            "t0_delta": scores[candidate_name]["overall"]["t0"]
            - scores["control"]["overall"]["t0"],
            "t1_delta": scores[candidate_name]["overall"]["t1"]
            - scores["control"]["overall"]["t1"],
            "paired_component_bootstrap_wp": wp_boot,
            "paired_component_bootstrap_by_target": by_target,
            "position_bin_deltas": {
                f"rows_{start}_{stop}": {
                    target: scores[candidate_name][f"rows_{start}_{stop}"][target]
                    - scores["control"][f"rows_{start}_{stop}"][target]
                    for target in ("wp", "t0", "t1")
                }
                for start, stop in TIME_BINS
            },
        }
    return {
        "fold": fold,
        "split_manifest_sha256": cell["split"]["sha256"],
        "evaluation_groups": len(groups),
        "evaluation_components": int(len(np.unique(components))),
        "scored_rows": int(mask.sum()),
        "runs": identities,
        "scores": scores,
        "comparisons": comparisons,
        "exposure_and_limits": (
            "Previously inspected development fold. Each within-fold contrast is paired by seed and parent. "
            "F0/F1 change seed, parent, target cache, teacher ancestry and propensity together; "
            "they do not isolate fold from seed or establish training-seed uncertainty. No public-score inference."
        ),
    }


def main() -> int:
    """Write one create-only grade after all artifact and coverage gates pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    raw_plan = PLAN_PATH.read_bytes()
    plan = json.loads(raw_plan)
    plan_sha = hashlib.sha256(raw_plan).hexdigest()
    pins = verify_sources(plan)
    verify_paired_plan(plan)

    splits: dict[int, dict[str, Any]] = {}
    cell_receipts: dict[int, dict[str, Any]] = {}
    for cell in plan["cells"]:
        fold = int(cell["fold"])
        split = load_json(ROOT / cell["split"]["path"])
        require(int(split["fold"]) == fold, f"F{fold}: split fold differs")
        splits[fold] = split
        cell_receipts[fold] = verify_cell_inputs(cell, split)

    predictions: dict[int, dict[str, np.ndarray]] = {}
    identities: dict[int, dict[str, dict[str, Any]]] = {}
    for cell in plan["cells"]:
        fold = int(cell["fold"])
        split, split_sha = splits[fold], cell["split"]["sha256"]
        predictions[fold], identities[fold] = {}, {}
        for spec in plan["runs"]:
            if int(spec["fold"]) != fold:
                continue
            run = spec["run_id"]
            provenance_path = ROOT / "artifacts/runs" / run / "provenance.json"
            provenance = load_json(provenance_path)
            receipt, updates = verify_training_receipt(spec, provenance, plan, cell, split, pins)
            checkpoint = ROOT / spec["checkpoint"]
            checkpoint_sha = sha256(checkpoint)
            require(
                provenance.get("checkpoint_sha256", {}).get(checkpoint.name) == checkpoint_sha,
                f"{run}: checkpoint hash absent from provenance",
            )
            checkpoint_info = checkpoint_diagnostic(checkpoint, run, spec["arm"], provenance)
            selected, pred_info = verify_prediction(spec, split, split_sha, checkpoint_sha)
            receipt.update(pred_info)
            receipt["checkpoint_diagnostic"] = checkpoint_info
            receipt["provenance_sha256"] = sha256(provenance_path)
            receipt["update_diagnostics"] = updates
            receipt["_updates"] = updates
            predictions[fold][spec["arm"]] = selected
            identities[fold][spec["arm"]] = receipt
        require(set(identities[fold]) == {"control", "history"}, f"F{fold}: incomplete pair")
        verify_pair(fold, identities[fold])

    # No target labels are opened until all four full-valid exports pass the gates above.
    target_path = ROOT / "cache/valid_targets.npz"
    require(sha256(target_path) == TARGET_SHA256, "official valid-target cache hash differs")
    expected_target_orders = {
        int(cell["fold"]): [
            int(row["row_group"]) for row in splits[int(cell["fold"])]["sequences"]["valid"]
        ]
        for cell in plan["cells"]
    }
    with np.load(target_path, allow_pickle=False) as archive:
        row_groups = list(map(int, archive["row_group"]))
        require(
            all(row_groups == expected for expected in expected_target_orders.values()),
            "official target row-group order differs from a fold's canonical full-valid order",
        )
        target_cache = {
            "row_group": np.asarray(row_groups, dtype=np.int64),
            "y": archive["y"],
            "mask": archive["mask"],
        }
    require(
        target_cache["y"].shape == (FULL_VALID, ROWS, 2)
        and target_cache["mask"].shape == (FULL_VALID, ROWS),
        "official target cache does not contain all 1,873 valid groups",
    )
    folds: dict[str, Any] = {}
    for cell in plan["cells"]:
        fold = int(cell["fold"])
        split = splits[fold]
        target_order = list(map(int, target_cache["row_group"]))
        expected_order = [int(row["row_group"]) for row in split["sequences"]["valid"]]
        require(target_order == expected_order, f"F{fold}: target row-group order differs")
        positions = {group: index for index, group in enumerate(target_order)}
        require(len(positions) == FULL_VALID, f"F{fold}: duplicate target row groups")
        folds[str(fold)] = {
            "ancestry_and_inputs": cell_receipts[fold],
            **grade_fold(
                fold, split, cell, predictions[fold], identities[fold], positions, target_cache
            ),
        }
    require(sha256(target_path) == TARGET_SHA256, "official target cache changed during grade")
    for runs in identities.values():
        for run in runs.values():
            run.pop("_updates", None)
    report = {
        "schema": "wnn.i1-strong-parent-warm-history-grade.v1",
        "experiment_id": plan["experiment_id"],
        "plan_status": plan["status"],
        "plan_sha256": plan_sha,
        "grader_sha256": sha256(Path(__file__)),
        "metric_id": "global_wp_clip2_v1",
        "fit_counts": {"fits_per_arm": 2, "total_new_fits": 4, "teacher_fits": 0},
        "fixed_mixture": {
            "control": 0.9,
            "history": 0.1,
            "operation": "float64 blend, cast to float32, then official clipping",
        },
        "bootstrap": {
            "unit": "paired dependency component within fold",
            "replicates": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "centered_moments": True,
            "per_target_method": "reuse canonical helper with selected target duplicated in both slots",
        },
        "source_sha256_verified": pins,
        "scorer_source_sha256": {
            "grade_prediction_bank": sha256(ROOT / "scripts/grade_prediction_bank.py"),
            "gmetric": sha256(ROOT / "src/wnn/gmetric.py"),
            "evaluation": sha256(ROOT / "src/wnn/evaluation.py"),
            "reused_i1_grade_helpers": sha256(
                ROOT / "research/campaign_2026-10-21/next_cycle_r1/i1/grade_i1_panel.py"
            ),
            "valid_targets": TARGET_SHA256,
        },
        "folds": folds,
        "interpretation_limit": (
            "Both folds are exposed. Fold-specific paired contrasts are primary; their different seeds, parents, "
            "caches, teacher ancestry and propensity models preclude a clean replication claim. No public-score "
            "or training-seed uncertainty is inferred, and no mixture coefficients are selected here."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "plan_sha256": plan_sha,
                "folds": {key: value["comparisons"] for key, value in folds.items()},
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
