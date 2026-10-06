"""Grade a fixed paired continuation against its same-seed warm parents.

The evaluator uses only frozen fold-0 validation groups. Predictions are
memmapped and scored in small sequence batches; it never searches weights or
checkpoint epochs. Its component interval is conditional on the supplied fixed
models, not training seeds or public transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_distill_cache import _audit_ancestor_closure  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, WARMUP  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

SEEDS = (3, 4)
EVALUATION_GROUPS = 468
SEQUENCE_ROWS = 20_000
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 271_828
SEQUENCES_PER_BATCH = 4
RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_parents(source_manifest: Mapping[str, object]) -> dict[int, dict[str, object]]:
    families = source_manifest.get("families")
    if not isinstance(families, list):
        raise ValueError("source-family manifest lacks a families list")
    parents: dict[int, dict[str, object]] = {}
    for item in families:
        if not isinstance(item, dict) or item.get("family") != "strong":
            continue
        seed = item.get("seed")
        if seed not in SEEDS:
            continue
        if item.get("status") != "existing_audited":
            raise ValueError(f"seed {seed}: warm parent is not registered as audited")
        if not item.get("checkpoint_sha256") or not item.get("provenance_sha256"):
            raise ValueError(f"seed {seed}: warm parent lacks frozen hashes")
        parents[int(seed)] = item
    if set(parents) != set(SEEDS):
        raise ValueError("source-family manifest must identify one strong parent per seed")
    return parents


def _expected_c0_runs(preregistration: Mapping[str, object]) -> set[str]:
    return set(_expected_c0_commands(preregistration))


def _expected_c0_commands(preregistration: Mapping[str, object]) -> dict[str, list[str]]:
    commands = preregistration.get("commands")
    if not isinstance(commands, list):
        raise ValueError("E-502 preregistration lacks command records")
    by_name: dict[str, list[str]] = {}
    for command in commands:
        if not isinstance(command, list) or "--name" not in command:
            raise ValueError("E-502 command is malformed")
        index = command.index("--name")
        if index + 1 >= len(command):
            raise ValueError("E-502 command has no run name")
        name = str(command[index + 1])
        if name in by_name:
            raise ValueError(f"E-502 repeats a command for {name}")
        by_name[name] = list(map(str, command))
    if len(by_name) != len(SEEDS):
        raise ValueError("E-502 must preregister exactly two C0 run names")
    return by_name


def _verify_command_arguments(args: Mapping[str, object], command: list[str], run: str) -> None:
    """Check every explicit E-502 option against the saved checkpoint args."""
    if len(command) < 3 or command[1] != "scripts/train_rnn.py":
        raise ValueError("E-502 command does not invoke the registered trainer")
    expected: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        option = command[index]
        if not option.startswith("--"):
            raise ValueError(f"{run}: malformed E-502 option near {option!r}")
        key = option[2:].replace("-", "_")
        if key in expected:
            raise ValueError(f"{run}: E-502 repeats --{key.replace('_', '-')}")
        index += 1
        if index < len(command) and not command[index].startswith("--"):
            expected[key] = command[index]
            index += 1
        else:
            expected[key] = True

    # The checkpoint binds the actual manifest contents. The preregistered path
    # may be relative while argparse records its absolute equivalent.
    expected.pop("split_manifest", None)
    for key, expected_value in expected.items():
        observed = args.get(key)
        if isinstance(expected_value, bool):
            matches = observed is expected_value
        elif isinstance(observed, bool) or observed is None:
            matches = False
        elif isinstance(observed, int):
            try:
                matches = int(expected_value) == observed
            except ValueError:
                matches = False
        elif isinstance(observed, float):
            try:
                matches = float(expected_value) == observed
            except ValueError:
                matches = False
        else:
            matches = str(observed) == expected_value
        if not matches:
            raise ValueError(f"{run}: checkpoint argument {key} differs from E-502")


def _verify_common_checkpoint(
    checkpoint: Mapping[str, object],
    run: str,
    seed: int,
    epoch: int,
    split_sha256: str,
    expected_init: str | None,
) -> Mapping[str, object]:
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{run}: checkpoint has no argument record")
    if args.get("name") != run or args.get("seed") != seed:
        raise ValueError(f"{run}: checkpoint name or seed differs")
    if (
        checkpoint.get("epoch") != epoch
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epoch
        or args.get("no_eval_during_fit") is not True
    ):
        raise ValueError(f"{run}: checkpoint was not fixed at the declared no-eval epoch")
    if args.get("split_manifest_sha256") != split_sha256 or args.get("valid_fold") != 0:
        raise ValueError(f"{run}: checkpoint belongs to another fold")
    if expected_init is not None and args.get("init") != expected_init:
        raise ValueError(f"{run}: continuation does not use its matching warm parent")
    return args


def _verify_provenance_arguments(
    args: Mapping[str, object],
    embedded_provenance: Mapping[str, object],
    run_provenance: Mapping[str, object],
    run: str,
) -> None:
    """Ensure the recursive auditor sees the same arguments as the checkpoint."""
    if embedded_provenance.get("args") != args or run_provenance.get("args") != args:
        raise ValueError(f"{run}: checkpoint and run provenance arguments disagree")


def _verify_lineage_arguments(runs_root: Path, lineage: list[dict[str, object]]) -> None:
    """Bind every recursively audited ancestor edge to its checkpoint bytes."""
    for ancestor in lineage:
        run, checkpoint_file = ancestor.get("run"), ancestor.get("checkpoint_file")
        if not isinstance(run, str) or not RUN_NAME.fullmatch(run):
            raise ValueError("lineage auditor returned an unsafe run name")
        if not isinstance(checkpoint_file, str) or not re.fullmatch(
            r"model(?:_ep[0-9]+)?\.pt", checkpoint_file
        ):
            raise ValueError(f"{run}: lineage auditor returned an unsafe checkpoint name")
        checkpoint = torch.load(
            runs_root / run / checkpoint_file, map_location="cpu", weights_only=False
        )
        args = checkpoint.get("args")
        embedded = checkpoint.get("provenance")
        if not isinstance(args, dict) or not isinstance(embedded, dict):
            raise ValueError(f"{run}: ancestor checkpoint lacks provenance arguments")
        provenance = _read_json(runs_root / run / "provenance.json")
        _verify_provenance_arguments(args, embedded, provenance, run)
        del checkpoint


def _validate_prediction_artifact(
    prediction_path: Path,
    metadata: Mapping[str, object],
    *,
    run: str,
    epoch: int,
    split_sha256: str,
    checkpoint_sha256: str,
    row_groups: list[int],
    shape: tuple[int, int, int],
) -> tuple[np.memmap, dict[str, object]]:
    """Bind a finite float32 prediction memmap to its run, checkpoint and rows."""
    if (
        metadata.get("run") != run
        or metadata.get("split") != "valid"
        or metadata.get("epoch") != epoch
        or metadata.get("split_manifest_sha256") != split_sha256
        or metadata.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"{run}: prediction sidecar disagrees with the fixed checkpoint")
    if metadata.get("row_groups") != row_groups:
        raise ValueError(f"{run}: prediction row order differs from valid_targets")
    prediction_sha256 = file_sha256(prediction_path)
    if metadata.get("prediction_sha256") != prediction_sha256:
        raise ValueError(f"{run}: prediction bytes differ from sidecar")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != shape or prediction.dtype != np.float32:
        raise ValueError(f"{run}: prediction array is incomplete or not float32")
    for start in range(0, len(prediction), SEQUENCES_PER_BATCH):
        if not np.isfinite(prediction[start : start + SEQUENCES_PER_BATCH]).all():
            raise ValueError(f"{run}: nonfinite prediction values")
    binding = {
        "run": run,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": prediction_sha256,
        "split_manifest_sha256": split_sha256,
        "row_groups_sha256": _list_sha256(row_groups),
        "inference_precision": metadata.get("inference_precision"),
    }
    return prediction, binding


def _list_sha256(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _official_mask(is_scored: np.ndarray) -> np.ndarray:
    scored = np.asarray(is_scored, dtype=bool)
    if scored.ndim != 2 or scored.shape[1] != SEQUENCE_ROWS:
        raise ValueError("expected is_scored rows for complete 20,000-step sequences")
    need_prediction = np.arange(SEQUENCE_ROWS) >= WARMUP
    return scored & need_prediction[None, :]


def _validate_target_index(
    row_groups: list[int],
    targets: np.ndarray,
    is_scored: np.ndarray,
    market_class: np.ndarray,
    valid_groups: tuple[int, ...],
) -> None:
    """Require the score cache to cover exactly the frozen validation index."""
    if (
        len(set(row_groups)) != len(row_groups)
        or set(row_groups) != set(map(int, valid_groups))
        or targets.shape != (len(row_groups), SEQUENCE_ROWS, 2)
        or targets.dtype != np.float32
        or is_scored.shape != (len(row_groups), SEQUENCE_ROWS)
        or market_class.shape != (len(row_groups),)
    ):
        raise ValueError("valid target cache is malformed or differs from frozen valid groups")


def _blend_equal_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Blend raw members first; the official scorer clips the resulting prediction."""
    left, right = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("equal-pair members must have identical prediction shapes")
    return ((left.astype(np.float64) + right.astype(np.float64)) * 0.5).astype(np.float32)


def compare_arrays(
    targets: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    is_scored: np.ndarray,
    components: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Score one fixed paired contrast with the official mask and component bootstrap."""
    y = np.asarray(targets, dtype=np.float32)
    mask = _official_mask(is_scored)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    reference_array = np.asarray(reference, dtype=np.float32)
    if y.shape != candidate_array.shape or y.shape != reference_array.shape:
        raise ValueError("target and prediction arrays must align")
    if y.shape[:2] != mask.shape or components.shape != (len(y),):
        raise ValueError("mask or component identities do not align with sequences")
    candidate_moments, reference_moments = paired_component_moments(
        y, candidate_array, reference_array, mask
    )
    result = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        components,
        n_boot=n_boot,
        seed=seed,
        centered=True,
    )
    return dict(result) | {"selected_rows": int(mask.sum())}


def _member(
    artifact_root: Path,
    run: str,
    seed: int,
    epoch: int,
    split,
    row_groups: list[int],
    *,
    expected_init: str | None,
    source_record: Mapping[str, object] | None = None,
    expected_command: list[str] | None = None,
) -> tuple[np.memmap, dict[str, object]]:
    if not RUN_NAME.fullmatch(run):
        raise ValueError(f"unsafe run name: {run!r}")
    runs_root = artifact_root / "artifacts" / "runs"
    run_root = runs_root / run
    checkpoint_path = run_root / f"model_ep{epoch}.pt"
    provenance_path = run_root / "provenance.json"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    provenance_sha256 = file_sha256(provenance_path)
    if source_record is not None and (
        checkpoint_sha256 != source_record.get("checkpoint_sha256")
        or provenance_sha256 != source_record.get("provenance_sha256")
    ):
        raise ValueError(f"{run}: checkpoint or provenance differs from SOURCE_FAMILIES")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_parent_init = expected_init
    args = _verify_common_checkpoint(
        checkpoint, run, seed, epoch, split.sha256, expected_parent_init
    )
    if expected_command is not None:
        _verify_command_arguments(args, expected_command, run)
    if expected_init is not None and (
        args.get("epochs") != 2
        or args.get("batch") != 16
        or args.get("chunk") != 2000
        or args.get("loss") != "wmse"
        or args.get("lr") != 0.0001
    ):
        raise ValueError(f"{run}: candidate differs from the fixed paired continuation budget")
    embedded_provenance = checkpoint.get("provenance")
    if not isinstance(embedded_provenance, dict):
        raise ValueError(f"{run}: checkpoint lacks embedded provenance")
    if not isinstance(embedded_provenance.get("args"), dict):
        raise ValueError(f"{run}: checkpoint embedded provenance lacks arguments")
    del checkpoint

    provenance = _read_json(provenance_path)
    _verify_provenance_arguments(args, embedded_provenance, provenance, run)

    lineage = _audit_ancestor_closure(
        run,
        split,
        runs_root,
        f"model_ep{epoch}.pt",
        checkpoint_sha256,
    )
    _verify_lineage_arguments(runs_root, lineage)
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError(f"{run}: provenance lacks fit-row membership")
    train_fit = set(map(int, fit.get("train", [])))
    valid_fit = set(map(int, fit.get("valid", [])))
    if valid_fit & set(split.evaluation_groups):
        raise ValueError(f"{run}: evaluation groups occur in direct fit membership")
    split.assert_fit_groups(train_fit, valid_fit)

    prediction_path = artifact_root / "cache" / f"pred_{run}_valid.npy"
    sidecar = _read_json(prediction_path.with_suffix(".json"))
    prediction, binding = _validate_prediction_artifact(
        prediction_path,
        sidecar,
        run=run,
        epoch=epoch,
        split_sha256=split.sha256,
        checkpoint_sha256=checkpoint_sha256,
        row_groups=row_groups,
        shape=(len(row_groups), SEQUENCE_ROWS, 2),
    )
    binding.update(
        seed=seed,
        provenance_sha256=provenance_sha256,
        lineage=lineage,
    )
    return prediction, binding


def _component_provenance(
    groups: list[int],
    components: np.ndarray,
    classes: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    for index, group in enumerate(groups):
        component = int(components[index])
        entry = grouped.setdefault(
            component,
            {"component": component, "row_groups": [], "class_counts": {}, "selected_rows": 0},
        )
        entry["row_groups"].append(int(group))
        label = str(int(classes[index]))
        class_counts = entry["class_counts"]
        class_counts[label] = int(class_counts.get(label, 0)) + 1
        entry["selected_rows"] += int(mask[index].sum())
    return [grouped[key] for key in sorted(grouped)]


def _markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        f"# {report['candidate_id']} paired continuation evaluation",
        "",
        f"Metric: `{METRIC_ID}`. Fold 0 is exposed development data, not a fresh holdout.",
        "The 95% intervals resample frozen evaluation components conditional on the fixed checkpoints;",
        "they do not include training-seed variation or establish public transfer.",
        "The comparison is continuation versus matching warm parents, not the public-best blend.",
        f"Candidate manifest validation: {report['candidate_manifest_validation']}.",
        "",
        "| Comparison | Candidate WP | Parent WP | Difference | 95% component interval |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    comparisons = report["comparisons"]
    for name, result in comparisons.items():
        lo, hi = result["ci95"]
        lines.append(
            f"| {name} | {result['candidate_wp']:.8f} | {result['reference_wp']:.8f} | "
            f"{result['mean_diff']:+.8f} | [{lo:+.8f}, {hi:+.8f}] |"
        )
    lines.extend(
        [
            "",
            f"Evaluation sequences: {report['n_sequences']}; components: {report['n_components']}; "
            f"selected rows: {report['selected_rows']}.",
            "The equal seed pair uses fixed 0.5/0.5 weights; target and prediction clipping occurs "
            "inside the official metric after that blend.",
            "",
        ]
    )
    return "\n".join(lines)


def grade(
    artifact_root: Path,
    candidate_prefix: str,
    candidate_id: str,
    output_dir: Path,
    *,
    repository_root: Path = ROOT,
    split_manifest: Path | None = None,
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Validate fixed members, score all fold-0 evaluation groups, and save reports."""
    if not RUN_NAME.fullmatch(candidate_prefix):
        raise ValueError("candidate prefix contains unsafe characters")
    if candidate_id not in {"C0", "C1", "C2", "C3", "C4"}:
        raise ValueError("candidate ID must be one of the preregistered C0-C4 recipes")
    artifact_root = artifact_root.resolve(strict=True)
    repository_root = repository_root.resolve(strict=True)
    prereg_path = (
        repository_root
        / "research/campaign_2026-10-06/bandit_continuation/local/E502_C0_PAIRED_v2.json"
    )
    source_path = (
        repository_root / "research/campaign_2026-10-06/bandit_continuation/SOURCE_FAMILIES.json"
    )
    preregistration = _read_json(prereg_path)
    source_manifest = _read_json(source_path)
    if (
        preregistration.get("id") != "E-502"
        or preregistration.get("status") != "preregistered"
        or preregistration.get("fold") != 0
        or preregistration.get("seeds") != list(SEEDS)
    ):
        raise ValueError("E-502 preregistration differs from this frozen grader")
    c0_commands = _expected_c0_commands(preregistration)
    if candidate_id == "C0" and {f"{candidate_prefix}_s{seed}_f0" for seed in SEEDS} != set(
        c0_commands
    ):
        raise ValueError("C0 candidate prefix differs from the preregistered run names")

    split_path = split_manifest or (
        artifact_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    )
    split = load_campaign_split(split_path, artifact_root)
    if split.fold != 0 or len(split.evaluation_groups) != EVALUATION_GROUPS:
        raise ValueError("expected the complete, frozen 468-group fold-0 evaluation panel")

    target_path = artifact_root / "cache" / "valid_targets.npz"
    target_sha256 = file_sha256(target_path)
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        targets = cache["y"]
        is_scored = cache["mask"]
        classes_all = cache["market_class"]
    _validate_target_index(row_groups, targets, is_scored, classes_all, split.valid_groups)
    position = {group: index for index, group in enumerate(row_groups)}
    groups = list(map(int, split.evaluation_groups))
    if len(set(groups)) != EVALUATION_GROUPS or any(group not in position for group in groups):
        raise ValueError("frozen evaluation groups are missing or duplicated in valid_targets")
    indices = np.asarray([position[group] for group in groups], dtype=np.int64)
    group_to_component = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    components = np.asarray([group_to_component[group] for group in groups], dtype=np.int64)
    classes = np.asarray(classes_all[indices], dtype=np.int64)
    selected_mask = _official_mask(np.asarray(is_scored[indices], dtype=bool))
    parents = _source_parents(source_manifest)

    candidate_predictions: dict[int, np.memmap] = {}
    parent_predictions: dict[int, np.memmap] = {}
    bindings: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        parent = parents[seed]
        parent_run = str(parent["run"])
        candidate_run = f"{candidate_prefix}_s{seed}_f0"
        parent_prediction, parent_binding = _member(
            artifact_root,
            parent_run,
            seed,
            int(parent["epoch_zero_based"]),
            split,
            row_groups,
            expected_init=None,
            source_record=parent,
        )
        candidate_prediction, candidate_binding = _member(
            artifact_root,
            candidate_run,
            seed,
            1,
            split,
            row_groups,
            expected_init=f"{parent_run}@ep{parent['epoch_zero_based']}",
            expected_command=c0_commands.get(candidate_run) if candidate_id == "C0" else None,
        )
        candidate_predictions[seed] = candidate_prediction
        parent_predictions[seed] = parent_prediction
        bindings[candidate_run] = candidate_binding
        bindings[parent_run] = parent_binding

    comparison_specs = {
        "seed3_candidate_vs_parent": ((candidate_predictions[3],), (parent_predictions[3],)),
        "seed4_candidate_vs_parent": ((candidate_predictions[4],), (parent_predictions[4],)),
        "equal_pair_candidate_vs_parent": (
            (candidate_predictions[3], candidate_predictions[4]),
            (parent_predictions[3], parent_predictions[4]),
        ),
    }
    moments: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (
            np.empty((EVALUATION_GROUPS, 2, 6), dtype=np.float64),
            np.empty((EVALUATION_GROUPS, 2, 6), dtype=np.float64),
        )
        for name in comparison_specs
    }
    for start in range(0, EVALUATION_GROUPS, SEQUENCES_PER_BATCH):
        stop = min(start + SEQUENCES_PER_BATCH, EVALUATION_GROUPS)
        take = indices[start:stop]
        y = np.asarray(targets[take], dtype=np.float32)
        mask = selected_mask[start:stop]
        for name, (candidate_members, parent_members) in comparison_specs.items():
            candidate_rows = (
                np.asarray(candidate_members[0][take], dtype=np.float32)
                if len(candidate_members) == 1
                else _blend_equal_pair(candidate_members[0][take], candidate_members[1][take])
            )
            parent_rows = (
                np.asarray(parent_members[0][take], dtype=np.float32)
                if len(parent_members) == 1
                else _blend_equal_pair(parent_members[0][take], parent_members[1][take])
            )
            candidate_moments, parent_moments = paired_component_moments(
                y, candidate_rows, parent_rows, mask
            )
            moments[name][0][start:stop] = candidate_moments
            moments[name][1][start:stop] = parent_moments

    comparisons: dict[str, dict[str, object]] = {}
    for name, (candidate_moments, parent_moments) in moments.items():
        result = paired_component_bootstrap(
            candidate_moments,
            parent_moments,
            components,
            n_boot=n_boot,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        comparisons[name] = dict(result) | {"selected_rows": int(selected_mask.sum())}

    component_provenance = _component_provenance(groups, components, classes, selected_mask)
    report: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_prefix": candidate_prefix,
        "candidate_manifest_validation": (
            "checkpoint CLI arguments matched to the frozen E-502 commands"
            if candidate_id == "C0"
            else "not performed by this E-502 grader; use the campaign recipe validator"
        ),
        "preregistration": {
            "id": preregistration["id"],
            "sha256": file_sha256(prereg_path),
            "commands": preregistration["commands"],
        },
        "source_families": {"sha256": file_sha256(source_path)},
        "split": {
            "path": str(split_path),
            "fold": split.fold,
            "sha256": split.sha256,
            "evaluation_groups": groups,
        },
        "targets": {
            "path": str(target_path),
            "sha256": target_sha256,
            "row_groups_sha256": _list_sha256(row_groups),
        },
        "metric_id": METRIC_ID,
        "selection": "need_prediction AND is_scored; need_prediction is step >= 99",
        "blend": "fixed 0.5/0.5 per-seed pair; blend raw predictions before metric clipping",
        "n_sequences": EVALUATION_GROUPS,
        "n_components": int(len(np.unique(components))),
        "selected_rows": int(selected_mask.sum()),
        "bootstrap": {
            "method": "paired frozen-component bootstrap over centered per-sequence moments",
            "replicates": n_boot,
            "seed": BOOTSTRAP_SEED,
            "interpretation": "conditional on the fixed checkpoints; excludes training-seed and public uncertainty",
        },
        "purpose": "matched continuation versus same-seed warm parents; not public-best-blend comparison",
        "public_forecast": None,
        "members": bindings,
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "components.json").write_text(
        json.dumps(
            {
                "fold": 0,
                "split_manifest_sha256": split.sha256,
                "selection": "need_prediction AND is_scored",
                "components": component_provenance,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "report.md").write_text(_markdown_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument("--candidate-id", choices=("C0", "C1", "C2", "C3", "C4"), default="C0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if args.bootstrap_replicates < 2:
        parser.error("--bootstrap-replicates must be at least two")
    report = grade(
        args.artifact_root,
        args.candidate_prefix,
        args.candidate_id,
        args.output_dir,
        repository_root=args.repository_root,
        split_manifest=args.split_manifest,
        n_boot=args.bootstrap_replicates,
    )
    print(json.dumps({name: value for name, value in report["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
