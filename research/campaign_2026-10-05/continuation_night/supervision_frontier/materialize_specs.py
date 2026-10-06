"""Write E-496 unsealed job templates; never starts scientific work."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any  # JSON specs are heterogeneous objects.

ROOT = Path(__file__).resolve().parents[4]
NIGHT = Path(__file__).resolve().parents[1]
FRONTIER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import continuation_recipes as recipes  # noqa: E402

BLOCKS = ((3, 0), (4, 0), (3, 1))
ALPHA_HORIZON = tuple(
    (alpha, horizon)
    for alpha in (0.50, 0.70, 0.85)
    for horizon in (0, 8)
)
SHUFFLE_SEED = 314159
DIRECTION_AUX = 0.1
REFERENCE_TOKEN = "${calibration.reference_mass}"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _set(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        raise ValueError(f"recipe is missing required option {flag}")
    argv[argv.index(flag) + 1] = value


def _append_or_set(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        _set(argv, flag, value)
    else:
        argv.extend((flag, value))


def _base_args(run: str, fold: int, seed: int) -> list[str]:
    argv = recipes.training_args(run, fold, seed) + ["--batch-order", "seeded"]
    _set(argv, "--distill-alpha", "0.7")
    _set(argv, "--distill-boot-h", "8")
    argv.extend(("--readout-activation", "gelu", "--distill-reanchor", "0"))
    return argv


def _spec(
    name: str,
    argv: list[str],
    inputs: list[str],
    artifacts: list[str],
    *,
    kind: str = "gpu",
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    source_paths = recipes.source_paths(ROOT)
    source_paths += [
        str((FRONTIER / name).relative_to(ROOT))
        for name in ("materialize_specs.py", "resolve_calibration.py")
    ]
    return {
        "version": 1,
        "name": name,
        "kind": kind,
        "argv": argv,
        "code": sorted(set(source_paths)),
        "inputs": sorted(
            set(
                inputs
                + [
                    str((NIGHT / relative).relative_to(ROOT))
                    for relative in (
                        "CONTRACT.md",
                        "activation_transfer/DESIGN.md",
                        "supervision_frontier/DESIGN.md",
                    )
                ]
            )
        ),
        "artifacts": artifacts,
        "limits": limits
        or {
            "cpus": 2,
            "ram_bytes": 18_000_000_000,
            "timeout_seconds": 14_400,
            "gpu_memory_bytes": 4_200_000_000,
            "gpu_stop_bytes": 4_800_000_000,
        },
        "environment": {
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "3",
            "WNN_PORTABLE_DATA_MANIFEST": recipes.CAMPAIGN + "/portable_inputs.json",
        },
    }


def _run_name(seed: int, fold: int, arm: str) -> str:
    return f"c20261005_e496_{arm}_s{seed}_f{fold}"


def _parent_artifacts(run: str) -> list[str]:
    return [
        f"artifacts/runs/{run}/{name}"
        for name in ("model_ep3.pt", "provenance.json", "history.json", "training_state.pt")
    ]


def _head_artifacts(run: str) -> list[str]:
    return [
        f"artifacts/runs/{run}/{name}"
        for name in ("model_ep0.pt", "provenance.json", "history.json", "training_state.pt")
    ]


def _prediction(run: str, epoch: int, fold: int) -> dict[str, Any]:
    return _spec(
        "predict_" + run,
        [
            "{python}",
            "scripts/predict_run.py",
            run,
            "valid",
            "--epoch",
            str(epoch),
            "--inference-precision",
            "fp32",
        ],
        [
            "datasets/valid.parquet",
            "cache/valid_targets.npz",
            f"{recipes.FOLDS}/fold{fold}.json",
            recipes.CAMPAIGN + "/portable_inputs.json",
            f"artifacts/runs/{run}/model_ep{epoch}.pt",
            f"artifacts/runs/{run}/provenance.json",
        ],
        [f"cache/pred_{run}_valid.{extension}" for extension in ("npy", "json")],
    )


def _arm_options(arm: str) -> tuple[list[str], str, bool]:
    """Return intervention args, denominator mode, and calibration-token status."""
    if arm.startswith("ah_"):
        _, alpha_text, horizon_text = arm.split("_")
        if not alpha_text.startswith("a") or not horizon_text.startswith("h"):
            raise ValueError(f"malformed alpha/horizon arm: {arm}")
        alpha = int(alpha_text[1:]) / 100
        return ["--distill-alpha", str(alpha), "--distill-boot-h", horizon_text[1:]], "chunk", False
    if arm == "direction_on":
        return ["--aux-sign", str(DIRECTION_AUX)], "chunk", False
    if arm == "den_fixed":
        return ["--distill-denominator", "fixed", "--distill-reference-mass", REFERENCE_TOKEN], "fixed", True
    if arm == "den_shuffled":
        return [
            "--distill-denominator",
            "shuffled",
            "--distill-reference-mass",
            REFERENCE_TOKEN,
            "--denominator-shuffle-seed",
            str(SHUFFLE_SEED),
        ], "shuffled", True
    if arm == "shared_alpha070_h8":
        return [], "chunk", False
    raise ValueError(f"unknown E-496 arm: {arm}")


def _screen_arms(
    family: str, seed: int, fold: int, endpoint: str
) -> tuple[dict[str, dict[str, float]], list[list[str]]]:
    shared = _run_name(seed, fold, "ah_a070_h8")
    if family == "alpha_bootstrap":
        names = [f"a{int(alpha * 100):03d}_h{horizon}" for alpha, horizon in ALPHA_HORIZON]
        runs = {
            name: _run_name(seed, fold, f"ah_a{int(alpha * 100):03d}_h{horizon}")
            for name, (alpha, horizon) in zip(names, ALPHA_HORIZON, strict=True)
        }
        comparisons = [
            [f"a{alpha:03d}_h8", f"a{alpha:03d}_h0"] for alpha in (50, 70, 85)
        ]
        comparisons += [
            [f"a{alpha:03d}_h{horizon}", f"a050_h{horizon}"]
            for horizon in (0, 8)
            for alpha in (70, 85)
        ]
        comparisons += [
            [f"a{alpha:03d}_h8", f"a{alpha:03d}_h0", "a050_h8", "a050_h0"]
            for alpha in (70, 85)
        ]
    elif family == "direction":
        runs = {"direction_off": shared, "direction_on_010": _run_name(seed, fold, "direction_on")}
        comparisons = [["direction_on_010", "direction_off"]]
    elif family == "denominator":
        runs = {
            "chunk": shared,
            "fixed": _run_name(seed, fold, "den_fixed"),
            "shuffled": _run_name(seed, fold, "den_shuffled"),
        }
        comparisons = [["fixed", "chunk"], ["shuffled", "fixed"], ["shuffled", "chunk"]]
    else:
        raise ValueError(f"unknown E-496 family: {family}")
    if endpoint == "complete":
        arms = {label: {run: 0.5, run + "_head": 0.5} for label, run in runs.items()}
    elif endpoint == "parent":
        arms = {label: {run: 1.0} for label, run in runs.items()}
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")
    return arms, comparisons


def _write_parent_and_head(
    seed: int, fold: int, arm: str, missing_inputs: list[dict[str, object]]
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    run = _run_name(seed, fold, arm)
    argv = _base_args(run, fold, seed)
    treatment, denominator, calibrated = _arm_options(arm)
    for index, value in enumerate(treatment):
        if value.startswith("--"):
            continue
        flag = treatment[index - 1]
        _append_or_set(argv, flag, value)
    parent_inputs = recipes.input_paths(fold)
    calibration_path = f"{FRONTIER.relative_to(ROOT)}/calibration/reference_s{seed}_f{fold}.json"
    if calibrated:
        parent_inputs.append(calibration_path)
    parent = _spec(run, argv, parent_inputs, _parent_artifacts(run))
    parent_path = FRONTIER / "specs" / f"{run}.{'template' if calibrated else 'draft'}.json"
    _write(parent_path, parent)
    missing_inputs.append(
        {
            "node": run,
            "spec": str(parent_path.relative_to(ROOT)),
            "inputs": [item for item in parent["inputs"] if not (ROOT / item).exists()],
            "unresolved": [calibration_path + "::reference_mass"] if calibrated else [],
        }
    )

    head = run + "_head"
    head_argv = recipes.head_args(head, run + "@ep3", fold) + ["--batch-order", "seeded"]
    head = _spec(
        head,
        head_argv,
        recipes.input_paths(fold, parent=run + "@ep3"),
        _head_artifacts(head),
    )
    head_path = FRONTIER / "specs" / f"{run}_head.draft.json"
    _write(head_path, head)
    missing_inputs.append(
        {
            "node": run + "_head",
            "spec": str(head_path.relative_to(ROOT)),
            "inputs": [item for item in head["inputs"] if not (ROOT / item).exists()],
            "unresolved": [f"artifacts/runs/{run}/model_ep3.pt"],
        }
    )
    return run, run + "_head", parent, head


def _write_calibration(seed: int, fold: int, missing_inputs: list[dict[str, object]]) -> str:
    path = f"{FRONTIER.relative_to(ROOT)}/calibration/reference_s{seed}_f{fold}.json"
    run = f"c20261005_e496_calibration_s{seed}_f{fold}"
    argv = _base_args(run, fold, seed)
    argv += ["--device", "cpu", "--calibration-epochs", "1", "--calibrate-distill-reference", path]
    argv = [
        token
        for token in argv
        if token not in {"--save-training-state", "--fast"}
    ]
    argv.append("--prop-cache")
    specification = _spec(
        run,
        argv,
        recipes.input_paths(fold),
        [path],
        kind="cpu",
        limits={
            "cpus": 1,
            "ram_bytes": 18_000_000_000,
            "timeout_seconds": 14_400,
            "gpu_memory_bytes": 0,
            "gpu_stop_bytes": 0,
        },
    )
    spec_path = FRONTIER / "specs" / f"{run}.draft.json"
    _write(spec_path, specification)
    missing_inputs.append(
        {
            "node": run,
            "spec": str(spec_path.relative_to(ROOT)),
            "inputs": [item for item in specification["inputs"] if not (ROOT / item).exists()],
            "unresolved": [path],
        }
    )
    return path


def _write_probe(seed: int, fold: int, arm: str, kind: str) -> str:
    run = f"c20261005_e496_probe_{arm}_s{seed}_f{fold}"
    argv = _base_args(run, fold, seed)
    for index, value in enumerate(_arm_options(arm)[0]):
        if not value.startswith("--"):
            _append_or_set(argv, _arm_options(arm)[0][index - 1], value)
    _set(argv, "--epochs", "1")
    _set(argv, "--checkpoint-epoch", "0")
    argv = [token for token in argv if token != "--save-training-state"]
    argv += ["--profile-steps", "2", "--profile-warmup", "1"]
    plan_path = FRONTIER / "probes" / f"{run}.plan.json"
    report = f"{FRONTIER.relative_to(ROOT)}/probes/{run}.json"
    plan = {"version": 1, "steps": [{"name": run, "argv": argv}], "report": report}
    _write(plan_path, plan)
    specification = _spec(
        run,
        ["{python}", "scripts/kaggle_sequence.py", "--plan", str(plan_path.relative_to(ROOT))],
        recipes.input_paths(fold) + [str(plan_path.relative_to(ROOT))],
        [report],
    )
    if _arm_options(arm)[2]:
        specification["inputs"].append(
            f"{FRONTIER.relative_to(ROOT)}/calibration/reference_s{seed}_f{fold}.json"
        )
        specification["inputs"].sort()
    specification["code"] = sorted(set(specification["code"] + ["scripts/kaggle_sequence.py", "scripts/kaggle_common.py"]))
    specification["limits"]["timeout_seconds"] = 900
    calibrated = _arm_options(arm)[2]
    spec_path = FRONTIER / "specs" / f"{run}.{'template' if calibrated else 'draft'}.json"
    _write(spec_path, specification)
    return str(spec_path.relative_to(ROOT))


def materialize() -> dict[str, Any]:
    """Materialize fixed, unsealed endpoint specs and dependency inventory."""
    missing_inputs: list[dict[str, object]] = []
    nodes: list[dict[str, Any]] = []
    for seed, fold in BLOCKS:
        calibration_path = _write_calibration(seed, fold, missing_inputs)
        calibration_node = f"calibration_s{seed}_f{fold}"
        nodes.append(
            {
                "id": calibration_node,
                "kind": "fit_only_cpu_calibration",
                "spec": str((FRONTIER / "specs" / f"c20261005_e496_calibration_s{seed}_f{fold}.draft.json").relative_to(ROOT)),
                "outputs": [calibration_path],
                "depends_on": [],
                "launch_authorized": False,
            }
        )
        arms = [f"ah_a{int(alpha * 100):03d}_h{horizon}" for alpha, horizon in ALPHA_HORIZON]
        arms += ["direction_on", "den_fixed", "den_shuffled"]
        # The alpha=.70/horizon=8/aux=0/chunk arm is the shared block control.
        arms = [arm for arm in arms if arm != "ah_a070_h8"]
        arms.insert(0, "ah_a070_h8")
        for arm in arms:
            run, head_run, parent, head = _write_parent_and_head(seed, fold, arm, missing_inputs)
            calibrated = _arm_options(arm)[2]
            parent_node = f"parent_s{seed}_f{fold}_{arm}"
            head_node = f"head_s{seed}_f{fold}_{arm}"
            nodes.append(
                {
                    "id": parent_node,
                    "kind": "parent_fit",
                    "arm": arm,
                    "spec": str((FRONTIER / "specs" / f"{run}.{'template' if calibrated else 'draft'}.json").relative_to(ROOT)),
                    "outputs": parent["artifacts"],
                    "depends_on": [calibration_node] if calibrated else [],
                    "unresolved": [calibration_path + "::reference_mass"] if calibrated else [],
                }
            )
            nodes.append(
                {
                    "id": head_node,
                    "kind": "head_fit",
                    "arm": arm,
                    "spec": str((FRONTIER / "specs" / f"{run}_head.draft.json").relative_to(ROOT)),
                    "outputs": head["artifacts"],
                    "depends_on": [parent_node],
                }
            )
            for phase, run_name, epoch, dependency in (
                ("parent_prediction", run, 3, parent_node),
                ("head_prediction", head_run, 0, head_node),
            ):
                pred = _prediction(run_name, epoch, fold)
                pred_path = FRONTIER / "specs" / f"{run_name}_{phase}.draft.json"
                _write(pred_path, pred)
                pred_node = f"{phase}_s{seed}_f{fold}_{arm}"
                nodes.append(
                    {
                        "id": pred_node,
                        "kind": "prediction",
                        "spec": str(pred_path.relative_to(ROOT)),
                        "outputs": pred["artifacts"],
                        "depends_on": [dependency],
                    }
                )

        if (seed, fold) == BLOCKS[0]:
            for arm, kind in (
                ("ah_a085_h8", "alpha_horizon_envelope"),
                ("direction_on", "direction_auxiliary"),
                ("den_fixed", "fixed_denominator"),
                ("den_shuffled", "shuffled_denominator"),
            ):
                spec_path = _write_probe(seed, fold, arm, kind)
                nodes.append(
                    {
                        "id": f"probe_{kind}_s{seed}_f{fold}",
                        "kind": "resource_probe_plan",
                        "spec": spec_path,
                        "outputs": [],
                        "depends_on": [calibration_node]
                        if _arm_options(arm)[2]
                        else [],
                        "unresolved": [calibration_path + "::reference_mass"]
                        if _arm_options(arm)[2]
                        else [],
                        "launch_authorized": False,
                    }
                )

        for family in ("alpha_bootstrap", "direction", "denominator"):
            for endpoint in ("parent", "complete"):
                arms_map, comparisons = _screen_arms(family, seed, fold, endpoint)
                endpoint_spec = {
                    "version": 1,
                    "experiment": "E-496",
                    "family": family,
                    "folds": [fold],
                    "seed": seed,
                    "endpoint": endpoint,
                    "n_boot": 2000,
                    "metric": "global_wp_clip2_v1",
                    "arms": arms_map,
                    "comparisons": [contrast for contrast in comparisons if len(contrast) == 2],
                    "interactions": [
                        {
                            "id": f"a{alpha:03d}_boot_interaction",
                            "cells": [
                                f"a{alpha:03d}_h8",
                                f"a{alpha:03d}_h0",
                                "a050_h8",
                                "a050_h0",
                            ],
                            "operation": "(cell0-cell1)-(cell2-cell3)",
                        }
                        for alpha in (70, 85)
                    ]
                    if family == "alpha_bootstrap"
                    else [],
                    "class_mix_sensitivity": {
                        "status": "secondary_inferred_scenario",
                        "pub143_scenario": [0.183, 0.160, 0.198, 0.459],
                    },
                    "report": ["both_targets", "class_by_event_age", "paired_component_bootstrap", "per_seed_fold"],
                    "selection": "none; every registered arm and contrast is reported",
                }
                name = f"e496_{family}_s{seed}_f{fold}_{endpoint}.json"
                endpoint_path = FRONTIER / "endpoints" / name
                _write(endpoint_path, endpoint_spec)
                deps: set[str] = set()
                for run_names in arms_map.values():
                    for run in run_names:
                        arm_slug = arm_from_run(run)
                        deps.add(f"parent_prediction_s{seed}_f{fold}_{arm_slug}")
                        if endpoint == "complete":
                            deps.add(f"head_prediction_s{seed}_f{fold}_{arm_slug}")
                nodes.append(
                    {
                        "id": f"endpoint_{family}_{endpoint}_s{seed}_f{fold}",
                        "kind": "score_endpoint_spec",
                        "spec": str(endpoint_path.relative_to(ROOT)),
                        "outputs": [],
                        "depends_on": sorted(deps),
                    }
                )

    graph = {
        "version": 1,
        "experiment": "E-496",
        "hids": ["H-180", "H-134", "H-126"],
        "launch_authorized": False,
        "nodes": nodes,
        "missing_inputs": missing_inputs,
        "unresolved_templates": [
            node["spec"]
            for node in nodes
            if node.get("unresolved")
        ],
    }
    _write(FRONTIER / "dependency_graph.json", graph)
    return graph


def arm_from_run(run: str) -> str:
    """Recover the registered arm slug from a concrete run identity."""
    for prefix in ("c20261005_e496_",):
        if run.startswith(prefix):
            stem = run[len(prefix) :]
            for suffix in ("_head",):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            return stem.rsplit("_s", 1)[0]
    raise ValueError(f"not an E-496 run identity: {run}")


if __name__ == "__main__":
    graph = materialize()
    print(json.dumps({"nodes": len(graph["nodes"]), "missing_input_entries": len(graph["missing_inputs"]),
                      "unresolved_templates": len(graph["unresolved_templates"]), "launch_authorized": False}))
