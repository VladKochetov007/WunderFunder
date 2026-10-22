from __future__ import annotations

import hashlib
import json
import math

from bootstrap import EXPERIMENT, PACKET, ROOT, SOURCE, STATE
from feature_common import legacy, sha, training_arguments, verify_source, write
from feature_common import read as read
from wnn.splits import load_campaign_split
from wnn.training.membership import load_component_fit_subset


def subset_manifest(fold: int, fraction: float) -> dict:
    """Select nested, complete fitting components without accessing labels."""
    import numpy as np

    split = load_campaign_split(ROOT / "splits" / f"fold{fold}.json", ROOT / "data")
    allowed = {"train": set(split.train_groups), "valid": set(split.valid_groups)}
    eligible = [(s, g, c) for s, g, c in split.components if g in allowed[s]]
    components = sorted(
        {c for _, _, c in eligible},
        key=lambda c: hashlib.sha256(f"4h22:{fold}:{c}".encode()).digest(),
    )
    chosen = set(components[: max(1, math.ceil(len(components) * fraction))])
    groups = {s: sorted(g for ss, g, c in eligible if ss == s and c in chosen) for s in allowed}
    members = sorted((s, g) for s in groups for g in groups[s])
    evaluated = sorted(split.evaluation_groups)
    value = {
        "version": 1,
        "experiment_id": EXPERIMENT,
        "fold": fold,
        "fraction_of_components": fraction,
        "selection_seed": "sha256:4h22:fold:component",
        "split_manifest_sha256": split.sha256,
        "selected_groups": groups,
        "selected_components": sorted(chosen),
        "selected_members_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "selected_component_sha256": hashlib.sha256(
            np.asarray(sorted(chosen), dtype="<i8").tobytes()
        ).hexdigest(),
        "evaluation_groups": evaluated,
        "evaluation_group_count": len(evaluated),
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(evaluated, dtype="<i8").tobytes()
        ).hexdigest(),
        "evaluation_exposure": "previously exposed development components",
    }
    path = PACKET / "subsets" / f"f{fold}_p{round(100 * fraction)}.json"
    write(path, value, fresh=True)
    load_component_fit_subset(path, split, expected_experiment_id=EXPERIMENT)
    return dict(
        path=str(path),
        sha256=sha(path),
        train=len(groups["train"]),
        valid=len(groups["valid"]),
        components=len(chosen),
    )


def argv_for(job: dict, profile: bool) -> list[str]:
    """Amend only fitting membership; retain the frozen six-epoch FP32 recipe."""
    base = dict(job, family=job["mechanism"])
    argv = training_arguments(ROOT, base, mode="profile" if profile else "fit")
    argv += [
        "--fit-subset-manifest",
        job["subset"]["path"],
        "--fit-subset-experiment-id",
        EXPERIMENT,
    ]
    if profile:
        for flag, value in [("--profile-steps", "8"), ("--profile-warmup", "2")]:
            argv[argv.index(flag) + 1] = value
        argv += ["--max-batches", "10"]
    return argv


def prepare() -> None:
    """Freeze 24 fits, exact commands, dependency hashes and unchanged scoring."""
    if (PACKET / "PLAN.json").exists():
        raise FileExistsError("Preparation is single-use; preserve the earlier packet")
    identity = verify_source(ROOT)
    modules = legacy(ROOT)
    gpus = (
        __import__("subprocess")
        .check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
        .splitlines()
    )
    assert len(gpus) == 8
    cells = [(0, 1201), (2, 1202), (3, 1203)]
    subsets = {(f, p): subset_manifest(f, p) for f, _ in cells for p in (0.10, 0.25)}
    jobs = []
    for panel, (mechanism, fraction, arms) in enumerate(
        [
            ("age", 0.10, ["group", "field"]),
            ("innovation", 0.10, ["moments", "innovations"]),
            ("age", 0.25, ["group", "field"]),
            ("innovation", 0.25, ["moments", "innovations"]),
        ]
    ):
        for cell_index, (fold, seed) in enumerate(cells):
            for arm_index, arm in enumerate(arms):
                index = 2 * cell_index + arm_index
                gpu = 2 * panel + index // 3
                name = f"short4h_{mechanism}_p{round(100 * fraction)}_{arm}_f{fold}_s{seed}"
                work = STATE / "work" / name
                work.mkdir(parents=True, exist_ok=False)
                inputs = modules["fit_workspace"].prepare_workspace(
                    work, ROOT / "data", ROOT / "splits" / f"fold{fold}.json", fold
                )
                job = dict(
                    id=name,
                    mechanism=mechanism,
                    family=f"{mechanism}_p{round(100 * fraction)}",
                    arm=arm,
                    fold=fold,
                    seed=seed,
                    fraction=fraction,
                    gpu_index=gpu,
                    gpu_uuid=gpus[gpu],
                    subset=subsets[fold, fraction],
                    slot=index % 3,
                    cpus=[2 * (3 * gpu + index % 3), 2 * (3 * gpu + index % 3) + 1],
                )
                write(
                    work / "INPUTS.json",
                    dict(inputs, actual_fit_subset=job["subset"], source=identity),
                    fresh=True,
                )
                for phase in ["profile", "fit"]:
                    phase_work = work
                    if phase == "profile":
                        phase_work = STATE / "profiles" / name
                        phase_work.mkdir(parents=True, exist_ok=False)
                        modules["fit_workspace"].prepare_workspace(
                            phase_work, ROOT / "data", ROOT / "splits" / f"fold{fold}.json", fold
                        )
                    prediction = [
                        name,
                        "valid",
                        "--epoch",
                        "5",
                        "--campaign-split-manifest",
                        str(ROOT / "splits" / f"fold{fold}.json"),
                        "--campaign-evaluation-only",
                        "--device",
                        "cuda",
                        "--inference-precision",
                        "fp32",
                    ]
                    modules["worker"]._make_request(
                        phase_work / f"{phase}_request.json",
                        mode="profile" if phase == "profile" else "fit",
                        name=name,
                        source_root=SOURCE,
                        work=phase_work,
                        train_argv=argv_for(job, phase == "profile"),
                        predict_argv=prediction,
                    )
                job["requests"] = {
                    str(w): sha(w)
                    for w in [
                        STATE / "profiles" / name / "profile_request.json",
                        work / "fit_request.json",
                    ]
                }
                jobs.append(job)
    dependencies = {
        str(p): sha(p) for p in (ROOT / "runtime/parallel-expansion-v1/code").glob("*.py")
    }
    plan = dict(
        dependencies=dependencies,
        schema="tabdata-short4h-plan.v1",
        experiment=EXPERIMENT,
        stage="prepared_not_started",
        user_authority="22 October: 8xB300, three jobs per GPU, STRICT four hours node wall time",
        hard_wall_seconds=14400,
        service_runtime_max_seconds=13500,
        service_stop_grace_seconds=120,
        profile_budget_seconds=900,
        grade_cutoff_seconds=13140,
        allow_partial_under_hard_cutoff=True,
        max_stop_and_finalizer_allowance_seconds=240,
        all_in_service_bound_seconds=13740,
        fit_cutoff_seconds=11100,
        final_export_cutoff_seconds=12900,
        cells=cells,
        fits=24,
        mechanism_comparisons=2,
        fidelity_panels=4,
        jobs=jobs,
        source=identity,
        endpoint="six epochs, fixed epoch5; full20k sequences; full fixed exposed evaluation; no best checkpoint",
        measured_subset_runtime_seconds=None,
        budget_gpu_hours_ceiling=32,
        fidelity_caveat="10pct/25pct complete-component fractions, not full-data results; fidelity panels share cells and are not independent replications",
        selection_policy="retain every cell; compare paired endpoint WP; no auto promotion or blanket zero-crossing rejection",
    )
    write(PACKET / "PLAN.json", plan, fresh=True)
    files = {
        str(p.relative_to(PACKET)): sha(p)
        for p in PACKET.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    write(PACKET / "MANIFEST.json", dict(files=files), fresh=True)
    print(
        json.dumps(
            dict(
                stage=plan["stage"],
                fits=24,
                panels=4,
                subsets=subsets_to_list(subsets),
                plan_sha256=sha(PACKET / "PLAN.json"),
            )
        )
    )


def subsets_to_list(subsets: dict) -> list[dict]:
    """Report the selected data size without dumping source identities."""
    return [dict(fold=f, fraction=p, **s) for (f, p), s in subsets.items()]


if __name__ == "__main__":
    prepare()
