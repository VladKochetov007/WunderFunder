"""Expand a cohort arm file into warm_queue_worker run specs for one machine.

    .venv/bin/python scripts/make_cohort_specs.py ARMS.json --root /kaggle/working/repo \
        --python /usr/bin/python3 --queue QUEUE_DIR [--only A00,A01] [--no-score-refs]

Each arm is a flag diff against the cohort's base argv. A spec runs fit -> predict ->
official fold-0 panel score. Paths in argv are written relative to ``--root``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_ID = re.compile(r"[A-Za-z0-9_.-]{1,96}\Z")


def apply_diff(base: list[str], diff: dict[str, object]) -> list[str]:
    """Set, add or remove ``--flag`` entries; value None removes, True means bare flag."""
    argv = list(base)
    for flag, value in diff.items():
        if flag in argv:
            index = argv.index(flag)
            has_value = index + 1 < len(argv) and not argv[index + 1].startswith("--")
            del argv[index : index + (2 if has_value else 1)]
        if value is None:
            continue
        argv += [flag] if value is True else [flag, str(value)]
    return argv


def build_spec(cohort: dict, arm: dict, root: str, python: str, score_refs: bool) -> dict:
    """Return one queue spec whose argv contains only root-relative project paths."""
    seed = int(arm.get("seed", cohort["seed"]))
    run = f"{cohort['run_prefix']}_{arm['id'].lower()}_s{seed}_f0"
    if not RUN_ID.fullmatch(run):
        raise ValueError(f"unsafe run name {run!r}")
    fmt = {"seed": seed, "root": root}
    base = [str(token).format(**fmt) for token in cohort["base_argv"]]
    diff = {k: (str(v).format(**fmt) if isinstance(v, str) else v) for k, v in arm["diff"].items()}
    flags = apply_diff(base, {"--name": run, "--seed": seed} | diff)
    epoch = int(flags[flags.index("--checkpoint-epoch") + 1])
    trainer = f"{root}/{cohort['trainer']}"
    score = [python, f"{root}/scripts/score_fold0_panel.py", f"{run}@ep{epoch}"]
    if score_refs:
        for reference in cohort.get("score_references", []):
            score += ["--reference", reference.format(**fmt)]
    score += ["--out", f"{root}/artifacts/runs/{run}/grade_fold0.json"]
    env = {k: v.format(**fmt) for k, v in cohort.get("env", {}).items()}
    return {
        "run_id": run,
        "priority": int(arm.get("priority", 100)),
        "arm": arm["id"],
        "hypothesis": arm.get("hypothesis", cohort.get("hypothesis")),
        "notes": arm.get("notes", ""),
        "steps": [
            {
                "name": "fit",
                "argv": [python, trainer, *flags],
                "timeout_seconds": int(cohort["fit_timeout"]),
                "env": env,
                "cwd": root,
            },
            {
                "name": "predict",
                "argv": [
                    python,
                    f"{root}/scripts/predict_run.py",
                    run,
                    "valid",
                    "--epoch",
                    str(epoch),
                ],
                "timeout_seconds": 1800,
                "env": env,
                "cwd": root,
            },
            {"name": "score", "argv": score, "timeout_seconds": 900, "env": env, "cwd": root},
        ],
        "expected_outputs": [
            f"{root}/artifacts/runs/{run}/model_ep{epoch}.pt",
            f"{root}/artifacts/runs/{run}/provenance.json",
            f"{root}/artifacts/runs/{run}/grade_fold0.json",
            f"{root}/cache/pred_{run}_valid.json",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", type=Path)
    parser.add_argument("--root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--no-score-refs", action="store_true")
    args = parser.parse_args()
    cohort = json.loads(args.arms.read_text())
    wanted = {x for x in args.only.split(",") if x}
    pending = args.queue / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    for arm in cohort["arms"]:
        if wanted and arm["id"] not in wanted:
            continue
        spec = build_spec(cohort, arm, args.root.rstrip("/"), args.python, not args.no_score_refs)
        path = pending / f"{spec['run_id']}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(spec, indent=1))
        print(path)


if __name__ == "__main__":
    main()
