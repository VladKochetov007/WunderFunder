"""Expand a cohort arm file into warm_queue_worker run specs for one machine.

    .venv/bin/python scripts/make_cohort_specs.py ARMS.json --root /kaggle/working/repo \
        --python /usr/bin/python3 --queue QUEUE_DIR [--only A00,A01] [--no-score-refs]

Each arm is a flag diff against the cohort's base argv. A spec runs fit -> predict ->
official source-qualified fold panel score. Paths in argv are written relative to ``--root``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

RUN_ID = re.compile(r"[A-Za-z0-9_.-]{1,96}\Z")
REFERENCE_MEMBER = re.compile(r"([A-Za-z0-9_.-]{1,96}@ep[0-9]+)(?:\*([^,*]+))?\Z")
DEFAULT_F0_SPLIT = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
MAX_SPLIT_MANIFEST_BYTES = 2_000_000


def _format_path(value: object, fmt: dict[str, object]) -> str:
    """Format a declared project path without accepting an empty value."""
    if not isinstance(value, str) or not value:
        raise ValueError("split path must be a non-empty string")
    return value.format(**fmt)


def _flag_values(argv: list[str], names: tuple[str, ...]) -> dict[str, str]:
    """Read the single value for any matching CLI flag and reject malformed repeats."""
    values: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token, separator, inline_value = argv[index].partition("=")
        if token in names:
            if token in values:
                raise ValueError(f"malformed or repeated {token} in trainer arguments")
            if separator:
                if not inline_value:
                    raise ValueError(f"malformed or repeated {token} in trainer arguments")
                values[token] = inline_value
                index += 1
            else:
                if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                    raise ValueError(f"malformed or repeated {token} in trainer arguments")
                values[token] = argv[index + 1]
                index += 2
        else:
            index += 1
    return values


def _same_project_path(left: str, right: str, root: str) -> bool:
    """Compare project paths after making relative paths root-relative."""
    left_path = Path(left)
    right_path = Path(right)
    if not left_path.is_absolute():
        left_path = Path(root) / left_path
    if not right_path.is_absolute():
        right_path = Path(root) / right_path
    return left_path.resolve(strict=False) == right_path.resolve(strict=False)


def _declared_split_path(
    cohort: dict[str, object], flags: list[str], root: str, fmt: dict[str, object]
) -> str | None:
    """Bind the cohort split path to any trainer split argument."""
    declared_values = [
        cohort[key] for key in ("split_manifest", "split_path") if cohort.get(key) is not None
    ]
    if len(declared_values) == 2 and not _same_project_path(
        _format_path(declared_values[0], fmt), _format_path(declared_values[1], fmt), root
    ):
        raise ValueError("cohort split_manifest and split_path disagree")
    declared = _format_path(declared_values[0], fmt) if declared_values else None
    trainer_values = _flag_values(flags, ("--split-manifest", "--split"))
    if len(trainer_values) == 2 and not _same_project_path(
        trainer_values["--split-manifest"], trainer_values["--split"], root
    ):
        raise ValueError("trainer --split-manifest and --split disagree")
    trainer_path = next(iter(trainer_values.values()), None)
    if declared is not None and trainer_path is None:
        raise ValueError("declared cohort split path is missing from trainer arguments")
    if (
        declared is not None
        and trainer_path is not None
        and not _same_project_path(declared, trainer_path, root)
    ):
        raise ValueError("cohort split path disagrees with trainer split argument")
    return declared or trainer_path


def _validate_fold_flags(flags: list[str], fold: int) -> None:
    """Reject trainer fold options that contradict the cohort fold."""
    values = _flag_values(flags, ("--fold", "--valid-fold"))
    parsed: list[int] = []
    for flag, value in values.items():
        try:
            parsed.append(int(value))
        except ValueError as error:
            raise ValueError(f"{flag} must be an integer") from error
    if any(value != fold for value in parsed):
        raise ValueError("trainer fold argument disagrees with cohort fold")
    n_folds = _flag_values(flags, ("--n-valid-folds",))
    if n_folds:
        try:
            count = int(n_folds["--n-valid-folds"])
        except ValueError as error:
            raise ValueError("--n-valid-folds must be an integer") from error
        if count <= fold:
            raise ValueError("trainer fold is outside --n-valid-folds")


def _validate_split_file(path: str, root: str, fold: int) -> None:
    """Check the declared split file binds to the selected cohort fold."""
    split_path = Path(path)
    if not split_path.is_absolute():
        split_path = Path(root) / split_path
    with split_path.open("rb") as stream:
        raw = stream.read(MAX_SPLIT_MANIFEST_BYTES + 1)
    if not raw or len(raw) > MAX_SPLIT_MANIFEST_BYTES:
        raise ValueError("declared split manifest is empty or exceeds the read bound")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("declared split manifest must be a JSON object")
    manifest_fold = manifest.get("fold")
    n_folds = manifest.get("n_folds")
    if not isinstance(manifest_fold, int) or isinstance(manifest_fold, bool):
        raise ValueError("declared split manifest lacks an integer fold")
    if manifest_fold != fold:
        raise ValueError("declared split manifest fold disagrees with cohort fold")
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or not 0 <= fold < n_folds:
        raise ValueError("cohort fold is outside the declared split manifest")


def _validate_references(references: list[str]) -> None:
    """Reject malformed run/epoch blend members before writing a queue spec."""
    for reference in references:
        if not isinstance(reference, str) or not reference:
            raise ValueError("score references must be non-empty strings")
        members = reference.split(",")
        if any(not REFERENCE_MEMBER.fullmatch(member) for member in members):
            raise ValueError(f"malformed score reference {reference!r}")
        for member in members:
            match = REFERENCE_MEMBER.fullmatch(member)
            assert match is not None
            weight = match.group(2)
            if weight is not None:
                try:
                    numeric_weight = float(weight)
                except ValueError as error:
                    raise ValueError(
                        f"malformed score reference weight in {reference!r}"
                    ) from error
                if not math.isfinite(numeric_weight):
                    raise ValueError(f"non-finite score reference weight in {reference!r}")


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


def build_spec(
    cohort: dict[str, object],
    arm: dict[str, object],
    root: str,
    python: str,
    score_refs: bool,
) -> dict[str, object]:
    """Return one queue spec whose argv contains only root-relative project paths."""
    root = str(Path(root).resolve())
    seed = int(arm.get("seed", cohort["seed"]))
    fold = int(cohort.get("fold", 0))
    if fold < 0:
        raise ValueError("cohort fold must be a non-negative integer")
    run = f"{cohort['run_prefix']}_{arm['id'].lower()}_s{seed}_f{fold}"
    if not RUN_ID.fullmatch(run):
        raise ValueError(f"unsafe run name {run!r}")
    fmt = {"seed": seed, "root": root}
    base = [str(token).format(**fmt) for token in cohort["base_argv"]]
    diff = {k: (str(v).format(**fmt) if isinstance(v, str) else v) for k, v in arm["diff"].items()}
    flags = apply_diff(base, {"--name": run, "--seed": seed} | diff)
    _validate_fold_flags(flags, fold)
    split_path = _declared_split_path(cohort, flags, root, fmt)
    if fold > 0 and split_path is None:
        raise ValueError("non-zero cohort folds require a declared trainer split path")
    if split_path is not None:
        _validate_split_file(split_path, root, fold)
        split_path = str((Path(root) / split_path).resolve())
    checkpoint_epoch = _flag_values(flags, ("--checkpoint-epoch",))
    if "--checkpoint-epoch" not in checkpoint_epoch:
        raise ValueError("trainer arguments must declare --checkpoint-epoch")
    epoch = int(checkpoint_epoch["--checkpoint-epoch"])
    if epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    trainer = f"{root}/{cohort['trainer']}"
    run_at_epoch = f"{run}@ep{epoch}"
    references = arm.get("score_references", cohort.get("score_references", []))
    if not isinstance(references, list):
        raise ValueError("score_references must be a list")
    references = [str(reference).format(**fmt) for reference in references] if score_refs else []
    _validate_references(references)
    grade_path = arm.get("grade_path", cohort.get("grade_path"))
    if grade_path is None:
        grade_path = f"{root}/artifacts/runs/{run}/grade_fold{fold}.json"
    else:
        grade_path = _format_path(grade_path, fmt)
        if not Path(grade_path).is_absolute():
            grade_path = f"{root}/{grade_path}"
    score_script = "score_fold_panel.py" if split_path is not None else "score_fold0_panel.py"
    score = [python, f"{root}/scripts/{score_script}", run_at_epoch]
    if split_path is not None:
        score += ["--split", split_path, "--expected-fold", str(fold)]
    for reference in references:
        score += ["--reference", reference]
    score += ["--out", str(grade_path)]
    env = {k: v.format(**fmt) for k, v in cohort.get("env", {}).items()}
    spec = {
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
            str(grade_path),
            f"{root}/cache/pred_{run}_valid.npy",
            f"{root}/cache/pred_{run}_valid.json",
        ],
    }
    experiment_id = arm.get("experiment_id", cohort.get("experiment_id"))
    if experiment_id is not None:
        hypothesis_id = arm.get(
            "hypothesis_id",
            arm.get("hypothesis", cohort.get("hypothesis_id", cohort.get("hypothesis"))),
        )
        receipt_path = arm.get("receipt_path", cohort.get("receipt_path"))
        ledger_path = arm.get("ledger_path", cohort.get("ledger_path"))
        for label, value in (
            ("experiment_id", experiment_id),
            ("hypothesis_id", hypothesis_id),
            ("receipt_path", receipt_path),
            ("ledger_path", ledger_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"comparison requires a non-empty {label}")
        if not references or len(set(references)) != len(references):
            raise ValueError("comparison requires non-empty, unique score_references")
        split_metadata_path = split_path or arm.get(
            "split_path", cohort.get("split_path", f"{root}/{DEFAULT_F0_SPLIT}")
        )
        resolved_receipt = str((Path(root) / _format_path(receipt_path, fmt)).resolve())
        resolved_ledger = str((Path(root) / _format_path(ledger_path, fmt)).resolve())
        resolved_split = str((Path(root) / _format_path(split_metadata_path, fmt)).resolve())
        if (
            len(
                {str(Path(grade_path).resolve()), resolved_receipt, resolved_ledger, resolved_split}
            )
            != 4
        ):
            raise ValueError("comparison grade, receipt, ledger and split paths must be distinct")
        comparison = {
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "candidate": run_at_epoch,
            "references": references,
            "grade_path": str(grade_path),
            "receipt_path": resolved_receipt,
            "ledger_path": resolved_ledger,
            "split_path": resolved_split,
            "fold": fold,
            "expected_fold": fold,
        }
        fidelity = arm.get("fidelity", cohort.get("fidelity"))
        if fidelity is not None:
            if not isinstance(fidelity, dict):
                raise ValueError("comparison fidelity must be an object")
            comparison["fidelity"] = fidelity
        spec["comparison"] = comparison
    return spec


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
