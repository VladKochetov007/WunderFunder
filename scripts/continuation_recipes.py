"""Materialize fixed continuation jobs without launching or choosing checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "research/campaign_2026-10-04"
FOLDS = "research/campaign_2026-09-29/artifacts/splits"

# Each entry changes only the named intervention from the current combo.
VARIANTS: dict[str, list[str]] = {
    "combo": [],
    "ctrl": ["--optim", "adamw", "--chunk", "2000", "--distill-boot-h", "0"],
    "width384_adamw": [
        "--hidden",
        "384",
        "--optim",
        "adamw",
        "--chunk",
        "2000",
        "--distill-boot-h",
        "0",
    ],
    "width384_combo": ["--hidden", "384"],
    "ema_rows": ["--ema-reference-rows", "2000"],
    "credit_detached": ["--update-rows", "2000", "--detach-rows", "1000"],
    "credit_connected": ["--update-rows", "2000", "--detach-rows", "2000"],
    "since_off": ["--since-weight", "0"],
    "thinning": ["--thin-prob", "1", "--thin-range", "0.1,0.4"],
    "lstm128": ["--cell", "lstm", "--hidden", "128"],
    "gru128": ["--hidden", "128"],
    "structured_multi": ["--arch", "structured", "--decay-init", "multi", "--optim", "adamw"],
    "structured_single": ["--arch", "structured", "--decay-init", "single", "--optim", "adamw"],
    "memory_content": [
        "--arch",
        "event-content",
        "--memory-arm",
        "endpoint-content",
        "--optim",
        "adamw",
    ],
    "memory_position": [
        "--arch",
        "event-content",
        "--memory-arm",
        "endpoint-position",
        "--optim",
        "adamw",
    ],
    "memory_raw_clock": [
        "--arch",
        "event-content",
        "--memory-arm",
        "raw-code-content",
        "--optim",
        "adamw",
    ],
}


def training_args(name: str, fold: int | None, seed: int) -> list[str]:
    """Reproduce the registered combo recipe; final fitting keeps old holdout out."""
    teacher = (
        "ab67_final_since_s3,ab67_final_since_s4"
        if fold is None
        else f"c20260929_since_s3_f{fold},c20260929_since_s4_f{fold}"
    )
    propensity = "prop_v1" if fold is None else f"c20260929_prop_f{fold}"
    args = [
        "{python}",
        "scripts/train_rnn.py",
        "--name",
        name,
        "--train-source",
        "train+valid",
        "--full-valid",
        "--loss",
        "wmse",
        "--aux-struct",
        "0.3",
        "--aux-next",
        "0.3",
        "--propensity",
        propensity,
        "--prop-power",
        "2",
        "--prop-eps",
        "0.03",
        "--valid-true-mask",
        "0.03",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--batch",
        "32",
        "--chunk",
        "1000",
        "--epochs",
        "4",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--seed",
        str(seed),
        "--optim",
        "muon",
        "--muon-scope",
        "rnn",
        "--ema-teacher",
        "1",
        "--ema-decay",
        "0.999",
        "--eval-ema",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--distill",
        teacher,
        "--distill-alpha",
        "0.5",
        "--distill-boot-h",
        "8",
        "--fast",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--save-training-state",
    ]
    if fold is None:
        # Historical final evaluation participates in RNG consumption, never selection.
        args += ["--eval-source", "holdout"]
    else:
        args += ["--split-manifest", f"{FOLDS}/fold{fold}.json", "--no-eval-during-fit"]
    return args


def head_args(name: str, parent: str, fold: int | None) -> list[str]:
    """Keep head FT AdamW, one epoch and historical inherited auxiliaries."""
    args = [
        "{python}",
        "scripts/train_rnn.py",
        "--name",
        name,
        "--init",
        parent,
        "--train-source",
        "valid",
        "--full-valid",
        "--mask-weight",
        "--seed",
        "3",
        "--epochs",
        "1",
        "--batch",
        "16",
        "--chunk",
        "2000",
        "--fast",
        "--inherit-init-preprocessing",
        "--loss",
        "gcorr",
        "--optim",
        "adamw",
        "--freeze",
        "inp,gru",
        "--lr",
        "0.0005",
        "--wd",
        "0.01",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "0",
        "--save-training-state",
    ]
    if fold is None:
        args += ["--eval-source", "holdout"]
    else:
        args += ["--split-manifest", f"{FOLDS}/fold{fold}.json", "--no-eval-during-fit"]
    return args


def source_paths(root: Path) -> list[str]:
    """Include trainer imports and the shared scientific specification itself."""
    return sorted(
        [
            *(str(path.relative_to(root)) for path in (root / "src/wnn").glob("*.py")),
            "scripts/train_rnn.py",
            "scripts/predict_run.py",
            "scripts/continuation_recipes.py",
        ]
    )


def input_paths(fold: int | None, *, parent: str | None = None) -> list[str]:
    """Declare the actual dependency closure; raw data remains separate on Kaggle."""
    inputs = [
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "datasets/valid_mask.parquet",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        "cache/chains.npz",
        "cache/chain_folds.npz",
        f"{CAMPAIGN}/portable_inputs.json",
    ]
    if fold is not None:
        inputs.append(f"{FOLDS}/fold{fold}.json")
    if parent:
        run, _, epoch = parent.partition("@ep")
        inputs.append(f"artifacts/runs/{run}/model_ep{epoch}.pt")
        inputs.append(f"artifacts/runs/{run}/provenance.json")
    else:
        propensity = "prop_v1" if fold is None else f"c20260929_prop_f{fold}"
        inputs.append(f"artifacts/runs/{propensity}/model.pt")
        for seed in (3, 4):
            teacher = (
                f"ab67_final_since_s{seed}" if fold is None else f"c20260929_since_s{seed}_f{fold}"
            )
            inputs += [
                f"artifacts/runs/{teacher}/model.pt",
                f"artifacts/runs/{teacher}/provenance.json",
            ]
            for split in ("train", "valid"):
                inputs += [
                    f"cache/pred_{teacher}_{split}.npy",
                    f"cache/pred_{teacher}_{split}.json",
                ]
    return sorted(set(inputs))


def recipe(trial: str, fold: int | None, seed: int, backend: str, root: Path = ROOT) -> dict:
    """Return an unsealed job; the runner seals content hashes after all code checks."""
    if fold is not None and fold not in range(4):
        raise ValueError("fold must be 0..3 or final")
    if backend not in {"local", "kaggle"}:
        raise ValueError("training recipes currently use local or Kaggle CUDA")
    suffix = "final" if fold is None else f"f{fold}"
    name = f"c20261004_{trial}_s{seed}_{suffix}"
    parent = None
    if trial == "combo_head":
        if seed != 3:
            raise ValueError("registered head comparison uses parent seed3, FT seed3")
        parent = (
            "c20261004_combo_s3_final@ep3" if fold is None else f"c20261003_combo_s3_f{fold}@ep3"
        )
        argv = head_args(name, parent, fold)
        epoch = 0
    else:
        if trial not in VARIANTS:
            raise ValueError(f"unknown implemented recipe: {trial}")
        argv = training_args(name, fold, seed) + VARIANTS[trial]
        epoch = 3
    local = backend == "local"
    return {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": argv,
        "code": source_paths(root),
        "inputs": input_paths(fold, parent=parent),
        "artifacts": [
            f"artifacts/runs/{name}/{file}"
            for file in (
                f"model_ep{epoch}.pt",
                "provenance.json",
                "history.json",
                "training_state.pt",
            )
        ],
        "limits": {
            "cpus": 8 if local else 2,
            "ram_bytes": (24 if local else 10) * 1024**3,
            "timeout_seconds": 43200,
            "gpu_memory_bytes": 4_200_000_000 if local else 11 * 1024**3,
            "gpu_stop_bytes": 4_800_000_000 if local else 13 * 1024**3,
        },
        "environment": {
            "WNN_LOADER_WORKERS": "2" if local else "1",
            "WNN_LOADER_RAM_GB": "3",
            "WNN_PORTABLE_DATA_MANIFEST": f"{CAMPAIGN}/portable_inputs.json",
        },
    }


def merged_screen() -> dict:
    """Freeze equal parent/head mixtures without searching weights on any fold."""
    return {
        "folds": [0, 1, 2, 3],
        "n_boot": 2000,
        "arms": {
            "CTRL": {"c20261001_d256_s3_f{fold}": 0.5, "c20261001_ft_d256_s3_f{fold}h": 0.5},
            "COMBO": {"c20261003_combo_s3_f{fold}": 0.5, "c20261004_combo_head_s3_f{fold}": 0.5},
        },
        "comparisons": [["COMBO", "CTRL"]],
        "class_mix": {"pub143": [0.183, 0.160, 0.198, 0.459]},
    }


def main() -> None:
    """List or emit recipes; launch authorization belongs to the campaign runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", choices=[*VARIANTS, "combo_head", "merged_screen", "list"])
    parser.add_argument("--fold", default="0", choices=["0", "1", "2", "3", "final"])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--backend", choices=["local", "kaggle"], default="local")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trial == "list":
        payload = {
            "recipes": [*VARIANTS, "combo_head"],
            "status": "require sealed code and implementation checks",
        }
    elif args.trial == "merged_screen":
        payload = merged_screen()
    else:
        payload = recipe(
            args.trial, None if args.fold == "final" else int(args.fold), args.seed, args.backend
        )
    value = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(value)
    else:
        print(value, end="")


if __name__ == "__main__":
    main()
