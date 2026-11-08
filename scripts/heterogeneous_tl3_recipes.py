from __future__ import annotations

import argparse
import json
from pathlib import Path

PREFIX = "het6_"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
FIT_SEQUENCES = 10325
EVALUATION_SEQUENCES = 1873
STUDENT_SEEDS = (1301, 1302)
CONTROL_CACHE = "gru-only-targets"
HETEROGENEOUS_CACHE = "heterogeneous-targets"
CONTROL_TEACHERS = (
    ("tl3c_fin256_s3", 3, 0.5),
    ("tl3c_fin256_s4", 3, 0.5),
)
HETEROGENEOUS_TEACHERS = (
    ("tl3c_fin256_s3", 3, 0.35),
    ("tl3c_fin256_s4", 3, 0.35),
    ("g2-galam-lstm128", 2, 0.10),
    (PREFIX + "lstm_s1201", 3, 0.10),
    (PREFIX + "tcn_s1202", 3, 0.10),
)


def common_arguments(name: str, seed: int, split: Path) -> list[str]:
    """Freeze the shared fresh-fit recipe; checkpoint epochs are zero based."""
    return [
        "--name",
        PREFIX + name,
        "--seed",
        str(seed),
        "--train-source",
        "train",
        "--train-seqs",
        str(FIT_SEQUENCES),
        "--eval-source",
        "dev",
        "--full-valid",
        "--split-manifest",
        str(split),
        "--precision",
        "fp32",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--dropout",
        "0",
        "--batch",
        "32",
        "--chunk",
        "1000",
        "--epochs",
        "4",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--optim",
        "adamw",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--beta2",
        "0.999",
        "--sched",
        "onecycle",
        "--warmup-frac",
        "0.05",
        "--clip",
        "1",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--ema-teacher",
        "1",
        "--ema-decay",
        "0.999",
        "--eval-ema",
        "--aux-next",
        "0.3",
        "--aux-struct",
        "0.3",
        "--aux-weight",
        "0.5",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--distill-boot-h",
        "0",
        "--distill-family-weight",
        "0",
        "--train-eval-seqs",
        "32",
        "--sync-free",
        "--fused-adam",
        "--pin-memory",
        "--grad-stats",
        "--save-training-state",
    ]


def _replace(arguments: list[str], option: str, value: str) -> None:
    arguments[arguments.index(option) + 1] = value


def _node(name: str, arguments: list[str], role: str, dependencies: list[str]) -> dict[str, object]:
    return {
        "id": name,
        "run": PREFIX + name,
        "role": role,
        "args": arguments,
        "parents": dependencies,
        "epochs": 4,
        "endpoint": 3,
        "expected_updates": (FIT_SEQUENCES // 32) * (20000 // 1000) * 4,
        "export_train": role == "teacher",
        "export_valid": True,
    }


def build_nodes(
    split: Path,
    run_root: Path,
    control_cache: Path | None = None,
    heterogeneous_cache: Path | None = None,
) -> list[dict[str, object]]:
    """Materialize two teachers and four paired students without launching work.

    Cache paths must point to separately validated fitting-only preweighted caches.
    A different path is not permission to alter either cache's frozen mixture.
    """
    control = control_cache if control_cache is not None else run_root / CONTROL_CACHE
    heterogeneous = (
        heterogeneous_cache if heterogeneous_cache is not None else run_root / HETEROGENEOUS_CACHE
    )
    if control.resolve() == heterogeneous.resolve():
        raise ValueError("control and heterogeneous target caches must be distinct")
    lstm = common_arguments("lstm_s1201", 1201, split)
    _replace(lstm, "--cell", "lstm")
    tcn = common_arguments("tcn_s1202", 1202, split)
    for key, value in (("--arch", "tcn"), ("--hidden", "128"), ("--layers", "12")):
        _replace(tcn, key, value)
    tcn += ["--tcn-mix", "0"]
    nodes = [
        _node("lstm_s1201", lstm, "teacher", []),
        _node("tcn_s1202", tcn, "teacher", []),
    ]
    for seed in STUDENT_SEEDS:
        for arm, cache, parents in (
            ("control", control, [record[0] for record in CONTROL_TEACHERS]),
            (
                "heterogeneous",
                heterogeneous,
                [record[0] for record in HETEROGENEOUS_TEACHERS],
            ),
        ):
            name = f"{arm}_s{seed}"
            arguments = common_arguments(name, seed, split)
            arguments += [
                "--distill-cache-root",
                str(cache),
                "--distill-cache-variant",
                "source",
                "--distill-cache-scale",
                "1",
                "--distill-alpha",
                "0.5",
            ]
            nodes.append(_node(name, arguments, "student", parents))
    if len(nodes) != 6 or sum(int(node["expected_updates"]) for node in nodes) != 154560:
        raise ValueError("six-fit scientific inventory changed")
    return nodes


def materialize_plan(split: Path, run_root: Path) -> dict[str, object]:
    """Return a source-independent recipe plan, not execution or resource admission."""
    return {
        "schema": "wnn.heterogeneous-tl3-recipes.v1",
        "expected_split_sha256": SPLIT_SHA256,
        "fitting_train_sequences": FIT_SEQUENCES,
        "fitting_valid_sequences": 0,
        "evaluation_valid_sequences": EVALUATION_SEQUENCES,
        "new_fits": 6,
        "new_teacher_fits": 2,
        "new_student_fits": 4,
        "configured_epochs": 24,
        "expected_total_updates": 154560,
        "control_contributors": CONTROL_TEACHERS,
        "heterogeneous_contributors": HETEROGENEOUS_TEACHERS,
        "cache_contract": "raw weighted prediction sum before one training-time clip; sum=1",
        "baseline": "fresh canonical TL3 .34 W18 + .33 tiedlead_s5 + .33 tiedlead_s6",
        "baseline_dependency": "W18 cache repair/final fit is separate from these six new fits",
        "nodes": build_nodes(split, run_root),
        "status": "recipe_only_receiver_profiles_and_frozen_source_required",
    }


def main() -> None:
    """Write a create-only JSON plan; this command never starts training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_plan(args.split_manifest, args.run_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
