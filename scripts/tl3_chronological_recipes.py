"""Frozen finite predictor graph for the fresh train-only TL3 adaptation."""

from __future__ import annotations

from pathlib import Path

PREFIX = "tl3c_"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
FINAL_WEIGHTS = {"w18final_distilled_s5": 0.34, "fttall_tlead_s5": 0.33, "fttall_tlead_s6": 0.33}
W18_COEFFICIENTS = {
    "fbc1k_s5": ("strong", 0.3, 3),
    "ffull_s3": ("full_finetuned", 0.1, 1),
    "ffull_s4": ("full_finetuned", 0.1, 1),
    "frawW_s3ft": ("raw_wmse", 0.125, 1),
    "frawW_s4ft": ("raw_wmse", 0.125, 1),
    "orgbase": ("samefold_baseline", 0.0625, 3),
}


def _base(name: str, seed: int, hidden: int, split: Path) -> list[str]:
    """Keep every train, metric, precision and checkpoint decision explicit."""
    return [
        "--name",
        PREFIX + name,
        "--seed",
        str(seed),
        "--hidden",
        str(hidden),
        "--layers",
        "2",
        "--dropout",
        "0",
        "--train-source",
        "train",
        "--train-seqs",
        "10325",
        "--eval-source",
        "dev",
        "--full-valid",
        "--split-manifest",
        str(split),
        "--precision",
        "fp32",
        "--checkpoint-policy",
        "fixed",
        "--loss-norm",
        "chunk",
        "--sched",
        "onecycle",
        "--warmup-frac",
        "0.05",
        "--wd",
        "0.01",
        "--clip",
        "1",
        "--beta2",
        "0.999",
        "--sync-free",
        "--fused-adam",
        "--pin-memory",
        "--grad-stats",
        "--save-training-state",
        "--train-eval-seqs",
        "0" if name == "fin256_s3" else "32",
    ]


def _auxiliary() -> list[str]:
    """Use the recovered structural and next-change heads."""
    return ["--aux-next", "0.3", "--aux-struct", "0.3", "--aux-weight", "0.5"]


def _ema() -> list[str]:
    """Preserve the teacher-consistency and inference averaging recipe."""
    return ["--ema-teacher", "1", "--ema-decay", "0.999", "--eval-ema"]


def _node(name: str, args: list[str], parents: list[str], export: bool) -> dict[str, object]:
    """Bind one named scientific fit to its exact dependencies and endpoint."""
    batch = int(args[args.index("--batch") + 1])
    chunk = int(args[args.index("--chunk") + 1])
    epochs = int(args[args.index("--epochs") + 1])
    return {
        "id": name,
        "run": PREFIX + name,
        "args": args,
        "parents": parents,
        "export_train": export,
        "epochs": epochs,
        "endpoint": epochs - 1,
        "expected_updates": (10325 // batch) * (20000 // chunk) * epochs,
    }


def build_nodes(split: Path, run_root: Path) -> list[dict[str, object]]:
    """Materialize only the17 declared predictor fits; never fit propensity/valid."""
    nodes = []
    for seed in (3, 4):
        for family, hidden, raw in (
            ("fin256", 256, False),
            ("fmu128", 128, False),
            ("frawW", 128, True),
        ):
            name = f"{family}_s{seed}"
            args = _base(name, seed, hidden, split)
            args += [
                "--batch",
                "32",
                "--chunk",
                "2000",
                "--epochs",
                "4",
                "--lr",
                ".001",
                "--loss",
                "wmse",
            ]
            args += _auxiliary() + _ema() + ["--since-weight", "32", "--since-floor", ".25"]
            args += (
                ["--optim", "adamw"]
                if family == "fin256"
                else ["--optim", "tl3-muon", "--muon-lr-mult", "6"]
            )
            if raw:
                args += ["--raw-only"]
            nodes.append(_node(name, args, [], family == "fin256"))
    args = (
        _base("orgbase", 3, 128, split) + _ema() + ["--since-weight", "32", "--since-floor", ".25"]
    )
    args += [
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "4",
        "--lr",
        ".001",
        "--loss",
        "wmse",
        "--optim",
        "tl3-muon",
        "--muon-lr-mult",
        "6",
    ]
    nodes.append(_node("orgbase", args, [], True))
    for seed in (3, 4):
        for name, parent, hidden in (
            (f"fin256ft_s{seed}", f"fin256_s{seed}", 256),
            (f"ffull_s{seed}", f"fmu128_s{seed}", 128),
            (f"frawW_s{seed}ft", f"frawW_s{seed}", 128),
        ):
            args = _base(name, seed, hidden, split) + _auxiliary()
            args += [
                "--batch",
                "16",
                "--chunk",
                "2000",
                "--epochs",
                "2",
                "--lr",
                ".0001",
                "--loss",
                "gcorr",
                "--optim",
                "adamw",
                "--init",
                PREFIX + parent + "@ep3",
                "--inherit-init-preprocessing",
            ]
            if name.startswith("frawW"):
                args += ["--raw-only"]
            nodes.append(_node(name, args, [parent], True))
    args = (
        _base("fbc1k_s5", 5, 256, split)
        + _auxiliary()
        + _ema()
        + ["--since-weight", "32", "--since-floor", ".25"]
    )
    args += [
        "--batch",
        "32",
        "--chunk",
        "1000",
        "--epochs",
        "4",
        "--lr",
        ".001",
        "--loss",
        "wmse",
        "--optim",
        "tl3-muon",
        "--muon-lr-mult",
        "3",
        "--distill",
        PREFIX + "fin256_s3," + PREFIX + "fin256_s4",
        "--distill-alpha",
        ".5",
        "--distill-boot-h",
        "8",
    ]
    nodes.append(_node("fbc1k_s5", args, ["fin256_s3", "fin256_s4"], True))
    for seed in (5, 6):
        name = f"fttall_tlead_s{seed}"
        args = (
            _base(name, seed, 256, split)
            + _auxiliary()
            + _ema()
            + ["--since-weight", "32", "--since-floor", ".25"]
        )
        args += [
            "--enc",
            "tiedlead",
            "--batch",
            "32",
            "--chunk",
            "1000",
            "--epochs",
            "4",
            "--lr",
            ".001",
            "--loss",
            "wmse",
            "--optim",
            "tl3-muon",
            "--muon-lr-mult",
            "3",
            "--distill",
            PREFIX + "fin256ft_s3," + PREFIX + "fin256ft_s4",
            "--distill-alpha",
            ".5",
            "--distill-boot-h",
            "8",
        ]
        nodes.append(_node(name, args, ["fin256ft_s3", "fin256ft_s4"], False))
    args = _base("w18final_distilled_s5", 5, 256, split) + _auxiliary()
    args += [
        "--batch",
        "16",
        "--chunk",
        "2000",
        "--epochs",
        "2",
        "--lr",
        ".0001",
        "--loss",
        "wmse",
        "--optim",
        "adamw",
        "--init",
        PREFIX + "fbc1k_s5@ep3",
        "--inherit-init-preprocessing",
        "--distill-cache-root",
        str(run_root / "w18-mixed-targets"),
        "--distill-alpha",
        ".7",
        "--distill-family-weight",
        ".1",
    ]
    nodes.append(_node("w18final_distilled_s5", args, list(W18_COEFFICIENTS), False))
    if len(nodes) != 17 or sum(int(node["expected_updates"]) for node in nodes) != 257740:
        raise ValueError("frozen TL3 fit/update inventory changed")
    return nodes
