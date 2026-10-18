"""Fixed labels-only F1/seed-1001 pair recipe for the canonical trainer."""

from __future__ import annotations

from pathlib import Path

SOURCE_COMMIT = "a266b036b02e8e012c6eb31072291dad618a169d"
PROFILE_SPEC_SHA256 = "1ca416cac32fb075b4b9d16683e9a07634f3926e625e62576468c99b2b722d56"
FOLD1_SHA256 = "d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9"
BASE_RECIPE = {
    "fold": 1,
    "seed": 1001,
    "epochs": 6,
    "checkpoint_epoch": 5,
    "train_source": "train+valid",
    "train_seqs": 10607,
    "valid_train_fraction": 1.0,
    "valid_repeat": 1,
    "n_valid_folds": 4,
    "loss": "wmse",
    "loss_norm": "chunk",
    "precision": "fp32",
    "batch": 8,
    "chunk": 2000,
    "dropout": 0.0,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "scheduler": "onecycle",
    "warmup_fraction": 0.05,
    "head_lr_multiplier": 1.0,
    "gradient_clip": 1.0,
    "batch_order": "seeded",
    "teacher": None,
    "propensity": None,
    "initializer": None,
    "evaluation_during_fit": False,
    "checkpoint_policy": "fixed",
    "cold_initialization": True,
}
ARMS = (
    {"name": "rr18_r01_wide_f1_s1001", "hidden": 512, "layers": 1, "parameters": 2_210_306},
    {"name": "rr18_r01_deep_f1_s1001", "hidden": 384, "layers": 2, "parameters": 2_151_554},
)


def validate_fit_spec(spec: dict[str, object]) -> tuple[int, int]:
    """Require a sealed, fixed F1 pair and populated measured runtime deadlines."""
    if spec.get("schema_version") != 1 or spec.get("state") != "SEALED_FOR_F1_PAIR":
        raise ValueError("fit spec is not sealed for the F1 pair")
    expected = {
        "experiment_id": "E-RR18-01-LABELS",
        "hypothesis_id": "H-RR17-01",
        "cell": "F1/s1001",
        "source_commit": SOURCE_COMMIT,
        "fold_manifest_sha256": FOLD1_SHA256,
        "recipe": BASE_RECIPE,
        "arms": list(ARMS),
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"fit spec changes the authorized {key}")
    runtime = spec.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("fit spec lacks measured runtime allocation")
    internal = runtime.get("internal_deadline_seconds")
    external = runtime.get("external_cancel_deadline_seconds_from_submit")
    if type(internal) is not int or type(external) is not int or internal <= 0:
        raise ValueError("measured runtime deadlines are unset or invalid")
    if external < internal + 60:
        raise ValueError("external deadline must leave at least 60 seconds for graceful collection")
    return internal, external


def training_args(arm: dict[str, object], split_path: Path) -> list[str]:
    """Build the exact canonical-trainer arguments for one fixed arm."""
    args = [
        "--name",
        str(arm["name"]),
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        str(arm["hidden"]),
        "--layers",
        str(arm["layers"]),
        "--dropout",
        "0",
        "--batch",
        "8",
        "--epochs",
        "6",
        "--train-seqs",
        "10607",
        "--train-source",
        "train+valid",
        "--valid-train-fraction",
        "1",
        "--valid-repeat",
        "1",
        "--valid-fold",
        "1",
        "--n-valid-folds",
        "4",
        "--full-valid",
        "--eval-source",
        "vfold",
        "--split-manifest",
        str(split_path),
        "--seed",
        "1001",
        "--batch-order",
        "seeded",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--precision",
        "fp32",
        "--lr",
        "0.0001",
        "--optim",
        "adamw",
        "--wd",
        "0.01",
        "--beta2",
        "0.999",
        "--sched",
        "onecycle",
        "--warmup-frac",
        "0.05",
        "--head-lr-mult",
        "1",
        "--clip",
        "1",
        "--chunk",
        "2000",
        "--grad-stats",
        "--fused-adam",
        "--sync-free",
        "--deterministic",
        "--no-eval-during-fit",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "5",
        "--save-training-state",
    ]
    return args


def prediction_args(arm: dict[str, object], split_path: Path) -> list[str]:
    """Build the canonical feature-only FP32 export for the fixed fold endpoint."""
    return [
        str(arm["name"]),
        "valid",
        "--epoch",
        "5",
        "--campaign-split-manifest",
        str(split_path),
        "--campaign-evaluation-only",
        "--inference-precision",
        "fp32",
    ]
