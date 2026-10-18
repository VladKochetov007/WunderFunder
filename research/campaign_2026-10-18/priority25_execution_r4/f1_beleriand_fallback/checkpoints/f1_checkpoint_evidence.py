from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from f1_checkpoint_common import (
    ARCHIVE_MANIFEST_SHA256,
    DATASET_LENGTH,
    FOLD1_SHA256,
    PLAN_SHA256,
    REGISTRY_SHA256,
    REMOTE_ROOT,
    SCHEDULE_UPDATES,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_FILE_COUNT,
    SOURCE_REVISION,
    mapping,
    read_json,
    require,
    sha256_file,
)


def validate_manifest_membership(
    manifest_path: Path, contract: Mapping[str, object], provenance: Mapping[str, object]
) -> dict[str, object]:
    """Recompute exact F1 fit and evaluation row orders from the sealed manifest."""
    import numpy as np

    require(manifest_path.stat().st_size <= 2_000_000, "fold manifest size bound")
    require(sha256_file(manifest_path) == FOLD1_SHA256, "fold1 manifest SHA256")
    manifest = read_json(manifest_path)
    require(
        (manifest["version"], manifest["fold"], manifest["n_folds"]) == (1, 1, 4), "fold identity"
    )
    fit = mapping(manifest["fit"], "manifest fit membership")
    evaluation = mapping(manifest["evaluation"], "manifest evaluation membership")
    train = cast(list[int], fit["train"])
    valid = cast(list[int], fit["valid"])
    eval_valid = cast(list[int], evaluation["valid"])
    require((len(train), len(valid), len(eval_valid)) == (10_533, 1_416, 457), "fold1 group counts")
    require(
        all(len(set(groups)) == len(groups) for groups in (train, valid, eval_valid)),
        "duplicate manifest group",
    )
    train_order = np.random.default_rng(1001).permutation(np.asarray(train, dtype="<i8"))[:10607]
    valid_order = np.asarray(valid, dtype="<i8")
    eval_order = np.random.default_rng(12345).permutation(np.asarray(eval_valid, dtype="<i8"))
    fit_rows = mapping(provenance["fit_row_groups"], "provenance fit row groups")
    require(fit_rows["train"] == train_order.tolist(), "ordered train membership")
    require(fit_rows["valid"] == valid_order.tolist(), "ordered fit-valid membership")
    require(
        provenance["evaluation_row_groups"] == eval_order.tolist(), "ordered evaluation membership"
    )
    components = mapping(manifest["sequences"], "manifest source-qualified sequences")
    component_of: dict[tuple[str, int], int] = {}
    for split in ("train", "valid"):
        rows = cast(list[dict[str, object]], components[split])
        for row in rows:
            key = (split, int(row["row_group"]))
            require(key not in component_of, "duplicate source-qualified sequence")
            component_of[key] = int(row["component"])
    fit_components = {component_of[("train", int(group))] for group in train}
    fit_components.update(component_of[("valid", int(group))] for group in valid)
    eval_components = {component_of[("valid", int(group))] for group in eval_valid}
    require(not fit_components.intersection(eval_components), "fit/evaluation component overlap")

    def identity(groups: object) -> dict[str, object]:
        values = np.asarray(groups, dtype="<i8")
        return {
            "count": len(values),
            "ordered_int64_le_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }

    expected_membership = {
        "train": identity(train_order),
        "valid": identity(valid_order),
        "valid_repeat": 1,
        "dataset_length": DATASET_LENGTH,
    }
    require(contract["fit_membership"] == expected_membership, "contract fit membership identity")
    require(
        provenance["fit_membership_identity"] == expected_membership,
        "provenance fit membership identity",
    )
    require(provenance.get("split_manifest_sha256") == FOLD1_SHA256, "provenance fold manifest SHA")
    return {
        "manifest_sha256": FOLD1_SHA256,
        "fit_train_groups": len(train),
        "fit_valid_groups": len(valid),
        "evaluation_groups": len(eval_valid),
        "evaluation_components": len(eval_components),
        "fit_membership_identity": expected_membership,
        "ordered_evaluation_membership_sha256": identity(eval_order)["ordered_int64_le_sha256"],
    }


def _verify_registry(contract: Mapping[str, object], plan: Mapping[str, object], path: Path) -> str:
    """Bind full parquet and target-cache identities to the sealed portable registry."""
    require(sha256_file(path) == REGISTRY_SHA256, "portable registry SHA256")
    registry = read_json(path)
    sources = mapping(registry["sources"], "portable registry sources")
    bindings = mapping(plan["data_bindings"], "plan data bindings")
    for split in ("train", "valid"):
        dataset = mapping(mapping(bindings["datasets"], "datasets")[split], f"{split} binding")
        entry = mapping(sources[f"datasets/{split}.parquet"], f"{split} registry entry")
        original = mapping(entry["original_identity"], f"{split} original identity")
        current = mapping(
            mapping(contract["datasets"], "contract datasets")[split], f"{split} contract identity"
        )
        require(
            (dataset["sha256"], dataset["bytes"]) == (entry["sha256"], original["bytes"]),
            f"{split} sealed registry identity",
        )
        require(
            (current["sha256"], current["bytes"]) == (entry["sha256"], original["bytes"]),
            f"{split} checkpoint dataset identity",
        )
    target_indexes = mapping(bindings["target_indexes"], "target index plan bindings")
    for split in ("train", "valid"):
        target = mapping(target_indexes[split], f"{split} target binding")
        entry = mapping(sources[f"cache/{split}_targets.npz"], f"{split} target registry entry")
        require(
            (target["sha256"], target["bytes"])
            == (entry["sha256"], entry["original_identity"]["bytes"]),
            f"{split} target cache identity",
        )
    return sha256_file(path)


def _verify_source_map(source_root: Path, contract: Mapping[str, object]) -> int:
    """Rehash every implementation module and trainer source in the checkpoint contract."""
    sources = mapping(contract["sources"], "checkpoint source map")
    require(len(sources) == SOURCE_FILE_COUNT, "source map file count")
    for relative, digest in sources.items():
        path = (
            (source_root / "scripts/train_rnn.py")
            if relative == "trainer"
            else (source_root / "src/wnn" / str(relative))
        )
        require(
            ".." not in Path(str(relative)).parts and path.is_file(), f"source file path {relative}"
        )
        require(sha256_file(path) == digest, f"source map hash {relative}")
    return len(sources)


def _verify_archive_manifest(path: Path) -> str:
    """Bind the source tree to the sealed code archive inventory."""
    require(sha256_file(path) == ARCHIVE_MANIFEST_SHA256, "portable archive manifest SHA256")
    manifest = read_json(path)
    require(manifest.get("source_revision") == "a266b036", "archive source revision")
    archives = cast(list[dict[str, object]], manifest["archives"])
    require(
        any(row.get("sha256") == SOURCE_ARCHIVE_SHA256 for row in archives),
        "canonical source archive SHA",
    )
    return sha256_file(path)


def _verify_contract(
    contract: Mapping[str, object], plan: Mapping[str, object], spec: Mapping[str, object]
) -> None:
    """Check the frozen cold labels-only F1 fit recipe and strict FP32 contract."""
    options = mapping(contract["options"], "contract options")
    arm = str(spec["arm"])
    expected = {
        "seed": 1001,
        "valid_fold": 1,
        "precision": "fp32",
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": spec["hidden"],
        "layers": spec["layers"],
        "batch": 8,
        "chunk": 2000,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "train_source": "train+valid",
        "train_seqs": 10607,
        "valid_train_fraction": 1.0,
        "valid_repeat": 1,
        "n_valid_folds": 4,
        "full_valid": True,
        "eval_source": "vfold",
        "no_eval_during_fit": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "batch_order": "seeded",
        "deterministic": True,
        "optim": "adamw",
        "lr": 0.0001,
        "beta2": 0.999,
        "wd": 0.01,
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "head_lr_mult": 1.0,
        "clip": 1.0,
        "distill": "",
        "propensity": "",
        "init": "",
        "final_distill_initializer": False,
        "inherit_init_preprocessing": False,
        "dropout": 0,
        "device": "cuda",
        "sync_free": True,
        "fused_adam": True,
        "grad_stats": True,
        "save_training_state": True,
        "split_manifest_sha256": FOLD1_SHA256,
    }
    for key, value in expected.items():
        require(options.get(key) == value, f"{arm} recipe option {key}")
    require(
        (contract["epochs"], contract["total_steps"]) == (6, SCHEDULE_UPDATES), "fixed schedule"
    )
    loader = mapping(contract["loader"], "training loader")
    require(
        (loader["batch_size"], loader["drop_last"], loader["dataset_length"])
        == (8, True, DATASET_LENGTH),
        "loader shape",
    )
    runtime = mapping(contract["runtime"], "runtime flags")
    require(
        runtime.get("matmul_tf32") is False and runtime.get("cudnn_tf32") is False,
        "strict FP32 runtime",
    )
    require(
        contract["teacher_predictions"] == {} and contract["consumed_caches"] == {},
        "teacher/cache inputs",
    )
    ancestors = mapping(contract["ancestors"], "cold-start ancestry")
    require(
        ancestors.get("initial_checkpoint_sha256") is None
        and ancestors.get("propensity_checkpoint_sha256") is None,
        "cold-start ancestry",
    )
    require(options.get("split_manifest") is None, "resume contract split path exclusion")
    require(plan["source_revision"] == SOURCE_REVISION[:8], "plan source revision")


def _verify_collection(
    snapshot: Path, receipt_path: Path, spec: Mapping[str, object], plan_path: Path
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Check exact phase artifacts and copied attempt/log evidence."""
    receipt = read_json(receipt_path)
    run = str(spec["run"])
    require(
        (receipt.get("arm"), receipt.get("phase"), receipt.get("run"), receipt.get("attempt"))
        == (spec["arm"], spec["phase"], run, spec["attempt"]),
        "collection identity",
    )
    require(
        receipt.get("source_directory") == str(REMOTE_ROOT / "data/artifacts/runs" / run),
        "collection source directory",
    )
    require(receipt.get("snapshot_directory") == str(snapshot), "collection snapshot directory")
    require(
        receipt.get("plan_sha256") == PLAN_SHA256 and sha256_file(plan_path) == PLAN_SHA256,
        "collection plan SHA",
    )
    require(
        receipt.get("source_stats_before") == receipt.get("source_stats_after"),
        "source generation stability",
    )
    require(
        receipt.get("collector_dependencies")
        == {
            "f1_checkpoint_common.py": sha256_file(
                Path(__file__).with_name("f1_checkpoint_common.py")
            )
        },
        "collection helper dependency hashes",
    )
    require(
        receipt.get("collector_sha256")
        == sha256_file(Path(__file__).with_name("collect_f1_checkpoint.py")),
        "collection helper SHA256",
    )
    expected_names = set(cast(tuple[str, ...], spec["artifacts"]))
    files = {"attempt.json", "training.log"}
    require(
        set(mapping(receipt["source_stats_before"], "source stats")) == expected_names | files,
        "source stats file set",
    )
    artifacts = mapping(receipt["artifacts"], "collection artifact map")
    require(set(artifacts) == expected_names, "collection artifact names")
    result: dict[str, dict[str, object]] = {}
    for name in expected_names:
        entry = mapping(artifacts[name], f"artifact {name}")
        path = snapshot / name
        require(path.is_file() and not path.is_symlink(), f"artifact {name} file type")
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        require(dict(entry) == actual, f"artifact {name} integrity")
        result[name] = actual
    for field, name in (("attempt_evidence", "attempt.json"), ("log_evidence", "training.log")):
        evidence = mapping(receipt[field], field)
        path = snapshot / name
        require(path.is_file() and not path.is_symlink(), f"{field} file type")
        require(
            evidence.get("path") == str(path)
            and evidence.get("bytes") == path.stat().st_size
            and evidence.get("sha256") == sha256_file(path),
            f"{field} integrity",
        )
    attempt = mapping(receipt["attempt_evidence"], "attempt evidence")
    require(attempt.get("id") == spec["attempt"], "collected attempt id")
    if spec["phase"] != "epoch0":
        require(
            (attempt.get("status"), attempt.get("exit_status")) == ("finished", 0),
            "collected terminal status",
        )
    return result, receipt
