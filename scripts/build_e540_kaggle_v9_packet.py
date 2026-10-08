#!/usr/bin/env python3
"""Build a private V9 packet that reuses V8 linear and fits only PLE."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_assets import prepare as prepare_private_assets  # noqa: E402
from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256, sha256_file  # noqa: E402

LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540"
PACKET_ID = "E540_FP32_V9_REUSE_CONTROL"
KERNEL_DATASET = "vladkochetov07/e540-v9-ple-reuse-control-20261008"
CONTROL_DATASET = "vladkochetov07/e540-v9-control-assets-20261008"
SPEC_NAME = "e540-v9-ple-reuse-control-20261008"
AUTH_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V9_ROOT_CONDITIONAL_AUTHORIZATION.json"
)
SUBSET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_FIT_SUBSET_V5.json"
)
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
ASSET_MANIFEST_RELATIVE = Path(
    "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
)
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
V8_TERMINAL_RECEIPT_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V8_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json"
)
V8_REVIEW_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V8_ROOT_INDEPENDENT_SOURCE_REVIEW_20261008_R1.json"
)
V8_AUTH_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V8_ROOT_CONDITIONAL_AUTHORIZATION.json"
)
EXPECTED_V8 = {
    "source_commit": "942178ef647fd4f49c90d1f6846dbd927cfab4fa",
    "source_review_sha256": "34c2d33457035bf338d3191ae01b590f8593c9b31f226f99902558023a4052b4",
    "worker_sha256": "2a76f211620640b896f404d69817f991c7c35f7ec3c75129878aa388593efe39",
    "code_manifest_sha256": "c1396bb5fd29a0d4c85216bdfa63e3b9e84204b3369aeb98cd1634a80e1ed8fc",
    "recipe_sha256": "00f5ddfaa7d97f147d97d7cc4d7d2ec15a976c052985be12026686cfb266dedb",
    "conditional_authorization_sha256": "4823742074306346b934ac0474e029553318ad40df8ee2c20ecc79d55e0444a0",
    "terminal_receipt_sha256": "93a6956dffc9f12fd480b0e9374be9e67654767456f6a7edbfe51e4f680ad5f9",
    "result_sha256": "88a415b843966c3f8f1ef4e4551759f9698bdc5658479cfb6ca1c0814dc16493",
    "provenance_sha256": "ef342abcc11b94eabed66bd4d1e29009fc645079bb17bb95f5f63d83b7e82794",
    "args_sha256": "931dccef9138bf3a41fc18c27b24af5202a3c842fc017f95af84d6b4f3e169d4",
    "model_sha256": "73880506cf241502d961ce2cba20819085056fef05f5ea8e56d586535bbfb946",
    "model_ep0_sha256": "17c91be768a7f41cd4c6ae0b2bbce26bcf7d6fbf5e4f6509e38c4be6bd44403a",
    "history_sha256": "679b6c7bf3d8b93027c6076ab8bbe6216a9c76d94885a7a06ffeaa64929aaba0",
    "subset_manifest_sha256": "ce84816336bc457d487cbddf2f7214ef4d8b0b02e83f9c8e60921b9ce2374d4d",
    "fit_members_sha256": "28747d40caefee2b559d16d7d05ec57e7c4b5079965acb2e590fa0cefec9580e",
    "split_manifest_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
}
EXTERNAL_NAMES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
)
CONTROL_NAMES = ("model.pt", "model_ep0.pt", "provenance.json", "history.json")
ARTIFACTS = [
    "e540_cohort",
    "artifacts/runs/e540v8-linear-s540-fp32",
    "artifacts/runs/e540v9-ple-s540-fp32",
    "cache/pred_e540v8-linear-s540-fp32-eval_valid.npy",
    "cache/pred_e540v8-linear-s540-fp32-eval_valid.json",
    "cache/pred_e540v9-ple-s540-fp32-eval_valid.npy",
    "cache/pred_e540v9-ple-s540-fp32-eval_valid.json",
]


def write_json(path: Path, value: object) -> None:
    """Write deterministic, finite packet metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def descriptor(path: Path, logical_name: str | None = None) -> dict[str, object]:
    """Hash one small local source or metadata file in bounded memory."""
    return {
        "path": logical_name or path.relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_base_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_paths() -> list[Path]:
    return [
        Path("scripts/build_e540_kaggle_v9_packet.py"),
        Path("scripts/e540_cli_preflight.py"),
        Path("scripts/e540_kaggle_v9_worker.py"),
        Path("scripts/kaggle_assets.py"),
        Path("scripts/kaggle_common.py"),
        Path("scripts/kaggle_bundle.py"),
        Path("scripts/kaggle_gpu_runner.py"),
        Path("scripts/kaggle_kernel.py"),
        Path("scripts/predict_run.py"),
        Path("scripts/prepare_e540_fit_subset.py"),
        Path("scripts/train_rnn.py"),
        Path("src/wnn/__init__.py"),
        *sorted(
            path.relative_to(ROOT)
            for path in (ROOT / "src/wnn").rglob("*.py")
            if path.name != "__init__.py"
        ),
    ]


def _local_input_paths() -> list[Path]:
    names = [
        SPLIT_RELATIVE,
        PORTABLE_RELATIVE,
        ASSET_MANIFEST_RELATIVE,
        Path("cache/chains.npz"),
        Path("cache/chain_folds.npz"),
        SUBSET_RELATIVE,
        V8_TERMINAL_RECEIPT_RELATIVE,
        V8_REVIEW_RELATIVE,
        V8_AUTH_RELATIVE,
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V8_FP32.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_FP32_V8.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/README_V8.md"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V9_FP32_REUSE_CONTROL.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_FP32_V9.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/README_V9.md"
        ),
        AUTH_RELATIVE,
    ]
    return names


def _expected_external(portable: dict[str, Any]) -> list[dict[str, object]]:
    sources = portable["sources"]
    return [
        {
            "path": name,
            "size": int(sources[name]["original_identity"]["bytes"]),
            "sha256": str(sources[name]["sha256"]),
        }
        for name in EXTERNAL_NAMES
    ]


def _validate_v8_identity(
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review_path = ROOT / V8_REVIEW_RELATIVE
    authorization_path = ROOT / V8_AUTH_RELATIVE
    terminal_path = ROOT / V8_TERMINAL_RECEIPT_RELATIVE
    review = json.loads(review_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        sha256_file(review_path) != EXPECTED_V8["source_review_sha256"]
        or review.get("status") != "source_approved_conditional_fit"
        or review.get("source_commit") != EXPECTED_V8["source_commit"]
        or review.get("worker_sha256") != EXPECTED_V8["worker_sha256"]
        or sha256_file(authorization_path) != EXPECTED_V8["conditional_authorization_sha256"]
        or authorization.get("status") != "approved"
        or authorization.get("authorized") is not True
        or authorization.get("code_manifest_sha256") != EXPECTED_V8["code_manifest_sha256"]
        or authorization.get("recipe_sha256") != EXPECTED_V8["recipe_sha256"]
        or sha256_file(ROOT / V8_TERMINAL_RECEIPT_RELATIVE)
        != EXPECTED_V8["terminal_receipt_sha256"]
        or sha256_file(result_path) != EXPECTED_V8["result_sha256"]
    ):
        raise ValueError("V8 source review, authorization, terminal receipt, or result changed")
    if (
        terminal.get("kernel", {}).get("provider_terminal_status") != "KernelWorkerStatus.ERROR"
        or terminal.get("kernel", {}).get("run_status") != "failed"
        or terminal.get("failure_diagnosis", {}).get("worker_error")
        != "ValueError: checkpoint files do not match trainer provenance hashes"
        or terminal.get("artifact_collection", {}).get("all_declared_artifacts_verified")
        is not True
        or terminal.get("arm_outcomes", {}).get("linear", {}).get("checkpoint_produced") is not True
        or terminal.get("arm_outcomes", {}).get("ple", {}).get("fit_started") is not False
        or terminal.get("arm_outcomes", {}).get("grader_run") is not False
    ):
        raise ValueError("V8 terminal receipt does not prove the expected linear-only outcome")
    if (
        result.get("status") != "failed"
        or result.get("returncode") != 1
        or result.get("device", {}).get("name") != "Tesla T4"
    ):
        raise ValueError("V8 worker result differs from the downloaded terminal receipt")
    return review, authorization, terminal


def _validate_control_outputs(
    control_dir: Path, result_path: Path, subset: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_manifest = {item["path"]: item for item in result.get("artifacts", {}).get("files", [])}
    provenance_path = control_dir / "provenance.json"
    model_path = control_dir / "model.pt"
    epoch0_path = control_dir / "model_ep0.pt"
    history_path = control_dir / "history.json"
    required = {
        "model.pt": model_path,
        "model_ep0.pt": epoch0_path,
        "provenance.json": provenance_path,
        "history.json": history_path,
    }
    for filename, path in required.items():
        relative = f"artifacts/runs/e540v8-linear-s540-fp32/{filename}"
        entry = result_manifest.get(relative)
        if (
            not isinstance(entry, dict)
            or path.stat().st_size != entry.get("size")
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(
                f"V8 linear output differs from its terminal artifact manifest: {filename}"
            )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    args = provenance.get("args")
    numeric = provenance.get("numeric_preprocessing")
    if not isinstance(args, dict) or not isinstance(numeric, dict):
        raise ValueError("V8 linear external provenance is malformed")
    if (
        sha256_file(provenance_path) != EXPECTED_V8["provenance_sha256"]
        or canonical_sha256(args) != EXPECTED_V8["args_sha256"]
        or provenance.get("checkpoint_sha256")
        != {
            "model.pt": EXPECTED_V8["model_sha256"],
            "model_ep0.pt": EXPECTED_V8["model_ep0_sha256"],
        }
        or provenance.get("fit_subset_manifest_sha256") != EXPECTED_V8["subset_manifest_sha256"]
        or provenance.get("fit_subset_members_sha256") != EXPECTED_V8["fit_members_sha256"]
        or provenance.get("split_manifest_sha256") != EXPECTED_V8["split_manifest_sha256"]
        or provenance.get("fit_row_groups") != subset.get("selected_groups")
        or not isinstance(provenance.get("evaluation_row_groups"), list)
        or len(provenance["evaluation_row_groups"]) != len(subset.get("evaluation_groups", []))
        or len(set(provenance["evaluation_row_groups"])) != len(provenance["evaluation_row_groups"])
        or set(provenance["evaluation_row_groups"]) != set(subset.get("evaluation_groups", []))
        or args.get("name") != "e540v8-linear-s540-fp32"
        or args.get("seed") != 540
        or args.get("numeric_map") != "linear"
        or args.get("precision") != "fp32"
        or args.get("loss") != "wmse"
        or args.get("loss_norm") != "chunk"
        or args.get("epochs") != 1
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 0
        or numeric.get("mode") != "linear"
        or numeric.get("fit_members") != subset.get("selected_groups")
    ):
        raise ValueError("V8 control provenance does not match the frozen V8 recipe and V9 subset")
    if (
        provenance.get("trainer_source_sha256")
        != "31b72e53ae81a771d6cff42d8a48a4c06cdc523fabc8c7d00e51d20710942853"
        or provenance.get("model_source_sha256")
        != "874fbf1ab5756c7dabe9e388d9d619c0a29c218d28db36a7f8747957df453181"
        or provenance.get("run_identity_source_sha256")
        != "7033ea007b9c7dcf498d3ab1e31aaeacb5725b7009fe719e9071c05b99272a76"
        or numeric.get("module_source_sha256")
        != "229a7e0f032cd65bbf6700fa9372c08016b20e4719cb2f9eddd540ff8c468720"
    ):
        raise ValueError("V8 control training source hashes are not the reviewed V8 files")
    updates = provenance.get("update_diagnostics")
    if not isinstance(updates, list) or len(updates) != 1 or not isinstance(updates[0], dict):
        raise ValueError("V8 control provenance lacks its epoch-0 update record")
    update = updates[0]
    epoch = update.get("epoch")
    cumulative = provenance.get("cumulative_updates")
    grad_stats = update.get("grad_stats")
    if (
        not isinstance(epoch, dict)
        or not isinstance(cumulative, dict)
        or not isinstance(grad_stats, dict)
        or epoch.get("optimizer_attempts") != 72_996
        or epoch.get("optimizer_completed_updates") != 72_996
        or epoch.get("amp_skipped_updates") != 0
        or epoch.get("completed_batches") != 924
        or cumulative.get("optimizer_completed_updates") != 72_996
        or grad_stats.get("steps") != 72_996
        or grad_stats.get("nonfinite_steps") != 0
        or provenance.get("selected_epoch") != 0
        or len(history) != 1
        or history[0].get("epoch") != 0
        or history[0].get("evaluation") != "withheld"
        or history[0].get("update_diagnostics")
        != {
            key: update[key]
            for key in (
                "epoch",
                "cumulative",
                "sample_order_identity",
                "normalization_identity",
                "grad_stats",
            )
        }
    ):
        raise ValueError("V8 control update, finiteness or fixed-epoch evidence is incomplete")
    files = {
        "model.pt": {"size_bytes": model_path.stat().st_size, "sha256": sha256_file(model_path)},
        "model_ep0.pt": {
            "size_bytes": epoch0_path.stat().st_size,
            "sha256": sha256_file(epoch0_path),
        },
        "provenance.json": {
            "size_bytes": provenance_path.stat().st_size,
            "sha256": sha256_file(provenance_path),
        },
        "history.json": {
            "size_bytes": history_path.stat().st_size,
            "sha256": sha256_file(history_path),
        },
    }
    control_manifest = {
        "schema": "wnn.e540-v8-linear-control.v1",
        "producer": {
            "packet_id": "E540_FP32_V8",
            "source_commit": EXPECTED_V8["source_commit"],
            "source_review_sha256": EXPECTED_V8["source_review_sha256"],
            "worker_sha256": EXPECTED_V8["worker_sha256"],
            "code_manifest_sha256": EXPECTED_V8["code_manifest_sha256"],
            "recipe_sha256": EXPECTED_V8["recipe_sha256"],
            "conditional_authorization_sha256": EXPECTED_V8["conditional_authorization_sha256"],
            "terminal_receipt_sha256": EXPECTED_V8["terminal_receipt_sha256"],
            "result_sha256": EXPECTED_V8["result_sha256"],
            "provenance_sha256": EXPECTED_V8["provenance_sha256"],
        },
        "run_name": args["name"],
        "args_sha256": canonical_sha256(args),
        "subset_manifest_sha256": provenance["fit_subset_manifest_sha256"],
        "fit_members_sha256": provenance["fit_subset_members_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "fit_group_counts": {
            key: len(value) for key, value in provenance["fit_row_groups"].items()
        },
        "evaluation_group_count": len(provenance["evaluation_row_groups"]),
        "training_source_hashes": {
            "scripts/train_rnn.py": provenance["trainer_source_sha256"],
            "src/wnn/nn.py": provenance["model_source_sha256"],
            "src/wnn/run_identity.py": provenance["run_identity_source_sha256"],
            "src/wnn/numeric_ple.py": numeric["module_source_sha256"],
        },
        "files": files,
        "update_diagnostics": {
            "epoch": epoch,
            "cumulative": cumulative,
            "sample_order_identity": update["sample_order_identity"],
            "normalization_identity": update["normalization_identity"],
            "grad_stats": grad_stats,
        },
    }
    return control_manifest, provenance, result


def _ensure_pending_authorization(
    path: Path,
    *,
    code_manifest_sha256: str,
    recipe_sha256: str,
    subset_sha256: str,
    control_manifest_sha256: str,
    control_args_sha256: str,
    control_model_sha256: str,
    control_epoch0_sha256: str,
    control_provenance_sha256: str,
) -> dict[str, object]:
    """Create a fail-closed V9 placeholder; only root can approve this exact packet."""
    expected = {
        "schema": "wnn.e540-conditional-fit-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
        "conditional_fit_after_profile_gate": True,
        "packet_id": PACKET_ID,
        "code_manifest_sha256": code_manifest_sha256,
        "recipe_sha256": recipe_sha256,
        "subset_manifest_sha256": subset_sha256,
        "kernel_process_wall_seconds_max": 6000,
        "worker_wall_seconds_max": 5200,
        "control_manifest_sha256": control_manifest_sha256,
        "control_args_sha256": control_args_sha256,
        "control_model_sha256": control_model_sha256,
        "control_epoch0_sha256": control_epoch0_sha256,
        "control_provenance_sha256": control_provenance_sha256,
        "authorization_id": None,
        "provider_launch_authorized": False,
        "profile_gate": {
            "profile_arm": "ple",
            "precision": "fp32",
            "grad_scaler_enabled": False,
            "optimizer_completed_updates": 316,
            "gradient_stats_steps": 316,
            "gradient_stats_nonfinite_steps": 0,
            "amp_init_scale_applied": False,
            "allocator_reserved_bytes_max": 4_300_000_000,
            "projected_worker_seconds_max": 5200,
            "control_fit": "already completed in V8; validation and export only",
        },
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "approved" or existing.get("authorized") is True:
            for key, value in expected.items():
                if (
                    key not in {"status", "authorized", "authorization_id"}
                    and existing.get(key) != value
                ):
                    raise ValueError(f"existing approved V9 root authorization differs in {key}")
            return existing
        write_json(path, expected)
        return expected
    write_json(path, expected)
    return expected


def build_packet(output_dir: Path, control_run_dir: Path) -> dict[str, object]:
    """Stage private control assets and seal a V9 source packet without data/fit reads."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V9 output directory: {output_dir}")
    subset_path = ROOT / SUBSET_RELATIVE
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    subset_sha = sha256_file(subset_path)
    if (
        subset_sha != EXPECTED_V8["subset_manifest_sha256"]
        or subset.get("selected_component_count") != 512
        or subset.get("evaluation_group_count") != 468
        or subset.get("selected_members_sha256") != EXPECTED_V8["fit_members_sha256"]
    ):
        raise ValueError("frozen V5 subset identity or 512/468 membership changed")
    control_run_dir = control_run_dir.resolve(strict=True)
    result_path = control_run_dir / "result.json"
    review, v8_authorization, terminal = _validate_v8_identity(result_path)
    del review, v8_authorization, terminal

    portable_path = ROOT / PORTABLE_RELATIVE
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    external_descriptors = _expected_external(portable)
    v5_packet_path = LINEAGE / "kaggle_v5_final_20261008/E540_KAGGLE_PACKET_V5.json"
    v5_packet = json.loads(v5_packet_path.read_text(encoding="utf-8"))
    if external_descriptors != v5_packet["external_input_files"]:
        raise ValueError("V9 mounted data identities differ from the frozen V5/V8 identities")
    asset_manifest_path = ROOT / ASSET_MANIFEST_RELATIVE
    if sha256_file(asset_manifest_path) != ASSET_MANIFEST_SHA256:
        raise ValueError("authorized private fold0 assets manifest changed")
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if asset_manifest.get("dataset") != ASSET_DATASET or asset_manifest.get("private") is not True:
        raise ValueError("existing private assets manifest identity or privacy flag changed")
    asset_entries = {entry["path"]: entry for entry in asset_manifest["files"]}
    for name in ("cache/train_targets.npz", "cache/valid_targets.npz"):
        expected = next(item for item in external_descriptors if item["path"] == name)
        entry = asset_entries.get(name)
        if (
            entry is None
            or entry.get("sha256") != expected["sha256"]
            or entry.get("size") != expected["size"]
        ):
            raise ValueError(f"private target asset descriptor changed: {name}")
    for name in ("cache/chains.npz", "cache/chain_folds.npz"):
        path = ROOT / name
        if path.stat().st_size > 2_000_000:
            raise ValueError(f"refusing unexpected large feature-only graph cache: {name}")
        prior = next(item for item in v5_packet["local_input_files"] if item["path"] == name)
        if path.stat().st_size != prior["size"] or sha256_file(path) != prior["sha256"]:
            raise ValueError(f"feature-only graph cache changed from V5: {name}")

    control_dir = control_run_dir / "artifacts" / "artifacts" / "runs" / "e540v8-linear-s540-fp32"
    control_manifest, control_provenance, result = _validate_control_outputs(
        control_dir, result_path, subset
    )
    del control_provenance

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    control_stage = output_dir / "control_asset_stage"
    control_stage.mkdir()
    for filename in CONTROL_NAMES:
        shutil.copyfile(control_dir / filename, control_stage / filename)
    write_json(control_stage / "control-manifest.json", control_manifest)
    control_manifest_sha = sha256_file(control_stage / "control-manifest.json")
    control_assets_dir = output_dir / "control_assets_upload"
    prepare_private_assets(
        control_stage,
        [*CONTROL_NAMES, "control-manifest.json"],
        control_assets_dir,
        CONTROL_DATASET,
    )
    uploaded_assets = json.loads((control_assets_dir / "assets-manifest.json").read_text())
    uploaded_entries = {entry["path"]: entry for entry in uploaded_assets["files"]}
    control_external_descriptors = []
    external_args = [
        "datasets/train.parquet=train.parquet",
        "datasets/valid.parquet=valid.parquet",
        f"cache/train_targets.npz={ASSET_DATASET}:{asset_entries['cache/train_targets.npz']['filename']}",
        f"cache/valid_targets.npz={ASSET_DATASET}:{asset_entries['cache/valid_targets.npz']['filename']}",
    ]
    control_logical_names = {
        filename: f"e540_control/{filename}"
        for filename in (*CONTROL_NAMES, "control-manifest.json")
    }
    for filename, logical in control_logical_names.items():
        entry = uploaded_entries[filename]
        control_external_descriptors.append(
            {"path": logical, "size": entry["size"], "sha256": entry["sha256"]}
        )
        external_args.append(f"{logical}={CONTROL_DATASET}:{entry['filename']}")

    source_paths = _source_paths()
    source_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in source_paths}
    code_descriptors = [descriptor(ROOT / path) for path in source_paths]
    code_manifest_sha = canonical_sha256(sorted(code_descriptors, key=lambda item: item["path"]))
    recipe_path = LINEAGE / "RECIPE_V9_FP32_REUSE_CONTROL.json"
    recipe_sha = sha256_file(recipe_path)
    if (
        source_hashes.get("scripts/train_rnn.py")
        != control_manifest["training_source_hashes"]["scripts/train_rnn.py"]
    ):
        raise ValueError("V9 train_rnn source differs from the reused V8 control trainer")
    authorization = _ensure_pending_authorization(
        ROOT / AUTH_RELATIVE,
        code_manifest_sha256=code_manifest_sha,
        recipe_sha256=recipe_sha,
        subset_sha256=subset_sha,
        control_manifest_sha256=control_manifest_sha,
        control_args_sha256=control_manifest["args_sha256"],
        control_model_sha256=control_manifest["files"]["model.pt"]["sha256"],
        control_epoch0_sha256=control_manifest["files"]["model_ep0.pt"]["sha256"],
        control_provenance_sha256=control_manifest["files"]["provenance.json"]["sha256"],
    )
    local_paths = _local_input_paths()
    local_descriptors = [descriptor(ROOT / path) for path in local_paths]
    if authorization.get("status") == "approved":
        from e540_kaggle_v9_worker import validate_conditional_authorization

        validate_conditional_authorization(
            ROOT / AUTH_RELATIVE,
            {
                "packet_id": PACKET_ID,
                "code_manifest_sha256": code_manifest_sha,
                "recipe_sha256": recipe_sha,
                "subset_manifest_sha256": subset_sha,
                "kernel_process_wall_seconds_max": 6000,
                "worker_wall_seconds_max": 5200,
                "control_manifest_sha256": control_manifest_sha,
                "control_args_sha256": control_manifest["args_sha256"],
                "control_model_sha256": control_manifest["files"]["model.pt"]["sha256"],
                "control_epoch0_sha256": control_manifest["files"]["model_ep0.pt"]["sha256"],
                "control_provenance_sha256": control_manifest["files"]["provenance.json"]["sha256"],
            },
        )

    all_external = [*external_descriptors, *control_external_descriptors]
    spec = {
        "version": 1,
        "name": SPEC_NAME,
        "kind": "single_profile_then_conditional_candidate_fit_and_paired_export",
        "argv": ["{python}", "scripts/e540_kaggle_v9_worker.py"],
        "code": code_descriptors,
        "code_sha256": code_manifest_sha,
        "inputs": [*local_descriptors, *all_external],
        "artifacts": ARTIFACTS,
        "limits": {
            "cpus": 2,
            "ram_bytes": 8 * 1024**3,
            "timeout_seconds": 5200,
            "gpu_memory_bytes": 4_300_000_000,
            "gpu_stop_bytes": 5_000_000_000,
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "WNN_THREADS": "2",
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_PORTABLE_DATA_MANIFEST": PORTABLE_RELATIVE.as_posix(),
            "WNN_DEFERRED_IDENTITY_PATHS": ",".join(EXTERNAL_NAMES),
            "WNN_E540_PACKET_ID": PACKET_ID,
            "WNN_E540_AUTHORIZATION_PATH": AUTH_RELATIVE.as_posix(),
            "WNN_E540_CODE_MANIFEST_SHA256": code_manifest_sha,
            "WNN_E540_RECIPE_SHA256": recipe_sha,
            "WNN_E540_SUBSET_SHA256": subset_sha,
            "WNN_E540_KERNEL_WALL_SECONDS": "6000",
            "WNN_E540_WORKER_WALL_SECONDS": "5200",
            "WNN_E540_CONTROL_MANIFEST_SHA256": control_manifest_sha,
            "WNN_E540_CONTROL_ARGS_SHA256": control_manifest["args_sha256"],
            "WNN_E540_CONTROL_MODEL_SHA256": control_manifest["files"]["model.pt"]["sha256"],
            "WNN_E540_CONTROL_EPOCH0_SHA256": control_manifest["files"]["model_ep0.pt"]["sha256"],
            "WNN_E540_CONTROL_PROVENANCE_SHA256": control_manifest["files"]["provenance.json"][
                "sha256"
            ],
        },
    }
    spec_path = output_dir / f"{SPEC_NAME}.spec.json"
    write_json(spec_path, spec)
    build_bundle(
        ROOT,
        [spec_path],
        output_dir / "bundle",
        KERNEL_DATASET,
        external_args,
        [ASSET_DATASET, CONTROL_DATASET],
    )
    packet = {
        "schema": "wnn.e540-kaggle-cohort.v4",
        "packet_id": PACKET_ID,
        "status": "prepared_pending_independent_root_review_and_conditional_authorization",
        "fit_admitted": False,
        "provider_launch_authorized": False,
        "grade_owner": "root independent grader",
        "generated_from": {
            "source_checkout": str(ROOT),
            "git_base_commit": _git_base_commit(),
            "source_files_are_hash_bound": True,
            "source_sha256": source_hashes,
            "code_manifest_sha256": code_manifest_sha,
        },
        "recipe": {
            "path": "RECIPE_V9_FP32_REUSE_CONTROL.json",
            "sha256": recipe_sha,
        },
        "control": {
            "source_packet_id": "E540_FP32_V8",
            "control_manifest_sha256": control_manifest_sha,
            "args_sha256": control_manifest["args_sha256"],
            "model_sha256": control_manifest["files"]["model.pt"]["sha256"],
            "model_ep0_sha256": control_manifest["files"]["model_ep0.pt"]["sha256"],
            "provenance_sha256": control_manifest["files"]["provenance.json"]["sha256"],
            "asset_dataset": CONTROL_DATASET,
            "asset_manifest_sha256": sha256_file(control_assets_dir / "assets-manifest.json"),
            "private_assets_upload_directory": str(control_assets_dir),
            "linear_refit": False,
        },
        "data": {
            "split_manifest_sha256": sha256_file(ROOT / SPLIT_RELATIVE),
            "portable_registry_sha256": sha256_file(portable_path),
            "private_targets_asset_manifest_sha256": ASSET_MANIFEST_SHA256,
            "mounted_source_identities": external_descriptors,
            "control_assets": control_external_descriptors,
            "subset_manifest_sha256": subset_sha,
            "selected_component_count": 512,
            "evaluation_group_count": 468,
            "no_local_full_parquet_footer_or_target_npz_read": True,
        },
        "conditional_authorization": {
            "receipt": AUTH_RELATIVE.as_posix(),
            "sha256": sha256_file(ROOT / AUTH_RELATIVE),
            "status": authorization["status"],
            "owner": "root",
            "required_before_mounted_training_data_hashing": True,
            "conditional_fit_after_ple_profile_gate": True,
            "bound_control_manifest_sha256": control_manifest_sha,
            "bound_control_args_sha256": control_manifest["args_sha256"],
        },
        "execution": {
            "kernel_process_wall_seconds_max": 6000,
            "worker_wall_seconds_max": 5200,
            "runner_finalization_reserve_seconds": 300,
            "source_and_asset_uploads_private": True,
            "profile_arms": ["ple"],
            "fit_arms": ["ple"],
            "prediction_arms": ["reused_v8_linear", "fresh_v9_ple"],
            "provider_mutations_performed": 0,
            "kaggle_push_performed": False,
        },
        "bundle": {
            "archive_sha256": sha256_file(output_dir / "bundle/bundle.wnn"),
            "manifest_sha256": sha256_file(output_dir / "bundle/bundle-manifest.json"),
            "bootstrap_sha256": sha256_file(output_dir / "bundle/kernel/kernel.py"),
            "kernel_metadata_sha256": sha256_file(
                output_dir / "bundle/kernel/kernel-metadata.json"
            ),
        },
    }
    write_json(output_dir / "E540_KAGGLE_PACKET_V9.json", packet)
    shutil.rmtree(control_stage)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = build_packet(args.output_dir, args.control_run_dir)
    print(
        json.dumps(
            {
                "packet": str(args.output_dir / "E540_KAGGLE_PACKET_V9.json"),
                "bundle_archive_sha256": packet["bundle"]["archive_sha256"],
                "control_manifest_sha256": packet["control"]["control_manifest_sha256"],
                "authorization_status": packet["conditional_authorization"]["status"],
                "provider_mutations_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
