"""Exact fit, feature-export, and W18 cache stages for the finite final-submit graph."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import final_submit_chain_contract as contract
import final_submit_chain_runtime as runtime

from wnn.final_submit_manifest import FinalFitMembership
from wnn.training.final_submit_bridge_cache import validate_final_distillation_cache

FIT_CAP_SECONDS = 21_600
NODE_CAP_SECONDS = 43_200


def build_fit_argv(
    *,
    node: Mapping[str, Any],
    python: Path,
    source_root: Path,
    data_root: Path,
    plan: Mapping[str, Any],
    profile_path: Path,
) -> list[str]:
    """Build a fit command from only the exact plan argv or fixed native recipe."""
    node_id = str(node["node_id"])
    if node_id == "source4_lstm192x1":
        native = plan["optional_source4_lstm"]
        config = Path(str(native["config_path"]))
        return [
            str(python),
            str(source_root / "scripts/train_final_lstm.py"),
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--output",
            str(data_root / "artifacts/runs" / str(node["run"])),
            "--final-fit-manifest",
            str(node["membership_path"]),
            "--final-fit-source-validation-receipt",
            str(node["audit_path"]),
            "--final-fit-source-validation-sha256",
            contract.SOURCE_AUDIT_SHA256,
            "--profile-receipt",
            str(profile_path),
        ]
    return [str(python), str(source_root / "scripts/train_rnn.py"), *node["args"]]


def execute_node(
    *,
    node: Mapping[str, Any],
    plan: Mapping[str, Any],
    membership: FinalFitMembership,
    source_root: Path,
    data_root: Path,
    controller_root: Path,
    python: Path,
    gpu_uuid: str,
    gpu_visible: str,
    profile_path: Path,
    trial: int,
    progress: Callable[[str, Mapping[str, object]], None],
) -> dict[str, object]:
    """Run a fresh fixed-endpoint fit, verify its checkpoint, then export teacher features."""
    node_id, run = str(node["node_id"]), str(node["run"])
    output_dir = data_root / "artifacts/runs" / run
    if output_dir.exists():
        raise FileExistsError(f"fresh final-fit output already exists: {output_dir}")
    membership_path, audit_path = Path(str(node["membership_path"])), Path(str(node["audit_path"]))
    fit_node = dict(node) | {"membership_path": str(membership_path), "audit_path": str(audit_path)}
    argv = build_fit_argv(
        node=fit_node,
        python=python,
        source_root=source_root,
        data_root=data_root,
        plan=plan,
        profile_path=profile_path,
    )
    fit_receipt = controller_root / "receipts/fits" / f"{node_id}.json"
    fit_log = controller_root / "logs/fits" / f"{node_id}.log"
    fit_start = time.time()
    fit_process = runtime.run_stage(
        argv=[
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=10s",
            f"{FIT_CAP_SECONDS - 10}s",
            *argv,
        ],
        source_root=source_root,
        data_root=data_root,
        controller_root=controller_root,
        gpu_uuid=gpu_uuid,
        gpu_visible=gpu_visible,
        stage=1,
        trial=trial,
        timeout=FIT_CAP_SECONDS,
        output_dir=output_dir,
        receipt=fit_receipt,
        log=fit_log,
        progress=lambda value: progress("fit", value),
    )
    checkpoint = runtime.validate_fit_checkpoint(
        node=node, output_dir=output_dir, membership=membership
    )
    checkpoint_hash = contract.sha256_file(checkpoint)
    result: dict[str, object] = {
        "status": "fit_complete",
        "node_id": node_id,
        "run": run,
        "fit_started_at_epoch": fit_start,
        "fit_finished_at_epoch": time.time(),
        "fit_process_receipt": str(fit_receipt),
        "fit_process": fit_process,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "exports": {},
    }
    if node.get("export_train") is True:
        for split in ("train", "valid"):
            elapsed = time.time() - fit_start
            remaining = int(NODE_CAP_SECONDS - elapsed)
            timeout = min(FIT_CAP_SECONDS, remaining)
            if timeout <= 20:
                raise TimeoutError(f"node wall budget is exhausted before {run}/{split} export")
            prediction_path = data_root / "cache" / f"pred_{run}_{split}.npy"
            sidecar_path = prediction_path.with_suffix(".json")
            if prediction_path.exists() or sidecar_path.exists():
                raise FileExistsError(f"final-fit export already exists: {prediction_path}")
            export_argv = [
                str(python),
                str(source_root / "scripts/predict_run.py"),
                run,
                split,
                "--epoch",
                str(node["endpoint"]),
                "--final-fit-manifest",
                str(membership_path),
                "--final-fit-source-validation-receipt",
                str(audit_path),
                "--final-fit-source-validation-sha256",
                contract.SOURCE_AUDIT_SHA256,
                "--device",
                "cuda",
            ]
            export_receipt = controller_root / "receipts/exports" / f"{node_id}-{split}.json"
            export_log = controller_root / "logs/exports" / f"{node_id}-{split}.log"
            process = runtime.run_stage(
                argv=[
                    "/usr/bin/timeout",
                    "--signal=TERM",
                    "--kill-after=10s",
                    f"{timeout - 10}s",
                    *export_argv,
                ],
                source_root=source_root,
                data_root=data_root,
                controller_root=controller_root,
                gpu_uuid=gpu_uuid,
                gpu_visible=gpu_visible,
                stage=2,
                trial=trial,
                timeout=timeout,
                output_dir=data_root / "cache",
                receipt=export_receipt,
                log=export_log,
                progress=lambda value, split_name=split: progress(f"export:{split_name}", value),
            )
            exported = runtime.validate_export(data_root, run, split, membership)
            result["exports"][split] = {
                "status": "complete",
                "path": str(exported),
                "sha256": contract.sha256_file(exported),
                "receipt": str(export_receipt),
                "process": process,
                "prediction_role": "fitting-only teacher features; no independent score",
                "independent_score_claim": "forbidden",
            }
    result["status"] = "complete"
    return result


def build_w18_cache(
    *,
    source_root: Path,
    data_root: Path,
    controller_root: Path,
    python: Path,
    membership_path: Path,
    audit_path: Path,
    membership: FinalFitMembership,
    output: Path,
    progress: Callable[[Mapping[str, object]], None],
) -> dict[str, object]:
    """Build and validate the one frozen six-teacher W18 cache after all source exports."""
    if output.exists():
        raise FileExistsError("unrecognized preexisting W18 cache; refusing to reuse it")
    argv = runtime.cache_argv(
        python=python,
        source_root=source_root,
        data_root=data_root,
        membership_path=membership_path,
        audit_path=audit_path,
        output=output,
    )
    receipt = controller_root / "receipts/cache/w18.json"
    log = controller_root / "logs/cache/w18.log"
    process = runtime.run_stage(
        argv=[
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=10s",
            "21590s",
            *argv,
        ],
        source_root=source_root,
        data_root=data_root,
        controller_root=controller_root,
        gpu_uuid="cpu-cache",
        gpu_visible="",
        stage=3,
        trial=1,
        timeout=21_600,
        output_dir=output,
        receipt=receipt,
        log=log,
        progress=progress,
        use_gpu=False,
    )
    metadata = validate_final_distillation_cache(
        output, membership, "source", require_families=True
    )
    return {
        "status": "complete",
        "path": str(output),
        "schema": metadata["metadata"].get("format"),
        "cache_sha256": metadata.get("metadata_sha256"),
        "receipt": str(receipt),
        "process": process,
    }
