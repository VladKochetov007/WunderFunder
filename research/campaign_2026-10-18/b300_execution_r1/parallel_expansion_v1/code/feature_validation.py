"""Independent post-fit gate for one causal-feature successor cell."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_validation_checks import check_prior, check_request, check_updates, job_spec
from feature_validation_runtime import (
    check_model,
    check_phase,
    checkpoint_helpers,
    r01_helpers,
    read_json,
    require_owned_file,
    sha256,
    verify_source_pin,
)


def _load_fit_context(
    root: Path, work: Path, fold: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, str]]:
    """Bind work receipts and membership to the immutable four-fold manifest."""
    inputs_path = work / "INPUTS.json"
    request_path = work / "worker_request.json"
    phase_path = work / "fit_phase.json"
    receipt_paths = (inputs_path, request_path, phase_path)
    for path in receipt_paths:
        require_owned_file(path, work)
    before_hashes = {str(path): sha256(path) for path in receipt_paths}
    inputs, request, phase = map(read_json, receipt_paths)
    if (
        inputs.get("schema") != "tabdata-l4-inputs.v1"
        or inputs.get("fold") != fold
        or inputs.get("n_folds") != 4
    ):
        raise ValueError("feature INPUTS.json does not bind the requested four-fold cell")

    split_path = root / "splits" / f"fold{fold}.json"
    require_owned_file(split_path, root)
    split_sha = sha256(split_path)
    split = read_json(split_path)
    fit_train = inputs.get("fit_train_groups")
    fit_valid = inputs.get("fit_valid_groups")
    evaluation = inputs.get("evaluation_groups")
    if (
        inputs.get("split_manifest") != str(split_path)
        or inputs.get("split_manifest_sha256") != split_sha
        or split.get("fold") != fold
        or split.get("n_folds") != 4
        or fit_train != split.get("fit", {}).get("train")
        or fit_valid != split.get("fit", {}).get("valid")
        or evaluation != split.get("evaluation", {}).get("valid")
        or not isinstance(fit_train, list)
        or not isinstance(fit_valid, list)
        or not isinstance(evaluation, list)
        or set(fit_valid) & set(evaluation)
    ):
        raise ValueError("feature workspace membership differs from the frozen fold")
    return inputs, request, phase, split_sha, before_hashes


def validate_fit(root: Path, work: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Validate a completed fixed-epoch-5 fit without starting an actor or scorer."""
    mode, control, run_name, width = job_spec(job)
    if work.is_symlink():
        raise ValueError("feature fit workspace cannot be a symlink")
    root, work = root.resolve(strict=True), work.resolve(strict=True)
    if not work.is_dir() or work == root or not work.is_relative_to(root):
        raise ValueError("feature fit workspace is not an owned project directory")
    fold, seed = job["fold"], job["seed"]
    inputs, request, phase, split_sha, before_hashes = _load_fit_context(root, work, fold)
    split_path = root / "splits" / f"fold{fold}.json"

    phase_hashes = check_phase(root, work, phase, job)
    import feature_common

    actor_source_pin = feature_common.verify_source(root)
    code_dir = root / "runtime/b300-r01-v1/code"
    identity_path = root / "runtime/pr15-host-v2/checkpoint_identity.py"
    if code_dir.is_symlink() or identity_path.is_symlink():
        raise ValueError("feature validator runtime dependencies cannot be symlinks")
    with r01_helpers(code_dir) as (common, fit_helper, fit_recipe):
        source_pin = verify_source_pin(root, inputs, common)
        output_dir = work / "artifacts/runs" / run_name
        if output_dir.is_symlink():
            raise ValueError("fit output directory cannot be a symlink")
        output_names = [
            *(f"model_ep{epoch}.pt" for epoch in range(6)),
            "model.pt",
            "provenance.json",
            "history.json",
            "training_state.pt",
        ]
        for name in output_names:
            require_owned_file(output_dir / name, work)
        provenance_path = output_dir / "provenance.json"
        history_path = output_dir / "history.json"
        provenance = read_json(provenance_path)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        args = provenance.get("args")
        if not isinstance(args, dict):
            raise ValueError("fit provenance has no saved trainer argument map")
        source_root = root / "runtime/source-a266-lf-subset-v1"
        expected_argv = fit_recipe.build_training_argv(
            common._json(common._RECIPE),
            arm="wide",
            fold=fold,
            seed=seed,
            run_name=run_name,
            split_manifest=split_path,
            mode="fit",
        ) + ["--research-features", mode, "--research-feature-control", control]
        request_sha = check_request(
            request,
            args,
            run_name=run_name,
            fold=fold,
            seed=seed,
            expected_argv=expected_argv,
            source_root=str(source_root),
            work_dir=str(work),
        )
        producer_identity = source_pin["producer_identity"]
        if (
            provenance.get("trainer_implementation") != producer_identity
            or args.get("research_features") != mode
            or args.get("research_feature_control") != control
            or args.get("research_feature_adapter_only") is True
            or args.get("device") != "cuda"
        ):
            raise ValueError("saved trainer arguments differ from the declared feature arm")
        input_receipt = dict(inputs)
        input_receipt["split_manifest_sha256"] = split_sha
        # The frozen R01 receipt helper expects producer-file identity, while INPUTS
        # binds the wider portable package manifest. Both maps were checked above.
        input_receipt["trainer_source_identity"] = {
            "source_identity_sha256": producer_identity["sha256"],
            "files": producer_identity["files"],
        }
        base = fit_helper.extract_fit_receipt(
            work,
            output_dir,
            run_name=run_name,
            arm="wide",
            fold=fold,
            seed=seed,
            input_receipt=input_receipt,
            adapter_bundle_identity={"schema": "feature-validation-only"},
        )

    checkpoint_path = output_dir / "model_ep5.pt"
    selected_path = output_dir / "model.pt"
    with checkpoint_helpers(identity_path) as identity_helper:
        identity = identity_helper.verify_checkpoint_pair(
            checkpoint_path,
            selected_path,
            expected_epoch=5,
            expected_input_width=width,
        )
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("fixed epoch checkpoint is not an object")
        checkpoint_provenance = checkpoint.get("provenance", {})
        if not isinstance(checkpoint_provenance, dict):
            raise ValueError("fixed epoch checkpoint provenance is malformed")
        if (
            checkpoint.get("epoch") != 5
            or checkpoint.get("n_in") != width
            or checkpoint.get("args") != args
            or provenance.get("selected_epoch") != 5
            or provenance.get("split_manifest_sha256") != split_sha
            or checkpoint_provenance.get("split_manifest_sha256") != split_sha
            or checkpoint_provenance.get("trainer_implementation") != producer_identity
        ):
            raise ValueError("fixed checkpoint metadata differs from fit provenance or source")
        recorded_hashes = provenance.get("checkpoint_sha256")
        if (
            not isinstance(recorded_hashes, dict)
            or recorded_hashes.get("model_ep5.pt") != identity.fixed_file_sha256
            or recorded_hashes.get("model.pt") != identity.selected_file_sha256
        ):
            raise ValueError("trainer provenance does not bind both epoch-5 checkpoint archives")
        check_model(checkpoint, width, torch)
        feature_source_sha = source_pin["feature_source_sha256"]
        prior_sha = check_prior(
            checkpoint,
            provenance,
            inputs,
            mode=mode,
            control=control,
            seed=seed,
            feature_source_sha256=feature_source_sha,
        )
        total_steps = check_updates(history, provenance, checkpoint)
        for key in (
            "update_diagnostics",
            "trainer_implementation",
            "causal_feature_prior",
            "causal_feature_control",
            "causal_feature_prior_source_sha256",
        ):
            if checkpoint_provenance.get(key) != provenance.get(key):
                raise ValueError(f"fixed checkpoint and provenance disagree on {key}")

    outputs = base.get("outputs", {})
    evidence_hashes = {
        f"artifacts/runs/{run_name}/model_ep5.pt": identity.fixed_file_sha256,
        f"artifacts/runs/{run_name}/model.pt": identity.selected_file_sha256,
        f"artifacts/runs/{run_name}/provenance.json": sha256(provenance_path),
        f"artifacts/runs/{run_name}/history.json": sha256(history_path),
        f"artifacts/runs/{run_name}/training_state.pt": sha256(output_dir / "training_state.pt"),
    }
    if any(
        outputs.get(name, {}).get("sha256") != digest for name, digest in evidence_hashes.items()
    ):
        raise ValueError("fit outputs changed during independent artifact validation")
    if any(sha256(Path(path)) != digest for path, digest in before_hashes.items()):
        raise ValueError("feature fit identity receipts changed during validation")
    return {
        "schema": "tabdata-feature-fit-validation.v1",
        "status": "fit_completed_fixed_epoch5",
        "id": run_name,
        "family": job["family"],
        "arm": job["arm"],
        "fold": fold,
        "seed": seed,
        "feature_mode": mode,
        "feature_control": control,
        "input_width": width,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": identity.fixed_file_sha256,
        "selected_sha256": identity.selected_file_sha256,
        "payload_sha256": identity.payload_sha256,
        "model_state_sha256": identity.model_state_sha256,
        "provenance_sha256": evidence_hashes[f"artifacts/runs/{run_name}/provenance.json"],
        "history_sha256": evidence_hashes[f"artifacts/runs/{run_name}/history.json"],
        "training_state_sha256": evidence_hashes[f"artifacts/runs/{run_name}/training_state.pt"],
        "feature_prior_sha256": prior_sha,
        "total_steps": total_steps,
        "source_pin": source_pin,
        "actor_source_pin": actor_source_pin,
        "inputs_sha256": before_hashes[str(work / "INPUTS.json")],
        "request_sha256": before_hashes[str(work / "worker_request.json")],
        "trainer_argv_sha256": request_sha,
        "fit_phase_sha256": phase_hashes["phase_sha256"],
        "fit_log_sha256": phase_hashes["log_sha256"],
        "resource_receipt_sha256": phase_hashes["resource_receipt_sha256"],
        "split_manifest_sha256": split_sha,
    }
