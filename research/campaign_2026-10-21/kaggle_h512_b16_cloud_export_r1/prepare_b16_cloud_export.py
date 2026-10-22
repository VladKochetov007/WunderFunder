"""Prepare sealed runner specs for recovered B16 prediction exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCRATCH = Path("/home/vlad/development/wunder/research_tmp")
FIT = "r01_h512_f1_s1004_kaggle_fit_r2"
SWA = "r01_h512_f1_s1004_kaggle_swa345_r2"
CAMPAIGN = "research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2"
COLLECTION_MANIFEST_SHA256 = "b45a283af8d55a292d1f211bb389fe945581086c0c45c67437563f601438af86"
WORKER_TIMEOUT_SECONDS = 700
KERNEL_WALL_SECONDS = 1_700
FINALIZATION_RESERVE_SECONDS = 300
QUOTA_FLOOR_SECONDS = 2_345
SOURCE_PINS = {
    "pyproject.toml": "18d727bb46bbe5e1d09164616555029dd7f13f362128ec4fcc424c76389cd278",
    "uv.lock": "6776378c6216d219e58e24f006416c11aa0f9d9e1d8e5e080d7e25d67b07e173",
    "scripts/predict_run.py": "904ddc9a2cafd194dc8d6ab9bdf1524120d6a089b83e33e2eb731e67abed6754",
    "scripts/train_rnn.py": "74530d3499bba1b5647e459105cb263bfc7205d7f3aa928684dfb279c2677345",
}
CODE_PATHS = {
    "scripts/kaggle_common.py",
    "scripts/kaggle_gpu_runner.py",
    "scripts/kaggle_kernel.py",
    "scripts/predict_run.py",
    "scripts/train_rnn.py",
    "utils.py",
}
INPUT_PATHS = {
    "datasets/train.parquet",
    "datasets/valid.parquet",
    f"{CAMPAIGN}/fold1_profile_view.json",
    f"{CAMPAIGN}/fold1_portable_inputs.json",
    "research/campaign_2026-09-29/artifacts/splits/fold1.json",
    "cache/chains.npz",
    "cache/chain_folds.npz",
}


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object at a pinned artifact boundary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    """Hash a small sealed artifact with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Match the existing Kaggle runner's canonical JSON digest."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def descriptor(root: Path, logical_path: str) -> dict[str, object]:
    """Describe exact staged bytes for the existing worker runner."""
    relative = Path(logical_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a safe relative artifact path: {logical_path}")
    path = root / logical_path
    return {"path": logical_path, "size": path.stat().st_size, "sha256": sha256_file(path)}


def stage_link(
    root: Path, recovery_root: Path, logical_path: str, expected: dict[str, Any]
) -> None:
    """Expose one recovered small artifact by symlink in the disposable source stage."""
    source = recovery_root / logical_path
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = descriptor(recovery_root, logical_path)
    if observed["size"] != expected["size"] or observed["sha256"] != expected["sha256"]:
        raise ValueError(f"Recovered artifact identity differs: {logical_path}")
    target = root / logical_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or descriptor(root, logical_path) != observed:
            raise ValueError(f"Refusing to replace staged source path: {logical_path}")
        return
    target.symlink_to(source.resolve())


def load_source_descriptors(source_spec: Path, source_root: Path) -> list[dict[str, object]]:
    """Select the exact predictor/runtime closure, excluding every fit worker."""
    spec = read_object(source_spec)
    code = spec.get("code")
    if not isinstance(code, list) or canonical_sha256(
        sorted(code, key=lambda item: item["path"])
    ) != spec.get("code_sha256"):
        raise ValueError("Frozen B16 execution spec code aggregate is invalid")
    descriptors = {item["path"]: item for item in code}
    selected = [
        item
        for name, item in descriptors.items()
        if name in CODE_PATHS or name.startswith("src/wnn/")
    ]
    if not CODE_PATHS.issubset({item["path"] for item in selected}):
        raise ValueError("Frozen source spec lacks an export runner dependency")
    for name, expected_sha256 in SOURCE_PINS.items():
        if name in descriptors:
            item = descriptors[name]
            if item["sha256"] != expected_sha256:
                raise ValueError(f"Frozen source pin changed: {name}")
        source_file = source_root / name
        if not source_file.is_file() or sha256_file(source_file) != expected_sha256:
            raise ValueError(f"Staged source pin changed or is absent: {name}")
        if name not in descriptors:
            selected.append(descriptor(source_root, name))
    return sorted(selected, key=lambda item: item["path"])


def make_worker_spec(
    name: str,
    run: str,
    code: list[dict[str, object]],
    inputs: list[dict[str, object]],
    timeout_seconds: int,
) -> dict[str, object]:
    """Bind one fixed FP32 evaluation-only call to the existing runner."""
    return {
        "version": 1,
        "name": name,
        "kind": "prediction_export",
        "argv": [
            "{python}",
            "scripts/predict_run.py",
            run,
            "valid",
            "--campaign-split-manifest",
            f"{CAMPAIGN}/fold1_profile_view.json",
            "--campaign-evaluation-only",
            "--device",
            "cuda",
            "--inference-precision",
            "fp32",
            "--output-name",
            run,
            "--epoch",
            "5",
            "--loader-workers",
            "0",
        ],
        "artifacts": [
            f"cache/pred_{run}_valid.npy",
            f"cache/pred_{run}_valid.json",
        ],
        "code": code,
        "code_sha256": canonical_sha256(code),
        "inputs": inputs,
        "limits": {
            "cpus": 1,
            "ram_bytes": 4_000_000_000,
            "gpu_memory_bytes": 4_000_000_000,
            "gpu_stop_bytes": 5_000_000_000,
            "timeout_seconds": timeout_seconds,
        },
        "environment": {
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONHASHSEED": "0",
            "WNN_DEFERRED_IDENTITY_PATHS": "datasets/train.parquet,datasets/valid.parquet",
            "WNN_PORTABLE_DATA_MANIFEST": f"{CAMPAIGN}/fold1_portable_inputs.json",
        },
    }


def prepare(args: argparse.Namespace) -> None:
    """Verify recovered identities and emit two sealed prediction specs."""
    source_root = args.source_root.resolve()
    recovery_root = args.recovery_root.resolve()
    output_dir = args.output_dir.resolve()
    for path in (source_root, recovery_root, output_dir):
        if not path.is_relative_to(SCRATCH.resolve()):
            raise ValueError(f"Temporary stage must be on physical scratch: {path}")
    if (
        args.worker_timeout_seconds > WORKER_TIMEOUT_SECONDS
        or args.kernel_wall_seconds > KERNEL_WALL_SECONDS
    ):
        raise ValueError("Requested timeout exceeds the reviewed B16 export cap")
    if args.worker_timeout_seconds * 2 + FINALIZATION_RESERVE_SECONDS > args.kernel_wall_seconds:
        raise ValueError("Two worker caps and the 300-second finalizer exceed the kernel cap")
    receipt = read_object(args.recovery_receipt)
    if (
        receipt.get("status") != "OFFLINE_ARTIFACT_RECOVERY_COMPLETE_NOT_GRADED"
        or receipt.get("source_commit") != "68ff682977b822f38351aac0f621146f5f13b440"
        or receipt.get("initial_parameter_sha256") is not None
        or receipt.get("initialization_identity_status") != "not_emitted_by_pinned_trainer_source"
        or receipt.get("optimizer_chunk_updates") != 44_760
        or receipt.get("fit_membership")
        != {"train_groups": 10_533, "valid_fit_groups": 1_416, "evaluation_groups": 457}
    ):
        raise ValueError("Offline recovery receipt differs from the preserved B16 evidence")
    if sha256_file(args.collection_manifest) != COLLECTION_MANIFEST_SHA256:
        raise ValueError("Original B16 terminal collection manifest hash changed")
    collection = read_object(args.collection_manifest)
    fit_run = f"artifacts/runs/{FIT}"
    fit_checkpoint = f"{fit_run}/model_ep5.pt"
    fit_provenance = f"{fit_run}/provenance.json"
    fit_files = receipt["copied_fit_artifacts"]
    fit_expected = {
        fit_checkpoint: fit_files[fit_checkpoint],
        fit_provenance: fit_files[fit_provenance],
    }
    for path, expected in fit_expected.items():
        collected = collection[f"experiment/{path}"]
        if (
            collected["bytes"] != expected["size_bytes"]
            or collected["sha256"] != expected["sha256"]
        ):
            raise ValueError(f"Original collection and recovery receipt differ: {path}")
        stage_link(
            source_root,
            recovery_root,
            path,
            {"size": expected["size_bytes"], "sha256": expected["sha256"]},
        )
    swa_run = f"artifacts/runs/{SWA}"
    swa_checkpoint = f"{swa_run}/model_ep5.pt"
    swa_derivation = str(receipt["swa_derivation_receipt_path"])
    for path, expected_sha256 in (
        (swa_checkpoint, receipt["swa_checkpoint_sha256"]),
        (swa_derivation, receipt["swa_derivation_receipt_sha256"]),
    ):
        recovered = descriptor(recovery_root, path)
        if recovered["sha256"] != expected_sha256:
            raise ValueError(f"SWA recovery identity differs: {path}")
        stage_link(source_root, recovery_root, path, recovered)
    source_spec = read_object(args.base_execution_spec)
    input_descriptors = {item["path"]: item for item in source_spec["inputs"]}
    if not INPUT_PATHS.issubset(input_descriptors):
        raise ValueError("Frozen B16 source spec lacks a required input identity")
    inputs = [input_descriptors[path] for path in sorted(INPUT_PATHS)]
    inputs.extend(
        descriptor(source_root, path)
        for path in (fit_checkpoint, fit_provenance, swa_checkpoint, swa_derivation)
    )
    code = load_source_descriptors(args.base_execution_spec, source_root)
    specs = [
        make_worker_spec("export_fit_cp5", FIT, code, inputs, args.worker_timeout_seconds),
        make_worker_spec("export_swa345", SWA, code, inputs, args.worker_timeout_seconds),
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    for spec in specs:
        (output_dir / f"{spec['name']}.json").write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema": "wnn.kaggle-h512-b16-export-spec-stage.v1",
        "status": "LOCAL_SPECS_ONLY_NO_BUNDLE_UPLOAD_OR_KERNEL_LAUNCH",
        "source_commit": receipt["source_commit"],
        "source_execution_spec_sha256": sha256_file(args.base_execution_spec),
        "preparation_script_sha256": sha256_file(Path(__file__)),
        "recovery_receipt_sha256": sha256_file(args.recovery_receipt),
        "collection_manifest_sha256": COLLECTION_MANIFEST_SHA256,
        "source_bundle_code_paths": len(code),
        "worker_code_sha256": canonical_sha256(code),
        "input_paths": [item["path"] for item in inputs],
        "initial_parameter_sha256": None,
        "initialization_identity_status": receipt["initialization_identity_status"],
        "optimizer_chunk_updates": receipt["optimizer_chunk_updates"],
        "fit_checkpoint": fit_expected[fit_checkpoint],
        "swa_checkpoint_sha256": receipt["swa_checkpoint_sha256"],
        "swa_derivation_receipt_sha256": receipt["swa_derivation_receipt_sha256"],
        "workers": [spec["name"] for spec in specs],
        "timeout_seconds_per_worker": args.worker_timeout_seconds,
        "aggregate_worker_cap_seconds": args.worker_timeout_seconds * 2,
        "kernel_wall_seconds": args.kernel_wall_seconds,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "quota_gate": {
            "minimum_fresh_remaining_seconds": QUOTA_FLOOR_SECONDS,
            "require_no_other_reservations": True,
            "prior_observation_utc": "2026-10-21 19:56 UTC",
            "prior_observation_is_current_admission": False,
        },
        "spec_sha256": {
            f"{spec['name']}.json": sha256_file(output_dir / f"{spec['name']}.json")
            for spec in specs
        },
    }
    (output_dir / "STAGING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def patch_kernel_deadline(path: Path, wall_seconds: int) -> None:
    """Set the generated bootstrap's internal deadline before provider submission."""
    kernel = path.resolve()
    if not kernel.is_relative_to(SCRATCH.resolve()):
        raise ValueError("Generated kernel must stay on physical scratch")
    if wall_seconds != KERNEL_WALL_SECONDS:
        raise ValueError("Kernel deadline must match the staged 1,700-second B16 export cap")
    receipt_path = kernel.parent.parent / "KERNEL_DEADLINE_PATCH_R1.json"
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite deadline patch receipt: {receipt_path}")
    source = kernel.read_text(encoding="utf-8")
    old = "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200"
    new = f"KERNEL_PROCESS_WALL_SECONDS_MAX = {wall_seconds:_}"
    if source.count(old) != 1:
        raise ValueError("Generated kernel did not match the pinned wall-time source")
    kernel.write_text(source.replace(old, new), encoding="utf-8")
    receipt = {
        "schema": "wnn.kaggle-h512-b16-kernel-deadline-patch.v1",
        "status": "LOCAL_KERNEL_PATCH_ONLY_NO_PROVIDER_ACTION",
        "kernel_path": str(kernel),
        "kernel_sha256": sha256_file(kernel),
        "kernel_wall_seconds": wall_seconds,
        "worker_timeout_seconds_per_job": WORKER_TIMEOUT_SECONDS,
        "finalization_collection_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "quota_gate_minimum_fresh_remaining_seconds": QUOTA_FLOOR_SECONDS,
        "requires_fresh_quota_and_reservation_check": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Prepare local specs or patch a generated kernel deadline; no provider calls."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--recovery-root", type=Path, required=True)
    prepare_parser.add_argument("--recovery-receipt", type=Path, required=True)
    prepare_parser.add_argument("--collection-manifest", type=Path, required=True)
    prepare_parser.add_argument("--base-execution-spec", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--worker-timeout-seconds", type=int, default=WORKER_TIMEOUT_SECONDS
    )
    prepare_parser.add_argument("--kernel-wall-seconds", type=int, default=KERNEL_WALL_SECONDS)
    prepare_parser.set_defaults(handler=prepare)
    patch_parser = commands.add_parser("patch-kernel-deadline")
    patch_parser.add_argument("--kernel", type=Path, required=True)
    patch_parser.add_argument("--kernel-wall-seconds", type=int, default=KERNEL_WALL_SECONDS)
    patch_parser.set_defaults(
        handler=lambda args: patch_kernel_deadline(args.kernel, args.kernel_wall_seconds)
    )
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
