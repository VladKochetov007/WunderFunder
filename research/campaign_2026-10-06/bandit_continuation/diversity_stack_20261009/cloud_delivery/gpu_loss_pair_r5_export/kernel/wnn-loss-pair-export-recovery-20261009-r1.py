from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT, REPO = WORK / "out", WORK / "wnn"
R4_CODE = "vladkochetov07/wnn-loss-pair-code-20261009-r4"
R4_RECOVERY = "vladkochetov07/wnn-loss-pair-r4-recovery-20261009-r1"
FOLD0_DATA = "vladkochetov07/wnn-c20261004-fold0-assets"
SOURCE_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
R4_SOURCE_COMMIT = "1bc0fbacf53cdc08a8c4664c6f66f38f8980092c"
R4_ASSET_SHA = "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
R4_ARCHIVE_SHA = "a3c616f6a9e2f55c2747cf771f8d76ce33cdd90b847e454c75df9cea6fedaa18"
R4_CODE_MANIFEST_SHA = "0a6fa1567dde9ac5fc41d27eed8d9edb6f90a93afb891db5a4080770b65e4351"
RECOVERY_MANIFEST_SHA = "0d9d5b6d53017b8ef837c369d7aaa47ce2a676c37a1165cda71d6735b446bcfd"
R4_TERMINAL_RECEIPT_SHA = "1bed2d7befbdd25b487f15f0b8788b8e514783058a2c7d5f683b31ff70a7500c"
R4_FIT_REVIEW_SHA = "701987320ad045a4688e633cef19d59fb8b940cb1a75f53b089deaeb0a070e47"
R4_OLD_CACHE_RECEIPT_SHA = "f6452f4b03abf7c66a219008458e431a8c1b6a2ffba961f63e5f57d1e936076f"
PREDICTOR_ORIGINAL_SHA = "74d1eac17add3f9461e1205bac0eb5b6a78facd567898d679b15bdfa5bed4f5c"
CACHE_MANIFEST_SHA = "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
CACHE_KEY = "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6"
SOURCE_RECEIPT_SHA = "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7"
PHASE_SHA = "cee62c0cdfa0a8b13998690fa49a7b111b56798a8ff6baf09e0a06b315c008e5"
CPU_VALIDATION_SHA = "84686dff926036b37362a3ea28edcf2166d0bb44bee11b2fcbd64082e4b5cf35"
CPU_PROFILE_SHA = "c371f3c70e79e69de8b382b442869d2e06e72ea4890a16d19b22923d7c82825e"
SOURCE_VERIFY_SHA = "1ccd91ccec85bd157cbd0244ae9032316b88d781a0caf19daf7a027f3a908567"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
REGISTRY_SHA = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
MEMBERSHIP_SHA = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
FOLD0_MANIFEST_SHA = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
MODEL_SHA = {
    "wmse": "71fcba3d9eaff3710f8fd42f3778ef74cd425f11977668a5575c21563a82c309",
    "huber_d1": "b61b5026386273b11db871f06f09eda7654af5814c066e850274e32d9cdfca77",
}
EPOCH1_SHA = {
    "wmse": "da9763cf34280224561f6115baf413fa9779bf9381a590088a9a04e8bc01bc90",
    "huber_d1": "6424b906c5b894598665843f043d7cdb28774211f8d9a0eb2474d20e37562390",
}
PROVENANCE_SHA = {
    "wmse": "c3e99c0c310288450f86fd007f8dfb9ffe80a5acd13857ebce455be83803477c",
    "huber_d1": "18e71073b7ac70fdac9aa57a3bd15fc0c55122472f9ace3eeeda0ab3befe8fe5",
}
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY_PATH = "research/campaign_2026-10-04/portable_inputs.json"
MEMBERSHIP_PATH = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/loss_pair_r1_membership_fold0_f03_s20261009.json"
ARM_RUN = {"wmse": "loss_pair_r3_wmse", "huber_d1": "loss_pair_r3_huber_d1"}


class Rejected(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def phase(name: str, **values: Any) -> None:
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": name, **values}
    with (WORK / "phase.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def mounted_manifest(dataset: str, expected_sha: str) -> tuple[Path, dict[str, Any]]:
    matches = []
    depth0 = len(INPUT.parts)
    for directory, dirs, files in os.walk(INPUT, followlinks=False):
        current = Path(directory)
        if len(current.parts) - depth0 >= 8:
            dirs[:] = []
        else:
            dirs.sort()
        if "assets-manifest.json" in files:
            path = current / "assets-manifest.json"
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("dataset") == dataset:
                matches.append((path, value))
    if len(matches) != 1:
        raise Rejected(f"expected one mounted dataset manifest for {dataset}, found {len(matches)}")
    path, value = matches[0]
    if sha(path) != expected_sha or value.get("version") != 1 or value.get("private") is not True:
        raise Rejected(f"mounted dataset manifest differs for {dataset}")
    seen_paths: set[str] = set()
    seen_payloads: dict[str, tuple[str, int]] = {}
    checked: set[str] = set()
    for row in value.get("files", []):
        rel, name = str(row.get("path", "")), str(row.get("filename", ""))
        pure = PurePosixPath(rel)
        info = (str(row.get("sha256", "")), int(row.get("size", -1)))
        if not rel or pure.is_absolute() or ".." in pure.parts or Path(name).name != name or not name.endswith(".wnn") or rel in seen_paths:
            raise Rejected(f"invalid payload manifest row: {rel}")
        if name in seen_payloads and seen_payloads[name] != info:
            raise Rejected(f"same opaque payload name has conflicting pins: {name}")
        seen_paths.add(rel)
        seen_payloads[name] = info
        if name in checked:
            continue
        payload = path.parent / name
        if not payload.is_file() or payload.is_symlink() or payload.stat().st_size != info[1] or sha(payload) != info[0]:
            raise Rejected(f"mounted payload failed size/hash verification: {rel}")
        checked.add(name)
    return path.parent, value


def discover_fold0_asset_manifest(input_root: Path) -> tuple[Path, dict[str, Any]]:
    matches = []
    depth0 = len(input_root.parts)
    if not input_root.is_dir():
        raise Rejected(f"Kaggle input directory is missing: {input_root}")
    for directory, dirs, files in os.walk(input_root, followlinks=False):
        current = Path(directory)
        if len(current.parts) - depth0 >= 8:
            dirs[:] = []
        else:
            dirs.sort()
        if "assets-manifest.json" not in files:
            continue
        path = current / "assets-manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("dataset") == FOLD0_DATA:
            matches.append((path, value))
    if len(matches) != 1:
        raise Rejected(
            f"expected one fold-0 manifest for {FOLD0_DATA}, found {len(matches)}"
        )
    path, manifest = matches[0]
    if sha(path) != FOLD0_MANIFEST_SHA:
        raise Rejected("fold-0 asset manifest hash differs from its sealed pin")
    if manifest.get("version") != 1 or manifest.get("private") is not True:
        raise Rejected("fold-0 asset manifest metadata is unsupported")
    return path, manifest


def extract_code(root: Path, manifest: dict[str, Any]) -> None:
    rows = {str(row["path"]): row for row in manifest["files"]}
    archive_name, cm_name = "wnn_loss_pair_src.tar.gz", "source-code-manifest.json"
    required = {archive_name, cm_name, SPLIT_PATH, REGISTRY_PATH, MEMBERSHIP_PATH}
    if set(rows) != required or sha(root / rows[cm_name]["filename"]) != R4_CODE_MANIFEST_SHA:
        raise Rejected("R4 code asset inventory/manifest differs")
    cm_path = root / rows[cm_name]["filename"]
    code_manifest = json.loads(cm_path.read_text(encoding="utf-8"))
    if code_manifest.get("source_commit") != R4_SOURCE_COMMIT:
        raise Rejected("R4 source commit differs")
    archive = root / rows[archive_name]["filename"]
    if sha(archive) != R4_ARCHIVE_SHA or REPO.exists():
        raise Rejected("R4 code archive differs or worktree already exists")
    files = {str(row["path"]): row for row in code_manifest["files"]}
    REPO.mkdir(parents=True)
    seen = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts or member.name not in files or member.name in seen or not member.isfile():
                raise Rejected("R4 code archive has an unexpected member")
            dest = REPO.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise Rejected(f"cannot read archive member {member.name}")
            with source, dest.open("xb") as output:
                shutil.copyfileobj(source, output, 1 << 20)
            seen.add(member.name)
    if seen != set(files):
        raise Rejected("R4 code archive omitted a pinned member")
    for rel, row in files.items():
        path = REPO / rel
        if path.stat().st_size != int(row["size_bytes"]) or sha(path) != row["sha256"]:
            raise Rejected(f"R4 code payload differs: {rel}")
    for rel, pin in ((SPLIT_PATH, SPLIT_SHA), (REGISTRY_PATH, REGISTRY_SHA), (MEMBERSHIP_PATH, MEMBERSHIP_SHA)):
        dest = REPO / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rows[rel]["filename"], dest)
        if sha(dest) != pin:
            raise Rejected(f"R4 split or membership differs: {rel}")
    for rel in ("datasets", "cache", "artifacts/runs"):
        (REPO / rel).mkdir(parents=True, exist_ok=True)


def recover_outputs(root: Path, manifest: dict[str, Any]) -> int:
    count = 0
    for row in manifest["files"]:
        rel = PurePosixPath(str(row["path"]))
        source = root / row["filename"]
        if rel.parts[0] == "out" and rel.parts[:2] == ("out", "arms"):
            dest = OUT.joinpath(*rel.parts[1:])
        elif rel.as_posix() == "out/shared/device_profile_gate.json":
            dest = OUT / "shared/device_profile_gate.json"
        elif rel.parts[0] == "out" and rel.parts[:2] == ("out", "logs"):
            dest = OUT / "recovery/logs" / rel.name
        elif rel.parts[0] == "out":
            dest = OUT / "recovery" / f"r4_{rel.name}"
        else:
            dest = OUT / "recovery" / rel.name
        if dest.exists():
            raise Rejected(f"recovery output collision: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if dest.stat().st_size != int(row["size"]) or sha(dest) != row["sha256"]:
            raise Rejected(f"recovered output differs: {rel.as_posix()}")
        count += 1
    if count != 45:
        raise Rejected(f"expected 45 recovered R4 rows, found {count}")
    if sha(OUT / "recovery/terminal_operator_receipt.json") != R4_TERMINAL_RECEIPT_SHA:
        raise Rejected("R4 terminal receipt differs")
    if sha(OUT / "recovery/root_fit_review.json") != R4_FIT_REVIEW_SHA:
        raise Rejected("R4 completed-fit review differs")
    return count


def patch_predictor() -> dict[str, str]:
    path = REPO / "scripts/predict_run.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREDICTOR_ORIGINAL_SHA:
        raise Rejected("R4 prediction entrypoint SHA differs")
    changes = (
        (
            b'    parser.add_argument("--expected-parsed-cache-validation-sha256", default="")\n',
            b'    parser.add_argument("--expected-parsed-cache-validation-sha256", default="")\n'
            b'    parser.add_argument("--expected-checkpoint-cache-validation-sha256", default="")\n',
        ),
        (
            b'        args.expected_parsed_cache_validation_sha256,\n        args.expected_parsed_cache_key,\n',
            b'        args.expected_parsed_cache_validation_sha256,\n        args.expected_parsed_cache_key,\n'
            b'        args.expected_checkpoint_cache_validation_sha256,\n',
        ),
        (
            b'            != args.expected_parsed_cache_validation_sha256\n',
            b'            != args.expected_checkpoint_cache_validation_sha256\n',
        ),
    )
    for before, after in changes:
        if raw.count(before) != 1:
            raise Rejected("R4 predictor receipt bridge patch was not a unique exact match")
        raw = raw.replace(before, after, 1)
    path.write_bytes(raw)
    receipt = {
        "schema": "wnn.loss-pair.r4-predictor-receipt-bridge.v1",
        "original_sha256": PREDICTOR_ORIGINAL_SHA,
        "patched_sha256": hashlib.sha256(raw).hexdigest(),
        "original_checkpoint_cache_receipt_sha256": R4_OLD_CACHE_RECEIPT_SHA,
        "change": "the mounted cache uses a fresh runtime receipt while each unchanged R4 checkpoint must match its original receipt",
    }
    write_json(OUT / "recovery/predictor_patch_receipt.json", receipt)
    return receipt


def prediction_argv(repo: Path, name: str, manifest: Path, validation: Path, validation_sha: str) -> list[str]:
    return [
        sys.executable, str(repo / "scripts/predict_run.py"), name, "valid",
        "--inference-precision", "fp32",
        "--parsed-cache-manifest", str(manifest),
        "--parsed-cache-validation-receipt", str(validation),
        "--expected-parsed-cache-manifest-sha256", CACHE_MANIFEST_SHA,
        "--expected-parsed-cache-validation-sha256", validation_sha,
        "--expected-parsed-cache-key", CACHE_KEY,
        "--expected-checkpoint-cache-validation-sha256", R4_OLD_CACHE_RECEIPT_SHA,
    ]


def verify_checkpoint(arm: str, run_name: str, arm_dir: Path) -> dict[str, Any]:
    import torch

    model_path = arm_dir / "model/model.pt"
    epoch1_path = arm_dir / "model/model_ep1.pt"
    provenance_path = arm_dir / "model/provenance.json"
    if (
        sha(model_path) != MODEL_SHA[arm]
        or sha(epoch1_path) != EPOCH1_SHA[arm]
        or sha(provenance_path) != PROVENANCE_SHA[arm]
    ):
        raise Rejected(f"R4 {arm} checkpoint/history provenance pins differ")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    epoch1 = torch.load(epoch1_path, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("provenance", {})
    cache_provenance = provenance.get("parsed_sequence_cache", {})
    if (
        checkpoint.get("args", {}).get("name") != run_name
        or int(checkpoint.get("epoch", -1)) != 1
        or int(checkpoint.get("args", {}).get("valid_fold", -1)) != 0
        or provenance.get("split_manifest_sha256") != SPLIT_SHA
        or cache_provenance.get("cache_key") != CACHE_KEY
        or cache_provenance.get("manifest_sha256") != CACHE_MANIFEST_SHA
        or cache_provenance.get("validation_receipt_sha256") != R4_OLD_CACHE_RECEIPT_SHA
    ):
        raise Rejected(f"R4 {arm} checkpoint source/cache provenance differs")
    state, fixed_state = checkpoint.get("model"), epoch1.get("model")
    if (
        not isinstance(state, dict)
        or not isinstance(fixed_state, dict)
        or set(state) != set(fixed_state)
    ):
        raise Rejected(f"R4 {arm} model/epoch-1 state layout differs")
    for key, tensor in state.items():
        if (
            not torch.is_tensor(tensor)
            or not torch.isfinite(tensor).all()
            or not torch.equal(tensor, fixed_state[key])
        ):
            raise Rejected(f"R4 {arm} final checkpoint differs from fixed epoch-1: {key}")
    return {
        "checkpoint_sha256": MODEL_SHA[arm],
        "epoch1_sha256": EPOCH1_SHA[arm],
        "checkpoint_epoch": 1,
        "provenance_sha256": PROVENANCE_SHA[arm],
    }


def validate_prediction_metadata(
    arm: str,
    run_name: str,
    sidecar: dict[str, Any],
    prediction_sha: str,
    expected_groups: list[int],
    validation_sha: str,
    cache_manifest: dict[str, Any],
) -> None:
    import numpy as np

    group_digest = hashlib.sha256(np.asarray(expected_groups, dtype="<i8").tobytes()).hexdigest()
    cache_identity = sidecar.get("parsed_sequence_cache") or {}
    if (
        len(expected_groups) != 468
        or sidecar.get("run") != run_name
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 1
        or sidecar.get("checkpoint_sha256") != MODEL_SHA[arm]
        or sidecar.get("prediction_sha256") != prediction_sha
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("split_manifest_sha256") != SPLIT_SHA
        or sidecar.get("row_groups") != expected_groups
        or sidecar.get("evaluation_membership_sha256") != group_digest
        or cache_identity.get("cache_key") != CACHE_KEY
        or cache_identity.get("manifest_sha256") != CACHE_MANIFEST_SHA
        or cache_identity.get("validation_receipt_sha256") != validation_sha
        or cache_identity.get("membership_sha256")
        != cache_manifest.get("membership_artifact", {}).get("sha256")
        or cache_manifest.get("membership_artifact", {}).get("sha256") != MEMBERSHIP_SHA
    ):
        raise Rejected(
            f"{arm} prediction sidecar does not bind the recovered epoch-1 model and fresh cache"
        )


def verify_prediction_sidecar(
    arm: str,
    run_name: str,
    arm_dir: Path,
    validation_sha: str,
    cache_manifest: dict[str, Any],
) -> dict[str, Any]:
    sidecar_path = arm_dir / "predictions_valid.json"
    prediction_path = arm_dir / "predictions_valid.npy"
    membership = json.loads((REPO / MEMBERSHIP_PATH).read_text(encoding="utf-8"))
    rows = membership["memberships"]["evaluation_valid"]
    expected_groups = [int(row["row_group"]) for row in rows]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    validate_prediction_metadata(
        arm,
        run_name,
        sidecar,
        sha(prediction_path),
        expected_groups,
        validation_sha,
        cache_manifest,
    )

    membership_path = arm_dir / "prediction_membership.json"
    membership_sidecar = json.loads(membership_path.read_text(encoding="utf-8"))
    membership_sidecar["prediction_sidecar_sha256"] = sha(sidecar_path)
    membership_sidecar["checkpoint_lineage"] = {
        "checkpoint_sha256": MODEL_SHA[arm],
        "original_run": run_name,
        "checkpoint_epoch": 1,
        "epoch1_checkpoint_sha256": EPOCH1_SHA[arm],
        "provenance_sha256": PROVENANCE_SHA[arm],
        "predecessor_kernel": "vladkochetov07/wnn-loss-pair-gpu-20261009-v4",
        "recovery_asset_manifest_sha256": RECOVERY_MANIFEST_SHA,
    }
    membership_sidecar["cache_receipt_relocation"] = {
        "cache_key": CACHE_KEY,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA,
        "checkpoint_original_validation_receipt_sha256": R4_OLD_CACHE_RECEIPT_SHA,
        "inference_runtime_validation_receipt_sha256": validation_sha,
        "source_stage_kernel": SOURCE_KERNEL,
        "source_stage_receipt_sha256": SOURCE_RECEIPT_SHA,
        "membership_sha256": MEMBERSHIP_SHA,
        "split_manifest_sha256": SPLIT_SHA,
    }
    write_json(membership_path, membership_sidecar)
    return {
        "sidecar_sha256": sha(sidecar_path),
        "prediction_sha256": sidecar["prediction_sha256"],
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "checkpoint_epoch": sidecar["epoch"],
        "cache_validation_receipt_sha256": validation_sha,
    }


def main() -> int:
    started = time.monotonic()
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        if (WORK / "boot_marker.json").exists():
            raise Rejected("R5 working directory is not empty")
        write_json(WORK / "boot_marker.json", {"event": "kernel_boot", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid()})
        OUT.mkdir()
        (OUT / "logs").mkdir()
        (OUT / "recovery").mkdir()
        phase("boot_verified")

        code_root, code_asset = mounted_manifest(R4_CODE, R4_ASSET_SHA)
        recovery_root, recovery_asset = mounted_manifest(R4_RECOVERY, RECOVERY_MANIFEST_SHA)
        extract_code(code_root, code_asset)
        recovered_count = recover_outputs(recovery_root, recovery_asset)
        phase("r4_artifacts_recovered", row_count=recovered_count)

        fold_manifest, _ = discover_fold0_asset_manifest(INPUT)
        sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
        from wnn.fold0_assets import verify_and_link_fold0_assets

        fold_receipt = OUT / "shared/fold0_assets_validation.json"
        verify_and_link_fold0_assets(fold_manifest, REPO / REGISTRY_PATH, REPO / SPLIT_PATH, REPO / "cache", fold_receipt)
        fold_receipt_sha = sha(fold_receipt)
        phase("fold0_assets_revalidated", receipt_sha256=fold_receipt_sha)

        patch_receipt = patch_predictor()
        model_receipts = {}
        for arm, run_name in ARM_RUN.items():
            arm_dir = OUT / "arms" / arm
            model_receipts[arm] = verify_checkpoint(arm, run_name, arm_dir)
            run_dir = REPO / "artifacts/runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(arm_dir / "model/model.pt", run_dir / "model.pt")
            if sha(run_dir / "model.pt") != MODEL_SHA[arm]:
                raise Rejected(f"R4 {arm} checkpoint changed during inference staging")

        import loss_pair_gpu_worker as worker

        worker.ROOT, worker.INPUT, worker.WORK, worker.OUT = REPO, INPUT, WORK, OUT
        worker.SOURCE_DATASET = R4_CODE
        worker._STARTED = started
        worker._FOLD0_ASSET_RECEIPT_SHA256 = fold_receipt_sha
        worker.OUTPUT_CAP_BYTES = 600_000_000
        worker.CGROUP_CURRENT_CAP_BYTES = 24_000_000_000
        worker.MIN_MEMORY_AVAILABLE_BYTES = 2_000_000_000
        worker.MIN_START_DISK_FREE_BYTES = 4_000_000_000
        worker._sample("recovery_start", {"parent": os.getpid()})

        cache_manifest, source_receipt_path = worker._mounted_cache()
        source_receipt = worker._validate_source_stage(cache_manifest, source_receipt_path)
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
        validation = OUT / "shared/cache_validation_runtime.json"
        profile = OUT / "shared/cache_reader_profile_runtime.json"
        deadline = time.monotonic() + 3300
        profile_argv = [
            sys.executable, str(REPO / "scripts/profile_parsed_sequence_cache.py"),
            "--manifest", str(cache_manifest), "--expected-manifest-sha256", CACHE_MANIFEST_SHA,
            "--validation-receipt", str(validation), "--profile-output", str(profile),
            "--per-role", "2", "--seed", "20261009",
        ]
        profile_result = worker._wait_one(profile_argv, "cache_profile_verify", OUT / "logs/cache_profile_verify.log", deadline, worker._environment())
        if profile_result.get("returncode") != 0 or profile_result.get("guard_stop"):
            raise Rejected(f"fresh cache validation/profile failed: {profile_result}")
        validation_sha, profile_sha = sha(validation), sha(profile)
        profile_value = json.loads(profile.read_text(encoding="utf-8"))
        if profile_value.get("cache_key") != CACHE_KEY or profile_value.get("validation_receipt_sha256") != validation_sha:
            raise Rejected("fresh cache profile does not bind the current validation receipt")
        phase("fresh_cache_validated", validation_sha256=validation_sha, profile_sha256=profile_sha, wall_seconds=profile_result.get("wall_seconds"))

        devices = worker._gpu_inventory()
        worker._write_json(OUT / "gpu_inventory.json", {"schema": "wnn.loss-pair.gpu-inventory.v1", "devices": devices})
        arms = {arm: OUT / "arms" / arm for arm in ARM_RUN}
        specs = []
        for index, (arm, run_name) in enumerate(ARM_RUN.items()):
            command = prediction_argv(REPO, run_name, cache_manifest, validation, validation_sha)
            env = worker._environment(gpu=str(index), arm_dir=arms[arm], run_name=run_name)
            specs.append((command, f"predict_{arm}", OUT / "logs" / f"predict_{arm}.log", env))
        prediction_rows = worker._wait_parallel(specs, deadline)
        if any(row.get("returncode") != 0 or row.get("guard_stop") for row in prediction_rows):
            raise Rejected(f"one or more prediction exports failed: {prediction_rows}")
        predictions = {row["label"].removeprefix("predict_"): row for row in prediction_rows}
        for arm, run_name in ARM_RUN.items():
            worker._collect_prediction({"manifest_path": cache_manifest, "manifest": manifest}, run_name, arms[arm])
            predictions[arm]["sidecar_identity"] = verify_prediction_sidecar(
                arm, run_name, arms[arm], validation_sha, manifest
            )
            for suffix in (".npy", ".json"):
                produced = REPO / "cache" / f"pred_{run_name}_valid{suffix}"
                if produced.is_file() and not produced.is_symlink():
                    produced.unlink()
        phase("predictions_collected", arms=sorted(ARM_RUN))

        fit_rows = [
            {"label": "fit_wmse", "fit_state": "completed_and_root_verified_in_R4", "wall_seconds": None},
            {"label": "fit_huber_d1", "fit_state": "completed_and_root_verified_in_R4", "wall_seconds": None},
        ]
        contract, _ = worker._collect_shared(
            arms,
            ARM_RUN,
            validation,
            profile,
            source_receipt,
            {"source_commit": R4_SOURCE_COMMIT, "asset_manifest_sha256": R4_ASSET_SHA, "archive_sha256": R4_ARCHIVE_SHA, "code_manifest_sha256": R4_CODE_MANIFEST_SHA},
            cache_manifest,
            fit_rows,
            predictions,
        )
        contract["recovery"] = {
            "mode": "export_only",
            "predecessor_kernel": "vladkochetov07/wnn-loss-pair-gpu-20261009-v4",
            "predecessor_terminal_receipt_sha256": R4_TERMINAL_RECEIPT_SHA,
            "predecessor_root_fit_review_sha256": R4_FIT_REVIEW_SHA,
            "recovery_asset_manifest_sha256": RECOVERY_MANIFEST_SHA,
            "checkpoint_cache_receipt_sha256": R4_OLD_CACHE_RECEIPT_SHA,
            "fresh_runtime_cache_receipt_sha256": validation_sha,
            "predictor_original_sha256": PREDICTOR_ORIGINAL_SHA,
            "predictor_patched_sha256": patch_receipt["patched_sha256"],
            "fits_retrained": False,
            "checkpoint_files_rewritten": False,
            "fit_process_wall_seconds": None,
            "logged_epoch_train_seconds_by_arm": {"wmse": [57.7, 23.1], "huber_d1": [57.7, 23.1]},
        }
        contract["evaluation"] = {"state": "pending_independent_grade", "metric_id": "global_wp_clip2_v1", "fold": 0, "fixed_evaluation_count": 468, "membership_sha256": MEMBERSHIP_SHA, "exposure": "exposed fold-0 screen; not public transfer evidence"}
        worker._write_json(OUT / "training_contract.json", contract)
        receipt = {
            "schema": "wnn.loss-pair.export-recovery-run-receipt.v1",
            "format_version": 1,
            "state": "complete_exports_pending_independent_grade",
            "source_kernel": SOURCE_KERNEL,
            "source_dataset": R4_CODE,
            "recovery_dataset": R4_RECOVERY,
            "cache_manifest_sha256": CACHE_MANIFEST_SHA,
            "cache_validation_sha256": validation_sha,
            "cache_profile_sha256": profile_sha,
            "training_contract_sha256": sha(OUT / "training_contract.json"),
            "prediction_results": predictions,
            "model_receipts": model_receipts,
            "gpu_inventory": devices,
            "source_stage_receipt_sha256": SOURCE_RECEIPT_SHA,
            "fit_results": fit_rows,
            "evaluation_state": "pending_independent_grade",
            "worker_wall_seconds": time.monotonic() - started,
            "scientific_claim": "none; independent grader owns evaluation",
        }
        worker._write_json(OUT / "export_recovery_run_receipt.json", receipt)
        worker._write_json(OUT / "commands.json", {"commands": worker._COMMANDS})
        phase("worker_complete", training_contract_sha256=receipt["training_contract_sha256"])
        write_json(WORK / "final_status.json", {"state": receipt["state"], "elapsed_seconds": time.monotonic() - started})
        return 0
    except BaseException as exc:
        failure = {"state": "failed_export_recovery", "error_type": type(exc).__name__, "error": str(exc), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            write_json(WORK / "failure.json", failure)
            write_json(WORK / "final_status.json", {"state": "failed_export_recovery"})
            phase("worker_failed", error_type=failure["error_type"], error=failure["error"])
        except OSError:
            pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
