from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

PREPARATION = Path(__file__).parent
EXPORTER = PREPARATION.parent / "first_cell_evaluation_export_v3"
ARMS = ("control", "ema", "sparse", "nll", "allthree")


def sha(raw: bytes) -> str:
    """Hash metadata bytes only."""
    return hashlib.sha256(raw).hexdigest()


def require(condition: object, message: str) -> None:
    """Reject a failed metadata boundary."""
    if not condition:
        raise ValueError(message)


def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
    """Read one bounded regular file through a no-follow descriptor and verify stability."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"unsafe file: {path}")
    require(before.st_size <= cap, f"file exceeds cap: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            f"file changed before read: {path}",
        )
        raw = bytearray()
        while len(raw) <= cap:
            block = os.read(descriptor, min(65_536, cap + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(len(raw) == before.st_size and len(raw) <= cap, f"file changed or exceeded cap: {path}")
    require(
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        f"file changed while read: {path}",
    )
    return bytes(raw), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def pinned_json(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Read one small ROOT metadata file and retain its exact source pin."""
    raw, _ = stable_file(path, 4_000_000)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("metadata root must be an object")
    return value, {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


def decode_capture(item: dict[str, object]) -> bytes:
    """Strictly join a transport's base64 field to its recorded bytes and digest."""
    raw = base64.b64decode(item["base64"], validate=True)
    if type(item.get("bytes")) is not int or (len(raw), sha(raw)) != (
        item["bytes"],
        item["sha256"],
    ):
        raise ValueError("capture base64/bytes/SHA join failed")
    return raw


def validate_scorer_data_identity(binding: dict[str, object], manifest: dict[str, object]) -> None:
    """Call only the new frozen scorer's pure metadata leaf, never its data/CLI path."""
    directory = EXPORTER.parent / "first_cell_evaluation_score_v3"
    names = ("first_cell_score_core", "first_cell_score_contract")
    if any(name in sys.modules for name in names):
        raise ValueError("pure scorer metadata dependency already imported")
    if any(directory.rglob("__pycache__")) or any(directory.rglob("*.pyc")):
        raise ValueError("pure scorer metadata dependency has preexisting bytecode")
    pins = {
        "first_cell_score_core.py": (
            27226,
            "ffe8a6e7aad38b131a78783504ef55c761d4e8037c13bac279f20c5e55f6249e",
        ),
        "first_cell_score_contract.py": (
            19485,
            "7784c91713dbfd7d11ec3cfd276eb8a62607289067999f4ecd5c03a3f9a18038",
        ),
    }
    for name, pin in pins.items():
        raw, _ = stable_file(directory / name, 100000)
        if (len(raw), sha(raw)) != pin:
            raise ValueError("scorer pure metadata dependency pin differs")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(directory))
    module = importlib.import_module("first_cell_score_contract")
    for name, pin in pins.items():
        origin = Path(sys.modules[name.removesuffix(".py")].__file__)
        if origin.resolve(strict=True) != (directory / name).resolve(strict=True):
            raise ValueError("pure metadata scorer dependency origin differs")
        raw, _ = stable_file(origin, 100000)
        if (len(raw), sha(raw)) != pin:
            raise ValueError("pure metadata scorer dependency changed during import")
    module._validate_data_identity(binding, manifest)


def validate_metadata(
    observation: dict[str, object], binding: dict[str, object], binding_pin: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Check exact source/model/representation/group/output joins without array reads."""
    if observation.get("schema") != "fg19.first-cell-export-bounded-status.v1":
        raise ValueError("exact native raw capture schema differs")
    props = observation["systemd"]
    if (
        props.get("ActiveState"),
        props.get("SubState"),
        props.get("MainPID"),
        props.get("ExecMainCode"),
        props.get("ExecMainStatus"),
        props.get("Result"),
    ) != ("active", "exited", "0", "1", "0", "success"):
        raise ValueError("export is not a successful native terminal observation")
    if props.get("Id") != observation.get("unit"):
        raise ValueError("exact native systemd Id differs from captured unit")
    if props["InvocationID"] != observation["invocation_id"]:
        raise ValueError("native invocation join failed")
    metadata = observation["terminal_metadata"]
    expected_names = {"EXPORT_MANIFEST_V1.json"} | {
        f"{arm}/PREDICTION_SIDECAR.json" for arm in ARMS
    }
    if set(metadata) != expected_names or observation["array_payloads_read_or_hashed"] is not False:
        raise ValueError("metadata-only capture membership differs")
    manifest = json.loads(decode_capture(metadata["EXPORT_MANIFEST_V1.json"]))
    if set(manifest) != {
        "schema",
        "status",
        "created_utc",
        "binding_sha256",
        "source",
        "split",
        "data_identity",
        "ordered_groups",
        "membership_sha256",
        "arms",
        "quality_or_metric_claim",
    }:
        raise ValueError("completion manifest exact field membership differs")
    if (
        manifest.get("schema"),
        manifest.get("status"),
        manifest.get("quality_or_metric_claim"),
    ) != (
        "fg19.first-cell-evaluation-export-manifest.v1",
        "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED",
        False,
    ):
        raise ValueError("export completion semantics differ")
    if (
        manifest["binding_sha256"] != binding_pin["sha256"]
        or manifest["source"] != binding["source"]
    ):
        raise ValueError("export source/binding differs")
    validate_scorer_data_identity(binding, manifest)
    groups = manifest["ordered_groups"]
    if (
        len(groups) != 468
        or len(set(groups)) != 468
        or any(type(g) is not int or not 0 <= g < 1873 for g in groups)
    ):
        raise ValueError("evaluation group membership is not 468 unique strict integers")
    membership_sha = sha("".join(f"{group}\n" for group in groups).encode("ascii"))
    if manifest["membership_sha256"] != membership_sha:
        raise ValueError("evaluation membership digest differs")
    if manifest["split"] != {
        **binding["split"],
        "role": "fit-excluded-exposed-development-evaluation",
    }:
        raise ValueError("source-qualified split differs")
    if [row["arm"] for row in manifest["arms"]] != list(ARMS):
        raise ValueError("fixed five-arm export order differs")
    outputs = {}
    for arm, row, recipe in zip(ARMS, manifest["arms"], binding["arms"], strict=True):
        raw = decode_capture(metadata[f"{arm}/PREDICTION_SIDECAR.json"])
        sidecar = json.loads(raw)
        if (len(raw), sha(raw)) != (row["sidecar_bytes"], row["sidecar_sha256"]):
            raise ValueError("sidecar exact manifest join differs")
        for key in ("arm", "run_name", "unit", "invocation_id", "representation_mode"):
            if sidecar[key] != recipe[key] or row[key] != recipe[key]:
                raise ValueError(f"fixed arm metadata differs: {arm}/{key}")
        expected_checkpoint = {**recipe["model"], "loaded_sha256": recipe["model"]["sha256"]}
        if sidecar["checkpoint"] != expected_checkpoint or row["checkpoint"] != expected_checkpoint:
            raise ValueError("fixed epoch checkpoint hash differs")
        if sidecar["selected_epoch"] != 5 or sidecar["successful_updates"] != 44760:
            raise ValueError("fixed checkpoint epoch/update selection differs")
        for key in ("terminal_acceptance", "exact_stop"):
            if sidecar[key] != recipe[key]:
                raise ValueError("accepted and stopped endpoint receipt differs")
        if (
            sidecar["ordered_groups"] != groups
            or sidecar["membership_sha256"] != membership_sha
            or sidecar["source"] != binding["source"]
            or sidecar["split"] != manifest["split"]
            or sidecar["binding_sha256"] != binding_pin["sha256"]
            or sidecar["feature_only"] != {"targets_read": False, "scored_mask_read": False}
            or sidecar["scoring_performed"] is not False
        ):
            raise ValueError("feature-only source/split/membership contract differs")
        prediction = sidecar["prediction"]
        if sidecar["representation"] != {
            "mode": recipe["representation_mode"],
            **binding["representation"],
        } or sidecar["inference"] != {
            "precision": "strict_fp32",
            "autocast": False,
            "tf32": False,
            "chunk_rows": 2000,
            "sequence_batch_size": 1,
        }:
            raise ValueError("representation/strict FP32 inference contract differs")
        if (
            prediction["shape"] != [468, 20000, 2]
            or prediction["dtype"] != "float32-little-endian"
            or prediction["data_bytes"] != 74880000
            or prediction["bytes"] != 74880000 + prediction["npy_header_bytes"]
            or prediction["sha256"] != row["prediction_sha256"]
            or prediction["bytes"] != row["prediction_bytes"]
            or type(prediction["npy_header_bytes"]) is not int
            or not 0 < prediction["npy_header_bytes"] <= 4096
        ):
            raise ValueError("bounded FP32 prediction artifact metadata differs")
        if (
            not isinstance(prediction["sha256"], str)
            or len(prediction["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in prediction["sha256"])
        ):
            raise ValueError("prediction artifact SHA is malformed")
        expected_prediction = {
            "path": "predictions.npy",
            "bytes": row["prediction_bytes"],
            "npy_header_bytes": prediction["npy_header_bytes"],
            "data_bytes": 74880000,
            "sha256": row["prediction_sha256"],
            "dtype": "float32-little-endian",
            "shape": [468, 20000, 2],
        }
        if prediction != expected_prediction:
            raise ValueError("prediction metadata exact fields differ")
        expected_sidecar = {
            "schema": "fg19.first-cell-feature-only-prediction.v1",
            "status": "PREDICTION_EXPORT_ONLY_NOT_SCORED",
            "binding_sha256": binding_pin["sha256"],
            "arm": arm,
            "run_name": recipe["run_name"],
            "unit": recipe["unit"],
            "invocation_id": recipe["invocation_id"],
            "selected_epoch": 5,
            "successful_updates": 44760,
            "terminal_acceptance": recipe["terminal_acceptance"],
            "exact_stop": recipe["exact_stop"],
            "checkpoint": expected_checkpoint,
            "representation_mode": recipe["representation_mode"],
            "source": binding["source"],
            "split": manifest["split"],
            "ordered_groups": groups,
            "membership_encoding": "ASCII decimal row-group IDs, one per line, final LF",
            "membership_sha256": membership_sha,
            "feature_only": {"targets_read": False, "scored_mask_read": False},
            "representation": {"mode": recipe["representation_mode"], **binding["representation"]},
            "inference": {
                "precision": "strict_fp32",
                "autocast": False,
                "tf32": False,
                "chunk_rows": 2000,
                "sequence_batch_size": 1,
            },
            "prediction": expected_prediction,
            "scoring_performed": False,
        }
        if sidecar != expected_sidecar:
            raise ValueError("sidecar exact fields or consumer metadata differ")
        if row != {
            "arm": arm,
            "directory": arm,
            "run_name": recipe["run_name"],
            "unit": recipe["unit"],
            "invocation_id": recipe["invocation_id"],
            "representation_mode": recipe["representation_mode"],
            "selected_epoch": 5,
            "successful_updates": 44760,
            "checkpoint": expected_checkpoint,
            "membership_sha256": membership_sha,
            "prediction_path": f"{arm}/predictions.npy",
            "sidecar_path": f"{arm}/PREDICTION_SIDECAR.json",
            "prediction_bytes": prediction["bytes"],
            "prediction_sha256": prediction["sha256"],
            "sidecar_bytes": len(raw),
            "sidecar_sha256": sha(raw),
        }:
            raise ValueError("manifest arm row exact fields or relative paths differ")
        outputs[arm] = {"prediction": prediction, "checkpoint": expected_checkpoint}
    return manifest, outputs


def reviewed_binding_pin(review: dict[str, object], relative_path: str) -> tuple[int, str]:
    """Join the actual V3 review's unique file rows to the exact binding path."""
    if (review.get("schema"), review.get("status")) != (
        "fg19.first-cell-export-source-review.v3",
        "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
    ):
        raise ValueError("actual independent V3 source review schema/status differs")
    files = review.get("reviewed_files")
    if not isinstance(files, list) or not files:
        raise ValueError("actual V3 source review has no file list")
    pins = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("reviewed file row is not an object")
        path, size, digest = item.get("path"), item.get("bytes"), item.get("sha256")
        if (
            not isinstance(path, str)
            or path in pins
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("reviewed file path or byte/SHA pin is malformed/duplicate")
        pins[path] = (size, digest)
    if relative_path not in pins:
        raise ValueError("actual V3 review omits exact binding path")
    return pins[relative_path]


def main() -> None:
    """Create a ROOT metadata acceptance only after the actual bounded terminal capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    binding, binding_pin = pinned_json(EXPORTER / "EXPORT_BINDING_V3.json")
    review, review_pin = pinned_json(EXPORTER / "EXPORT_SOURCE_REVIEW_V3.json")
    binding_relative = str((EXPORTER / "EXPORT_BINDING_V3.json").relative_to(EXPORTER.parents[4]))
    if (binding_pin["bytes"], binding_pin["sha256"]) != reviewed_binding_pin(
        review, binding_relative
    ):
        raise ValueError("binding differs from independently reviewed bytes")
    raw_path = args.capture_root / "STDOUT.raw.json"
    observation, raw_pin = pinned_json(raw_path)
    transport, transport_pin = pinned_json(args.capture_root / "TRANSPORT.json")
    if (
        type(transport["ssh_returncode"]) is not int
        or transport["ssh_returncode"] != 0
        or transport["stdout_bytes"] != raw_pin["bytes"]
        or transport["stdout_sha256"] != raw_pin["sha256"]
        or transport["persisted_before_parse"] is not True
    ):
        raise ValueError("actual raw-first transport binding differs")
    launch_acceptance, launch_acceptance_pin = pinned_json(
        args.record_root / "LAUNCH_ACCEPTANCE.json"
    )
    if (launch_acceptance.get("schema"), launch_acceptance.get("status")) != (
        "fg19.first-cell-export-launch-acceptance.v1",
        "ACCEPTED_SINGLE_EXPORT_LAUNCH_NOT_QUALITY",
    ):
        raise ValueError("ROOT launch acceptance schema/status differs")
    for name in ("ADMISSION.json", "LAUNCH.json"):
        local = stable_file(args.record_root / name, 1_000_000)[0]
        if (len(local), sha(local)) != (
            launch_acceptance["receipts"][name]["bytes"],
            launch_acceptance["receipts"][name]["sha256"],
        ):
            raise ValueError("ROOT launch acceptance original receipt pins differ")
        if decode_capture(observation[name]) != local:
            raise ValueError("original launch receipt capture differs")
    launch = json.loads(stable_file(args.record_root / "LAUNCH.json", 1_000_000)[0])
    if (observation["unit"], observation["invocation_id"]) != (
        launch["unit"],
        launch["invocation_id"],
    ):
        raise ValueError("terminal-to-launch unit/invocation differs")
    if (launch_acceptance["unit"], launch_acceptance["invocation_id"]) != (
        launch["unit"],
        launch["invocation_id"],
    ):
        raise ValueError("ROOT launch acceptance differs from actual invocation")
    manifest, outputs = validate_metadata(observation, binding, binding_pin)
    acceptance = {
        "schema": "fg19.first-cell-export-acceptance.v1",
        "status": "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": observation["unit"],
        "invocation_id": observation["invocation_id"],
        "native_terminal": {
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
        },
        "source_review": review_pin,
        "binding": binding_pin,
        "launch_acceptance": launch_acceptance_pin,
        "native_raw_capture": raw_pin,
        "native_raw_transport": transport_pin,
        "manifest_capture": {
            k: v
            for k, v in observation["terminal_metadata"]["EXPORT_MANIFEST_V1.json"].items()
            if k != "base64"
        },
        "membership_sha256": manifest["membership_sha256"],
        "evaluation_groups": 468,
        "outputs": outputs,
        "exposure": "fit-excluded EXPOSED development groups, not a pristine/public holdout",
        "quality_result": None,
        "array_payloads_read_or_hashed_by_ROOT": False,
        "stop_status": "PENDING_EXACT_STOP",
    }
    target = args.record_root / "EXPORT_ACCEPTANCE.json"
    encoded = (json.dumps(acceptance, indent=2) + "\n").encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"path": str(target), "bytes": len(encoded), "sha256": sha(encoded)}))


if __name__ == "__main__":
    main()
