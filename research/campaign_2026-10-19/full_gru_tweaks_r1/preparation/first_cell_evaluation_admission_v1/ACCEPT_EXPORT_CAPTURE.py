from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

PREPARATION = Path(__file__).parent
EXPORTER = PREPARATION.parent / "first_cell_evaluation_export_v1"
ARMS = ("control", "ema", "sparse", "nll", "allthree")


def sha(raw: bytes) -> str:
    """Hash metadata bytes only."""
    return hashlib.sha256(raw).hexdigest()


def pinned_json(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Read one small ROOT metadata file and retain its exact source pin."""
    if path.is_symlink() or path.stat().st_size > 4_000_000:
        raise ValueError(f"unsafe or oversized ROOT metadata: {path}")
    raw = path.read_bytes()
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


def validate_metadata(
    observation: dict[str, object], binding: dict[str, object], binding_pin: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Check exact source/model/representation/group/output joins without array reads."""
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
    if props["InvocationID"] != observation["invocation_id"]:
        raise ValueError("native invocation join failed")
    metadata = observation["terminal_metadata"]
    expected_names = {"EXPORT_MANIFEST_V1.json"} | {
        f"{arm}/PREDICTION_SIDECAR.json" for arm in ARMS
    }
    if set(metadata) != expected_names or observation["array_payloads_read_or_hashed"] is not False:
        raise ValueError("metadata-only capture membership differs")
    manifest = json.loads(decode_capture(metadata["EXPORT_MANIFEST_V1.json"]))
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
    identity = manifest["data_identity"]
    expected_sources = binding["registered_data"]["source_identities"]
    before, after = (
        identity["observed_sources_before_split"],
        identity["observed_sources_after_split"],
    )
    if (
        identity["registered_source_identities"] != expected_sources
        or before != after
        or set(before) != set(expected_sources)
    ):
        raise ValueError("six registered source membership/snapshots differ")
    for name, observed in before.items():
        if any(observed[key] != expected_sources[name][key] for key in observed):
            raise ValueError("registered source observed identity differs")
    groups = manifest["ordered_groups"]
    if len(groups) != 468 or len(set(groups)) != 468 or any(type(g) is not int for g in groups):
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
        outputs[arm] = {"prediction": prediction, "checkpoint": expected_checkpoint}
    return manifest, outputs


def main() -> None:
    """Create a ROOT metadata acceptance only after the actual bounded terminal capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    binding, binding_pin = pinned_json(EXPORTER / "EXPORT_BINDING_V1.json")
    review, review_pin = pinned_json(EXPORTER / "EXPORT_SOURCE_REVIEW_V1.json")
    if review["status"] != "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR":
        raise ValueError("independent paired source review is absent")
    if (binding_pin["bytes"], binding_pin["sha256"]) != (
        review["producer"]["binding"]["bytes"],
        review["producer"]["binding"]["sha256"],
    ):
        raise ValueError("binding differs from independently reviewed bytes")
    raw_path = args.capture_root / "STDOUT.raw.json"
    observation, raw_pin = pinned_json(raw_path)
    transport, transport_pin = pinned_json(args.capture_root / "TRANSPORT.json")
    if (
        transport["ssh_returncode"] != 0
        or transport["stdout_bytes"] != raw_pin["bytes"]
        or transport["stdout_sha256"] != raw_pin["sha256"]
        or transport["persisted_before_parse"] is not True
    ):
        raise ValueError("actual raw-first transport binding differs")
    launch_acceptance, launch_acceptance_pin = pinned_json(
        args.record_root / "LAUNCH_ACCEPTANCE.json"
    )
    for name in ("ADMISSION.json", "LAUNCH.json"):
        local = (args.record_root / name).read_bytes()
        if decode_capture(observation[name]) != local:
            raise ValueError("original launch receipt capture differs")
    launch = json.loads((args.record_root / "LAUNCH.json").read_bytes())
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
