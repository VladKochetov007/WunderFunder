from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import stat
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

PREPARATION = Path(__file__).parent
EXPORTER = PREPARATION.parent / "first_cell_evaluation_export_v4"
RECORD_V6 = Path(
    "/Users/nb/Documents/wunder/WunderFunder-maintenance/research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/export_v5"
)
ARMS = ("control", "ema", "sparse", "nll", "allthree")
EXPORT_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4/EXPORT_SOURCE_REVIEW_V4.json"
EXPORT_REVIEW_PIN = (8712, "64c93d8279a2daeb15f2750e6366a0fe94a268c7b8e9a04dbdb28e4a36a56e73")
EXPORT_REVIEWED_FILES = {
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4/export_first_cell_v4.py": (
        66790,
        "87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json": (
        15796,
        "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4/EXPORT_SOURCE_MANIFEST.json": (
        27569,
        "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4/test_export_v4_input_lead_metadata.py": (
        11096,
        "bd6d20042ac06ab5f0f368a16a1a9ec711c6af2e0ce09a925286cced38d10569",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3/EXPORT_SOURCE_REVIEW_V3.json": (
        7284,
        "ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a",
        "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v5/first_cell_score_contract.py": (
        20165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v5/first_cell_score_core.py": (
        27226,
        "30c456e7db875e2c075a6200c857c375087b18d9c952c354ae9bd3dceb0a0143",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v5/first_cell_score_predictions.py": (
        11056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v5/score_first_cell.py": (
        10630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v5/dependency_manifest.json": (
        7638,
        "d7dd24836c01543afeb80cbdd6d6b58ee8f4d464805f763fd9bd76e5083fbd61",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v5/test_score_v5_api_manifest_gate.py": (
        10537,
        "d2c9c3279eaa742bbcaca21336316262d56829a9ed838cba718a6746b2a838b6",
        None,
    ),
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v4/SCORE_V4_CONTRACT_SOURCE_ROW_REJECTION_V1.json": (
        5510,
        "3ff0858cb85c45efb6e447957d2a8db0a542ce94bb6b1946a84bb58c6ab7a727",
        None,
    ),
}
V6_BODY = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v6/REMOTE_SUBMISSION_V6.py",
    "bytes": 249896,
    "sha256": "d963b7df0349fcef3ae4804502d8b4b9430ff667132bd1b91552e73801807d10",
}
V6_BINDING = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v6/ADMISSION_SOURCE_BINDING_V6.json",
    "bytes": 22263,
    "sha256": "1175c095fa21b0cf2d26544ea9817f943fb5270b6539813a7b78353220c645c9",
}
V6_SOURCE_REVIEW = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v6/REMOTE_SUBMISSION_SOURCE_REVIEW_V6.json",
    "bytes": 10614,
    "sha256": "401f9d217282dd2a3d436e8a541b819e364ac56983f5b3d754adbcd6f4abcf33",
}
UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
ROOT = EXPORTER.parents[4]
STATUS_PREPARATION = (
    ROOT
    / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1"
)
STATUS_TEMPLATE = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/READ_EXPORT_TEMPLATE_V4.py",
    "bytes": 14786,
    "sha256": "208a185c19b3a7fa4fb620192d0dae22bd701a8856dc3f44aa0951ddb7b4254d",
}
STATUS_TEMPLATE_REVIEW = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/THIRD_UNIT_ROOT_TOOLS_SOURCE_REVIEW_V1.json",
    "bytes": 7363,
    "sha256": "8a548c071bac10223bfcbb3de327161d19bed123cd48ee35e03a371cddb8fc0f",
}
STATUS_DRIVER = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_V5.py",
    "bytes": 10396,
    "sha256": "874a8aa14cb6be6284d8899ac4266e41b70e0f7e6d94bce0d882b58130f36c2a",
}
STATUS_DRIVER_REVIEW = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_SOURCE_REVIEW_V5.json",
    "bytes": 3564,
    "sha256": "d214f775aa71de962b9af5b1b5707ccb339b5132ae67e004e082f5fc22040ecd",
}
STATUS_DRIVER_REVIEW_CORRECTION = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_V5_REVIEW_PIN_CORRECTION_V1.json",
    "bytes": 2157,
    "sha256": "9d1737e9a13fe9a5d7ba81a6753198c047301cbd80f259bb2a01bc22564ec2ab",
}
STATUS_FIXTURE = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/test_root_capture_export_status_v5_terminal_metadata.py",
    "bytes": 4206,
    "sha256": "1427c7c71d18d990d5529410c7c294b2846b6a57e26cbf2ad3dc19755304b1cb",
    "observed_sha256": "1427c7c71d18d990d5529410c7d294b2846b6a57e26cbf2ad3dc19755304b1cb",
}
STATUS_INVERSE_PROOF = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/ROOT_CAPTURE_EXPORT_STATUS_V5_INVERSE_PROOF.json",
    "bytes": 934,
    "sha256": "8abe137400fc6454e99749f9eea7f679c696ce9e930332eddbf641b5ae356c1a",
}
STATUS_TRANSPORT_KEYS = {
    "schema",
    "started_utc",
    "finished_utc",
    "body_bytes",
    "body_sha256",
    "ssh_returncode",
    "timed_out",
    "stdout_bytes",
    "stdout_sha256",
    "stderr_bytes",
    "stderr_sha256",
    "persisted_before_parse",
    "retry_performed",
}
SCORER_MODULE_PINS = {
    "first_cell_score_core": (
        "first_cell_score_core.py",
        27226,
        "30c456e7db875e2c075a6200c857c375087b18d9c952c354ae9bd3dceb0a0143",
    ),
    "first_cell_score_contract": (
        "first_cell_score_contract.py",
        20165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
    ),
}
LAUNCH_ACCEPTANCE_KEYS = {
    "schema",
    "status",
    "created_utc",
    "unit",
    "invocation_id",
    "receipts",
    "body",
    "source_review",
    "raw_transport",
    "scientific_fits_started",
    "quality_result",
    "note",
}
TRANSPORT_KEYS = {
    "schema",
    "started_utc",
    "finished_utc",
    "body_bytes",
    "body_sha256",
    "source_review_bytes",
    "source_review_sha256",
    "ssh_returncode",
    "timed_out",
    "stdout_bytes",
    "stdout_sha256",
    "stderr_bytes",
    "stderr_sha256",
    "persisted_before_parse",
    "retry_performed",
}


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


def run_pinned_source_validator(
    directory: Path,
    module_pins: dict[str, tuple[str, int, str]],
    binding: dict[str, object],
    manifest: dict[str, object],
) -> None:
    """Compile exact private source bytes in memory without pyc-based imports."""
    names = tuple(module_pins)
    if any(name in sys.modules for name in names):
        raise ValueError("pure scorer metadata dependency already imported")
    if any(directory.rglob("__pycache__")) or any(directory.rglob("*.pyc")):
        raise ValueError("pure scorer metadata dependency has preexisting bytecode")
    pinned: dict[str, tuple[Path, bytes, tuple[int, str]]] = {}
    for module_name, (filename, expected_bytes, expected_sha) in module_pins.items():
        path = directory / filename
        raw, _ = stable_file(path, 100000)
        expected = (expected_bytes, expected_sha)
        if (len(raw), sha(raw)) != expected:
            raise ValueError("scorer pure metadata dependency pin differs")
        pinned[module_name] = (path, raw, expected)

    sys.dont_write_bytecode = True
    modules: dict[str, types.ModuleType] = {}
    try:
        core_name = "first_cell_score_core"
        contract_name = "first_cell_score_contract"
        if names != (core_name, contract_name):
            raise ValueError("scorer metadata module order differs from the fixed closure")
        for module_name in names:
            path, raw, _ = pinned[module_name]
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__package__ = ""
            sys.modules[module_name] = module
            modules[module_name] = module
            code = compile(raw, str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
            if Path(module.__file__).resolve(strict=True) != path.resolve(strict=True):
                raise ValueError("pure metadata scorer dependency origin differs")

        modules[contract_name]._validate_data_identity(binding, manifest)
        for path, _, expected in pinned.values():
            raw_after, _ = stable_file(path, 100000)
            if (len(raw_after), sha(raw_after)) != expected:
                raise ValueError("scorer pure metadata dependency changed during compilation")
    finally:
        for name in names:
            sys.modules.pop(name, None)
    if any(directory.rglob("__pycache__")) or any(directory.rglob("*.pyc")):
        raise ValueError("pinned source compilation unexpectedly created bytecode")


def validate_scorer_data_identity(binding: dict[str, object], manifest: dict[str, object]) -> None:
    """Call only the new frozen scorer's pure metadata leaf, never its data/CLI path."""
    run_pinned_source_validator(
        EXPORTER.parent / "first_cell_evaluation_score_v5",
        SCORER_MODULE_PINS,
        binding,
        manifest,
    )


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


def validate_export_review_rows(review: dict[str, object]) -> dict[str, tuple[int, str]]:
    """Require the exact twelve reviewed-file rows, with no omissions or additions."""
    if (review.get("schema"), review.get("status")) != (
        "fg19.first-cell-export-source-review.v4",
        "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
    ):
        raise ValueError("actual independent V4 source review schema/status differs")
    files = review.get("reviewed_files")
    if not isinstance(files, list) or len(files) != len(EXPORT_REVIEWED_FILES):
        raise ValueError("actual V4 source review does not have the exact twelve-row list")
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
        if path not in EXPORT_REVIEWED_FILES:
            raise ValueError("reviewed file path is not in the exact V4 review set")
        expected_size, expected_digest, expected_status = EXPORT_REVIEWED_FILES[path]
        expected_keys = {"path", "bytes", "sha256"}
        if expected_status is not None:
            expected_keys.add("status")
        if set(item) != expected_keys or (size, digest) != (expected_size, expected_digest):
            raise ValueError("reviewed file row differs from the exact pinned V4 review")
        if expected_status is not None and item["status"] != expected_status:
            raise ValueError("reviewed file row status differs from its pinned review")
        pins[path] = (size, digest)
    if pins != {path: (size, digest) for path, (size, digest, _) in EXPORT_REVIEWED_FILES.items()}:
        raise ValueError("reviewed file set is not the exact twelve pinned rows")
    return pins


def reviewed_binding_pin(review: dict[str, object], relative_path: str) -> tuple[int, str]:
    """Join the exact V4 review's unique file rows to the binding path."""
    pins = validate_export_review_rows(review)
    if relative_path not in pins:
        raise ValueError("actual V4 review omits exact binding path")
    return pins[relative_path]


def validate_root_launch_acceptance(
    acceptance: dict[str, object],
    transport: dict[str, object],
    transport_path: Path,
    raw_pin: dict[str, object],
    receipt_pins: dict[str, tuple[int, str]],
    unit: object,
    invocation_id: object,
) -> None:
    """Join ROOT's reviewed body, source review, two receipts, and raw SSH transport."""
    if set(acceptance) != LAUNCH_ACCEPTANCE_KEYS:
        raise ValueError("ROOT launch acceptance exact field set differs")
    if (acceptance.get("schema"), acceptance.get("status")) != (
        "fg19.first-cell-export-launch-acceptance.v1",
        "ACCEPTED_SINGLE_EXPORT_LAUNCH_NOT_QUALITY",
    ):
        raise ValueError("ROOT launch acceptance schema/status differs")
    if acceptance.get("body") != V6_BODY or acceptance.get("source_review") != V6_SOURCE_REVIEW:
        raise ValueError("ROOT launch acceptance body/source-review pins differ from V6")
    if acceptance.get("unit") != unit or acceptance.get("invocation_id") != invocation_id:
        raise ValueError("ROOT launch acceptance unit/invocation differs from original launch")
    if (
        not isinstance(unit, str)
        or unit != UNIT
        or not isinstance(invocation_id, str)
        or len(invocation_id) != 32
        or any(char not in "0123456789abcdef" for char in invocation_id)
    ):
        raise ValueError("ROOT launch unit or invocation identifier is malformed")
    receipts = acceptance.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) != {"ADMISSION.json", "LAUNCH.json"}:
        raise ValueError("ROOT launch acceptance receipt set differs")
    if set(receipt_pins) != set(receipts):
        raise ValueError("local original receipt set differs from launch acceptance")
    for name, (size, digest) in receipt_pins.items():
        if receipts[name] != {"bytes": size, "sha256": digest}:
            raise ValueError(f"ROOT launch acceptance receipt pin differs: {name}")

    if set(transport) != TRANSPORT_KEYS:
        raise ValueError("raw-first SSH transport exact field set differs")
    try:
        started = datetime.fromisoformat(transport["started_utc"])
        finished = datetime.fromisoformat(transport["finished_utc"])
        accepted = datetime.fromisoformat(acceptance["created_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("ROOT launch or transport timestamp is malformed") from error
    if (
        started.tzinfo is None
        or finished.tzinfo is None
        or accepted.tzinfo is None
        or finished < started
        or accepted < finished
    ):
        raise ValueError("ROOT launch and SSH transport timestamps are unordered/unqualified")
    if (
        transport.get("schema") != "fg19.first-cell-export-root-ssh-transport.v1"
        or type(transport.get("ssh_returncode")) is not int
        or transport.get("ssh_returncode") != 0
        or transport.get("timed_out") is not False
        or type(transport.get("body_bytes")) is not int
        or transport.get("body_bytes") != V6_BODY["bytes"]
        or transport.get("body_sha256") != V6_BODY["sha256"]
        or type(transport.get("source_review_bytes")) is not int
        or transport.get("source_review_bytes") != V6_SOURCE_REVIEW["bytes"]
        or transport.get("source_review_sha256") != V6_SOURCE_REVIEW["sha256"]
        or type(transport.get("stdout_bytes")) is not int
        or transport.get("stdout_bytes") != raw_pin.get("bytes")
        or transport.get("stdout_sha256") != raw_pin.get("sha256")
        or type(transport.get("stderr_bytes")) is not int
        or transport.get("stderr_bytes") != 0
        or transport.get("stderr_sha256") != sha(b"")
        or transport.get("persisted_before_parse") is not True
        or transport.get("retry_performed") is not False
    ):
        raise ValueError("root SSH transport does not bind exact V6 body/review and raw capture")
    raw_transport = acceptance.get("raw_transport")
    expected_raw_transport = {"path": str(transport_path), **transport}
    if raw_transport != expected_raw_transport:
        raise ValueError("ROOT launch acceptance raw transport differs from persisted transport")
    if (
        type(acceptance.get("scientific_fits_started")) is not int
        or acceptance.get("scientific_fits_started") != 0
        or acceptance.get("quality_result") is not None
        or acceptance.get("note")
        != "ROOT acceptance binds exact stdin bytes; no worker stdin self-read or worker body-SHA claim."
    ):
        raise ValueError("ROOT launch acceptance scope/quality fields differ")


def validate_status_review_pair(review: dict[str, object], correction: dict[str, object]) -> None:
    """Bind the frozen V5 review and its explicit fixture-pin correction together."""
    if (review.get("schema"), review.get("status")) != (
        "fg19.first-cell-export-status-capture-source-review.v5",
        "PASS_SOURCE_ONLY",
    ):
        raise ValueError("V5 status source-review schema/status differs")
    candidate = review.get("candidate")
    reader = review.get("bound_reader")
    fixture = review.get("focused_fixture")
    if candidate != STATUS_DRIVER or reader != {
        "path": STATUS_TEMPLATE["path"],
        "bytes": STATUS_TEMPLATE["bytes"],
        "sha256": STATUS_TEMPLATE["sha256"],
        "unit": UNIT,
        "private_directory": "fg19_first_cell_f0_s1001_evaluation_v3",
    }:
        raise ValueError("V5 review candidate or bound reader pin differs")
    if not isinstance(fixture, dict) or (
        fixture.get("path"),
        fixture.get("bytes"),
        fixture.get("sha256"),
    ) != (
        STATUS_FIXTURE["path"],
        STATUS_FIXTURE["bytes"],
        STATUS_FIXTURE["sha256"],
    ):
        raise ValueError("frozen V5 source review fixture transcription differs unexpectedly")
    fixture_correction = correction.get("fixture_pin_correction")
    if (
        correction.get("schema") != "fg19.first-cell-export-status-source-review-pin-correction.v1"
        or correction.get("status") != "CORRECTION_RECONCILIATION_ONLY"
        or correction.get("frozen_source_review") != STATUS_DRIVER_REVIEW
        or not isinstance(fixture_correction, dict)
        or (
            fixture_correction.get("path"),
            fixture_correction.get("bytes"),
            fixture_correction.get("sha256_recorded_in_frozen_review"),
            fixture_correction.get("sha256_observed_for_exact_path"),
        )
        != (
            STATUS_FIXTURE["path"],
            STATUS_FIXTURE["bytes"],
            STATUS_FIXTURE["sha256"],
            STATUS_FIXTURE["observed_sha256"],
        )
    ):
        raise ValueError("V5 review correction does not bind the preserved review and fixture typo")
    reviewed_executable = correction.get("reviewed_executable_subject")
    if not isinstance(reviewed_executable, dict) or any(
        reviewed_executable.get(key) != STATUS_DRIVER[key] for key in ("path", "bytes", "sha256")
    ):
        raise ValueError("V5 correction does not affirm the correct executable pin")


def _constant_assignments(tree: ast.Module, names: set[str]) -> dict[str, ast.Assign]:
    found: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            name = node.targets[0].id
            if name in found:
                raise ValueError(f"reader template has duplicate binding constant: {name}")
            found[name] = node
    if set(found) != names:
        raise ValueError("reader template does not expose the exact five literal bindings")
    return found


def bind_reader_template(template_raw: bytes, literal_values: dict[str, object]) -> bytes:
    """Apply exactly the reader's five reviewed actual-receipt literal substitutions."""
    names = {"INVOCATION", "ADMISSION_BYTES", "ADMISSION_SHA256", "LAUNCH_BYTES", "LAUNCH_SHA256"}
    if set(literal_values) != names:
        raise ValueError("actual reader binding does not contain the exact five literals")
    template_text = template_raw.decode("utf-8", errors="strict")
    tree = ast.parse(template_text)
    assignments = _constant_assignments(tree, names)
    lines = template_text.splitlines(keepends=True)
    for name, node in sorted(assignments.items(), key=lambda item: item[1].lineno, reverse=True):
        if not isinstance(literal_values[name], (str, int)) or type(literal_values[name]) not in (
            str,
            int,
        ):
            raise ValueError("actual reader literal has an invalid strict scalar type")
        lines[node.lineno - 1 : node.end_lineno] = [f"{name} = {literal_values[name]!r}\n"]
    body = "".join(lines).encode("utf-8")
    body_tree = ast.parse(body.decode("utf-8", errors="strict"))
    body_assignments = _constant_assignments(body_tree, names)
    for name, original in assignments.items():
        body_assignments[name].value = original.value
    if ast.dump(body_tree, include_attributes=False) != ast.dump(tree, include_attributes=False):
        raise ValueError("bound reader changes operational AST beyond the five receipt literals")
    return body


def validate_reader_binding(
    body_raw: bytes,
    ast_binding: dict[str, object],
    template_raw: bytes,
    source_review_raw: bytes,
    literal_values: dict[str, object],
) -> dict[str, object]:
    """Join the status-reader body and AST proof to the exact reviewed template/receipt literals."""
    expected_body = bind_reader_template(template_raw, literal_values)
    body_pin = {"bytes": len(body_raw), "sha256": sha(body_raw)}
    expected = {
        "schema": "fg19.first-cell-export-reader-actual-binding.v1",
        "status": "EXACT_FIVE_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL",
        "template_bytes": len(template_raw),
        "template_sha256": sha(template_raw),
        "source_review_bytes": len(source_review_raw),
        "source_review_sha256": sha(source_review_raw),
        "actual_reader_bytes": body_pin["bytes"],
        "actual_reader_sha256": body_pin["sha256"],
        "actual_literal_values": literal_values,
        "all_other_AST_identical": True,
    }
    if body_raw != expected_body or ast_binding != expected:
        raise ValueError(
            "status reader body/AST binding differs from the reviewed five-literal fork"
        )
    return body_pin


def validate_status_transport(
    transport: dict[str, object],
    stdout_raw: bytes,
    stderr_raw: bytes,
    body_pin: dict[str, object],
) -> None:
    """Validate the status child's own raw-first SSH transport, separate from launch transport."""
    if set(transport) != STATUS_TRANSPORT_KEYS:
        raise ValueError("status-capture transport exact field set differs")
    try:
        started = datetime.fromisoformat(transport["started_utc"])
        finished = datetime.fromisoformat(transport["finished_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("status-capture transport timestamp is malformed") from error
    if (
        started.tzinfo is None
        or finished.tzinfo is None
        or finished < started
        or transport.get("schema") != "fg19.first-cell-export-status-root-transport.v1"
        or type(transport.get("ssh_returncode")) is not int
        or transport.get("ssh_returncode") != 0
        or transport.get("timed_out") is not False
        or type(transport.get("body_bytes")) is not int
        or (transport.get("body_bytes"), transport.get("body_sha256"))
        != (body_pin["bytes"], body_pin["sha256"])
        or type(transport.get("stdout_bytes")) is not int
        or (transport.get("stdout_bytes"), transport.get("stdout_sha256"))
        != (len(stdout_raw), sha(stdout_raw))
        or type(transport.get("stderr_bytes")) is not int
        or (transport.get("stderr_bytes"), transport.get("stderr_sha256"))
        != (len(stderr_raw), sha(stderr_raw))
        or transport.get("persisted_before_parse") is not True
        or transport.get("retry_performed") is not False
        or stderr_raw != b""
    ):
        raise ValueError("status-capture transport does not bind its own body and raw streams")


def validate_status_reply(
    observation: dict[str, object],
    receipt_bytes: dict[str, bytes],
    unit: str,
    invocation_id: str,
) -> None:
    """Join status-reader JSON to the original launch identity and receipts."""
    if observation.get("schema") != "fg19.first-cell-export-bounded-status.v1":
        raise ValueError("status reader response schema differs")
    if (observation.get("unit"), observation.get("invocation_id")) != (unit, invocation_id):
        raise ValueError("status reader response unit/invocation differs from original launch")
    try:
        captured = datetime.fromisoformat(observation["captured_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("status reader capture timestamp is malformed") from error
    if captured.tzinfo is None:
        raise ValueError("status reader capture timestamp is not timezone-qualified")
    systemd = observation.get("systemd")
    if not isinstance(systemd, dict) or (
        systemd.get("Id"),
        systemd.get("InvocationID"),
    ) != (unit, invocation_id):
        raise ValueError("status reader systemd identity differs from original launch")
    if observation.get("array_payloads_read_or_hashed") is not False:
        raise ValueError("status reader array-payload boundary is not false")
    for name, expected in receipt_bytes.items():
        item = observation.get(name)
        if not isinstance(item, dict) or decode_capture(item) != expected:
            raise ValueError(f"status reader raw receipt join differs: {name}")
    log = observation.get("log_tail")
    if not isinstance(log, dict):
        raise ValueError("status reader log-tail metadata is not an object")
    log_raw = decode_capture(log)
    if len(log_raw) > 65536:
        raise ValueError("status reader log tail exceeds reviewed cap")


def validate_capture_roots(
    launch_root: Path, capture_root: Path, *, expected_launch_root: Path = RECORD_V6
) -> None:
    """Require a fixed launch root and a distinct, direct, non-symlink status child."""
    if (
        launch_root != expected_launch_root
        or not launch_root.is_absolute()
        or not capture_root.is_absolute()
    ):
        raise ValueError("launch record root must be the exact caller-bound absolute root")
    if capture_root.parent != launch_root or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", capture_root.name
    ):
        raise ValueError("status capture must be a distinct single-component child of launch root")
    for path in (launch_root, capture_root):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError("launch/status evidence root is not a real unredirected directory")


def validate_capture_pair(
    *,
    launch_root: Path,
    capture_root: Path,
    launch_stdout_raw: bytes,
    launch_transport: dict[str, object],
    launch_transport_path: Path,
    launch_acceptance: dict[str, object],
    launch_receipts: dict[str, bytes],
    status_stdout_raw: bytes,
    status_stderr_raw: bytes,
    status_transport: dict[str, object],
    status_transport_path: Path,
    status_body_raw: bytes,
    status_ast_binding: dict[str, object],
    status_template_raw: bytes,
    status_source_review_raw: bytes,
    expected_launch_root: Path = RECORD_V6,
) -> tuple[dict[str, object], dict[str, object], dict[str, tuple[int, str]], dict[str, object]]:
    """Validate two separately rooted raw-first captures and return the status observation."""
    validate_capture_roots(launch_root, capture_root, expected_launch_root=expected_launch_root)
    if launch_transport_path != launch_root / "TRANSPORT.json":
        raise ValueError("launch transport path is not the fixed launch-root transport")
    if status_transport_path != capture_root / "TRANSPORT.json":
        raise ValueError("status transport path is not the distinct capture-child transport")
    if set(launch_receipts) != {"ADMISSION.json", "LAUNCH.json"}:
        raise ValueError("launch capture does not contain exactly the two original receipts")
    receipt_pins = {name: (len(raw), sha(raw)) for name, raw in launch_receipts.items()}
    launch_raw_pin = {"bytes": len(launch_stdout_raw), "sha256": sha(launch_stdout_raw)}
    validate_root_launch_acceptance(
        launch_acceptance,
        launch_transport,
        launch_transport_path,
        launch_raw_pin,
        receipt_pins,
        UNIT,
        json.loads(launch_receipts["LAUNCH.json"]).get("invocation_id"),
    )
    launch_reply = json.loads(launch_stdout_raw)
    if set(launch_reply) != {"unit", "outcome", "admission_base64", "launch_base64"} or (
        launch_reply["unit"],
        launch_reply["outcome"],
    ) != (UNIT, "submitted"):
        raise ValueError("launch raw response is not the exact four-key successful submit reply")
    for name, key in (
        ("ADMISSION.json", "admission_base64"),
        ("LAUNCH.json", "launch_base64"),
    ):
        if base64.b64decode(launch_reply[key], validate=True) != launch_receipts[name]:
            raise ValueError(
                f"launch raw response does not join the saved original receipt: {name}"
            )
    admission = json.loads(launch_receipts["ADMISSION.json"])
    launch = json.loads(launch_receipts["LAUNCH.json"])
    invocation_id = launch.get("invocation_id")
    if (
        admission.get("schema") != "fg19.first-cell-export-admission.v1"
        or admission.get("unit") != UNIT
        or launch.get("schema") != "fg19.first-cell-export-launch.v1"
        or launch.get("unit") != UNIT
        or launch.get("admission_sha256") != receipt_pins["ADMISSION.json"][1]
        or not isinstance(invocation_id, str)
        or len(invocation_id) != 32
        or any(char not in "0123456789abcdef" for char in invocation_id)
    ):
        raise ValueError(
            "original launch receipt schemas, unit, admission join, or invocation differ"
        )

    status_raw_pin = {"bytes": len(status_stdout_raw), "sha256": sha(status_stdout_raw)}
    literal_values = {
        "INVOCATION": invocation_id,
        "ADMISSION_BYTES": receipt_pins["ADMISSION.json"][0],
        "ADMISSION_SHA256": receipt_pins["ADMISSION.json"][1],
        "LAUNCH_BYTES": receipt_pins["LAUNCH.json"][0],
        "LAUNCH_SHA256": receipt_pins["LAUNCH.json"][1],
    }
    status_body_pin = validate_reader_binding(
        status_body_raw,
        status_ast_binding,
        status_template_raw,
        status_source_review_raw,
        literal_values,
    )
    validate_status_transport(
        status_transport, status_stdout_raw, status_stderr_raw, status_body_pin
    )
    observation = json.loads(status_stdout_raw)
    validate_status_reply(observation, launch_receipts, UNIT, invocation_id)
    return (
        observation,
        launch,
        receipt_pins,
        {
            "launch_raw": launch_raw_pin,
            "launch_transport": {"path": str(launch_transport_path), **launch_transport},
            "status_raw": {"path": str(capture_root / "STDOUT.raw.json"), **status_raw_pin},
            "status_body": {"path": str(capture_root / "REMOTE.py"), **status_body_pin},
            "status_ast_binding": status_ast_binding,
        },
    )


def main() -> None:
    """Accept one export only after independent launch and status captures join."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    validate_capture_roots(args.record_root, args.capture_root)

    binding, binding_pin = pinned_json(EXPORTER / "EXPORT_BINDING_V4.json")
    review, review_pin = pinned_json(ROOT / EXPORT_REVIEW_PATH)
    if (review_pin["bytes"], review_pin["sha256"]) != EXPORT_REVIEW_PIN:
        raise ValueError("independent V4 exporter/source review pin differs")
    binding_relative = str((EXPORTER / "EXPORT_BINDING_V4.json").relative_to(ROOT))
    if (binding_pin["bytes"], binding_pin["sha256"]) != reviewed_binding_pin(
        review, binding_relative
    ):
        raise ValueError("binding differs from independently reviewed bytes")

    # Keep the status-reader source review, its one-field-pin correction, and the
    # capture driver separate from both actual raw SSH transports.
    status_template_raw, _ = stable_file(ROOT / STATUS_TEMPLATE["path"], 100_000)
    status_template_review_raw, _ = stable_file(ROOT / STATUS_TEMPLATE_REVIEW["path"], 100_000)
    status_driver_raw, _ = stable_file(ROOT / STATUS_DRIVER["path"], 100_000)
    status_driver_review_raw, status_driver_review_pin = stable_file(
        ROOT / STATUS_DRIVER_REVIEW["path"], 100_000
    )
    status_correction_raw, status_correction_pin = stable_file(
        ROOT / STATUS_DRIVER_REVIEW_CORRECTION["path"], 100_000
    )
    status_fixture_raw, _ = stable_file(ROOT / STATUS_FIXTURE["path"], 100_000)
    status_inverse_raw, _ = stable_file(ROOT / STATUS_INVERSE_PROOF["path"], 100_000)
    for raw, expected, label in (
        (status_template_raw, STATUS_TEMPLATE, "status reader template"),
        (status_template_review_raw, STATUS_TEMPLATE_REVIEW, "status reader source review"),
        (status_driver_raw, STATUS_DRIVER, "status capture driver"),
        (status_driver_review_raw, STATUS_DRIVER_REVIEW, "status capture source review"),
        (status_correction_raw, STATUS_DRIVER_REVIEW_CORRECTION, "status review correction"),
        (status_inverse_raw, STATUS_INVERSE_PROOF, "status driver inverse proof"),
    ):
        if (len(raw), sha(raw)) != (expected["bytes"], expected["sha256"]):
            raise ValueError(f"pinned {label} differs")
    if (len(status_fixture_raw), sha(status_fixture_raw)) != (
        STATUS_FIXTURE["bytes"],
        STATUS_FIXTURE["observed_sha256"],
    ):
        raise ValueError("status fixture bytes differ from the corrected observed pin")
    status_driver_review = json.loads(status_driver_review_raw)
    status_correction = json.loads(status_correction_raw)
    validate_status_review_pair(status_driver_review, status_correction)

    body_raw, _ = stable_file(ROOT / V6_BODY["path"], 1_000_000)
    source_review_raw, _ = stable_file(ROOT / V6_SOURCE_REVIEW["path"], 100_000)
    source_binding_raw, _ = stable_file(ROOT / V6_BINDING["path"], 100_000)
    if (len(body_raw), sha(body_raw)) != (V6_BODY["bytes"], V6_BODY["sha256"]):
        raise ValueError("exact V6 stdin source body file differs from ROOT pin")
    if (len(source_review_raw), sha(source_review_raw)) != (
        V6_SOURCE_REVIEW["bytes"],
        V6_SOURCE_REVIEW["sha256"],
    ):
        raise ValueError("exact V6 independent source review file differs from ROOT pin")
    if (len(source_binding_raw), sha(source_binding_raw)) != (
        V6_BINDING["bytes"],
        V6_BINDING["sha256"],
    ):
        raise ValueError("exact V6 source-binding file differs from ROOT pin")

    launch_receipts = {}
    for name in ("ADMISSION.json", "LAUNCH.json"):
        launch_receipts[name] = stable_file(args.record_root / name, 1_000_000)[0]
    launch_stdout_raw = stable_file(args.record_root / "STDOUT.raw.json", 4_000_000)[0]
    launch_transport, launch_transport_pin = pinned_json(args.record_root / "TRANSPORT.json")
    launch_acceptance, launch_acceptance_pin = pinned_json(
        args.record_root / "LAUNCH_ACCEPTANCE.json"
    )

    # A status capture is a direct child with its own raw streams/transport/body.
    # The original launch response/transport remain at args.record_root.
    status_stdout_path = args.capture_root / "STDOUT.raw.json"
    status_stdout_raw = stable_file(status_stdout_path, 4_000_000)[0]
    status_stderr_path = args.capture_root / "STDERR.raw.txt"
    status_stderr_raw, status_stderr_details = stable_file(status_stderr_path, 1_000_000)
    status_transport_path = args.capture_root / "TRANSPORT.json"
    status_transport, status_transport_pin = pinned_json(status_transport_path)
    status_body_path = args.capture_root / "REMOTE.py"
    status_body_raw = stable_file(status_body_path, 100_000)[0]
    status_ast_path = args.capture_root / "AST_BINDING.json"
    status_ast_binding, status_ast_pin = pinned_json(status_ast_path)

    # The reader also persists exact receipt copies under its own child folder.
    for name, expected in launch_receipts.items():
        captured = stable_file(args.capture_root / name, 1_000_000)[0]
        if captured != expected:
            raise ValueError(f"status child receipt copy differs from launch-root bytes: {name}")

    observation, launch, receipt_pins, capture_pins = validate_capture_pair(
        launch_root=args.record_root,
        capture_root=args.capture_root,
        launch_stdout_raw=launch_stdout_raw,
        launch_transport=launch_transport,
        launch_transport_path=args.record_root / "TRANSPORT.json",
        launch_acceptance=launch_acceptance,
        launch_receipts=launch_receipts,
        status_stdout_raw=status_stdout_raw,
        status_stderr_raw=status_stderr_raw,
        status_transport=status_transport,
        status_transport_path=status_transport_path,
        status_body_raw=status_body_raw,
        status_ast_binding=status_ast_binding,
        status_template_raw=status_template_raw,
        status_source_review_raw=status_template_review_raw,
    )
    if (launch["unit"], launch["invocation_id"]) != (
        launch_acceptance["unit"],
        launch_acceptance["invocation_id"],
    ):
        raise ValueError("ROOT launch acceptance differs from actual original invocation")
    manifest, outputs = validate_metadata(observation, binding, binding_pin)

    launch_raw_pin = capture_pins["launch_raw"]
    status_raw_pin = capture_pins["status_raw"]
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
        "launch_raw_capture": {
            "path": str(args.record_root / "STDOUT.raw.json"),
            **launch_raw_pin,
        },
        "launch_raw_transport": launch_transport_pin,
        "launch_receipts": {
            name: {"bytes": size, "sha256": digest} for name, (size, digest) in receipt_pins.items()
        },
        "status_capture_root": str(args.capture_root),
        "native_raw_capture": status_raw_pin,
        "native_raw_transport": status_transport_pin,
        "status_stderr_capture": {
            "path": str(status_stderr_path),
            "bytes": status_stderr_details["bytes"],
            "sha256": sha(status_stderr_raw),
        },
        "status_reader_body": capture_pins["status_body"],
        "status_reader_ast_binding": {
            "path": str(status_ast_path),
            **status_ast_pin,
        },
        "status_driver_review": {
            "path": str(ROOT / STATUS_DRIVER_REVIEW["path"]),
            "bytes": status_driver_review_pin["bytes"],
            "sha256": sha(status_driver_review_raw),
        },
        "status_driver_review_pin_correction": {
            "path": str(ROOT / STATUS_DRIVER_REVIEW_CORRECTION["path"]),
            "bytes": status_correction_pin["bytes"],
            "sha256": sha(status_correction_raw),
        },
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
