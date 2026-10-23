from __future__ import annotations

import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

# Conditional one-shot admission for the fixed first-cell score consumer.

# The renderer prepends this exact reviewed stdlib helper source. These type-only
# bindings keep the unrendered template checkable and never replace runtime helpers.
if TYPE_CHECKING:
    require = cast(Callable[[bool, str], None], None)
    digest = cast(Callable[[bytes], str], None)
    json_object = cast(Callable[[bytes, str], dict[str, object]], None)
    stable_file = cast(Callable[[Path, int], tuple[bytes, dict[str, int]]], None)
    file_hash = cast(Callable[[Path, int], tuple[str, dict[str, int]]], None)
    _bounded_gpu_query = cast(Callable[[list[str]], tuple[str, str]], None)
    _memory_bytes = cast(Callable[[str], int], None)
    _bounded_proc_read = cast(Callable[[Path, int], bytes], None)
    _proc_starttime = cast(Callable[[Path, str], str], None)
    show = cast(Callable[[str, tuple[str, ...], int], dict[str, str]], None)
    write_exclusive = cast(Callable[[Path, bytes], None], None)
    write_json_exclusive = cast(Callable[[Path, dict[str, object]], None], None)


PROJECT_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA_ROOT = PROJECT_BASE / "data"
SOURCE_ROOT = PROJECT_BASE / "source-fg19-cache-68ff6829"
SPLIT_PATH = PROJECT_BASE / "splits/fold0.json"
SCRATCH_ROOT = PROJECT_BASE / "research_tmp"
FIT_RECEIPTS_ROOT = SCRATCH_ROOT / "fg19_child_allthree_fit_v1"
PROFILE_ROOT = SCRATCH_ROOT / "fg19_child_allthree_profile_v1"
CACHE_ROOT = SCRATCH_ROOT / "fg19_t3_fit_homogeneous_cache_v1"
CACHE_RECEIPTS_ROOT = PROJECT_BASE / "research/full_gru_tweaks_r1/cache_cpu_v2"
EXPORT_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_evaluation_v1"
EXPORT_ROOT = EXPORT_D / "exports"
SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v1"
SCORE_ACCEPTED_ROOT = SCORE_D / "accepted_export"
RESULT_PATH = EXPORT_D / "RESULT.json"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
PORTABLE_REGISTRY = PROJECT_BASE / "support-source/terminal-v1/PORTABLE_SOURCES.json"
UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"
EXPORT_UNIT = "computer-vision-homework-fg19-first-cell-export-v1.service"
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_SET_SHA256 = "9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8"
EXPORT_BINDING_BYTES = 15_822
EXPORT_BINDING_SHA256 = "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c"
SOURCE_MANIFEST_BYTES = 27_569
SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SOURCE_REVIEW_BYTES = 9_633
SOURCE_REVIEW_SHA256 = "cf44af7624e81ee99e9705bf08e69b359f7048354df069cad35b26c7ec927c15"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
PORTABLE_REGISTRY_BYTES = 2_157
PORTABLE_REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
FIT_ADMISSION_BYTES = 12_912
FIT_ADMISSION_SHA256 = "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"
FIT_LAUNCH_BYTES = 9_594
FIT_LAUNCH_SHA256 = "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538"
PARENT_CHECKPOINT_SHA256 = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
DICTIONARY_SHA256 = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
CACHE_METADATA_SHA256 = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA256 = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_STOP_SHA256 = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
PROFILE_PINS = {
    "ADMISSION.json": "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
    "LAUNCH.json": "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
    "unit.log": "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
}
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
    "PARENT_EXACT_STOP.json": "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
    "DICTIONARY_ACCEPTANCE.json": "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
    "DICTIONARY_EXACT_STOP.json": "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
    "PROFILE_TERMINAL_ACCEPTANCE.json": "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
    "PROFILE_EXACT_STOP.json": "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    "CONTROL_TERMINAL_ACCEPTANCE.json": "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    "CONTROL_EXACT_STOP.json": "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
    "EMA_TERMINAL_ACCEPTANCE.json": "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8",
    "EMA_EXACT_STOP.json": "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf",
    "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1",
    "DICTIONARY_CHILD_EXACT_STOP.json": "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3",
    "NLL_CHILD_TERMINAL_ACCEPTANCE.json": "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5",
    "NLL_CHILD_EXACT_STOP.json": "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1",
}
SCORER_FILES = {
    "first_cell_score_contract.py": (
        19_485,
        "09155537feec1d9cd398f6a729ab88201b82e814c1a8c4ac796fe4283940a90d",
    ),
    "first_cell_score_core.py": (
        27_226,
        "e115d12868b8d9c6ac85e1b497b0bc514559d78fff5dada0b0c350e18b69669a",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}
ARMS = (
    ("control", "base"),
    ("ema", "ema"),
    ("sparse", "sparse"),
    ("nll", "base"),
    ("allthree", "ema_sparse"),
)
PREDICTION_SHAPE = [468, 20_000, 2]
PREDICTION_DATA_BYTES = 74_880_000
MAX_PREDICTION_BYTES = PREDICTION_DATA_BYTES + 4_096
MAX_METADATA_BYTES = 16_000_000
MAX_SOURCE_BYTES = 128_000_000
MAX_SOURCE_AGGREGATE_BYTES = 10_000_000
WORKER_CPU_QUOTA = "50%"
WORKER_MEMORY_HIGH_BYTES = 1_400_000_000
WORKER_MEMORY_MAX_BYTES = 1_500_000_000
WORKER_RUNTIME_SECONDS = 600
PARENT_MEMORY_TARGET_BYTES = 17_000_000_000
PARENT_MEMORY_RESERVE_BYTES = 1_000_000_000
MINIMUM_FILESYSTEM_FREE_BYTES = 25 * 1024**3
EXPECTED_FILESYSTEM_DEVICE = 64_512
UNBOUND = "UNBOUND_ACTUAL_EXPORT_ACCEPTANCE_OR_STOP"
EXPORT_ACCEPTANCE_BYTES = 0
EXPORT_ACCEPTANCE_SHA256 = UNBOUND
EXPORT_EXACT_STOP_BYTES = 0
EXPORT_EXACT_STOP_SHA256 = UNBOUND
PAYLOAD_B64: dict[str, str] = {"__UNBOUND_PAYLOADS__": "__UNBOUND_PAYLOADS__"}


def validate_export_authority(
    acceptance: dict[str, object],
    exact_stop: dict[str, object],
    *,
    acceptance_sha256: str,
    binding_sha256: str,
    review_sha256: str,
    expected_unit: str,
) -> str:
    """Validate a root-owned fixed-export acceptance and its exact stop receipt."""
    if acceptance.get("schema") != "fg19.first-cell-export-acceptance.v1":
        raise ValueError("export acceptance schema differs")
    if acceptance.get("status") != "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED":
        raise ValueError("fixed prediction export is not root-accepted")
    unit, invocation = acceptance.get("unit"), acceptance.get("invocation_id")
    if unit != expected_unit or not is_invocation_id(invocation):
        raise ValueError("export acceptance unit/invocation differs")
    native = acceptance.get("native_terminal")
    if not isinstance(native, dict) or set(native) != {
        "active_state",
        "sub_state",
        "main_pid",
        "exec_main_code",
        "exec_main_status",
    }:
        raise ValueError("export native terminal fields differ")
    if (
        (native.get("active_state"), native.get("sub_state")) != ("active", "exited")
        or any(
            type(native.get(key)) is not int
            for key in ("main_pid", "exec_main_code", "exec_main_status")
        )
        or (native.get("main_pid"), native.get("exec_main_code"), native.get("exec_main_status"))
        != (0, 1, 0)
    ):
        raise ValueError("export native terminal outcome is not strict 0/1/0")
    binding = acceptance.get("binding")
    review = acceptance.get("source_review")
    if not isinstance(binding, dict) or (binding.get("bytes"), binding.get("sha256")) != (
        EXPORT_BINDING_BYTES,
        binding_sha256,
    ):
        raise ValueError("accepted export binding pin differs")
    if not isinstance(review, dict) or (review.get("bytes"), review.get("sha256")) != (
        SOURCE_REVIEW_BYTES,
        review_sha256,
    ):
        raise ValueError("accepted producer/consumer review pin differs")
    if (
        acceptance.get("evaluation_groups") != 468
        or type(acceptance.get("evaluation_groups")) is not int
    ):
        raise ValueError("accepted exposed membership count differs")
    membership = acceptance.get("membership_sha256")
    if not is_sha256(membership):
        raise ValueError("accepted exposed membership digest is malformed")
    if (
        acceptance.get("quality_result") is not None
        or acceptance.get("stop_status") != "PENDING_EXACT_STOP"
    ):
        raise ValueError("export acceptance includes a score or unexpected stop state")
    if acceptance.get("array_payloads_read_or_hashed_by_ROOT") is not False:
        raise ValueError("ROOT acceptance exceeds metadata-only export review")
    outputs = acceptance.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {arm for arm, _mode in ARMS}:
        raise ValueError("accepted output membership differs")
    for arm, _mode in ARMS:
        row = outputs[arm]
        if not isinstance(row, dict) or set(row) != {"prediction", "checkpoint"}:
            raise ValueError(f"accepted output row is malformed: {arm}")
        prediction, checkpoint = row["prediction"], row["checkpoint"]
        if not isinstance(prediction, dict) or not isinstance(checkpoint, dict):
            raise ValueError(f"accepted output pins are malformed: {arm}")
        if (
            prediction.get("shape") != PREDICTION_SHAPE
            or prediction.get("dtype") != "float32-little-endian"
            or prediction.get("data_bytes") != PREDICTION_DATA_BYTES
            or type(prediction.get("npy_header_bytes")) is not int
            or not 0 < prediction["npy_header_bytes"] <= 4_096
            or prediction.get("bytes") != PREDICTION_DATA_BYTES + prediction["npy_header_bytes"]
            or prediction.get("path") != "predictions.npy"
            or not is_sha256(prediction.get("sha256"))
        ):
            raise ValueError(f"accepted prediction metadata differs: {arm}")
        if (
            checkpoint.get("relative_path") is None
            or type(checkpoint.get("bytes")) is not int
            or checkpoint["bytes"] <= 0
            or not is_sha256(checkpoint.get("sha256"))
            or checkpoint.get("loaded_sha256") != checkpoint.get("sha256")
        ):
            raise ValueError(f"accepted checkpoint metadata differs: {arm}")
    exact_stop_schema = "fg19-first-cell-export-exact-owned-stop.v1"
    if exact_stop.get("schema") != exact_stop_schema:
        raise ValueError("export exact-stop schema differs")
    if set(exact_stop) != {
        "schema",
        "captured_utc",
        "unit",
        "invocation_id",
        "acceptance_sha256",
        "before",
        "after",
        "populated_before",
        "populated_after",
        "stop_returncode",
        "stop_stdout",
        "stop_stderr",
        "scope",
    }:
        raise ValueError("export exact-stop fields differ")
    if (
        exact_stop.get("unit"),
        exact_stop.get("invocation_id"),
        exact_stop.get("acceptance_sha256"),
    ) != (unit, invocation, acceptance_sha256):
        raise ValueError("export exact-stop identity or acceptance digest differs")
    before, after = exact_stop.get("before"), exact_stop.get("after")
    if (
        type(exact_stop.get("stop_returncode")),
        exact_stop.get("stop_returncode"),
        exact_stop.get("stop_stdout"),
        exact_stop.get("stop_stderr"),
    ) != (int, 0, "", ""):
        raise ValueError("exact stop command did not return strict zero with empty streams")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("exact-stop observations are malformed")
    if (
        before.get("Id"),
        before.get("ActiveState"),
        before.get("SubState"),
        before.get("MainPID"),
        before.get("InvocationID"),
        before.get("ExecMainCode"),
        before.get("ExecMainStatus"),
        before.get("Result"),
    ) != (unit, "active", "exited", "0", invocation, "1", "0", "success"):
        raise ValueError("export unit was not the accepted invocation before exact stop")
    if (
        after.get("ActiveState"),
        after.get("SubState"),
        after.get("MainPID"),
        after.get("LoadState"),
    ) != ("inactive", "dead", "0", "not-found"):
        raise ValueError("export unit is not absent and stopped after exact stop")
    if after.get("InvocationID") != "":
        raise ValueError("export invocation remains loaded after exact stop")
    if exact_stop.get("populated_before") != {} or exact_stop.get("populated_after") != {}:
        raise ValueError("exact-stop cgroup was populated")
    if exact_stop.get("scope") != (
        "Only the one successful exited export unit. All old native/support and foreign units "
        "excluded; no file or payload mutation."
    ):
        raise ValueError("exact-stop scope differs")
    return str(invocation)


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def is_invocation_id(value: object) -> bool:
    """Return whether a value is a lowercase 32-hex systemd invocation ID."""
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(char in "0123456789abcdef" for char in value)
    )


def payload_bytes() -> dict[str, bytes]:
    """Decode and verify exactly four scorer modules and two accepted receipts."""
    require(
        UNBOUND not in (EXPORT_ACCEPTANCE_SHA256, EXPORT_EXACT_STOP_SHA256),
        "actual export acceptance and exact-stop pins are not bound",
    )
    names = set(SCORER_FILES) | {"EXPORT_ACCEPTANCE.json", "EXPORT_EXACT_STOP.json"}
    require(set(PAYLOAD_B64) == names, "scorer and export-authority payload membership differs")
    result: dict[str, bytes] = {}
    for name, encoded in PAYLOAD_B64.items():
        raw = base64.b64decode(encoded, validate=True)
        if name in SCORER_FILES:
            expected_bytes, expected_sha = SCORER_FILES[name]
            require(
                (len(raw), digest(raw)) == (expected_bytes, expected_sha),
                f"scorer code pin differs: {name}",
            )
        elif name == "EXPORT_ACCEPTANCE.json":
            require(
                (len(raw), digest(raw)) == (EXPORT_ACCEPTANCE_BYTES, EXPORT_ACCEPTANCE_SHA256),
                "accepted export receipt pin differs",
            )
        elif name == "EXPORT_EXACT_STOP.json":
            require(
                (len(raw), digest(raw)) == (EXPORT_EXACT_STOP_BYTES, EXPORT_EXACT_STOP_SHA256),
                "export stop receipt pin differs",
            )
        else:
            raise RuntimeError(f"unapproved staged payload name: {name}")
        result[name] = raw
    require(
        sum(map(len, result.values())) <= 2_000_000, "bounded scorer staging payload exceeds cap"
    )
    return result


def path_stat(path: Path, *, expected_symlink: bool | None = False) -> dict[str, object]:
    """Observe one registered path without opening or hashing its payload."""
    link_stat = path.lstat()
    is_link = path.is_symlink()
    require(
        expected_symlink is None or is_link is expected_symlink,
        f"registered symlink status differs: {path}",
    )
    target_stat = path.stat()
    require(stat.S_ISREG(target_stat.st_mode), f"registered source is not a regular file: {path}")
    return {
        "path": str(path),
        "resolved": str(path.resolve(strict=True)),
        "islink": is_link,
        "device": target_stat.st_dev,
        "bytes": target_stat.st_size,
        "inode": target_stat.st_ino,
        "mtime_ns": target_stat.st_mtime_ns,
        "link_device": link_stat.st_dev,
        "link_inode": link_stat.st_ino,
    }


def check_registered_data(
    binding: dict[str, object], fit_admission: dict[str, object]
) -> dict[str, object]:
    """Recheck the exact six fit-registered paths and portable registry by metadata only."""
    registered = binding["registered_data"]
    sources = registered["source_identities"]
    expected_sources = fit_admission.get("data_identity_stats_only")
    require(
        isinstance(sources, dict) and isinstance(expected_sources, dict),
        "six-source admission map is missing",
    )
    require(
        sources == expected_sources and len(sources) == 6,
        "export binding differs from original six-source fit admission",
    )
    registry_raw, registry_info = stable_file(PORTABLE_REGISTRY, 2_000_000)
    require(
        (len(registry_raw), digest(registry_raw))
        == (PORTABLE_REGISTRY_BYTES, PORTABLE_REGISTRY_SHA256),
        "portable registry pin differs",
    )
    registry = json_object(registry_raw, "portable source registry")
    registry_sources = registry.get("sources")
    require(
        isinstance(registry_sources, dict) and set(registry_sources) == set(sources),
        "portable registry source membership differs",
    )
    require(
        type(registry.get("version")) is int
        and registry.get("version") == 1
        and registry.get("split_manifest_sha256") == SPLIT_SHA256,
        "portable registry version or split differs",
    )
    observed: dict[str, object] = {}
    for relative, expected in sources.items():
        require(isinstance(expected, dict), f"registered path identity malformed: {relative}")
        entry = registry_sources[relative]
        require(
            isinstance(entry, dict) and entry.get("sha256") == expected.get("registered_sha256"),
            f"portable source digest differs: {relative}",
        )
        identity = entry.get("original_identity")
        require(
            isinstance(identity, dict)
            and identity.get("bytes") == expected.get("bytes")
            and identity.get("mtime_ns") == expected.get("mtime_ns"),
            f"portable source stat differs: {relative}",
        )
        path = DATA_ROOT / relative
        current = path_stat(path, expected_symlink=expected["islink"])
        comparable = {
            key: current[key]
            for key in ("path", "resolved", "islink", "device", "bytes", "inode", "mtime_ns")
        }
        require(
            comparable == {key: expected[key] for key in comparable},
            f"registered source identity changed: {relative}",
        )
        observed[relative] = comparable
    return {
        "registry": {**registry_info, "sha256": PORTABLE_REGISTRY_SHA256},
        "sources_stat_only": observed,
        "payload_bytes_read_or_hashed": False,
    }


def check_source_tree(binding: dict[str, object], source_manifest_raw: bytes) -> dict[str, object]:
    """Hash the exact 167 admitted source files under byte and aggregate limits."""
    manifest = json_object(source_manifest_raw, "source manifest")
    source = binding["source"]
    require(
        manifest.get("schema") == "fg19.teacher-cache-source-manifest.v1",
        "canonical source manifest schema differs",
    )
    require(
        manifest.get("source_commit") == SOURCE_COMMIT
        and manifest.get("source_root") == str(SOURCE_ROOT),
        "canonical source revision or root differs",
    )
    files = manifest.get("files")
    require(
        isinstance(files, list) and len(files) == 167,
        "canonical source manifest does not contain 167 files",
    )
    require(
        (source.get("commit"), source.get("file_count"), source.get("source_set_sha256"))
        == (SOURCE_COMMIT, 167, SOURCE_SET_SHA256),
        "binding source identity differs",
    )
    require(
        source.get("manifest_bytes") == SOURCE_MANIFEST_BYTES
        and source.get("manifest_sha256") == SOURCE_MANIFEST_SHA256,
        "source manifest binding pin differs",
    )
    require(
        SOURCE_ROOT.is_dir()
        and not SOURCE_ROOT.is_symlink()
        and SOURCE_ROOT.resolve(strict=True) == SOURCE_ROOT,
        "canonical source root is redirected",
    )
    seen: set[str] = set()
    aggregate = hashlib.sha256()
    total_bytes = 0
    results: dict[str, object] = {}
    for row in sorted(files, key=lambda item: item["path"]):
        relative, expected_bytes, expected_sha = (
            row.get("path"),
            row.get("bytes"),
            row.get("sha256"),
        )
        require(
            isinstance(relative, str) and relative not in seen,
            "source manifest path duplicate or malformed",
        )
        require(
            type(expected_bytes) is int
            and 0 <= expected_bytes <= MAX_SOURCE_BYTES
            and is_sha256(expected_sha),
            f"source manifest row malformed: {relative}",
        )
        seen.add(relative)
        path = SOURCE_ROOT / relative
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            f"unsafe source manifest path: {relative}",
        )
        ancestor = path
        while ancestor != SOURCE_ROOT:
            require(not ancestor.is_symlink(), f"source path traverses a symbolic link: {relative}")
            ancestor = ancestor.parent
        require(
            path.resolve(strict=True).is_relative_to(SOURCE_ROOT),
            f"source path escaped pinned root: {relative}",
        )
        observed_sha, info = file_hash(path, MAX_SOURCE_BYTES)
        require(
            (info["bytes"], observed_sha) == (expected_bytes, expected_sha),
            f"canonical source file differs: {relative}",
        )
        total_bytes += info["bytes"]
        require(
            total_bytes <= MAX_SOURCE_AGGREGATE_BYTES, "canonical source aggregate exceeded cap"
        )
        aggregate.update(relative.encode("utf-8") + b"\0" + observed_sha.encode("ascii") + b"\n")
        results[relative] = {**info, "sha256": observed_sha}
    require(aggregate.hexdigest() == SOURCE_SET_SHA256, "canonical source aggregate digest differs")
    return {
        "source_commit": SOURCE_COMMIT,
        "file_count": len(results),
        "aggregate_bytes": total_bytes,
        "source_set_sha256": aggregate.hexdigest(),
        "files": results,
    }


def validate_acceptance_outputs(
    acceptance: dict[str, object],
    binding: dict[str, object],
    manifest: dict[str, object],
    manifest_pin: dict[str, object],
    sidecars: dict[str, tuple[dict[str, object], dict[str, object]]],
) -> dict[str, object]:
    """Join root-accepted fixed outputs to the existing manifest and sidecars without opening arrays."""
    accepted_manifest = acceptance.get("manifest_capture")
    require(
        isinstance(accepted_manifest, dict)
        and (accepted_manifest.get("bytes"), accepted_manifest.get("sha256"))
        == (manifest_pin["bytes"], manifest_pin["sha256"]),
        "accepted export manifest pin differs from current bytes",
    )
    require(
        manifest.get("schema") == "fg19.first-cell-evaluation-export-manifest.v1"
        and manifest.get("status") == "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED"
        and manifest.get("quality_or_metric_claim") is False,
        "export result is not a complete prediction-only bundle",
    )
    require(
        manifest.get("binding_sha256") == EXPORT_BINDING_SHA256
        and manifest.get("source") == binding.get("source"),
        "export manifest binding/source differs",
    )
    groups = manifest.get("ordered_groups")
    require(
        isinstance(groups, list)
        and len(groups) == 468
        and len(set(groups)) == 468
        and all(type(group) is int for group in groups),
        "export membership is not 468 strict unique groups",
    )
    membership_sha = digest("".join(f"{group}\n" for group in groups).encode("ascii"))
    require(
        membership_sha == acceptance.get("membership_sha256") == manifest.get("membership_sha256"),
        "accepted and exported membership digest differs",
    )
    require(
        manifest.get("split")
        == {**binding["split"], "role": "fit-excluded-exposed-development-evaluation"},
        "export split is not the fixed source-qualified F0 split",
    )
    arms = manifest.get("arms")
    require(
        isinstance(arms, list)
        and [row.get("arm") for row in arms if isinstance(row, dict)]
        == [name for name, _mode in ARMS],
        "export manifest arm order differs",
    )
    output_rows = acceptance.get("outputs")
    binding_arms = binding.get("arms")
    require(
        isinstance(output_rows, dict) and isinstance(binding_arms, list) and len(binding_arms) == 5,
        "accepted or bound arm rows are malformed",
    )
    joined: dict[str, object] = {}
    for manifest_row, bound_arm, (arm_name, mode) in zip(arms, binding_arms, ARMS, strict=True):
        require(
            manifest_row.get("arm") == arm_name
            and manifest_row.get("representation_mode") == mode
            and bound_arm.get("arm") == arm_name
            and bound_arm.get("representation_mode") == mode,
            f"fixed arm/mode differs: {arm_name}",
        )
        sidecar, sidecar_pin = sidecars[arm_name]
        accepted_row = output_rows[arm_name]
        prediction = sidecar.get("prediction")
        require(
            isinstance(prediction, dict) and isinstance(accepted_row, dict),
            f"prediction/acceptance row malformed: {arm_name}",
        )
        accepted_prediction = accepted_row.get("prediction")
        accepted_checkpoint = accepted_row.get("checkpoint")
        require(
            isinstance(accepted_prediction, dict) and isinstance(accepted_checkpoint, dict),
            f"accepted output row malformed: {arm_name}",
        )
        checkpoint = {**bound_arm["model"], "loaded_sha256": bound_arm["model"]["sha256"]}
        require(
            sidecar.get("schema") == "fg19.first-cell-feature-only-prediction.v1"
            and sidecar.get("status") == "PREDICTION_EXPORT_ONLY_NOT_SCORED",
            f"prediction sidecar status differs: {arm_name}",
        )
        require(
            sidecar.get("checkpoint") == checkpoint
            and accepted_checkpoint == checkpoint
            and manifest_row.get("checkpoint") == checkpoint,
            f"fixed model checkpoint pin differs: {arm_name}",
        )
        require(
            sidecar.get("selected_epoch") == manifest_row.get("selected_epoch") == 5
            and sidecar.get("successful_updates")
            == manifest_row.get("successful_updates")
            == 44_760,
            f"fixed model endpoint differs: {arm_name}",
        )
        require(
            sidecar.get("terminal_acceptance") == bound_arm.get("terminal_acceptance")
            and sidecar.get("exact_stop") == bound_arm.get("exact_stop"),
            f"fit acceptance/stop lineage differs: {arm_name}",
        )
        require(
            sidecar.get("binding_sha256") == EXPORT_BINDING_SHA256
            and sidecar.get("representation_mode") == mode
            and sidecar.get("ordered_groups") == groups
            and sidecar.get("membership_sha256") == membership_sha,
            f"sidecar binding, mode, or membership differs: {arm_name}",
        )
        require(
            sidecar.get("feature_only") == {"targets_read": False, "scored_mask_read": False}
            and sidecar.get("scoring_performed") is False,
            f"export is not feature-only: {arm_name}",
        )
        require(
            prediction.get("shape") == PREDICTION_SHAPE
            and prediction.get("dtype") == "float32-little-endian"
            and prediction.get("data_bytes") == PREDICTION_DATA_BYTES
            and type(prediction.get("npy_header_bytes")) is int
            and 0 < prediction["npy_header_bytes"] <= 4_096
            and prediction.get("bytes") == PREDICTION_DATA_BYTES + prediction["npy_header_bytes"]
            and prediction.get("path") == "predictions.npy"
            and is_sha256(prediction.get("sha256")),
            f"prediction array contract differs: {arm_name}",
        )
        require(
            accepted_prediction == prediction
            and manifest_row.get("prediction_bytes") == prediction["bytes"]
            and manifest_row.get("prediction_sha256") == prediction["sha256"]
            and manifest_row.get("sidecar_bytes") == sidecar_pin["bytes"]
            and manifest_row.get("sidecar_sha256") == sidecar_pin["sha256"],
            f"accepted output metadata differs: {arm_name}",
        )
        arm_dir = EXPORT_ROOT / arm_name
        require(
            arm_dir.is_dir() and not arm_dir.is_symlink(),
            f"prediction arm directory is unsafe: {arm_name}",
        )
        array_path = arm_dir / "predictions.npy"
        array_info = path_stat(array_path, expected_symlink=False)
        require(
            array_info["bytes"] == prediction["bytes"],
            f"prediction array size differs before admission: {arm_name}",
        )
        joined[arm_name] = {
            "sidecar": sidecar_pin,
            "prediction_stat_only": array_info,
            "prediction_sha256_from_root_acceptance_not_rehashed": prediction["sha256"],
            "checkpoint_sha256_from_root_acceptance_not_rehashed": checkpoint["sha256"],
        }
    return joined


def check_export_bundle(payload: dict[str, bytes]) -> dict[str, object]:
    """Revalidate export metadata and stat arrays; leave array hashing/mapping to the worker."""
    binding_path = EXPORT_D / "EXPORT_BINDING_V1.json"
    manifest_path = EXPORT_ROOT / "EXPORT_MANIFEST_V1.json"
    binding_raw, binding_info = stable_file(binding_path, 2_000_000)
    require(
        (len(binding_raw), digest(binding_raw)) == (EXPORT_BINDING_BYTES, EXPORT_BINDING_SHA256),
        "read-only export binding changed",
    )
    binding = json_object(binding_raw, "export binding")
    require(
        binding.get("schema") == "fg19.first-cell-evaluation-export-binding.v1"
        and (
            binding.get("project_base"),
            binding.get("data_root"),
            binding.get("source_root"),
            binding.get("evidence_root"),
            binding.get("output_dir"),
        )
        == (str(PROJECT_BASE), str(DATA_ROOT), str(SOURCE_ROOT), str(EXPORT_D), str(EXPORT_ROOT)),
        "export binding paths differ",
    )
    manifest_raw, manifest_info = stable_file(manifest_path, 4_000_000)
    manifest_pin = {**manifest_info, "bytes": len(manifest_raw), "sha256": digest(manifest_raw)}
    manifest = json_object(manifest_raw, "export completion manifest")
    acceptance = json_object(payload["EXPORT_ACCEPTANCE.json"], "root export acceptance")
    exact_stop = json_object(payload["EXPORT_EXACT_STOP.json"], "root export exact stop")
    invocation = validate_export_authority(
        acceptance,
        exact_stop,
        acceptance_sha256=EXPORT_ACCEPTANCE_SHA256,
        binding_sha256=EXPORT_BINDING_SHA256,
        review_sha256=SOURCE_REVIEW_SHA256,
        expected_unit=EXPORT_UNIT,
    )
    expected_sidecars = {arm: f"{arm}/PREDICTION_SIDECAR.json" for arm, _mode in ARMS}
    sidecars: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for arm, relative in expected_sidecars.items():
        raw, info = stable_file(EXPORT_ROOT / relative, 2_000_000)
        row = next(
            item
            for item in manifest.get("arms", [])
            if isinstance(item, dict) and item.get("arm") == arm
        )
        require(
            (len(raw), digest(raw)) == (row.get("sidecar_bytes"), row.get("sidecar_sha256")),
            f"export sidecar pin differs: {arm}",
        )
        sidecars[arm] = (
            json_object(raw, f"{arm} sidecar"),
            {"path": relative, "bytes": len(raw), "sha256": digest(raw), **info},
        )
    joined = validate_acceptance_outputs(acceptance, binding, manifest, manifest_pin, sidecars)
    return {
        "export_binding": {**binding_info, "sha256": EXPORT_BINDING_SHA256, "read_only": True},
        "source_manifest": {
            "path": str(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json"),
            "bytes": SOURCE_MANIFEST_BYTES,
            "sha256": SOURCE_MANIFEST_SHA256,
            "read_only": True,
        },
        "export_manifest": {**manifest_info, "sha256": digest(manifest_raw), "read_only": True},
        "export_invocation_id": invocation,
        "five_accepted_outputs_stat_only": joined,
        "array_payloads_read_or_hashed_before_worker": False,
    }


def check_predecessors(fit_admission: dict[str, object]) -> dict[str, object]:
    """Recheck the fourteen readonly receipts, profile metadata, and two ancestor stats."""
    staged = fit_admission.get("staged_receipt_sha256")
    require(
        isinstance(staged, dict) and staged == READONLY_RECEIPTS,
        "AllThree predecessor receipt map differs",
    )
    receipts: dict[str, object] = {}
    for name, expected_sha in READONLY_RECEIPTS.items():
        raw, info = stable_file(FIT_RECEIPTS_ROOT / name, 1_000_000)
        require(digest(raw) == expected_sha, f"accepted predecessor receipt differs: {name}")
        receipts[name] = {**info, "sha256": expected_sha, "read_only": True}
    profile: dict[str, object] = {}
    for name, expected_sha in PROFILE_PINS.items():
        raw, info = stable_file(PROFILE_ROOT / name, 1_000_000)
        require(digest(raw) == expected_sha, f"profile metadata differs: {name}")
        profile[name] = {**info, "sha256": expected_sha, "read_only": True}
    ancestry = (
        (
            PROJECT_BASE / "data/artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt",
            fit_admission.get("parent_checkpoint_identity"),
            PARENT_CHECKPOINT_SHA256,
            3_988_711,
            779_788_689,
        ),
        (
            PROJECT_BASE / "data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt",
            fit_admission.get("dictionary"),
            DICTIONARY_SHA256,
            22_557,
            733_618_561,
        ),
    )
    observed: dict[str, object] = {}
    for path, recorded, expected_sha, expected_bytes, expected_inode in ancestry:
        require(
            isinstance(recorded, dict)
            and recorded.get("path") == str(path)
            and recorded.get("sha256") == expected_sha,
            "parent or dictionary accepted identity differs",
        )
        require(not path.is_symlink(), f"ancestor checkpoint path is a symlink: {path}")
        current = path.stat()
        require(
            stat.S_ISREG(current.st_mode)
            and (current.st_size, current.st_dev, current.st_ino)
            == (expected_bytes, EXPECTED_FILESYSTEM_DEVICE, expected_inode),
            f"ancestor checkpoint stat differs: {path}",
        )
        require(
            (recorded.get("bytes"), recorded.get("device"), recorded.get("inode"))
            == (current.st_size, current.st_dev, current.st_ino),
            f"ancestor checkpoint admission stat differs: {path}",
        )
        observed[path.name] = {
            "path": str(path),
            "bytes": current.st_size,
            "device": current.st_dev,
            "inode": current.st_ino,
            "accepted_sha256_not_rehashed": expected_sha,
        }
    return {
        "readonly_receipts": receipts,
        "profile_metadata": profile,
        "ancestor_artifact_stats_only": observed,
        "model_payloads_read_or_hashed": False,
    }


def check_cache(fit_admission: dict[str, object]) -> dict[str, object]:
    """Recheck cache receipts, metadata, and training-array stat identities only."""
    cache = fit_admission.get("cache")
    require(isinstance(cache, dict), "accepted cache metadata is absent")
    metadata_raw, metadata_info = stable_file(CACHE_ROOT / "metadata.json", MAX_METADATA_BYTES)
    acceptance_raw, acceptance_info = stable_file(
        CACHE_RECEIPTS_ROOT / "TERMINAL_ACCEPTANCE.json", 10_000
    )
    stop_raw, stop_info = stable_file(CACHE_RECEIPTS_ROOT / "EXACT_STOP.json", 10_000)
    require(
        digest(metadata_raw) == CACHE_METADATA_SHA256
        and digest(acceptance_raw) == CACHE_ACCEPTANCE_SHA256
        and digest(stop_raw) == CACHE_STOP_SHA256,
        "accepted cache metadata or receipts changed",
    )
    metadata = json_object(metadata_raw, "homogeneous cache metadata")
    acceptance = json_object(acceptance_raw, "homogeneous cache acceptance")
    require(
        metadata.get("split_manifest_sha256") == SPLIT_SHA256
        and acceptance.get("split_manifest_sha256") == SPLIT_SHA256,
        "homogeneous cache split identity differs",
    )
    stats, outputs = cache.get("array_stats_only"), acceptance.get("output")
    require(isinstance(stats, dict) and isinstance(outputs, dict), "cache array metadata missing")
    arrays: dict[str, object] = {}
    for name, size, count, expected_sha, expected_inode in (
        (
            "train",
            1_685_600_128,
            10_535,
            "4abbb9be31ec1135a5e0ff5bb87c93a2db6e44943e95d23ff70835b8044c5194",
            739_677_840,
        ),
        (
            "valid",
            224_800_128,
            1_405,
            "6963d3c565caf469b023a61ba6d2dd376d929b2bcf724af88492dc8adb76f591",
            739_677_841,
        ),
    ):
        path = CACHE_ROOT / f"main_homogeneous_{name}.npy"
        observed = path_stat(path, expected_symlink=False)
        recorded = stats.get(name)
        output = outputs.get(name)
        require(
            isinstance(recorded, dict) and isinstance(output, dict),
            f"cache {name} accepted stats are missing",
        )
        compared = {key: observed[key] for key in ("bytes", "device", "inode", "mtime_ns")}
        require(
            compared == recorded
            and (observed["bytes"], observed["device"], observed["inode"])
            == (size, EXPECTED_FILESYSTEM_DEVICE, expected_inode),
            f"cache {name} stat differs",
        )
        require(
            (output.get("bytes"), output.get("sha256"), output.get("shape"), output.get("dtype"))
            == (size, expected_sha, [count, 20_000, 2], "float32"),
            f"cache {name} acceptance differs",
        )
        arrays[name] = {**compared, "sha256_from_acceptance_not_rehashed": expected_sha}
    return {
        "metadata": {**metadata_info, "sha256": CACHE_METADATA_SHA256},
        "acceptance": {**acceptance_info, "sha256": CACHE_ACCEPTANCE_SHA256},
        "exact_stop": {**stop_info, "sha256": CACHE_STOP_SHA256},
        "array_stat_only": arrays,
        "cache_payloads_read_or_hashed": False,
    }


def check_parent_resources() -> dict[str, object]:
    """Require the two-core parent envelope, empty scoped cgroup, and physical scratch."""
    parent = show(
        "wnn-campaign.slice",
        ("ControlGroup", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax"),
    )
    require(
        (
            parent.get("CPUQuotaPerSecUSec"),
            parent.get("MemoryHigh"),
            parent.get("MemoryMax"),
            parent.get("MemorySwapMax"),
        )
        == ("2s", "18000000000", "20000000000", "0"),
        "aggregate parent resource envelope differs",
    )
    group = parent.get("ControlGroup", "")
    require(
        group not in ("", "/") and group.endswith("/wnn-campaign.slice"),
        "parent cgroup is not narrowly scoped",
    )
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    require(
        root.is_dir() and root.resolve(strict=True) == root,
        "parent cgroup is unavailable or redirected",
    )
    inventory: list[Path] = []
    stack = [root]
    total_dirs = 0
    aggregate = 0
    populated: dict[str, str] = {}
    while stack:
        current = stack.pop()
        total_dirs += 1
        require(total_dirs <= 128, "parent cgroup directory cap exceeded")
        count = 0
        with os.scandir(current) as entries:
            for entry in entries:
                count += 1
                require(count <= 256, "parent cgroup entry cap exceeded")
                require(not entry.is_symlink(), "parent cgroup traversal encountered symlink")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    require(total_dirs + len(stack) <= 128, "parent cgroup directory cap exceeded")
        inventory.append(current / "cgroup.procs")
    require(0 < len(inventory) <= 128, "parent cgroup process-file inventory differs")
    for path in inventory:
        raw = _bounded_proc_read(path, 32_768)
        aggregate += len(raw)
        require(aggregate <= 262_144, "parent cgroup process metadata aggregate exceeds cap")
        if raw.strip():
            populated[str(path)] = raw.decode("ascii", errors="strict")
    require(not populated, "campaign parent cgroup is populated")
    current_raw = _bounded_proc_read(root / "memory.current", 4_096).decode("ascii").strip()
    require(current_raw.isdigit(), "parent memory.current is unavailable")
    current = int(current_raw)
    require(
        current + WORKER_MEMORY_MAX_BYTES + PARENT_MEMORY_RESERVE_BYTES
        < PARENT_MEMORY_TARGET_BYTES,
        "aggregate memory headroom is insufficient",
    )
    events = {
        name: _bounded_proc_read(root / name, 4_096).decode("ascii")
        for name in ("memory.events", "memory.events.local")
    }
    local = dict(
        line.split()
        for line in events["memory.events.local"].splitlines()
        if len(line.split()) == 2
    )
    require(
        all(local.get(key) == "0" for key in ("oom", "oom_kill", "oom_group_kill")),
        "parent local OOM counters are nonzero",
    )
    require(
        SCRATCH_ROOT.is_dir()
        and not SCRATCH_ROOT.is_symlink()
        and SCRATCH_ROOT.resolve(strict=True) == SCRATCH_ROOT,
        "physical scratch root is redirected",
    )
    device = PROJECT_BASE.stat().st_dev
    require(
        device == SCRATCH_ROOT.stat().st_dev == EXPECTED_FILESYSTEM_DEVICE,
        "project scratch filesystem device differs",
    )
    filesystem = subprocess.run(
        ["findmnt", "-T", str(SCRATCH_ROOT), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        filesystem.returncode == 0 and filesystem.stdout.strip() == "ext4",
        "scratch filesystem is not ext4",
    )
    free = os.statvfs(SCRATCH_ROOT).f_bavail * os.statvfs(SCRATCH_ROOT).f_frsize
    require(
        free >= MINIMUM_FILESYSTEM_FREE_BYTES,
        "physical scratch free space is below the fixed reserve",
    )
    return {
        "parent_properties": parent,
        "cgroup": str(root),
        "populated": populated,
        "memory_current_bytes": current,
        "memory_events": events,
        "filesystem_device": device,
        "scratch_filesystem": filesystem.stdout.strip(),
        "scratch_free_bytes": free,
        "minimum_scratch_free_bytes": MINIMUM_FILESYSTEM_FREE_BYTES,
    }


def read_gpu_owner_details(
    proc: Path,
    owner_uid: int,
    current_uid: int,
    read_proc: Callable[[Path, int], bytes],
) -> tuple[str, str | None]:
    """Read each GPU owner's cgroup and same-UID command metadata only."""
    cgroup = read_proc(proc / "cgroup", 32_768).decode("ascii", errors="strict").strip()
    if owner_uid != current_uid:
        return cgroup, None
    command = read_proc(proc / "cmdline", 8_192).decode("utf-8", errors="replace")
    return cgroup, command


def check_selected_gpu() -> dict[str, object]:
    """Observe all selected-GPU owners and preserve every process without action."""
    prefix = ["nvidia-smi", "-i", GPU_UUID]
    identity, _ = _bounded_gpu_query(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    )
    lines = [line.strip() for line in identity.splitlines() if line.strip()]
    require(len(lines) == 1, "selected GPU inventory did not return one device")
    fields = [part.strip() for part in lines[0].split(",")]
    require(
        len(fields) == 4 and fields[0] == GPU_UUID and fields[1], "selected GPU identity differs"
    )
    free, total = _memory_bytes(fields[2]), _memory_bytes(fields[3])
    require(total > 0 and free <= total, "selected GPU memory metadata is inconsistent")
    listing, _ = _bounded_gpu_query(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"]
    )
    process_rows = [line.strip() for line in listing.splitlines() if line.strip()]
    if len(process_rows) == 1 and process_rows[0].startswith("No running processes found"):
        process_rows = []
    require(len(process_rows) <= 256, "selected GPU process inventory exceeds cap")
    owners: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in process_rows:
        parts = [part.strip() for part in row.split(",")]
        require(
            len(parts) == 3
            and parts[0].isascii()
            and parts[0].isdigit()
            and int(parts[0]) > 0
            and str(int(parts[0])) == parts[0]
            and parts[1] == GPU_UUID
            and parts[0] not in seen,
            "selected GPU process row is malformed or duplicated",
        )
        seen.add(parts[0])
        proc = Path("/proc") / parts[0]
        before, start_before = proc.stat(follow_symlinks=False), _proc_starttime(proc, parts[0])
        cgroup, command = read_gpu_owner_details(
            proc, before.st_uid, os.getuid(), _bounded_proc_read
        )
        after, start_after = proc.stat(follow_symlinks=False), _proc_starttime(proc, parts[0])
        require(
            (before.st_ino, before.st_uid, start_before)
            == (after.st_ino, after.st_uid, start_after),
            "GPU owner PID changed during observation",
        )
        campaign_owned = "wnn-campaign.slice" in cgroup or (
            command is not None and str(PROJECT_BASE) in command
        )
        require(not campaign_owned, "campaign project process currently owns the selected GPU")
        owners.append(
            {
                "pid": int(parts[0]),
                "gpu_uuid": parts[1],
                "used_memory": parts[2],
                "uid": after.st_uid,
                "cgroup": cgroup,
                "cgroup_inspected": True,
                "cmdline_inspected": command is not None,
                "proc_starttime": start_before,
                "preserved": True,
                "action": "none",
            }
        )
    return {
        "uuid": GPU_UUID,
        "name": fields[1],
        "free_bytes": free,
        "total_bytes": total,
        "process_count": len(owners),
        "owners": owners,
        "foreign_process_action": "none",
    }


def check_paths_and_inputs(payload: dict[str, bytes]) -> dict[str, object]:
    """Perform every read-only path, receipt, source, data, and result-absence gate."""
    require(
        PROJECT_BASE.is_dir()
        and not PROJECT_BASE.is_symlink()
        and PROJECT_BASE.resolve(strict=True) == PROJECT_BASE,
        "project base is redirected",
    )
    require(
        DATA_ROOT.is_dir()
        and not DATA_ROOT.is_symlink()
        and DATA_ROOT.resolve(strict=True) == DATA_ROOT,
        "data root is redirected",
    )
    require(
        CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
        "canonical project Python is unavailable",
    )
    require(
        EXPORT_D.is_dir()
        and not EXPORT_D.is_symlink()
        and EXPORT_D.resolve(strict=True) == EXPORT_D,
        "accepted export evidence root is missing or redirected",
    )
    require(
        EXPORT_ROOT.is_dir()
        and not EXPORT_ROOT.is_symlink()
        and EXPORT_ROOT.resolve(strict=True) == EXPORT_ROOT,
        "accepted prediction export root is missing or redirected",
    )
    require(
        not os.path.lexists(SCORE_D), "private score directory already exists; never reuse or retry"
    )
    require(not os.path.lexists(RESULT_PATH), "create-only score result path already exists")
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found",
        "score support unit already exists; never reuse",
    )
    binding_raw, _ = stable_file(EXPORT_D / "EXPORT_BINDING_V1.json", 2_000_000)
    binding = json_object(binding_raw, "export binding")
    require(
        (len(binding_raw), digest(binding_raw)) == (EXPORT_BINDING_BYTES, EXPORT_BINDING_SHA256),
        "fixed export binding pin differs",
    )
    manifest_raw, source_info = stable_file(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json", 2_000_000)
    require(
        (len(manifest_raw), digest(manifest_raw))
        == (SOURCE_MANIFEST_BYTES, SOURCE_MANIFEST_SHA256),
        "source manifest pin differs",
    )
    fit_admission_raw, fit_admission_info = stable_file(
        FIT_RECEIPTS_ROOT / "ADMISSION.json", 20_000
    )
    fit_launch_raw, fit_launch_info = stable_file(FIT_RECEIPTS_ROOT / "LAUNCH.json", 20_000)
    require(
        (len(fit_admission_raw), digest(fit_admission_raw))
        == (FIT_ADMISSION_BYTES, FIT_ADMISSION_SHA256),
        "original AllThree fit admission differs",
    )
    require(
        (len(fit_launch_raw), digest(fit_launch_raw)) == (FIT_LAUNCH_BYTES, FIT_LAUNCH_SHA256),
        "original AllThree fit launch differs",
    )
    fit_admission = json_object(fit_admission_raw, "original AllThree admission")
    fit_launch = json_object(fit_launch_raw, "original AllThree launch")
    require(
        fit_admission.get("run_name") == "fg19_child_allthree_f0_s1001_fit_v1"
        and fit_admission.get("unit")
        == "computer-vision-homework-fg19-child-allthree-fit-v1.service"
        and fit_launch.get("unit") == fit_admission.get("unit")
        and is_invocation_id(fit_launch.get("invocation_id")),
        "original AllThree fit identity differs",
    )
    require(
        fit_launch.get("invocation_id") == "94d3a7132a434d448345b0206bd3daa3",
        "original AllThree invocation differs",
    )
    require(
        not SPLIT_PATH.is_symlink() and SPLIT_PATH.resolve(strict=True) == SPLIT_PATH,
        "canonical split path is redirected",
    )
    split_raw, split_info = stable_file(SPLIT_PATH, 2_000_000)
    require(
        digest(split_raw) == SPLIT_SHA256 and json_object(split_raw, "canonical split"),
        "canonical F0 split bytes differ",
    )
    require(
        binding.get("split", {}).get("sha256") == SPLIT_SHA256
        and binding.get("source", {}).get("commit") == SOURCE_COMMIT,
        "export binding source or split differs",
    )
    registered = check_registered_data(binding, fit_admission)
    source_tree = check_source_tree(binding, manifest_raw)
    predecessors = check_predecessors(fit_admission)
    cache = check_cache(fit_admission)
    export_bundle = check_export_bundle(payload)
    parent = check_parent_resources()
    gpu = check_selected_gpu()
    model_stats: dict[str, object] = {}
    for arm in binding.get("arms", []):
        model = arm.get("model")
        require(isinstance(model, dict), "accepted model checkpoint metadata is malformed")
        relative = Path(str(model.get("relative_path", "")))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "unsafe accepted checkpoint path",
        )
        path = DATA_ROOT / relative
        ancestor = path
        while ancestor != DATA_ROOT:
            require(
                not ancestor.is_symlink(), f"checkpoint path traverses a symbolic link: {ancestor}"
            )
            ancestor = ancestor.parent
        observed = path_stat(path, expected_symlink=False)
        require(
            observed["bytes"] == model.get("bytes"),
            f"accepted checkpoint size differs: {arm.get('arm')}",
        )
        model_stats[str(arm.get("arm"))] = {
            key: observed[key] for key in ("path", "bytes", "device", "inode", "mtime_ns")
        }
    sources_unchanged = stable_file(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json", 2_000_000)
    require(sources_unchanged[0] == manifest_raw, "source manifest changed during preflight")
    return {
        "binding": {"bytes": len(binding_raw), "sha256": digest(binding_raw), "read_only": True},
        "source_manifest": {
            **source_info,
            "sha256": digest(manifest_raw),
            "file_count": 167,
            "read_only": True,
        },
        "source_tree": source_tree,
        "split": {**split_info, "sha256": SPLIT_SHA256},
        "registered_six_sources_stat_only": registered,
        "fit_admission": {
            **fit_admission_info,
            "sha256": digest(fit_admission_raw),
            "read_only": True,
        },
        "fit_launch": {**fit_launch_info, "sha256": digest(fit_launch_raw), "read_only": True},
        "predecessors_profile_parent_dictionary": predecessors,
        "cache_stat_only": cache,
        "accepted_export": export_bundle,
        "fit_checkpoints_stat_only": model_stats,
        "parent_disk_cgroup_headroom": parent,
        "selected_gpu_owners": gpu,
        "result_absent_before_mutation": True,
        "score_unit_absent_before_mutation": True,
    }


def stage_packet(payload: dict[str, bytes]) -> dict[str, object]:
    """Create the private scorer directory and stage only four modules plus root receipts."""
    SCORE_D.mkdir(mode=0o700, exist_ok=False)
    accepted_dir = SCORE_D / "accepted_export"
    accepted_dir.mkdir(mode=0o700, exist_ok=False)
    temp_dir = SCORE_D / "tmp"
    temp_dir.mkdir(mode=0o700, exist_ok=False)
    for name in SCORER_FILES:
        write_exclusive(SCORE_D / name, payload[name])
    write_exclusive(accepted_dir / "EXPORT_ACCEPTANCE.json", payload["EXPORT_ACCEPTANCE.json"])
    write_exclusive(accepted_dir / "EXPORT_EXACT_STOP.json", payload["EXPORT_EXACT_STOP.json"])
    for directory in (accepted_dir, temp_dir, SCORE_D):
        fd = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    staged: dict[str, object] = {}
    for relative, raw in (
        *((name, payload[name]) for name in SCORER_FILES),
        ("accepted_export/EXPORT_ACCEPTANCE.json", payload["EXPORT_ACCEPTANCE.json"]),
        ("accepted_export/EXPORT_EXACT_STOP.json", payload["EXPORT_EXACT_STOP.json"]),
    ):
        disk, info = stable_file(SCORE_D / relative, len(raw))
        require(disk == raw, f"staged source/receipt changed: {relative}")
        staged[relative] = {**info, "sha256": digest(raw)}
    return staged


def bounded_submit(command: list[str]) -> tuple[int, bytes, bytes]:
    """Capture one systemd-run response within a fixed timeout and stream cap."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        start_new_session=True,
    )
    require(
        process.stdout is not None and process.stderr is not None, "systemd-run pipes unavailable"
    )
    output = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + 30
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                require(
                    remaining > 0,
                    "submission response timed out; outcome is unknown and must not be retried",
                )
                events = selector.select(remaining)
                require(
                    events,
                    "submission response timed out; outcome is unknown and must not be retried",
                )
                for key, _ in events:
                    label = streams[key.fileobj]
                    chunk = os.read(key.fileobj.fileno(), min(4096, 65_537 - len(output[label])))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output[label].extend(chunk)
                    require(
                        len(output[label]) <= 65_536,
                        "submission output exceeded stream cap; outcome unknown",
                    )
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, RuntimeError, subprocess.SubprocessError):
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    return returncode, bytes(output["stdout"]), bytes(output["stderr"])


def submit_once(admission: dict[str, object]) -> dict[str, object]:
    """Submit exactly one CPU score unit and durably capture its invocation ID."""
    command = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=" + WORKER_CPU_QUOTA,
        "--property=MemoryHigh=" + str(WORKER_MEMORY_HIGH_BYTES),
        "--property=MemoryMax=" + str(WORKER_MEMORY_MAX_BYTES),
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=" + str(WORKER_RUNTIME_SECONDS),
        "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(SCORE_D / "unit.log"),
        "--property=StandardError=append:" + str(SCORE_D / "unit.log"),
        "--working-directory=" + str(SCORE_D),
        "--setenv=CUDA_VISIBLE_DEVICES=",
        "--setenv=WNN_PROJECT_ROOT=" + str(DATA_ROOT),
        "--setenv=WNN_PORTABLE_DATA_MANIFEST=" + str(PORTABLE_REGISTRY),
        "--setenv=WNN_THREADS=1",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--setenv=VECLIB_MAXIMUM_THREADS=1",
        "--setenv=TMPDIR=" + str(SCORE_D / "tmp"),
        str(CANONICAL_PYTHON),
        "-E",
        "-B",
        "-s",
        str(SCORE_D / "score_first_cell.py"),
        "--evidence-root",
        str(EXPORT_D),
        "--export-root",
        str(EXPORT_ROOT),
        "--binding",
        str(EXPORT_D / "EXPORT_BINDING_V1.json"),
        "--source-manifest",
        str(EXPORT_D / "EXPORT_SOURCE_MANIFEST.json"),
        "--source-root",
        str(SOURCE_ROOT),
        "--data-root",
        str(DATA_ROOT),
        "--out",
        str(RESULT_PATH),
    ]
    try:
        returncode, stdout, stderr = bounded_submit(command)
        submission: dict[str, object] = {
            "outcome": "submitted" if returncode == 0 else "submit_failed_or_unknown",
            "returncode": returncode,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "argv": command,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        submission = {
            "outcome": "submit_error_or_timeout_unknown_never_retry",
            "returncode": None,
            "error": repr(error),
            "stdout_b64": "",
            "stderr_b64": "",
            "argv": command,
        }
    fields = (
        "Id",
        "LoadState",
        "InvocationID",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
        "ControlGroup",
    )
    post: dict[str, object] = {"outcome": "captured", "properties": {}}
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            properties = show(UNIT, fields)
            post["properties"] = properties
            if properties.get("Id") == UNIT and is_invocation_id(properties.get("InvocationID")):
                break
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            post = {"outcome": "query_failed_unknown", "error": repr(error), "properties": {}}
        time.sleep(0.25)
    properties = post.get("properties", {})
    invocation = properties.get("InvocationID") if isinstance(properties, dict) else None
    if not is_invocation_id(invocation):
        write_json_exclusive(
            SCORE_D / "LAUNCH_UNRESOLVED.json",
            {
                "schema": "fg19.first-cell-score-unresolved-launch.v1",
                "unit": UNIT,
                "invocation_id": None,
                "submission": submission,
                "systemctl_after_submission": post,
                "unknown_outcome_policy": "Inspect only this exact unit and score directory; never retry.",
            },
        )
        raise RuntimeError(
            "score invocation ID unresolved; durable unknown outcome retained, never retry"
        )
    launch = {
        "schema": "fg19.first-cell-score-launch.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "invocation_id": invocation,
        "export_unit": EXPORT_UNIT,
        "export_invocation_id": admission["export_invocation_id"],
        "admission_sha256": digest(
            (json.dumps(admission, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        ),
        "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Inspect only this exact unit and score directory; never retry.",
    }
    write_json_exclusive(SCORE_D / "LAUNCH.json", launch)
    if submission.get("outcome") != "submitted" or submission.get("returncode") != 0:
        raise RuntimeError(
            "submission outcome is not one confirmed launch; preserve records and never retry"
        )
    return launch


def main() -> None:
    """Run all read-only gates before staging the one private CPU score packet."""
    payload = payload_bytes()
    acceptance = json_object(payload["EXPORT_ACCEPTANCE.json"], "root export acceptance")
    exact_stop = json_object(payload["EXPORT_EXACT_STOP.json"], "root export exact stop")
    checks = check_paths_and_inputs(payload)
    export_invocation = validate_export_authority(
        acceptance,
        exact_stop,
        acceptance_sha256=EXPORT_ACCEPTANCE_SHA256,
        binding_sha256=EXPORT_BINDING_SHA256,
        review_sha256=SOURCE_REVIEW_SHA256,
        expected_unit=EXPORT_UNIT,
    )
    staged = stage_packet(payload)
    admission = {
        "schema": "fg19.first-cell-score-admission.v1",
        "status": "FRESH_GATES_PASSED_SUBMISSION_PENDING",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "export_unit": EXPORT_UNIT,
        "export_invocation_id": export_invocation,
        "export_acceptance": {
            "bytes": EXPORT_ACCEPTANCE_BYTES,
            "sha256": EXPORT_ACCEPTANCE_SHA256,
            "staged_path": "accepted_export/EXPORT_ACCEPTANCE.json",
        },
        "export_exact_stop": {
            "bytes": EXPORT_EXACT_STOP_BYTES,
            "sha256": EXPORT_EXACT_STOP_SHA256,
            "staged_path": "accepted_export/EXPORT_EXACT_STOP.json",
        },
        "checks": checks,
        "staged_files": staged,
        "result_path": str(RESULT_PATH),
        "result_create_only_and_absent_before_mutation": True,
        "resource_limits": {
            "cpu_quota": WORKER_CPU_QUOTA,
            "memory_high_bytes": WORKER_MEMORY_HIGH_BYTES,
            "memory_max_bytes": WORKER_MEMORY_MAX_BYTES,
            "memory_swap_max_bytes": 0,
            "runtime_seconds": WORKER_RUNTIME_SECONDS,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "scope": "One source-qualified streaming score of the fixed five epoch-5 F0/seed-1001 arms over 468 exposed development groups; no inference, fit, retuning, model load, new endpoint selection, or public-transfer claim.",
        "unknown_outcome_policy": "Never retry or recreate this unit, score directory, or create-only result path; inspect only the exact durable receipt and unit.",
    }
    write_json_exclusive(SCORE_D / "ADMISSION.json", admission)
    write_exclusive(SCORE_D / "unit.log", b"")
    directory_fd = os.open(
        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    launch = submit_once(admission)
    admission_raw, _ = stable_file(SCORE_D / "ADMISSION.json", 1_000_000)
    launch_raw, _ = stable_file(SCORE_D / "LAUNCH.json", 1_000_000)
    require(
        len(admission_raw) + len(launch_raw) <= 4_000_000,
        "bounded score launch response exceeds 4 MB",
    )
    durable_admission = json_object(admission_raw, "durable score admission")
    durable_launch = json_object(launch_raw, "durable score launch")
    require(
        durable_admission.get("unit") == UNIT
        and durable_launch.get("unit") == UNIT
        and durable_launch.get("invocation_id") == launch.get("invocation_id")
        and is_invocation_id(durable_launch.get("invocation_id"))
        and durable_launch.get("submission", {}).get("outcome") == "submitted"
        and durable_launch.get("submission", {}).get("returncode") == 0,
        "durable score receipts do not bind one successful launch",
    )
    print(
        json.dumps(
            {
                "unit": UNIT,
                "outcome": "submitted",
                "admission_base64": base64.b64encode(admission_raw).decode("ascii"),
                "launch_base64": base64.b64encode(launch_raw).decode("ascii"),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
