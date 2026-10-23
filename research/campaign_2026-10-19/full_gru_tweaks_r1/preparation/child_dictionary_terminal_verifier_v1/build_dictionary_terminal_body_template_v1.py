"""Build the conditional dictionary body from reviewed EMA V3 gate mechanics."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent
BASE = CAMPAIGN / "preparation/child_ema_terminal_verifier_v3/REMOTE_SUBMISSION_TEMPLATE.py"
TEMPLATE = OUTPUT / "REMOTE_SUBMISSION_TEMPLATE.py"
BASE_BYTES = 59_676
BASE_SHA256 = "31bc9986f3d6e0728d36a723cc954899727150afc9d965cf91993c7c0d3f1a12"

HEADER = '''"""One-shot, create-only CPU admission for the dictionary integrity check."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path

PAYLOAD = {}
FINAL_REVIEW_BINDING = {}
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA = B / "data"
RUN = DATA / "artifacts/runs/fg19_child_dictionary_f0_s1001_fit_v1"
SOURCE = B / "source-fg19-cache-68ff6829"
SPLIT = B / "splits/fold0.json"
CACHE = B / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"
DICT = DATA / "artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
PARENT_CHECKPOINT = DATA / "artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt"
REGISTRY = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
EVIDENCE = B / "research_tmp/fg19_child_dictionary_fit_v1"
PROFILE_EVIDENCE = B / "research_tmp/fg19_child_allthree_profile_v1"
CACHE_RECEIPTS = B / "research/full_gru_tweaks_r1/cache_cpu_v2"
D = B / "research_tmp/fg19_child_dictionary_verify_v1"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
UNIT = "computer-vision-homework-fg19-child-dictionary-verify-v1.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
PATH_EXCEPTION_ENV = "FG19_APPROVED_PATH_COMPAT_EXCEPTION"
PATH_EXCEPTION_VALUE = "pathlib._local.PosixPath:exact-dictionary-path-only"
INVOCATION = "73b10b699e034a1b9952b2377b845095"
RUN_NAME = "fg19_child_dictionary_f0_s1001_fit_v1"
CONTROL_RUN = "fg19_child_control_f0_s1001_fit_v1"
CONTROL_UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
CONTROL_INVOCATION = "e66e0d0703314e4fb5d47dcdbfcaf202"
EMA_RUN = "fg19_child_ema_f0_s1001_fit_v1"
EMA_UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
EMA_INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
EMA_MODEL_EP5_SHA = "6e90bc4dba8add0caf670097519f08f3ce239d0b64c37c47b3760aa22af41862"
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
ADMISSION_SHA = "b13ccf5834fd210ce02ca6c3aea7251d8dcae3f63823341da15a77abcbc75063"
LAUNCH_SHA = "9cbcc8b4df3886620429cae42d678d2e281aff980402f5b19d63d9f515b6ad3c"
LAUNCH_ACCEPTANCE_SHA = "33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95"
BODY_SHA = "9f545489dd80c13831b1a02c75dddf4059f3cbfb84311b2400db4830b4fdd77e"
BODY_BYTES = 196_111
SOURCE_REVIEW_SHA = "67528bf25430a7bd422e233f3ce84059b14d372f59b9b033cdd70e674634528a"
SOURCE_REVIEW_BYTES = 7_213
ROOT_FIT_PREPARATION_SHA = "2c0ee4b298a744f3c3204c46f5a269f3ea03c73d1a521c6db05dfa7a4144cf66"
ROOT_FIT_PREPARATION_BYTES = 2_343
ROOT_REVIEW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_terminal_check_v1/ROOT_PREPARATION.json"
TERMINAL_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/TERMINAL_EVIDENCE.json"
NATIVE_RAW_PATH = "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"
TERMINAL_SHA = "__RENDER_TERMINAL_SHA256__"
NATIVE_RAW_SHA = "__RENDER_NATIVE_RAW_SHA256__"
NATIVE_RAW_BYTES = 0
CHECKER_SHA = "__RENDER_CHECKER_SHA256__"
CHECKER_BYTES = 0
CONTROL_ACCEPTANCE_SHA = "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
CONTROL_STOP_SHA = "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
PARENT_ACCEPTANCE_SHA = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
PARENT_STOP_SHA = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
PARENT_CHECKPOINT_SHA = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
DICTIONARY_ACCEPTANCE_SHA = "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
DICTIONARY_STOP_SHA = "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158"
DICTIONARY_SHA = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
PROFILE_ACCEPTANCE_SHA = "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045"
PROFILE_STOP_SHA = "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce"
PROFILE_ADMISSION_SHA = "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5"
PROFILE_LAUNCH_SHA = "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44"
PROFILE_LOG_SHA = "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289"
EMA_ACCEPTANCE_SHA = "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8"
EMA_STOP_SHA = "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf"
CACHE_METADATA_SHA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_EXACT_STOP_SHA = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
REGISTRY_SHA = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
CONTROL_CHECKER_SHA = "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167"
PARENT_CHECKER_SHA = "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41"
SUPPORT_SHA = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
TEACHER_SHA = "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5"
EMA_CHECKER_SHA = "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d"
MAX_SOURCE_BYTES = 128_000_000
MAX_JSON_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_ARTIFACT_BYTES = 128_000_000
SUPPORT_MAX_BYTES = 1_500_000_000
PARENT_TARGET_BYTES = 17_000_000_000
PARENT_RESERVE_BYTES = 1_000_000_000
EXPECTED_HASHES = {
    "child_dictionary_terminal_check_v1.py": CHECKER_SHA,
    "child_control_terminal_check_v3.py": CONTROL_CHECKER_SHA,
    "parent_fit_terminal_check_v3.py": PARENT_CHECKER_SHA,
    "propensity_terminal_support.py": SUPPORT_SHA,
    "teacher_s5_terminal_check_v1.py": TEACHER_SHA,
    "child_ema_terminal_check_v3.py": EMA_CHECKER_SHA,
    "SOURCE_MANIFEST.json": MANIFEST_SHA,
    "LAUNCH_ACCEPTANCE.json": LAUNCH_ACCEPTANCE_SHA,
    "TERMINAL_EVIDENCE.json": TERMINAL_SHA,
    "NATIVE_STDOUT.raw.json": NATIVE_RAW_SHA,
}
MAX_INPUT_BYTES = {
    "child_dictionary_terminal_check_v1.py": 80_000,
    "child_control_terminal_check_v3.py": 80_000,
    "parent_fit_terminal_check_v3.py": 70_000,
    "propensity_terminal_support.py": 15_000,
    "teacher_s5_terminal_check_v1.py": 40_000,
    "child_ema_terminal_check_v3.py": 40_000,
    "SOURCE_MANIFEST.json": 30_000,
    "LAUNCH_ACCEPTANCE.json": 6_000,
    "TERMINAL_EVIDENCE.json": 16_000,
    "NATIVE_STDOUT.raw.json": 2_000_000,
}
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE_SHA,
    "PARENT_EXACT_STOP.json": PARENT_STOP_SHA,
    "DICTIONARY_ACCEPTANCE.json": DICTIONARY_ACCEPTANCE_SHA,
    "DICTIONARY_EXACT_STOP.json": DICTIONARY_STOP_SHA,
    "PROFILE_TERMINAL_ACCEPTANCE.json": PROFILE_ACCEPTANCE_SHA,
    "PROFILE_EXACT_STOP.json": PROFILE_STOP_SHA,
    "CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA,
    "CONTROL_EXACT_STOP.json": CONTROL_STOP_SHA,
    "EMA_TERMINAL_ACCEPTANCE.json": EMA_ACCEPTANCE_SHA,
    "EMA_EXACT_STOP.json": EMA_STOP_SHA,
}
'''

CHECK_NATIVE = '''def check_native(
    payload: dict[str, bytes],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Bind the actual dictionary launch, strict native exit evidence, and current unit."""
    admission_raw, _ = stable_file(EVIDENCE / "ADMISSION.json", 20_000)
    launch_raw, _ = stable_file(EVIDENCE / "LAUNCH.json", 20_000)
    require(digest(admission_raw) == ADMISSION_SHA, "original dictionary admission digest differs")
    require(digest(launch_raw) == LAUNCH_SHA, "original dictionary launch digest differs")
    admission = json_object(admission_raw, "dictionary admission")
    launch = json_object(launch_raw, "dictionary launch")
    accepted = json_object(payload["LAUNCH_ACCEPTANCE.json"], "dictionary launch acceptance")
    terminal_raw = payload["TERMINAL_EVIDENCE.json"]
    terminal = json_object(terminal_raw, "dictionary native terminal evidence")
    capture_raw = payload["NATIVE_STDOUT.raw.json"]
    capture = json_object(capture_raw, "dictionary native raw capture")
    control_predecessor = {
        "acceptance_sha256": CONTROL_ACCEPTANCE_SHA,
        "exact_stop_sha256": CONTROL_STOP_SHA,
        "invocation_id": CONTROL_INVOCATION,
        "run_name": CONTROL_RUN,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "unit": CONTROL_UNIT,
    }
    ema_predecessor = {
        "acceptance_sha256": EMA_ACCEPTANCE_SHA,
        "exact_stop_sha256": EMA_STOP_SHA,
        "invocation_id": EMA_INVOCATION,
        "model_ep5_sha256": EMA_MODEL_EP5_SHA,
        "run_name": EMA_RUN,
        "selected_epoch": 5,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "successful_updates": 44_760,
        "unit": EMA_UNIT,
    }
    require(
        (
            admission.get("schema"), admission.get("run_name"), admission.get("unit"),
            admission.get("source_commit"), admission.get("source_files_verified"),
            admission.get("source_manifest_sha256"), admission.get("split_sha256"),
            admission.get("control_predecessor"), admission.get("ema_predecessor"),
        )
        == (
            "fg19-child-dictionary-fit-admission.v1", RUN_NAME, NATIVE_UNIT, COMMIT, 167,
            MANIFEST_SHA, SPLIT_SHA, control_predecessor, ema_predecessor,
        ),
        "original dictionary admission identity or predecessor binding differs",
    )
    require(admission.get("populated") == {}, "original fit admission had populated descendants")
    require(
        admission.get("staged_receipt_sha256") == READONLY_RECEIPTS,
        "original staged predecessor receipt map differs",
    )
    submission = launch.get("submission")
    require(
        (
            launch.get("schema"), launch.get("unit"), launch.get("control_predecessor"),
            launch.get("ema_predecessor"), isinstance(submission, dict),
            submission.get("outcome") if isinstance(submission, dict) else None,
            submission.get("returncode") if isinstance(submission, dict) else None,
        )
        == (
            "fg19-child-dictionary-fit-launch.v1", NATIVE_UNIT, control_predecessor,
            ema_predecessor, True, "submitted", 0,
        ),
        "original dictionary launch identity differs",
    )
    expected_refs = {
        "admission": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/ADMISSION.json",
            "bytes": 10_937, "sha256": ADMISSION_SHA,
        },
        "launch": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/LAUNCH.json",
            "bytes": 8_072, "sha256": LAUNCH_SHA,
        },
        "body": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/REMOTE_SUBMISSION.py",
            "bytes": BODY_BYTES, "sha256": BODY_SHA,
        },
        "source_review": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/DICTIONARY_FINAL_BODY_SOURCE_REVIEW_V2.json",
            "bytes": SOURCE_REVIEW_BYTES, "sha256": SOURCE_REVIEW_SHA,
        },
        "root_preparation": {
            "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_dictionary_fit_v1/ROOT_ADMISSION_PREPARATION.json",
            "bytes": ROOT_FIT_PREPARATION_BYTES, "sha256": ROOT_FIT_PREPARATION_SHA,
        },
    }
    require(
        (
            accepted.get("schema"), accepted.get("status"), accepted.get("run_name"),
            accepted.get("unit"), accepted.get("invocation_id"),
            {key: accepted.get(key) for key in expected_refs},
            accepted.get("source_commit"), accepted.get("source_file_count"),
            accepted.get("source_manifest_sha256"), accepted.get("representation_mode"),
            accepted.get("mean_variance_objective"), accepted.get("native_exit_observed"),
            accepted.get("expected_terminal_updates_not_observed"), accepted.get("quality_score"),
        )
        == (
            "fg19-child-dictionary-fit-launch-acceptance.v1", "ACCEPTED_ACTUAL_ONE_LAUNCH_ONLY",
            RUN_NAME, NATIVE_UNIT, INVOCATION, expected_refs, COMMIT, 167, MANIFEST_SHA,
            "sparse", "wmse", False, 44_760, None,
        ),
        "original dictionary root launch acceptance differs",
    )
    require(
        (
            terminal.get("schema"), terminal.get("run_name"), terminal.get("unit"),
            terminal.get("invocation_id"), terminal.get("active_state"), terminal.get("sub_state"),
            terminal.get("raw_bytes"), terminal.get("raw_sha256"), terminal.get("raw_path"),
        )
        == (
            "fg19.child-dictionary-fit-native-terminal-evidence.v1", RUN_NAME, NATIVE_UNIT,
            INVOCATION, "active", "exited", NATIVE_RAW_BYTES, NATIVE_RAW_SHA, NATIVE_RAW_PATH,
        ),
        "typed dictionary terminal evidence differs",
    )
    require(
        type(terminal.get("main_pid")) is int and terminal["main_pid"] == 0
        and type(terminal.get("exec_main_code")) is int and terminal["exec_main_code"] == 1
        and type(terminal.get("exec_main_status")) is int and terminal["exec_main_status"] == 0,
        "native process/exit fields must be exact integers",
    )
    require(
        len(capture_raw) == NATIVE_RAW_BYTES and digest(capture_raw) == NATIVE_RAW_SHA,
        "raw-first dictionary capture bytes differ from typed proof",
    )
    require(
        (
            capture.get("schema"), capture.get("expected_run"), capture.get("expected_unit"),
            capture.get("expected_invocation_id"), capture.get("captured_utc"),
        )
        == (
            "fg19-child-control-startup-capture.v2", RUN_NAME, NATIVE_UNIT, INVOCATION,
            terminal.get("captured_utc"),
        ),
        "raw native capture identity differs",
    )
    captured_unit = capture.get("unit_properties")
    require(isinstance(captured_unit, dict), "captured native unit properties are missing")
    require(
        tuple(captured_unit.get(key) for key in (
            "InvocationID", "ActiveState", "SubState", "MainPID", "ExecMainCode",
            "ExecMainStatus", "Result",
        )) == (INVOCATION, "active", "exited", "0", "1", "0", "success"),
        "captured native unit exit differs",
    )
    require(captured_receipt(capture, "ADMISSION.json") == admission_raw,
            "captured original admission bytes differ")
    require(captured_receipt(capture, "LAUNCH.json") == launch_raw,
            "captured original launch bytes differ")
    current = show(NATIVE_UNIT, (
        "LoadState", "InvocationID", "ActiveState", "SubState", "MainPID", "ExecMainCode",
        "ExecMainStatus", "Result", "CPUUsageNSec",
    ))
    require(
        tuple(current.get(key) for key in (
            "LoadState", "InvocationID", "ActiveState", "SubState", "MainPID",
            "ExecMainCode", "ExecMainStatus", "Result",
        )) == ("loaded", INVOCATION, "active", "exited", "0", "1", "0", "success"),
        "fresh dictionary invocation is not the exact successful exited unit",
    )
    require(current.get("CPUUsageNSec", "").isdigit() and int(current["CPUUsageNSec"]) > 0,
            "native CPU usage evidence is missing")
    raw_log = captured_receipt(capture, "unit.log")
    log, log_stat = stable_file(EVIDENCE / "unit.log", MAX_JSON_BYTES)
    require(log == raw_log, "native unit log differs from its exact captured bytes")
    artifacts = check_run_stats(capture)
    return current, {
        "typed": terminal, "raw_capture_sha256": NATIVE_RAW_SHA,
        "raw_capture_bytes": len(capture_raw), "unit_log": {**log_stat, "sha256": digest(log)},
    }, artifacts
'''

CHECK_PARENT = '''def check_parent() -> dict[str, object]:
    """Enforce empty campaign descendants, one-time support headroom, and physical scratch."""
    fields = ("ControlGroup", "MemoryCurrent", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax")
    parent = show("wnn-campaign.slice", fields)
    require(
        (parent.get("CPUQuotaPerSecUSec"), parent.get("MemoryHigh"), parent.get("MemoryMax"), parent.get("MemorySwapMax"))
        == ("2s", "18000000000", "20000000000", "0"),
        "parent resource envelope differs",
    )
    control_group = parent.get("ControlGroup", "")
    require(control_group not in ("", "/") and control_group.endswith("/wnn-campaign.slice"),
            "empty/unscoped parent cgroup is NA; refusing host-root fallback")
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    require(cgroup.is_dir() and cgroup.resolve(strict=True) == cgroup, "parent cgroup is unavailable or redirected")
    proc_files = [cgroup / "cgroup.procs", *cgroup.rglob("cgroup.procs")]
    require(0 < len(proc_files) < 128, "parent populated-descendant inventory is NA or over its bound")
    populated: dict[str, str] = {}
    for path in proc_files:
        value = path.read_text(encoding="ascii")
        if value.strip():
            populated[str(path)] = value
    require(not populated, "campaign cgroup descendants are populated")
    current_raw = (cgroup / "memory.current").read_text(encoding="ascii").strip()
    require(current_raw.isdigit(), "parent memory current is unavailable")
    current = int(current_raw)
    require(current + SUPPORT_MAX_BYTES + PARENT_RESERVE_BYTES < PARENT_TARGET_BYTES,
            "parent headroom after one support reservation is insufficient")
    events = {name: (cgroup / name).read_text(encoding="ascii") for name in ("memory.events", "memory.events.local")}
    scratch = B / "research_tmp"
    require(scratch.is_dir() and not scratch.is_symlink() and scratch.resolve(strict=True) == scratch,
            "research_tmp is not a direct physical directory")
    free = os.statvfs(B).f_bavail * os.statvfs(B).f_frsize
    require(free >= 25 * 1024**3, "project filesystem free space is below 25 GiB")
    project_device = os.stat(B).st_dev
    scratch_device = os.stat(scratch).st_dev
    require(project_device == scratch_device == 64_512, "project/research_tmp device identity differs")
    result = subprocess.run(["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"],
                            capture_output=True, text=True, timeout=10, check=False)
    require(result.returncode == 0 and result.stdout.strip() == "ext4",
            "research_tmp is not on the expected ext4 filesystem")
    return {
        "systemd": parent, "cgroup": str(cgroup), "populated": populated,
        "memory_current_bytes": current,
        "headroom_formula": "current + 1500000000 support max + 1000000000 reserve < 17000000000 target",
        "memory_events": events, "free_bytes": free, "project_device": project_device,
        "research_tmp_device": scratch_device, "research_tmp_fstype": result.stdout.strip(),
    }
'''

MAIN = '''def main() -> None:
    """Run all fresh gates, stage fixed inputs create-only, and submit one CPU verifier."""
    require(
        FINAL_REVIEW_BINDING.get("schema")
        == "fg19.child-dictionary-terminal-verifier-render-binding.v1"
        and FINAL_REVIEW_BINDING.get("status")
        == "ROOT_PREPARATION_AND_SOURCE_REVIEW_AND_NATIVE_EXIT0_BOUND",
        "final root/source review and native-exit binding is missing",
    )
    checker_pin = FINAL_REVIEW_BINDING.get("checker")
    terminal_pin = FINAL_REVIEW_BINDING.get("native_terminal")
    source_review_pin = FINAL_REVIEW_BINDING.get("source_review")
    root_preparation_pin = FINAL_REVIEW_BINDING.get("root_preparation")
    require(
        isinstance(checker_pin, dict)
        and (checker_pin.get("bytes"), checker_pin.get("sha256")) == (CHECKER_BYTES, CHECKER_SHA),
        "final checker pin differs from the exact outer payload guard",
    )
    require(
        isinstance(terminal_pin, dict)
        and (terminal_pin.get("typed_sha256"), terminal_pin.get("raw_bytes"), terminal_pin.get("raw_sha256"))
        == (TERMINAL_SHA, NATIVE_RAW_BYTES, NATIVE_RAW_SHA),
        "final native evidence pin differs from exact embedded bytes",
    )
    require(
        isinstance(source_review_pin, dict)
        and source_review_pin.get("status") == "PASS_SOURCE_ONLY"
        and isinstance(source_review_pin.get("sha256"), str)
        and len(source_review_pin["sha256"]) == 64
        and isinstance(root_preparation_pin, dict)
        and root_preparation_pin.get("status")
        == "PASS_LOCAL_PREPARATION_ONLY_NOT_STAGED_ADMITTED_OR_EXECUTED"
        and isinstance(root_preparation_pin.get("sha256"), str)
        and len(root_preparation_pin["sha256"]) == 64,
        "final checker source-review/root-preparation pins are missing",
    )
    path_exception = require_path_exception()
    require(B.is_dir() and not B.is_symlink() and B.resolve(strict=True) == B,
            "project root is unavailable or redirected")
    require((B / "research_tmp").is_dir() and not (B / "research_tmp").is_symlink(),
            "research_tmp is not a physical directory")
    require(CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
            "canonical project Python is unavailable")
    require(D.parent == B / "research_tmp" and not os.path.lexists(D),
            "create-only verifier output already exists")
    require(show(UNIT, ("LoadState",)).get("LoadState") == "not-found",
            "support unit already exists")
    payload = payload_bytes()
    require(set(payload) == set(EXPECTED_HASHES), "fixed verifier input set differs")
    admission_raw, _ = stable_file(EVIDENCE / "ADMISSION.json", 20_000)
    admission = json_object(admission_raw, "dictionary admission")
    physical_code_root(SOURCE)
    source = check_source(payload, admission)
    split = check_split(payload, admission)
    registered = check_registered_data(admission)
    native, terminal, run_stats = check_native(payload)
    predecessors = check_predecessors(admission)
    cache = check_cache(payload, admission, split)
    parent = check_parent()
    body_raw, body_stat = stable_file(Path(__file__), 250_000)
    staged_files = {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in payload.items()}
    limits = {
        "cpu_quota": "50%", "memory_high_bytes": 1_400_000_000,
        "memory_max_bytes": SUPPORT_MAX_BYTES, "swap_max_bytes": 0,
        "runtime_seconds": 600, "cuda_visible_devices": "", "blas_threads": 1,
    }
    D.mkdir(mode=0o700, exist_ok=False)
    for name, raw in payload.items():
        write_exclusive(D / name, raw)
    verifier_admission = {
        "schema": "fg19.child-dictionary-terminal-verifier-admission.v1",
        "created_utc": utc_now(), "status": "FRESH_GATES_PASSED_SUPPORT_SUBMISSION_PENDING",
        "run_name": RUN_NAME, "native_unit": NATIVE_UNIT, "native_invocation_id": INVOCATION,
        "terminal_evidence_sha256": TERMINAL_SHA, "terminal_raw_sha256": NATIVE_RAW_SHA,
        "source": source, "split": split, "registered_data_stat_only": registered,
        "terminal_run_artifacts_stat_only": run_stats, "native_unit_snapshot": native,
        "typed_terminal_evidence": terminal, "predecessors_read_only": predecessors,
        "cache_stat_only": cache, "parent": parent, "staged_files": staged_files,
        "rendered_body": {"bytes": len(body_raw), "sha256": digest(body_raw), **body_stat},
        "final_local_review_binding": FINAL_REVIEW_BINDING,
        "resource_limits": limits,
        "approved_path_compatibility_exception": {
            "environment_variable": PATH_EXCEPTION_ENV, "value": path_exception,
            "global": "pathlib._local.PosixPath",
            "scope": "training_state.pt only; one exact dictionary-cache path constructor; weights_only=True",
        },
        "scope": "Six prehashed code files and bounded source/native/acceptance metadata only; existing receipts remain read-only; no source restage, native stop, model/state load, or cache-array/dictionary payload hash.",
        "unknown_outcome_policy": "Inspect this exact unit and durable output; never retry or recreate this output directory.",
    }
    write_json_exclusive(D / "VERIFIER_ADMISSION.json", verifier_admission)
    command = [
        "systemd-run", "--user", "--unit=" + UNIT, "--property=Type=exec",
        "--property=RemainAfterExit=yes", "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=50%", "--property=MemoryHigh=1400000000",
        "--property=MemoryMax=1500000000", "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=600", "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(D / "unit.log"),
        "--property=StandardError=append:" + str(D / "unit.log"),
        "--working-directory=" + str(D),
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "WNN_THREADS": "1", "RAYON_NUM_THREADS": "1",
        "TMPDIR": str(B / "research_tmp"), "PYTHONPATH": str(D) + ":" + str(SOURCE / "src"),
        "WNN_PROJECT_ROOT": str(DATA), PATH_EXCEPTION_ENV: path_exception,
    }
    command.extend("--setenv=" + key + "=" + value for key, value in environment.items())
    command.extend([
        str(CANONICAL_PYTHON), "-E", "-B", "-u", str(D / "child_dictionary_terminal_check_v1.py"),
        "--run-dir", str(RUN), "--source-root", str(SOURCE),
        "--source-manifest", str(D / "SOURCE_MANIFEST.json"), "--split", str(SPLIT),
        "--evidence-root", str(EVIDENCE), "--profile-evidence-root", str(PROFILE_EVIDENCE),
        "--admission", str(EVIDENCE / "ADMISSION.json"), "--launch", str(EVIDENCE / "LAUNCH.json"),
        "--launch-acceptance", str(D / "LAUNCH_ACCEPTANCE.json"),
        "--terminal-evidence", str(D / "TERMINAL_EVIDENCE.json"),
        "--cache-root", str(CACHE), "--cache-acceptance", str(CACHE_RECEIPTS / "TERMINAL_ACCEPTANCE.json"),
        "--cache-exact-stop", str(CACHE_RECEIPTS / "EXACT_STOP.json"),
        "--output", str(D / "VERIFICATION.json"),
    ])
    write_json_exclusive(D / "VERIFIER_SUBMISSION_INTENT.json", {
        "created_utc": utc_now(), "unit": UNIT, "command": command,
        "unknown_outcome_policy": "Never retry; inspect this exact unit and durable output.",
    })
    submission: dict[str, object] = {"outcome": "unknown"}
    try:
        result = subprocess.run(command, capture_output=True, timeout=25, check=False)
        submission = {
            "outcome": "submitted" if result.returncode == 0 else "unknown_or_failed",
            "returncode": result.returncode,
            "stdout_b64": base64.b64encode(result.stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(result.stderr).decode("ascii"),
        }
    except subprocess.TimeoutExpired as error:
        submission = {
            "outcome": "submit_timeout_unknown_never_retry", "returncode": None,
            "stdout_b64": base64.b64encode(text_bytes(error.stdout)).decode("ascii"),
            "stderr_b64": base64.b64encode(text_bytes(error.stderr)).decode("ascii"),
        }
    except OSError as error:
        submission = {
            "outcome": "submit_error_unknown_never_retry", "returncode": None,
            "error": repr(error), "stdout_b64": "", "stderr_b64": "",
        }
    post_fields = (
        "LoadState", "InvocationID", "ActiveState", "SubState", "MainPID", "ExecMainCode",
        "ExecMainStatus", "ControlGroup", "Result",
    )
    try:
        post = {"outcome": "captured", "properties": show(UNIT, post_fields)}
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        post = {"outcome": "query_failed_unknown", "error": repr(error)}
    support_properties = post.get("properties", {}) if isinstance(post, dict) else {}
    support_invocation = support_properties.get("InvocationID") if isinstance(support_properties, dict) else None
    launch_record = {
        "schema": "fg19.child-dictionary-terminal-verifier-launch.v1", "created_utc": utc_now(),
        "unit": UNIT, "invocation_id": support_invocation, "native_unit": NATIVE_UNIT,
        "native_invocation_id": INVOCATION, "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Durably retain this outcome; inspect exact unit and output; never retry or create another output directory.",
    }
    write_json_exclusive(D / "VERIFIER_LAUNCH.json", launch_record)
    response = {
        "unit": UNIT, "outcome": submission["outcome"],
        "admission_base64": base64.b64encode((D / "VERIFIER_ADMISSION.json").read_bytes()).decode("ascii"),
        "launch_base64": base64.b64encode((D / "VERIFIER_LAUNCH.json").read_bytes()).decode("ascii"),
    }
    print(json.dumps(response, sort_keys=True))
'''


def _sha256(raw: bytes) -> str:
    """Return the exact SHA-256 digest of a bounded code template."""
    return hashlib.sha256(raw).hexdigest()


def _function_sources(source: str) -> dict[str, str]:
    """Return selected top-level function source slices from the pinned baseline."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = "".join(lines[node.lineno - 1 : node.end_lineno])
    return result


def build_template() -> str:
    """Compose a dictionary-only admission body from unchanged EMA fresh gates."""
    raw = BASE.read_bytes()
    if len(raw) != BASE_BYTES or _sha256(raw) != BASE_SHA256:
        raise ValueError("reviewed EMA V3 fresh-gate template bytes changed")
    functions = _function_sources(raw.decode("utf-8"))
    selected = (
        "require",
        "require_path_exception",
        "digest",
        "utc_now",
        "json_object",
        "payload_bytes",
        "stable_file",
        "file_hash",
        "physical_code_root",
        "show",
        "write_exclusive",
        "write_json_exclusive",
        "captured_receipt",
        "check_run_stats",
        "check_source",
        "check_split",
        "check_registered_data",
        "check_receipt_file",
        "check_predecessors",
        "check_cache",
        "text_bytes",
    )
    missing = set(selected) - set(functions)
    if missing:
        raise ValueError(f"pinned EMA base is missing required pure gates: {sorted(missing)}")
    body = (
        "\n\n".join(
            [
                HEADER.rstrip(),
                *(functions[name].rstrip() for name in selected),
                CHECK_NATIVE.rstrip(),
                CHECK_PARENT.rstrip(),
                MAIN.rstrip(),
                'if __name__ == "__main__":\n    main()',
            ]
        )
        + "\n"
    )
    ast.parse(body)
    if "check_v1_failure" in body or "check_v2_failure" in body:
        raise ValueError("superseded EMA failure requirements leaked into dictionary body")
    return body


def main() -> int:
    """Create the reviewed-derived template once; never render an admission body."""
    if TEMPLATE.exists() or TEMPLATE.is_symlink():
        raise FileExistsError(f"conditional body template already exists: {TEMPLATE}")
    body = build_template().encode("utf-8")
    descriptor = os.open(TEMPLATE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    formatter = REPOSITORY / ".venv/bin/ruff"
    if not formatter.is_file():
        raise FileNotFoundError(f"configured Ruff executable is missing: {formatter}")
    subprocess.run([str(formatter), "format", str(TEMPLATE)], check=True)
    body = TEMPLATE.read_bytes()
    directory = os.open(OUTPUT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"template": str(TEMPLATE), "bytes": len(body), "sha256": _sha256(body)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
