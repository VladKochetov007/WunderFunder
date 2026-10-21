"""Exercise the frozen receipt checker against exact local receipt metadata only."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import posixpath
import re
import unittest
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
TEMPLATE = CAMPAIGN / "preparation/first_cell_evaluation_admission_v4/ADMISSION_BODY_TEMPLATE_V4.py"
EXPORT_BINDING = CAMPAIGN / "preparation/first_cell_evaluation_export_v2/EXPORT_BINDING_V1.json"
RECORDS = CAMPAIGN / "records"
ALLTHREE_ADMISSION = RECORDS / "child_allthree_fit_v1/ADMISSION.json"
ALLTHREE_LAUNCH = RECORDS / "child_allthree_fit_v1/LAUNCH.json"
LOCAL_RECEIPTS = {
    "CONTROL_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_control_fit_v1/TERMINAL_ACCEPTANCE.json",
        11_883,
        "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    ),
    "CONTROL_EXACT_STOP.json": (
        RECORDS / "child_control_fit_v1/exact_stop/STDOUT.raw.json",
        3_089,
        "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
    ),
    "EMA_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_ema_fit_v1/TERMINAL_ACCEPTANCE.json",
        13_016,
        "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8",
    ),
    "EMA_EXACT_STOP.json": (
        RECORDS / "child_ema_fit_v1/exact_stop/EXACT_STOP.json",
        3_093,
        "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf",
    ),
    "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_dictionary_fit_v1/TERMINAL_ACCEPTANCE.json",
        14_914,
        "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1",
    ),
    "DICTIONARY_CHILD_EXACT_STOP.json": (
        RECORDS / "child_dictionary_fit_v1/exact_stop/EXACT_STOP.json",
        3_113,
        "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3",
    ),
    "NLL_CHILD_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_nll_fit_v1/TERMINAL_ACCEPTANCE.json",
        15_860,
        "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5",
    ),
    "NLL_CHILD_EXACT_STOP.json": (
        RECORDS / "child_nll_fit_v1/exact_stop/EXACT_STOP.json",
        3_140,
        "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1",
    ),
}
FIXED_RECEIPTS = {
    "ADMISSION.json": (
        ALLTHREE_ADMISSION,
        12_912,
        "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540",
    ),
    "LAUNCH.json": (
        ALLTHREE_LAUNCH,
        9_594,
        "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538",
    ),
    **LOCAL_RECEIPTS,
    "ALLTHREE_TERMINAL_ACCEPTANCE.json": (
        RECORDS / "child_allthree_fit_v1/TERMINAL_ACCEPTANCE.json",
        20_208,
        "fada500ede63a175d7d055d38fbef5d38110aef481a91771ae3c5717c1e7026f",
    ),
    "ALLTHREE_EXACT_STOP.json": (
        RECORDS / "child_allthree_fit_v1/exact_stop/EXACT_STOP.json",
        3_208,
        "0154b3f35c4fb849d0db7e7b60ce1f4ce2fa01ab2203c17db65877f89353f13d",
    ),
}
B_RECEIPT_ROOT = (
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_allthree_fit_v1"
)


class FakePath:
    """Small POSIX path value used to ensure no B/model path is touched locally."""

    def __init__(self, value: str) -> None:
        self.value = posixpath.normpath(value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"FakePath({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakePath) and self.value == other.value

    def __truediv__(self, other: object) -> FakePath:
        return FakePath(posixpath.join(self.value, str(other)))

    @property
    def name(self) -> str:
        return posixpath.basename(self.value)

    @property
    def parent(self) -> FakePath:
        return FakePath(posixpath.dirname(self.value))

    @property
    def parts(self) -> tuple[str, ...]:
        return PurePosixPath(self.value).parts

    def is_absolute(self) -> bool:
        return self.value.startswith("/")

    def is_symlink(self) -> bool:
        return False


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _literal_assignments(source: str, names: set[str]) -> dict[str, object]:
    """Extract selected literal assignments without executing the module."""
    tree = ast.parse(source, filename=str(TEMPLATE))
    result: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names and node.value is not None:
            if isinstance(node.value, ast.Name) and node.value.id in result:
                result[target.id] = result[node.value.id]
            else:
                result[target.id] = ast.literal_eval(node.value)
    if set(result) != names:
        raise AssertionError(f"missing literal constants: {names - set(result)}")
    return result


def _checked_local_receipt(name: str) -> bytes:
    path, expected_bytes, expected_sha = FIXED_RECEIPTS[name]
    raw = path.read_bytes()
    if (len(raw), _digest(raw)) != (expected_bytes, expected_sha):
        raise AssertionError(f"local receipt pin differs: {name}")
    return raw


def _runtime(
    launch_override: bytes | None = None,
) -> tuple[object, dict[str, object], dict[str, int], list[str], list[str]]:
    """Compile only check_fit_receipts and attach bounded local metadata stubs."""
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TEMPLATE))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_fit_receipts"
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one checker function")
    function_module = ast.Module(body=[matches[0]], type_ignores=[])
    constants = _literal_assignments(
        source,
        {
            "FIT_ADMISSION_BYTES",
            "FIT_ADMISSION_SHA256",
            "FIT_LAUNCH_BYTES",
            "FIT_LAUNCH_SHA256",
            "FIT_LAUNCH_INVOCATION",
            "NATIVE_UNIT",
            "NATIVE_INVOCATION",
            "CANONICAL_COMMIT",
            "CANONICAL_MANIFEST_SHA256",
        },
    )
    local_by_name = {
        name: _checked_local_receipt(name)
        for name in FIXED_RECEIPTS
        if name not in {"ALLTHREE_TERMINAL_ACCEPTANCE.json", "ALLTHREE_EXACT_STOP.json"}
    }
    if launch_override is not None:
        local_by_name["LAUNCH.json"] = launch_override
        constants["FIT_LAUNCH_BYTES"] = len(launch_override)
        constants["FIT_LAUNCH_SHA256"] = _digest(launch_override)

    stable_calls: list[str] = []
    stat_calls: list[str] = []

    def require(condition: object, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def stable_file(path: FakePath, max_bytes: int) -> tuple[bytes, dict[str, object]]:
        name = path.name
        if name not in local_by_name:
            raise AssertionError(f"unexpected local receipt lookup: {path}")
        raw = local_by_name[name]
        if len(raw) > max_bytes:
            raise AssertionError(f"receipt exceeds caller byte limit: {name}")
        stable_calls.append(name)
        return raw, {
            "bytes": len(raw),
            "device": 1,
            "inode": len(stable_calls),
            "mtime_ns": 1,
            "islink": False,
            "resolved": f"/local-fixture/{name}",
        }

    def parse_json(raw: bytes, name: str) -> dict[str, object]:
        value = json.loads(raw.decode("utf-8", errors="strict"))
        if not isinstance(value, dict):
            raise RuntimeError(f"{name} is not an object")
        return value

    def stat_regular(path: FakePath) -> dict[str, object]:
        stat_calls.append(path.value)
        model = next(
            arm["model"]
            for arm in binding["arms"]
            if str(arm["model"]["relative_path"]) in path.value
        )
        return {
            "bytes": model["bytes"],
            "device": 1,
            "inode": len(stat_calls),
            "mtime_ns": 1,
            "islink": False,
            "resolved": path.value,
        }

    namespace: dict[str, object] = {
        **constants,
        "Path": FakePath,
        "FIT_RECEIPTS_ROOT": FakePath(B_RECEIPT_ROOT),
        "DATA_ROOT": FakePath("/home/grinchenkoeg/computer-vision-homework-20261018/data"),
        "base64": base64,
        "digest": _digest,
        "require": require,
        "stable_file": stable_file,
        "stat_regular": stat_regular,
        "parse_json": parse_json,
        "_is_invocation_id": lambda value: (
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None
        ),
    }
    exec(compile(function_module, str(TEMPLATE), "exec"), namespace)
    binding = json.loads(EXPORT_BINDING.read_bytes())
    payload = {
        "ALLTHREE_TERMINAL_ACCEPTANCE.json": _checked_local_receipt(
            "ALLTHREE_TERMINAL_ACCEPTANCE.json"
        ),
        "ALLTHREE_EXACT_STOP.json": _checked_local_receipt("ALLTHREE_EXACT_STOP.json"),
    }
    result = namespace["check_fit_receipts"](payload, binding)
    return result, binding, constants, stable_calls, stat_calls


class AdmissionV4ActualReceiptTests(unittest.TestCase):
    def test_full_checker_accepts_exact_receipts_and_reads_only_fake_checkpoint_stats(self) -> None:
        result, binding, _, stable_calls, stat_calls = _runtime()
        self.assertEqual(
            set(stable_calls),
            {
                "ADMISSION.json",
                "LAUNCH.json",
                "CONTROL_TERMINAL_ACCEPTANCE.json",
                "CONTROL_EXACT_STOP.json",
                "EMA_TERMINAL_ACCEPTANCE.json",
                "EMA_EXACT_STOP.json",
                "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json",
                "DICTIONARY_CHILD_EXACT_STOP.json",
                "NLL_CHILD_TERMINAL_ACCEPTANCE.json",
                "NLL_CHILD_EXACT_STOP.json",
            },
        )
        self.assertEqual(len(stable_calls), 10)
        self.assertEqual(len(stat_calls), 5)
        self.assertEqual(
            {Path(arm["model"]["relative_path"]).as_posix() for arm in binding["arms"]},
            {"/".join(path.split("/data/", 1)[-1].split("/")) for path in stat_calls},
        )
        original_launch = json.loads(_checked_local_receipt("LAUNCH.json"))
        self.assertNotIn("invocation_id", original_launch)
        self.assertNotIn("properties", original_launch)
        self.assertEqual(
            result["original_launch"]["submitted_invocation_id"],
            "94d3a7132a434d448345b0206bd3daa3",
        )
        self.assertEqual(
            result["original_launch"]["captured_invocation_id"],
            result["original_launch"]["submitted_invocation_id"],
        )
        self.assertEqual(result["original_launch"]["captured_main_pid"], "1875143")
        self.assertEqual(result["original_launch"]["captured_active_state"], "active")
        self.assertEqual(result["original_launch"]["captured_sub_state"], "running")
        self.assertEqual(len(result["five_accepted_stop_receipt_pairs_and_model_stats_only"]), 5)
        self.assertFalse(result["checkpoint_payloads_read_or_hashed"])

    def test_nested_invocation_mismatch_is_rejected(self) -> None:
        launch = json.loads(_checked_local_receipt("LAUNCH.json"))
        after = launch["systemctl_after_submission"]
        captured = base64.b64decode(after["stdout_b64"], validate=True).decode("utf-8")
        after["stdout_b64"] = base64.b64encode(
            captured.replace(
                "InvocationID=94d3a7132a434d448345b0206bd3daa3",
                "InvocationID=00000000000000000000000000000000",
            ).encode("utf-8")
        ).decode("ascii")
        modified = json.dumps(launch, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(RuntimeError, "systemctl properties do not join"):
            _runtime(modified)

    def test_top_level_invocation_fields_cannot_replace_missing_nested_capture(self) -> None:
        launch = json.loads(_checked_local_receipt("LAUNCH.json"))
        launch["invocation_id"] = "94d3a7132a434d448345b0206bd3daa3"
        launch["properties"] = {"InvocationID": "94d3a7132a434d448345b0206bd3daa3"}
        launch["systemctl_after_submission"]["stdout_b64"] = base64.b64encode(
            b"MainPID=1875143\\nActiveState=active\\nSubState=running\\n"
        ).decode("ascii")
        modified = json.dumps(launch, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(RuntimeError, "systemctl properties do not join"):
            _runtime(modified)


if __name__ == "__main__":
    unittest.main()
