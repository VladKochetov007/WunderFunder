"""Synthetic-fake-stat test of the exact original AllThree receipt preflight."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[5]
BODY = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V5.py"
FIT_RECORDS = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1"
)
EXPECTED_INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
EXPECTED_UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"


def body_constant(tree: ast.Module, name: str) -> object:
    """Read one exact top-level literal pin from the template AST."""
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one top-level {name} assignment")
    return ast.literal_eval(matches[0].value)


def build_preflight(launch_bytes: bytes | None = None) -> tuple[object, list[tuple[str, int]]]:
    """Compile only the new fit-receipt helper with synthetic stable-file stats."""
    admission_bytes = (FIT_RECORDS / "ADMISSION.json").read_bytes()
    launch_raw = (
        (FIT_RECORDS / "LAUNCH.json").read_bytes() if launch_bytes is None else launch_bytes
    )
    tree = ast.parse(BODY.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_original_fit_receipts"
    ]
    if len(functions) != 1:
        raise AssertionError("the V5 receipt-preflight helper is not unique")
    fake_stats = {
        "ADMISSION.json": {
            "bytes": len(admission_bytes),
            "device": 64512,
            "inode": 1001,
            "mtime_ns": 11,
        },
        "LAUNCH.json": {"bytes": len(launch_raw), "device": 64512, "inode": 1002, "mtime_ns": 12},
    }
    calls: list[tuple[str, int]] = []

    def stable_file(path: Path, maximum: int) -> tuple[bytes, dict[str, int]]:
        calls.append((path.name, maximum))
        if maximum != 20_000 or path.name not in fake_stats:
            raise AssertionError("receipt preflight requested an unexpected path or size cap")
        raw = admission_bytes if path.name == "ADMISSION.json" else launch_raw
        return raw, dict(fake_stats[path.name])

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def json_object(raw: bytes, label: str) -> dict[str, object]:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} is not a JSON object")
        return value

    namespace = {
        "FIT_RECEIPTS_ROOT": Path("/synthetic/fit-receipts"),
        "FIT_ADMISSION_BYTES": body_constant(tree, "FIT_ADMISSION_BYTES"),
        "FIT_ADMISSION_SHA256": body_constant(tree, "FIT_ADMISSION_SHA256"),
        "FIT_LAUNCH_BYTES": (
            body_constant(tree, "FIT_LAUNCH_BYTES") if launch_bytes is None else len(launch_raw)
        ),
        "FIT_LAUNCH_SHA256": (
            body_constant(tree, "FIT_LAUNCH_SHA256")
            if launch_bytes is None
            else hashlib.sha256(launch_raw).hexdigest()
        ),
        "base64": base64,
        "stable_file": stable_file,
        "require": require,
        "digest": digest,
        "json_object": json_object,
        "is_invocation_id": lambda value: (
            isinstance(value, str)
            and len(value) == 32
            and all(char in "0123456789abcdef" for char in value)
        ),
    }
    module = ast.Module(body=[copy.deepcopy(functions[0])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BODY), "exec"), namespace)
    return namespace["check_original_fit_receipts"], calls


class FitReceiptPreflightTest(unittest.TestCase):
    def test_actual_pinned_receipts_pass_with_fake_stat_rows(self) -> None:
        admission_raw = (FIT_RECORDS / "ADMISSION.json").read_bytes()
        launch_raw = (FIT_RECORDS / "LAUNCH.json").read_bytes()
        check, calls = build_preflight()
        result = check()
        self.assertEqual(calls, [("ADMISSION.json", 20_000), ("LAUNCH.json", 20_000)])
        self.assertEqual(result[0], admission_raw)
        self.assertEqual(result[2], launch_raw)
        self.assertEqual(result[1]["inode"], 1001)
        self.assertEqual(result[3]["inode"], 1002)
        self.assertEqual(result[4]["run_name"], "fg19_child_allthree_f0_s1001_fit_v1")
        captured = base64.b64decode(
            result[5]["systemctl_after_submission"]["stdout_b64"], validate=True
        ).decode("utf-8")
        self.assertIn(f"InvocationID={EXPECTED_INVOCATION}\n", captured)

    def test_wrong_nested_invocation_rejects_misleading_top_level_id(self) -> None:
        launch = json.loads((FIT_RECORDS / "LAUNCH.json").read_bytes())
        launch["invocation_id"] = EXPECTED_INVOCATION
        post = launch["systemctl_after_submission"]
        captured = base64.b64decode(post["stdout_b64"], validate=True).decode("utf-8")
        wrong = "00000000000000000000000000000000"
        self.assertIn(f"InvocationID={EXPECTED_INVOCATION}\n", captured)
        post["stdout_b64"] = base64.b64encode(
            captured.replace(EXPECTED_INVOCATION, wrong).encode("utf-8")
        ).decode("ascii")
        mutated = json.dumps(launch, sort_keys=True).encode("utf-8")
        check, _ = build_preflight(mutated)
        with self.assertRaisesRegex(RuntimeError, "systemctl properties do not join"):
            check()

    def test_wrong_top_level_id_cannot_override_nested_captured_id(self) -> None:
        launch = json.loads((FIT_RECORDS / "LAUNCH.json").read_bytes())
        launch["invocation_id"] = "00000000000000000000000000000000"
        mutated = json.dumps(launch, sort_keys=True).encode("utf-8")
        check, _ = build_preflight(mutated)
        result = check()
        self.assertEqual(result[5]["unit"], EXPECTED_UNIT)

    def test_fixture_uses_nested_property_and_preserves_narrow_source_scope(self) -> None:
        text = BODY.read_text(encoding="utf-8")
        start = text.index("def check_original_fit_receipts")
        end = text.index("def check_paths_and_inputs", start)
        helper = text[start:end]
        self.assertIn('fit_launch.get("systemctl_after_submission")', helper)
        self.assertIn('capture_properties.get("InvocationID")', helper)
        self.assertNotIn('fit_launch.get("invocation_id")', helper)
        self.assertIn('fit_launch.get("submission")', helper)


if __name__ == "__main__":
    unittest.main()
