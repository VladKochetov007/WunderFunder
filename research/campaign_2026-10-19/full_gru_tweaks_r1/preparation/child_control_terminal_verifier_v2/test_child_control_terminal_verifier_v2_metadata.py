"""Exercise actual early CONTROL receipts without opening fitted payloads."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
PACKET = PREPARATION / "child_control_terminal_verifier_v2"
RECORDS = CAMPAIGN / "records"
CONTROL = RECORDS / "child_control_fit_v1"
PROFILE = RECORDS / "child_allthree_profile_v1"
SOURCE_ROOT = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-68ff6829"
)
sys.path.insert(0, str(PREPARATION))
sys.path.insert(0, str(PACKET))

checker = importlib.import_module("child_control_terminal_check_v2")
parent_v3 = importlib.import_module("parent_fit_terminal_check_v3")


def _raw(path: Path) -> bytes:
    return path.read_bytes()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(_raw(path))
    if not isinstance(value, dict):
        raise AssertionError(f"expected metadata object: {path}")
    return value


class ActualMetadataPredicateTests(unittest.TestCase):
    """Check exact stored metadata and early checker branches only."""

    def test_original_launch_and_acceptance_are_separate_bindings(self) -> None:
        """Original launch schema/result and accepted native body SHA are joined."""
        admission_raw = _raw(CONTROL / "ADMISSION.json")
        launch_raw = _raw(CONTROL / "LAUNCH.json")
        acceptance_raw = _raw(CONTROL / "LAUNCH_ACCEPTANCE.json")
        launch = json.loads(launch_raw)
        acceptance = json.loads(acceptance_raw)
        self.assertNotIn("body_sha256", launch)
        self.assertEqual(acceptance["body_sha256"], checker.LAUNCH_BODY_SHA)
        checker._check_control_launch_records(
            launch, _sha(launch_raw), acceptance, _sha(acceptance_raw), _sha(admission_raw)
        )
        wrong_body = dict(acceptance)
        wrong_body["body_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "CONTROL launch acceptance"):
            checker._check_control_launch_records(
                launch, _sha(launch_raw), wrong_body, _sha(acceptance_raw), _sha(admission_raw)
            )
        wrong_launch = dict(launch)
        wrong_launch["unit"] = "computer-vision-homework-wrong.service"
        with self.assertRaisesRegex(ValueError, "CONTROL launch record"):
            checker._check_control_launch_records(
                wrong_launch,
                _sha(launch_raw),
                acceptance,
                _sha(acceptance_raw),
                _sha(admission_raw),
            )

    def test_terminal_path_calls_launch_binding_before_later_gates(self) -> None:
        """The exact launch and acceptance guard is on the real terminal path."""
        tree = ast.parse((PACKET / "child_control_terminal_check_v2.py").read_text())
        verify = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "verify_terminal"
        )
        calls = sorted(
            (node.lineno, node.func.id)
            for node in ast.walk(verify)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
        names = [name for _, name in calls]
        self.assertLess(
            names.index("_check_control_launch_records"), names.index("_check_terminal")
        )

    def test_actual_terminal_worker_and_predecessor_receipts(self) -> None:
        """Actual compact receipts satisfy early terminal, worker, and ancestry gates."""
        admission = _json(CONTROL / "ADMISSION.json")
        launch = _json(CONTROL / "LAUNCH.json")
        terminal_raw = _raw(CONTROL / "TERMINAL_EVIDENCE.json")
        terminal = json.loads(terminal_raw)
        capture_raw = _raw(CONTROL / "terminal_native_capture_v1/STDOUT.raw.json")
        capture = json.loads(capture_raw)
        self.assertEqual(_sha(capture_raw), terminal["raw_sha256"])
        self.assertEqual(len(capture_raw), terminal["raw_bytes"])
        self.assertEqual(capture["unit_properties"]["InvocationID"], checker.INVOCATION)
        for name in ("ADMISSION.json", "LAUNCH.json"):
            receipt = capture["receipts"][name]
            self.assertTrue(receipt["present"] and receipt["regular_file"])
            self.assertEqual(
                base64.b64decode(receipt["base64"], validate=True), _raw(CONTROL / name)
            )
        checker._check_terminal(terminal)
        parent_v3._check_worker(launch, admission, SOURCE_ROOT)
        self.assertEqual(len(terminal_raw), 713)
        with tempfile.TemporaryDirectory(prefix="control-v2-receipts-") as temporary:
            evidence = Path(temporary) / "evidence"
            profile = Path(temporary) / "profile"
            evidence.mkdir()
            profile.mkdir()
            receipts = {
                "PARENT_TERMINAL_ACCEPTANCE.json": RECORDS
                / "parent_fit_r2_v1/TERMINAL_ACCEPTANCE.json",
                "PARENT_EXACT_STOP.json": RECORDS / "parent_fit_r2_v1/exact_stop/STDOUT.raw.json",
                "DICTIONARY_ACCEPTANCE.json": RECORDS / "dictionary_f0_v1/ACCEPTANCE.json",
                "DICTIONARY_EXACT_STOP.json": RECORDS
                / "dictionary_f0_v1/exact_stop/STDOUT.raw.json",
                "PROFILE_TERMINAL_ACCEPTANCE.json": RECORDS
                / "child_allthree_profile_v1/TERMINAL_ACCEPTANCE.json",
                "PROFILE_EXACT_STOP.json": RECORDS
                / "child_allthree_profile_v1/exact_stop/STDOUT.raw.json",
            }
            for name, source in receipts.items():
                shutil.copyfile(source, evidence / name)
            for name, source in {
                "ADMISSION.json": PROFILE / "ADMISSION.json",
                "LAUNCH.json": PROFILE / "LAUNCH.json",
                "unit.log": PROFILE / "capture/unit.log",
            }.items():
                shutil.copyfile(source, profile / name)
            lineage = checker._check_predecessor_receipts(evidence, profile, admission)
        self.assertEqual(lineage["parent_acceptance_sha256"], checker.PARENT_ACCEPTANCE_SHA)
        self.assertEqual(lineage["dictionary_acceptance_sha256"], checker.DICTIONARY_ACCEPTANCE_SHA)
        self.assertEqual(lineage["profile_exact_stop_sha256"], checker.PROFILE_STOP_SHA)
        self.assertEqual(lineage["profile_initial_state_sha256"], checker.PROFILE_INITIAL_STATE_SHA)
        accepted_profile = _json(PROFILE / "TERMINAL_ACCEPTANCE.json")
        state_raw = json.dumps(
            accepted_profile["initial_state"], sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(_sha(state_raw), checker.PROFILE_INITIAL_STATE_SHA)

    def test_actual_cache_receipt_schema_matches_without_array_access(self) -> None:
        """The exact accepted cache/stop metadata satisfies the fresh gate shape."""
        cache = _json(CONTROL / "ADMISSION.json")["cache"]
        acceptance_raw = _raw(RECORDS / "teacher_cache_cpu_build_v2/TERMINAL_ACCEPTANCE.json")
        stop_raw = _raw(RECORDS / "teacher_cache_cpu_build_v2/exact_stop/STDOUT.raw.json")
        acceptance, stop = json.loads(acceptance_raw), json.loads(stop_raw)
        self.assertEqual(
            (_sha(acceptance_raw), _sha(stop_raw)),
            (cache["acceptance_sha256"], cache["exact_stop_sha256"]),
        )
        self.assertEqual(acceptance["metadata_sha256"], cache["metadata_sha256"])
        self.assertEqual(acceptance["source_manifest_sha256"], checker.SOURCE_MANIFEST_SHA)
        self.assertEqual(
            (acceptance["source_files_verified"], acceptance["unit"], acceptance["invocation_id"]),
            (167, cache["unit"], cache["invocation_id"]),
        )
        before, after = stop["before"][cache["unit"]], stop["after"][cache["unit"]]
        self.assertEqual(
            (before["InvocationID"], before["ExecMainStatus"]), (cache["invocation_id"], "0")
        )
        self.assertEqual(stop["stop"][cache["unit"]]["returncode"], 0)
        self.assertEqual(
            (
                after["LoadState"],
                after["ActiveState"],
                after["SubState"],
                after["InvocationID"],
                stop["populated"],
            ),
            ("not-found", "inactive", "dead", "", {}),
        )
        for name, count, rows in (("train", 1_685_600_128, 10_535), ("valid", 224_800_128, 1_405)):
            output = acceptance["output"][name]
            stats = cache["array_stats_only"][name]
            self.assertEqual(
                (output["bytes"], output["shape"], output["dtype"]),
                (count, [rows, 20_000, 2], "float32"),
            )
            self.assertEqual(
                (output["stat_only"]["inode"], output["stat_only"]["mtime_ns"]),
                (stats["inode"], stats["mtime_ns"]),
            )
            self.assertFalse(output["stat_only"]["payload_read_or_hashed_by_capture"])

    def test_checker_cli_lists_all_required_receipt_arguments(self) -> None:
        """CLI help is exercised without invoking terminal or artifact checks."""
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "WNN_THREADS": "1",
                "PYTHONPATH": str(PREPARATION),
            }
        )
        result = subprocess.run(
            [sys.executable, str(PACKET / "child_control_terminal_check_v2.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in (
            "--run-dir",
            "--source-root",
            "--source-manifest",
            "--split",
            "--evidence-root",
            "--profile-evidence-root",
            "--admission",
            "--launch",
            "--launch-acceptance",
            "--terminal-evidence",
            "--cache-root",
            "--cache-acceptance",
            "--cache-exact-stop",
            "--output",
        ):
            self.assertIn(flag, result.stdout)

    def test_rendered_body_payload_bytes_match_all_local_pins(self) -> None:
        """The submitted wrapper accepts its exact 12 pinned code/metadata inputs."""
        body_path = PACKET / "REMOTE_SUBMISSION.py"
        tree = ast.parse(body_path.read_text(encoding="utf-8"))
        module = ast.Module(
            body=[
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "main"
                    and not node.value.args
                    and not node.value.keywords
                )
            ],
            type_ignores=[],
        )
        namespace: dict[str, object] = {"__name__": "v2_payload_fixture"}
        exec(compile(module, str(body_path), "exec"), namespace)
        payload = namespace["payload_bytes"]()
        self.assertEqual(len(payload), 12)
        checker_raw = _raw(PACKET / "child_control_terminal_check_v2.py")
        self.assertEqual(namespace["CHECKER_SHA"], _sha(checker_raw))
        self.assertEqual(payload["child_control_terminal_check_v2.py"], checker_raw)
        self.assertEqual(
            namespace["UNIT"], "computer-vision-homework-fg19-child-control-verify-v2.service"
        )
        self.assertEqual(
            namespace["D"],
            Path(
                "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_control_verify_v2"
            ),
        )

    def test_cold_private_dependency_closure_uses_exact_four_files(self) -> None:
        """The isolated V2 import resolves only to the four exact staged files."""
        expected = {
            "child_control_terminal_check_v2.py": "cbaff895c6d451a671b14ff77d9c4b26dad69dcaa5c4ab0a8a642129cc3c7c9e",
            "parent_fit_terminal_check_v3.py": "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
            "propensity_terminal_support.py": "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
            "teacher_s5_terminal_check_v1.py": "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
        }
        child_script = """
import os, sys
from pathlib import Path
private = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(private))
blocked = (Path('/home/grinchenkoeg/computer-vision-homework-20261018/data'), Path('/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp'))
attempts = []
def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(args[0]).resolve()
    if any(path == root or path.is_relative_to(root) for root in blocked):
        attempts.append(str(path))
        raise AssertionError('payload open during import')
sys.addaudithook(audit)
import child_control_terminal_check_v2, parent_fit_terminal_check_v3, propensity_terminal_support, teacher_s5_terminal_check_v1
for module in (child_control_terminal_check_v2, parent_fit_terminal_check_v3, propensity_terminal_support, teacher_s5_terminal_check_v1):
    assert Path(module.__file__).resolve().parent == private
assert not attempts
print('cold-import-ok')
"""
        with tempfile.TemporaryDirectory(prefix="control-v2-cold-") as temporary:
            private = Path(temporary)
            for name, expected_sha in expected.items():
                source = (
                    PACKET / name
                    if name == "child_control_terminal_check_v2.py"
                    else PREPARATION / name
                )
                raw = _raw(source)
                self.assertEqual(_sha(raw), expected_sha)
                (private / name).write_bytes(raw)
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            result = subprocess.run(
                [sys.executable, "-I", "-c", child_script, str(private)],
                capture_output=True,
                text=True,
                timeout=40,
                check=False,
                env=environment,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cold-import-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
