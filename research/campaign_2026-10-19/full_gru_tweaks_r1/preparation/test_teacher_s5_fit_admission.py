"""Focused synthetic tests for the conditional S5 admission gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import admit_teacher_s5_fit as admission

NATIVE_UNIT = "computer-vision-homework-fg19-teacher-s4-v1.service"
VERIFIER_UNIT = "computer-vision-homework-fg19-teacher-s4-terminal-check-v1.service"
NATIVE_INVOCATION = "b6d71599e0044970b8641e3e241c40a4"
VERIFIER_INVOCATION = "a" * 32
MODEL_SHA = "1" * 64
CHECKPOINT_SHA = "2" * 64


def _acceptance() -> dict[str, object]:
    """Return a synthetic S4 root acceptance matching the fixed predecessor identity."""
    return {
        "schema": "fg19.teacher-s4-terminal-root-acceptance.v1",
        "status": "ACCEPTED",
        "run_name": "fg19_t3_h384x2_e3_s4_v1",
        "source_commit": "af21d6d8409c46b36830f0bdddd50e67ee17025f",
        "recipe_sha256": "c460b098d09773786552e5b2c98596fab3319a05108d49bd5693da43ee8a7d88",
        "successful_updates": 22380,
        "skipped_updates": 0,
        "nonfinite_updates": 0,
        "native_unit": {
            "unit": NATIVE_UNIT,
            "InvocationID": NATIVE_INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        },
        "verifier_unit": {
            "unit": VERIFIER_UNIT,
            "InvocationID": VERIFIER_INVOCATION,
            "ActiveState": "active",
            "SubState": "exited",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        },
        "verification_sha256": "3" * 64,
        "model_sha256": MODEL_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "model_tensor_fingerprint": "4" * 64,
        "files": {
            "model.pt": {"sha256": MODEL_SHA},
            "model_ep2.pt": {"sha256": CHECKPOINT_SHA},
        },
    }


def _stop(acceptance: dict[str, object]) -> dict[str, object]:
    """Return a synthetic exact-stop capture for the accepted S4 and verifier units."""
    native = acceptance["native_unit"]
    verifier = acceptance["verifier_unit"]
    units = (native, verifier)
    return {
        "before": {
            unit["unit"]: {
                "InvocationID": unit["InvocationID"],
                "ActiveState": "active",
                "SubState": "exited",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
            }
            for unit in units
        },
        "stop": {unit["unit"]: {"returncode": 0} for unit in units},
        "after": {
            unit["unit"]: {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "InvocationID": "",
            }
            for unit in units
        },
        "populated": {},
    }


class TeacherS5AdmissionTests(unittest.TestCase):
    """Check fixed lineage, raw-byte digest gates, and exact owned-unit stops."""

    def test_binding_changes_only_seed_and_run_name_in_argv(self) -> None:
        """Ensure the S5 recipe remains the reviewed S4 recipe with the successor seed."""
        prep = admission.C / "preparation"
        s4 = json.loads((prep / "TEACHER_S4_FIT_BINDING.json").read_bytes())
        s5 = json.loads((prep / "TEACHER_S5_FIT_BINDING.json").read_bytes())
        expected_argv = list(s4["argv"])
        expected_argv[expected_argv.index("fg19_t3_h384x2_e3_s4_v1")] = "fg19_t3_h384x2_e3_s5_v1"
        expected_argv[expected_argv.index("4")] = "5"
        self.assertEqual(s5["argv"], expected_argv)
        for key in (
            "recipe",
            "fit_scope",
            "resource_envelope",
            "runtime_environment",
            "source_binding",
        ):
            self.assertEqual(s5[key], s4[key])
        self.assertEqual(s5["run"]["seed"], 5)
        self.assertEqual(s5["run"]["name"], "fg19_t3_h384x2_e3_s5_v1")
        self.assertEqual(
            s5["parent_teacher_dependency"]["required_invocation_id"], NATIVE_INVOCATION
        )
        self.assertEqual(
            s5["parent_teacher_dependency"]["required_recipe_sha256"],
            "c460b098d09773786552e5b2c98596fab3319a05108d49bd5693da43ee8a7d88",
        )

    def test_acceptance_is_returned_as_the_hash_checked_raw_bytes(self) -> None:
        """Confirm the predecessor receipt remains byte-identical across its gate."""
        value = _acceptance()
        raw = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        expected_sha = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TERMINAL_ACCEPTANCE.json"
            path.write_bytes(raw)
            actual_raw, actual, actual_sha = admission._read_s4_acceptance(path, expected_sha)
        self.assertEqual(actual_raw, raw)
        self.assertEqual(actual, value)
        self.assertEqual(actual_sha, expected_sha)

    def test_acceptance_rejects_wrong_caller_digest_or_native_invocation(self) -> None:
        """Reject mismatched bytes and a predecessor from another native invocation."""
        value = _acceptance()
        raw = (json.dumps(value) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TERMINAL_ACCEPTANCE.json"
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "caller-provided digest"):
                admission._read_s4_acceptance(path, "0" * 64)
            value["native_unit"]["InvocationID"] = "b" * 32
            path.write_text(json.dumps(value) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "InvocationID"):
                admission._read_s4_acceptance(path, digest)

            value = _acceptance()
            value["successful_updates"] = 22379
            path.write_text(json.dumps(value) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "successful_updates"):
                admission._read_s4_acceptance(path, digest)

    def test_stop_requires_exact_owned_units_and_empty_campaign_population(self) -> None:
        """Require both accepted units stopped, gone, and no populated campaign cgroup."""
        value = _acceptance()
        acceptance_raw = (json.dumps(value) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            acceptance_path = Path(directory) / "TERMINAL_ACCEPTANCE.json"
            acceptance_path.write_bytes(acceptance_raw)
            _, accepted, _ = admission._read_s4_acceptance(
                acceptance_path, hashlib.sha256(acceptance_raw).hexdigest()
            )
            stop_value = _stop(accepted)
            stop_raw = (json.dumps(stop_value, separators=(",", ":")) + "\n").encode()
            stop_path = Path(directory) / "STDOUT.raw.json"
            stop_path.write_bytes(stop_raw)
            actual_raw, actual, _ = admission._read_s4_stop(
                stop_path, hashlib.sha256(stop_raw).hexdigest(), accepted
            )
            self.assertEqual(actual_raw, stop_raw)
            self.assertEqual(actual, stop_value)

            stop_value["populated"] = {"/sys/fs/cgroup/wnn-campaign.slice/cgroup.procs": "42"}
            stop_path.write_text(json.dumps(stop_value) + "\n")
            bad_sha = hashlib.sha256(stop_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "populated campaign cgroup"):
                admission._read_s4_stop(stop_path, bad_sha, accepted)

            stop_value = _stop(accepted)
            stop_value["after"]["unowned.service"] = {}
            stop_path.write_text(json.dumps(stop_value) + "\n")
            bad_sha = hashlib.sha256(stop_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "accepted owned units only"):
                admission._read_s4_stop(stop_path, bad_sha, accepted)


if __name__ == "__main__":
    unittest.main()
