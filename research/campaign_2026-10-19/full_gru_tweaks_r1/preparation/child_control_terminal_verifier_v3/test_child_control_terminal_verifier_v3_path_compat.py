"""Exercise only the V3 restricted Path decoder and pinned failure metadata."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch

CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
PACKET = PREPARATION / "child_control_terminal_verifier_v3"
RECORDS = CAMPAIGN / "records"
V2_FAILURE = RECORDS / "child_control_fit_v1/terminal_verification_v2/capture/STDOUT.raw.json"
CONTROL_LAUNCH = RECORDS / "child_control_fit_v1/LAUNCH.json"
sys.path.insert(0, str(PREPARATION))
sys.path.insert(0, str(PACKET))

checker = importlib.import_module("child_control_terminal_check_v3")


class UnexpectedPathGlobal:
    """Fixture class whose unpickler hook must remain unreachable."""

    loaded = False

    def __setstate__(self, state: object) -> None:
        type(self).loaded = True
        self.state = state


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _empty_safe_globals() -> Iterator[None]:
    """Temporarily model the checker's exact-empty registration boundary."""
    previous = torch.serialization.get_safe_globals()
    torch.serialization.clear_safe_globals()
    try:
        yield
    finally:
        torch.serialization.clear_safe_globals()
        if previous:
            torch.serialization.add_safe_globals(previous)


def _definitions_only(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
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
    ]
    namespace: dict[str, object] = {"__name__": "v3_metadata_fixture"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class PathCompatibilityTests(unittest.TestCase):
    """Prove the one-path exception remains restricted and opt-in."""

    def test_exact_synthetic_dictionary_path_uses_weights_only_and_one_constructor(self) -> None:
        """The exact Python 3.13 reducer form loads one primitive path, not fitted data."""
        self.assertEqual(
            checker.POSIXPATH_PICKLE_ARGS,
            (
                "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt",
            ),
        )
        with (
            _empty_safe_globals(),
            tempfile.TemporaryDirectory(prefix="control-v3-path-fixture-") as temporary,
        ):
            path = Path(temporary) / "synthetic-training-state.pt"
            torch.save(
                {"contract": {"options": {"representation_cache": checker.DICTIONARY_CACHE_PATH}}},
                path,
            )
            loaded, digest, identity, proof = checker._load_control_training_state(path, 1_000_000)
            raw_digest = _sha(path.read_bytes())
            file_bytes = path.stat().st_size
            after_load_globals = torch.serialization.get_safe_globals()
        self.assertEqual(digest, raw_digest)
        self.assertEqual(identity["bytes"], file_bytes)
        self.assertIs(
            type(loaded["contract"]["options"]["representation_cache"]), checker.PosixPath
        )
        self.assertEqual(proof["static_unknown_globals_with_baseline"], [checker.POSIXPATH_GLOBAL])
        self.assertEqual(proof["constructor_calls"], 1)
        self.assertIs(proof["weights_only"], True)
        self.assertEqual(proof["field"], "contract.options.representation_cache")
        self.assertEqual(proof["value"], str(checker.DICTIONARY_CACHE_PATH))
        self.assertTrue(proof["safe_globals_empty_after_load"])
        self.assertEqual(after_load_globals, [])

    def test_decoder_rejects_wrong_argument_type_count_and_path(self) -> None:
        """Only one exact full-path string argument reaches the inert constructor."""
        for args in (
            (),
            (str(checker.DICTIONARY_CACHE_PATH), "extra"),
            (Path(checker.DICTIONARY_CACHE_PATH),),
            ("/tmp/unrelated.pt",),
        ):
            calls = [0]
            restore = checker._path_restore_function(calls)
            with self.subTest(args=args), self.assertRaises(ValueError):
                restore(*args)
            self.assertEqual(calls[0], 1)

    def test_path_walker_rejects_wrong_location_value_and_multiple_paths(self) -> None:
        """A decoded Path is valid only once at contract.options.representation_cache."""
        valid = {"contract": {"options": {"representation_cache": checker.DICTIONARY_CACHE_PATH}}}
        proof = checker._validate_training_state_paths(valid)
        self.assertEqual(proof["occurrence_count"], 1)
        structured_array = np.empty(1, dtype=[("hidden", object)])
        structured_array["hidden"][0] = checker.DICTIONARY_CACHE_PATH
        structured_scalar = np.array((checker.DICTIONARY_CACHE_PATH,), dtype=[("hidden", object)])[
            ()
        ]
        for invalid in (
            {"other": checker.DICTIONARY_CACHE_PATH},
            {"contract": {"options": {"representation_cache": Path("/tmp/wrong.pt")}}},
            {
                "contract": {
                    "options": {"representation_cache": checker.DICTIONARY_CACHE_PATH},
                    "extra": checker.DICTIONARY_CACHE_PATH,
                }
            },
            {"other": np.array([checker.DICTIONARY_CACHE_PATH], dtype=object)},
            {**valid, "hidden": structured_array},
            {**valid, "hidden": structured_scalar},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                checker._validate_training_state_paths(invalid)

    def test_static_global_gate_rejects_other_classes_before_unpickling(self) -> None:
        """An additional class global fails static review before its hook can run."""
        UnexpectedPathGlobal.loaded = False
        with (
            _empty_safe_globals(),
            tempfile.TemporaryDirectory(prefix="control-v3-global-fixture-") as temporary,
        ):
            path = Path(temporary) / "synthetic-unknown-global.pt"
            torch.save({"unknown": UnexpectedPathGlobal()}, path)
            with self.assertRaisesRegex(ValueError, "static globals differ"):
                checker._load_control_training_state(path, 1_000_000)
        self.assertFalse(UnexpectedPathGlobal.loaded)

    def test_entry_reset_removes_an_unrelated_registered_global(self) -> None:
        """Preexisting registrations cannot be accepted as part of the Path baseline."""
        with _empty_safe_globals():
            torch.serialization.add_safe_globals([UnexpectedPathGlobal])
            self.assertIn(UnexpectedPathGlobal, torch.serialization.get_safe_globals())
            with self.assertRaisesRegex(ValueError, "unexpected ambient"):
                checker._load_control_training_state(
                    Path("/definitely/not/opened/training_state.pt"), 1_000_000
                )
            checker._reset_safe_globals()
            self.assertEqual(torch.serialization.get_safe_globals(), [])

    def test_comparison_copy_normalizes_only_verified_path_and_preserves_original(self) -> None:
        """The loaded contract remains typed while only a local comparison copy is stringified."""
        original = {
            "representation_cache": checker.DICTIONARY_CACHE_PATH,
            "seed": 1001,
        }
        compared = checker._validated_options_for_launch_comparison(original)
        self.assertIs(type(original["representation_cache"]), checker.PosixPath)
        self.assertEqual(compared["representation_cache"], str(checker.DICTIONARY_CACHE_PATH))
        self.assertEqual(compared["seed"], 1001)
        self.assertIsNot(compared, original)
        with self.assertRaisesRegex(ValueError, "dictionary cache Path differs"):
            checker._validated_options_for_launch_comparison(
                {"representation_cache": Path("/tmp/wrong.pt")}
            )

    def test_switch_is_exact_and_checker_requires_it_before_terminal_reads(self) -> None:
        """The approved environment marker is exact and the verify path checks it first."""
        key = checker.PATH_COMPAT_EXCEPTION_ENV
        with mock.patch.dict(os.environ, {key: ""}), self.assertRaisesRegex(ValueError, key):
            checker._require_path_compatibility_exception()
        with mock.patch.dict(os.environ, {key: checker.PATH_COMPAT_EXCEPTION_VALUE}):
            self.assertEqual(
                checker._require_path_compatibility_exception(), checker.PATH_COMPAT_EXCEPTION_VALUE
            )
        tree = ast.parse((PACKET / "child_control_terminal_check_v3.py").read_text())
        verify = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "verify_terminal"
        )
        verify_statements = verify.body[1:] if ast.get_docstring(verify) else verify.body
        first = verify_statements[0]
        self.assertIsInstance(first, ast.Assign)
        self.assertIsInstance(first.value, ast.Call)
        self.assertEqual(first.value.func.id, "_require_path_compatibility_exception")
        second = verify_statements[1]
        self.assertIsInstance(second, ast.Expr)
        self.assertEqual(second.value.func.id, "_reset_safe_globals")

    def test_wrapper_requires_switch_as_first_admission_action_and_records_it(self) -> None:
        """The actual rendered wrapper can only stage after the explicit switch passes."""
        namespace = _definitions_only(PACKET / "REMOTE_SUBMISSION.py")
        wrapper = namespace["require_path_exception"]
        key = namespace["PATH_EXCEPTION_ENV"]
        expected = namespace["PATH_EXCEPTION_VALUE"]
        with mock.patch.dict(os.environ, {key: ""}), self.assertRaisesRegex(RuntimeError, key):
            wrapper()
        with mock.patch.dict(os.environ, {key: expected}):
            self.assertEqual(wrapper(), expected)
        tree = ast.parse((PACKET / "REMOTE_SUBMISSION.py").read_text())
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_statements = main.body[1:] if ast.get_docstring(main) else main.body
        self.assertIsInstance(main_statements[0], ast.Assign)
        self.assertEqual(main_statements[0].value.func.id, "require_path_exception")
        environment_assignment = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "environment"
                for target in node.targets
            )
        )
        self.assertIsInstance(environment_assignment.value, ast.Dict)
        self.assertTrue(
            any(
                isinstance(key, ast.Name)
                and key.id == "PATH_EXCEPTION_ENV"
                and isinstance(value, ast.Name)
                and value.id == "path_exception"
                for key, value in zip(
                    environment_assignment.value.keys,
                    environment_assignment.value.values,
                    strict=True,
                )
            )
        )

    def test_v2_failure_capture_pin_is_metadata_only(self) -> None:
        """The V2 failure identity is fixed without opening any fitted payload."""
        raw = V2_FAILURE.read_bytes()
        self.assertEqual(len(raw), 26_978)
        self.assertEqual(
            _sha(raw), "d7f81c3773ee27c48611623689eda482b41dfc6ddee7776c61f905db8e76f7d2"
        )
        capture = json.loads(raw)
        self.assertEqual(capture["invocation_id"], "c890ef75142844a9a8f61ae21e38a97f")
        self.assertEqual(
            (
                capture["unit"]["MainPID"],
                capture["unit"]["ExecMainCode"],
                capture["unit"]["ExecMainStatus"],
                capture["unit"]["InvocationID"],
            ),
            ("0", "1", "1", "c890ef75142844a9a8f61ae21e38a97f"),
        )
        self.assertFalse(capture["terminal_success"])

    def test_pinned_control_launch_uses_the_source_qualified_cache_path(self) -> None:
        """The exact original launch metadata names the accepted dictionary cache path."""
        raw = CONTROL_LAUNCH.read_bytes()
        self.assertEqual(
            _sha(raw), "afd19428c8b6034a9dd40fac78d6adb78b70f3d95f1f964917cbadc828ed27e9"
        )
        launch = json.loads(raw)
        command = launch["command"]
        positions = [
            index for index, value in enumerate(command) if value == "--representation-cache"
        ]
        self.assertEqual(len(positions), 1)
        self.assertEqual(command[positions[0] + 1], str(checker.DICTIONARY_CACHE_PATH))

    def test_rendered_payload_matches_frozen_checker_bytes(self) -> None:
        """Only the exact checker and metadata inputs pass the body pin function."""
        body = PACKET / "REMOTE_SUBMISSION.py"
        namespace = _definitions_only(body)
        payload = namespace["payload_bytes"]()
        checker_raw = (PACKET / "child_control_terminal_check_v3.py").read_bytes()
        self.assertEqual(namespace["CHECKER_SHA"], _sha(checker_raw))
        self.assertEqual(payload["child_control_terminal_check_v3.py"], checker_raw)
        self.assertEqual(
            namespace["UNIT"], "computer-vision-homework-fg19-child-control-verify-v3.service"
        )
        self.assertEqual(
            namespace["D"],
            Path(
                "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_control_verify_v3"
            ),
        )


if __name__ == "__main__":
    unittest.main()
