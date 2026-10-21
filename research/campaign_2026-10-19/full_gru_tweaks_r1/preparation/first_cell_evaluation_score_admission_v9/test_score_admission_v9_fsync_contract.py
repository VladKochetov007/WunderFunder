"""Focused stdlib checks for the one V9 post-submit directory-fsync insertion."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

REPOSITORY = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
V8_DIRECTORY = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v8"
V9_DIRECTORY = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v9"
V8_BODY = V8_DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE_V8.py"
V8_BINDING = V8_DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V8.json"
V9_BODY = V9_DIRECTORY / "REMOTE_SUBMISSION_V9.py"
V9_BINDING = V9_DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V9.json"
BLOCK = b"""    directory_fd = os.open(
        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
"""
MARKER = b"    launch = submit_once(admission)\n"
FOLLOWING = b'    admission_raw, _ = stable_file(SCORE_D / "ADMISSION.json", 1_000_000)\n'
UNCHANGED_BINDING_KEYS = (
    "export_authority",
    "helper_source",
    "resource_limits",
    "score_unit",
    "scorer_payloads",
    "source_review",
    "staged_worker_files",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def payload_mapping(path: Path) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PAYLOAD_B64"
    ]
    if len(matches) != 1 or matches[0].value is None:
        raise AssertionError("expected one literal payload mapping")
    return ast.literal_eval(matches[0].value)


class ScoreAdmissionV9FsyncTests(unittest.TestCase):
    """Verify ordering and frozen code/receipt payload identity without imports."""

    def test_single_fsync_block_follows_submit_and_precedes_launch_reread(self) -> None:
        prior = V8_BODY.read_bytes()
        current = V9_BODY.read_bytes()
        self.assertEqual(prior.count(MARKER + FOLLOWING), 1)
        self.assertEqual(current.count(MARKER + BLOCK + FOLLOWING), 1)
        self.assertEqual(current.count(BLOCK), 2)
        self.assertEqual(
            current.replace(MARKER + BLOCK + FOLLOWING, MARKER + FOLLOWING, 1),
            prior,
        )
        tree = ast.parse(current.decode("utf-8", errors="strict"))
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        launch_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "launch"
        )
        self.assertEqual(type(main.body[launch_index + 1]), ast.Assign)
        self.assertEqual(type(main.body[launch_index + 2]), ast.Try)
        launch_reread_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "launch_raw"
                for top_target in node.targets
                for target in ast.walk(top_target)
            )
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "stable_file"
                and any(
                    isinstance(arg, ast.BinOp)
                    and isinstance(arg.left, ast.Name)
                    and arg.left.id == "SCORE_D"
                    and isinstance(arg.right, ast.Constant)
                    and arg.right.value == "LAUNCH.json"
                    for arg in call.args
                )
                for call in ast.walk(node)
            )
        )
        self.assertEqual(launch_reread_index, launch_index + 4)
        reply_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        )
        self.assertGreater(reply_index, launch_reread_index)

    def test_six_embedded_payloads_and_binding_contract_are_unchanged(self) -> None:
        self.assertEqual(payload_mapping(V8_BODY), payload_mapping(V9_BODY))
        prior = json.loads(V8_BINDING.read_bytes())
        current = json.loads(V9_BINDING.read_bytes())
        for key in UNCHANGED_BINDING_KEYS:
            self.assertEqual(current[key], prior[key], key)
        self.assertEqual(
            current["rendered_source"],
            {
                "path": V9_BODY.relative_to(REPOSITORY).as_posix(),
                "bytes": V9_BODY.stat().st_size,
                "sha256": sha256(V9_BODY.read_bytes()),
            },
        )
        self.assertEqual(
            set(current["scorer_payloads"]),
            {
                "first_cell_score_contract.py",
                "first_cell_score_core.py",
                "first_cell_score_predictions.py",
                "score_first_cell.py",
                "EXPORT_ACCEPTANCE.json",
                "EXPORT_EXACT_STOP.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
