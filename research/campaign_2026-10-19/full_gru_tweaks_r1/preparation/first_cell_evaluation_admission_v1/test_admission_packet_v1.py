"""Focused static contract tests for the one-shot export admission source."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
TEMPLATE = DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE.py"
SOURCE_BINDING = DIRECTORY / "ADMISSION_SOURCE_BINDING_V1.json"
EXPORT_SOURCE_REVIEW = (
    DIRECTORY.parent / "first_cell_evaluation_export_v1" / "EXPORT_SOURCE_REVIEW_V1.json"
)
EXPECTED_EXPORT_REVIEW = (
    9_633,
    "cf44af7624e81ee99e9705bf08e69b359f7048354df069cad35b26c7ec927c15",
)
EXPECTED_PAYLOADS = {
    "export_first_cell_v1.py": (
        64_315,
        "5e34aa96762c4bd9db934df6b87666f37b80bc8e1ef7e20e6a74a097d33beb00",
    ),
    "EXPORT_BINDING_V1.json": (
        15_822,
        "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c",
    ),
    "EXPORT_SOURCE_MANIFEST.json": (
        27_569,
        "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
    ),
    "ALLTHREE_TERMINAL_ACCEPTANCE.json": (
        20_208,
        "fada500ede63a175d7d055d38fbef5d38110aef481a91771ae3c5717c1e7026f",
    ),
    "ALLTHREE_EXACT_STOP.json": (
        3_208,
        "0154b3f35c4fb849d0db7e7b60ce1f4ce2fa01ab2203c17db65877f89353f13d",
    ),
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def module_ast() -> tuple[bytes, ast.Module]:
    raw = TEMPLATE.read_bytes()
    return raw, ast.parse(raw.decode("utf-8", errors="strict"), filename=str(TEMPLATE))


def find_assignment(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.Assign(targets=[node.target], value=node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node
    raise AssertionError(f"assignment not found: {name}")


class AdmissionPacketContractTests(unittest.TestCase):
    def test_frozen_local_bindings_match_final_template_and_five_payloads(self) -> None:
        raw, tree = module_ast()
        binding = json.loads(SOURCE_BINDING.read_text(encoding="utf-8"))
        self.assertEqual(binding["rendered_source"]["bytes"], len(raw))
        self.assertEqual(binding["rendered_source"]["sha256"], sha256(raw))
        self.assertEqual(
            binding["rendered_source"]["path"], str(TEMPLATE.relative_to(DIRECTORY.parents[4]))
        )
        payloads = ast.literal_eval(find_assignment(tree, "PAYLOAD_B64").value)
        self.assertEqual(set(payloads), set(EXPECTED_PAYLOADS))
        for name, encoded in payloads.items():
            size, expected_sha = EXPECTED_PAYLOADS[name]
            decoded = base64.b64decode(encoded, validate=True)
            self.assertEqual((len(decoded), sha256(decoded)), (size, expected_sha))
            self.assertEqual(binding["payloads"][name]["bytes"], size)
            self.assertEqual(binding["payloads"][name]["sha256"], expected_sha)

    def test_exact_producer_consumer_review_is_bound_to_admission_inputs(self) -> None:
        raw = EXPORT_SOURCE_REVIEW.read_bytes()
        review = json.loads(raw)
        binding = json.loads(SOURCE_BINDING.read_text(encoding="utf-8"))
        review_pin = binding["producer_consumer_review"]
        self.assertEqual((len(raw), sha256(raw)), EXPECTED_EXPORT_REVIEW)
        self.assertEqual(review_pin["bytes"], EXPECTED_EXPORT_REVIEW[0])
        self.assertEqual(review_pin["sha256"], EXPECTED_EXPORT_REVIEW[1])
        self.assertEqual(review_pin["schema"], review["schema"])
        self.assertEqual(review_pin["status"], "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR")
        for entry_name, review_entry in (
            ("export_first_cell_v1.py", review_pin["producer_source"]),
            ("EXPORT_BINDING_V1.json", review_pin["producer_binding"]),
            ("EXPORT_SOURCE_MANIFEST.json", review_pin["source_manifest"]),
        ):
            pinned = binding["payloads"][entry_name]
            self.assertEqual(review_entry["path"], pinned["path"])
            self.assertEqual(review_entry["bytes"], pinned["bytes"])
            self.assertEqual(review_entry["sha256"], pinned["sha256"])

    def test_stdin_body_is_self_contained_and_only_uses_standard_library(self) -> None:
        _, tree = module_ast()
        names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        self.assertNotIn("__file__", names)
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        self.assertTrue(
            set(imports)
            <= {
                "__future__",
                "base64",
                "datetime",
                "hashlib",
                "json",
                "os",
                "pathlib",
                "selectors",
                "stat",
                "subprocess",
                "time",
            }
        )
        self.assertNotIn("torch", imports)
        self.assertNotIn("numpy", imports)

    def test_preflight_precedes_directory_mutation_and_single_submission(self) -> None:
        _, tree = module_ast()
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        main = functions["main"]
        source = ast.get_source_segment(TEMPLATE.read_text(encoding="utf-8"), main)
        self.assertIsNotNone(source)
        main_calls = [
            node.func.id
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual(main_calls.count("preflight"), 1)
        self.assertEqual(main_calls.count("write_payloads"), 1)
        self.assertEqual(main_calls.count("submit_once"), 1)
        self.assertLess(source.index("preflight(payload)"), source.index("write_payloads(payload)"))
        self.assertLess(source.index("write_payloads(payload)"), source.index('"ADMISSION.json"'))
        self.assertLess(source.index('"ADMISSION.json"'), source.index("submit_once(admission)"))

    def test_historical_fit_units_are_never_queried_or_mutated(self) -> None:
        _, tree = module_ast()
        show_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "show"
        ]
        self.assertTrue(show_calls)
        shown_literals = [
            node.args[0].value
            for node in show_calls
            if node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        self.assertEqual(shown_literals, ["wnn-campaign.slice"])
        source = TEMPLATE.read_text(encoding="utf-8")
        self.assertNotIn("systemctl stop", source)
        self.assertNotIn("systemd-run stop", source)
        self.assertIn("no historical unit query", source)

    def test_bounded_export_profile_and_reply_are_exact(self) -> None:
        _, tree = module_ast()
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        submit_source = ast.get_source_segment(
            TEMPLATE.read_text(encoding="utf-8"), functions["submit_once"]
        )
        self.assertIn('"--property=CPUQuota=100%"', submit_source)
        self.assertIn('"--property=MemoryMax=" + str(WORKER_MEMORY_MAX_BYTES)', submit_source)
        self.assertIn('"--property=MemorySwapMax=0"', submit_source)
        self.assertIn('"--property=RuntimeMaxSec=" + str(WORKER_RUNTIME_SECONDS)', submit_source)
        for variable in (
            "TMPDIR=",
            "WNN_PORTABLE_DATA_MANIFEST=",
            "WNN_LOADER_WORKERS=0",
            "NVIDIA_TF32_OVERRIDE=0",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        ):
            self.assertIn(variable, submit_source)
        main = functions["main"]
        reply_dicts = []
        for node in ast.walk(main):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if {"unit", "outcome", "admission_base64", "launch_base64"} <= keys:
                reply_dicts.append(keys)
        self.assertEqual(reply_dicts, [{"unit", "outcome", "admission_base64", "launch_base64"}])

    def test_only_five_fixed_payload_files_are_staged(self) -> None:
        _, tree = module_ast()
        write_payloads = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_payloads"
        )
        assignments = {
            target.id: node.value
            for node in write_payloads.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        stage_map = assignments["stage_map"]
        self.assertIsInstance(stage_map, ast.Dict)
        paths = {key.value for key in stage_map.keys if isinstance(key, ast.Constant)}
        self.assertEqual(
            paths,
            {
                "export_first_cell_v1.py",
                "EXPORT_BINDING_V1.json",
                "EXPORT_SOURCE_MANIFEST.json",
                "accepted_receipts/ALLTHREE_TERMINAL_ACCEPTANCE.json",
                "accepted_receipts/ALLTHREE_EXACT_STOP.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
