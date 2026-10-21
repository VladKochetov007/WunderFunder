"""Metadata-only fixture for the V12 cache receipt schema correction."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast

PACKET = Path(__file__).resolve().parent
CAMPAIGN = Path(__file__).resolve().parents[2]
RECORDS = CAMPAIGN / "records"
BODY = PACKET / "REMOTE_SUBMISSION_V12.py"
V11_BODY = (
    CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11.py"
)
CACHE_RECORDS = RECORDS / "teacher_cache_cpu_build_v2"
METADATA_PATH = CACHE_RECORDS / "capture/METADATA.json"
ACCEPTANCE_PATH = CACHE_RECORDS / "TERMINAL_ACCEPTANCE.json"
STOP_PATH = CACHE_RECORDS / "exact_stop/STDOUT.raw.json"
FIT_ADMISSION_PATH = RECORDS / "child_allthree_fit_v1/ADMISSION.json"


def literal(tree: ast.Module, name: str) -> object:
    """Return one top-level literal constant without importing the worker body."""
    matches = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        )
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one literal {name}")
    value = matches[0].value
    if value is None:
        raise AssertionError(f"literal {name} has no value")
    return ast.literal_eval(value)


CacheCheck = Callable[[dict[str, object]], dict[str, object]]


def cache_checker() -> tuple[CacheCheck, dict[str, object], dict[str, list[str]]]:
    """Compile only the read-only check_cache helper with synthetic stat hooks."""
    tree = ast.parse(BODY.read_text(encoding="utf-8"), filename=str(BODY))
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_cache"
    ]
    if len(function_nodes) != 1:
        raise AssertionError("V12 check_cache helper is missing or duplicated")

    metadata_raw = METADATA_PATH.read_bytes()
    acceptance_raw = ACCEPTANCE_PATH.read_bytes()
    stop_raw = STOP_PATH.read_bytes()
    fit_admission = json.loads(FIT_ADMISSION_PATH.read_bytes())
    raw_by_name = {
        "metadata.json": metadata_raw,
        "TERMINAL_ACCEPTANCE.json": acceptance_raw,
        "EXACT_STOP.json": stop_raw,
    }
    calls: dict[str, list[str]] = {"stable": [], "stat": []}

    def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
        if path.name not in raw_by_name:
            raise AssertionError(f"unexpected receipt read: {path}")
        if path.name == "metadata.json" and cap != literal(tree, "MAX_METADATA_BYTES"):
            raise AssertionError("metadata cap differs")
        if path.name != "metadata.json" and cap != 10_000:
            raise AssertionError("receipt cap differs")
        calls["stable"].append(path.name)
        raw = raw_by_name[path.name]
        return raw, {"bytes": len(raw), "device": 64512, "inode": 100, "mtime_ns": 1}

    def path_stat(path: Path, *, expected_symlink: bool | None = False) -> dict[str, object]:
        if expected_symlink is not False:
            raise AssertionError("cache outputs must use the non-symlink path gate")
        if path.name not in ("main_homogeneous_train.npy", "main_homogeneous_valid.npy"):
            raise AssertionError(f"unexpected array-stat path: {path}")
        name = "train" if path.name.endswith("train.npy") else "valid"
        calls["stat"].append(path.name)
        return {
            **fit_admission["cache"]["array_stats_only"][name],
            "path": str(path),
            "islink": False,
        }

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def json_object(raw: bytes, name: str) -> dict[str, object]:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"{name} is not an object")
        return value

    namespace: dict[str, object] = {
        "Path": Path,
        "MAX_METADATA_BYTES": literal(tree, "MAX_METADATA_BYTES"),
        "CACHE_METADATA_SHA256": literal(tree, "CACHE_METADATA_SHA256"),
        "CACHE_ACCEPTANCE_SHA256": literal(tree, "CACHE_ACCEPTANCE_SHA256"),
        "CACHE_STOP_SHA256": literal(tree, "CACHE_STOP_SHA256"),
        "SPLIT_SHA256": literal(tree, "SPLIT_SHA256"),
        "EXPECTED_FILESYSTEM_DEVICE": literal(tree, "EXPECTED_FILESYSTEM_DEVICE"),
        "CACHE_ROOT": Path("/synthetic/cache"),
        "CACHE_RECEIPTS_ROOT": Path("/synthetic/cache-receipts"),
        "stable_file": stable_file,
        "path_stat": path_stat,
        "require": require,
        "digest": digest,
        "json_object": json_object,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function_nodes[0]], type_ignores=[]))
    exec(compile(module, str(BODY), "exec"), namespace)
    return cast(CacheCheck, namespace["check_cache"]), fit_admission, calls


class CacheSchemaTests(unittest.TestCase):
    def test_actual_cache_receipts_pass_with_synthetic_array_stats(self) -> None:
        check, fit_admission, calls = cache_checker()
        result = check(fit_admission)
        tree = ast.parse(BODY.read_text(encoding="utf-8"))
        self.assertEqual(
            calls["stable"], ["metadata.json", "TERMINAL_ACCEPTANCE.json", "EXACT_STOP.json"]
        )
        self.assertEqual(
            calls["stat"], ["main_homogeneous_train.npy", "main_homogeneous_valid.npy"]
        )
        self.assertEqual(result["metadata"]["sha256"], literal(tree, "CACHE_METADATA_SHA256"))
        self.assertEqual(result["acceptance"]["sha256"], literal(tree, "CACHE_ACCEPTANCE_SHA256"))
        self.assertEqual(result["exact_stop"]["sha256"], literal(tree, "CACHE_STOP_SHA256"))
        self.assertFalse(result["cache_payloads_read_or_hashed"])

    def test_actual_acceptance_joins_split_through_metadata_and_fit_receipt(self) -> None:
        metadata = json.loads(METADATA_PATH.read_bytes())
        acceptance = json.loads(ACCEPTANCE_PATH.read_bytes())
        fit_admission = json.loads(FIT_ADMISSION_PATH.read_bytes())
        cache = fit_admission["cache"]
        tree = ast.parse(BODY.read_text(encoding="utf-8"))
        self.assertEqual(metadata["split_manifest_sha256"], literal(tree, "SPLIT_SHA256"))
        self.assertNotIn("split_manifest_sha256", acceptance)
        self.assertEqual(
            acceptance["metadata_sha256"], hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest()
        )
        self.assertEqual(cache["metadata_sha256"], acceptance["metadata_sha256"])
        self.assertEqual(
            cache["acceptance_sha256"], hashlib.sha256(ACCEPTANCE_PATH.read_bytes()).hexdigest()
        )
        self.assertEqual(
            cache["exact_stop_sha256"], hashlib.sha256(STOP_PATH.read_bytes()).hexdigest()
        )
        self.assertEqual(
            (acceptance["unit"], acceptance["invocation_id"]),
            (cache["unit"], cache["invocation_id"]),
        )
        self.assertEqual(set(cache["array_stats_only"]), {"train", "valid"})
        for name, count in (("train", 10_535), ("valid", 1_405)):
            output = acceptance["output"][name]
            self.assertEqual(output["shape"], [count, 20_000, 2])
            self.assertEqual(output["dtype"], "float32")
            stat_only = output["stat_only"]
            self.assertEqual(
                {key: stat_only[key] for key in ("bytes", "device", "inode", "mtime_ns")},
                cache["array_stats_only"][name],
            )

    def test_v12_changes_only_the_cache_schema_predicate_and_preserves_payloads(self) -> None:
        v11_raw = V11_BODY.read_bytes()
        v12_raw = BODY.read_bytes()
        old = b"""    require(
        metadata.get("split_manifest_sha256") == SPLIT_SHA256
        and acceptance.get("split_manifest_sha256") == SPLIT_SHA256,
        "homogeneous cache split identity differs",
    )"""
        new = b"""    require(
        metadata.get("split_manifest_sha256") == SPLIT_SHA256
        and acceptance.get("metadata_sha256") == CACHE_METADATA_SHA256,
        "homogeneous cache metadata-to-split identity differs",
    )"""
        self.assertEqual(v11_raw.count(old), 1)
        self.assertEqual(v12_raw.count(new), 1)
        self.assertEqual(v12_raw.replace(new, old), v11_raw)
        v11_tree = ast.parse(v11_raw.decode("utf-8"))
        v12_tree = ast.parse(v12_raw.decode("utf-8"))
        self.assertEqual(literal(v11_tree, "PAYLOAD_B64"), literal(v12_tree, "PAYLOAD_B64"))
        self.assertEqual(len(literal(v12_tree, "PAYLOAD_B64")), 6)


if __name__ == "__main__":
    unittest.main()
