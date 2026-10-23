from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent
V11 = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v11"
V12_BODY = PACKET / "REMOTE_SUBMISSION_V12.py"
V12_BINDING = PACKET / "SCORE_ADMISSION_SOURCE_BINDING_V12.json"
PROOF_PATH = PACKET / "ACTUAL_SCORE_ADMISSION_V12_CACHE_SCHEMA_PROOF.json"


def pin(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def assigned_literal(tree: ast.Module, name: str) -> object:
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
        raise AssertionError(f"expected one top-level assignment for {name}")
    value = matches[0].value
    if value is None:
        raise AssertionError(f"top-level assignment for {name} has no value")
    return ast.literal_eval(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    old_path = V11 / "REMOTE_SUBMISSION_V11.py"
    old_raw = old_path.read_bytes()
    new_raw = V12_BODY.read_bytes()
    old_block = b"""    require(\n        metadata.get("split_manifest_sha256") == SPLIT_SHA256\n        and acceptance.get("split_manifest_sha256") == SPLIT_SHA256,\n        "homogeneous cache split identity differs",\n    )"""
    new_block = b"""    require(\n        metadata.get("split_manifest_sha256") == SPLIT_SHA256\n        and acceptance.get("metadata_sha256") == CACHE_METADATA_SHA256,\n        "homogeneous cache metadata-to-split identity differs",\n    )"""
    require(old_raw.count(old_block) == 1, "V11 cache predicate is not unique")
    require(new_raw.count(new_block) == 1, "V12 cache predicate is not unique")
    require(new_raw.replace(new_block, old_block) == old_raw, "V12 inverse is not exact V11")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    require(
        ast.dump(
            ast.parse(new_raw.replace(new_block, old_block).decode("utf-8")),
            include_attributes=False,
        )
        == ast.dump(old_tree, include_attributes=False),
        "V12 AST inverse does not equal V11",
    )
    old_payloads = assigned_literal(old_tree, "PAYLOAD_B64")
    new_payloads = assigned_literal(new_tree, "PAYLOAD_B64")
    require(old_payloads == new_payloads and len(new_payloads) == 6, "six worker payloads changed")

    records = CAMPAIGN / "records"
    cache_dir = records / "teacher_cache_cpu_build_v2"
    metadata_path = cache_dir / "capture/METADATA.json"
    acceptance_path = cache_dir / "TERMINAL_ACCEPTANCE.json"
    stop_path = cache_dir / "exact_stop/STDOUT.raw.json"
    fit_admission_path = records / "child_allthree_fit_v1/ADMISSION.json"
    metadata_raw = metadata_path.read_bytes()
    acceptance_raw = acceptance_path.read_bytes()
    stop_raw = stop_path.read_bytes()
    metadata = json.loads(metadata_raw)
    acceptance = json.loads(acceptance_raw)
    exact_stop = json.loads(stop_raw)
    fit_admission = json.loads(fit_admission_path.read_bytes())
    cache = fit_admission["cache"]

    require(
        metadata["split_manifest_sha256"]
        == "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
        "actual cache metadata split differs",
    )
    require("split_manifest_sha256" not in acceptance, "unexpected acceptance split field appeared")
    require(
        acceptance["metadata_sha256"] == hashlib.sha256(metadata_raw).hexdigest(),
        "acceptance metadata join differs",
    )
    require(
        acceptance["source_manifest_sha256"]
        == "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
        "acceptance source-manifest identity differs",
    )
    require(
        acceptance["unit"] == cache["unit"]
        and acceptance["invocation_id"] == cache["invocation_id"],
        "cache fit-admission receipt identity differs",
    )
    require(
        cache["metadata_sha256"] == hashlib.sha256(metadata_raw).hexdigest(),
        "fit admission metadata identity differs",
    )
    require(
        cache["acceptance_sha256"] == hashlib.sha256(acceptance_raw).hexdigest(),
        "fit admission acceptance identity differs",
    )
    require(
        cache["exact_stop_sha256"] == hashlib.sha256(stop_raw).hexdigest(),
        "fit admission exact-stop identity differs",
    )
    require(
        set(cache)
        == {
            "metadata_sha256",
            "acceptance_sha256",
            "exact_stop_sha256",
            "unit",
            "invocation_id",
            "array_stats_only",
        },
        "fit admission cache schema differs",
    )
    require(
        set(cache["array_stats_only"]) == {"train", "valid"}, "cache array stat membership differs"
    )
    require(
        acceptance["status"] == "ACCEPTED"
        and acceptance["schema"] == "fg19-teacher-cache-cpu-terminal-acceptance.v2",
        "cache acceptance status/schema differs",
    )
    require(
        acceptance["output"]["train"]["shape"] == [10535, 20000, 2], "train output metadata differs"
    )
    require(
        acceptance["output"]["valid"]["shape"] == [1405, 20000, 2], "valid output metadata differs"
    )
    for name in ("train", "valid"):
        stat_only = acceptance["output"][name]["stat_only"]
        expected = cache["array_stats_only"][name]
        require(
            {key: stat_only[key] for key in ("bytes", "device", "inode", "mtime_ns")} == expected,
            f"actual {name} stat-only admission does not match acceptance",
        )
        require(acceptance["output"][name]["dtype"] == "float32", f"actual {name} dtype differs")
    unit = cache["unit"]
    before = exact_stop["before"][unit]
    after = exact_stop["after"][unit]
    require(
        before["InvocationID"] == cache["invocation_id"], "cache exact-stop before identity differs"
    )
    require(
        (
            before["ActiveState"],
            before["SubState"],
            before["ExecMainCode"],
            before["ExecMainStatus"],
        )
        == ("active", "exited", "1", "0"),
        "cache exact-stop before state differs",
    )
    require(
        (after["LoadState"], after["ActiveState"], after["SubState"], after["InvocationID"])
        == ("not-found", "inactive", "dead", ""),
        "cache exact-stop after state differs",
    )
    require(
        exact_stop["stop"][unit] == {"returncode": 0, "stdout": "", "stderr": ""},
        "cache exact-stop command differs",
    )
    require(exact_stop["populated"] == {}, "cache exact-stop cgroup population differs")

    binding = json.loads(V12_BINDING.read_text())
    require(
        binding["schema"] == "fg19.first-cell-score-admission-source-binding.v12",
        "V12 binding schema differs",
    )
    require(binding["rendered_source"] == pin(V12_BODY), "V12 binding body pin differs")
    require(
        binding["scorer_payloads"]
        == json.loads((V11 / "SCORE_ADMISSION_SOURCE_BINDING_V11.json").read_text())[
            "scorer_payloads"
        ],
        "V12 binding payloads differ from V11",
    )

    proof = {
        "schema": "fg19.first-cell-score-admission-cache-schema-proof.v12",
        "status": "PASS_V12_ACTUAL_CACHE_SCHEMA_AND_EXACT_INVERSE",
        "candidate": pin(V12_BODY),
        "parent": pin(old_path),
        "binding": pin(V12_BINDING),
        "narrow_change": "One check_cache predicate changes from nonexistent acceptance.split_manifest_sha256 to actual acceptance.metadata_sha256; metadata.split_manifest_sha256 remains checked against the fixed split pin.",
        "inverse": {"restored_v11_bytes_exactly": True, "full_ast_matches_v11_after_inverse": True},
        "six_payload_b64_values_unchanged": True,
        "actual_metadata_only_inputs": {
            "cache_metadata": pin(metadata_path),
            "cache_acceptance": pin(acceptance_path),
            "cache_exact_stop": pin(stop_path),
            "allthree_fit_admission": pin(fit_admission_path),
            "acceptance_has_split_manifest_sha256": False,
            "acceptance_metadata_sha256_matches_metadata": True,
            "fit_admission_cache_identity_matches_actual_receipts": True,
            "train_valid_stat_shapes_match_actual_acceptance": True,
            "exact_stop_native_identity_matches_fit_admission": True,
            "model_or_array_payload_read_or_hashed": False,
        },
        "historical_failure": pin(
            CAMPAIGN / "records/first_cell_evaluation_v1/score_v1/STDERR.raw.txt"
        ),
        "source_review_required": True,
        "tests_run": False,
        "B_or_SSH_action": False,
    }
    PROOF_PATH.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
