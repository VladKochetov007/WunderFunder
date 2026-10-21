"""Prove the helper V2 source differs from V1 only at the three scoped nodes."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
V1 = BASE / "preparation/first_cell_evaluation_score_acceptance_v1/score_result_acceptance.py"
V2 = HERE / "score_result_acceptance_v2.py"
V1_PIN = (45_795, "5987d527382f0ce33e72b4c17ae020d03485df3057b398ae285940ead29d528b")
EXPECTED_CHANGED_NODES = {
    "SCORE_UNIT",
    "EXPECTED_SCORE_CODE_PINS",
    "_validate_registered_observations",
}
EXPECTED_CORE_PIN = (
    26_965,
    "961f566eaeec0cd7e9ab48788666d03da9b0482810a11f16a019b083cfc65005",
)
METADATA_INPUTS = {
    "export_binding_v4": (
        BASE / "preparation/first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json",
        15_796,
        "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
    ),
    "score_v2_admission": (
        BASE / "records/first_cell_evaluation_v1/score_v2/ADMISSION.json",
        69_650,
        "1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631",
    ),
    "alias_parent_diagnostic": (
        BASE
        / "records/first_cell_evaluation_v1/score_v2/alias_parent_diagnostic_v1/STDOUT.raw.json",
        3_218,
        "e6c08c71185d34c8763ec4fc983d72a91003d3626e6dc2da6a53926c49cb8380",
    ),
}


def read_pinned(path: Path, byte_count: int, digest: str) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != byte_count:
        raise ValueError(f"input is not the expected regular metadata file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"input changed before read: {path}")
        raw = os.read(fd, byte_count + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        len(raw) != byte_count
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity
    ):
        raise ValueError(f"input changed during read: {path}")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"input SHA-256 differs: {path}")
    return raw


def top_level_nodes(raw: bytes, filename: str) -> dict[str, str]:
    tree = ast.parse(raw.decode("utf-8", errors="strict"), filename=filename)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = ast.dump(node, include_attributes=False)
    return result


v1_raw = read_pinned(V1, *V1_PIN)
v2_raw = V2.read_bytes()
v1_nodes = top_level_nodes(v1_raw, str(V1))
v2_nodes = top_level_nodes(v2_raw, str(V2))
if set(v1_nodes) != set(v2_nodes):
    raise ValueError("helper V2 added or removed top-level nodes")
changed_nodes = {name for name in v1_nodes if v1_nodes[name] != v2_nodes[name]}
if changed_nodes != EXPECTED_CHANGED_NODES:
    raise ValueError(f"helper V2 changed nodes differ: {sorted(changed_nodes)}")

v2_tree = ast.parse(v2_raw.decode("utf-8", errors="strict"), filename=str(V2))
assignments = {
    node.targets[0].id: node.value
    for node in v2_tree.body
    if isinstance(node, ast.Assign)
    and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
}
if (
    ast.literal_eval(assignments["SCORE_UNIT"])
    != "computer-vision-homework-fg19-first-cell-score-v2.service"
):
    raise ValueError("helper V2 score unit literal differs")
code_pins = ast.literal_eval(assignments["EXPECTED_SCORE_CODE_PINS"])
if code_pins["first_cell_score_core.py"] != EXPECTED_CORE_PIN:
    raise ValueError("helper V2 core pin differs from the frozen V6 source")
for name, expected in {
    "first_cell_score_contract.py": (
        20_165,
        "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}.items():
    if code_pins[name] != expected:
        raise ValueError(f"helper V2 unchanged scorer pin differs: {name}")

inputs: dict[str, dict[str, object]] = {}
parsed: dict[str, dict[str, object]] = {}
for name, (path, byte_count, digest) in METADATA_INPUTS.items():
    raw = read_pinned(path, byte_count, digest)
    inputs[name] = {"path": str(path), "bytes": len(raw), "sha256": digest}
    parsed[name] = json.loads(raw)
binding = parsed["export_binding_v4"]
admission = parsed["score_v2_admission"]
diagnostic = parsed["alias_parent_diagnostic"]
source_identities = binding["registered_data"]["source_identities"]
admission_sources = admission["checks"]["registered_six_sources_stat_only"]["sources_stat_only"]
if set(source_identities) != set(admission_sources) or len(source_identities) != 6:
    raise ValueError("actual six-source binding/admission membership differs")
fields = ("path", "resolved", "islink", "device", "inode", "bytes", "mtime_ns")
for relative, identity in source_identities.items():
    if {key: identity[key] for key in fields} != {
        key: admission_sources[relative][key] for key in fields
    }:
        raise ValueError(f"actual admission and binding leaf pin differ: {relative}")
    leaf = diagnostic["leaves"][relative]
    if {key: leaf[key] for key in fields} != {key: identity[key] for key in fields}:
        raise ValueError(f"actual parent diagnostic and binding leaf pin differ: {relative}")

cache_parent = diagnostic["parents"]["cache"]
datasets_parent = diagnostic["parents"]["datasets"]
if cache_parent["resolved"] != cache_parent["path"] or cache_parent["islink"] is not False:
    raise ValueError("actual cache parent is not the recorded physical/self directory")
if (
    datasets_parent["resolved"] != "/home/grinchenkoeg/WunderFunder/datasets"
    or datasets_parent["islink"] is not True
):
    raise ValueError("actual datasets parent alias differs")
if diagnostic["parents"] != diagnostic["parents_after"]:
    raise ValueError("actual parent diagnostic differs across its two observations")
chain_folds_target_parent = str(Path(source_identities["cache/chain_folds.npz"]["resolved"]).parent)
if chain_folds_target_parent == cache_parent["resolved"]:
    raise ValueError(
        "fixture does not distinguish leaf target parent from actual parent resolution"
    )

proof = {
    "schema": "fg19.first-cell-score-acceptance-helper-v2-delta-proof.v1",
    "status": "PASS_NARROW_ALIAS_AND_SCORE_V2_REBASE",
    "created_utc": datetime.now(UTC).isoformat(),
    "source_delta": {
        "v1": {"path": str(V1), "bytes": len(v1_raw), "sha256": hashlib.sha256(v1_raw).hexdigest()},
        "v2": {"path": str(V2), "bytes": len(v2_raw), "sha256": hashlib.sha256(v2_raw).hexdigest()},
        "changed_top_level_nodes": sorted(changed_nodes),
        "added_or_removed_nodes": [],
        "score_unit": "computer-vision-homework-fg19-first-cell-score-v2.service",
        "score_core_v6_pin": {"bytes": EXPECTED_CORE_PIN[0], "sha256": EXPECTED_CORE_PIN[1]},
    },
    "actual_metadata_inputs": inputs,
    "actual_alias_join": {
        "six_leaf_identity_sets_match": True,
        "cache_parent_path": cache_parent["path"],
        "cache_parent_resolved": cache_parent["resolved"],
        "datasets_parent_path": datasets_parent["path"],
        "datasets_parent_resolved": datasets_parent["resolved"],
        "cache_chain_folds_leaf_resolved": source_identities["cache/chain_folds.npz"]["resolved"],
        "parent_before_after_equal": True,
        "payload_reads_or_hashes": 0,
    },
    "scope": {
        "helper_imported_or_executed": False,
        "scorer_imported_or_executed": False,
        "b_access_or_unit_query": False,
        "model_prediction_data_cache_array_payload_access": False,
        "quality_claim": False,
    },
}
destination = HERE / "HELPER_V2_DELTA_PROOF_FINAL.json"
encoded = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
fd = os.open(
    destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644
)
try:
    offset = 0
    while offset < len(encoded):
        offset += os.write(fd, encoded[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
directory_fd = os.open(HERE, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(
    json.dumps(
        {
            "status": proof["status"],
            "proof": str(destination),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
        sort_keys=True,
    )
)
