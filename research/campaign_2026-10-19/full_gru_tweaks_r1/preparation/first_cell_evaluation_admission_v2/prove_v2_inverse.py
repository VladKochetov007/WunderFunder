"""Prove V2 differs from the frozen V1 body only in gpu_owners."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
DIRECTORY = Path(__file__).resolve().parent
V1_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1"
)
V1_BODY = V1_DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE.py"
V1_BINDING = V1_DIRECTORY / "ADMISSION_SOURCE_BINDING_V1.json"
V1_REVIEW = V1_DIRECTORY / "REMOTE_SUBMISSION_SOURCE_REVIEW_V1.json"
V2_BODY = DIRECTORY / "REMOTE_SUBMISSION_V2.py"
V2_BINDING = DIRECTORY / "ADMISSION_SOURCE_BINDING_V2.json"
OUTPUT = DIRECTORY / "V2_INVERSE_BYTE_PROOF.json"
V1_BODY_PIN = {
    "bytes": 243_515,
    "sha256": "e9f9295605726cbc4ba7a3f702263b70c7022c3878a7cdfdf9485a173c2914aa",
}
V1_BINDING_PIN = {
    "bytes": 10_886,
    "sha256": "f56c67ac18a0a78296cfae1f8667d0ba73a6d7a0e4ce76b57f7786c4b7a55a09",
}
V1_REVIEW_PIN = {
    "bytes": 3_001,
    "sha256": "b03a098ea6c0df8265979fac40f54b02b3355bfca3fcc4e74d3fdc34ba461330",
    "status": "REJECT_SOURCE_ONLY_NO_B_ATTEMPT",
}


def sha256(raw: bytes) -> str:
    """Return a SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def function_span(raw: bytes, name: str) -> tuple[int, int, bytes]:
    """Return one exact UTF-8 function span without importing or executing its module."""
    source = raw.decode("utf-8", errors="strict")
    tree = ast.parse(source)
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one top-level {name} function")
    node = matches[0]
    lines = raw.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return start, end, raw[start:end]


def main() -> None:
    """Bind predecessor and candidate, then require exact single-function inversion."""
    v1_raw, v1_binding_raw, v1_review_raw, v2_raw, v2_binding_raw = (
        V1_BODY.read_bytes(),
        V1_BINDING.read_bytes(),
        V1_REVIEW.read_bytes(),
        V2_BODY.read_bytes(),
        V2_BINDING.read_bytes(),
    )
    if (len(v1_raw), sha256(v1_raw)) != tuple(V1_BODY_PIN.values()):
        raise ValueError("frozen V1 body pin differs")
    if (len(v1_binding_raw), sha256(v1_binding_raw)) != tuple(V1_BINDING_PIN.values()):
        raise ValueError("frozen V1 binding pin differs")
    if (len(v1_review_raw), sha256(v1_review_raw)) != (
        V1_REVIEW_PIN["bytes"],
        V1_REVIEW_PIN["sha256"],
    ):
        raise ValueError("frozen V1 negative review pin differs")
    v1_review = json.loads(v1_review_raw)
    if v1_review.get("status") != V1_REVIEW_PIN["status"]:
        raise ValueError("V1 negative source-review status differs")
    v2_binding = json.loads(v2_binding_raw)
    if v2_binding.get("rendered_source") != {
        "bytes": len(v2_raw),
        "path": str(V2_BODY.relative_to(REPOSITORY)),
        "sha256": sha256(v2_raw),
    }:
        raise ValueError("V2 binding does not describe its exact rendered body")

    v1_start, v1_end, v1_function = function_span(v1_raw, "gpu_owners")
    v2_start, v2_end, v2_function = function_span(v2_raw, "gpu_owners")
    prefix_equal = v1_raw[:v1_start] == v2_raw[:v2_start]
    suffix_equal = v1_raw[v1_end:] == v2_raw[v2_end:]
    reconstructed = v2_raw[:v2_start] + v1_function + v2_raw[v2_end:]
    if not prefix_equal or not suffix_equal or reconstructed != v1_raw:
        raise ValueError("V2 is not an exact single-function change from V1")

    proof = {
        "schema": "fg19.first-cell-export-admission-v2-inverse-byte-proof.v1",
        "status": "PASS_EXACT_SINGLE_FUNCTION_INVERSE",
        "scope": "Local byte comparison only; bodies were parsed for AST spans but never imported or executed; no B action or payload reads.",
        "predecessor": {
            "body": {
                "path": str(V1_BODY.relative_to(REPOSITORY)),
                "bytes": len(v1_raw),
                "sha256": sha256(v1_raw),
            },
            "binding": {
                "path": str(V1_BINDING.relative_to(REPOSITORY)),
                "bytes": len(v1_binding_raw),
                "sha256": sha256(v1_binding_raw),
            },
            "negative_source_review": {
                "path": str(V1_REVIEW.relative_to(REPOSITORY)),
                "bytes": len(v1_review_raw),
                "sha256": sha256(v1_review_raw),
                "status": v1_review["status"],
            },
        },
        "candidate": {
            "body": {
                "path": str(V2_BODY.relative_to(REPOSITORY)),
                "bytes": len(v2_raw),
                "sha256": sha256(v2_raw),
            },
            "binding": {
                "path": str(V2_BINDING.relative_to(REPOSITORY)),
                "bytes": len(v2_binding_raw),
                "sha256": sha256(v2_binding_raw),
            },
        },
        "changed_function": {
            "name": "gpu_owners",
            "v1_bytes": len(v1_function),
            "v1_sha256": sha256(v1_function),
            "v2_bytes": len(v2_function),
            "v2_sha256": sha256(v2_function),
            "v1_span_bytes": [v1_start, v1_end],
            "v2_span_bytes": [v2_start, v2_end],
        },
        "inverse": {
            "prefix_equal": prefix_equal,
            "suffix_equal": suffix_equal,
            "replace_candidate_function_with_predecessor_function": True,
            "reconstructed_bytes": len(reconstructed),
            "reconstructed_sha256": sha256(reconstructed),
            "equals_exact_v1_body": reconstructed == v1_raw,
        },
    }
    OUTPUT.write_text(json.dumps(proof, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof["candidate"]["body"], sort_keys=True))


if __name__ == "__main__":
    main()
