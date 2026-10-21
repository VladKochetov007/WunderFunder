"""Static inverse and provenance proof for the V9 local renderer fork.

This script reads only source/metadata files. It does not import project code,
run either renderer, inspect prediction/model/data payloads, or contact B.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREPARATION = CAMPAIGN / "preparation"
HERE = PREPARATION / "first_cell_evaluation_score_admission_v8"
V8_RENDERER = HERE / "ACTUAL_RENDER_SCORE_ADMISSION_V8.py"
V9_RENDERER = HERE / "ACTUAL_RENDER_SCORE_ADMISSION_V9.py"
PLAN = HERE / "ACTUAL_BINDING_PLAN_V8.json"
V8_PLAN = HERE / "SCORE_ADMISSION_BINDING_PLAN_V8.json"
NEGATIVE = HERE / "LOCAL_RENDER_V8_HELPER_PIN_NEGATIVE.json"
V6_BINDING = PREPARATION / "first_cell_evaluation_admission_v6/ADMISSION_SOURCE_BINDING_V6.json"
EXTRACTION = PREPARATION / "first_cell_evaluation_admission_v1/V5_STDLIB_HELPER_EXTRACTION_V1.json"
HELPER = PREPARATION / "first_cell_evaluation_admission_v1/V5_STDLIB_HELPERS_EXTRACTED.py"
PROOF_PATH = HERE / "ACTUAL_RENDER_SCORE_ADMISSION_V9_SOURCE_PROOF.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pin(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "path": path.relative_to(REPOSITORY).as_posix(),
        "sha256": sha256(data),
    }


def normalized_ast(raw: bytes) -> str:
    tree = ast.parse(raw.decode("utf-8", errors="strict"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id in {"HELPER_BYTES", "HELPER_SHA256"}:
            node.value = ast.Constant(value=f"<normalized:{node.targets[0].id}>")
    return ast.dump(tree, include_attributes=False)


def assignment_value(raw: bytes, name: str) -> object:
    tree = ast.parse(raw.decode("utf-8", errors="strict"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1 or not isinstance(matches[0].value, ast.Constant):
        raise AssertionError(f"expected exactly one literal assignment for {name}")
    return matches[0].value.value


def main() -> None:
    v8_raw = V8_RENDERER.read_bytes()
    v9_raw = V9_RENDERER.read_bytes()
    old_bytes = b"HELPER_BYTES = 8_065"
    new_bytes = b"HELPER_BYTES = 8_125"
    old_sha = b'HELPER_SHA256 = "f82fca9bc56125433b8255aa2d1f85a438cc93e6e64f7a6634ace90a2735a9ce"'
    new_sha = b'HELPER_SHA256 = "c8d3b5b6645446f183dec84e79154c94dcdcb7e3dcc4f641685d39b88a26fa43"'
    if v8_raw.count(old_bytes) != 1 or v9_raw.count(new_bytes) != 1:
        raise AssertionError("helper byte pin assignment is absent or duplicated")
    if v8_raw.count(old_sha) != 1 or v9_raw.count(new_sha) != 1:
        raise AssertionError("helper SHA assignment is absent or duplicated")
    restored = v9_raw.replace(new_bytes, old_bytes, 1).replace(new_sha, old_sha, 1)
    if restored != v8_raw:
        raise AssertionError("V9 is not an exact inverse of the two helper-pin edits")
    if normalized_ast(v8_raw) != normalized_ast(v9_raw):
        raise AssertionError("V9 changes operational AST beyond the two data pins")
    if assignment_value(v9_raw, "HELPER_BYTES") != 8125:
        raise AssertionError("V9 helper byte literal mismatch")
    if assignment_value(v9_raw, "HELPER_SHA256") != (
        "c8d3b5b6645446f183dec84e79154c94dcdcb7e3dcc4f641685d39b88a26fa43"
    ):
        raise AssertionError("V9 helper digest literal mismatch")

    helper_raw = HELPER.read_bytes()
    extraction = json.loads(EXTRACTION.read_text(encoding="utf-8"))
    v6 = json.loads(V6_BINDING.read_text(encoding="utf-8"))
    helper_meta = v6["helper_source"]
    if helper_meta["path"] != HELPER.relative_to(REPOSITORY).as_posix():
        raise AssertionError("V6 helper provenance points to another source")
    if helper_meta["provenance_path"] != EXTRACTION.relative_to(REPOSITORY).as_posix():
        raise AssertionError("V6 helper provenance record path differs")
    if helper_meta["provenance_sha256"] != sha256(EXTRACTION.read_bytes()):
        raise AssertionError("V6 extraction provenance digest differs")
    if helper_meta["provenance"] != extraction:
        raise AssertionError("V6 helper provenance object differs from extraction record")
    helper_output = extraction["output"]
    if helper_output != {
        "bytes": len(helper_raw),
        "path": HELPER.relative_to(REPOSITORY).as_posix(),
        "sha256": sha256(helper_raw),
    }:
        raise AssertionError("current helper bytes do not match exact extraction record")
    if (helper_meta["bytes"], helper_meta["sha256"]) != (
        len(helper_raw),
        sha256(helper_raw),
    ):
        raise AssertionError("V6 helper pin does not match exact extracted helper")
    if len(extraction["helpers"]) != 12 or any(
        not item.get("source_exact") for item in extraction["helpers"]
    ):
        raise AssertionError("extraction record does not describe 12 exact source spans")

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    v8_plan = json.loads(V8_PLAN.read_text(encoding="utf-8"))
    negative = json.loads(NEGATIVE.read_text(encoding="utf-8"))
    acceptance = plan["export_authority"]["acceptance"]
    exact_stop = plan["export_authority"]["exact_stop"]
    actual_receipts = [acceptance, exact_stop]
    for receipt in actual_receipts:
        receipt_path = Path(receipt["path"])
        receipt_raw = receipt_path.read_bytes()
        if (len(receipt_raw), sha256(receipt_raw)) != (
            receipt["bytes"],
            receipt["sha256"],
        ):
            raise AssertionError(f"actual root authority receipt pin mismatch: {receipt_path}")
    if acceptance["expected_schema"] != "fg19.first-cell-export-acceptance.v1":
        raise AssertionError("unexpected export acceptance schema")
    if acceptance["expected_status"] != "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED":
        raise AssertionError("unexpected export acceptance status")
    if exact_stop["expected_schema"] != "fg19-first-cell-export-exact-owned-stop.v1":
        raise AssertionError("unexpected export exact-stop schema")
    if (
        Path(acceptance["path"]).relative_to(REPOSITORY).as_posix()
        != negative["authority_receipts"]["root_export_acceptance"]["path"]
    ):
        raise AssertionError("V8 negative record does not bind the same acceptance")
    if (
        Path(exact_stop["path"]).relative_to(REPOSITORY).as_posix()
        != negative["authority_receipts"]["root_exact_stop"]["path"]
    ):
        raise AssertionError("V8 negative record does not bind the same exact stop")
    if v8_plan["frozen_inputs"]["stdlib_helper_source"]["bytes"] != 8065:
        raise AssertionError("original V8 plan does not document the rejected old pin")
    if v8_plan["frozen_inputs"]["stdlib_helper_source"]["sha256"] != (
        "f82fca9bc56125433b8255aa2d1f85a438cc93e6e64f7a6634ace90a2735a9ce"
    ):
        raise AssertionError("original V8 plan old helper hash differs")
    if (
        plan["frozen_inputs"]["stdlib_helper_source"]
        != v8_plan["frozen_inputs"]["stdlib_helper_source"]
    ):
        raise AssertionError("actual plan changed the frozen V8 historical helper metadata")
    if "frozen_inputs" in V9_RENDERER.read_text(encoding="utf-8"):
        raise AssertionError("renderer unexpectedly consumes historical frozen_inputs metadata")

    body_output = HERE / "REMOTE_SUBMISSION_TEMPLATE_V8.py"
    binding_output = HERE / "SCORE_ADMISSION_SOURCE_BINDING_V8.json"
    if body_output.exists() or binding_output.exists():
        raise AssertionError("renderer output already exists; proof must precede one render")

    result = {
        "schema": "fg19.first-cell-score-admission-v9-renderer-source-proof.v1",
        "status": "PASS_V9_HELPER_PIN_INVERSE_AND_LINEAGE_ONLY",
        "scope": (
            "Static byte/AST inverse, helper extraction lineage, and receipt/plan pin check. "
            "No renderer was executed; no project module, model, prediction, target, or array "
            "payload was read; no B/network/unit action was performed."
        ),
        "renderer_inverse": {
            "v8_actual_fork": pin(V8_RENDERER),
            "v9_candidate": pin(V9_RENDERER),
            "only_changed_assignments": [
                {"name": "HELPER_BYTES", "from": 8065, "to": 8125},
                {
                    "name": "HELPER_SHA256",
                    "from": "f82fca9bc56125433b8255aa2d1f85a438cc93e6e64f7a6634ace90a2735a9ce",
                    "to": "c8d3b5b6645446f183dec84e79154c94dcdcb7e3dcc4f641685d39b88a26fa43",
                },
            ],
            "inverse_bytes_equal_v8": True,
            "normalized_ast_equal": True,
        },
        "helper_lineage": {
            "current_helper": pin(HELPER),
            "v6_admission_source_binding": pin(V6_BINDING),
            "v5_extraction_record": pin(EXTRACTION),
            "extraction_method": extraction["extractor"]["source_representation"],
            "exact_helper_functions": len(extraction["helpers"]),
            "v6_provenance_object_equals_extraction_record": True,
            "v9_literals_match_current_extracted_helper": True,
            "plan_helper_pin_is_unchanged_historical_v8_metadata": True,
            "renderer_does_not_consume_plan_frozen_inputs": True,
        },
        "actual_plan_and_receipts": {
            "plan": pin(PLAN),
            "prior_v8_plan": pin(V8_PLAN),
            "v8_fail_closed_record": pin(NEGATIVE),
            "root_export_acceptance": {
                **pin(Path(acceptance["path"])),
                "expected_schema": acceptance["expected_schema"],
                "expected_status": acceptance["expected_status"],
            },
            "root_export_exact_stop": {
                **pin(Path(exact_stop["path"])),
                "expected_schema": exact_stop["expected_schema"],
            },
        },
        "render_outputs_absent": {
            "body": body_output.relative_to(REPOSITORY).as_posix(),
            "binding": binding_output.relative_to(REPOSITORY).as_posix(),
            "both_absent": True,
        },
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    PROOF_PATH.write_bytes(encoded)
    print(
        f"wrote {PROOF_PATH.relative_to(REPOSITORY)} bytes={len(encoded)} sha256={sha256(encoded)}"
    )


if __name__ == "__main__":
    main()
