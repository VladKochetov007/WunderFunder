"""Prove the V5 conditional packet contains only its scoped V4 repairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
REPOSITORY = Path(__file__).resolve().parents[5]
V4 = PREPARATION / "first_cell_evaluation_score_admission_v4"
V5_BODY = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V5.py"
V5_RENDERER = PACKET / "render_score_admission_v5.py"
V5_PLAN = PACKET / "SCORE_ADMISSION_BINDING_PLAN_V5.json"
EXPECTED_HELPER_BYTES = 4_565
EXPECTED_HELPER_SHA256 = "f278212454a25da7bf573d54600c810a2b9786fff3741845cbf039fd4cf89a19"

OLD_BODY_BLOCK = """    fit_admission_raw, fit_admission_info = stable_file(
        FIT_RECEIPTS_ROOT / "ADMISSION.json", 20_000
    )
    fit_launch_raw, fit_launch_info = stable_file(FIT_RECEIPTS_ROOT / "LAUNCH.json", 20_000)
    require(
        (len(fit_admission_raw), digest(fit_admission_raw))
        == (FIT_ADMISSION_BYTES, FIT_ADMISSION_SHA256),
        "original AllThree fit admission differs",
    )
    require(
        (len(fit_launch_raw), digest(fit_launch_raw)) == (FIT_LAUNCH_BYTES, FIT_LAUNCH_SHA256),
        "original AllThree fit launch differs",
    )
    fit_admission = json_object(fit_admission_raw, "original AllThree admission")
    fit_launch = json_object(fit_launch_raw, "original AllThree launch")
    require(
        fit_admission.get("run_name") == "fg19_child_allthree_f0_s1001_fit_v1"
        and fit_admission.get("unit")
        == "computer-vision-homework-fg19-child-allthree-fit-v1.service"
        and fit_launch.get("unit") == fit_admission.get("unit")
        and is_invocation_id(fit_launch.get("invocation_id")),
        "original AllThree fit identity differs",
    )
    require(
        fit_launch.get("invocation_id") == "94d3a7132a434d448345b0206bd3daa3",
        "original AllThree invocation differs",
    )
"""
NEW_BODY_CALL = """    (
        fit_admission_raw,
        fit_admission_info,
        fit_launch_raw,
        fit_launch_info,
        fit_admission,
        fit_launch,
    ) = check_original_fit_receipts()
"""
HELPER_MARKER = "def check_paths_and_inputs(payload: dict[str, bytes]) -> dict[str, object]:\n"
RENDERER_REPLACEMENTS = (
    ("SCORE_ADMISSION_BODY_TEMPLATE_V4.py", "SCORE_ADMISSION_BODY_TEMPLATE_V5.py"),
    ("SCORE_ADMISSION_BINDING_PLAN_V4.json", "SCORE_ADMISSION_BINDING_PLAN_V5.json"),
    ("EXPORT_SOURCE_REVIEW_V1.json", "EXPORT_SOURCE_REVIEW_V2.json"),
    (
        "fg19.first-cell-score-admission-source-binding.v4",
        "fg19.first-cell-score-admission-source-binding.v5",
    ),
    ("SCORE_ADMISSION_SOURCE_BINDING_V4.json", "SCORE_ADMISSION_SOURCE_BINDING_V5.json"),
)


def sha256(raw: bytes) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(raw).hexdigest()


def helper_source(source: str) -> str:
    """Return the one V5 receipt helper for its independent expected pin."""
    start = source.index("def check_original_fit_receipts")
    stop = source.index(HELPER_MARKER, start)
    return source[start:stop]


def transform_body(source: str, expected_helper: str) -> str:
    """Reconstruct V5 by adding the pinned helper and replacing one call site."""
    if source.count(OLD_BODY_BLOCK) != 1 or source.count(HELPER_MARKER) != 1:
        raise AssertionError("V4 fit-receipt replacement span is not unique")
    source = source.replace(HELPER_MARKER, expected_helper + HELPER_MARKER, 1)
    return source.replace(OLD_BODY_BLOCK, NEW_BODY_CALL, 1)


def transform_renderer(source: str) -> str:
    """Apply only packet-version labels and the corrected V2 review filename."""
    for old, new in RENDERER_REPLACEMENTS:
        if source.count(old) != 1:
            raise AssertionError(f"renderer replacement is not unique: {old}")
        source = source.replace(old, new, 1)
    return source


def transformed_plan() -> bytes:
    """Apply only V5 labels and unbind receipts for the pending exporter revision."""
    plan = json.loads((V4 / "SCORE_ADMISSION_BINDING_PLAN_V4.json").read_bytes())
    plan["schema"] = "fg19.first-cell-score-admission-binding-plan.v5"
    plan["status"] = "LOCAL_CONDITIONAL_TEMPLATE_ONLY_UNBOUND_NOT_REVIEWED_OR_ADMITTED_V5"
    plan["scope"] = (
        "V5 narrowly fixes the original AllThree fit launch identity join by decoding the nested "
        "systemctl-after-submission InvocationID and fixes the source-review filename to "
        "EXPORT_SOURCE_REVIEW_V2.json. V4 remains preserved. The latest exporter V3/scorer V3 "
        "pins and future export acceptance/exact-stop remain unbound; this packet is not "
        "renderable or admissible."
    )
    for key in ("acceptance", "exact_stop"):
        plan["export_authority"][key]["path"] = None
        plan["export_authority"][key]["bytes"] = None
        plan["export_authority"][key]["sha256"] = None
    plan["limits"] = [
        "The prior actual export V1 attempt failed before checkpoint or prediction payload reads; a corrected exporter V3 and scorer V3 pin update are pending. This conditional V5 retains the current scorer V2 pins only as a review baseline and must not be rendered against them.",
        "The next export acceptance and its exact-stop receipt do not yet exist; paths, bytes, and SHA-256 pins remain null and the renderer gate is closed.",
        "This local packet contains no bound remote body or score invocation, and authorizes no B read, staging, launch, score execution, or quality claim.",
        "Root owns final dependency rebinding, independent review, fresh runtime admission, any single launch, and interpretation of later scalar results.",
    ]
    return (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    """Verify exact narrow reconstruction and write a compact proof receipt."""
    v4_body_path = V4 / "SCORE_ADMISSION_BODY_TEMPLATE_V4.py"
    v4_body = v4_body_path.read_text(encoding="utf-8")
    actual_body = V5_BODY.read_bytes()
    helper = helper_source(actual_body.decode("utf-8"))
    helper_raw = helper.encode("utf-8")
    if (len(helper_raw), sha256(helper_raw)) != (
        EXPECTED_HELPER_BYTES,
        EXPECTED_HELPER_SHA256,
    ):
        raise SystemExit("V5 receipt helper differs from its fixed reviewed candidate pin")
    reconstructed_body = transform_body(v4_body, helper).encode()
    if reconstructed_body != actual_body:
        raise SystemExit("V5 body has changes outside the single receipt-preflight repair")

    v4_renderer = (V4 / "render_score_admission_v4.py").read_text(encoding="utf-8")
    reconstructed_renderer = transform_renderer(v4_renderer).encode()
    actual_renderer = V5_RENDERER.read_bytes()
    if reconstructed_renderer != actual_renderer:
        raise SystemExit("V5 renderer has changes beyond versioning and the V2 review path fix")

    reconstructed_plan = transformed_plan()
    actual_plan = V5_PLAN.read_bytes()
    if reconstructed_plan != actual_plan:
        raise SystemExit("V5 plan differs from its explicit conditional-scope transformation")

    proof = {
        "schema": "fg19.first-cell-score-admission-v5-inverse-byte-proof.v1",
        "status": "PASS_EXACT_NARROW_V4_RECONSTRUCTION",
        "scope": "Local source-byte proof only. No renderer execution, B action, module import, score, model, array, or data payload access.",
        "body": {
            "base_path": str(v4_body_path.relative_to(REPOSITORY)),
            "base_bytes": len(v4_body.encode()),
            "base_sha256": sha256(v4_body.encode()),
            "output_path": str(V5_BODY.relative_to(REPOSITORY)),
            "output_bytes": len(actual_body),
            "output_sha256": sha256(actual_body),
            "helper_bytes": len(helper_raw),
            "helper_sha256": sha256(helper_raw),
            "transformation": "Add one helper that checks the pinned original AllThree admission/launch bytes and strict nested submission/systemctl metadata; replace only the inline erroneous top-level invocation lookup with that helper call.",
            "inverse_byte_equality": True,
        },
        "renderer": {
            "base_bytes": len(v4_renderer.encode()),
            "base_sha256": sha256(v4_renderer.encode()),
            "output_bytes": len(actual_renderer),
            "output_sha256": sha256(actual_renderer),
            "transformation": "Only V5 packet filenames/schema, V5 source-binding filename, and EXPORT_SOURCE_REVIEW_V1.json to EXPORT_SOURCE_REVIEW_V2.json.",
            "inverse_byte_equality": True,
        },
        "plan": {
            "base_bytes": len((V4 / "SCORE_ADMISSION_BINDING_PLAN_V4.json").read_bytes()),
            "base_sha256": sha256((V4 / "SCORE_ADMISSION_BINDING_PLAN_V4.json").read_bytes()),
            "output_bytes": len(actual_plan),
            "output_sha256": sha256(actual_plan),
            "transformation": "Only V5 schema/status/scope and explicit unbinding of future export acceptance/exact-stop paths and pins with corresponding pending-exporter limits.",
            "inverse_byte_equality": True,
        },
        "v4_preserved": True,
        "latest_exporter_v3_and_scorer_v3_required_before_render": True,
        "actual_export_receipts_unbound": True,
    }
    output = PACKET / "SCORE_ADMISSION_V5_INVERSE_PROOF.json"
    raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(raw)
    print(
        json.dumps(
            {
                "path": str(output),
                "bytes": len(raw),
                "sha256": sha256(raw),
                "status": proof["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
