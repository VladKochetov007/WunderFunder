"""Render the conditional score-admission stdin body after ROOT binds export receipts."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[5]
DIRECTORY = Path(__file__).resolve().parent
SCORE_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3"
)
EXPORT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3"
)
RECEIPT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1"
)
BODY_TEMPLATE = DIRECTORY / "SCORE_ADMISSION_BODY_TEMPLATE_V6.py"
PLAN_PATH = DIRECTORY / "SCORE_ADMISSION_BINDING_PLAN_V6.json"
HELPER_PATH = RECEIPT_DIRECTORY / "V5_STDLIB_HELPERS_EXTRACTED.py"
OUTPUT_PATH = DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE.py"
BOUNDARY = 'PAYLOAD_B64: dict[str, str] = {"__UNBOUND_PAYLOADS__": "__UNBOUND_PAYLOADS__"}'
HELPER_BYTES = 8_065
HELPER_SHA256 = "f82fca9bc56125433b8255aa2d1f85a438cc93e6e64f7a6634ace90a2735a9ce"
SOURCE_PINS = {
    "first_cell_score_contract.py": (
        19_485,
        "7784c91713dbfd7d11ec3cfd276eb8a62607289067999f4ecd5c03a3f9a18038",
    ),
    "first_cell_score_core.py": (
        27_226,
        "ffe8a6e7aad38b131a78783504ef55c761d4e8037c13bac279f20c5e55f6249e",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}
SOURCE_REVIEW_PIN = (
    7_284,
    "ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a",
)
SCORE_MANIFEST_PIN = (
    6_495,
    "2bff7842f8c6e9be3920458cc575cd216444ccb97b23570765a63c591387705e",
)
SOURCE_REVIEWED_FILE_PINS = (
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3/export_first_cell_v3.py",
        66_080,
        "9bfcd5110d200212bff45d2188f04e2eb094fbaf5cd78831ed03db5a130af8cb",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3/EXPORT_BINDING_V3.json",
        15_798,
        "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3/EXPORT_SOURCE_MANIFEST.json",
        27_569,
        "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v3/test_export_v3_source_manifest_api_pins.py",
        11_832,
        "759cd0f0a578c0fe1fa11ac5cf43f1205fd5cc0d21c20bc7d030ff2d164603fc",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v2/EXPORT_SOURCE_REVIEW_V2.json",
        10_250,
        "f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af",
        "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3/first_cell_score_contract.py",
        19_485,
        "7784c91713dbfd7d11ec3cfd276eb8a62607289067999f4ecd5c03a3f9a18038",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3/first_cell_score_core.py",
        27_226,
        "ffe8a6e7aad38b131a78783504ef55c761d4e8037c13bac279f20c5e55f6249e",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3/first_cell_score_predictions.py",
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3/score_first_cell.py",
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3/dependency_manifest.json",
        6_495,
        "2bff7842f8c6e9be3920458cc575cd216444ccb97b23570765a63c591387705e",
        None,
    ),
    (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v3/test_score_v3_binding_metadata.py",
        6_399,
        "a755bbeb082be953961e3c0c0b6c0a16da6ddfc09afb74b4985f8eaa10843368",
        None,
    ),
)
SOURCE_REVIEW_JOIN_KEYS = (
    "exporter_declared_pin_matches_exact_source_file",
    "exporter_source_manifest_pin_matches_exact_file",
    "exporter_binding_pin_matches_exact_file",
    "score_dependency_manifest_producer_path_bytes_and_sha_match_exporter",
    "score_dependency_manifest_binding_filename_bytes_and_sha_match_binding_v3",
    "score_core_binding_bytes_and_sha_match_binding_v3",
    "score_contract_requires_evaluation_v2_root_and_EXPORT_BINDING_V3_json",
    "binding_evidence_root_and_output_dir_match_score_contract_paths",
    "five_arm_order_shapes_dtype_and_complete_export_status_are_consistent",
)


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _load_body_module() -> ModuleType:
    """Load the pure template module without invoking its remote entrypoint."""
    spec = importlib.util.spec_from_file_location("score_admission_body_template", BODY_TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("score body template could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_source_review_object(review: object) -> dict[str, object]:
    """Validate the V3 review's actual reviewed_files array, without invented subobjects."""
    if not isinstance(review, dict) or (
        review.get("schema"),
        review.get("status"),
    ) != (
        "fg19.first-cell-export-source-review.v3",
        "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
    ):
        raise ValueError("V3 producer/scorer source review schema or status differs")
    rows = review.get("reviewed_files")
    if not isinstance(rows, list) or len(rows) != len(SOURCE_REVIEWED_FILE_PINS):
        raise ValueError("V3 source review reviewed_files membership differs")
    observed_paths: list[str] = []
    for row, expected in zip(rows, SOURCE_REVIEWED_FILE_PINS, strict=True):
        expected_path, expected_bytes, expected_sha, expected_status = expected
        expected_keys = {"path", "bytes", "sha256"}
        if expected_status is not None:
            expected_keys.add("status")
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError("V3 reviewed_files row has unexpected fields")
        path = row.get("path")
        if not isinstance(path, str) or path in observed_paths:
            raise ValueError("V3 source review contains a duplicate or malformed file path")
        observed_paths.append(path)
        if (
            path != expected_path
            or type(row.get("bytes")) is not int
            or row.get("bytes") != expected_bytes
            or not isinstance(row.get("sha256"), str)
            or row.get("sha256") != expected_sha
            or (expected_status is not None and row.get("status") != expected_status)
        ):
            raise ValueError(f"V3 source review file pin differs: {expected_path}")
    join = review.get("producer_consumer_join")
    if (
        not isinstance(join, dict)
        or set(join) != set(SOURCE_REVIEW_JOIN_KEYS)
        or any(join.get(key) is not True for key in SOURCE_REVIEW_JOIN_KEYS)
    ):
        raise ValueError("V3 source review producer/consumer join is not a complete pass")
    return {"status": review["status"], "reviewed_file_count": len(observed_paths)}


def validate_source_review_bytes(raw: bytes) -> dict[str, object]:
    """Validate exact review bytes and its declared array-shaped evidence pins."""
    if (len(raw), sha256(raw)) != SOURCE_REVIEW_PIN:
        raise ValueError("independent producer/consumer source review pin differs")
    review = json.loads(raw)
    _validate_source_review_object(review)
    return review


def _pinned_source_review() -> dict[str, Any]:
    """Require the exact V3 review bytes and actual reviewed_files array contract."""
    path = EXPORT_DIRECTORY / "EXPORT_SOURCE_REVIEW_V3.json"
    raw = path.read_bytes()
    review = validate_source_review_bytes(raw)
    return {
        "path": str(path.relative_to(REPOSITORY)),
        "bytes": len(raw),
        "sha256": sha256(raw),
        "status": review["status"],
        "reviewed_file_count": len(review["reviewed_files"]),
    }


def render() -> tuple[bytes, dict[str, object]]:
    """Render one source-pinned admission body only after actual ROOT receipt pins exist."""
    plan_raw = PLAN_PATH.read_bytes()
    plan = json.loads(plan_raw)
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "fg19.first-cell-score-admission-binding-plan.v6"
    ):
        raise ValueError("score admission binding plan schema differs")
    score_manifest_path = SCORE_DIRECTORY / "dependency_manifest.json"
    score_manifest_raw = score_manifest_path.read_bytes()
    if (len(score_manifest_raw), sha256(score_manifest_raw)) != SCORE_MANIFEST_PIN:
        raise ValueError("frozen scorer V3 dependency manifest pin differs")
    export_authority = plan.get("export_authority")
    if not isinstance(export_authority, dict):
        raise ValueError("export authority inputs are missing")
    receipt_inputs: dict[str, tuple[Path, int, str]] = {}
    for key, staged_name in (
        ("acceptance", "EXPORT_ACCEPTANCE.json"),
        ("exact_stop", "EXPORT_EXACT_STOP.json"),
    ):
        item = export_authority.get(key)
        if not isinstance(item, dict) or item.get("path") in (None, ""):
            raise RuntimeError("actual ROOT export acceptance/exact-stop paths are still unbound")
        size, expected_sha = item.get("bytes"), item.get("sha256")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise RuntimeError("actual ROOT export acceptance/exact-stop pins are still unbound")
        receipt_inputs[staged_name] = (Path(item["path"]), size, expected_sha)

    helper_raw = HELPER_PATH.read_bytes()
    if (len(helper_raw), sha256(helper_raw)) != (HELPER_BYTES, HELPER_SHA256):
        raise ValueError("V5 stdlib helper extraction pin differs")
    source_payloads: dict[str, bytes] = {}
    payload_rows: list[str] = []
    payload_pins: dict[str, dict[str, object]] = {}
    for name, (expected_bytes, expected_sha) in SOURCE_PINS.items():
        path = SCORE_DIRECTORY / name
        raw = path.read_bytes()
        if (len(raw), sha256(raw)) != (expected_bytes, expected_sha):
            raise ValueError(f"frozen score module changed: {name}")
        source_payloads[name] = raw
        payload_rows.append(f"    {name!r}: {base64.b64encode(raw).decode('ascii')!r},")
        payload_pins[name] = {
            "path": str(path.relative_to(REPOSITORY)),
            "bytes": len(raw),
            "sha256": sha256(raw),
        }
    for staged_name, (path, expected_bytes, expected_sha) in receipt_inputs.items():
        raw = path.read_bytes()
        if (len(raw), sha256(raw)) != (expected_bytes, expected_sha):
            raise ValueError(f"root-authority export receipt changed: {staged_name}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"root-authority receipt is not an object: {staged_name}")
        source_payloads[staged_name] = raw
        payload_rows.append(f"    {staged_name!r}: {base64.b64encode(raw).decode('ascii')!r},")
        payload_pins[staged_name] = {"path": str(path), "bytes": len(raw), "sha256": sha256(raw)}

    body_raw = BODY_TEMPLATE.read_bytes()
    body_text = body_raw.decode("utf-8", errors="strict")
    if body_text.count(BOUNDARY) != 1:
        raise ValueError("score admission body must contain one payload placeholder")
    payload_block = "PAYLOAD_B64: dict[str, str] = {\n" + "\n".join(payload_rows) + "\n}"
    body_text = body_text.replace(BOUNDARY, payload_block, 1)
    replacements = {
        "EXPORT_ACCEPTANCE_BYTES = 0": f"EXPORT_ACCEPTANCE_BYTES = {receipt_inputs['EXPORT_ACCEPTANCE.json'][1]}",
        "EXPORT_ACCEPTANCE_SHA256 = UNBOUND": f"EXPORT_ACCEPTANCE_SHA256 = {receipt_inputs['EXPORT_ACCEPTANCE.json'][2]!r}",
        "EXPORT_EXACT_STOP_BYTES = 0": f"EXPORT_EXACT_STOP_BYTES = {receipt_inputs['EXPORT_EXACT_STOP.json'][1]}",
        "EXPORT_EXACT_STOP_SHA256 = UNBOUND": f"EXPORT_EXACT_STOP_SHA256 = {receipt_inputs['EXPORT_EXACT_STOP.json'][2]!r}",
    }
    for old, new in replacements.items():
        if body_text.count(old) != 1:
            raise ValueError(f"body has an ambiguous pin insertion point: {old}")
        body_text = body_text.replace(old, new, 1)
    body_text = body_text.replace("from __future__ import annotations\n", "", 1)
    rendered = (helper_raw.decode("utf-8", errors="strict") + "\n" + body_text).encode("utf-8")
    tree = ast.parse(rendered.decode("utf-8", errors="strict"), filename=str(OUTPUT_PATH))
    if any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree)):
        raise ValueError("remote stdin body must not inspect its filename")
    if sum(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body) != 1:
        raise ValueError("remote body must have exactly one main entrypoint")
    module = _load_body_module()
    acceptance = json.loads(source_payloads["EXPORT_ACCEPTANCE.json"])
    exact_stop = json.loads(source_payloads["EXPORT_EXACT_STOP.json"])
    expected_export_unit = plan.get("export_authority", {}).get("expected_unit")
    if expected_export_unit != "computer-vision-homework-fg19-first-cell-export-v2.service":
        raise ValueError("plan export unit differs from the reviewed V3 exporter identity")
    module.validate_export_authority(
        acceptance,
        exact_stop,
        acceptance_sha256=receipt_inputs["EXPORT_ACCEPTANCE.json"][2],
        binding_sha256="372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
        review_sha256=SOURCE_REVIEW_PIN[1],
        expected_unit=expected_export_unit,
    )
    source_review = _pinned_source_review()
    rendered_binding: dict[str, object] = {
        "schema": "fg19.first-cell-score-admission-source-binding.v6",
        "scope": "Conditional local binding of one CPU-only streaming score admission; no B action, scoring, target/prediction read, or model payload access.",
        "source_review": source_review,
        "helper_source": {
            "path": str(HELPER_PATH.relative_to(REPOSITORY)),
            "bytes": len(helper_raw),
            "sha256": sha256(helper_raw),
        },
        "body_template": {
            "path": str(BODY_TEMPLATE.relative_to(REPOSITORY)),
            "bytes": len(body_raw),
            "sha256": sha256(body_raw),
        },
        "scorer_dependency_manifest": {
            "path": str(score_manifest_path.relative_to(REPOSITORY)),
            "bytes": len(score_manifest_raw),
            "sha256": sha256(score_manifest_raw),
        },
        "scorer_payloads": payload_pins,
        "export_authority": {
            key: payload_pins[name]
            for key, name in (
                ("acceptance", "EXPORT_ACCEPTANCE.json"),
                ("exact_stop", "EXPORT_EXACT_STOP.json"),
            )
        },
        "rendered_source": {
            "path": str(OUTPUT_PATH.relative_to(REPOSITORY)),
            "bytes": len(rendered),
            "sha256": sha256(rendered),
        },
        "score_unit": "computer-vision-homework-fg19-first-cell-score-v1.service",
        "export_evidence_root": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v2",
        "score_evidence_root": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_score_v1",
        "result_path": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v2/RESULT.json",
        "resource_limits": {
            "cpu_quota": "50%",
            "memory_high_bytes": 1_400_000_000,
            "memory_max_bytes": 1_500_000_000,
            "memory_swap_max_bytes": 0,
            "runtime_seconds": 600,
            "cuda_visible_devices": "",
            "blas_threads": 1,
        },
        "outer_reply_keys": ["unit", "outcome", "admission_base64", "launch_base64"],
        "staged_worker_files": [
            *SOURCE_PINS,
            "accepted_export/EXPORT_ACCEPTANCE.json",
            "accepted_export/EXPORT_EXACT_STOP.json",
        ],
        "non_staged_readonly_export_inputs": [
            "EXPORT_BINDING_V3.json",
            "EXPORT_SOURCE_MANIFEST.json",
            "exports/EXPORT_MANIFEST_V1.json",
            "exports/<arm>/{PREDICTION_SIDECAR.json,predictions.npy}",
        ],
    }
    return rendered, rendered_binding


def main() -> None:
    """Write only a fully bound local stdin body and source-binding receipt."""
    rendered, binding = render()
    OUTPUT_PATH.write_bytes(rendered)
    (DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V6.json").write_text(
        json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding["rendered_source"], sort_keys=True))


if __name__ == "__main__":
    main()
