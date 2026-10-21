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
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_v1"
)
EXPORT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v1"
)
RECEIPT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1"
)
BODY_TEMPLATE = DIRECTORY / "SCORE_ADMISSION_BODY_TEMPLATE_V2.py"
PLAN_PATH = DIRECTORY / "SCORE_ADMISSION_BINDING_PLAN_V2.json"
HELPER_PATH = RECEIPT_DIRECTORY / "V5_STDLIB_HELPERS_EXTRACTED.py"
OUTPUT_PATH = DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE.py"
BOUNDARY = 'PAYLOAD_B64: dict[str, str] = {"__UNBOUND_PAYLOADS__": "__UNBOUND_PAYLOADS__"}'
HELPER_BYTES = 8_065
HELPER_SHA256 = "f82fca9bc56125433b8255aa2d1f85a438cc93e6e64f7a6634ace90a2735a9ce"
SOURCE_PINS = {
    "first_cell_score_contract.py": (
        19_485,
        "09155537feec1d9cd398f6a729ab88201b82e814c1a8c4ac796fe4283940a90d",
    ),
    "first_cell_score_core.py": (
        27_226,
        "e115d12868b8d9c6ac85e1b497b0bc514559d78fff5dada0b0c350e18b69669a",
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
    9_633,
    "cf44af7624e81ee99e9705bf08e69b359f7048354df069cad35b26c7ec927c15",
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


def _pinned_source_review() -> dict[str, Any]:
    """Require the existing independent exporter/scorer source-pair review."""
    path = EXPORT_DIRECTORY / "EXPORT_SOURCE_REVIEW_V1.json"
    raw = path.read_bytes()
    if (len(raw), sha256(raw)) != SOURCE_REVIEW_PIN:
        raise ValueError("independent producer/consumer source review pin differs")
    review = json.loads(raw)
    if (
        not isinstance(review, dict)
        or review.get("status") != "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR"
    ):
        raise ValueError("producer/consumer source review is not a pass")
    return {
        "path": str(path.relative_to(REPOSITORY)),
        "bytes": len(raw),
        "sha256": sha256(raw),
        "status": review["status"],
    }


def render() -> tuple[bytes, dict[str, object]]:
    """Render one source-pinned admission body only after actual ROOT receipt pins exist."""
    plan_raw = PLAN_PATH.read_bytes()
    plan = json.loads(plan_raw)
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "fg19.first-cell-score-admission-binding-plan.v2"
    ):
        raise ValueError("score admission binding plan schema differs")
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
    module.validate_export_authority(
        acceptance,
        exact_stop,
        acceptance_sha256=receipt_inputs["EXPORT_ACCEPTANCE.json"][2],
        binding_sha256="fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c",
        review_sha256=SOURCE_REVIEW_PIN[1],
        expected_unit="computer-vision-homework-fg19-first-cell-export-v1.service",
    )
    source_review = _pinned_source_review()
    rendered_binding: dict[str, object] = {
        "schema": "fg19.first-cell-score-admission-source-binding.v2",
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
        "export_evidence_root": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v1",
        "score_evidence_root": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_score_v1",
        "result_path": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v1/RESULT.json",
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
            "EXPORT_BINDING_V1.json",
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
    (DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V2.json").write_text(
        json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding["rendered_source"], sort_keys=True))


if __name__ == "__main__":
    main()
