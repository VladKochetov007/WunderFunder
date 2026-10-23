"""Render the versioned first-cell export admission stdin body."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
DIRECTORY = Path(__file__).resolve().parent
V1_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1"
)
EXPORT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v2"
)
EXPORT_REVIEW = EXPORT_DIRECTORY / "EXPORT_SOURCE_REVIEW_V2.json"
RECORD_DIRECTORY = (
    REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1"
)
HELPERS = V1_DIRECTORY / "V5_STDLIB_HELPERS_EXTRACTED.py"
HELPER_PROVENANCE = V1_DIRECTORY / "V5_STDLIB_HELPER_EXTRACTION_V1.json"
BODY = DIRECTORY / "ADMISSION_BODY_TEMPLATE_V4.py"
OUTPUT = DIRECTORY / "REMOTE_SUBMISSION_V4.py"
OUTPUT_BINDING = DIRECTORY / "ADMISSION_SOURCE_BINDING_V4.json"
FIXTURE_SOURCE = DIRECTORY / "test_admission_v4_actual_receipts.py"
INVERSE_PROOF_SOURCE = DIRECTORY / "prove_admission_v4_inverse.py"
RENDERER_SOURCE = Path(__file__)
V1_BODY = V1_DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE.py"
V1_BINDING = V1_DIRECTORY / "ADMISSION_SOURCE_BINDING_V1.json"
V1_REVIEW = V1_DIRECTORY / "REMOTE_SUBMISSION_SOURCE_REVIEW_V1.json"
FAILED_V2_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2"
)
FAILED_V2_BODY = FAILED_V2_DIRECTORY / "REMOTE_SUBMISSION_V2.py"
FAILED_V2_TRANSPORT = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v1/TRANSPORT.json"
)
FAILED_V2_STDERR = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v1/STDERR.raw.txt"
)
FAILED_V3_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v3"
)
FAILED_V3_BODY = FAILED_V3_DIRECTORY / "REMOTE_SUBMISSION_V3.py"
FAILED_V3_REVIEW = FAILED_V3_DIRECTORY / "REMOTE_SUBMISSION_SOURCE_REVIEW_V3.json"
FAILED_V3_TRANSPORT = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v2/TRANSPORT.json"
)
FAILED_V3_STDERR = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v2/STDERR.raw.txt"
)
FAILED_V3_PREFLIGHT = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v2/preflight_outcome_v1/STDOUT.raw.json"
)
FAILED_V3_PINS = {
    "body": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "first_cell_evaluation_admission_v3/REMOTE_SUBMISSION_V3.py",
        "bytes": 243_576,
        "sha256": "d7e2ca4852da54d8d1554ff3271dd3d4f369575a6c26c7dc19990a2ca54cb509",
    },
    "source_review": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "first_cell_evaluation_admission_v3/REMOTE_SUBMISSION_SOURCE_REVIEW_V3.json",
        "bytes": 9_040,
        "sha256": "959414c45fb4965993f6968d9fa33310faec95d0e3b2c354c0e12bddcf654269",
    },
    "transport": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/"
        "first_cell_evaluation_v1/export_v2/TRANSPORT.json",
        "bytes": 729,
        "sha256": "f1d6c833909f0897d7adfd8ce456eb4cd9bfbdfe9ce42d536aaf9489bf087955",
    },
    "stderr": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/"
        "first_cell_evaluation_v1/export_v2/STDERR.raw.txt",
        "bytes": 298,
        "sha256": "f8d4b72462ed5bda5533c593c9dee7dfdcfa5ae6fa538853620abe6640a833e3",
    },
    "preflight_capture": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/"
        "first_cell_evaluation_v1/export_v2/preflight_outcome_v1/STDOUT.raw.json",
        "bytes": 586,
        "sha256": "0b5e67d8b840c47683a3df79c5d4f692a04f48b6786d017860b1e1c60ad058c0",
    },
}

INPUT_PINS = {
    "export_first_cell_v1.py": (
        EXPORT_DIRECTORY / "export_first_cell_v1.py",
        64_315,
        "1587a29b2269f6592a6487fc48765a8e86d4cc23aab4951f00966d970ca3dfa1",
    ),
    "EXPORT_BINDING_V1.json": (
        EXPORT_DIRECTORY / "EXPORT_BINDING_V1.json",
        15_798,
        "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9",
    ),
    "EXPORT_SOURCE_MANIFEST.json": (
        EXPORT_DIRECTORY / "EXPORT_SOURCE_MANIFEST.json",
        27_569,
        "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
    ),
    "ALLTHREE_TERMINAL_ACCEPTANCE.json": (
        RECORD_DIRECTORY / "TERMINAL_ACCEPTANCE.json",
        20_208,
        "fada500ede63a175d7d055d38fbef5d38110aef481a91771ae3c5717c1e7026f",
    ),
    "ALLTHREE_EXACT_STOP.json": (
        RECORD_DIRECTORY / "exact_stop/EXACT_STOP.json",
        3_208,
        "0154b3f35c4fb849d0db7e7b60ce1f4ce2fa01ab2203c17db65877f89353f13d",
    ),
}
V5_SOURCE_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v5/REMOTE_SUBMISSION_V5.py",
    "bytes": 710_848,
    "sha256": "4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374",
}
EXPORT_REVIEW_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v2/EXPORT_SOURCE_REVIEW_V2.json",
    "bytes": 10_250,
    "sha256": "f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af",
    "schema": "fg19.first-cell-export-source-review.v2",
    "status": "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
}
FAILED_V2_PINS = {
    "body": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "first_cell_evaluation_admission_v2/REMOTE_SUBMISSION_V2.py",
        "bytes": 243_608,
        "sha256": "a15b2cec55657f3f630fcd18a4fe20037fe08c3186bb0f046aac5480c6e0aeab",
    },
    "transport": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/"
        "first_cell_evaluation_v1/export_v1/TRANSPORT.json",
        "bytes": 729,
        "sha256": "1abf3a11a02e3ea6ba7ef959b08c5d84a5eb3a3f052f9ae3d1dd2ae68cd2afb1",
    },
    "stderr": {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/records/"
        "first_cell_evaluation_v1/export_v1/STDERR.raw.txt",
        "bytes": 769,
        "sha256": "6aab5654e7568faf1e59db72f8870aa9427127807fa9da590d7b21191ddf4245",
    },
}
V1_BODY_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/REMOTE_SUBMISSION_TEMPLATE.py",
    "bytes": 243_515,
    "sha256": "e9f9295605726cbc4ba7a3f702263b70c7022c3878a7cdfdf9485a173c2914aa",
}
V1_BINDING_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/ADMISSION_SOURCE_BINDING_V1.json",
    "bytes": 10_886,
    "sha256": "f56c67ac18a0a78296cfae1f8667d0ba73a6d7a0e4ce76b57f7786c4b7a55a09",
}
V1_REVIEW_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v1/REMOTE_SUBMISSION_SOURCE_REVIEW_V1.json",
    "bytes": 3_001,
    "sha256": "b03a098ea6c0df8265979fac40f54b02b3355bfca3fcc4e74d3fdc34ba461330",
    "status": "REJECT_SOURCE_ONLY_NO_B_ATTEMPT",
}
PLACEHOLDER = 'PAYLOAD_B64: dict[str, str] = {"__UNBOUND_PAYLOADS__": "__UNBOUND_PAYLOADS__"}'


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def render() -> tuple[bytes, dict[str, object]]:
    """Verify every fixed input and render a stdin-safe body without source self-reads."""
    helpers = HELPERS.read_bytes()
    body = BODY.read_bytes()
    helper_provenance_raw = HELPER_PROVENANCE.read_bytes()
    helper_provenance = json.loads(helper_provenance_raw)
    if not isinstance(helper_provenance, dict):
        raise ValueError("helper provenance must be a JSON object")
    if helper_provenance.get("source") != V5_SOURCE_PIN:
        raise ValueError("stdlib helper extraction no longer binds the exact frozen V5 body")
    if helper_provenance.get("output") != {
        "path": str(HELPERS.relative_to(REPOSITORY)),
        "bytes": len(helpers),
        "sha256": sha256(helpers),
    }:
        raise ValueError("extracted helper bytes differ from their provenance")
    body_text = body.decode("utf-8", errors="strict")
    if body_text.count(PLACEHOLDER) != 1:
        raise ValueError("admission body must contain exactly one payload placeholder")

    v1_body_raw = V1_BODY.read_bytes()
    v1_binding_raw = V1_BINDING.read_bytes()
    v1_review_raw = V1_REVIEW.read_bytes()
    for raw, pin, label in (
        (v1_body_raw, V1_BODY_PIN, "V1 rendered body"),
        (v1_binding_raw, V1_BINDING_PIN, "V1 source binding"),
        (v1_review_raw, V1_REVIEW_PIN, "V1 negative review"),
    ):
        if (len(raw), sha256(raw)) != (pin["bytes"], pin["sha256"]):
            raise ValueError(f"frozen {label} pin differs")
    v1_review = json.loads(v1_review_raw)
    if not isinstance(v1_review, dict) or v1_review.get("status") != V1_REVIEW_PIN["status"]:
        raise ValueError("frozen V1 review is not the expected preserved negative")
    if v1_review.get("candidate") != {
        **V1_BODY_PIN,
        "path": V1_BODY_PIN["path"],
    }:
        raise ValueError("frozen V1 review candidate binding differs")

    export_review_raw = EXPORT_REVIEW.read_bytes()
    if (len(export_review_raw), sha256(export_review_raw)) != (
        EXPORT_REVIEW_PIN["bytes"],
        EXPORT_REVIEW_PIN["sha256"],
    ):
        raise ValueError("fixed exporter producer-consumer source review pin differs")
    export_review = json.loads(export_review_raw)
    if not isinstance(export_review, dict):
        raise ValueError("exporter source review must be a JSON object")
    if (export_review.get("schema"), export_review.get("status")) != (
        EXPORT_REVIEW_PIN["schema"],
        EXPORT_REVIEW_PIN["status"],
    ):
        raise ValueError("exporter source review is not the expected passing review")

    failed_v2_body_raw = FAILED_V2_BODY.read_bytes()
    failed_v2_transport_raw = FAILED_V2_TRANSPORT.read_bytes()
    failed_v2_stderr_raw = FAILED_V2_STDERR.read_bytes()
    for raw, pin, label in (
        (failed_v2_body_raw, FAILED_V2_PINS["body"], "failed V2 body"),
        (failed_v2_transport_raw, FAILED_V2_PINS["transport"], "failed V2 transport"),
        (failed_v2_stderr_raw, FAILED_V2_PINS["stderr"], "failed V2 stderr"),
    ):
        if (len(raw), sha256(raw)) != (pin["bytes"], pin["sha256"]):
            raise ValueError(f"preserved V2 preflight evidence pin differs: {label}")
    failed_v2_transport = json.loads(failed_v2_transport_raw)
    if not isinstance(failed_v2_transport, dict) or (
        failed_v2_transport.get("body_bytes"),
        failed_v2_transport.get("body_sha256"),
        failed_v2_transport.get("ssh_returncode"),
        failed_v2_transport.get("stdout_bytes"),
        failed_v2_transport.get("stderr_bytes"),
        failed_v2_transport.get("stderr_sha256"),
        failed_v2_transport.get("persisted_before_parse"),
        failed_v2_transport.get("retry_performed"),
    ) != (
        FAILED_V2_PINS["body"]["bytes"],
        FAILED_V2_PINS["body"]["sha256"],
        1,
        0,
        FAILED_V2_PINS["stderr"]["bytes"],
        FAILED_V2_PINS["stderr"]["sha256"],
        True,
        False,
    ):
        raise ValueError("preserved V2 preflight transport fields differ")

    failed_v3_rows = (
        (FAILED_V3_BODY.read_bytes(), FAILED_V3_PINS["body"], "failed V3 body"),
        (FAILED_V3_REVIEW.read_bytes(), FAILED_V3_PINS["source_review"], "failed V3 source review"),
        (FAILED_V3_TRANSPORT.read_bytes(), FAILED_V3_PINS["transport"], "failed V3 transport"),
        (FAILED_V3_STDERR.read_bytes(), FAILED_V3_PINS["stderr"], "failed V3 stderr"),
        (
            FAILED_V3_PREFLIGHT.read_bytes(),
            FAILED_V3_PINS["preflight_capture"],
            "V3 read-only preflight capture",
        ),
    )
    failed_v3_raw = {}
    for raw, pin, label in failed_v3_rows:
        if (len(raw), sha256(raw)) != (pin["bytes"], pin["sha256"]):
            raise ValueError(f"preserved V3 preflight evidence pin differs: {label}")
        failed_v3_raw[pin["path"]] = raw
    failed_v3_transport = json.loads(failed_v3_raw[FAILED_V3_PINS["transport"]["path"]])
    failed_v3_capture = json.loads(failed_v3_raw[FAILED_V3_PINS["preflight_capture"]["path"]])
    failed_v3_stderr = failed_v3_raw[FAILED_V3_PINS["stderr"]["path"]]
    if not isinstance(failed_v3_transport, dict) or (
        failed_v3_transport.get("body_bytes"),
        failed_v3_transport.get("body_sha256"),
        failed_v3_transport.get("ssh_returncode"),
        failed_v3_transport.get("stdout_bytes"),
        failed_v3_transport.get("stderr_bytes"),
        failed_v3_transport.get("stderr_sha256"),
        failed_v3_transport.get("persisted_before_parse"),
        failed_v3_transport.get("retry_performed"),
    ) != (
        FAILED_V3_PINS["body"]["bytes"],
        FAILED_V3_PINS["body"]["sha256"],
        1,
        0,
        FAILED_V3_PINS["stderr"]["bytes"],
        FAILED_V3_PINS["stderr"]["sha256"],
        True,
        False,
    ):
        raise ValueError("preserved V3 preflight transport fields differ")
    if (
        not isinstance(failed_v3_capture, dict)
        or failed_v3_capture.get("schema") != "fg19.first-cell-export-preflight-outcome.v2"
        or failed_v3_capture.get("no_mutations") is not True
        or failed_v3_capture.get("payloads_read_or_hashed") is not False
        or failed_v3_capture.get("evidence_lexists") is not False
        or failed_v3_capture.get("unit_properties", {}).get("LoadState") != "not-found"
        or failed_v3_capture.get("unit_properties", {}).get("MainPID") != "0"
        or b"original AllThree launch receipt differs" not in failed_v3_stderr
    ):
        raise ValueError("preserved V3 launch-receipt preflight failure differs")
    failed_v3_review = json.loads(failed_v3_raw[FAILED_V3_PINS["source_review"]["path"]])
    if (
        not isinstance(failed_v3_review, dict)
        or failed_v3_review.get("status") != "PASS_SOURCE_ONLY_ADMISSION_V3"
        or failed_v3_review.get("candidate")
        != {
            "path": FAILED_V3_PINS["body"]["path"],
            "bytes": FAILED_V3_PINS["body"]["bytes"],
            "sha256": FAILED_V3_PINS["body"]["sha256"],
        }
    ):
        raise ValueError("preserved V3 source review candidate differs")

    payloads: dict[str, bytes] = {}
    payload_pins: dict[str, dict[str, object]] = {}
    encoded_rows: list[str] = []
    for name, (path, expected_bytes, expected_sha) in INPUT_PINS.items():
        raw = path.read_bytes()
        actual = (len(raw), sha256(raw))
        if actual != (expected_bytes, expected_sha):
            raise ValueError(f"fixed payload pin differs: {name}: {actual}")
        payloads[name] = raw
        encoded = base64.b64encode(raw).decode("ascii")
        encoded_rows.append(f"    {json.dumps(name)}: {json.dumps(encoded)},")
        payload_pins[name] = {
            "path": str(path.relative_to(REPOSITORY)),
            "bytes": len(raw),
            "sha256": sha256(raw),
        }

    reviewed_producer = export_review.get("producer")
    if not isinstance(reviewed_producer, dict):
        raise ValueError("exporter review must include its reviewed producer")
    expected_reviewed_files = {
        "export_first_cell_v1.py": reviewed_producer,
        "EXPORT_BINDING_V1.json": reviewed_producer.get("binding"),
    }
    for name, reviewed_file in expected_reviewed_files.items():
        if not isinstance(reviewed_file, dict):
            raise ValueError(f"exporter review is missing reviewed input {name}")
        expected_path, expected_bytes, expected_sha = INPUT_PINS[name]
        if (
            reviewed_file.get("path") != str(expected_path.relative_to(REPOSITORY))
            or reviewed_file.get("bytes") != expected_bytes
            or reviewed_file.get("sha256") != expected_sha
        ):
            raise ValueError(f"exporter review's producer input pin differs: {name}")
    reviewed_manifest = export_review.get("scope_authority", {}).get("source_manifest")
    manifest_path, manifest_bytes, manifest_sha = INPUT_PINS["EXPORT_SOURCE_MANIFEST.json"]
    if not isinstance(reviewed_manifest, dict) or (
        reviewed_manifest.get("path") != str(manifest_path.relative_to(REPOSITORY))
        or reviewed_manifest.get("bytes") != manifest_bytes
        or reviewed_manifest.get("sha256") != manifest_sha
    ):
        raise ValueError("exporter review's source-manifest input pin differs")

    payload_map = "PAYLOAD_B64: dict[str, str] = {\n" + "\n".join(encoded_rows) + "\n}"
    rendered_body = body_text.replace(PLACEHOLDER, payload_map, 1)
    rendered = (helpers.decode("utf-8", errors="strict") + "\n\n" + rendered_body).encode("utf-8")
    tree = ast.parse(rendered.decode("utf-8", errors="strict"), filename=str(OUTPUT))
    if any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree)):
        raise ValueError("remote stdin body must not inspect its own filename")
    if sum(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body) != 1:
        raise ValueError("remote body must have exactly one main entrypoint")
    rendered_pins = {
        "bytes": len(rendered),
        "sha256": sha256(rendered),
        "path": str(OUTPUT.relative_to(REPOSITORY)),
    }
    local_sources = {}
    for name, path in (
        ("renderer", RENDERER_SOURCE),
        ("focused_fixture", FIXTURE_SOURCE),
        ("inverse_proof_script", INVERSE_PROOF_SOURCE),
    ):
        raw = path.read_bytes()
        local_sources[name] = {
            "path": str(path.relative_to(REPOSITORY)),
            "bytes": len(raw),
            "sha256": sha256(raw),
        }
    binding: dict[str, object] = {
        "schema": "fg19.first-cell-export-admission-source-binding.v4",
        "scope": "Local static binding of one versioned remote admission stdin body; no remote action or model/array/data payload access.",
        "supersedes_v1": {
            "body": V1_BODY_PIN,
            "binding": V1_BINDING_PIN,
            "negative_source_review": V1_REVIEW_PIN,
            "single_change": "Only gpu_owners now reads /proc/<pid>/cmdline for same-UID processes; the campaign-cgroup check still applies to all UIDs.",
        },
        "supersedes_v2_failed_preflight": {
            "body": FAILED_V2_PINS["body"],
            "transport": FAILED_V2_PINS["transport"],
            "stderr": FAILED_V2_PINS["stderr"],
            "outcome": "check_support_helpers failed before evidence-directory creation or systemd-run because two binding paths incorrectly included preparation/; no retry was performed.",
        },
        "supersedes_v3_failed_preflight": {
            **FAILED_V3_PINS,
            "outcome": "The V3 body failed in check_fit_receipts before staging or systemd-run because its parser expected top-level invocation_id/properties absent from the exact original LAUNCH.json. The captured invocation ID is in the strict base64 systemctl_after_submission.stdout_b64 properties. The read-only outcome capture reports the unit not-found and evidence directory absent at its capture time; V4 still repeats every fresh admission gate. No retry or mutation was performed by V3.",
        },
        "local_review_sources": local_sources,
        "helper_source": {
            "path": str(HELPERS.relative_to(REPOSITORY)),
            "bytes": len(helpers),
            "sha256": sha256(helpers),
            "provenance_path": str(HELPER_PROVENANCE.relative_to(REPOSITORY)),
            "provenance_bytes": len(helper_provenance_raw),
            "provenance_sha256": sha256(helper_provenance_raw),
            "provenance": helper_provenance,
        },
        "body_template": {
            "path": str(BODY.relative_to(REPOSITORY)),
            "bytes": len(body),
            "sha256": sha256(body),
        },
        "payloads": payload_pins,
        "producer_consumer_review": {
            **EXPORT_REVIEW_PIN,
            "producer_source": payload_pins["export_first_cell_v1.py"],
            "producer_binding": payload_pins["EXPORT_BINDING_V1.json"],
            "source_manifest": payload_pins["EXPORT_SOURCE_MANIFEST.json"],
        },
        "rendered_source": rendered_pins,
        "remote_reply_contract": ["unit", "outcome", "admission_base64", "launch_base64"],
        "remote_receipts": {
            "admission": {"path": "ADMISSION.json", "unit_field": True},
            "launch": {
                "path": "LAUNCH.json",
                "unit_field": True,
                "invocation_id": "strict lowercase 32-hex",
            },
        },
        "staging": [
            "export_first_cell_v1.py",
            "EXPORT_BINDING_V1.json",
            "EXPORT_SOURCE_MANIFEST.json",
            "accepted_receipts/ALLTHREE_TERMINAL_ACCEPTANCE.json",
            "accepted_receipts/ALLTHREE_EXACT_STOP.json",
        ],
    }
    return rendered, binding


def main() -> None:
    rendered, binding = render()
    OUTPUT.write_bytes(rendered)
    OUTPUT_BINDING.write_text(
        json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding["rendered_source"], sort_keys=True))


if __name__ == "__main__":
    main()
