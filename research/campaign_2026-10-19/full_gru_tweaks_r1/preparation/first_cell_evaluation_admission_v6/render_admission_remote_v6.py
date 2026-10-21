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
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4"
)
EXPORT_REVIEW = EXPORT_DIRECTORY / "EXPORT_SOURCE_REVIEW_V4.json"
RECORD_DIRECTORY = (
    REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1"
)
FAILED_EXPORT_V4_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/export_v3"
)
FAILED_EXPORT_V4_STARTUP_DIRECTORY = FAILED_EXPORT_V4_DIRECTORY / "startup_health_v1"
LATEST_FAILED_EXPORT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/export_v4"
)
LATEST_FAILED_EXPORT_STARTUP = LATEST_FAILED_EXPORT_DIRECTORY / "startup_health_v1"
HELPERS = V1_DIRECTORY / "V5_STDLIB_HELPERS_EXTRACTED.py"
HELPER_PROVENANCE = V1_DIRECTORY / "V5_STDLIB_HELPER_EXTRACTION_V1.json"
BODY = DIRECTORY / "ADMISSION_BODY_TEMPLATE_V6.py"
OUTPUT = DIRECTORY / "REMOTE_SUBMISSION_V6.py"
OUTPUT_BINDING = DIRECTORY / "ADMISSION_SOURCE_BINDING_V6.json"
FIXTURE_SOURCE = DIRECTORY / "test_admission_v6_export_review_contract.py"
HISTORY_FIXTURE_SOURCE = DIRECTORY / "test_admission_v6_historical_failure_binding.py"
INVERSE_PROOF_SOURCE = DIRECTORY / "prove_admission_v6_inverse.py"
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
    "export_first_cell_v4.py": (
        EXPORT_DIRECTORY / "export_first_cell_v4.py",
        66_790,
        "87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e",
    ),
    "EXPORT_BINDING_V4.json": (
        EXPORT_DIRECTORY / "EXPORT_BINDING_V4.json",
        15_796,
        "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
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
FAILED_EXPORT_V4_PINS = {
    "launch_acceptance": {
        "path": FAILED_EXPORT_V4_DIRECTORY / "LAUNCH_ACCEPTANCE.json",
        "bytes": 2_236,
        "sha256": "cf66be8426133d6bda4fd9e0174b9c28cc2cf9ae55dc511c3affcf330152a07b",
    },
    "root_transport": {
        "path": FAILED_EXPORT_V4_DIRECTORY / "TRANSPORT.json",
        "bytes": 731,
        "sha256": "433f495358fb4ccbbb597d7a69b475c27a6f6b0c92002579283bfa52433197b7",
    },
    "startup_observation": {
        "path": FAILED_EXPORT_V4_STARTUP_DIRECTORY / "STDOUT.raw.json",
        "bytes": 44_429,
        "sha256": "18124f5744d2e2edc031d94819718d0a6a6745c2bd38886db5d902fbbc16ea05",
    },
    "startup_transport": {
        "path": FAILED_EXPORT_V4_STARTUP_DIRECTORY / "TRANSPORT.json",
        "bytes": 608,
        "sha256": "15c3eca21dab0d6758eb37bab6385af23ccfc75cf0cf1913d2b26ea21de29d2f",
    },
    "unit_log_tail": {
        "path": FAILED_EXPORT_V4_STARTUP_DIRECTORY / "unit.log.tail",
        "bytes": 827,
        "sha256": "d7779f1d24a84204577c27a351fe3e4fc813d8c460e12953c606a13806f1be42",
    },
    "failure_review": {
        "path": FAILED_EXPORT_V4_STARTUP_DIRECTORY / "ACTUAL_FAILURE_METADATA_REVIEW.json",
        "bytes": 14_887,
        "sha256": "ecc08aef8871050c71046e44041022162d27d0706d759f54e58fa4a0fad29905",
    },
}
LATEST_FAILED_EXPORT_PINS = {
    "launch_acceptance": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "LAUNCH_ACCEPTANCE.json",
        "bytes": 2_242,
        "sha256": "3f0eabf089c50463c58298dc2e4ce25fe9b2736242fe49799ec4dcbcbd1e1d0a",
    },
    "root_transport": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "TRANSPORT.json",
        "bytes": 731,
        "sha256": "8411f88ff20dfc56fafdf8067ed17ba192e0a2f82e8514455f147dd89ef358ec",
    },
    "root_stdout": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "STDOUT.raw.json",
        "bytes": 38_528,
        "sha256": "c23f7108dab26711ec0168c923c1f3fb4d564d39284b5c4c7bfa4815ce9b029b",
    },
    "root_stderr": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "STDERR.raw.txt",
        "bytes": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "admission": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "ADMISSION.json",
        "bytes": 24_660,
        "sha256": "00e24339145b43c921f0f7ea3ce2de2b58f76cb49bebbb42ac9e0231c28b1b09",
    },
    "launch": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "LAUNCH.json",
        "bytes": 4_130,
        "sha256": "47d39b3657a8486ddac3210ad9ed86ccf00fd913b8fb850798d43804f95c2df8",
    },
    "launch_review": {
        "path": LATEST_FAILED_EXPORT_DIRECTORY / "ACTUAL_LAUNCH_METADATA_REVIEW.json",
        "bytes": 12_509,
        "sha256": "3650ca24de670bdeed4da3859b504e11831548200e7302a6ab310dd632b5c002",
    },
    "startup_observation": {
        "path": LATEST_FAILED_EXPORT_STARTUP / "STDOUT.raw.json",
        "bytes": 45_026,
        "sha256": "a1b1799aa1024cc7e8d721dbca2edefa7e5e605dc68314de31f44a1c4ccc4638",
    },
    "startup_stderr": {
        "path": LATEST_FAILED_EXPORT_STARTUP / "STDERR.raw.txt",
        "bytes": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "startup_transport": {
        "path": LATEST_FAILED_EXPORT_STARTUP / "TRANSPORT.json",
        "bytes": 608,
        "sha256": "f0884b78e415337e7389fc2f52353c21c561f406528a86c314365b24dd004ed5",
    },
    "unit_log_tail": {
        "path": LATEST_FAILED_EXPORT_STARTUP / "unit.log.tail",
        "bytes": 1_200,
        "sha256": "1a33f07c4337fe8837bb5a227148f51f6893de7e6e7878954b5fbc0e5d2f7c5b",
    },
    "observation": {
        "path": LATEST_FAILED_EXPORT_STARTUP / "OBSERVATION.json",
        "bytes": 1_539,
        "sha256": "536940b9f9afde989e669213233c4a5d29ee6ade7052f681f39ff22dd95c722d",
    },
    "failure_review": {
        "path": LATEST_FAILED_EXPORT_STARTUP / "ACTUAL_FAILURE_METADATA_REVIEW.json",
        "bytes": 12_980,
        "sha256": "ce81c21d2dd485ed320978e9f455a1363f9e9567898dffb298538a376067b2fc",
    },
}
V5_SOURCE_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v5/REMOTE_SUBMISSION_V5.py",
    "bytes": 710_848,
    "sha256": "4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374",
}
EXPORT_REVIEW_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v4/EXPORT_SOURCE_REVIEW_V4.json",
    "bytes": 8_712,
    "sha256": "64c93d8279a2daeb15f2750e6366a0fe94a268c7b8e9a04dbdb28e4a36a56e73",
    "schema": "fg19.first-cell-export-source-review.v4",
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


def reviewed_file_map(review: object) -> dict[str, dict[str, object]]:
    """Validate V4's actual reviewed_files array and index its unique exact pins."""
    if not isinstance(review, dict):
        raise ValueError("exporter source review must be a JSON object")
    rows = review.get("reviewed_files")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("exporter V4 review must contain its exact 12 reviewed-file rows")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("exporter V4 reviewed-file row is malformed")
        row_keys = set(row)
        base_keys = {"path", "bytes", "sha256"}
        is_prior_review = row.get("path") == (
            "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
            "first_cell_evaluation_export_v3/EXPORT_SOURCE_REVIEW_V3.json"
        )
        if row_keys != base_keys and not (is_prior_review and row_keys == base_keys | {"status"}):
            raise ValueError("exporter V4 reviewed-file row has unexpected fields")
        if (
            type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])
        ):
            raise ValueError("exporter V4 reviewed-file row is malformed")
        if is_prior_review and row.get("status") != "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR":
            raise ValueError("exporter V4 previous-review status pin differs")
        if row["path"] in result:
            raise ValueError("exporter V4 review has a duplicate reviewed-file path")
        result[row["path"]] = row
    return result


def checked_producer_review_pins(review: object) -> dict[str, dict[str, object]]:
    """Join review V4's actual reviewed_files rows to the three staged producer inputs."""
    if not isinstance(review, dict) or (
        review.get("schema"),
        review.get("status"),
    ) != (
        EXPORT_REVIEW_PIN["schema"],
        EXPORT_REVIEW_PIN["status"],
    ):
        raise ValueError("exporter review is not the exact passing V4 source review")
    reviewed_files = reviewed_file_map(review)
    producer_review_pins: dict[str, dict[str, object]] = {}
    for name in (
        "export_first_cell_v4.py",
        "EXPORT_BINDING_V4.json",
        "EXPORT_SOURCE_MANIFEST.json",
    ):
        path, expected_bytes, expected_sha = INPUT_PINS[name]
        relative = str(path.relative_to(REPOSITORY))
        expected_reviewed = {
            "path": relative,
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }
        if reviewed_files.get(relative) != expected_reviewed:
            raise ValueError(f"exporter review's producer input pin differs: {name}")
        producer_review_pins[name] = expected_reviewed
    return producer_review_pins


def bind_previous_export_failure() -> dict[str, object]:
    """Pin the first actual export failure as root-only metadata, never a B input."""
    raw_by_name: dict[str, bytes] = {}
    pin_rows: dict[str, dict[str, object]] = {}
    for name, pin in FAILED_EXPORT_V4_PINS.items():
        path = pin["path"]
        raw = path.read_bytes()
        expected = (pin["bytes"], pin["sha256"])
        if (len(raw), sha256(raw)) != expected:
            raise ValueError(f"preserved first export failure pin differs: {name}")
        relative = str(path.relative_to(REPOSITORY))
        pin_rows[name] = {
            "path": relative,
            "bytes": len(raw),
            "sha256": sha256(raw),
        }
        raw_by_name[name] = raw

    launch_acceptance = json.loads(raw_by_name["launch_acceptance"])
    root_transport = json.loads(raw_by_name["root_transport"])
    startup = json.loads(raw_by_name["startup_observation"])
    startup_transport = json.loads(raw_by_name["startup_transport"])
    failure_review = json.loads(raw_by_name["failure_review"])
    unit_log_tail = raw_by_name["unit_log_tail"]
    if not isinstance(launch_acceptance, dict) or not isinstance(root_transport, dict):
        raise ValueError("historical root launch records are malformed")
    if (
        launch_acceptance.get("schema") != "fg19.first-cell-export-launch-acceptance.v1"
        or launch_acceptance.get("status") != "ACCEPTED_SINGLE_EXPORT_LAUNCH_NOT_QUALITY"
        or launch_acceptance.get("unit")
        != "computer-vision-homework-fg19-first-cell-export-v1.service"
        or launch_acceptance.get("invocation_id") != "da6c9e1f5897478f9f3f686de58e39cc"
        or launch_acceptance.get("scientific_fits_started") != 0
        or launch_acceptance.get("quality_result") is not None
    ):
        raise ValueError("historical export launch acceptance identity differs")
    root_transport_fields = (
        "schema",
        "started_utc",
        "finished_utc",
        "body_bytes",
        "body_sha256",
        "source_review_bytes",
        "source_review_sha256",
        "ssh_returncode",
        "timed_out",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "persisted_before_parse",
        "retry_performed",
    )
    if any(
        launch_acceptance.get("raw_transport", {}).get(key) != root_transport.get(key)
        for key in root_transport_fields
    ) or (
        root_transport.get("ssh_returncode"),
        root_transport.get("timed_out"),
        root_transport.get("stderr_bytes"),
        root_transport.get("persisted_before_parse"),
        root_transport.get("retry_performed"),
    ) != (0, False, 0, True, False):
        raise ValueError("historical root transport does not match its launch acceptance")
    if not isinstance(startup, dict) or not isinstance(startup_transport, dict):
        raise ValueError("historical startup observation is malformed")
    systemd = startup.get("systemd")
    if (
        startup.get("schema") != "fg19.first-cell-export-bounded-status.v1"
        or startup.get("unit") != launch_acceptance.get("unit")
        or startup.get("invocation_id") != launch_acceptance.get("invocation_id")
        or startup.get("array_payloads_read_or_hashed") is not False
        or startup.get("terminal_metadata") != {}
        or not isinstance(systemd, dict)
        or (
            systemd.get("Id"),
            systemd.get("InvocationID"),
            systemd.get("ActiveState"),
            systemd.get("SubState"),
            systemd.get("Result"),
            systemd.get("ExecMainCode"),
            systemd.get("ExecMainStatus"),
            systemd.get("MainPID"),
        )
        != (
            launch_acceptance.get("unit"),
            launch_acceptance.get("invocation_id"),
            "failed",
            "failed",
            "exit-code",
            "1",
            "1",
            "0",
        )
    ):
        raise ValueError("historical startup observation does not bind the failed invocation")
    if (
        startup_transport.get("schema"),
        startup_transport.get("ssh_returncode"),
        startup_transport.get("timed_out"),
        startup_transport.get("stdout_bytes"),
        startup_transport.get("stdout_sha256"),
        startup_transport.get("stderr_bytes"),
        startup_transport.get("persisted_before_parse"),
        startup_transport.get("retry_performed"),
    ) != (
        "fg19.first-cell-export-status-root-transport.v1",
        0,
        False,
        pin_rows["startup_observation"]["bytes"],
        pin_rows["startup_observation"]["sha256"],
        0,
        True,
        False,
    ):
        raise ValueError("historical startup transport does not bind the raw observation")
    if (
        not isinstance(failure_review, dict)
        or failure_review.get("schema")
        != "fg19.first-cell-export-startup-failure-metadata-review.v1"
        or failure_review.get("status") != "PASS_METADATA_ONLY_SOURCE_GATE_FAILURE_CONFIRMED"
        or failure_review.get("captured_native_observation", {}).get("strict_success_tuple_matched")
        is not False
        or failure_review.get("captured_native_observation", {}).get(
            "array_payloads_read_or_hashed"
        )
        is not False
        or failure_review.get("execution_state", {}).get("attempt_retried") is not False
        or failure_review.get("execution_state", {}).get("new_fit_or_score") is not False
        or failure_review.get("execution_state", {}).get(
            "output_directory_or_prediction_artifact_observed"
        )
        is not False
        or failure_review.get("source_gate_failure", {})
        .get("canonical_api_pin_comparison", {})
        .get("classification")
        != "Deterministic metadata-shape comparison bug; the audited API sizes and content hashes are equal, and the captured exception does not establish corrupted or changed source bytes."
        or b"canonical API pin differs from the full source manifest" not in unit_log_tail
    ):
        raise ValueError("historical first export failure classification differs")

    return {
        "classification": "FIRST_EXPORT_INVOCATION_FAILED_IN_SOURCE_METADATA_GATE",
        "historical_only": True,
        "root_only_not_staged_to_B": True,
        "old_unit_must_not_be_queried_stopped_or_retried": True,
        "unit": launch_acceptance["unit"],
        "invocation_id": launch_acceptance["invocation_id"],
        "files": pin_rows,
    }


def bind_latest_failed_export() -> dict[str, object]:
    """Pin the second export attempt's saved failure metadata as root-only history."""
    raw_by_name: dict[str, bytes] = {}
    pin_rows: dict[str, dict[str, object]] = {}
    for name, pin in LATEST_FAILED_EXPORT_PINS.items():
        raw = pin["path"].read_bytes()
        if (len(raw), sha256(raw)) != (pin["bytes"], pin["sha256"]):
            raise ValueError(f"preserved latest failed export pin differs: {name}")
        relative = str(pin["path"].relative_to(REPOSITORY))
        pin_rows[name] = {"path": relative, "bytes": len(raw), "sha256": sha256(raw)}
        raw_by_name[name] = raw

    launch_acceptance = json.loads(raw_by_name["launch_acceptance"])
    transport = json.loads(raw_by_name["root_transport"])
    launch_review = json.loads(raw_by_name["launch_review"])
    startup = json.loads(raw_by_name["startup_observation"])
    startup_transport = json.loads(raw_by_name["startup_transport"])
    observation = json.loads(raw_by_name["observation"])
    failure_review = json.loads(raw_by_name["failure_review"])
    unit_log_tail = raw_by_name["unit_log_tail"]
    expected_unit = "computer-vision-homework-fg19-first-cell-export-v2.service"
    expected_invocation = "555122f41c884850a6bd4e847c14e82f"

    if not isinstance(launch_acceptance, dict) or not isinstance(transport, dict):
        raise ValueError("latest failed export launch records are malformed")
    if (
        launch_acceptance.get("schema") != "fg19.first-cell-export-launch-acceptance.v1"
        or launch_acceptance.get("status") != "ACCEPTED_SINGLE_EXPORT_LAUNCH_NOT_QUALITY"
        or launch_acceptance.get("unit") != expected_unit
        or launch_acceptance.get("invocation_id") != expected_invocation
        or launch_acceptance.get("scientific_fits_started") != 0
        or launch_acceptance.get("quality_result") is not None
    ):
        raise ValueError("latest failed export launch identity differs")
    if (
        transport.get("schema") != "fg19.first-cell-export-root-ssh-transport.v1"
        or transport.get("ssh_returncode") != 0
        or transport.get("timed_out") is not False
        or transport.get("stderr_bytes") != 0
        or transport.get("persisted_before_parse") is not True
        or transport.get("retry_performed") is not False
        or transport.get("stdout_bytes") != pin_rows["root_stdout"]["bytes"]
        or transport.get("stdout_sha256") != pin_rows["root_stdout"]["sha256"]
        or transport.get("stderr_bytes") != pin_rows["root_stderr"]["bytes"]
        or transport.get("stderr_sha256") != pin_rows["root_stderr"]["sha256"]
    ):
        raise ValueError("latest failed export transport does not bind saved raw streams")
    transport_fields = (
        "schema",
        "started_utc",
        "finished_utc",
        "body_bytes",
        "body_sha256",
        "source_review_bytes",
        "source_review_sha256",
        "ssh_returncode",
        "timed_out",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "persisted_before_parse",
        "retry_performed",
    )
    if any(
        launch_acceptance.get("raw_transport", {}).get(key) != transport.get(key)
        for key in transport_fields
    ):
        raise ValueError("latest failed export acceptance does not join root transport")
    receipt_pins = launch_acceptance.get("receipts")
    if not isinstance(receipt_pins, dict) or receipt_pins != {
        "ADMISSION.json": {
            "bytes": pin_rows["admission"]["bytes"],
            "sha256": pin_rows["admission"]["sha256"],
        },
        "LAUNCH.json": {
            "bytes": pin_rows["launch"]["bytes"],
            "sha256": pin_rows["launch"]["sha256"],
        },
    }:
        raise ValueError("latest failed export acceptance receipt pins differ")

    if not isinstance(launch_review, dict) or (
        launch_review.get("schema"),
        launch_review.get("status"),
    ) != (
        "fg19.first-cell-export-actual-launch-metadata-review.v1",
        "PASS_LOCAL_ACTUAL_LAUNCH_METADATA_ONLY",
    ):
        raise ValueError("latest failed export launch metadata review differs")
    actual_launch = launch_review.get("actual_launch")
    if not isinstance(actual_launch, dict) or (
        actual_launch.get("unit"),
        actual_launch.get("invocation_id"),
        actual_launch.get("outcome"),
        actual_launch.get("scientific_fits_started"),
        actual_launch.get("quality_result"),
        actual_launch.get("terminal_or_export_result_reviewed"),
    ) != (expected_unit, expected_invocation, "submitted", 0, None, False):
        raise ValueError("latest failed export launch review identity differs")

    if not isinstance(startup, dict) or not isinstance(startup_transport, dict):
        raise ValueError("latest failed export startup capture is malformed")
    systemd = startup.get("systemd")
    if (
        startup.get("schema") != "fg19.first-cell-export-bounded-status.v1"
        or startup.get("unit") != expected_unit
        or startup.get("invocation_id") != expected_invocation
        or startup.get("terminal_metadata") != {}
        or startup.get("array_payloads_read_or_hashed") is not False
        or not isinstance(startup.get("parent_cgroup"), dict)
        or startup["parent_cgroup"].get("populated") != {}
        or not isinstance(systemd, dict)
        or (
            systemd.get("Id"),
            systemd.get("InvocationID"),
            systemd.get("ActiveState"),
            systemd.get("SubState"),
            systemd.get("Result"),
            systemd.get("ExecMainCode"),
            systemd.get("ExecMainStatus"),
            systemd.get("MainPID"),
        )
        != (expected_unit, expected_invocation, "failed", "failed", "exit-code", "1", "1", "0")
    ):
        raise ValueError("latest failed export observation identity differs")
    if not isinstance(observation, dict) or (
        observation.get("schema"),
        observation.get("unit"),
        observation.get("invocation_id"),
        observation.get("array_payloads_read_or_hashed"),
        observation.get("terminal_metadata_files"),
        observation.get("systemd"),
    ) != (
        "fg19.first-cell-export-bounded-status.v1",
        expected_unit,
        expected_invocation,
        False,
        [],
        systemd,
    ):
        raise ValueError("latest failed export normalized observation differs")
    if (
        startup_transport.get("schema"),
        startup_transport.get("ssh_returncode"),
        startup_transport.get("timed_out"),
        startup_transport.get("stdout_bytes"),
        startup_transport.get("stdout_sha256"),
        startup_transport.get("stderr_bytes"),
        startup_transport.get("persisted_before_parse"),
        startup_transport.get("retry_performed"),
    ) != (
        "fg19.first-cell-export-status-root-transport.v1",
        0,
        False,
        pin_rows["startup_observation"]["bytes"],
        pin_rows["startup_observation"]["sha256"],
        0,
        True,
        False,
    ):
        raise ValueError("latest failed export capture transport differs")

    if not isinstance(failure_review, dict) or (
        failure_review.get("schema"),
        failure_review.get("status"),
    ) != (
        "fg19.first-cell-export-startup-failure-metadata-review.v2",
        "PASS_METADATA_ONLY_ACTUAL_CHECKPOINT_RECIPE_FAILURE_CONFIRMED_NO_QUALITY",
    ):
        raise ValueError("latest failed export metadata review differs")
    captured = failure_review.get("captured_native_observation")
    execution = failure_review.get("execution_state")
    failure_classification = failure_review.get("failure_classification")
    if (
        not isinstance(captured, dict)
        or not isinstance(execution, dict)
        or not isinstance(failure_classification, dict)
    ):
        raise ValueError("latest failed export metadata claims are malformed")
    failure = failure_classification.get("execution_order")
    if (
        captured.get("unit") != expected_unit
        or captured.get("invocation_id") != expected_invocation
        or captured.get("strict_success_tuple_matched") is not False
        or captured.get("terminal_metadata") != {}
        or execution.get("retry_performed") is not False
        or execution.get("stop_or_restart_performed_by_reviewer") is not False
        or execution.get("remaining_four_checkpoint_loads_reached") is not False
        or execution.get("prediction_arrays_or_sidecars_observed") is not False
        or execution.get("complete_export_manifest_or_acceptance") is not False
        or execution.get("score_or_quality_result") is not False
        or execution.get("new_fit_or_training") is not False
        or not isinstance(failure, dict)
        or failure.get("failure_location")
        != "export_first_cell_v3.py:_validate_checkpoint, required argument loop; captured worker line 985"
        or failure.get("failure_message")
        != "ValueError: checkpoint recipe argument differs: input_lead"
        or b"ValueError: checkpoint recipe argument differs: input_lead" not in unit_log_tail
    ):
        raise ValueError("latest failed export classification or no-quality boundary differs")

    return {
        "classification": "SECOND_EXPORT_INVOCATION_FAILED_AT_CONTROL_CHECKPOINT_ARGUMENT_GATE",
        "historical_only": True,
        "root_only_not_staged_to_B": True,
        "old_unit_must_not_be_queried_stopped_or_retried": True,
        "unit": expected_unit,
        "invocation_id": expected_invocation,
        "failure_message": "ValueError: checkpoint recipe argument differs: input_lead",
        "files": pin_rows,
    }


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

    previous_export_failure = bind_previous_export_failure()
    latest_export_failure = bind_latest_failed_export()

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

    producer_review_pins = checked_producer_review_pins(export_review)
    reviewed_files = reviewed_file_map(export_review)

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
        ("export_review_contract_fixture", FIXTURE_SOURCE),
        ("historical_failure_fixture", HISTORY_FIXTURE_SOURCE),
        ("inverse_proof_script", INVERSE_PROOF_SOURCE),
    ):
        raw = path.read_bytes()
        local_sources[name] = {
            "path": str(path.relative_to(REPOSITORY)),
            "bytes": len(raw),
            "sha256": sha256(raw),
        }
    binding: dict[str, object] = {
        "schema": "fg19.first-cell-export-admission-source-binding.v6",
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
        "supersedes_previous_actual_export_failure": previous_export_failure,
        "supersedes_latest_actual_export_failure": latest_export_failure,
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
            "reviewed_file_count": len(reviewed_files),
            "producer_source": producer_review_pins["export_first_cell_v4.py"],
            "producer_binding": producer_review_pins["EXPORT_BINDING_V4.json"],
            "source_manifest": producer_review_pins["EXPORT_SOURCE_MANIFEST.json"],
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
            "export_first_cell_v4.py",
            "EXPORT_BINDING_V4.json",
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
