"""Static contract proof for the conditional ROOT score-stop transport driver."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREPARATION = CAMPAIGN / "preparation/first_cell_evaluation_score_stop_v1"
DRIVER = PREPARATION / "ROOT_CAPTURE_SCORE_STOP_V1.py"
TEMPLATE = PREPARATION / "STOP_SCORE_TEMPLATE_V1.py"
REVIEW = PREPARATION / "SOURCE_REVIEW_V1.json"
PROOF = PREPARATION / "ROOT_CAPTURE_SCORE_STOP_V1_PROOF.json"
EXPORT_DRIVER = (
    CAMPAIGN / "records/first_cell_evaluation_v1/export_v5/exact_stop/ROOT_CAPTURE_STOP.py"
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pin(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.relative_to(REPOSITORY).as_posix(),
        "sha256": sha256(raw),
    }


def top_level_constants(tree: ast.Module) -> dict[str, object]:
    constants: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            with contextlib.suppress(ValueError, TypeError):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    return constants


def score_stop_payload_keys(template_tree: ast.Module) -> set[str]:
    payloads: list[ast.Dict] = []
    for node in ast.walk(template_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr == "dumps"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            payloads.append(node.args[0])
    if len(payloads) != 1:
        raise AssertionError("expected one JSON object emitted by the frozen stop template")
    keys = {ast.literal_eval(key) for key in payloads[0].keys}
    if any(not isinstance(key, str) for key in keys):
        raise AssertionError("stop template response keys must be literal strings")
    return keys


def driver_stop_keys(driver_tree: ast.Module) -> set[str]:
    for node in driver_tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "EXPECTED_STOP_KEYS"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
        ):
            values = ast.literal_eval(node.value.args[0])
            if isinstance(values, set) and all(isinstance(value, str) for value in values):
                return values
    raise AssertionError("driver exact stop key set is not an explicit literal frozenset")


def main() -> None:
    driver_raw = DRIVER.read_bytes()
    template_raw = TEMPLATE.read_bytes()
    driver_tree = ast.parse(driver_raw.decode("utf-8", errors="strict"))
    template_tree = ast.parse(template_raw.decode("utf-8", errors="strict"))
    driver_constants = top_level_constants(driver_tree)
    template_constants = top_level_constants(template_tree)

    expected_keys = score_stop_payload_keys(template_tree)
    if len(expected_keys) != 14:
        raise AssertionError(f"frozen score stop schema has {len(expected_keys)} keys, expected 14")
    if driver_stop_keys(driver_tree) != expected_keys:
        raise AssertionError("driver exact key set differs from frozen score stop template")
    for name in ("ACCEPTANCE_SCHEMA", "ACCEPTANCE_STATUS", "RESULT_SCHEMA", "RESULT_STATUS"):
        if driver_constants.get(name) != template_constants.get(name):
            raise AssertionError(f"driver acceptance/result contract differs: {name}")
    if driver_constants.get("SCORE_UNIT") != template_constants.get("SCORE_UNIT"):
        raise AssertionError("driver unit differs from frozen score-stop template")
    if driver_constants.get("SCORE_INVOCATION") != "UNBOUND_ROOT_SCORE_INVOCATION":
        raise AssertionError("conditional invocation must remain explicitly unbound")
    for name in ("BODY_PIN", "REVIEW_PIN", "ACCEPTANCE_PIN"):
        value = driver_constants.get(name)
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and value[0] == 0
            and isinstance(value[1], str)
            and value[1].startswith("UNBOUND_ROOT_SCORE_")
        ):
            raise AssertionError(f"conditional {name} is unexpectedly bound")

    main_function = next(
        (
            node
            for node in driver_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
        None,
    )
    if main_function is None:
        raise AssertionError("driver main entrypoint is absent")
    calls = [node for node in ast.walk(main_function) if isinstance(node, ast.Call)]
    ssh_runs = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    if len(ssh_runs) != 1:
        raise AssertionError("driver must have exactly one subprocess.run attempt")
    run_call = ssh_runs[0]
    keywords = {keyword.arg: keyword.value for keyword in run_call.keywords}
    if (
        not isinstance(keywords.get("timeout"), ast.Constant)
        or keywords["timeout"].value != 60
        or not isinstance(keywords.get("input"), ast.Name)
        or keywords["input"].id != "body"
    ):
        raise AssertionError(
            "driver SSH call must use the pinned body and a single 60-second bound"
        )
    commands = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets)
        and isinstance(node.value, ast.List)
    ]
    if len(commands) != 1:
        raise AssertionError("driver SSH argv must be a single explicit literal list")
    argv = [ast.literal_eval(item) for item in commands[0].value.elts]
    if argv != [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "beleriand",
        "python3 -E -B -",
    ]:
        raise AssertionError("driver SSH command is not the one bounded, no-retry command")

    output_write_lines: dict[str, int] = {}
    validation_lines: list[int] = []
    for node in calls:
        if isinstance(node.func, ast.Name) and node.func.id == "_write_exclusive" and node.args:
            try:
                output_name = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            if output_name in {
                "STDOUT.raw.json",
                "STDERR.raw.txt",
                "TRANSPORT.json",
                "EXACT_STOP.json",
            }:
                output_write_lines[output_name] = node.lineno
        if isinstance(node.func, ast.Name) and node.func.id == "_validate_stop":
            validation_lines.append(node.lineno)
    if set(output_write_lines) != {
        "STDOUT.raw.json",
        "STDERR.raw.txt",
        "TRANSPORT.json",
        "EXACT_STOP.json",
    }:
        raise AssertionError("driver does not exclusively persist all four expected stop artifacts")
    if len(validation_lines) != 1:
        raise AssertionError("driver must validate exactly one captured stop response")
    if (
        max(
            output_write_lines[key]
            for key in ("STDOUT.raw.json", "STDERR.raw.txt", "TRANSPORT.json")
        )
        >= validation_lines[0]
    ):
        raise AssertionError("raw stdout/stderr/transport must be durable before response parsing")
    if output_write_lines["EXACT_STOP.json"] <= validation_lines[0]:
        raise AssertionError("exact stop receipt must be written only after validation")

    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if review.get("status") != "PASS_CONDITIONAL_SOURCE_ONLY_STOP_TEMPLATE_UNBOUND":
        raise AssertionError("frozen conditional score-stop source review status differs")
    result = {
        "schema": "fg19.first-cell-score-stop-root-driver-proof.v1",
        "status": "PASS_CONDITIONAL_ROOT_TRANSPORT_CONTRACT_ONLY_UNBOUND",
        "scope": "Static stdlib metadata/AST review only; driver not executed, no B/network/unit query, no model or array payload access.",
        "driver": pin(DRIVER),
        "frozen_score_stop_template": pin(TEMPLATE),
        "frozen_score_stop_source_review": pin(REVIEW),
        "reference_export_exact_stop_driver": {
            **pin(EXPORT_DRIVER),
            "read_as_source_only": True,
        },
        "stop_response_contract": {
            "source": "STOP_SCORE_TEMPLATE_V1.py emitted JSON object",
            "exact_key_count": 14,
            "keys": sorted(expected_keys),
            "result_sha256_joined_to_ROOT_acceptance": True,
            "before_exit": {
                "ActiveState": "active",
                "SubState": "exited",
                "MainPID": "0",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "Result": "success",
            },
            "after_state": {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": "0",
                "InvocationID": "",
            },
            "populated_snapshots": "both exact empty objects",
        },
        "transport": {
            "one_ssh_invocation": True,
            "ssh_connection_attempts": 1,
            "timeout_seconds": 60,
            "stdout_stderr_transport_fsynced_before_parse": True,
            "create_only_outputs": [
                "STDOUT.raw.json",
                "STDERR.raw.txt",
                "TRANSPORT.json",
                "EXACT_STOP.json",
            ],
            "timeout_or_unknown_outcome_retries": False,
        },
        "identity_literals": {
            "D": "planned records/first_cell_evaluation_v1/score_v1/exact_stop; remains conditional on durable score acceptance",
            "body_review_acceptance_pins": "explicitly unbound placeholders",
            "score_unit": driver_constants["SCORE_UNIT"],
            "score_invocation": "UNBOUND_ROOT_SCORE_INVOCATION",
            "root_score_acceptance": "must bind accepted native 0/1/0 exit and positive result byte/SHA/path pin before SSH",
        },
        "summary_key_count_correction": "The frozen source template emits 14 keys including result_sha256; the historical task shorthand of 13 is not treated as schema authority.",
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    PROOF.write_bytes(encoded)
    print(f"wrote {PROOF.relative_to(REPOSITORY)} bytes={len(encoded)} sha256={sha256(encoded)}")


if __name__ == "__main__":
    main()
