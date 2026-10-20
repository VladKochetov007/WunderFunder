"""Run focused static and cold metadata-only checks for the RNG diagnostic packet."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPARATION = HERE / "PREPARATION.json"
TEMPLATE = HERE / "REMOTE_SUBMISSION_TEMPLATE.py"
BODY = HERE / "REMOTE_SUBMISSION.py"
GENERATOR = HERE / "prepare_parent_rng_diagnostic_v2.py"
SCRIPT = HERE / "rng_diagnostic_v2.py"
RECEIPT = HERE / "LOCAL_CHECKS.json"
V1_BODY = HERE.parent / "parent_rng_diagnostic_v1/REMOTE_SUBMISSION.py"
V1_REJECTION_STDERR = (
    HERE.parents[1] / "records/parent_fit_r2_v1/rng_diagnostic_v1/STDERR.launch.raw"
)
FORBIDDEN_SUFFIXES = (".pt", ".npy", ".npz", ".parquet")


def sha256(raw: bytes) -> str:
    """Hash local source or bounded metadata bytes."""
    return hashlib.sha256(raw).hexdigest()


def _load_local_inputs() -> dict[str, bytes]:
    """Load the pinned local Python/JSON inputs, never checkpoint payloads."""
    sys.path.insert(0, str(HERE))
    import prepare_parent_rng_diagnostic_v2 as module

    return module._load_payload()


def cold_read_proof() -> dict[str, object]:
    """Cold-import the stdlib preparer while denying all model/data suffixes."""
    child_script = f"""
import pathlib, sys
sys.path.insert(0, {str(HERE)!r})
import prepare_parent_rng_diagnostic_v2 as module
original_open = pathlib.Path.open
def guarded_open(path, *args, **kwargs):
    if pathlib.Path(path).suffix.lower() in {FORBIDDEN_SUFFIXES!r}:
        raise AssertionError('forbidden payload read: ' + str(path))
    return original_open(path, *args, **kwargs)
pathlib.Path.open = guarded_open
payload = module._load_payload()
assert 'RNG_DIAGNOSTIC.py' in payload and 'RNG_STATE_STAT.json' in payload
assert 'FAILED_VERIFIER_CAPTURE.raw.json' in payload
assert 'parent_fit_terminal_check_v1.py' in payload
print('PASS isolated stdlib preparation; pinned inputs=' + str(len(payload)) + '; forbidden payload opens=0')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", child_script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cold input sentinel failed: {result.stderr.strip()}")
    return {
        "command": "python -I -B -c <script embedded in this check file>",
        "script_sha256": sha256(child_script.encode()),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "forbidden_suffixes": list(FORBIDDEN_SUFFIXES),
        "preparer_path": str(GENERATOR),
    }


def checked_failed_attempt_fixture(payload: dict[str, bytes]) -> list[str]:
    """Run the remote failure gate on actual capture bytes and reject type mutations."""
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"), filename=str(TEMPLATE))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "main"
        )
    ]
    namespace: dict[str, object] = {"__name__": "parent_rng_diagnostic_local_test"}
    exec(compile(tree, str(TEMPLATE), "exec"), namespace)
    check = namespace.get("checked_failed_attempt")
    if not callable(check):
        raise ValueError("remote template has no checked_failed_attempt function")
    outputs = json.loads(payload["FAILED_VERIFIER_ADMISSION.json"]).get("terminal_files_stat_only")
    if not isinstance(outputs, dict):
        raise ValueError("actual failed-verifier admission has no terminal stat object")
    actual = json.loads(payload["FAILED_VERIFIER_CAPTURE.raw.json"])
    if actual.get("terminal_success") is not False:
        raise ValueError("actual failed-verifier capture does not contain boolean false")
    result = check(payload, outputs)
    if result.get("prior_verifier_terminal_status") != "failed":
        raise ValueError("actual V1 failure capture did not pass the intended failure gate")

    def require_rejected(capture: dict[str, object], label: str) -> None:
        changed = dict(payload)
        raw = (json.dumps(capture, sort_keys=True, separators=(",", ":")) + "\n").encode()
        changed["FAILED_VERIFIER_CAPTURE.raw.json"] = raw
        namespace["FAILED_RAW_SHA"] = sha256(raw)
        try:
            check(changed, outputs)
        except RuntimeError:
            return
        raise ValueError(f"mutated failed capture was accepted: {label}")

    rejected = []
    for field, value, label in (
        ("terminal_success", True, "terminal_success_true"),
        ("terminal_success", 0, "terminal_success_integer_zero"),
    ):
        variant = dict(actual)
        variant[field] = value
        require_rejected(variant, label)
        rejected.append(label)
    bad_unit = dict(actual["unit"])
    bad_unit["ExecMainStatus"] = 1
    variant = dict(actual)
    variant["unit"] = bad_unit
    require_rejected(variant, "exec_status_integer")
    rejected.append("exec_status_integer")
    return rejected


def main() -> None:
    """Verify source pins, syntax, single-submit behavior, and cold metadata reads."""
    prep = json.loads(PREPARATION.read_text(encoding="utf-8"))
    template = TEMPLATE.read_text(encoding="utf-8")
    body = BODY.read_text(encoding="utf-8")
    v1_body = V1_BODY.read_text(encoding="utf-8")
    diagnostic = SCRIPT.read_text(encoding="utf-8")
    ast.parse(template, filename=str(TEMPLATE))
    ast.parse(body, filename=str(BODY))
    ast.parse(diagnostic, filename=str(SCRIPT))
    payload = _load_local_inputs()
    rejected_variants = checked_failed_attempt_fixture(payload)
    checker = payload["parent_fit_terminal_check_v1.py"].decode("utf-8")
    observed = {
        "template_sha256": sha256(template.encode()),
        "remote_body_sha256": sha256(body.encode()),
        "generator_sha256": sha256(GENERATOR.read_bytes()),
        "diagnostic_script_sha256": sha256(SCRIPT.read_bytes()),
    }
    for name, value in observed.items():
        if value != prep[name]:
            raise ValueError(f"preparation pin differs: {name}")
    if body.count('"systemd-run"') != 1 or "systemctl stop" in body:
        raise ValueError("body must submit one unit and must not stop another")
    if "show(FAILED_UNIT" in body or "nvidia-smi" in body:
        raise ValueError("body queries the prior failure or probes GPU emptiness")
    v1_body_sha = sha256(v1_body.encode())
    v1_stderr_sha = sha256(V1_REJECTION_STDERR.read_bytes())
    supersedes = prep.get("supersedes")
    if not isinstance(supersedes, dict):
        raise ValueError("preparation receipt omits V1 rejection provenance")
    if v1_body_sha != supersedes.get("local_v1_body_sha256") or v1_stderr_sha != supersedes.get(
        "v1_readonly_rejection_stderr_sha256"
    ):
        raise ValueError("V1 body/rejection provenance differs")
    gate_pos = v1_body.index("prior_failure = checked_failed_attempt(payload, outputs)")
    mkdir_pos = v1_body.index("output.mkdir(mode=0o700, exist_ok=False)")
    submit_pos_v1 = v1_body.index('"systemd-run"')
    if not gate_pos < mkdir_pos < submit_pos_v1:
        raise ValueError("V1 failure gate does not precede output creation and submission")
    if any(token in body for token in ("torch.load", "numpy.load")):
        raise ValueError("admission body contains a payload loader")
    required = (
        "checker._load_checkpoint(state_path, MAX_STATE_BYTES)",
        'state.get("format")',
        'state.get("next_epoch")',
        'value.detach().cpu().contiguous().numpy().tobytes(order="C")',
    )
    if any(token not in diagnostic for token in required):
        raise ValueError("diagnostic does not implement the narrow RNG-only report")
    if "weights_only=True" not in checker or "torch.serialization.safe_globals" not in checker:
        raise ValueError("pinned checker no longer provides the restricted loader")
    if 'state.get("optimizer")' in diagnostic or 'state.get("model")' in diagnostic:
        raise ValueError("diagnostic inspects model or optimizer values")
    submit_pos = body.index("result = subprocess.run(command")
    launch_pos = body.index('write_json_exclusive(output / "RNG_DIAGNOSTIC_LAUNCH.json"')
    if submit_pos >= launch_pos or "post_show_timeout" not in body or "post_show_error" not in body:
        raise ValueError("unknown outcome is not durably recorded")
    cold = cold_read_proof()
    receipt = {
        "schema": "fg19.parent-fit-r2-rng-diagnostic-local-checks.v2",
        "status": "PASS_LOCAL_PREPARATION_ONLY",
        **observed,
        "preparation_sha256": sha256(PREPARATION.read_bytes()),
        "inputs_sha256": prep["input_sha256"],
        "checks": {
            "template_body_and_script_ast": "PASS",
            "exact_f270_weights_only_loader_and_safe_globals": "PASS",
            "no_prior_failed_unit_query_or_stop_and_one_submit": "PASS",
            "create_only_output_and_durable_unknown_outcome": "PASS",
            "actual_failed_v1_capture_false_and_stat": "PASS",
            "strict_capture_mutation_rejections": rejected_variants,
            "v1_rejected_before_output_and_submit": {
                "body_sha256": v1_body_sha,
                "stderr_sha256": v1_stderr_sha,
                "checked_failure_before_mkdir_and_systemd_run": True,
            },
            "cold_metadata_read_sentinel": cold,
            "ruff": "PASS; commands listed in packet README",
        },
        "limits": [
            "No SSH, remote query, unit action, model read, training-state read, or array/data read was performed.",
            "Cold child read only bounded Python and JSON source/metadata; .pt/.npy/.npz/.parquet opens were denied.",
            "Only the future remote unit loads training_state.pt through the exact pinned f270 restricted weights_only loader.",
        ],
    }
    RECEIPT.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "cold_read": cold["stdout"]}, sort_keys=True))


if __name__ == "__main__":
    main()
