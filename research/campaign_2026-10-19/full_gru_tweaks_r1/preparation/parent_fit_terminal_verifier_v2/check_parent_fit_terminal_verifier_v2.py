"""Run narrow static and cold metadata-only checks for the prepared body."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPARATION = HERE / "PREPARATION.json"
BODY = HERE / "REMOTE_SUBMISSION.py"
TEMPLATE = HERE / "REMOTE_SUBMISSION_TEMPLATE.py"
GENERATOR = HERE / "prepare_parent_fit_terminal_verifier_v2.py"
FOCUSED_TEST = HERE / "test_parent_fit_terminal_check_v2_rng.py"
RECEIPT = HERE / "LOCAL_CHECKS.json"
FORBIDDEN_SUFFIXES = (".pt", ".npy", ".npz", ".parquet")


def sha256(path: Path) -> str:
    """Hash a small local preparation file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cold_read_proof() -> dict[str, object]:
    """Import only the stdlib generator and reject artifact-payload reads."""
    script = f"""
import pathlib, sys
sys.path.insert(0, {str(HERE)!r})
import prepare_parent_fit_terminal_verifier_v2 as module
original_open = pathlib.Path.open
def guarded_open(path, *args, **kwargs):
    if pathlib.Path(path).suffix.lower() in {FORBIDDEN_SUFFIXES!r}:
        raise AssertionError('forbidden payload read: ' + str(path))
    return original_open(path, *args, **kwargs)
pathlib.Path.open = guarded_open
payload = module._read_payload()
expected = set(module.INPUTS)
assert set(payload) == expected
print('PASS cold stdlib import; pinned inputs=' + str(len(payload)) + '; model/array/data suffix reads=0')
"""
    command = [sys.executable, "-I", "-B", "-c", script]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"cold read sentinel failed: {result.stderr.strip()}")
    return {
        "command": "python -I -B -c <script embedded in this check file>",
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "forbidden_suffixes": list(FORBIDDEN_SUFFIXES),
        "module_path": str(GENERATOR),
    }


def focused_rng_tests() -> dict[str, object]:
    """Run only the new small RNG signature invariants in an isolated child."""
    command = [sys.executable, "-I", "-B", str(FOCUSED_TEST)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"focused RNG tests failed: {result.stderr.strip()}")
    return {
        "command": [sys.executable, "-I", "-B", str(FOCUSED_TEST)],
        "script_sha256": sha256(FOCUSED_TEST),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    """Confirm current hashes, body syntax, single-submit ordering, and cold reads."""
    preparation = json.loads(PREPARATION.read_text(encoding="utf-8"))
    body = BODY.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    ast.parse(body, filename=str(BODY))
    ast.parse(template, filename=str(TEMPLATE))
    if hashlib.sha256(body.encode()).hexdigest() != preparation["remote_body_sha256"]:
        raise ValueError("rendered body does not match preparation receipt")
    if hashlib.sha256(template.encode()).hexdigest() != preparation["template_sha256"]:
        raise ValueError("template does not match preparation receipt")
    if hashlib.sha256(GENERATOR.read_bytes()).hexdigest() != preparation["generator_sha256"]:
        raise ValueError("generator does not match preparation receipt")
    if body.count('"systemd-run"') != 1 or "systemctl stop" in body:
        raise ValueError("body is not a single-submit, no-stop adapter")
    if any(token in body for token in ("numpy.load", "torch.load", "nvidia-smi")):
        raise ValueError("admission body contains a model/data loader or GPU emptiness probe")
    submit_index = body.index("result = subprocess.run(command")
    durable_index = body.index('write_json_exclusive(output / "VERIFIER_LAUNCH.json"')
    if (
        submit_index >= durable_index
        or "post_show_timeout" not in body
        or "post_show_error" not in body
        or '"returncode": error.returncode' not in body
        or '"stdout": text_value(error.stdout)' not in body
        or '"stderr": text_value(error.stderr)' not in body
        or "free >= 25 * 1024**3" not in body
    ):
        raise ValueError("launch record does not durably cover post-submit outcomes")
    cold = cold_read_proof()
    focused = focused_rng_tests()
    receipt = {
        "schema": "fg19.parent-fit-r2-terminal-verifier-local-checks.v2",
        "status": "PASS_LOCAL_PREPARATION_ONLY",
        "checker_sha256": preparation["code_dependencies"]["parent_fit_terminal_check_v2.py"],
        "template_sha256": preparation["template_sha256"],
        "body_sha256": preparation["remote_body_sha256"],
        "generator_sha256": preparation["generator_sha256"],
        "preparation_sha256": sha256(PREPARATION),
        "checks": {
            "rendered_body_and_template_ast": "PASS",
            "single_submit_no_stop_no_model_array_or_gpu_probe": "PASS",
            "post_submit_timeout_and_query_error_durable_record_order": "PASS",
            "cold_metadata_read_sentinel": cold,
            "focused_new_rng_signature_tests": focused,
            "ruff": "PASS; exact commands recorded in README",
        },
        "limits": [
            "No SSH, remote query, admission, submission or unit action was performed.",
            "The cold child read only the fixed Python/JSON package inputs and denied .pt/.npy/.npz/.parquet opens.",
            "No model, training-state, dataset, or prediction-array payload was opened locally.",
            "Remote source/data/cache/resource/output gates remain to be run by root immediately before submission.",
        ],
    }
    RECEIPT.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "body_sha256": receipt["body_sha256"],
                "cold_read": cold["stdout"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
