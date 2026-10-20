"""Run bounded local checks for the one-shot child all-three profile adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
PROBE_PATH = PREPARATION / "child_allthree_profile_probe_v1.py"
CONTRACT_PATH = PREPARATION / "child_allthree_profile_contract_v1.py"
HELPER_PATH = PREPARATION / "propensity_terminal_support.py"
BODY = PACKET / "REMOTE_SUBMISSION.py"
TEMPLATE = PACKET / "REMOTE_SUBMISSION_TEMPLATE.py"
PROOF = PACKET / "private_d_closure_proof.py"
RECEIPT = PACKET / "LOCAL_CHECKS.json"
HELPER_SHA256 = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_adapter():
    spec = importlib.util.spec_from_file_location("child_allthree_profile_adapter", BODY)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the stdlib-only adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_rejected(callable_, message: str) -> None:
    try:
        callable_()
    except RuntimeError:
        return
    raise AssertionError(message)


def main() -> int:
    if BODY.read_bytes() != TEMPLATE.read_bytes():
        raise AssertionError("rendered body differs from its reviewed template")
    adapter = _load_adapter()
    staged, pins = adapter._decode_payloads()
    expected = {
        "DICTIONARY_ACCEPTANCE.json",
        "PARENT_EXACT_STOP.json",
        "PARENT_TERMINAL_ACCEPTANCE.json",
        "child_allthree_profile_contract_v1.py",
        "child_allthree_profile_probe_v1.py",
        "propensity_terminal_support.py",
    }
    if set(staged) != expected or set(pins) != expected:
        raise AssertionError("staged private-D payload closure is incomplete")
    if pins["propensity_terminal_support.py"] != HELPER_SHA256:
        raise AssertionError("restricted-loader helper SHA is not pinned")
    argv = adapter._trainer_argv()
    if (
        "--inherit-init-preprocessing" not in argv
        or "--profile-steps" not in argv
        or argv[argv.index("--profile-steps") + 1] != "4"
        or "--profile-warmup" not in argv
        or argv[argv.index("--profile-warmup") + 1] != "2"
        or "--max-batches" not in argv
        or argv[argv.index("--max-batches") + 1] != "6"
        or any(
            flag in argv
            for flag in ("--student-frontier-preflight-only", "--calibrate-distill-reference")
        )
    ):
        raise AssertionError("fixed profile argv does not bind the reviewed 60-update profile")

    # Exercise the pre-staging parent checkpoint guard only with a tiny synthetic file.
    with tempfile.TemporaryDirectory(prefix="child-profile-parent-fixture-") as directory:
        root = Path(directory).resolve()
        checkpoint = root / "runs" / "model_ep3.pt"
        checkpoint.parent.mkdir()
        raw = b"synthetic checkpoint identity fixture"
        checkpoint.write_bytes(raw)
        adapter.B = root
        adapter.PARENT_CHECKPOINT = checkpoint
        adapter.PARENT_EP3_BYTES = len(raw)
        adapter.PARENT_EP3_SHA256 = _sha(raw)
        identity = adapter._parent_checkpoint_identity()
        if identity["sha256"] != _sha(raw) or identity["bytes"] != len(raw):
            raise AssertionError("synthetic checkpoint identity did not bind")
        checkpoint.write_bytes(raw + b"changed")
        _assert_rejected(
            adapter._parent_checkpoint_identity,
            "changed synthetic checkpoint was not rejected",
        )
        checkpoint.unlink()
        external = root.parent / f"{root.name}-outside.pt"
        external.write_bytes(raw)
        checkpoint.symlink_to(external)
        _assert_rejected(
            adapter._parent_checkpoint_identity,
            "symlinked synthetic checkpoint was not rejected",
        )
        external.unlink()

    # Import the staged files from an isolated temporary D; no model or array reader runs.
    with tempfile.TemporaryDirectory(prefix="child-profile-import-") as directory:
        workspace = Path(directory).resolve()
        private_dir = workspace / "private-D"
        source_src = workspace / "source" / "src"
        private_dir.mkdir()
        source_src.mkdir(parents=True)
        for name in (
            "child_allthree_profile_probe_v1.py",
            "child_allthree_profile_contract_v1.py",
            "propensity_terminal_support.py",
        ):
            (private_dir / name).write_bytes(staged[name])
        proof_copy = private_dir / PROOF.name
        proof_copy.write_bytes(PROOF.read_bytes())
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(source_src),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        completed = subprocess.run(
            [sys.executable, "-S", str(proof_copy)],
            cwd=workspace,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        import_proof = json.loads(completed.stdout)
        for entry in import_proof["modules"].values():
            if Path(entry["path"]).parent != private_dir:
                raise AssertionError("profile import escaped private-D closure")

    receipt = {
        "schema": "child-allthree-profile-local-checks.v1",
        "status": "PASS",
        "body_sha256": _sha(BODY.read_bytes()),
        "template_sha256": _sha(TEMPLATE.read_bytes()),
        "check_script_sha256": _sha(Path(__file__).read_bytes()),
        "private_d_proof_script_sha256": _sha(PROOF.read_bytes()),
        "probe_sha256": _sha(PROBE_PATH.read_bytes()),
        "contract_sha256": _sha(CONTRACT_PATH.read_bytes()),
        "helper_sha256": _sha(HELPER_PATH.read_bytes()),
        "focused_probe_test_sha256": _sha(
            (PREPARATION / "test_child_allthree_profile_probe_v1.py").read_bytes()
        ),
        "staged_payload_sha256": pins,
        "parent_checkpoint_guard": "PASS synthetic exact bytes; rejects mutation and symlink",
        "profile_argv": "PASS fixed 60-update profile and inherited parent normalization",
        "private_d_cold_import": import_proof,
        "commands": [
            ".venv/bin/ruff check <REMOTE_SUBMISSION.py> <REMOTE_SUBMISSION_TEMPLATE.py> check_local.py private_d_closure_proof.py",
            "python3 -m py_compile <REMOTE_SUBMISSION.py> <REMOTE_SUBMISSION_TEMPLATE.py> check_local.py private_d_closure_proof.py",
            "python3 check_local.py",
        ],
        "remote_actions": False,
        "model_array_or_dataset_payloads_read": False,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
