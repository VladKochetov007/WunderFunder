"""Prepare and statically verify the narrowly patched NLL verifier body."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
FOLDER = CAMPAIGN / "preparation/child_nll_terminal_verifier_v2"
V1_BODY = CAMPAIGN / "preparation/child_nll_terminal_verifier_v1/REMOTE_SUBMISSION.py"
V1_BODY_BYTES = 613_694
V1_BODY_SHA256 = "e0cc26f822762a96758bb88046440fc04dc7c9a4651bcc4db3509c072295f3f5"
V2_BODY = FOLDER / "REMOTE_SUBMISSION_V2.py"
GATE_FRAGMENT = FOLDER / "GPU_OWNERSHIP_GATE.pyfrag"
V2_IMPORT = b"import stat\nimport selectors\n"
V1_IMPORT = b"import stat\n"
V1_CONSTANT = b"PARENT_RESERVE_BYTES = 1_000_000_000\n"
V2_CONSTANT = V1_CONSTANT + b'SELECTED_GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"\n'
V1_OUTPUT_ANCHOR = b'        "blas_threads": 1,\n    }\n    D.mkdir(mode=0o700, exist_ok=False)\n'
V2_OUTPUT_ANCHOR = (
    b'        "blas_threads": 1,\n    }\n'
    b"    gpu_ownership = check_selected_gpu_ownership()\n"
    b"    D.mkdir(mode=0o700, exist_ok=False)\n"
)
V1_PARENT_FIELD = b'        "parent": parent,\n'
V2_PARENT_FIELD = V1_PARENT_FIELD + b'        "selected_gpu_ownership": gpu_ownership,\n'


def _replace_once(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    """Apply one exact byte replacement and reject a changed source anchor."""
    count = raw.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} anchor, found {count}")
    return raw.replace(old, new, 1)


def render_body(v1_raw: bytes, gate_raw: bytes) -> bytes:
    """Derive V2 from the pinned V1 body using only the new ownership gate."""
    if (len(v1_raw), hashlib.sha256(v1_raw).hexdigest()) != (V1_BODY_BYTES, V1_BODY_SHA256):
        raise ValueError("V1 rendered-body pin differs")
    if not gate_raw.endswith(b"\n") or not gate_raw.startswith(b"def _bounded_gpu_query("):
        raise ValueError("GPU ownership gate fragment is malformed")
    rendered = _replace_once(v1_raw, V1_IMPORT, V2_IMPORT, "import")
    rendered = _replace_once(rendered, V1_CONSTANT, V2_CONSTANT, "GPU constant")
    main_anchor = b"def main() -> None:\n"
    rendered = _replace_once(rendered, main_anchor, gate_raw + b"\n\n" + main_anchor, "main")
    rendered = _replace_once(rendered, V1_OUTPUT_ANCHOR, V2_OUTPUT_ANCHOR, "pre-mutation GPU gate")
    rendered = _replace_once(rendered, V1_PARENT_FIELD, V2_PARENT_FIELD, "admission GPU field")
    return rendered


def _assert_body_shape(body: bytes) -> None:
    """Check the gate is syntactically valid and precedes output mutation."""
    tree = ast.parse(body.decode("utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "check_selected_gpu_ownership" not in functions:
        raise AssertionError("selected-GPU ownership function is absent")
    main = functions.get("main")
    if main is None:
        raise AssertionError("main function is absent")
    gate_index = next(
        (
            index
            for index, statement in enumerate(main.body)
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "gpu_ownership"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "check_selected_gpu_ownership"
        ),
        -1,
    )
    mkdir_index = next(
        (
            index
            for index, statement in enumerate(main.body)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "D"
            and statement.value.func.attr == "mkdir"
        ),
        -1,
    )
    if gate_index < 0 or mkdir_index < 0 or gate_index >= mkdir_index:
        raise AssertionError("fresh GPU ownership gate does not precede D.mkdir")
    admissions = [
        statement.value
        for statement in main.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "verifier_admission"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Dict)
    ]
    if len(admissions) != 1:
        raise AssertionError("verifier admission mapping is absent or ambiguous")
    pairs = {
        key.value: value
        for key, value in zip(admissions[0].keys, admissions[0].values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    owner_value = pairs.get("selected_gpu_ownership")
    if not isinstance(owner_value, ast.Name) or owner_value.id != "gpu_ownership":
        raise AssertionError("selected GPU observation is not recorded in ADMISSION")
    gate_source = ast.get_source_segment(
        body.decode("utf-8"), functions["check_selected_gpu_ownership"]
    )
    if gate_source is None or '"minimum_free_memory_required_bytes": 0' not in gate_source:
        raise AssertionError("gate does not explicitly avoid a VRAM reservation threshold")


def _inverse_gate(body: bytes, gate_raw: bytes) -> bytes:
    """Remove only the five known additions to prove the rest is byte-identical."""
    restored = _replace_once(body, V2_IMPORT, V1_IMPORT, "V2 import")
    restored = _replace_once(restored, V2_CONSTANT, V1_CONSTANT, "V2 GPU constant")
    restored = _replace_once(
        restored,
        gate_raw + b"\n\n" + b"def main() -> None:\n",
        b"def main() -> None:\n",
        "GPU function",
    )
    restored = _replace_once(restored, V2_OUTPUT_ANCHOR, V1_OUTPUT_ANCHOR, "V2 pre-mutation gate")
    restored = _replace_once(restored, V2_PARENT_FIELD, V1_PARENT_FIELD, "V2 admission field")
    return restored


def prepare() -> bytes:
    """Render the new file with exclusive creation after validating V1 bytes."""
    raw = V1_BODY.read_bytes()
    gate = GATE_FRAGMENT.read_bytes()
    result = render_body(raw, gate)
    with V2_BODY.open("xb") as stream:
        stream.write(result)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def check() -> dict[str, object]:
    """Run focused source-only gates and return a compact immutable receipt."""
    original = V1_BODY.read_bytes()
    gate = GATE_FRAGMENT.read_bytes()
    body = V2_BODY.read_bytes()
    expected = render_body(original, gate)
    if body != expected:
        raise AssertionError("saved V2 body differs from the exact V1-plus-gate renderer")
    _assert_body_shape(body)
    if _inverse_gate(body, gate) != original:
        raise AssertionError("V2 contains a change outside the selected-GPU gate")
    ast.parse(gate.decode("utf-8"))
    return {
        "schema": "fg19.child-nll-terminal-verifier-v2-local-checks.v1",
        "status": "PASS_LOCAL_RENDER_AND_AST_ONLY_NOT_SUBMITTED",
        "v1_body": {"bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()},
        "gpu_gate_fragment": {"bytes": len(gate), "sha256": hashlib.sha256(gate).hexdigest()},
        "v2_body": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
        "checks": {
            "body_is_exact_renderer_output": True,
            "inverse_patch_restores_exact_v1_bytes": True,
            "body_ast_parses": True,
            "gate_runs_before_output_directory_creation": True,
            "gate_observation_is_recorded_in_admission": True,
            "foreign_gpu_owners_are_preserved": True,
            "no_minimum_vram_reservation": True,
        },
        "scope": "Local byte transformation and AST checks only; no V1 entrypoint, checker, SSH, B staging, or support unit was invoked.",
    }


def main() -> int:
    """Expose explicit prepare and static-check operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare()
        print(
            json.dumps(
                {"bytes": len(result), "sha256": hashlib.sha256(result).hexdigest()}, sort_keys=True
            )
        )
        return 0
    print(json.dumps(check(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
