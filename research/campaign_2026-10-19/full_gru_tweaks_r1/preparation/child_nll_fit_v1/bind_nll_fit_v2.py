"""Versioned NLL binder adding exact original dictionary launch-acceptance lineage."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
V1_BINDER = HERE / "bind_nll_fit.py"
V1_TEMPLATE = HERE / "NLL_LAUNCH_TEMPLATE.py"
V2_TEMPLATE = HERE / "NLL_LAUNCH_TEMPLATE_V2.py"
V2_BODY = HERE / "REMOTE_SUBMISSION_V2.py"
V2_BINDING = HERE / "NLL_FINAL_BINDING_RECEIPT_V2.json"

V1_BINDER_BYTES = 40_287
V1_BINDER_SHA256 = "54ef01de5b5c7af4e77a37f40400320aaa1674445a9a4ad80a37b65de4ad9f41"
V1_TEMPLATE_BYTES = 203_526
V1_TEMPLATE_SHA256 = "7f75525a14527c9cc11e0b44da7969846120fe5231ed2d4da0f6169d82fd8dc3"
V2_TEMPLATE_BYTES = 203_735
V2_TEMPLATE_SHA256 = "366598622361efac4bb69f1145e5cb4f8349bb5c7ba56484738dff608563483b"
DICTIONARY_LAUNCH_ACCEPTANCE_SHA256 = (
    "33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95"
)
REMOTE_ANCHOR = (
    "    acceptance = json.loads(acceptance_raw)\n    exact_stop = json.loads(stop_raw)\n"
)
REMOTE_PATCH = (
    "    acceptance = json.loads(acceptance_raw)\n"
    '    if acceptance.get("launch_acceptance_sha256") != '
    "NLL_DICTIONARY_CHILD_LAUNCH_ACCEPTANCE_SHA256:\n"
    '        raise RuntimeError("dictionary child acceptance does not bind its exact original launch acceptance")\n'
    "    exact_stop = json.loads(stop_raw)\n"
)
LOCAL_ANCHOR = "    acceptance = json.loads(acceptance_raw)\n    stop = json.loads(stop_raw)\n"
LOCAL_PATCH = (
    "    acceptance = json.loads(acceptance_raw)\n"
    "    _require_exact_launch_acceptance(acceptance)\n"
    "    stop = json.loads(stop_raw)\n"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
        raise ValueError(f"frozen NLL V1 input size/type differs: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
        ):
            raise ValueError(f"frozen NLL V1 input changed before read: {path}")
        raw = bytearray()
        while len(raw) <= expected_bytes:
            chunk = os.read(descriptor, min(65_536, expected_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != expected_bytes or _sha256(raw) != expected_sha256:
        raise ValueError(f"frozen NLL V1 input SHA/length differs: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    ):
        raise ValueError(f"frozen NLL V1 input changed while reading: {path}")
    return bytes(raw)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def _require_exact_launch_acceptance(acceptance: object) -> str:
    if not isinstance(acceptance, dict):
        raise ValueError("dictionary child terminal acceptance must be a JSON object")
    actual = acceptance.get("launch_acceptance_sha256")
    if not isinstance(actual, str) or actual != DICTIONARY_LAUNCH_ACCEPTANCE_SHA256:
        raise ValueError(
            "dictionary child terminal acceptance does not bind the exact original launch acceptance"
        )
    return actual


def _load_v1_binder() -> ModuleType:
    _read_pinned(V1_BINDER, V1_BINDER_BYTES, V1_BINDER_SHA256)
    spec = importlib.util.spec_from_file_location("fg19_frozen_nll_binder_v1", V1_BINDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned local NLL V1 binder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _derive_template() -> bytes:
    raw = _read_pinned(V1_TEMPLATE, V1_TEMPLATE_BYTES, V1_TEMPLATE_SHA256)
    source = raw.decode("utf-8")
    source = _replace_once(source, REMOTE_ANCHOR, REMOTE_PATCH, "remote receipt validator")
    return source.encode("utf-8")


def _check_template() -> dict[str, object]:
    expected = _derive_template()
    actual = _read_pinned(V2_TEMPLATE, V2_TEMPLATE_BYTES, V2_TEMPLATE_SHA256)
    if actual != expected:
        raise ValueError("frozen NLL V2 template is not the exact V1-plus-gate derivation")
    if V2_BODY.exists() or V2_BINDING.exists():
        raise ValueError("NLL V2 rendered body/binding must remain absent before actual receipts")
    if REMOTE_PATCH.strip() not in actual.decode("utf-8"):
        raise ValueError("NLL V2 remote gate is absent")
    return {
        "status": "PASS_LOCAL_V2_UNBOUND_TEMPLATE_ONLY",
        "template": {"bytes": len(actual), "sha256": _sha256(actual)},
        "v1_template": {"bytes": V1_TEMPLATE_BYTES, "sha256": V1_TEMPLATE_SHA256},
        "required_acceptance_field": "launch_acceptance_sha256",
        "required_exact_value": DICTIONARY_LAUNCH_ACCEPTANCE_SHA256,
        "body_absent": True,
    }


def _validate_receipts(acceptance_raw: bytes, stop_raw: bytes) -> dict[str, object]:
    v1 = _load_v1_binder()
    acceptance_sha = _sha256(acceptance_raw)
    stop_sha = _sha256(stop_raw)
    acceptance = json.loads(acceptance_raw)
    exact_launch_sha = _require_exact_launch_acceptance(acceptance)
    v1._validate_child_receipts(acceptance_raw, stop_raw, acceptance_sha, stop_sha)
    return {
        "acceptance_sha256": acceptance_sha,
        "exact_stop_sha256": stop_sha,
        "launch_acceptance_sha256": exact_launch_sha,
    }


def _render_body(acceptance_raw: bytes, stop_raw: bytes) -> bytes:
    if len(acceptance_raw) > 64_000 or len(stop_raw) > 64_000:
        raise ValueError("actual dictionary predecessor receipt exceeds fixed bound")
    _validate_receipts(acceptance_raw, stop_raw)
    source = _check_template_bytes().decode("utf-8")
    v1 = _load_v1_binder()
    replacements = {
        f'NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256 = "{v1.UNBOUND_ACCEPTANCE}"': (
            f'NLL_DICTIONARY_CHILD_ACCEPTANCE_SHA256 = "{_sha256(acceptance_raw)}"'
        ),
        f'NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256 = "{v1.UNBOUND_EXACT_STOP}"': (
            f'NLL_DICTIONARY_CHILD_EXACT_STOP_SHA256 = "{_sha256(stop_raw)}"'
        ),
        f'NLL_DICTIONARY_CHILD_ACCEPTANCE_B64 = "{v1.ACCEPTANCE_B64_SENTINEL}"': (
            "NLL_DICTIONARY_CHILD_ACCEPTANCE_B64 = "
            f'"{base64.b64encode(acceptance_raw).decode("ascii")}"'
        ),
        f'NLL_DICTIONARY_CHILD_EXACT_STOP_B64 = "{v1.EXACT_STOP_B64_SENTINEL}"': (
            f'NLL_DICTIONARY_CHILD_EXACT_STOP_B64 = "{base64.b64encode(stop_raw).decode("ascii")}"'
        ),
    }
    for old, new in replacements.items():
        source = _replace_once(source, old, new, "V2 dictionary receipt slot")
    if v1.UNBOUND_ACCEPTANCE in source or v1.UNBOUND_EXACT_STOP in source:
        raise ValueError("rendered NLL V2 body retains an unbound marker")
    if REMOTE_PATCH.strip() not in source:
        raise ValueError("rendered NLL V2 body lost the exact launch-acceptance gate")
    return source.encode("utf-8")


def _check_template_bytes() -> bytes:
    return _read_pinned(V2_TEMPLATE, V2_TEMPLATE_BYTES, V2_TEMPLATE_SHA256)


def _write_create_only(path: Path, raw: bytes, mode: int | None = None) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    if mode is not None:
        path.chmod(mode)


def _render() -> dict[str, object]:
    _check_template()
    v1 = _load_v1_binder()
    acceptance_raw = v1._read_regular(
        v1.CHILD_ACCEPTANCE, 64_000, "actual dictionary child acceptance"
    )
    stop_raw = v1._read_regular(v1.CHILD_EXACT_STOP, 64_000, "actual dictionary child exact stop")
    receipt_check = _validate_receipts(acceptance_raw, stop_raw)
    body = _render_body(acceptance_raw, stop_raw)
    if V2_BODY.exists() or V2_BINDING.exists():
        raise FileExistsError("NLL V2 output already exists; refusing overwrite")
    _write_create_only(V2_BODY, body, 0o700)
    record = {
        "schema": "fg19-child-nll-fit-final-binding.v2",
        "status": "PASS_LOCAL_RENDERED_BODY_AWAITING_INDEPENDENT_SOURCE_REVIEW",
        "body": {"path": V2_BODY.name, "bytes": len(body), "sha256": _sha256(body)},
        "predecessor_receipts": receipt_check,
        "dictionary_child_is_sequential_predecessor_only": True,
        "initializer_remains": "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
        "no_b_action": True,
    }
    binding = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_create_only(V2_BINDING, binding)
    return {"status": record["status"], "body": record["body"], "binding_sha256": _sha256(binding)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-template", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.prepare_template:
        raw = _derive_template()
        _write_create_only(V2_TEMPLATE, raw)
        result = {
            "status": "CREATED_LOCAL_UNBOUND_TEMPLATE_ONLY",
            "bytes": len(raw),
            "sha256": _sha256(raw),
        }
    elif args.check:
        result = _check_template()
    else:
        result = _render()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
