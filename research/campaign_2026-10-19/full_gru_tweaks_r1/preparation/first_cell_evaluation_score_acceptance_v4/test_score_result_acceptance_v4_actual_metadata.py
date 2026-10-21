"""Exercise helper V4 on captured scalar metadata only; never open export arrays."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import stat
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
HELPER = C / "preparation/first_cell_evaluation_score_acceptance_v4/score_result_acceptance_v4.py"
INPUTS = {
    "result_raw": (
        "records/first_cell_evaluation_v1/score_v4/status_capture_v2/RESULT.json",
        33_852,
        "1646d8656a042ce49b355f22521134206d601c1dbac3697991fcd308aa1a94d5",
    ),
    "capture_raw": (
        "records/first_cell_evaluation_v1/score_v4/status_capture_v2/STDOUT.raw.json",
        148_379,
        "eac8de6c6e1dcc46e12331dec243bfbe9b4acf185537246edf0e783becac53b2",
    ),
    "score_admission_raw": (
        "records/first_cell_evaluation_v1/score_v4/ADMISSION.json",
        69_650,
        "ddb88a8c4584adbb76d6220285cf050a800324f5984eea49f0e7c748a35162d7",
    ),
    "score_launch_raw": (
        "records/first_cell_evaluation_v1/score_v4/LAUNCH.json",
        4_193,
        "848b0c6f055a093b2600c3043cb41204d6047565e79e199bc367c14f94f4d2e4",
    ),
    "export_acceptance_raw": (
        "records/first_cell_evaluation_v1/export_v5/EXPORT_ACCEPTANCE.json",
        8_833,
        "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d",
    ),
    "export_exact_stop_raw": (
        "records/first_cell_evaluation_v1/export_v5/exact_stop/EXACT_STOP.json",
        1_140,
        "695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f",
    ),
    "export_binding_raw": (
        "preparation/first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json",
        15_796,
        "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
    ),
    "export_manifest_raw": (
        "records/first_cell_evaluation_v1/export_v5/status_20261021T1107Z/EXPORT_MANIFEST_V1.json",
        23_130,
        "6b93cabf0e0b4450a11823682dfeba9396076efbc7b4a2033673fc46b7f04c23",
    ),
}
MAX_LOCAL_METADATA = 250_000
EXPECTED_SCORE_INVOCATION = "9bd64d6759e8462bb40985b06a9a0ad6"
EXPECTED_EXPORT_INVOCATION = "78fa25d0abc44d199e3696d172dbc4f9"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(relative: str, expected_bytes: int, expected_sha: str) -> bytes:
    path = C / relative
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_LOCAL_METADATA:
        raise ValueError(f"metadata fixture input is not a bounded regular file: {relative}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"metadata fixture input changed before read: {relative}")
        chunks: list[bytes] = []
        remaining = MAX_LOCAL_METADATA + 1
        while remaining:
            chunk = os.read(descriptor, min(32_768, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) != expected_bytes or _sha(raw) != expected_sha:
        raise ValueError(f"metadata fixture bytes differ from pinned captured input: {relative}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"metadata fixture input changed during read: {relative}")
    return raw


def _load_helper():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("score_result_acceptance_v4_fixture", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load helper V4 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs() -> dict[str, bytes]:
    return {name: _read_pinned(*pin) for name, pin in INPUTS.items()}


def _mutated_kwargs(helper: Any, raw_inputs: dict[str, bytes], mutation: str) -> dict[str, Any]:
    result = json.loads(raw_inputs["result_raw"])
    if mutation == "checkpoint_sha":
        result["fixed_models"][0]["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "extra_loaded_sha":
        checkpoint = result["fixed_models"][0]["checkpoint"]
        checkpoint["loaded_sha256"] = checkpoint["sha256"]
    elif mutation == "invocation":
        result["fixed_models"][0]["invocation_id"] = "0" * 32
    else:
        raise ValueError(f"unknown fixture mutation: {mutation}")

    result_raw = json.dumps(
        result, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    capture = json.loads(raw_inputs["capture_raw"])
    capture["result_capture"] = {"read": True, "bytes": len(result_raw), "sha256": _sha(result_raw)}
    capture["files"]["RESULT.json"] = {
        "read": True,
        "bytes": len(result_raw),
        "sha256": _sha(result_raw),
        "base64": base64.b64encode(result_raw).decode("ascii"),
    }
    kwargs = dict(raw_inputs)
    kwargs["result_raw"] = result_raw
    kwargs["capture_raw"] = json.dumps(
        capture, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    kwargs["result_path"] = helper.RESULT_PATH
    return kwargs


class ActualMetadataAcceptanceFixture(unittest.TestCase):
    def test_actual_capture_and_export_receipts_build_acceptance_and_reject_tampering(self) -> None:
        helper = _load_helper()
        raw_inputs = _inputs()

        score_launch = json.loads(raw_inputs["score_launch_raw"])
        native_capture = json.loads(raw_inputs["capture_raw"])
        self.assertEqual(score_launch["invocation_id"], EXPECTED_SCORE_INVOCATION)
        self.assertEqual(native_capture["invocation_id"], EXPECTED_SCORE_INVOCATION)
        self.assertEqual(score_launch["export_invocation_id"], EXPECTED_EXPORT_INVOCATION)
        self.assertEqual(native_capture["unit"]["InvocationID"], EXPECTED_SCORE_INVOCATION)

        kwargs = {**raw_inputs, "result_path": helper.RESULT_PATH}
        acceptance = helper.build_root_acceptance(**kwargs)
        self.assertEqual(acceptance["schema"], helper.ACCEPTANCE_SCHEMA)
        self.assertEqual(acceptance["status"], helper.ACCEPTANCE_STATUS)
        self.assertEqual(acceptance["unit"], helper.SCORE_UNIT)
        self.assertEqual(acceptance["invocation_id"], EXPECTED_SCORE_INVOCATION)
        self.assertEqual(acceptance["native_terminal"]["sub_state"], "exited")
        self.assertEqual(acceptance["result"]["bytes"], INPUTS["result_raw"][1])
        self.assertEqual(acceptance["stop_status"], "PENDING_EXACT_STOP")
        self.assertIs(acceptance["score_gain_or_public_transfer_claim"], False)

        for mutation in ("checkpoint_sha", "extra_loaded_sha", "invocation"):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                helper.build_root_acceptance(**_mutated_kwargs(helper, raw_inputs, mutation))


if __name__ == "__main__":
    unittest.main()
