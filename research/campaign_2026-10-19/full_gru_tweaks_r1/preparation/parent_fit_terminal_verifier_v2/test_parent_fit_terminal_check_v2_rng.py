from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

PREPARATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREPARATION))

from parent_fit_terminal_check_v2 import (  # noqa: E402
    RNG_DIAGNOSTIC_TORCH_VERSION,
    _check_cuda_rng,
    _check_tensor_rng,
    _tensor_rng_signature,
    _validate_rng_diagnostic,
)


class ParentRngDiagnosticTests(unittest.TestCase):
    """Exercise the new byte-signature guards with tiny synthetic tensors."""

    def test_cuda_uint8_sixteen_bytes_matches_expected_digest(self) -> None:
        tensor = torch.arange(16, dtype=torch.uint8)
        signature = _tensor_rng_signature(tensor)
        self.assertEqual(signature["sha256"], hashlib.sha256(bytes(range(16))).hexdigest())
        expected = {
            "type": "builtins.list",
            "cardinality": 1,
            "components": [signature],
        }
        _check_cuda_rng([tensor], expected)

    def test_wrong_tensor_length_and_dtype_fail(self) -> None:
        tensor = torch.arange(16, dtype=torch.uint8)
        expected = {
            "type": "builtins.list",
            "cardinality": 1,
            "components": [_tensor_rng_signature(tensor)],
        }
        with self.assertRaises(ValueError):
            _check_cuda_rng([tensor[:15]], expected)
        with self.assertRaises(ValueError):
            _check_cuda_rng([torch.arange(16, dtype=torch.int16)], expected)

    def test_wrong_cuda_cardinality_and_digest_fail(self) -> None:
        tensor = torch.arange(16, dtype=torch.uint8)
        signature = _tensor_rng_signature(tensor)
        expected = {
            "type": "builtins.list",
            "cardinality": 1,
            "components": [signature],
        }
        with self.assertRaises(ValueError):
            _check_cuda_rng([tensor, tensor], expected)
        bad_digest = {**signature, "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            _check_tensor_rng(tensor, bad_digest, "synthetic CUDA RNG")

    def test_diagnostic_torch_version_is_bound(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "records/parent_fit_r2_v1/rng_diagnostic_v2/capture/RNG_DIAGNOSTIC.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_rng_diagnostic(
            report, report["state_file_stat_only"], RNG_DIAGNOSTIC_TORCH_VERSION
        )
        self.assertEqual(report["torch_version"], RNG_DIAGNOSTIC_TORCH_VERSION)
        report["torch_version"] = "unexpected"
        with self.assertRaises(ValueError):
            _validate_rng_diagnostic(
                report, report["state_file_stat_only"], RNG_DIAGNOSTIC_TORCH_VERSION
            )
        report["torch_version"] = RNG_DIAGNOSTIC_TORCH_VERSION
        with self.assertRaises(ValueError):
            _validate_rng_diagnostic(report, report["state_file_stat_only"], "unexpected")


if __name__ == "__main__":
    unittest.main()
