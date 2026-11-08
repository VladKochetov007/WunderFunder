"""CPU fixtures for precision receipts nested in bound Source6 producer ancestry."""

from __future__ import annotations

import copy
import unittest

from wnn.multistage_primary_refit_inputs import _strict_fp32


def _source6_producer() -> tuple[dict[str, object], dict[str, object]]:
    config = {
        "precision_mode": "fp32",
        "model": {"attention_backend": "math", "compile_blocks": False},
    }
    runtime = {
        "matmul_precision": "highest",
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast": False,
        "amp_scaler": False,
        "attention_backend": "math",
        "precision": {
            "mode": "fp32",
            "attention_backend": "math",
            "cublas_emulate_single_precision": None,
            "parameter_dtype": "float32",
            "gradient_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "activation_cache_dtype": "float32",
            "amp_scaler": False,
            "model_compute": "IEEE FP32",
            "model_forward": "strict FP32",
            "loss": "FP32",
            "optimizer": "FP32 outside model precision contexts",
        },
        "compiler": {
            "enabled": False,
            "backend": None,
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
            "recompile_limit": None,
        },
    }
    provenance = {"identity": {"runtime": runtime}, "identity_sha256": "a" * 64}
    producer = {
        "checkpoint_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "fit_members_sha256": "d" * 64,
        "provenance": provenance,
    }
    return config, producer


class Source6PrecisionTests(unittest.TestCase):
    def test_nested_source6_receipt_is_accepted_without_top_level_precision(self) -> None:
        config, producer = _source6_producer()
        provenance = producer["provenance"]
        runtime = provenance["identity"]["runtime"]
        self.assertNotIn("precision", producer)
        self.assertNotIn("precision", provenance)
        self.assertNotIn("precision", provenance["identity"])
        self.assertTrue(_strict_fp32(config, provenance, runtime))

    def test_wrong_precision_compiler_and_tf32_are_rejected(self) -> None:
        for section, key, value in (
            ("precision", "mode", "bf16_bf16x9"),
            ("precision", "parameter_dtype", "bfloat16"),
            ("precision", "cublas_emulate_single_precision", "1"),
            ("compiler", "enabled", True),
            ("compiler", "backend", "inductor"),
            (None, "matmul_tf32", True),
            (None, "autocast", True),
        ):
            config, producer = _source6_producer()
            provenance = producer["provenance"]
            runtime = provenance["identity"]["runtime"]
            target = runtime if section is None else runtime[section]
            target[key] = value
            with self.subTest(section=section, key=key):
                self.assertFalse(_strict_fp32(config, provenance, runtime))

    def test_conflicting_explicit_precision_is_not_overwritten(self) -> None:
        config, producer = _source6_producer()
        provenance = producer["provenance"]
        runtime = provenance["identity"]["runtime"]
        provenance["precision"] = copy.deepcopy(runtime["precision"])
        provenance["precision"]["parameter_dtype"] = "bfloat16"
        self.assertFalse(_strict_fp32(config, provenance, runtime))


if __name__ == "__main__":
    unittest.main()
