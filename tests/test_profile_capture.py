from __future__ import annotations

import argparse
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from wnn.optim import MuonAdamW
from wnn.training.cli import build_parser
from wnn.training.environment import TrainingEnvironment
from wnn.training.profile_capture import (
    ProfileCapture,
    install_capture_hooks,
    profile_argument_error,
    profile_state_summary,
    run_teacher_profile,
)


def _profile_argv() -> list[str]:
    return [
        "--name",
        "profile-fixture",
        "--seed",
        "3",
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        "384",
        "--layers",
        "2",
        "--head-layers",
        "1",
        "--epochs",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "2",
        "--batch",
        "32",
        "--chunk",
        "1000",
        "--optim",
        "muon",
        "--muon-scope",
        "all",
        "--muon-ortho-dtype",
        "fp32",
        "--strict-muon-fp32",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--beta2",
        "0.999",
        "--sched",
        "onecycle",
        "--warmup-frac",
        "0.05",
        "--clip",
        "1",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--precision",
        "fp32",
        "--aux-struct",
        "0.3",
        "--aux-next",
        "0.3",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--propensity",
        "fg19_propensity_clean_f0_seed0_e4",
        "--prop-power",
        "2",
        "--prop-eps",
        "0.03",
        "--valid-true-mask",
        "0.03",
        "--sync-free",
        "--pin-memory",
        "--grad-stats",
        "--max-batches",
        "6",
        "--profile-warmup",
        "2",
        "--profile-steps",
        "4",
        "--split-manifest",
        "fixture.json",
        "--train-source",
        "train",
        "--no-eval-during-fit",
    ]


def _profile_args() -> argparse.Namespace:
    return build_parser().parse_args(_profile_argv())


class ProfileCaptureTests(unittest.TestCase):
    def test_profile_recipe_requires_complete_grad_stats(self) -> None:
        args = _profile_args()
        self.assertIsNone(profile_argument_error(args))
        args.grad_stats = False
        self.assertIn("--grad-stats", profile_argument_error(args) or "")

    def test_profile_rejects_fast_unbound_propensity_cache(self) -> None:
        args = _profile_args()
        args.fast = True
        self.assertIn("propensity cache", profile_argument_error(args) or "")

    def test_state_summary_checks_fp32_finite_optimizer_state(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = MuonAdamW(
            [
                {
                    "params": [model.weight],
                    "muon": True,
                    "blocks": 1,
                    "muon_fp32_ortho": True,
                },
                {"params": [model.bias]},
            ]
        )
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        summary = profile_state_summary(model, optimizer)
        self.assertTrue(summary["model_all_fp32_finite"])
        self.assertTrue(summary["all_muon_groups_fp32_ortho"])
        self.assertTrue(summary["optimizer_float_state_all_fp32_finite"])

    def test_capture_hooks_retain_canonical_objects(self) -> None:
        environment = TrainingEnvironment()
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        environment.make_model = lambda _args, _n_in, _n_out: model
        environment.build_optimizer = lambda _model, _args: optimizer
        capture = ProfileCapture()
        install_capture_hooks(environment, capture)
        self.assertIs(environment.make_model({}, 2, 1), model)
        with (
            patch("wnn.training.profile_capture.torch.cuda.synchronize"),
            patch("wnn.training.profile_capture.torch.cuda.max_memory_allocated", return_value=10),
            patch("wnn.training.profile_capture.torch.cuda.max_memory_reserved", return_value=20),
        ):
            self.assertIs(
                environment.build_optimizer(model, argparse.Namespace(device="cuda")), optimizer
            )
        self.assertIs(capture.model, model)
        self.assertIs(capture.optimizer, optimizer)
        self.assertEqual(capture.setup_cuda_peak_allocated_bytes, 10)
        self.assertEqual(capture.setup_cuda_peak_reserved_bytes, 20)

    def test_peak_snapshots_precede_state_checks_and_include_setup_and_fit(self) -> None:
        order: list[str] = []
        emitted: dict[str, object] = {}
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        environment = SimpleNamespace(
            make_model=lambda _args, _n_in, _n_out: model,
            build_optimizer=lambda _model, _args: optimizer,
        )

        def run_fit(_argv: list[str], *, environment: SimpleNamespace) -> None:
            model_ref = environment.make_model({}, 211, 23)
            environment.build_optimizer(model_ref, argparse.Namespace(device="cuda"))
            order.append("fit_return")

        allocated_values = iter((101, 70))
        reserved_values = iter((202, 280))

        def read_allocated(_device: str) -> int:
            value = next(allocated_values)
            order.append(f"allocated-{value}")
            return value

        def read_reserved(_device: str) -> int:
            value = next(reserved_values)
            order.append(f"reserved-{value}")
            return value

        def summarize(_model: nn.Module, _optimizer: torch.optim.Optimizer) -> dict[str, bool]:
            order.append("state_summary")
            return {"synthetic_state_summary": True}

        def record_emit(_tag: str, payload: str, **_kwargs: object) -> None:
            emitted.update(json.loads(payload))

        original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            with (
                patch("wnn.training.profile_capture.configure_profile_runtime"),
                patch("wnn.training.environment.TrainingEnvironment", return_value=environment),
                patch("wnn.training.profile_capture.run_training", side_effect=run_fit),
                patch("wnn.training.profile_capture.profile_state_summary", side_effect=summarize),
                patch("wnn.training.profile_capture.emit", side_effect=record_emit),
                patch("wnn.training.profile_capture.torch.cuda.synchronize"),
                patch(
                    "wnn.training.profile_capture.torch.cuda.max_memory_allocated",
                    side_effect=read_allocated,
                ),
                patch(
                    "wnn.training.profile_capture.torch.cuda.max_memory_reserved",
                    side_effect=read_reserved,
                ),
                patch("wnn.training.profile_capture.torch.autocast", return_value=nullcontext()),
                patch("wnn.training.profile_capture.torch.is_autocast_enabled", return_value=False),
                patch(
                    "wnn.training.profile_capture.torch.get_float32_matmul_precision",
                    return_value="highest",
                ),
            ):
                run_teacher_profile(_profile_argv())
        finally:
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

        self.assertLess(order.index("allocated-70"), order.index("state_summary"))
        self.assertLess(order.index("reserved-280"), order.index("state_summary"))
        self.assertEqual(emitted["setup_cuda_peak_allocated_bytes"], 101)
        self.assertEqual(emitted["fit_cuda_peak_allocated_bytes"], 70)
        self.assertEqual(emitted["cuda_peak_allocated_bytes"], 101)
        self.assertEqual(emitted["setup_cuda_peak_reserved_bytes"], 202)
        self.assertEqual(emitted["fit_cuda_peak_reserved_bytes"], 280)
        self.assertEqual(emitted["cuda_peak_reserved_bytes"], 280)


if __name__ == "__main__":
    unittest.main()
