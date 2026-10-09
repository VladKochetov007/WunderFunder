from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
D02_CODE_ROOT = REPO_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "d02_muon_loss_successor_r2/code"
)
sys.path.insert(0, str(D02_CODE_ROOT / "src"))
import recipe_sampler as sampler  # noqa: E402

from wnn import optim as d02_optim  # noqa: E402
from wnn import training_dynamics as d02_dynamics  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash the exact D02 module that the test imports."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simple_model() -> nn.Module:
    """Build the smallest model shape accepted by the D02 recurrent optimizer."""

    class TinyGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=4, hidden_size=3, batch_first=True)
            self.head = nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            values, _ = self.gru(x)
            return self.head(values)

    return TinyGRU()


def optimizer_args(name: str) -> argparse.Namespace:
    """Return only the fields consumed by D02's actual optimizer builder."""
    return argparse.Namespace(
        optim=name,
        head_lr_mult=1.0,
        zoneout=0.0,
        momentum=0.0,
        optim_switch="",
        lr=0.001,
        beta2=0.999,
        wd=0.01,
        muon_scope="rnn",
        muon_ortho_dtype_policy="fp32",
        fused_adam=False,
    )


class D02SourcePolicyTests(unittest.TestCase):
    def test_imported_d02_source_matches_frozen_closure_and_not_live_main(self) -> None:
        spec = json.loads((HERE / "sampler_spec_r2.json").read_text(encoding="utf-8"))
        source = (HERE / "candidate_pool_r2.jsonl").read_text(encoding="utf-8").splitlines()[0]
        recipe = json.loads(source)["recipe"]
        pinned = recipe["source"]["sha256_by_path"]
        optimizer_path = D02_CODE_ROOT / "src/wnn/optim.py"
        dynamics_path = D02_CODE_ROOT / "src/wnn/training_dynamics.py"
        self.assertEqual(
            sha256_file(optimizer_path), pinned[optimizer_path.relative_to(REPO_ROOT).as_posix()]
        )
        self.assertEqual(
            sha256_file(dynamics_path), pinned[dynamics_path.relative_to(REPO_ROOT).as_posix()]
        )
        self.assertEqual(Path(d02_optim.__file__).resolve(), optimizer_path.resolve())
        self.assertEqual(Path(d02_dynamics.__file__).resolve(), dynamics_path.resolve())
        live_optimizer = REPO_ROOT / "src/wnn/optim.py"
        self.assertNotIn(
            live_optimizer.relative_to(REPO_ROOT).as_posix(), spec["source_code_paths"]
        )

    def test_actual_d02_orthogonalize_uses_fp32_for_low_precision_gradient(self) -> None:
        gradient = torch.randn(3, 4, 5, dtype=torch.float16)
        requested_dtypes: list[torch.dtype] = []
        original_to = torch.Tensor.to

        def tracked_to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            dtype = kwargs.get("dtype")
            if dtype is None and args and isinstance(args[0], torch.dtype):
                dtype = args[0]
            if isinstance(dtype, torch.dtype):
                requested_dtypes.append(dtype)
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", tracked_to):
            result = d02_optim.orthogonalize(gradient, dtype_policy="fp32")
        self.assertEqual(result.dtype, gradient.dtype)
        self.assertGreaterEqual(requested_dtypes.count(torch.float32), 1)
        self.assertEqual(requested_dtypes[-1], gradient.dtype)
        self.assertNotIn(torch.bfloat16, requested_dtypes)
        with self.assertRaisesRegex(ValueError, "explicit FP32"):
            d02_optim.orthogonalize(gradient, dtype_policy="bf16")

    def test_actual_d02_builder_supports_adamw_and_fp32_muon(self) -> None:
        adamw = d02_dynamics.build_optimizer(simple_model(), optimizer_args("adamw"))
        self.assertIsInstance(adamw, torch.optim.AdamW)
        muon_args = optimizer_args("muon")
        muon_args.precision = "fp32"
        muon = d02_dynamics.build_optimizer(simple_model(), muon_args)
        self.assertIsInstance(muon, d02_optim.MuonAdamW)
        self.assertTrue(
            all(
                group["orthogonalization_dtype_policy"] == "fp32"
                for group in muon.param_groups
                if group["muon"]
            )
        )
        record = d02_dynamics.muon_precision_record(muon, muon_args, torch.device("cpu"))
        self.assertEqual(record["selected_orthogonalization_dtype"], "float32")

    def test_actual_d02_scheduler_uses_frozen_r2_onecycle_parameters(self) -> None:
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.001, betas=(0.9, 0.999))
        args = argparse.Namespace(
            sched="onecycle",
            lr=0.001,
            warmup_frac=0.05,
            wsd_decay=0.2,
            wsd_final=0.01,
            optim="adamw",
            head_lr_mult=1.0,
        )
        captured: dict[str, object] = {}
        sentinel = object()

        def capture_onecycle(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return sentinel

        with patch.object(torch.optim.lr_scheduler, "OneCycleLR", capture_onecycle):
            scheduler = d02_dynamics.build_scheduler(optimizer, args, total=220)
        self.assertIs(scheduler, sentinel)
        self.assertEqual(captured["pct_start"], 0.05)
        self.assertEqual(captured["final_div_factor"], 100)
        self.assertEqual(captured["cycle_momentum"], True)
        self.assertEqual(captured["max_lr"], 0.001)
        self.assertEqual(
            inspect.signature(torch.optim.lr_scheduler.OneCycleLR).parameters["div_factor"].default,
            25.0,
        )

    def test_all_four_matched_recipes_share_one_admitted_source_but_remain_pending(self) -> None:
        rows = [
            json.loads(line)
            for line in (HERE / "candidate_pool_r2.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_id = {row["candidate_id"]: row for row in rows}
        expected_optimizers = {
            "A_adamw_wmse": ("adamw", "wmse", None),
            "B_adamw_huber_d1": ("adamw", "huber", 1.0),
            "C_muon_wmse": ("muon", "wmse", None),
            "D_muon_huber_d1": ("muon", "huber", 1.0),
        }
        source_hashes = set()
        for candidate_id, expected in expected_optimizers.items():
            recipe = by_id[candidate_id]["recipe"]
            observed = (
                recipe["optimizer"]["name"],
                recipe["loss"]["name"],
                recipe["loss"]["huber_delta"],
            )
            self.assertEqual(observed, expected)
            source_hashes.add(recipe["source"]["binding_hash"])
            self.assertEqual(
                recipe["implementation_dependency"]["state"], "pending_recipe_aware_runner"
            )
            with self.assertRaisesRegex(sampler.RecipeError, "validated recipe-aware runner"):
                sampler.assert_recipe_runner_ready(recipe)
        self.assertEqual(len(source_hashes), 1)


if __name__ == "__main__":
    unittest.main()
