"""Synthetic correctness checks for the E485 duration actor."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import fit_duration as e485
import numpy as np
import torch
from fit_activation_placement import PlacementRNN

from wnn.optim import muon_param_groups

probe_spec = importlib.util.spec_from_file_location(
    "e485_gpu_probe", Path(__file__).with_name("gpu_probe.py")
)
if probe_spec is None or probe_spec.loader is None:
    raise ImportError("could not load the E485 probe delta gate")
probe = importlib.util.module_from_spec(probe_spec)
probe_spec.loader.exec_module(probe)


class DurationActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_model_count_initialization_and_optimizer_partition(self) -> None:
        states = e485.make_initial_states()
        self.assertEqual(set(states), set(e485.SEEDS))
        state = states[e485.SEEDS[0]]
        alpha_zero = e485.initialize_arm(0.0, state)
        alpha_sine = e485.initialize_arm(0.1, state)
        self.assertEqual(
            sum(parameter.numel() for parameter in alpha_zero.parameters()),
            236_802,
        )
        for candidate in (alpha_zero, alpha_sine):
            candidate.load_state_dict(state, strict=True)
            self.assertEqual(set(candidate.state_dict()), set(state))
            for name, value in candidate.state_dict().items():
                self.assertTrue(torch.equal(value, state[name]), name)
        groups = muon_param_groups(alpha_zero, scope="rnn")
        muon = [group for group in groups if group.get("muon")]
        selected = {
            name
            for name, parameter in alpha_zero.named_parameters()
            if any(id(parameter) == id(item) for group in muon for item in group["params"])
        }
        self.assertEqual(selected, e485.MUON_NAMES)
        self.assertEqual(muon[0]["blocks"], 3)
        self.assertEqual(
            sum(len(group["params"]) for group in groups), len(list(alpha_zero.parameters()))
        )
        self.assertEqual(
            len({id(p) for group in groups for p in group["params"]}),
            len(list(alpha_zero.parameters())),
        )

    def test_alpha_zero_matches_native_forward_state_gradients_and_clipped_step(self) -> None:
        torch.manual_seed(991)
        native = PlacementRNN("gelu", "gelu", hidden=e485.HIDDEN)
        custom = e485.DurationCandidateRNN(0.0)
        custom.load_state_dict(native.state_dict(), strict=True)
        x_native = torch.randn(4, 256, e485.FEATURES, requires_grad=True)
        x_custom = x_native.detach().clone().requires_grad_(True)
        state_native = torch.randn(2, 4, e485.HIDDEN, requires_grad=True)
        state_custom = state_native.detach().clone().requires_grad_(True)
        prediction_native, final_native = native(x_native, state_native)
        prediction_custom, final_custom = custom(x_custom, state_custom)
        torch.testing.assert_close(prediction_custom, prediction_native, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(final_custom, final_native, atol=2e-5, rtol=2e-4)
        target = torch.randn_like(prediction_native)
        loss_native = (
            prediction_native - target
        ).square().mean() + 0.1 * final_native.square().mean()
        loss_custom = (
            prediction_custom - target
        ).square().mean() + 0.1 * final_custom.square().mean()
        loss_native.backward()
        loss_custom.backward()
        torch.testing.assert_close(x_custom.grad, x_native.grad, atol=5e-5, rtol=5e-4)
        torch.testing.assert_close(state_custom.grad, state_native.grad, atol=5e-5, rtol=5e-4)
        for (name_c, parameter_c), (name_n, parameter_n) in zip(
            custom.named_parameters(), native.named_parameters(), strict=True
        ):
            self.assertEqual(name_c, name_n)
            torch.testing.assert_close(
                parameter_c.grad, parameter_n.grad, atol=5e-5, rtol=5e-4, msg=name_c
            )
        optimizer_custom = e485.make_optimizer(custom)
        optimizer_native = e485.make_optimizer(native)
        custom_before = {
            name: parameter.detach().clone() for name, parameter in custom.named_parameters()
        }
        native_before = {
            name: parameter.detach().clone() for name, parameter in native.named_parameters()
        }
        clipped_custom, _ = e485.optimizer_update(
            custom, optimizer_custom, 0, 0, "synthetic-native-custom/custom"
        )
        clipped_native, _ = e485.optimizer_update(
            native, optimizer_native, 0, 0, "synthetic-native-custom/native"
        )
        self.assertEqual(clipped_custom, clipped_native)
        delta_reports = probe.compare_update_deltas(native, custom, native_before, custom_before)
        self.assertTrue(all(report["passed"] for report in delta_reports.values()))

    def test_local_clip_update_matches_frozen_e474_on_identical_gradients(self) -> None:
        torch.manual_seed(4_851)
        initial = e485.make_initial_states()[e485.SEEDS[0]]
        local = e485.initialize_arm(0.0, initial)
        frozen = e485.initialize_arm(0.0, initial)
        local_optimizer = e485.make_optimizer(local)
        frozen_optimizer = e485.make_optimizer(frozen)
        for index, ((name_local, parameter_local), (name_frozen, parameter_frozen)) in enumerate(
            zip(local.named_parameters(), frozen.named_parameters(), strict=True)
        ):
            self.assertEqual(name_local, name_frozen)
            values = torch.arange(parameter_local.numel(), dtype=parameter_local.dtype).reshape_as(
                parameter_local
            )
            gradient = ((values.remainder(31) - 15.0) * 0.01) + (index + 1) * 0.001
            parameter_local.grad = gradient.clone()
            parameter_frozen.grad = gradient.clone()
        gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(parameter.grad) for parameter in local.parameters()]
            )
        )
        self.assertGreater(float(gradient_norm), e485.CLIP_NORM)
        clipped, _ = e485.optimizer_update(local, local_optimizer, 0, 0, "synthetic-helper/local")
        frozen_telemetry = e485.e474.apply_optimizer_update(
            frozen,
            frozen_optimizer,
            e485.e474.parameter_families(frozen),
            0,
            "synthetic-helper/frozen-e474",
        )
        self.assertTrue(clipped)
        self.assertIsNotNone(frozen_telemetry)
        local_parameters = dict(local.named_parameters())
        frozen_parameters = dict(frozen.named_parameters())
        for name, local_parameter in local_parameters.items():
            frozen_parameter = frozen_parameters[name]
            self.assertEqual(
                local_parameter.detach().contiguous().numpy().tobytes(),
                frozen_parameter.detach().contiguous().numpy().tobytes(),
            )
            local_state = local_optimizer.state[local_parameter]
            frozen_state = frozen_optimizer.state[frozen_parameter]
            self.assertEqual(local_state.keys(), frozen_state.keys())
            for key, value in local_state.items():
                other = frozen_state[key]
                if isinstance(value, torch.Tensor):
                    self.assertIsInstance(other, torch.Tensor)
                    self.assertEqual(
                        value.detach().contiguous().numpy().tobytes(),
                        other.detach().contiguous().numpy().tobytes(),
                    )
                else:
                    self.assertEqual(value, other)

    def test_causal_chunk_parity_state_reset_and_candidate_bound(self) -> None:
        torch.manual_seed(123)
        model = e485.DurationCandidateRNN(0.1).eval()
        x = torch.randn(2, 19, e485.FEATURES)
        with torch.no_grad():
            whole, whole_state = model(x)
            first, state = model(x[:, :7])
            second, state = model(x[:, 7:13], state)
            third, final_state = model(x[:, 13:], state)
            chunked = torch.cat((first, second, third), dim=1)
            zero_state = torch.zeros(2, 2, e485.HIDDEN)
            reset, reset_state = model(x, zero_state)
            changed_future = x.clone()
            changed_future[:, 11:] += 100.0 * torch.randn_like(changed_future[:, 11:])
            perturbed, _ = model(changed_future)
        torch.testing.assert_close(chunked, whole, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(final_state, whole_state, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(reset, whole, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(reset_state, whole_state, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(perturbed[:, :11], whole[:, :11], atol=0, rtol=0)
        u = torch.linspace(-1, 1, 100_001)
        candidate = u + 0.1 * torch.sin(torch.pi * u)
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertLessEqual(float(candidate.abs().max()), 1.0 + 1e-6)

    def test_epoch_orders_follow_registered_canonical_shuffle(self) -> None:
        self.assertEqual(e485.UPDATES_PER_EPOCH, 474)
        self.assertEqual(e485.UPDATES_PER_TRAJECTORY, 1_896)
        for seed in e485.SEEDS:
            orders = [e485.canonical_epoch_order(seed, epoch) for epoch in range(4)]
            for epoch, order in enumerate(orders):
                expected = np.arange(24)
                np.random.default_rng(seed + epoch).shuffle(expected)
                np.testing.assert_array_equal(order, expected)
                np.testing.assert_array_equal(np.sort(order), np.arange(24))
            self.assertTrue(any(not np.array_equal(orders[0], order) for order in orders[1:]))


if __name__ == "__main__":
    unittest.main()
