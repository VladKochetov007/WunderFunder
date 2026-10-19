from __future__ import annotations

import unittest
from collections import Counter

from panel import (
    ARMS,
    GPU_UUIDS,
    REQUIRED_DENSITY_VARIANTS,
    DensityGate,
    OwnerGate,
    build_panel,
    cpu_pair,
    next_admissions,
    validate_panel,
)


def density_gate() -> DensityGate:
    return DensityGate(True, True, "a" * 64, 5, REQUIRED_DENSITY_VARIANTS)


def owner_gates(*, terminal: bool = False) -> dict[str, OwnerGate]:
    return {
        name: OwnerGate(
            status="terminal" if terminal else "live",
            verified=True,
            receipt_sha256=("b" if name == "r01" else "c") * 64,
            unit_inactive=terminal,
            cgroup_empty=terminal,
            actors_gone=terminal,
            canonical_complete=terminal,
        )
        for name in ("r01", "pr15")
    }


class FeatureSuccessorPanelTests(unittest.TestCase):
    def test_panel_is_four_matched_30_cell_successor_arms(self) -> None:
        panel = build_panel()
        jobs = panel["jobs"]
        self.assertEqual(len(jobs), 120)
        self.assertEqual(len({job["id"] for job in jobs}), 120)
        for arm in ARMS:
            selected = [
                job
                for job in jobs
                if job["experiment_id"] == arm.experiment_id and job["arm"] == arm.name
            ]
            self.assertEqual(len(selected), 30)
            self.assertEqual(
                Counter(job["fold"] for job in selected),
                Counter({0: 8, 1: 8, 2: 7, 3: 7}),
            )
            self.assertTrue(all(job["source_pr"] == arm.source_pr for job in selected))
            self.assertTrue(all(job["source_revision"].startswith("a266b036") for job in selected))
            self.assertTrue(all(job["cold_start"] for job in selected))
            self.assertTrue(all(job["initialization"] is None for job in selected))
            self.assertTrue(
                all(job["teacher"] is None and job["propensity"] is None for job in selected)
            )
            self.assertTrue(all(job["serving_export_ready"] is False for job in selected))

        age_jobs = [job for job in jobs if job["family"] == "field_age"]
        innovation_jobs = [job for job in jobs if job["family"] == "causal_innovation"]
        self.assertEqual({job["seed"] for job in jobs}, set(range(1101, 1109)))
        self.assertTrue(all(job["input_width"] == 323 for job in age_jobs))
        self.assertTrue(all(job["parameter_count"] == 2_267_650 for job in age_jobs))
        self.assertTrue(all(job["input_width"] == 291 for job in innovation_jobs))
        self.assertTrue(all(job["parameter_count"] == 2_251_266 for job in innovation_jobs))

        for family_jobs in (age_jobs, innovation_jobs):
            pairs: dict[str, list[dict[str, object]]] = {}
            for job in family_jobs:
                pairs.setdefault(str(job["pair_id"]), []).append(job)
            self.assertEqual(len(pairs), 30)
            for pair in pairs.values():
                self.assertEqual(len(pair), 2)
                self.assertEqual(
                    len({(job["fold"], job["seed"], job["gpu_index"]) for job in pair}), 1
                )

    def test_cpu_pairs_are_unique_and_numalocal(self) -> None:
        pairs = [cpu_pair(gpu, slot) for gpu in range(8) for slot in range(5)]
        self.assertEqual(len(set(pairs)), 40)
        self.assertEqual(min(cpu for pair in pairs for cpu in pair), 80)
        self.assertEqual(max(cpu for pair in pairs for cpu in pair), 159)
        for gpu in range(8):
            for slot in range(5):
                first, second = cpu_pair(gpu, slot)
                self.assertEqual(second, first + 2)
                self.assertEqual(first % 2, second % 2)
                self.assertEqual(first % 2, int(gpu >= 4))
        self.assertNotEqual(GPU_UUIDS[0], GPU_UUIDS[-1])

    def test_live_legacy_owners_reserve_four_lanes_and_seed_wave_spans_folds(self) -> None:
        panel = build_panel()
        pending = {str(job["id"]) for job in panel["jobs"]}
        wave = next_admissions(panel, pending, {}, owner_gates(), density_gate())
        self.assertEqual(len(wave), 8)
        self.assertEqual(
            Counter(job["gpu_index"] for job in wave),
            Counter({gpu: 1 for gpu in range(8)}),
        )
        self.assertEqual({job["slot_id"] for job in wave}, {0})
        self.assertEqual({job["seed"] for job in wave}, {1101})
        self.assertEqual({job["fold"] for job in wave}, {0, 1, 2, 3})
        self.assertEqual({job["family"] for job in wave}, {"field_age", "causal_innovation"})
        self.assertEqual(
            {job["gpu_index"] for job in wave if job["family"] == "field_age"},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            {job["gpu_index"] for job in wave if job["family"] == "causal_innovation"},
            {4, 5, 6, 7},
        )
        active = {str(job["id"]): (int(job["gpu_index"]), int(job["slot_id"])) for job in wave}
        remaining = pending - set(active)
        self.assertEqual(
            next_admissions(panel, remaining, active, owner_gates(), density_gate()), []
        )

    def test_one_live_legacy_owner_keeps_two_reservations(self) -> None:
        owners = owner_gates(terminal=True)
        owners["pr15"] = OwnerGate("live", True, "c" * 64)
        panel = build_panel()
        wave = next_admissions(
            panel,
            {str(job["id"]) for job in panel["jobs"]},
            {},
            owners,
            density_gate(),
        )
        self.assertEqual(len(wave), 24)
        self.assertEqual(
            Counter(job["gpu_index"] for job in wave),
            Counter({gpu: 3 for gpu in range(8)}),
        )
        self.assertEqual({job["slot_id"] for job in wave}, {0, 1, 2})

    def test_new_concurrency_caps_at_three_when_both_old_owners_are_terminal(self) -> None:
        panel = build_panel()
        pending = {str(job["id"]) for job in panel["jobs"]}
        wave = next_admissions(panel, pending, {}, owner_gates(terminal=True), density_gate())
        self.assertEqual(len(wave), 24)
        self.assertEqual(
            Counter(job["gpu_index"] for job in wave),
            Counter({gpu: 3 for gpu in range(8)}),
        )
        self.assertEqual({int(job["slot_id"]) for job in wave}, {0, 1, 2})
        cpu_ids = [cpu for job in wave for cpu in job["cpu_ids"]]
        self.assertEqual(len(cpu_ids), len(set(cpu_ids)))

    def test_fails_closed_without_density_or_full_terminal_receipts(self) -> None:
        panel = build_panel()
        pending = {str(job["id"]) for job in panel["jobs"]}
        bad_density = DensityGate(False, False, "a" * 64, 5, REQUIRED_DENSITY_VARIANTS)
        with self.assertRaisesRegex(ValueError, "density evidence"):
            next_admissions(panel, pending, {}, owner_gates(), bad_density)
        with self.assertRaisesRegex(ValueError, "both old R01 and PR15"):
            next_admissions(panel, pending, {}, {"r01": owner_gates()["r01"]}, density_gate())
        bad_terminal = owner_gates(terminal=True)
        bad_terminal["r01"] = OwnerGate("terminal", True, "b" * 64, True, False, True)
        with self.assertRaisesRegex(ValueError, "full drain"):
            next_admissions(panel, pending, {}, bad_terminal, density_gate())
        bad_terminal["r01"] = OwnerGate("terminal", True, "b" * 64, True, True, True)
        with self.assertRaisesRegex(ValueError, "full drain"):
            next_admissions(panel, pending, {}, bad_terminal, density_gate())

    def test_ledgers_reject_duplicate_ids_overlap_and_wrong_gpu(self) -> None:
        panel = build_panel()
        first = panel["jobs"][0]
        job_id = str(first["id"])
        with self.assertRaisesRegex(ValueError, "duplicate cell ids"):
            next_admissions(panel, [job_id, job_id], {}, owner_gates(), density_gate())
        with self.assertRaisesRegex(ValueError, "also be admitted as unstarted"):
            next_admissions(
                panel,
                {job_id},
                {job_id: (int(first["gpu_index"]), 0)},
                owner_gates(),
                density_gate(),
            )
        with self.assertRaisesRegex(ValueError, "frozen GPU assignment"):
            next_admissions(
                panel,
                set(),
                {job_id: ((int(first["gpu_index"]) + 1) % 8, 0)},
                owner_gates(),
                density_gate(),
            )

    def test_frozen_panel_rejects_recipe_or_parameter_tampering(self) -> None:
        panel = build_panel()
        panel["jobs"][0]["parameter_count"] += 1
        with self.assertRaisesRegex(ValueError, "pinned successor recipe"):
            validate_panel(panel)


if __name__ == "__main__":
    unittest.main()
