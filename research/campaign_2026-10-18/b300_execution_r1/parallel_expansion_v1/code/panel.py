"""Pure panel and admission planning for two cold-start feature successors.

This module does not inspect devices, processes, units, receipts, or files and
never starts work. Its caller must supply authenticated owner/resource gates and
the exact cells proved to have no prior evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
PANEL_SCHEMA = "tabdata-feature-successor-panel.v1"
GPU_UUIDS = (
    "GPU-daf615f5-ea55-860a-bfe1-41c36ab96c3e",
    "GPU-25ae492c-db98-e1ae-0f5e-c56365a422c7",
    "GPU-df6fe7d0-64e5-7af2-467d-c1699cb6fc67",
    "GPU-76951d88-add3-0d6b-5bee-c0390915ca47",
    "GPU-e6d65cb1-aba4-14f5-d4e5-bf24ef1ee3d8",
    "GPU-1fc4d921-dc78-6711-c2a8-ec0589b7f106",
    "GPU-2913ac69-eccb-63b0-1112-e09ea9598b57",
    "GPU-be85f55c-71a1-f389-eeed-55c63a0e3ce7",
)
FOLD_SEEDS = {
    0: tuple(range(1101, 1109)),
    1: tuple(range(1101, 1109)),
    2: tuple(range(1101, 1108)),
    3: tuple(range(1101, 1108)),
}
BASE_INPUT_WIDTH = 211  # raw112 + the canonical99 event features
BASE_WIDE_PARAMETERS = 2_210_306
HIDDEN_WIDTH = 512
MAX_NEW_SLOTS_PER_GPU = 5  # Stable slot IDs 0..4; not a concurrent admission count.
MAX_NEW_CONCURRENT_JOBS_PER_GPU = 3
MEASURED_TOTAL_ACTORS_PER_GPU = 5
OLD_RESERVED_SLOTS_PER_GPU = 4  # two R01 plus two PR15, including phase gaps
LEGACY_SLOTS_PER_LIVE_OWNER = 2
NEW_CPU_START = 80
CPUS_PER_SLOT_STRIDE = 16
GPU_CPU_STRIDE = 4
REQUIRED_DENSITY_VARIANTS = frozenset(
    {
        "group_age",
        "field_age",
        "causal_innovations:newest",
        "causal_innovations:none",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FIT_RECIPE = {
    "architecture": "gru",
    "hidden": 512,
    "layers": 1,
    "dropout": 0,
    "input_encoder": "linear",
    "epochs": 6,
    "fixed_checkpoint_epoch_zero_based": 5,
    "batch": 8,
    "chunk": 2000,
    "precision": "fp32",
    "autocast": False,
    "tf32": False,
    "optimizer": "adamw",
    "lr": 0.0001,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "scheduler": "onecycle",
    "warmup_fraction": 0.05,
    "head_lr_multiplier": 1,
    "gradient_clip": 1.0,
    "loss": "wmse",
    "loss_normalization": "chunk",
    "train_source": "train+valid excluding the assigned validation fold",
    "train_sequences": 10607,
    "valid_train_fraction": 1,
    "valid_repeat": 1,
    "n_valid_folds": 4,
    "full_valid": True,
    "eval_source": "vfold",
    "outer_evaluation_during_fit": False,
    "batch_order": "seeded",
    "deterministic": True,
    "fused_adam": True,
    "sync_free": True,
    "pin_memory": False,
    "loader_workers": 0,
    "grad_stats": True,
    "save_training_state": True,
    "initialization": None,
    "teacher": None,
    "propensity": None,
}


@dataclass(frozen=True)
class Arm:
    """One fixed-width feature arm attached to a new successor experiment."""

    experiment_id: str
    source_pr: int
    family: str
    name: str
    feature_mode: str
    feature_control: str
    added_width: int


ARMS = (
    Arm(
        "E-B300-PR4-AGE-SUCCESSOR-V1",
        4,
        "field_age",
        "group_age",
        "group_age",
        "none",
        112,
    ),
    Arm(
        "E-B300-PR4-AGE-SUCCESSOR-V1",
        4,
        "field_age",
        "field_age",
        "field_age",
        "none",
        112,
    ),
    Arm(
        "E-B300-PR14-INNOVATION-SUCCESSOR-V1",
        14,
        "causal_innovation",
        "matched_moments",
        "causal_innovations",
        "newest",
        80,
    ),
    Arm(
        "E-B300-PR14-INNOVATION-SUCCESSOR-V1",
        14,
        "causal_innovation",
        "moments_plus_innovation",
        "causal_innovations",
        "none",
        80,
    ),
)


@dataclass(frozen=True)
class OwnerGate:
    """One root-verified observation for the old R01 or PR15 allocation."""

    status: str
    verified: bool
    receipt_sha256: str
    unit_inactive: bool = False
    cgroup_empty: bool = False
    actors_gone: bool = False
    canonical_complete: bool = False


@dataclass(frozen=True)
class DensityGate:
    """Root-verified five-actor-per-GPU feature-density measurement."""

    verified: bool
    passed: bool
    receipt_sha256: str
    max_actors_per_gpu: int
    variants: frozenset[str]


def _cell_specs() -> list[tuple[int, int, int]]:
    """Return seed-major paired cells, interleaving folds in every early wave."""
    cells = []
    for seed in sorted({seed for seeds in FOLD_SEEDS.values() for seed in seeds}):
        for fold in range(4):
            if seed in FOLD_SEEDS[fold]:
                cells.append((len(cells), fold, seed))
    if len(cells) != 30:
        raise AssertionError("successor panel must contain exactly 30 paired cells")
    return cells


def _arm_order(cell_index: int, family: str) -> tuple[Arm, Arm]:
    """Alternate which member of a pair is first to reduce order confounding."""
    pair = tuple(arm for arm in ARMS if arm.family == family)
    if len(pair) != 2:
        raise AssertionError(f"expected exactly two arms for {family}")
    return pair if cell_index % 2 == 0 else (pair[1], pair[0])


def build_panel() -> dict[str, object]:
    """Build 120 cold-start cells in two separately identified successors.

    Each pair shares fold, seed, and preferred GPU. The two feature families
    receive opposite four-GPU placement offsets. Device placement is advisory;
    the host controller must use the admission function below before launch.
    """
    jobs: list[dict[str, object]] = []
    cells = _cell_specs()
    for cell_index, fold, seed in cells:
        pair_key = f"f{fold}_s{seed}"
        for family_index, family in enumerate(("field_age", "causal_innovation")):
            gpu_index = (cell_index + family_index * 4) % len(GPU_UUIDS)
            for arm in _arm_order(cell_index, family):
                run_name = f"b300_{arm.family}_successor_{arm.name}_f{fold}_s{seed}"
                jobs.append(
                    {
                        "id": run_name,
                        "experiment_id": arm.experiment_id,
                        "source_pr": arm.source_pr,
                        "successor": True,
                        "family": arm.family,
                        "arm": arm.name,
                        "pair_id": f"{arm.family}_{pair_key}",
                        "pair_index": cell_index,
                        "fold": fold,
                        "seed": seed,
                        "gpu_index": gpu_index,
                        "gpu_uuid": GPU_UUIDS[gpu_index],
                        "source_revision": SOURCE_REVISION,
                        "architecture": "gru",
                        "hidden": HIDDEN_WIDTH,
                        "layers": 1,
                        "cold_start": True,
                        "initialization": None,
                        "teacher": None,
                        "propensity": None,
                        "feature_mode": arm.feature_mode,
                        "feature_control": arm.feature_control,
                        "feature_width": arm.added_width,
                        "input_width": BASE_INPUT_WIDTH + arm.added_width,
                        "parameter_count": BASE_WIDE_PARAMETERS + HIDDEN_WIDTH * arm.added_width,
                        "trainer_argv_delta": [
                            "--research-features",
                            arm.feature_mode,
                            "--research-feature-control",
                            arm.feature_control,
                        ],
                        "serving_export_ready": False,
                        "serving_gate": (
                            "streaming_graph_rejects_field_age_modes"
                            if family == "field_age"
                            else "streaming_graph_implemented; parity_required"
                        ),
                    }
                )

    panel = {
        "schema": PANEL_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "fit_recipe": dict(FIT_RECIPE),
        "cells_per_arm": 30,
        "jobs": jobs,
    }
    validate_panel(panel)
    return panel


def validate_panel(panel: Mapping[str, object]) -> None:
    """Reject duplicate, incomplete, or non-successor panel definitions."""
    jobs = panel.get("jobs")
    if (
        panel.get("schema") != PANEL_SCHEMA
        or panel.get("source_revision") != SOURCE_REVISION
        or not isinstance(jobs, list)
    ):
        raise ValueError("unsupported successor panel")
    recipe = panel.get("fit_recipe")
    if not isinstance(recipe, Mapping) or dict(recipe) != FIT_RECIPE:
        raise ValueError("successor fit recipe changed from the frozen six-epoch FP32 recipe")
    ids = [job.get("id") for job in jobs if isinstance(job, dict)]
    if len(jobs) != 120 or len(ids) != len(jobs) or len(set(ids)) != 120:
        raise ValueError("panel must contain 120 uniquely identified jobs")
    expected_cells = {(fold, seed): index for index, fold, seed in _cell_specs()}
    for arm in ARMS:
        selected = [
            job
            for job in jobs
            if job.get("experiment_id") == arm.experiment_id and job.get("arm") == arm.name
        ]
        if len(selected) != 30:
            raise ValueError(f"{arm.name} must contain 30 cells")
        for job in selected:
            fold, seed = job.get("fold"), job.get("seed")
            if type(fold) is not int or type(seed) is not int or (fold, seed) not in expected_cells:
                raise ValueError("successor cell is outside the frozen fold/seed panel")
            cell_index = expected_cells[(fold, seed)]
            expected_gpu = (cell_index + (4 if arm.source_pr == 14 else 0)) % len(GPU_UUIDS)
            if any(
                (
                    job.get("source_revision") != SOURCE_REVISION,
                    job.get("source_pr") != arm.source_pr,
                    job.get("family") != arm.family,
                    job.get("architecture") != "gru",
                    job.get("hidden") != HIDDEN_WIDTH,
                    job.get("layers") != 1,
                    job.get("cold_start") is not True,
                    job.get("successor") is not True,
                    job.get("initialization") is not None,
                    job.get("teacher") is not None,
                    job.get("propensity") is not None,
                    job.get("feature_mode") != arm.feature_mode,
                    job.get("feature_control") != arm.feature_control,
                    job.get("feature_width") != arm.added_width,
                    job.get("input_width") != BASE_INPUT_WIDTH + arm.added_width,
                    job.get("parameter_count")
                    != BASE_WIDE_PARAMETERS + HIDDEN_WIDTH * arm.added_width,
                    job.get("gpu_index") != expected_gpu,
                    job.get("gpu_uuid") != GPU_UUIDS[expected_gpu],
                    job.get("trainer_argv_delta")
                    != [
                        "--research-features",
                        arm.feature_mode,
                        "--research-feature-control",
                        arm.feature_control,
                    ],
                    job.get("serving_export_ready") is not False,
                )
            ):
                raise ValueError(f"{arm.name} cell does not match its pinned successor recipe")
    for family in ("field_age", "causal_innovation"):
        by_pair: dict[str, list[Mapping[str, object]]] = {}
        for job in jobs:
            if job.get("family") == family:
                by_pair.setdefault(str(job["pair_id"]), []).append(job)
        if len(by_pair) != 30 or any(len(pair) != 2 for pair in by_pair.values()):
            raise ValueError(f"{family} must have 30 two-arm pairs")
        for pair_id, pair in by_pair.items():
            if len({(job["fold"], job["seed"], job["gpu_index"]) for job in pair}) != 1:
                raise ValueError(f"{family} pair has mismatched fold, seed, or GPU")
            only = pair[0]
            if pair_id != f"{family}_f{only['fold']}_s{only['seed']}":
                raise ValueError(f"{family} pair identity is not canonical")


def cpu_pair(gpu_index: int, slot_id: int) -> tuple[int, int]:
    """Return the unique two-CPU, same-NUMA affinity for one new GPU slot."""
    if type(gpu_index) is not int or gpu_index not in range(8):
        raise ValueError("GPU index must be in [0, 7]")
    if type(slot_id) is not int or slot_id not in range(MAX_NEW_SLOTS_PER_GPU):
        raise ValueError("new slot id must be in [0, 4]")
    first = NEW_CPU_START + slot_id * CPUS_PER_SLOT_STRIDE
    first += gpu_index * GPU_CPU_STRIDE if gpu_index < 4 else (gpu_index - 4) * GPU_CPU_STRIDE + 1
    return first, first + GPU_CPU_STRIDE // 2


def _validate_gates(owner_gates: Mapping[str, OwnerGate], density_gate: DensityGate) -> int:
    """Validate root receipts and return reserved legacy slots per GPU."""
    if set(owner_gates) != {"r01", "pr15"}:
        raise ValueError("both old R01 and PR15 owner gates are required")
    if (
        density_gate.verified is not True
        or density_gate.passed is not True
        or type(density_gate.max_actors_per_gpu) is not int
        or density_gate.max_actors_per_gpu != MEASURED_TOTAL_ACTORS_PER_GPU
        or not REQUIRED_DENSITY_VARIANTS.issubset(density_gate.variants)
        or not _SHA256.fullmatch(density_gate.receipt_sha256)
    ):
        raise ValueError("verified five-actor density evidence for all four variants is required")

    reserved = 0
    for name, gate in owner_gates.items():
        if gate.verified is not True or not _SHA256.fullmatch(gate.receipt_sha256):
            raise ValueError(f"{name} owner lacks a verified, hash-bound observation")
        if gate.status == "live":
            reserved += LEGACY_SLOTS_PER_LIVE_OWNER
        elif gate.status == "terminal":
            if not (
                gate.canonical_complete is True
                and gate.unit_inactive is True
                and gate.cgroup_empty is True
                and gate.actors_gone is True
            ):
                raise ValueError(f"{name} terminal receipt does not prove full drain")
        else:
            raise ValueError(f"{name} owner status must be live or terminal")
    if reserved not in (0, 2, 4):
        raise AssertionError("legacy owner reservations must be 0, 2, or 4 slots per GPU")
    return reserved


def next_admissions(
    panel: Mapping[str, object],
    confirmed_unstarted_ids: Iterable[str],
    active_cells: Mapping[str, tuple[int, int]],
    owner_gates: Mapping[str, OwnerGate],
    density_gate: DensityGate,
) -> list[dict[str, object]]:
    """Return a pure next-wave plan, reserving four legacy lanes per GPU.

    ``confirmed_unstarted_ids`` must come from a root-validated ledger and may
    contain only cells with no prior start/artifact evidence. ``active_cells``
    must include every new fit/export/grade actor until its whole worker exits.
    Each live R01/PR15 owner reserves two per-GPU slots, irrespective of
    transient NVSMI process gaps or an empty export lane. A terminal reservation
    is released only after the caller proves the canonical family is complete,
    its unit is inactive, its cgroup is empty, and its owned actors are gone.
    """
    validate_panel(panel)
    old_reserved = _validate_gates(owner_gates, density_gate)
    jobs = panel["jobs"]
    by_id = {str(job["id"]): job for job in jobs}
    supplied_pending = list(confirmed_unstarted_ids)
    pending = set(supplied_pending)
    if len(supplied_pending) != len(pending):
        raise ValueError("unstarted ledger contains duplicate cell ids")
    if not pending.issubset(by_id):
        raise ValueError("unstarted ledger contains ids outside the frozen panel")
    if not pending.isdisjoint(active_cells):
        raise ValueError("active cells cannot also be admitted as unstarted")

    occupied: dict[int, set[int]] = {gpu: set() for gpu in range(8)}
    for job_id, placement in active_cells.items():
        if job_id not in by_id or not isinstance(placement, (tuple, list)) or len(placement) != 2:
            raise ValueError("active cell identity or placement is outside the panel")
        gpu_index, slot_id = placement
        cpu_pair(gpu_index, slot_id)
        if gpu_index != by_id[job_id]["gpu_index"]:
            raise ValueError("active cell was moved from its frozen GPU assignment")
        if slot_id in occupied[gpu_index]:
            raise ValueError("two active new jobs occupy the same GPU slot")
        occupied[gpu_index].add(slot_id)

    if old_reserved not in (0, 2, OLD_RESERVED_SLOTS_PER_GPU):
        raise AssertionError("unexpected legacy reservation count")
    new_capacity = min(
        MAX_NEW_CONCURRENT_JOBS_PER_GPU,
        density_gate.max_actors_per_gpu - old_reserved,
    )
    if new_capacity < 1:
        raise ValueError("no measured capacity remains after legacy reservations")
    for gpu_index, slots in occupied.items():
        if len(slots) > new_capacity or any(slot >= new_capacity for slot in slots):
            raise ValueError(f"active new jobs exceed the authorized GPU {gpu_index} capacity")

    plan: list[dict[str, object]] = []
    for job in jobs:
        job_id = str(job["id"])
        if job_id not in pending:
            continue
        gpu_index = int(job["gpu_index"])
        free = next(
            (slot for slot in range(new_capacity) if slot not in occupied[gpu_index]),
            None,
        )
        if free is None:
            continue
        scheduled = dict(job)
        scheduled["slot_id"] = free
        scheduled["cpu_ids"] = list(cpu_pair(gpu_index, free))
        plan.append(scheduled)
        occupied[gpu_index].add(free)
    return plan
