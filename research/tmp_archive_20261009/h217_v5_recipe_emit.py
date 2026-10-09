#!/usr/bin/env python3
"""Emit the versioned H217 V5 recipe. Not part of the packet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PACKET = Path(
    "/tmp/wnn-h217-full-feature-ladder/research/campaign_2026-10-06/"
    "bandit_continuation/continuous_execution_E529/vast/h217_packet_v5"
)
SPEC = importlib.util.spec_from_file_location("h217_v5_worker", PACKET / "worker.py")
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)

DISABLED = [
    "--init",
    "",
    "--distill",
    "",
    "--distill-boot-h",
    "0",
    "--propensity",
    "",
    "--prop-input",
    "",
    "--prop-power",
    "1",
    "--prop-eps",
    "0",
    "--valid-true-mask",
    "-1",
    "--ema-teacher",
    "0",
    "--aux-struct",
    "0",
    "--aux-next",
    "0",
    "--aux-tier",
    "0",
    "--aux-mask",
    "0",
    "--aux-sign",
    "0",
    "--aux-quote",
    "0",
    "--aux",
    "",
    "--aux-weight",
    "0",
    "--swa-frac",
    "0",
    "--fast",
    "--batch",
    "8",
    "--tcn-mix",
    "1",
]


def argv(arm: str, *, profile: bool) -> list[str]:
    name = WORKER.PROFILE_RUN_NAMES[arm] if profile else WORKER.RUN_NAMES[arm]
    command = [
        "scripts/train_rnn.py",
        "--name",
        name,
        "--device",
        "cuda",
        "--arch",
        "tcn",
        "--hidden",
        WORKER.ARM_HIDDEN[arm],
        "--layers",
        "12",
        "--dropout",
        "0",
        "--seed",
        "7",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--split-manifest",
        WORKER.MEMBERSHIP["split_manifest"],
        "--valid-fold",
        "0",
        "--epochs",
        "4",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--no-eval-during-fit",
        "--chunk",
        "2000",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--precision",
        "fp16",
        "--amp-initial-scale",
        "4096",
        *DISABLED,
    ]
    if profile:
        command.extend(["--profile-warmup", "3", "--profile-steps", "20"])
    return command


def branch(arm: str, *, profile: bool) -> dict[str, object]:
    name = WORKER.PROFILE_RUN_NAMES[arm] if profile else WORKER.RUN_NAMES[arm]
    kind = "profile" if profile else "fit"
    row: dict[str, object] = {
        "arm": arm,
        "argv": argv(arm, profile=profile),
        "attempt": f"e529-h217-v5-{arm.replace('_', '-')}-s7-f0-{kind}-b8-amp4096",
        "batch_size": 8,
        "output_path": f"artifacts/runs/{name}",
        "prelaunch_gate": (
            "Recheck destination absent, source hashes, exact data/fold membership, "
            "device identity and current quota/headroom. This command is not admitted."
        ),
        "run_name": name,
        "stage": kind,
    }
    if not profile:
        row["prediction_export"] = {
            "epoch": 3,
            "inference_precision": "fp32",
            "output_name": WORKER.OUTPUT_NAMES[arm],
            "split": "valid",
        }
        row["profile_attempt"] = (
            f"e529-h217-v5-{arm.replace('_', '-')}-s7-f0-profile-b8-amp4096"
        )
    return row


recipe = {
    "architecture": {
        arm: {
            "arch": "tcn",
            "blocks": 12,
            "channels": int(WORKER.ARM_HIDDEN[arm]),
            "initialization": "scratch",
            "input_width": 211,
            "mix": 1,
            "raw_only": False,
            "seed": 7,
        }
        for arm in WORKER.ARMS
    },
    "authorization": {
        "automatic_admission": False,
        "fit": False,
        "profile": False,
        "provider_create": False,
        "push": False,
        "valid_export": False,
    },
    "basis": {
        "all_data_training": "Future work. This packet fits only the sealed fold0 train+valid membership.",
        "owned_worker": (
            "Campaign-owned Vast attempt e529-vast-h217-pair-v4-20261008-a3 finished, "
            "was collected, and was destroyed. This packet does not touch any Vast machine."
        ),
        "question": (
            "Does retaining the existing full causal event features improve the TCN signal, "
            "and does scaling channel width yield useful teacher or blend diversity?"
        ),
        "sealed_project": "h217_packet_v4/project copied byte-for-byte. Trainer and model code are unchanged.",
        "v4_packet_sha256": "38bb1893b946c85850268a95afcf88c40593f06f0f5a956f71eafd599e681656",
        "v4_source_manifest_sha256": "7ea2bf3cfb6fee065148dfb1c4834d15b5cc92058bb4119153b814b6a940a8dc",
    },
    "cohort_execution": {
        "execution_order": [
            "profile tcn_full128 and tcn_full256 independently",
            "fit and valid-only fixed-epoch export each profile-passing arm independently",
        ],
        "failure_isolation": (
            "A failing profile, fit, or export is recorded for its own arm and does not "
            "suppress the other arm. Profile failure blocks only that arm's fit."
        ),
        "max_worker_seconds": 21600,
        "overall_status": (
            "delivered_ungraded if both arms complete; partial_delivery_ungraded if one "
            "completes and the other has a recorded failure or deferral; failed if neither delivers"
        ),
        "phase_caps_seconds": {"fit_each": 7200, "profile_each": 900, "valid_export_each": 1800},
        "setup_timeout_seconds": 450,
        "terminal_packaging_reserve_seconds": 600,
        "unallocated_worker_headroom_seconds": 750,
        "worst_case_declared_phase_sum_seconds": 19800,
    },
    "comparison_design": {
        "evaluation": (
            "Fold0 was previously exposed. Export the valid split only at fixed epoch 3 in FP32. "
            "Root grades the declared fold0 membership with the unchanged official metric. "
            "No train predictions, fresh holdout claim, public forecast, or worker-side score."
        ),
        "not_a_paired_factor": (
            "The delivered raw-only TCN128x12 is an exposed historical reference. "
            "Its quality is independently ungraded. It is not a factor level in this packet."
        ),
        "paired_factor": (
            "Channel width 128 versus 256 under one shared full causal 211-wide default "
            "feature path, seed 7, fold0 membership, and matched training treatment."
        ),
        "training_ancestry": (
            "Both arms train from independent scratch initialization. No teacher weights, "
            "E521 initialization, distillation, propensity weighting, or auxiliary objective."
        ),
    },
    "created_date": "2026-10-08",
    "expected_public_gain_or_prior": None,
    "exports": [
        {
            "argv": [
                "scripts/predict_run.py",
                WORKER.RUN_NAMES[arm],
                "valid",
                "--epoch",
                "3",
                "--output-name",
                WORKER.OUTPUT_NAMES[arm],
                "--inference-precision",
                "fp32",
            ],
            "arm": arm,
            "epoch": 3,
            "inference_precision": "fp32",
            "output_name": WORKER.OUTPUT_NAMES[arm],
            "run_name": WORKER.RUN_NAMES[arm],
            "split": "valid",
        }
        for arm in WORKER.ARMS
    ],
    "feature_view": WORKER.FEATURE_VIEW,
    "fit_branches": {"batch8": [branch(arm, profile=False) for arm in WORKER.ARMS]},
    "historical_raw_only_reference": {
        "admits_v5_fit": False,
        "attempt_id": "e529-vast-h217-pair-v4-20261008-a3",
        "checkpoint_sha256": "5fbb9815b17973dd790cd10ee11af63c1a241ca716df32d44a5b56f099af7dbc",
        "feature_view": "raw-only keeps the 211-wide ABI and zeros event channels",
        "fit_elapsed_seconds": 823.9716792257968,
        "paired_factor": False,
        "prediction_sha256": "d178d84a0bbd74158e5788713973fd7b5c16d5d6f71e6bcc976bce44e883f3aa",
        "quality_graded": False,
        "terminal_status": "partial_delivery_ungraded",
    },
    "hypothesis": (
        "Keeping the existing causal event channels, and training a wider scratch TCN "
        "on that same view, may produce teacher or blend diversity. This packet measures "
        "the two widths. It does not treat the ungraded raw-only run as a paired control."
    ),
    "membership": WORKER.MEMBERSHIP,
    "numerical_contract": {
        "amp_initial_scale": 4096,
        "finite_update_gate": (
            "Attempts, completions, skips, finite loss and unscaled-gradient evidence stay strict. "
            "Exactly 230 optimizer attempts, 230 completed updates, 0 AMP skips and 230 loss checks."
        ),
        "optimizer": "adamw",
        "precision": "fp16",
        "schedule": "onecycle",
        "target_treatment": "unchanged WMSE with chunk normalization",
        "weight_decay": 0.01,
        "learning_rate": 0.001,
    },
    "pending_root_review": {
        "fit_authorized": False,
        "grade_authorized": False,
        "profile_authorized": False,
        "provider_create_authorized": False,
        "push_authorized": False,
        "required_before_any_create_or_fit": [
            "Verify the V5 project tree is byte-identical to h217_packet_v4/project.",
            "Verify exactly two scratch full-feature TCNs, widths 128 and 256, layers 12, mix 1, seed 7, and no GRU or --raw-only arm.",
            "Verify startup source hashes, sealed input hashes, and fold0 membership of 11940 fitted groups with 468 excluded.",
            "Verify the strict 230-update finite profile gate and phase caps 900/7200/1800 inside a 21600-second cohort.",
            "Admit each fit only from that arm's measured profile projection. Historical raw-only 823.97 seconds does not admit V5.",
            "Fresh market, account, owned-instance, quota, headroom, and cost admission. This packet authorizes no create, push, profile, fit, or grade.",
        ],
        "status": "pending_root_review",
    },
    "profile_acceptance": (
        "Require exactly one config line, one PROFILE timing line and one PROFILE_UPDATE_EVIDENCE line. "
        "FP16 AMP initial scale must equal 4096. Attempts must equal completions plus AMP skips, "
        "loss checks must equal attempts, all loss and gradient checks must be finite, and skipped updates must be zero."
    ),
    "profile_gate": (
        "Use PROFILE.step_wall in milliseconds per outer DataLoader batch. The timer encloses the full "
        "internal chunk loop, so do not multiply by ten. Project four epochs as "
        "ceil(11940 / batch_size) * 4 * step_seconds * 1.5 + 600 and reject that arm if the projection "
        "exceeds the 7200-second fit cap."
    ),
    "profile_sequence": [branch(arm, profile=True) for arm in WORKER.ARMS],
    "recipe_id": "E529-H217-TCN-FULLFEATURE-CAPACITY-V5-AMP4096",
    "resource_caps": {
        "arrow_cpu_threads_per_process": 1,
        "arrow_io_threads_per_process": 1,
        "cohort_max_runtime_seconds": 21600,
        "cpu_threads": 6,
        "cuda_allocator_max_bytes": 18000000000,
        "loader_workers": 3,
        "observed_total_vram_max_bytes": 19000000000,
        "phase_caps_seconds": {"fit_each": 7200, "profile_each": 900, "valid_export_each": 1800},
        "prelaunch_condition": (
            "Requires separate root review and fresh owned Vast market, account, quota, and headroom. "
            "This prepared packet authorizes no paid allocation or provider create."
        ),
        "worker_memory_max_bytes": 28000000000,
    },
    "schema_version": 1,
    "scope": (
        "V5 full-feature capacity ladder: scratch TCN C128/L12/mix1 and scratch TCN C256/L12/mix1, "
        "seed 7, sealed fold0, train+valid membership, B8, chunk 2000, four epochs, fixed epoch 3, "
        "FP16 GradScaler initial scale 4096, and the default causal 211-wide feature path. "
        "--raw-only is omitted. No GRU arm and no all-data fit."
    ),
    "source_provenance": {
        "required_sha256": WORKER.REQUIRED_SOURCE_HASHES,
        "sealed_input_sha256": WORKER.SEALED_INPUT_HASHES,
        "trainer_and_model_code": "byte-identical to h217_packet_v4/project; not the root H218 or E540 trainer",
    },
    "status": "PREPARED_AWAITING_ROOT_REVIEW",
    "successor_gates": [
        "Root reviews this exact packet before any provider create.",
        "Recheck current Vast account, owned-worker policy, market, aggregate budget, and provider headroom.",
        "Startup must reproduce sealed source hashes, input hashes, and fold0 membership before a profile.",
        "Each profile, fit, and valid export is arm-local. A measured projection over 7200 seconds blocks only that fit.",
        "Export only fixed epoch-3 valid predictions in FP32. Root grades exposed fold0. The worker records no score.",
    ],
}
destination = PACKET / "recipe.json"
destination.write_text(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
runtime = json.loads((PACKET / "runtime_resources.json").read_text())
WORKER.validate_recipe(recipe, runtime)
print("recipe bytes", destination.stat().st_size)
