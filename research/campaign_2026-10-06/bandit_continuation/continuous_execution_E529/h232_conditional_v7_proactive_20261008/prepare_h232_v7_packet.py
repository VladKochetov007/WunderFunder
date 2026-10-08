from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
PACKET_ID = "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A"
FIT_RUN = "bd26_h232_full_s3_f0_amp4096_v7_proactive_20261008"
PROFILE_RUN = "bd26_h232_full_s3_f0_profile_b8_v7_proactive_20261008"
V6_RECEIPT_SHA256 = "95e5ddbb8027d1f8bbc56326dbde64c7943373bfd4edcb8b17720005f906b503"
WHOLE_WORKER_WALL_CAP_SECONDS = 16_000
MAX_PROJECTED_FIT_SECONDS = 12_000
VALID_EXPORT_CAP_SECONDS = 1_800
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKET))
from cache_pressure_recovery import (  # noqa: E402
    MAX_ADVICE_BYTES,
    MAX_RELIEF_EVENTS,
    MAX_TOTAL_ADVICE_BYTES,
    MIN_RELIEF_INTERVAL_SECONDS,
    PROACTIVE_CGROUP_HEADROOM_FRACTION,
)

PROACTIVE_TRIGGER_FRACTION = PROACTIVE_CGROUP_HEADROOM_FRACTION


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_v6_basis() -> dict[str, str]:
    """Bind the V7 packet to the terminal V6 receipt and unchanged source archive."""
    receipt = ROOT / "basis/V6_TERMINAL_POLL_RECEIPT.json"
    archive = ROOT / "basis/V6_SOURCE_ARCHIVE.tar"
    manifest = ROOT / "basis/V6_SOURCE_ARCHIVE_MANIFEST.json"
    receipt_hash = sha256_bytes(receipt.read_bytes())
    archive_hash = sha256_bytes(archive.read_bytes())
    manifest_hash = sha256_bytes(manifest.read_bytes())
    if receipt_hash != V6_RECEIPT_SHA256:
        raise ValueError("V6 terminal receipt differs from the assigned terminal evidence")
    if archive_hash != "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4":
        raise ValueError("V6 source archive basis hash does not match the reviewed packet")
    if manifest_hash != "6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42":
        raise ValueError("V6 source manifest basis hash does not match the reviewed packet")
    return {
        "terminal_receipt_sha256": receipt_hash,
        "source_archive_sha256": archive_hash,
        "source_archive_manifest_sha256": manifest_hash,
    }


def prepare() -> dict[str, object]:
    basis = verify_v6_basis()
    plan_path = PACKET / "H232_PROFILE_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prior_identity = json.loads(
        (PACKET / "prior_input_identity_v2.json").read_text(encoding="utf-8")
    )
    recipe_path = PACKET / "H232_AMP4096_SUCCESSOR_RECIPE.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["status"] = "same-science-recipe-v7-resource-repair-pending-root-admission"
    recipe["next_gate"] = (
        "Root independently reviews the V7 execution-repair diff and fresh quota. Scientific "
        "recipe fields remain the V6 seed-3 GRU384 B8 AMP4096 treatment."
    )
    recipe["runtime"]["worker_wall_cap_seconds"] = WHOLE_WORKER_WALL_CAP_SECONDS
    recipe["runtime"]["execution_budget_change_from_v6"] = (
        "16000-second whole-worker bound with profile admission reserving the full valid-export "
        "cap; all four fit epochs and fixed epoch-3 checkpoint remain required."
    )
    recipe["execution_repair_parent"] = (
        "E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V6"
    )
    write_json(recipe_path, recipe)
    plan["packet_id"] = PACKET_ID
    plan["status"] = (
        "v7-proactive-recovery-assembled-pending-independent-review-and-fresh-admission"
    )
    plan["next_gate"] = (
        "Root independently reviews the V7 repair diff, source/archive identity, and whole-worker "
        "budget, then writes a fresh admission. This packet does not authorize provider push or "
        "execution."
    )
    plan["fit_run_name"] = FIT_RUN
    plan["profile"]["trace"] = (
        "/kaggle/working/h232_conditional_v7_proactive_20261008/profile_update_trace.json"
    )
    plan["profile"]["run_name"] = PROFILE_RUN
    plan["source_recipe"] = {
        "embedded_member": "H232_AMP4096_SUCCESSOR_RECIPE.json",
        "path": "packet/H232_AMP4096_SUCCESSOR_RECIPE.json",
        "sha256": sha256_bytes((PACKET / "H232_AMP4096_SUCCESSOR_RECIPE.json").read_bytes()),
    }
    plan["source_recipe_sha256"] = plan["source_recipe"]["sha256"]
    plan["source_archive"]["changed_members_vs_H232_PROFILE_V2"] = [
        "scripts/train_rnn.py",
        "src/wnn/data.py",
    ]
    plan["source_archive"]["changed_members_vs_V6"] = []
    plan["source_archive"]["parent_v6_archive_sha256"] = basis["source_archive_sha256"]
    plan["source_archive"]["source_archive_sha256"] = basis["source_archive_sha256"]
    plan["source_archive"]["source_change"] = (
        "V7 retains the byte-identical V6 source archive, including its reviewed trainer "
        "compatibility repair and opt-in rowgroup-read receipt hook. No trainer, data, feature, "
        "target, order, membership, model, objective, optimizer, or AMP source change is made."
    )
    plan["conditional_fit"]["run_name"] = FIT_RUN
    plan["conditional_fit"]["outputs"] = [
        "checkpoint files",
        "provenance.json",
        "history.json",
        "profile_update_trace.json",
        "profile_gate_result.json",
        "phase/resource/hash/cache-pressure receipts",
        "all-valid ep3 FP32 prediction .npy plus verified JSON sidecar",
    ]
    plan["conditional_fit"]["postfit_validation"] = [
        "model.pt and model_ep3.pt exist and match trainer provenance SHA256",
        "selected_epoch is 3",
        "history contains exactly epochs 0..3 with evaluation withheld",
        "trainer args equal profile scientific args except run name and profile-only diagnostics",
        "fit row groups exactly match frozen fold0 train and valid groups",
        "evaluation mask exactly 468 groups and overlap with fit is zero",
        "valid export uses the fixed ep3 checkpoint and explicit FP32 inference",
        "valid export sidecar row_group order exactly matches the frozen valid-target index",
        "export is finite, float32, hash-bound, and evaluated offline by the root grader only",
    ]
    plan["conditional_fit"]["prediction_export"] = {
        "argv": [
            "python",
            "scripts/predict_run.py",
            FIT_RUN,
            "valid",
            "--epoch",
            "3",
            "--output-name",
            "h232_v7_proactive_fixed_ep3",
            "--inference-precision",
            "fp32",
        ],
        "output": "cache/pred_h232_v7_proactive_fixed_ep3_valid.npy",
        "shape": [1_873, 20_000, 2],
        "dtype": "float32",
        "cap_seconds": VALID_EXPORT_CAP_SECONDS,
        "score_or_metric": False,
    }
    plan.pop("resource_contract_v6", None)
    plan["resource_contract_v7"] = {
        "hard_gates_unchanged": {
            "host_and_cgroup_available_bytes_minimum": 6_500_000_000,
            "host_and_cgroup_available_fraction_minimum": 0.25,
        },
        "proactive_cgroup_headroom_trigger_fraction": PROACTIVE_TRIGGER_FRACTION,
        "trigger_policy": (
            "Attempt bounded file-cache relief when cgroup available fraction falls below 28%; "
            "the unchanged 6.5 GB and 25% gates remain the hard stop."
        ),
        "sample": "Same most-constrained cgroup memory.current, memory.max and bounded memory.stat read on every resource guard sample; record anon, file, active_file, inactive_file, shmem and workingset_refault_file.",
        "eligibility": {
            "inactive_file_minimum_bytes": 768_000_000,
            "active_file_to_inactive_file_max_ratio": 4.0,
            "anonymous_growth_maximum_bytes_per_sample": 128_000_000,
            "active_file_growth_maximum_bytes_per_sample": 256_000_000,
            "workingset_refault_file_maximum_bytes_per_sample": 64_000_000,
            "same_cgroup_prior_sample_required": True,
            "host_hard_gates_must_pass": True,
        },
        "action": {
            "method": (
                "POSIX_FADV_DONTNEED on recent compressed byte ranges recorded from the existing "
                "open PyArrow reader; no parquet footer is reopened during recovery"
            ),
            "input_files": [
                "identity-bound read-only datasets/train.parquet",
                "identity-bound read-only datasets/valid.parquet",
            ],
            "maximum_advised_bytes_per_pressure_episode": MAX_ADVICE_BYTES,
            "maximum_pressure_episodes_per_worker": MAX_RELIEF_EVENTS,
            "maximum_total_advice_bytes_per_worker": MAX_TOTAL_ADVICE_BYTES,
            "minimum_seconds_between_pressure_episodes": MIN_RELIEF_INTERVAL_SECONDS,
            "recovery_margin_above_unchanged_hard_gate_bytes": 512_000_000,
            "recheck": (
                "Reread resource snapshot and memory.stat. Keep the original 25% plus 512 MB "
                "recovery target. If a proactive-only attempt misses that target but the "
                "consistent post-advice host and cgroup samples still pass every hard gate, "
                "record the miss and continue. If hard pressure existed at trigger time, the "
                "full recovery target is required; otherwise fail closed."
            ),
            "fallback": (
                "At proactive-only pressure, eligibility veto, missing trace, unsupported advice, "
                "budget exhaustion, or a missed recovery margin does not stop a fit while the "
                "unchanged hard gates pass. Existing hard pressure still fails closed unless the "
                "full 512 MB recovery target is restored."
            ),
            "global_drop_caches_or_memory_reclaim_write": False,
        },
        "receipts": [
            "resource.json",
            "resource_abort.json",
            "memory_stat_phases.json",
            "rowgroup_reads.tsv",
            "cache_pressure_relief.jsonl",
        ],
        "advice_is_a_kernel_hint": True,
    }
    raw_sizes = {
        split: int(prior_identity["expected"][f"datasets/{split}.parquet"]["bytes"])
        for split in ("train", "valid")
    }
    fit_epochs = int(plan["scientific_shape"]["epochs"])
    raw_pair_bytes = sum(raw_sizes.values())
    # Bound four full fit passes, one additional full pass for setup/profile conservatism,
    # the one fallback identity-hash pass, and one full valid-export pass. The actual profile
    # is capped at 23 batches and the hash path already streams POSIX_FADV_DONTNEED advice.
    conservative_passes = fit_epochs + 2
    covered_read_bytes = conservative_passes * raw_pair_bytes + raw_sizes["valid"]
    bytes_per_episode = MAX_ADVICE_BYTES
    minimum_event_count_with_10pct_margin = math.ceil(covered_read_bytes * 1.10 / bytes_per_episode)
    if MAX_RELIEF_EVENTS * bytes_per_episode > MAX_TOTAL_ADVICE_BYTES:
        raise ValueError("cumulative relief byte cap is inconsistent with event cap")
    if minimum_event_count_with_10pct_margin > MAX_RELIEF_EVENTS:
        raise ValueError("cache-relief event cap does not cover the frozen read-horizon bound")
    plan["resource_contract_v7"]["budget_derivation"] = {
        "train_raw_file_bytes": raw_sizes["train"],
        "valid_raw_file_bytes": raw_sizes["valid"],
        "fit_epochs": fit_epochs,
        "full_raw_pair_bytes": raw_pair_bytes,
        "fit_full_raw_pass_ceiling_bytes": fit_epochs * raw_pair_bytes,
        "additional_full_raw_passes_for_setup_and_fallback_hash": 2,
        "setup_and_fallback_hash_ceiling_bytes": 2 * raw_pair_bytes,
        "valid_prediction_export_full_pass_ceiling_bytes": raw_sizes["valid"],
        "conservative_total_read_ceiling_bytes": covered_read_bytes,
        "minimum_events_with_10pct_margin": minimum_event_count_with_10pct_margin,
        "maximum_events": MAX_RELIEF_EVENTS,
        "maximum_bytes_per_event": bytes_per_episode,
        "maximum_cumulative_advice_bytes": MAX_TOTAL_ADVICE_BYTES,
        "cumulative_reserve_over_covered_ceiling_bytes": (
            MAX_TOTAL_ADVICE_BYTES - covered_read_bytes
        ),
        "minimum_seconds_between_events": MIN_RELIEF_INTERVAL_SECONDS,
        "maximum_advice_rate_bytes_per_second_if_every_event_hits_cap": (
            bytes_per_episode / MIN_RELIEF_INTERVAL_SECONDS
        ),
        "maximum_worker_lifetime_seconds": WHOLE_WORKER_WALL_CAP_SECONDS,
        "coverage_assumption": (
            "Each fit epoch reads no more than one complete pass of both pinned raw files; one extra "
            "complete pair pass bounds profile/setup exposure even though profile is capped at 23 batches; "
            "one more pair pass bounds identity-hash fallback; prediction export reads no more than one "
            "complete valid file. This sizes a worst-case advice budget, not a prediction of cache eviction."
        ),
        "hash_path_note": (
            "The raw identity fallback independently streams through its existing bounded "
            "POSIX_FADV_DONTNEED path; one full pair pass is nevertheless included in this reserve."
        ),
        "rate_and_failure_policy": (
            f"Reserve at most one {MAX_ADVICE_BYTES} byte advice episode per "
            f"{MIN_RELIEF_INTERVAL_SECONDS} seconds and {MAX_RELIEF_EVENTS} total; the existing "
            "memory.stat eligibility and post-advice gate-plus-512 MB recheck apply to every event. "
            "A proactive-only margin miss is recorded and may continue only while the unchanged "
            "hard gates pass; hard-pressure recovery remains fail closed."
        ),
    }
    plan.pop("current_v4_deliverable", None)
    plan["v7_execution_budget"] = {
        "whole_worker_wall_cap_seconds": WHOLE_WORKER_WALL_CAP_SECONDS,
        "maximum_projected_fit_seconds": MAX_PROJECTED_FIT_SECONDS,
        "fit_epochs": 4,
        "fixed_checkpoint_epoch_zero_based": 3,
        "valid_export_cap_seconds": VALID_EXPORT_CAP_SECONDS,
        "duration_cuts_training": False,
        "basis": (
            "V6 reached epoch 0 batch 1060/1492 at 1452.987 seconds. Its measured full-fit "
            "projection is about 8200 seconds before startup, profile, identity, and valid export. "
            "The profile gate admits only a full four-epoch fit plus the full export cap inside "
            "the 16000-second worker budget."
        ),
    }
    plan["prior_attempt_diagnosis"] = (
        "V6 passed the actual-device profile and reached epoch0 batch1060/1492 with finite "
        "updates and zero AMP skips. Its last eligible 768 MB advice raised cgroup headroom "
        "from 8.0364 GB to 8.4675 GB; the unchanged 25% plus 512 MB target was 8.5651 GB, "
        "a 97.5 MB shortfall. This was resource-censored with no checkpoint, prediction, or score."
    )
    plan["smallest_change_needed"] = (
        "Root independently reviews the V7 repair diff, source/archive and packet hashes, confirms "
        "fresh cloud headroom, and separately admits any provider run. The valid prediction remains "
        "an ungraded artifact for the root-owned official metric check."
    )
    plan["profile"]["lifetime_projection"].update(
        {
            "p50_fit_max_seconds": MAX_PROJECTED_FIT_SECONDS,
            "p90_worker_total_max_seconds": WHOLE_WORKER_WALL_CAP_SECONDS,
            "valid_export_cap_seconds": VALID_EXPORT_CAP_SECONDS,
        }
    )
    plan["profile"]["required_timed_conditions"] = [
        "profile input width 211" if item == "profile input width 160" else item
        for item in plan["profile"]["required_timed_conditions"]
    ]
    plan["conditional_fit"]["start_condition"] = (
        "same-worker profile gate status pass, exact V7 root review receipt and fresh quota "
        "resource admission, input identity and stat binding unchanged"
    )
    plan["screening_followup"]["current_v4_deliverable"] = (
        "This V7 packet owns one full four-epoch seed-3 fit only after the actual-device profile "
        "gate and fresh root admission pass. It exports the fixed ep3 FP32 valid prediction without "
        "scoring; it is not a candidate/control screen or scientific result by itself."
    )
    plan["screening_followup"]["smallest_change_needed"] = (
        "Root reviews the exact V7 resource-recovery and time-budget diff, refreshes quota, and "
        "admits one identified provider run. After collection, root owns offline grading and the "
        "fit remains exposed validation evidence."
    )
    plan["scope"] = (
        "Same frozen V6 seed-3 full-feature GRU384x2 scientific treatment, with a proactive "
        "bounded cache-recovery trigger and an explicit whole-worker time budget. No scientific "
        "recipe field changes from V6."
    )
    plan["source_archive"].update(
        {
            "sha256": basis["source_archive_sha256"],
            "manifest_sha256": basis["source_archive_manifest_sha256"],
            "parent_v6_archive_sha256": basis["source_archive_sha256"],
            "changed_members_vs_V6": [],
        }
    )
    plan["timing_recording"]["valid_prediction_export"] = (
        "Separate fixed ep3 FP32 subprocess with argv, exit, source/checkpoint/input/row-order/array hashes, "
        "resource snapshots and an 1800-second cap; no metric is computed in the worker."
    )
    plan["timing_recording"]["prediction_export"] = (
        "Included as a fixed ep3 FP32 validation-only export after a complete four-epoch fit; no "
        "metric is computed in the worker."
    )
    plan["actual_cloud_resources"]["wall_cap_seconds"] = WHOLE_WORKER_WALL_CAP_SECONDS
    plan["execution_repair_parent"] = (
        "E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V6"
    )
    plan["v6_basis_hashes"] = basis
    write_json(plan_path, plan)

    review = {
        "schema": "h232-independent-review-v1",
        "experiment_id": PACKET_ID,
        "scope": "conditional-profile-gated-fit",
        "status": "pending_independent_review",
        "conditional_fit_authorized": False,
        "fit_admission": "not_admitted",
        "provider_push_authorized": False,
        "reviewer": None,
        "reviewed_hashes": {},
        "note": "No V6 approval transfers to this V7 attempt. Root must independently review the exact V7 diff and replace this placeholder before any provider push.",
    }
    review_path = PACKET / "external_review.json"
    write_json(review_path, review)

    manifest = json.loads((PACKET / "source_archive_manifest.json").read_text(encoding="utf-8"))
    archive_bytes = (PACKET / "source_archive.tar").read_bytes()
    source_manifest_bytes = (PACKET / "source_archive_manifest.json").read_bytes()
    trainer = next(row for row in manifest["members"] if row["path"] == "scripts/train_rnn.py")
    run_identity = next(
        row for row in manifest["members"] if row["path"] == "src/wnn/run_identity.py"
    )
    recipe_bytes = (PACKET / "H232_AMP4096_SUCCESSOR_RECIPE.json").read_bytes()
    prior_bytes = (PACKET / "prior_input_identity_v2.json").read_bytes()
    plan_bytes = (PACKET / "H232_PROFILE_PLAN.json").read_bytes()
    helper_hashes = {
        name: sha256_bytes((PACKET / name).read_bytes())
        for name in (
            "profile_worker.py",
            "profile_kernel.py",
            "gpu_memory_domains.py",
            "verified_digest_handoff.py",
            "cache_pressure_recovery.py",
            "prior_input_identity_v2.json",
            "H232_PROFILE_PLAN.json",
            "H232_AMP4096_SUCCESSOR_RECIPE.json",
            "source_archive_manifest.json",
            "source_archive.tar",
        )
    }
    meta = {
        "schema": "h232-kaggle-conditional-packet-metadata-v1",
        "packet_id": PACKET_ID,
        "status": "assembled_pending_independent_review_and_fresh_admission",
        "fit_enabled": True,
        "fit_is_conditional_on_gate": True,
        "profile_only": False,
        "scoring_enabled": False,
        "grading_enabled": False,
        "prediction_export_enabled": True,
        "profile_and_fit_amp_initial_scale": 4096.0,
        "historical_fit_amp_initial_scale": 65536.0,
        "scientific_numeric_successor": False,
        "execution_resource_repair": True,
        "parent_packet_id": "E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V6",
        "v6_basis_hashes": basis,
        "source_recipe_sha256": sha256_bytes(recipe_bytes),
        "prior_input_identity_sha256": sha256_bytes(prior_bytes),
        "source_archive_sha256": sha256_bytes(archive_bytes),
        "source_archive_manifest_sha256": sha256_bytes(source_manifest_bytes),
        "trainer_sha256": trainer["sha256"],
        "run_identity_sha256": run_identity["sha256"],
        "external_review_sha256": sha256_bytes(review_path.read_bytes()),
        "reviewed_hashes": {},
        "current_packet_hashes": helper_hashes,
        "provider_push_authorized": False,
        "timing_receipts": [
            "phase.json",
            "input_identity.json",
            "hash_io_advice.json",
            "memory_stat_phases.json",
            "cache_pressure_relief.jsonl",
            "rowgroup_reads.tsv",
            "profile_update_trace.json",
            "profile_gate_result.json",
            "fit_result.json",
            "valid_prediction_export.json",
            "gpu_telemetry.json",
            "gpu_telemetry_fit.json",
        ],
        "plan_sha256": sha256_bytes(plan_bytes),
    }
    write_json(PACKET / "packet_meta.json", meta)
    kernel_metadata = {
        "code_file": "h232_profile_kernel_bundle.py",
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "id": "vladkochetov07/h232-gru384-b8-amp4096-v7-proactive",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": "H232 GRU384 B8 AMP4096 V7 Proactive Cache Recovery",
    }
    write_json(PACKET / "kernel-metadata.json", kernel_metadata)
    return {
        "packet_id": PACKET_ID,
        "packet_meta_sha256": sha256_bytes((PACKET / "packet_meta.json").read_bytes()),
        "external_review_sha256": meta["external_review_sha256"],
        "reviewed_hashes": {},
        "provider_push_authorized": False,
        "fit_run_name": FIT_RUN,
        "profile_run_name": PROFILE_RUN,
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
