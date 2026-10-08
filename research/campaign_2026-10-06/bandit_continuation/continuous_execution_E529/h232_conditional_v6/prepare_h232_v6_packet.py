from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
PACKET_ID = "E529_H232_FULLFEATURE_GRU384_B8_AMP4096_CONDITIONAL_PROFILE_FIT_V6"
FIT_RUN = "bd26_h232_full_s3_f0_amp4096_v6"
PROFILE_RUN = "bd26_h232_full_s3_f0_profile_b8_v6"
sys.path.insert(0, str(PACKET))
from cache_pressure_recovery import (  # noqa: E402
    MAX_ADVICE_BYTES,
    MAX_RELIEF_EVENTS,
    MAX_TOTAL_ADVICE_BYTES,
    MIN_RELIEF_INTERVAL_SECONDS,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare() -> dict[str, object]:
    plan_path = PACKET / "H232_PROFILE_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prior_identity = json.loads(
        (PACKET / "prior_input_identity_v2.json").read_text(encoding="utf-8")
    )
    plan["packet_id"] = PACKET_ID
    plan["status"] = "v6-resource-repair-assembled-pending-independent-review-and-fresh-admission"
    plan["next_gate"] = (
        "Independent review must bind every V6 executable, source archive, cache-pressure "
        "test and post-fit valid-export contract. Provider push and execution require a fresh "
        "root admission; this packet does not authorize a launch."
    )
    plan["fit_run_name"] = FIT_RUN
    plan["profile"]["trace"] = "/kaggle/working/h232_conditional_v6/profile_update_trace.json"
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
    plan["source_archive"]["source_change"] = (
        "V5's reviewed trainer-only compatibility repair is retained. V6 adds only an opt-in "
        "rowgroup-read and compressed byte-range receipt hook in data.py using the existing "
        "PyArrow reader for verified raw train/valid files; feature values, targets, order, "
        "membership, model, objective, optimizer and AMP policy are unchanged."
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
            "h232_v6_fixed_ep3",
            "--inference-precision",
            "fp32",
        ],
        "output": "cache/pred_h232_v6_fixed_ep3_valid.npy",
        "shape": [1_873, 20_000, 2],
        "dtype": "float32",
        "cap_seconds": 1_800,
        "score_or_metric": False,
    }
    plan["resource_contract_v6"] = {
        "hard_gates_unchanged": {
            "host_and_cgroup_available_bytes_minimum": 6_500_000_000,
            "host_and_cgroup_available_fraction_minimum": 0.25,
        },
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
            "recheck": "Reread resource snapshot and memory.stat; continue only if host gates pass and cgroup headroom exceeds the original byte/fraction gates plus the recovery margin.",
            "fallback": "Unsupported advice, missing trace, changed source identity, inconsistent snapshot, refault/anon/active-file veto, or insufficient recovery preserves the original abort.",
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
    plan["resource_contract_v6"]["budget_derivation"] = {
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
        "maximum_worker_lifetime_seconds": 34_000,
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
            "Any event/rate/byte cap or failed recheck preserves the original hard abort."
        ),
    }
    plan["current_v4_deliverable"] = (
        "V5 passed the profile and started fitting, then was resource-censored at epoch0 batch660/1492. "
        "V6 preserves that frozen H232 fit treatment and adds bounded raw-input cache-pressure recovery "
        "plus a fixed-epoch valid export; neither V5 nor V6 has produced a graded score here."
    )
    plan["smallest_change_needed"] = (
        "Root independently reviews the exact V6 source/worker/test hashes, confirms fresh cloud headroom, "
        "and separately admits any provider run. The valid prediction is an ungraded artifact for the "
        "root-owned official metric check."
    )
    plan["timing_recording"]["valid_prediction_export"] = (
        "Separate fixed ep3 FP32 subprocess with argv, exit, source/checkpoint/input/row-order/array hashes, "
        "resource snapshots and an 1800-second cap; no metric is computed in the worker."
    )
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
        "note": "V5 approval does not transfer to V6. Root must replace this placeholder with a separately reviewed receipt before launch.",
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
        "scientific_numeric_successor": True,
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
        "id": "vladkochetov07/h232-full-feature-gru384-b8-amp4096-conditional-profile-fit-v6",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": "H232 GRU384 B8 AMP4096 Conditional Profile Fit V6",
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
