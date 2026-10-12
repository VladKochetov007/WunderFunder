from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import pathlib
import subprocess
import sys
from collections.abc import Mapping


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_sha256(value: object) -> str:
    # The frozen producer stores heterogeneous PyTorch objects in this state.
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def systemd_properties(unit: str) -> dict[str, str]:
    names = (
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainStatus",
        "ExecMainCode",
        "MemoryCurrent",
        "MemoryPeak",
        "CPUQuotaPerSecUSec",
    )
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, *[f"--property={name}" for name in names]],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        key: value
        for line in result.stdout.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def check(checks: list[dict[str, object]], name: str, condition: bool, observed: object) -> None:
    checks.append({"name": name, "passed": bool(condition), "observed": observed})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the actual seed4 P2 producer state.")
    parser.add_argument("--state", type=pathlib.Path, required=True)
    parser.add_argument("--provenance", type=pathlib.Path, required=True)
    parser.add_argument("--attempt", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--root-release", type=pathlib.Path, required=True)
    parser.add_argument("--memory-margin", type=pathlib.Path, required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--prelaunch-parent-current-bytes", type=int, required=True)
    parser.add_argument("--prelaunch-observed-utc", required=True)
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema": "wnn.clean-t3-control-seed4-p2-terminal-validation.r1",
        "scope": "P2 terminal state/provenance validation only; no score or grade.",
        "recorded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "status": "FAIL",
        "checks": checks,
        "validator": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "state_loader_source": str(args.source_root / "wnn" / "training_state.py"),
            "operation": "wnn.training_state.load_training_state(path) from frozen source snapshot",
        },
    }

    try:
        sys.path.insert(0, str(args.source_root.resolve()))
        from wnn.run_identity import group_identity
        from wnn.training_state import load_training_state

        attempt = json.loads(args.attempt.read_text(encoding="utf-8"))
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        release = json.loads(args.root_release.read_text(encoding="utf-8"))
        margin = json.loads(args.memory_margin.read_text(encoding="utf-8"))
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        state = load_training_state(args.state)
        state_file_sha = sha256_file(args.state)

        parent = contract["source"]["parent_source"]
        state_contract = state["contract"]
        options = state_contract["options"]
        progress = state["progress"]
        loader = state["loader"]
        cache = provenance["distillation_cache"]
        fit_groups = provenance["fit_row_groups"]
        actual_group_ids = {
            split: group_identity(fit_groups[split]) for split in ("train", "valid")
        }
        valid_eval_overlap = sorted(
            set(fit_groups["valid"]) & set(provenance["evaluation_row_groups"])
        )

        check(
            checks,
            "attempt_terminal_finished_exit0",
            attempt.get("status") == "finished"
            and attempt.get("phase") == "terminal"
            and attempt.get("exit_status") == 0,
            {key: attempt.get(key) for key in ("status", "phase", "exit_status", "finished_utc")},
        )
        check(
            checks,
            "root_release_authorized_and_pinned",
            release.get("authorized") is True
            and release.get("contract_sha256") == sha256_file(args.contract),
            {
                "authorized": release.get("authorized"),
                "contract_sha256": release.get("contract_sha256"),
            },
        )
        check(
            checks,
            "memory_margin_rule_documented",
            margin.get("working_admission_target_bytes") == 17_000_000_000
            and margin.get("unchanged_hard_limit_bytes") == 20_000_000_000,
            {
                "working_admission_target_bytes": margin.get("working_admission_target_bytes"),
                "unchanged_hard_limit_bytes": margin.get("unchanged_hard_limit_bytes"),
            },
        )
        check(
            checks,
            "state_loader_full_epoch_format",
            state.get("format") == "wnn.epoch-training-state"
            and state.get("version") == 1
            and state.get("boundary") == "epoch",
            {
                "format": state.get("format"),
                "version": state.get("version"),
                "boundary": state.get("boundary"),
            },
        )
        check(checks, "next_epoch_4", state.get("next_epoch") == 4, state.get("next_epoch"))
        check(
            checks,
            "fixed_six_epoch_full_schedule",
            state_contract.get("epochs") == 6
            and state_contract.get("total_steps") == 44_760
            and options.get("epochs") == 6,
            {
                "epochs": state_contract.get("epochs"),
                "total_steps": state_contract.get("total_steps"),
                "options_epochs": options.get("epochs"),
            },
        )
        check(
            checks,
            "recipe_seed4_h256_adamw_fp16_alpha09",
            options.get("seed") == 4
            and options.get("hidden") == 256
            and options.get("layers") == 2
            and options.get("optim") == "adamw"
            and options.get("precision") == "fp16"
            and options.get("distill_alpha") == 0.9,
            {
                key: options.get(key)
                for key in ("seed", "hidden", "layers", "optim", "precision", "distill_alpha")
            },
        )
        check(
            checks,
            "trainer_source_pin",
            state_contract.get("sources", {}).get("trainer")
            == contract["source"]["trainer_sha256"],
            {
                "state_trainer_sha256": state_contract.get("sources", {}).get("trainer"),
                "pinned_trainer_sha256": contract["source"]["trainer_sha256"],
            },
        )
        check(
            checks,
            "actual_provenance_parent_and_split",
            provenance.get("initial_checkpoint_sha256") == parent["checkpoint_sha256"]
            and provenance.get("split_manifest_sha256") == parent["split_manifest_sha256"]
            and provenance.get("seed") == 4,
            {
                "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
                "split_manifest_sha256": provenance.get("split_manifest_sha256"),
                "seed": provenance.get("seed"),
            },
        )
        check(
            checks,
            "actual_fit_membership_pins",
            actual_group_ids["train"].get("count") == 10_535
            and actual_group_ids["train"].get("ordered_int64_le_sha256")
            == parent["fit_train_order_sha256"]
            and actual_group_ids["valid"].get("count") == 1_405
            and actual_group_ids["valid"].get("ordered_int64_le_sha256")
            == parent["fit_valid_order_sha256"]
            and len(provenance["evaluation_row_groups"]) == 468
            and not valid_eval_overlap
            and state_contract.get("fit_membership") == provenance.get("fit_membership_identity"),
            {
                "train": actual_group_ids["train"],
                "fit_valid": actual_group_ids["valid"],
                "evaluation_groups": len(provenance["evaluation_row_groups"]),
                "fit_valid_eval_overlap": len(valid_eval_overlap),
                "state_contract_matches_provenance": state_contract.get("fit_membership")
                == provenance.get("fit_membership_identity"),
            },
        )
        expected_parents = {
            ("clean_t3_f0_muonall_h384_e3_s3_r5", 2, 0.1),
            ("clean_t3_f0_muonall_h384_e3_s4_r5", 2, 0.1),
            ("clean_t3_f0_muonall_h384_e3_s5_r5", 2, 0.1),
        }
        actual_parents = {
            (item["run"], item["epoch"], item["coefficient"])
            for item in cache["source_coefficients"]
        }
        check(
            checks,
            "actual_teacher_cache_ancestry",
            expected_parents == actual_parents
            and cache.get("metadata_sha256") == contract["inputs"]["cache_metadata_sha256"]
            and cache.get("variant") == "homogeneous"
            and cache.get("epoch_convention")
            == {
                "student_epochs": 6,
                "student_checkpoint_policy": "fixed",
                "student_checkpoint_epoch": 5,
            },
            {
                "teacher_sources": sorted(actual_parents),
                "metadata_sha256": cache.get("metadata_sha256"),
                "variant": cache.get("variant"),
                "epoch_convention": cache.get("epoch_convention"),
            },
        )
        check(
            checks,
            "loader_and_full_state_sections",
            loader.get("batch_size") == 16
            and loader.get("num_workers") == 0
            and loader.get("dataset_length") == 11_940
            and all(
                isinstance(state.get(key), Mapping)
                for key in (
                    "model",
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "auxiliary",
                    "progress",
                    "rng",
                )
            ),
            {
                "loader": loader,
                "sections": {
                    key: isinstance(state.get(key), Mapping)
                    for key in (
                        "model",
                        "optimizer",
                        "scheduler",
                        "scaler",
                        "auxiliary",
                        "progress",
                        "rng",
                    )
                },
            },
        )
        check(
            checks,
            "actual_checkpoint_matches_provenance_model_ep3",
            isinstance(provenance.get("checkpoint_sha256"), Mapping)
            and provenance["checkpoint_sha256"].get("model_ep3.pt")
            == sha256_file(args.state.parent / "model_ep3.pt"),
            {
                "provenance_checkpoint_sha256": provenance.get("checkpoint_sha256"),
                "model_ep3_sha256": sha256_file(args.state.parent / "model_ep3.pt"),
            },
        )

        buffer_hashes = {
            key: component_sha256(state[key])
            for key in (
                "model",
                "optimizer",
                "scheduler",
                "scaler",
                "ema",
                "swa",
                "loader",
                "auxiliary",
                "progress",
                "rng",
            )
        }
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]
        scaler = state["scaler"]
        rng = state["rng"]
        aux = state["auxiliary"]
        receipt.update(
            {
                "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
                "attempt": {
                    "path": str(args.attempt),
                    "sha256": sha256_file(args.attempt),
                    "id": attempt.get("id"),
                    "unit": attempt.get("unit"),
                    "started_utc": attempt.get("started_utc"),
                    "finished_utc": attempt.get("finished_utc"),
                    "elapsed_seconds": attempt.get("elapsed_seconds"),
                    "exit_status": attempt.get("exit_status"),
                    "peak_ram_bytes": attempt.get("peak_ram_bytes"),
                    "peak_observed_vram_bytes": attempt.get("peak_observed_vram_bytes"),
                    "memory_event_deltas": attempt.get("final_memory_event_deltas"),
                    "parent_local_memory_event_deltas": attempt.get(
                        "final_parent_memory_event_deltas"
                    ),
                    "parent_hierarchical_memory_event_deltas": attempt.get(
                        "final_parent_hierarchical_memory_event_deltas"
                    ),
                },
                "inputs": {
                    "state_path": str(args.state),
                    "state_bytes": args.state.stat().st_size,
                    "state_sha256": state_file_sha,
                    "provenance_path": str(args.provenance),
                    "provenance_sha256": sha256_file(args.provenance),
                    "contract_path": str(args.contract),
                    "contract_sha256": sha256_file(args.contract),
                    "root_release_path": str(args.root_release),
                    "root_release_sha256": sha256_file(args.root_release),
                    "memory_margin_path": str(args.memory_margin),
                    "memory_margin_sha256": sha256_file(args.memory_margin),
                    "p2_model_ep3_sha256": sha256_file(args.state.parent / "model_ep3.pt"),
                },
                "state": {
                    "format": state.get("format"),
                    "version": state.get("version"),
                    "boundary": state.get("boundary"),
                    "next_epoch": state.get("next_epoch"),
                    "contract_sha256": hashlib.sha256(
                        json.dumps(state_contract, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "model_tensor_count": len(state["model"]),
                    "optimizer_parameter_group_count": len(optimizer["param_groups"]),
                    "optimizer_state_parameter_count": len(optimizer["state"]),
                    "scheduler": {
                        key: scheduler.get(key)
                        for key in ("_step_count", "last_epoch", "_total_steps")
                    },
                    "scaler": {
                        key: scaler.get(key)
                        for key in ("scale", "_growth_tracker", "growth_interval")
                    },
                    "loader": loader,
                    "progress": {
                        key: progress.get(key)
                        for key in ("step_count", "cumulative_updates", "fitting_seconds")
                    },
                    "auxiliary_keys": sorted(aux),
                    "rng_keys": sorted(rng),
                    "cuda_rng_states": len(rng.get("cuda", [])),
                    "component_sha256": buffer_hashes,
                    "component_sha256_method": "torch.save(component, BytesIO()) then SHA-256; whole-file SHA is authoritative.",
                },
                "actual_provenance": {
                    "seed": provenance.get("seed"),
                    "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
                    "split_manifest_sha256": provenance.get("split_manifest_sha256"),
                    "fit_membership_identity": provenance.get("fit_membership_identity"),
                    "evaluation_group_count": len(provenance["evaluation_row_groups"]),
                    "fit_valid_eval_overlap": len(valid_eval_overlap),
                    "distillation_cache": cache,
                },
                "fresh_support_admission": {
                    "captured_utc": args.prelaunch_observed_utc,
                    "parent_memory_current_bytes_before_support": args.prelaunch_parent_current_bytes,
                    "support_memory_bound_bytes": 1_500_000_000,
                    "sum_bytes": args.prelaunch_parent_current_bytes + 1_500_000_000,
                    "working_target_bytes": margin["working_admission_target_bytes"],
                    "within_target": args.prelaunch_parent_current_bytes + 1_500_000_000
                    <= margin["working_admission_target_bytes"],
                    "worker_reservation_active": False,
                    "support_cpu_cores": 1,
                    "support_timeout_seconds": 120,
                },
                "no_scientific_grade": True,
            }
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ImportError,
        subprocess.CalledProcessError,
    ) as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        receipt["status"] = "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "checks": checks,
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
