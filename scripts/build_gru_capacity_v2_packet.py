"""Build the frozen Kaggle packet for the seed-9 Muon capacity comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
V1_PACKET = CAMPAIGN / "gru512_768_capacity_v1_20261008/packet"
OUT = CAMPAIGN / "gru512_768_muon_capacity_v2_20261008/packet"
H232_PACKET = CAMPAIGN / "h232_conditional_v7b_reviewed_20261008/packet"
PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_CAPACITY_V2_20261008"
ORTHOGONALIZATION_POLICY = "native_bf16_if_supported_else_fp32_v1"
ABSENT_NONRUNTIME_SOURCE_ALLOWLIST = {
    "src/wnn/conditional_readout.py": (
        "present in the archived V1 manifest but absent from this frozen source branch; "
        "no imports or references occur in the included runtime source"
    ),
    "src/wnn/online_dynamics.py": (
        "present in the archived V1 manifest but absent from this frozen source branch; "
        "no imports or references occur in the included runtime source"
    ),
}
ADDITIONAL_RUNTIME_DEPENDENCIES = {
    "src/wnn/numeric_ple.py": {
        "sha256": "229a7e0f032cd65bbf6700fa9372c08016b20e4719cb2f9eddd540ff8c468720",
        "source_main_commit": "83b97650270c8bb8fdbbe5715f659b59bd20316c",
        "reason": "imported by scripts/train_rnn.py and src/wnn/nn.py at module load",
    }
}
RAW_INPUTS = {
    "datasets/train.parquet": {
        "bytes": 29_240_219_011,
        "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    },
    "datasets/valid.parquet": {
        "bytes": 5_615_509_128,
        "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    },
    "cache/chain_folds.npz": {
        "bytes": 550_072,
        "sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    },
    "cache/chains.npz": {
        "bytes": 650_142,
        "sha256": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
    },
    "cache/train_targets.npz": {
        "bytes": 1_909_335_790,
        "sha256": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    },
    "cache/valid_targets.npz": {
        "bytes": 337_186_198,
        "sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    },
}


def sha256_file(path: Path) -> str:
    """Hash one bounded source or packet file without loading it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic, reviewable JSON text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def copy_binding(source: Path, target: Path, expected_sha256: str) -> dict[str, object]:
    """Copy an already frozen input manifest and verify its exact bytes."""
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"frozen input manifest changed: {source.name}")
    target.write_bytes(source.read_bytes())
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": actual}


def source_members() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Rebuild runtime closure and record reviewed absent optional modules."""
    prior = json.loads((V1_PACKET / "source/source_archive_manifest.json").read_text())
    paths = [PurePosixPath(item["path"]) for item in prior["members"]]
    paths.extend(PurePosixPath(path) for path in ADDITIONAL_RUNTIME_DEPENDENCIES)
    members = []
    omitted = []
    for relative in sorted(paths):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source member: {relative}")
        path = ROOT / Path(*relative.parts)
        if path.is_symlink() or not path.is_file():
            reason = ABSENT_NONRUNTIME_SOURCE_ALLOWLIST.get(relative.as_posix())
            if reason is not None and not path.exists():
                omitted.append({"path": relative.as_posix(), "reason": reason})
                continue
            raise FileNotFoundError(path)
        addition = ADDITIONAL_RUNTIME_DEPENDENCIES.get(relative.as_posix())
        if addition is not None and sha256_file(path) != addition["sha256"]:
            raise ValueError(f"frozen runtime dependency bytes changed: {relative.as_posix()}")
        members.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for omission in omitted:
        module_stem = Path(omission["path"]).stem
        if any(
            module_stem in (ROOT / str(member["path"])).read_text(encoding="utf-8")
            for member in members
            if str(member["path"]).endswith(".py")
        ):
            raise ValueError(f"omitted source module is referenced by included code: {module_stem}")
    if len({str(member["path"]) for member in members}) != len(members):
        raise ValueError("source closure has duplicate members after adding runtime dependencies")
    return members, omitted


def write_source_archive(members: list[dict[str, object]]) -> tuple[str, str]:
    """Create a reproducible regular-file-only source archive and its manifest."""
    source_dir = OUT / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / "source_archive.tar"
    with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for entry in members:
            relative = PurePosixPath(str(entry["path"]))
            path = ROOT / Path(*relative.parts)
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    manifest_path = source_dir / "source_archive_manifest.json"
    manifest = {
        "schema": "wnn-gru-capacity-source-manifest-v1",
        "member_count": len(members),
        "members": members,
    }
    write_json(manifest_path, manifest)
    return sha256_file(archive_path), sha256_file(manifest_path)


def run_git_commit() -> str:
    """Capture the isolated source snapshot commit used by this packet."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((V1_PACKET / "portable_inputs.json").read_text())
    if set(source_manifest["sources"]) != set(RAW_INPUTS):
        raise ValueError("V1 portable input closure differs from the frozen six-source identity")
    for name, expected in RAW_INPUTS.items():
        entry = source_manifest["sources"][name]
        if entry.get("sha256") != expected["sha256"]:
            raise ValueError(f"portable source identity changed: {name}")
        identity = entry.get("original_identity", {})
        if identity.get("bytes") != expected["bytes"]:
            raise ValueError(f"portable source size changed: {name}")

    fold_binding = copy_binding(
        V1_PACKET / "fold0.json",
        OUT / "fold0.json",
        "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    )
    portable_binding = copy_binding(
        V1_PACKET / "portable_inputs.json",
        OUT / "portable_inputs.json",
        "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
    )
    prior_identity_binding = copy_binding(
        H232_PACKET / "prior_input_identity_v2.json",
        OUT / "prior_input_identity_v2.json",
        "cbea42868303968f7ee84105990a25848ce17add175b7161e116c80dfa63e46e",
    )

    members, omitted_members = source_members()
    member_hashes = {str(member["path"]): str(member["sha256"]) for member in members}
    archive_sha, source_manifest_sha = write_source_archive(members)
    source_manifest_path = OUT / "source/source_archive_manifest.json"
    source_manifest_bytes = source_manifest_path.stat().st_size

    kernel_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "id": "vladkochetov07/gru512-768-muon-cap-v2-20261008",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": "GRU512/768 Muon Capacity Seed9 V2",
    }
    write_json(OUT / "kernel-metadata.json", kernel_metadata)
    (OUT / "README.md").write_text(
        "# GRU512/768 Muon capacity packet V2\n\n"
        "This is a private, source-prepared Kaggle cohort for a bounded clean-fold "
        "screen. The pair is fresh seed-9 Muon512 versus Muon768; AdamW512 is an "
        "optional same-width control only when its actual profile and the root's "
        "predeclared runtime projection permit it.\n\n"
        "The source-only build has not been uploaded, pushed, fitted, scored, or "
        "graded. `ROOT_CONDITIONAL_FIT_AUTHORIZATION.json` is pending and grants "
        "no action. Both a packet review and fresh provider admission are required "
        "before execution.\n\n"
        "The worker verifies the exact raw train/valid files and four named fold-0 "
        "cache inputs, then shares those read-only mounts across arms. It profiles "
        "the required Muon pair at B8 and falls back to common B4 only after an "
        "explicit B8 CUDA allocation failure. The optional AdamW arm is profiled "
        "at the selected pair batch and cannot block the primary pair if it fails. "
        "The conditional fit gate requires four epochs, fixed epoch 3, the frozen "
        "clean split, and exact zero-skip update counts. Export uses feature-only "
        "FP32 prediction over exactly the 468 declared evaluation groups; the "
        "worker does not read target labels or scoring masks for that export and "
        "does not calculate a score.\n\n"
        "The trainer CLI receives profile and fit argv from one frozen constructor; "
        "the worker records sanitized phase heartbeats and private detailed receipts. "
        "The source tar explicitly includes the exact own-project `numeric_ple.py` "
        "runtime dependency required by unconditional trainer/model imports; its "
        "source snapshot and SHA-256 are recorded in the recipe. Two unrelated "
        "optional modules named in the older V1 archive are absent and unreferenced "
        "in this frozen runtime closure, so they are explicitly listed as omissions. "
        "The provider timeout is 6000 seconds, with an internal worker cap of 5200 "
        "seconds and a 300-second finalization reserve.\n",
        encoding="utf-8",
    )

    bound_files = {}
    for name in (
        "kernel.py",
        "capacity_worker.py",
        "train_entry.py",
        "verified_digest_handoff.py",
        "README.md",
        "kernel-metadata.json",
        "source/source_archive.tar",
        "source/source_archive_manifest.json",
        "fold0.json",
        "portable_inputs.json",
        "prior_input_identity_v2.json",
    ):
        path = OUT / name
        if not path.is_file():
            raise FileNotFoundError(f"packet runtime file is not ready: {name}")
        bound_files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    recipe = {
        "schema": "wnn-gru-optimizer-capacity-recipe-v2",
        "packet_id": PACKET_ID,
        "status": "source-prepared-pending-root-review-and-admission",
        "hypothesis_parent": "H-232",
        "scientific_question": (
            "At a fixed seed, data membership, objective and update budget, does the independent "
            "Muon recurrent update support a useful GRU width ladder from 512 to 768?"
        ),
        "comparison_scope": {
            "primary_pair": ["muon512", "muon768"],
            "optional_same_width_optimizer_control": "adamw512",
            "optional_control_rule": (
                "Profile AdamW512 once at the batch selected by the required Muon pair. A failed or "
                "missing optional-control profile cannot block the Muon pair and cannot be overridden "
                "by authorization. Fit AdamW512 only if its profile passes and measured remaining "
                "worker budget covers its complete fit and feature-only export after the Muon pair. "
                "No score or validation result may affect admission."
            ),
            "claim_scope": "bounded clean-fold screen; not a public-score forecast or submission claim",
        },
        "arms": [
            {"id": "muon512", "hidden": 512, "layers": 2, "optimizer": "muon"},
            {"id": "muon768", "hidden": 768, "layers": 2, "optimizer": "muon"},
            {"id": "adamw512", "hidden": 512, "layers": 2, "optimizer": "adamw", "optional": True},
        ],
        "initialization": "fresh seed-9 initialization; no checkpoint, teacher, or teammate weights",
        "input_view": {
            "features": 211,
            "raw_features": 112,
            "causal_event_features": 99,
            "encoder": "linear",
            "no_future_or_teacher_inputs": True,
        },
        "fit_membership": {
            "source": "train+valid through the frozen clean fold-0 component split",
            "train_groups": 10535,
            "valid_fit_groups": 1405,
            "fit_groups_total": 11940,
            "evaluation_groups_excluded": 468,
            "split_manifest_sha256": fold_binding["sha256"],
            "fit_evaluation_component_overlap": 0,
        },
        "sampler": {
            "mode": "seeded",
            "implementation": "EpochOrderSampler with independent torch.Generator per epoch",
            "seed": 9,
            "epoch_seed_formula": "(seed + 1009 * (epoch + 1)) mod 2^32",
            "pairing": (
                "All arms use the same ordered fit pools, seed, batch size, drop_last and epoch seed; "
                "the sampler does not consume model-initialization global RNG."
            ),
            "normalization_epoch": -1,
            "drop_last_disclosure": (
                "B8 selects all 11940 fit groups but consumes 11936 per epoch (4 not consumed); "
                "B4 consumes all 11940. The worker records counts and seed/epoch provenance. It does "
                "not claim an exact dropped group identity because the trainer does not persist the "
                "realized batch row-group sequence."
            ),
            "historical_comparison_limit": (
                "H232 used PyTorch RandomSampler/global CPU RNG. This explicit seeded sampler is a "
                "declared change, so this is not a pure width-only replay of historical H232."
            ),
        },
        "objective": {
            "loss": "weighted MSE on two primary targets",
            "target_clip": [-2.0, 2.0],
            "weight": "absolute magnitude of the clipped target",
            "normalization": "chunk",
            "chunk_rows": 2000,
            "global_gradient_clip_norm": 1.0,
            "auxiliary_losses": False,
            "propensity_weighting": False,
            "distillation": False,
            "EMA_or_SWA": False,
        },
        "optimizer": {
            "muon": {
                "implementation": "independently maintained project MuonAdamW",
                "scope": "GRU weight_ih_l* and weight_hh_l* only; three separate GRU gate blocks",
                "other_parameters": "project AdamW update for input projection, head, and biases",
                "new_numeric_policy": ORTHOGONALIZATION_POLICY,
                "policy": (
                    "FP32 Newton-Schulz matrix products except native CUDA BF16 when compute capability "
                    "major >= 8 and torch.cuda.is_bf16_supported() both hold. This is independent of "
                    "the model's FP16 autocast and GradScaler. T4/P100 use FP32."
                ),
            },
            "adamw_control": "torch AdamW on all trainable parameters; same LR, WD and OneCycle schedule",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "adam_beta2": 0.999,
            "schedule": "OneCycle",
        },
        "training": {
            "seed": 9,
            "epochs": 4,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch_zero_based": 3,
            "evaluation_during_fit": False,
            "model_precision": "fp16 autocast",
            "amp_init_scale": 4096.0,
            "amp_skip_gate": "exactly zero skips; every attempted update is counted and completed",
            "initial_batch": 8,
            "fallback_batch": 4,
            "batch_selection_rule": (
                "Profile both required Muon arms at B8 first. Only an explicit CUDA allocation OOM "
                "on a required arm may trigger a complete B4 reprofile of both required arms. Select "
                "B8 only if both required profiles pass; otherwise select B4 only if both B4 profiles "
                "pass. Profile the optional AdamW control once at the selected pair batch; its failure "
                "is recorded and cannot veto the primary pair. No numerical, data, or score-driven fallback."
            ),
            "batch_budgets": {
                "8": {
                    "batches_per_epoch": 1492,
                    "groups_consumed_per_epoch": 11936,
                    "unconsumed_selected_groups_per_epoch": 4,
                    "attempts_per_epoch": 14920,
                    "attempts_four_epochs": 59680,
                },
                "4": {
                    "batches_per_epoch": 2985,
                    "groups_consumed_per_epoch": 11940,
                    "unconsumed_selected_groups_per_epoch": 0,
                    "attempts_per_epoch": 29850,
                    "attempts_four_epochs": 119400,
                },
            },
        },
        "profile": {
            "warmup_outer_batches": 3,
            "timed_outer_batches": 20,
            "chunks_per_outer_batch": 10,
            "total_optimizer_attempts": 230,
            "warmup_attempts": 30,
            "timed_attempts": 200,
            "required_evidence": [
                "PROFILE and PROFILE_UPDATE_EVIDENCE emitted by the exact archived trainer",
                "230 attempts, 230 completed updates, 0 AMP skips and 230 loss checks",
                "finite profile losses and gradients with the GradScaler enabled at scale4096",
                "Muon compute-precision metadata matches the declared hardware policy",
                "stable one-device T4 identity, positive finite timing, measured GPU memory and host headroom",
            ],
            "actual_cli_note": (
                "The trainer supports --profile-warmup and --profile-steps; it does not support the "
                "older V1 draft's --profile-trace-output token, which is intentionally absent."
            ),
        },
        "fit_and_export_gate": {
            "primary_fit_arms": ["muon512", "muon768"],
            "optional_fit_arm": "adamw512",
            "profile_projection": (
                "For each fitted arm, use its actual PROFILE median outer-batch wall time times the "
                "selected full batches-per-epoch times four epochs, with a 1.20 safety factor; add "
                "measured elapsed setup/hash/profile time, fixed checkpoint and feature-only export "
                "reserves, and a 300-second worker finalization reserve. Reject if this does not fit "
                "the root-authorized 5200-second internal deadline. No extrapolation from H232 or TFLOPS."
            ),
            "strict_fit_contract": (
                "Four completed epochs, exact per-epoch attempts and batches, zero skips, finite "
                "checkpoint parameters, matching source/split/membership/args and fixed epoch-3 output."
            ),
            "prediction_export": (
                "FP32 feature-only inference over exactly the frozen 468 clean-fold evaluation groups; "
                "targets and scoring masks are not requested. No score is computed in the worker."
            ),
            "grading": "root-owned grader consumes only the frozen 468 groups in sidecar order",
        },
        "inputs": {
            "portable_manifest": portable_binding,
            "split_manifest": fold_binding,
            "prior_input_identity": prior_identity_binding,
            "source_members": RAW_INPUTS,
            "total_bytes": 37_103_450_341,
            "mounted_sources": [
                "vladkochetov07/wunder-funder",
                "vladkochetov07/wnn-c20261004-fold0-assets",
            ],
            "mount_use": (
                "Raw train/valid come from the approved raw dataset. Only the four named fold0 cache "
                "files are linked from the reusable asset mount. Other model/teacher files in that "
                "mount are not read. One read-only mounted copy is shared by every arm."
            ),
        },
        "source": {
            "independent_source_commit": run_git_commit(),
            "lineage": "own H232 V7 trainer baseline plus this branch's independent optimizer policy",
            "runtime_dependency_additions": ADDITIONAL_RUNTIME_DEPENDENCIES,
            "source_archive": {
                "path": "source/source_archive.tar",
                "bytes": (OUT / "source/source_archive.tar").stat().st_size,
                "sha256": archive_sha,
            },
            "source_manifest": {
                "path": "source/source_archive_manifest.json",
                "bytes": source_manifest_bytes,
                "sha256": source_manifest_sha,
                "member_count": len(members),
            },
            "omitted_members": omitted_members,
            "trainer_sha256": member_hashes["scripts/train_rnn.py"],
            "predictor_sha256": member_hashes["scripts/predict_run.py"],
            "optimizer_sha256": member_hashes["src/wnn/optim.py"],
            "sampling_sha256": member_hashes["src/wnn/training_sampling.py"],
            "file_bindings": bound_files,
            "source_restrictions": "no teammate code or weights; all arms initialize from scratch",
        },
        "runtime": {
            "provider": "Kaggle T4; one visible GPU (device zero)",
            "requested_kernel": kernel_metadata["id"],
            "provider_timeout_seconds": 6000,
            "worker_wall_cap_seconds": 5200,
            "worker_finalization_reserve_seconds": 300,
            "loader_workers": 0,
            "cpu_threads": 4,
            "gpu_allocator_fraction": 0.75,
            "network": "disabled",
            "all_in_vast_cost_cap_usd": 0.0,
        },
        "authorization": {
            "profile": "requires root-approved packet and fresh Kaggle quota admission",
            "conditional_fit": "requires a separate explicit root authorization embedded in the packet",
            "grading": False,
            "provider_push": False,
        },
    }
    write_json(OUT / "GRU512_768_MUON_CAPACITY_RECIPE_V2.json", recipe)
    recipe_sha = sha256_file(OUT / "GRU512_768_MUON_CAPACITY_RECIPE_V2.json")
    auth = {
        "schema": "wnn-gru-capacity-root-conditional-authorization-v2",
        "packet_id": PACKET_ID,
        "status": "pending_root_review",
        "authorized": False,
        "profile_authorized": False,
        "conditional_fit_authorized": False,
        "fit_arm_ids": [],
        "selected_batch": None,
        "recipe_sha256": recipe_sha,
        "source_archive_sha256": archive_sha,
        "source_manifest_sha256": source_manifest_sha,
        "portable_manifest_sha256": portable_binding["sha256"],
        "split_manifest_sha256": fold_binding["sha256"],
        "prior_input_identity_sha256": prior_identity_binding["sha256"],
        "kernel_metadata_sha256": bound_files["kernel-metadata.json"]["sha256"],
        "worker_sha256": bound_files["capacity_worker.py"]["sha256"],
        "entry_sha256": bound_files["kernel.py"]["sha256"],
        "train_entry_sha256": bound_files["train_entry.py"]["sha256"],
        "digest_handoff_sha256": bound_files["verified_digest_handoff.py"]["sha256"],
        "root_review_receipt": "ROOT_INDEPENDENT_REVIEW.json",
        "root_review_receipt_sha256": None,
        "max_worker_seconds": 5200,
        "all_in_vast_cost_cap_usd": 0.0,
        "notes": "Pending independent source review and fresh cloud admission; grants no action.",
    }
    write_json(OUT / "ROOT_CONDITIONAL_FIT_AUTHORIZATION.json", auth)
    bound_files["GRU512_768_MUON_CAPACITY_RECIPE_V2.json"] = {
        "bytes": (OUT / "GRU512_768_MUON_CAPACITY_RECIPE_V2.json").stat().st_size,
        "sha256": recipe_sha,
    }
    bound_files["ROOT_CONDITIONAL_FIT_AUTHORIZATION.json"] = {
        "bytes": (OUT / "ROOT_CONDITIONAL_FIT_AUTHORIZATION.json").stat().st_size,
        "sha256": sha256_file(OUT / "ROOT_CONDITIONAL_FIT_AUTHORIZATION.json"),
    }
    write_json(
        OUT / "PACKET_BUILD_RECEIPT.json",
        {
            "packet_id": PACKET_ID,
            "status": "source-prepared-pending-root-review-and-admission",
            "recipe_sha256": recipe_sha,
            "source_archive_sha256": archive_sha,
            "source_manifest_sha256": source_manifest_sha,
            "omitted_source_members": omitted_members,
            "authorization_sha256": sha256_file(OUT / "ROOT_CONDITIONAL_FIT_AUTHORIZATION.json"),
            "bound_files": bound_files,
        },
    )
    print(
        json.dumps(
            {
                "packet": str(OUT),
                "recipe_sha256": recipe_sha,
                "source_archive_sha256": archive_sha,
                "source_manifest_sha256": source_manifest_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
