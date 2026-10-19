from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import feature_validation_checks as checks  # noqa: E402
import feature_validation_runtime as runtime  # noqa: E402


@pytest.mark.parametrize(
    ("arm", "family", "mode", "control", "width"),
    [
        ("group", "age", "group_age", "none", 323),
        ("field", "age", "field_age", "none", 323),
        ("moments", "innovation", "causal_innovations", "newest", 291),
        ("innovations", "innovation", "causal_innovations", "none", 291),
    ],
)
def test_frozen_feature_arm_mapping(
    arm: str, family: str, mode: str, control: str, width: int
) -> None:
    job = {"id": "cell-f0-s1003", "arm": arm, "family": family, "fold": 0, "seed": 1003}
    assert checks.job_spec(job) == (mode, control, job["id"], width)


def _request_and_args() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    argv = [
        "--name",
        "cell-f0-s1003",
        "--valid-fold",
        "0",
        "--seed",
        "1003",
        "--hidden",
        "512",
        "--layers",
        "1",
        "--epochs",
        "6",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "5",
        "--precision",
        "fp32",
        "--batch",
        "8",
        "--chunk",
        "2000",
        "--no-eval-during-fit",
        "--research-features",
        "causal_innovations",
        "--research-feature-control",
        "newest",
    ]
    args = {
        "name": "cell-f0-s1003",
        "valid_fold": 0,
        "seed": 1003,
        "hidden": 512,
        "layers": 1,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "precision": "fp32",
        "batch": 8,
        "chunk": 2000,
        "no_eval_during_fit": True,
        "research_features": "causal_innovations",
        "research_feature_control": "newest",
    }
    request = {
        "schema": "pr15-host-worker-request.v1",
        "mode": "fit",
        "name": args["name"],
        "source_root": "/project/runtime/source-a266-lf-subset-v1",
        "work": "/project/state/work/cell-f0-s1003",
        "train_argv": argv,
        "training_argv_sha256": hashlib.sha256(
            json.dumps(argv, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return request, args, argv


def test_request_must_match_exact_source_work_and_canonical_argv() -> None:
    request, args, argv = _request_and_args()
    digest = checks.check_request(
        request,
        args,
        run_name="cell-f0-s1003",
        fold=0,
        seed=1003,
        expected_argv=argv,
        source_root=request["source_root"],
        work_dir=request["work"],
    )
    assert digest == request["training_argv_sha256"]

    changed = dict(request, source_root="/other/source")
    with pytest.raises(ValueError, match="request identity"):
        checks.check_request(
            changed,
            args,
            run_name="cell-f0-s1003",
            fold=0,
            seed=1003,
            expected_argv=argv,
            source_root=request["source_root"],
            work_dir=request["work"],
        )

    changed = dict(request, train_argv=argv[:-1])
    with pytest.raises(ValueError, match="exact R01 recipe"):
        checks.check_request(
            changed,
            args,
            run_name="cell-f0-s1003",
            fold=0,
            seed=1003,
            expected_argv=argv,
            source_root=request["source_root"],
            work_dir=request["work"],
        )


def _history() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    history: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    total = {"optimizer_attempts": 0, "optimizer_completed_updates": 0, "amp_skipped_updates": 0}
    for epoch in range(6):
        current = {
            "optimizer_attempts": 12,
            "optimizer_completed_updates": 12,
            "amp_skipped_updates": 0,
        }
        for key, value in current.items():
            total[key] += value
        record = {
            "epoch": epoch,
            "evaluation": "withheld",
            "update_diagnostics": {"epoch": current, "cumulative": dict(total)},
        }
        history.append(record)
        provenance_rows.append({"epoch_index": epoch, "epoch": current, "cumulative": dict(total)})
    provenance = {"cumulative_updates": dict(total), "update_diagnostics": provenance_rows}
    checkpoint = {"provenance": {"cumulative_updates": dict(total)}}
    return history, provenance, checkpoint


def test_six_epoch_receipt_requires_all_finite_updates_and_withheld_eval() -> None:
    history, provenance, checkpoint = _history()
    assert checks.check_updates(history, provenance, checkpoint) == 72

    history[3]["update_diagnostics"]["epoch"]["amp_skipped_updates"] = 1
    for record in history[3:]:
        record["update_diagnostics"]["cumulative"]["amp_skipped_updates"] = 1
    for record in provenance["update_diagnostics"][3:]:
        record["cumulative"]["amp_skipped_updates"] = 1
    provenance["cumulative_updates"]["amp_skipped_updates"] = 1
    checkpoint["provenance"]["cumulative_updates"]["amp_skipped_updates"] = 1
    with pytest.raises(ValueError, match="skipped or nonfinite updates"):
        checks.check_updates(history, provenance, checkpoint)


def test_innovation_prior_reconstructs_fit_only_sample_and_control() -> None:
    inputs = {"fit_train_groups": list(range(40)), "fit_valid_groups": [100, 101]}
    members = checks._prior_members(inputs, 1003)
    prior = {
        "schema": "causal_moments_v1",
        "mean": [0.0] * 16,
        "scale": [1.0] * 16,
        "fit_members": members,
        "fit_members_sha256": hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "sample_policy": "uniform_seeded_source_qualified_32_groups_v1",
        "sample_seed": 1003 + 20261012,
        "sample_stride": 64,
        "sample_count": len(members) * 313,
    }
    provenance = {
        "causal_feature_prior": prior,
        "causal_feature_control": {
            "kind": "newest",
            "zeroed_suffix_width": 16,
            "effective_capacity": "reduced",
        },
        "causal_feature_prior_source_sha256": "f" * 64,
    }
    checkpoint = {"research_feature_prior": prior}
    assert (
        checks.check_prior(
            checkpoint,
            provenance,
            inputs,
            mode="causal_innovations",
            control="newest",
            seed=1003,
            feature_source_sha256="f" * 64,
        )
        is not None
    )

    contaminated = dict(prior, fit_members=members + [{"split": "valid", "row_group": 999}])
    with pytest.raises(ValueError, match="membership or deterministic sampling"):
        checks.check_prior(
            {"research_feature_prior": contaminated},
            dict(provenance, causal_feature_prior=contaminated),
            inputs,
            mode="causal_innovations",
            control="newest",
            seed=1003,
            feature_source_sha256="f" * 64,
        )


def test_portable_source_pin_derives_actual_sixty_file_trainer_identity(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "version", lambda package: "99.1")
    files = {f"src/wnn/training/module_{index}.py": f"{index:064x}" for index in range(60)}
    files["src/wnn/training/README.md"] = "a" * 64
    manifest = {"files": files}
    identity = runtime.canonical_producer_identity(manifest)
    expected = dict(files)
    expected.pop("src/wnn/training/README.md")
    expected_sha = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert identity["schema"] == "wnn.training-source.v1"
    assert identity["sha256"] == expected_sha
    assert identity["files"] == expected
    assert identity["runtime_dependencies"] == {"attrs": "99.1"}


def test_model_gate_accepts_only_full_fp32_gru_geometry() -> None:
    torch = pytest.importorskip("torch")
    shapes = {
        "mean": (291,),
        "std": (291,),
        "inp.weight": (512, 291),
        "inp.bias": (512,),
        "gru.weight_ih_l0": (1536, 512),
        "gru.weight_hh_l0": (1536, 512),
        "gru.bias_ih_l0": (1536,),
        "gru.bias_hh_l0": (1536,),
        "head.0.weight": (512, 1024),
        "head.0.bias": (512,),
        "head.2.weight": (2, 512),
        "head.2.bias": (2,),
    }
    state = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in shapes.items()}
    state["std"].fill_(1.0)
    runtime.check_model({"model": state}, 291, torch)
    state["gru.weight_hh_l0"] = state["gru.weight_hh_l0"].double()
    with pytest.raises(ValueError, match="strict FP32"):
        runtime.check_model({"model": state}, 291, torch)
