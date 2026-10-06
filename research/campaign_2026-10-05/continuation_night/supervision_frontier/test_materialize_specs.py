from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import materialize_specs as materializer  # noqa: E402
from resolve_calibration import resolve_template  # noqa: E402

from wnn.campaign_spec import digest, validate_spec  # noqa: E402


def _portable_schema(spec: dict[str, object]) -> dict[str, object]:
    """Substitute test-only content identities to exercise the queue schema."""
    result = dict(spec)
    for field in ("code", "inputs"):
        paths = result[field]
        assert isinstance(paths, list)
        result[field] = [
            {"path": path, "size": 0, "sha256": "b" * 64}
            for path in sorted(paths)
        ]
    result["code_sha256"] = digest(result["code"])
    return result


def test_six_alpha_horizon_cells_have_exact_existing_flags() -> None:
    expected = {
        (0.5, 0): ("0.5", "0"),
        (0.5, 8): ("0.5", "8"),
        (0.7, 0): ("0.7", "0"),
        (0.7, 8): ("0.7", "8"),
        (0.85, 0): ("0.85", "0"),
        (0.85, 8): ("0.85", "8"),
    }
    assert len(materializer.ALPHA_HORIZON) == 6
    for alpha, horizon in materializer.ALPHA_HORIZON:
        treatment, denominator, unresolved = materializer._arm_options(
            f"ah_a{int(alpha * 100):03d}_h{horizon}"
        )
        flags = dict(zip(treatment[::2], treatment[1::2], strict=True))
        assert (flags["--distill-alpha"], flags["--distill-boot-h"]) == expected[(alpha, horizon)]
        assert denominator == "chunk"
        assert unresolved is False


def test_materializer_emits_fixed_endpoints_and_unresolved_templates() -> None:
    graph = materializer.materialize()
    assert graph["hids"] == ["H-180", "H-134", "H-126"]
    assert graph["launch_authorized"] is False
    nodes = graph["nodes"]
    kinds = [node["kind"] for node in nodes]
    assert kinds.count("fit_only_cpu_calibration") == 3
    assert kinds.count("parent_fit") == 27
    assert kinds.count("head_fit") == 27
    assert kinds.count("prediction") == 54
    assert kinds.count("resource_probe_plan") == 4
    assert kinds.count("score_endpoint_spec") == 18

    for seed, fold in materializer.BLOCKS:
        fixed = f"c20261005_e496_den_fixed_s{seed}_f{fold}.template.json"
        shuffled = f"c20261005_e496_den_shuffled_s{seed}_f{fold}.template.json"
        fixed_spec = json.loads((materializer.FRONTIER / "specs" / fixed).read_text())
        shuffled_spec = json.loads((materializer.FRONTIER / "specs" / shuffled).read_text())
        assert "${calibration.reference_mass}" in fixed_spec["argv"]
        assert "${calibration.reference_mass}" in shuffled_spec["argv"]
        assert not (materializer.FRONTIER / "specs" / fixed.replace("template", "draft")).exists()
        assert not (materializer.FRONTIER / "specs" / shuffled.replace("template", "draft")).exists()

    calibration = json.loads(
        (materializer.FRONTIER / "specs" / "c20261005_e496_calibration_s3_f0.draft.json").read_text()
    )
    argv = calibration["argv"]
    assert "--device" in argv and argv[argv.index("--device") + 1] == "cpu"
    assert "--fast" not in argv and "--fused-adam" not in argv and "--pin-memory" not in argv
    assert "--prop-cache" in argv
    assert calibration["limits"] == {
        "cpus": 1,
        "ram_bytes": 18_000_000_000,
        "timeout_seconds": 14_400,
        "gpu_memory_bytes": 0,
        "gpu_stop_bytes": 0,
    }

    for path in (materializer.FRONTIER / "specs").glob("*.json"):
        spec = json.loads(path.read_text())
        validate_spec(_portable_schema(spec))


def test_direction_is_separate_and_fixed_coefficient_is_not_a_sweep() -> None:
    direction, denominator, denominator_unresolved = materializer._arm_options("direction_on")
    assert direction == ["--aux-sign", "0.1"]
    assert denominator == "chunk" and denominator_unresolved is False
    for arm, mode in (("den_fixed", "fixed"), ("den_shuffled", "shuffled")):
        options, actual_mode, unresolved = materializer._arm_options(arm)
        assert "--aux-sign" not in options
        assert actual_mode == mode and unresolved is True
        assert materializer.REFERENCE_TOKEN in options


def test_endpoint_matrices_include_fixed_interactions_and_no_selection() -> None:
    graph = materializer.materialize()
    endpoint_nodes = [node for node in graph["nodes"] if node["kind"] == "score_endpoint_spec"]
    for node in endpoint_nodes:
        data = json.loads((ROOT / str(node["spec"])).read_text())
        assert data["selection"].startswith("none")
        assert data["n_boot"] == 2000
        if data["family"] == "alpha_bootstrap":
            assert len(data["arms"]) == 6
            assert len(data["interactions"]) == 2
            assert all(len(contrast) == 2 for contrast in data["comparisons"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration_fixture(root: Path, *, fold: int = 0) -> tuple[Path, Path]:
    """Synthetic tmp-only schema fixture; it is never a research calibration."""
    source_paths = [path for path in (ROOT / "src/wnn").rglob("*.py")]
    source_paths += [ROOT / "scripts/train_rnn.py"]
    for source in source_paths:
        relative = str(source.relative_to(ROOT))
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    fold_relative = f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    fold_path = root / fold_relative
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / fold_relative, fold_path)

    def write_bytes(relative: str, value: bytes) -> str:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return _sha256(path)

    def write_json(relative: str, value: object) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    fold_manifest = json.loads(fold_path.read_text())
    sources: dict[str, dict[str, object]] = {}
    for split in ("train", "valid"):
        relative = f"datasets/{split}.parquet"
        payload = f"{split} fixture data".encode()
        sha = write_bytes(relative, payload)
        identity = {"bytes": len(payload), "footer_bytes": 0, "footer_sha256": "0" * 64, "mtime_ns": 0}
        fold_manifest["sources"][relative] = identity
        sources[relative] = {"original_identity": identity, "sha256": sha}

    target_orders = {"train": list(range(9)), "valid": list(range(7))}
    for split in ("train", "valid"):
        relative = f"cache/{split}_targets.npz"
        target_path = root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target_path, row_group=np.asarray(target_orders[split], dtype=np.int64), seq_ix=np.arange(len(target_orders[split])))
        target_sha = _sha256(target_path)
        stat = target_path.stat()
        identity = {"bytes": stat.st_size, "index_sha256": "1" * 64, "mtime_ns": 0}
        fold_manifest["sources"][relative] = identity
        sources[relative] = {"original_identity": identity, "sha256": target_sha}

    for relative, payload in (("cache/chains.npz", b"chain fixture"), ("cache/chain_folds.npz", b"fold fixture")):
        sha = write_bytes(relative, payload)
        identity = {"bytes": len(payload), "mtime_ns": 0, "sha256": sha}
        fold_manifest["sources"][relative] = identity
        sources[relative] = {"original_identity": identity, "sha256": sha}

    fold_path.write_text(json.dumps(fold_manifest))
    fold_sha = _sha256(fold_path)
    write_json("research/campaign_2026-10-04/portable_inputs.json", {
        "version": 1,
        "split_manifest_sha256": fold_sha,
        "sources": sources,
    })

    for seed in (3, 4):
        run = f"c20260929_since_s{seed}_f{fold}"
        model_rel = f"artifacts/runs/{run}/model.pt"
        model_sha = write_bytes(model_rel, f"{run} model".encode())
        epoch3_rel = f"artifacts/runs/{run}/model_ep3.pt"
        epoch3_sha = write_bytes(epoch3_rel, f"{run} epoch 3 model".encode())
        fold_fit = fold_manifest["fit"]
        teacher_provenance = {
            "args": {
                "seed": seed,
                "valid_fold": fold,
                "split_manifest_sha256": fold_sha,
                "checkpoint_policy": "fixed",
                "checkpoint_epoch": 3,
            },
            "split_manifest_sha256": fold_sha,
            "fit_row_groups": {
                "train": np.random.default_rng(seed).permutation(np.asarray(fold_fit["train"], dtype=np.int64)).tolist(),
                "valid": fold_fit["valid"],
            },
            "checkpoint_sha256": {"model.pt": model_sha, "model_ep3.pt": epoch3_sha},
        }
        write_json(f"artifacts/runs/{run}/provenance.json", teacher_provenance)
        for split in ("train", "valid"):
            pred_relative = f"cache/pred_{run}_{split}.npy"
            pred_path = root / pred_relative
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, np.zeros((len(target_orders[split]), 2), dtype=np.float32))
            pred_sha = _sha256(pred_path)
            write_json(pred_relative.replace(".npy", ".json"), {
                "run": run,
                "split": split,
                "epoch": 3,
                "checkpoint_sha256": epoch3_sha,
                "prediction_sha256": pred_sha,
                "split_manifest_sha256": fold_sha,
                "row_groups": target_orders[split],
            })

    prop_name = f"c20260929_prop_f{fold}"
    prop_model_sha = write_bytes(f"artifacts/runs/{prop_name}/model.pt", b"propensity model")
    write_json(f"artifacts/runs/{prop_name}/provenance.json", {
        "checkpoint_sha256": {"model.pt": prop_model_sha},
    })
    consumed_caches: dict[str, str] = {}
    for split in ("train", "valid"):
        consumed_caches[f"{split}_targets.npz"] = _sha256(root / f"cache/{split}_targets.npz")
        prop_rel = f"cache/propcache_{prop_name}_{split}.npy"
        prop_path = root / prop_rel
        prop_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prop_path, np.ones((len(target_orders[split]),), dtype=np.float32))
        prop_json = prop_rel.replace(".npy", ".json")
        write_json(prop_json, {"model": prop_name, "model_sha256": prop_model_sha, "split": split})
        consumed_caches[Path(prop_rel).name] = _sha256(prop_path)
        consumed_caches[Path(prop_json).name] = _sha256(root / prop_json)

    sources_contract = {
        str(path.relative_to(root / "src/wnn")): _sha256(path)
        for path in sorted((root / "src/wnn").rglob("*.py"))
    }
    sources_contract["trainer"] = _sha256(root / "scripts/train_rnn.py")
    calibration_path = f"calibration/reference_s3_f{fold}.json"
    fold_fit = fold_manifest["fit"]
    expected_train = np.random.default_rng(3).permutation(np.asarray(fold_fit["train"], dtype=np.int64)).tolist()
    membership = {
        "train": {
            "count": len(expected_train),
            "ordered_int64_le_sha256": hashlib.sha256(np.asarray(expected_train, dtype="<i8").tobytes()).hexdigest(),
        },
        "valid": {
            "count": len(fold_fit["valid"]),
            "ordered_int64_le_sha256": hashlib.sha256(np.asarray(fold_fit["valid"], dtype="<i8").tobytes()).hexdigest(),
        },
        "valid_repeat": 1,
        "dataset_length": len(fold_fit["train"]) + len(fold_fit["valid"]),
    }
    options: dict[str, object] = {
        "seed": 3,
        "valid_fold": fold,
        "train_source": "train+valid",
        "device": "cpu",
        "distill": f"c20260929_since_s3_f{fold},c20260929_since_s4_f{fold}",
        "no_eval_during_fit": True,
        "batch_order": "seeded",
        "loss": "wmse",
        "hidden": 256,
        "layers": 2,
        "batch": 32,
        "chunk": 1000,
        "update_rows": 0,
        "origin_view": "off",
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "optim": "muon",
        "muon_scope": "rnn",
        "ema_teacher": 1.0,
        "ema_decay": 0.999,
        "eval_ema": True,
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "distill_denominator": "chunk",
        "distill_reference_mass": 0.0,
        "distill_alpha": 0.7,
        "distill_boot_h": 8,
        "aux_sign": 0.0,
        "distill_reanchor": 0.0,
        "calibration_epochs": 1,
        "calibrate_distill_reference": calibration_path,
        "propensity": prop_name,
        "prop_cache": True,
        "fast": False,
        "pin_memory": False,
        "fused_adam": False,
        "since_weight": 32.0,
        "since_floor": 0.25,
        "valid_true_mask": 0.03,
        "prop_power": 2.0,
        "prop_eps": 0.03,
        "valid_repeat": 1,
        "valid_train_fraction": 1.0,
        "include_heldout_valid": False,
        "split_manifest_sha256": fold_sha,
    }
    teacher_hashes = {
        f"pred_c20260929_since_s{seed}_f{fold}_{split}.npy": _sha256(
            root / f"cache/pred_c20260929_since_s{seed}_f{fold}_{split}.npy"
        )
        for seed in (3, 4)
        for split in ("train", "valid")
    }
    contract = {
        "options": options,
        "sources": sources_contract,
        "datasets": {
            split: {
                "bytes": (root / f"datasets/{split}.parquet").stat().st_size,
                "sha256": sources[f"datasets/{split}.parquet"]["sha256"],
            }
            for split in ("train", "valid")
        },
        "teacher_predictions": teacher_hashes,
        "consumed_caches": consumed_caches,
        "fit_membership": membership,
        "ancestors": {"propensity_checkpoint_sha256": prop_model_sha},
        "runtime": {"devices": []},
    }
    retained = membership["dataset_length"] // 32 * 32
    mass_count = retained // 32 * 20
    mass_values = [12.25] * mass_count
    mass_bytes = np.asarray(mass_values, dtype="<f8").tobytes()
    report = {
        "format": "wnn.distill-reference-mass",
        "version": 1,
        "reference_mass": 12.25,
        "updates": mass_count,
        "minimum": 12.25,
        "maximum": 12.25,
        "standard_deviation": 0.0,
        "zero_mass_updates": 0,
        "estimator": "mean over actual sampled fit batches and update windows",
        "update_masses": {
            "format": "wnn.float64-le-update-masses",
            "version": 1,
            "dtype": "<f8",
            "count": mass_count,
            "sha256": hashlib.sha256(mass_bytes).hexdigest(),
            "values": mass_values,
        },
        "calibration_epochs": 1,
        "sample_order": [{
            "sampler_class": "EpochOrderSampler",
            "batch_size": 32,
            "drop_last": True,
            "num_workers": 0,
            "persistent_workers": False,
            "retained_dataset_indices_sha256": "a" * 64,
            "retained_count": retained,
            "unique_retained_count": retained,
            "dropped_dataset_indices": list(range(retained, membership["dataset_length"])),
        }],
        "contract": contract,
    }
    calibration_file = root / calibration_path
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_text(json.dumps(report))
    template_path = root / "specs" / f"c20261005_e496_den_fixed_s3_f{fold}.template.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "version": 1,
        "name": f"c20261005_e496_den_fixed_s3_f{fold}",
        "kind": "gpu",
        "argv": ["{python}", "scripts/train_rnn.py", "--distill-reference-mass", "${calibration.reference_mass}"],
        "code": ["scripts/train_rnn.py"],
        "inputs": [calibration_path],
        "artifacts": [f"artifacts/e496/fixed_s3_f{fold}.json"],
        "limits": {
            "cpus": 2,
            "ram_bytes": 18_000_000_000,
            "timeout_seconds": 3600,
            "gpu_memory_bytes": 4_200_000_000,
            "gpu_stop_bytes": 4_800_000_000,
        },
        "environment": {},
    }
    template_path.write_text(json.dumps(template))
    return template_path, calibration_file


def test_resolver_binds_fit_only_report_and_emits_queue_schema(tmp_path: Path) -> None:
    template, calibration = _calibration_fixture(tmp_path)
    output = resolve_template(template, calibration, root=tmp_path)
    resolved = json.loads(output.read_text())
    argv = resolved["argv"]
    assert argv[argv.index("--distill-reference-mass") + 1] == "12.25"
    binding_path = output.with_name(output.name.replace(".draft.json", ".calibration_binding.json"))
    binding = json.loads(binding_path.read_text())
    assert binding["reference_mass"] == 12.25
    assert binding["file_sha256"] == _sha256(calibration)
    assert str(binding_path.relative_to(tmp_path)) in resolved["inputs"]
    assert "calibration_binding" not in resolved
    validate_spec(_portable_schema(resolved))


def test_resolver_rejects_wrong_fit_fold_before_writing_draft(tmp_path: Path) -> None:
    template, calibration = _calibration_fixture(tmp_path)
    report = json.loads(calibration.read_text())
    report["contract"]["options"]["valid_fold"] = 1
    calibration.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recipe, seed/fold"):
        resolve_template(template, calibration, root=tmp_path)
    assert not template.with_name(template.name.replace(".template.json", ".draft.json")).exists()


def test_resolver_recomputes_scalar_from_recorded_update_masses(tmp_path: Path) -> None:
    template, calibration = _calibration_fixture(tmp_path)
    report = json.loads(calibration.read_text())
    report.update(
        {
            "reference_mass": 20.0,
            "minimum": 0.0,
            "maximum": 40.0,
            "standard_deviation": 12.0,
            "zero_mass_updates": 0,
        }
    )
    calibration.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="reference_mass differs from recorded per-update masses"):
        resolve_template(template, calibration, root=tmp_path)
    assert not template.with_name(template.name.replace(".template.json", ".draft.json")).exists()


@pytest.mark.parametrize(
    ("mismatch", "error"),
    [
        ("membership", "fitting membership count/order"),
        ("source", "source closure"),
        ("updates", "per-update mass count"),
        ("dataset", "current dataset content"),
        ("teacher", "sidecar does not bind current prediction bytes"),
        ("teacher_epoch3_missing", "required calibration dependency is missing"),
        ("teacher_epoch3_content", "model/epoch-3 checkpoint hash differs"),
        ("propensity_cache", "consumed-cache hashes differ"),
    ],
)
def test_resolver_rejects_stale_bound_dependencies(
    tmp_path: Path, mismatch: str, error: str
) -> None:
    template, calibration = _calibration_fixture(tmp_path)
    report = json.loads(calibration.read_text())
    if mismatch == "membership":
        report["contract"]["fit_membership"]["train"]["ordered_int64_le_sha256"] = "0" * 64
    elif mismatch == "source":
        report["contract"]["sources"]["trainer"] = "0" * 64
    elif mismatch == "updates":
        report["updates"] -= 1
    elif mismatch == "dataset":
        dataset_path = tmp_path / "datasets/train.parquet"
        payload = bytearray(dataset_path.read_bytes())
        payload[0] ^= 1
        dataset_path.write_bytes(payload)
    elif mismatch == "teacher":
        prediction_path = tmp_path / "cache/pred_c20260929_since_s3_f0_train.npy"
        payload = bytearray(prediction_path.read_bytes())
        payload[-1] ^= 1
        prediction_path.write_bytes(payload)
    elif mismatch == "teacher_epoch3_missing":
        run = "c20260929_since_s3_f0"
        provenance_path = tmp_path / f"artifacts/runs/{run}/provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["checkpoint_sha256"]["model_ep3.pt"] = "0" * 64
        provenance_path.write_text(json.dumps(provenance))
        (tmp_path / f"artifacts/runs/{run}/model_ep3.pt").unlink()
        for split in ("train", "valid"):
            sidecar_path = tmp_path / f"cache/pred_{run}_{split}.json"
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["checkpoint_sha256"] = "0" * 64
            sidecar_path.write_text(json.dumps(sidecar))
    elif mismatch == "teacher_epoch3_content":
        checkpoint_path = tmp_path / "artifacts/runs/c20260929_since_s3_f0/model_ep3.pt"
        payload = bytearray(checkpoint_path.read_bytes())
        payload[-1] ^= 1
        checkpoint_path.write_bytes(payload)
    elif mismatch == "propensity_cache":
        cache_path = tmp_path / "cache/propcache_c20260929_prop_f0_train.npy"
        payload = bytearray(cache_path.read_bytes())
        payload[-1] ^= 1
        cache_path.write_bytes(payload)
    else:
        raise AssertionError(mismatch)
    calibration.write_text(json.dumps(report))

    with pytest.raises(ValueError, match=error):
        resolve_template(template, calibration, root=tmp_path)
    assert not template.with_name(template.name.replace(".template.json", ".draft.json")).exists()


def test_resolver_binds_nested_probe_plan_and_calibration(tmp_path: Path) -> None:
    template, calibration = _calibration_fixture(tmp_path)
    spec = json.loads(template.read_text())
    plan_path = tmp_path / "probes/denominator.plan.json"
    plan_path.parent.mkdir(parents=True)
    plan = {
        "version": 1,
        "steps": [{"name": "fit", "argv": ["scripts/train_rnn.py", "--distill-reference-mass", "${calibration.reference_mass}"]}],
        "report": "probes/report.json",
    }
    plan_path.write_text(json.dumps(plan))
    plan_relative = str(plan_path.relative_to(tmp_path))
    spec["argv"] = ["{python}", "scripts/kaggle_sequence.py", "--plan", plan_relative]
    spec["inputs"].append(plan_relative)
    template.write_text(json.dumps(spec))

    output = resolve_template(template, calibration, root=tmp_path)
    resolved = json.loads(output.read_text())
    resolved_plan_relative = resolved["argv"][resolved["argv"].index("--plan") + 1]
    resolved_plan = json.loads((tmp_path / resolved_plan_relative).read_text())
    step_argv = resolved_plan["steps"][0]["argv"]
    assert step_argv[step_argv.index("--distill-reference-mass") + 1] == "12.25"
    assert resolved_plan_relative in resolved["inputs"]
    binding_path = output.with_name(output.name.replace(".draft.json", ".calibration_binding.json"))
    binding = json.loads(binding_path.read_text())
    assert binding["plan_sha256"] == _sha256(plan_path)
    assert binding["resolved_plan_sha256"] == _sha256(tmp_path / resolved_plan_relative)
    validate_spec(_portable_schema(resolved))
