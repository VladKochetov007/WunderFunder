from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import cache_propensity as cache_propensity_script
from scripts import train_propensity as train_propensity_script
from wnn.propensity import (
    LEGACY_PROFILE,
    PROPENSITY_FIT_SOURCE_FILES,
    STRICT_FP32_PROFILE,
    PropensityNet,
    fit_normalization,
    ordered_groups_sha256,
    propensity,
)
from wnn.splits import CampaignSplit
from wnn.training import cache_identity
from wnn.training.cache_identity import prop_cache_path
from wnn.training.environment import TrainingEnvironment


def test_streaming_normalization_uses_all_ordered_fit_batches() -> None:
    model = PropensityNet(n_in=2, hidden=3)
    batches = [
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
    ]

    provenance = fit_normalization(model, iter(batches))

    expected = torch.cat([batch.reshape(-1, 2) for batch in batches]).to(torch.float64)
    torch.testing.assert_close(model.mean, expected.mean(0).float())
    torch.testing.assert_close(model.std, expected.std(0, unbiased=False).float())
    assert provenance["count"] == 4
    assert provenance["accumulator_dtype"] == "float64"
    assert provenance["buffer_dtype"] == "float32"
    assert provenance["method"] == "ordered_fit_population_welford_float64_v1"


def test_streaming_normalization_rejects_nonfinite_values() -> None:
    model = PropensityNet(n_in=2, hidden=3)
    batch = torch.tensor([[[1.0, float("nan")]]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        fit_normalization(model, [batch])


def test_propensity_precision_profile_is_explicit_and_legacy_default_is_preserved() -> None:
    model = PropensityNet(n_in=2, hidden=3).eval()
    inputs = torch.randn(2, 7, 2)
    legacy = propensity(model, inputs, chunk=3)
    explicit_legacy = propensity(model, inputs, chunk=3, profile=LEGACY_PROFILE)
    strict = propensity(model, inputs, chunk=3, profile=STRICT_FP32_PROFILE)

    torch.testing.assert_close(legacy, explicit_legacy)
    torch.testing.assert_close(legacy, strict)
    with pytest.raises(ValueError, match="float32 inputs"):
        propensity(model, inputs.to(torch.float64), profile=STRICT_FP32_PROFILE)


def test_strict_propensity_rejects_nonfinite_inputs_and_model_outputs() -> None:
    model = PropensityNet(n_in=2, hidden=3).eval()
    bad_inputs = torch.full((1, 2, 2), float("nan"))
    with pytest.raises(FloatingPointError, match="inputs"):
        propensity(model, bad_inputs, profile=STRICT_FP32_PROFILE)

    class NonfiniteModel(torch.nn.Module):
        def forward(self, inputs, hidden=None):
            logits = torch.full(inputs.shape[:2], float("inf"), device=inputs.device)
            return logits, hidden

    with pytest.raises(FloatingPointError, match="logits"):
        propensity(NonfiniteModel(), torch.ones((1, 2, 2)), profile=STRICT_FP32_PROFILE)


def test_strict_run_directory_is_create_only_but_legacy_reuses(tmp_path: Path) -> None:
    run = tmp_path / "artifacts/runs/clean"
    train_propensity_script._prepare_run_directory(run, strict=True)
    with pytest.raises(FileExistsError):
        train_propensity_script._prepare_run_directory(run, strict=True)
    train_propensity_script._prepare_run_directory(run, strict=False)

    dangling = tmp_path / "artifacts/runs/dangling"
    dangling.parent.mkdir(parents=True, exist_ok=True)
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        train_propensity_script._prepare_run_directory(dangling, strict=True)
    with pytest.raises(FloatingPointError, match="gradient"):
        train_propensity_script._require_finite(torch.tensor([float("inf")]), "gradient test")


def test_strict_cache_dataset_reads_feature_columns_without_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSequenceFile:
        def __init__(self, _path: Path) -> None:
            pass

        def read(self, index: int, *, targets: bool, mask: bool):
            assert index == 7
            assert not targets and not mask
            return type("Sequence", (), {"features": np.ones((2, 3), dtype=np.float32)})()

    monkeypatch.setattr(cache_propensity_script, "SequenceFile", FakeSequenceFile)
    dataset = cache_propensity_script.FeatureOnlySequences("valid", np.asarray([7]))

    assert torch.equal(dataset[0], torch.ones((2, 3)))


def _write_source(root: Path, source: Path, target: str) -> str:
    dest = root / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    dest.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _strict_cache_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], CampaignSplit]:
    root = Path(__file__).resolve().parents[1]
    fake_root = tmp_path / "project"
    for path in (*PROPENSITY_FIT_SOURCE_FILES, "scripts/cache_propensity.py"):
        _write_source(fake_root, root / path, path)
    (fake_root / "datasets").mkdir(parents=True)
    (fake_root / "datasets/valid.parquet").touch()
    cache_path = fake_root / "cache/propcache_test_valid.npy"
    cache_path.parent.mkdir(parents=True)
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(2, 20000))
    cache[:] = 0.25
    cache.flush()
    del cache

    mean = torch.zeros(1, dtype=torch.float32)
    std = torch.ones(1, dtype=torch.float32)
    normalization = {
        "method": "ordered_fit_population_welford_float64_v1",
        "accumulator_dtype": "float64",
        "buffer_dtype": "float32",
        "count": 20000,
        "std_floor": 0.001,
        "mean_sha256": hashlib.sha256(mean.numpy().astype("<f4").tobytes()).hexdigest(),
        "std_sha256": hashlib.sha256(std.numpy().astype("<f4").tobytes()).hexdigest(),
    }
    split = CampaignSplit(
        path=tmp_path / "split.json",
        sha256="c" * 64,
        fold=0,
        n_folds=2,
        train_groups=(0,),
        valid_groups=(1,),
        evaluation_groups=(0,),
        components=(("valid", 0, 0), ("valid", 1, 1)),
    )
    model_path = fake_root / "artifacts/runs/test/model.pt"
    model_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {"mean": mean, "std": std},
            "hidden": 1,
            "n_in": 1,
            "args": {},
            "provenance": {
                "precision_profile": STRICT_FP32_PROFILE,
                "split_manifest_sha256": split.sha256,
                "fit_row_groups": {"valid": [1]},
                "fit_row_groups_sha256": {"valid": ordered_groups_sha256([1])},
                "evaluation_row_groups": [0],
                "normalization": normalization,
                "source_sha256": {
                    relative: hashlib.sha256((fake_root / relative).read_bytes()).hexdigest()
                    for relative in PROPENSITY_FIT_SOURCE_FILES
                },
            },
        },
        model_path,
    )
    source_hashes = {
        relative: hashlib.sha256((fake_root / relative).read_bytes()).hexdigest()
        for relative in (*PROPENSITY_FIT_SOURCE_FILES, "scripts/cache_propensity.py")
    }
    metadata: dict[str, object] = {
        "schema": "wnn.propensity_cache.strict_fp32.v1",
        "model": "test",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "split": "valid",
        "rows": 2,
        "n_filled": 2,
        "precision_profile": STRICT_FP32_PROFILE,
        "inference_precision": "float32_no_autocast_tf32_disabled",
        "split_manifest_sha256": split.sha256,
        "fit_row_groups_sha256": ordered_groups_sha256([1]),
        "fit_row_groups_count": 1,
        "ordered_output_row_groups_sha256": ordered_groups_sha256([0, 1]),
        "populated_row_groups_sha256": ordered_groups_sha256([0, 1]),
        "checkpoint_normalization": normalization,
        "cache_shape": [2, 20000],
        "cache_dtype": "float32",
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
    }
    cache_path.with_suffix(".json").write_text(json.dumps(metadata))
    return cache_path, metadata, split


def test_strict_cache_validates_fit_membership_separately_from_output_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path, metadata, split = _strict_cache_fixture(tmp_path)
    monkeypatch.setattr(cache_identity, "SequenceFile", lambda _path: [None, None])
    environment = TrainingEnvironment(root=tmp_path / "project", source_root=tmp_path / "project")

    resolved = prop_cache_path("test", "valid", environment=environment, campaign_split=split)

    assert resolved == cache_path
    assert metadata["fit_row_groups_sha256"] != metadata["ordered_output_row_groups_sha256"]

    changed = dict(metadata)
    changed["ordered_output_row_groups_sha256"] = ordered_groups_sha256([1, 0])
    cache_path.with_suffix(".json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="row-group order"):
        prop_cache_path("test", "valid", environment=environment, campaign_split=split)


def test_strict_cache_rejects_nonfinite_and_out_of_range_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path, metadata, split = _strict_cache_fixture(tmp_path)
    monkeypatch.setattr(cache_identity, "SequenceFile", lambda _path: [None, None])
    environment = TrainingEnvironment(root=tmp_path / "project", source_root=tmp_path / "project")

    cache = np.load(cache_path, mmap_mode="r+")
    cache[0, 0] = np.nan
    cache.flush()
    del cache
    metadata["cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    cache_path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="invalid probability"):
        prop_cache_path("test", "valid", environment=environment, campaign_split=split)


def test_strict_cache_preflights_both_splits_and_publishes_create_only(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    paths = cache_propensity_script._strict_output_paths(tmp_path, "new")
    assert {split for split, _, _ in paths} == {"train", "valid"}

    dangling_sidecar = paths[1][2]
    dangling_sidecar.symlink_to(tmp_path / "missing-sidecar")
    with pytest.raises(FileExistsError, match="already exists"):
        cache_propensity_script._strict_output_paths(tmp_path, "new")

    temp_array = cache_propensity_script._reserve_temp(paths[0][1])
    temp_sidecar = cache_propensity_script._reserve_temp(paths[0][2])
    temp_array.write_bytes(b"array")
    temp_sidecar.write_bytes(b"metadata")
    paths[0][1].write_bytes(b"racer")
    with pytest.raises(FileExistsError):
        cache_propensity_script._publish_pair(temp_array, temp_sidecar, paths[0][1], paths[0][2])
    assert temp_array.exists() and temp_sidecar.exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "/tmp/outside", "a/b", "a\\b"])
def test_cache_name_must_be_a_single_path_component(name: str) -> None:
    with pytest.raises(ValueError, match="single path component"):
        cache_propensity_script._validate_cache_name(name)


def test_cache_consumer_rejects_escaping_name_before_checkpoint_lookup(tmp_path: Path) -> None:
    environment = TrainingEnvironment(root=tmp_path, source_root=tmp_path)
    with pytest.raises(ValueError, match="single path component"):
        prop_cache_path("../outside", "valid", environment=environment)


def test_cache_cli_rejects_escaping_name_before_checkpoint_or_runtime_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["cache_propensity.py", "../outside"])
    monkeypatch.setattr(
        cache_propensity_script,
        "configure_torch",
        lambda: pytest.fail("runtime setup must follow name validation"),
    )
    with pytest.raises(SystemExit) as error:
        cache_propensity_script.main()
    assert error.value.code == 2


def test_strict_source_inventory_includes_portable_manifest_verifier() -> None:
    assert "src/wnn/portability.py" in PROPENSITY_FIT_SOURCE_FILES


def test_reserved_cache_memmap_is_valid_npy_and_rejects_symlink_swap(tmp_path: Path) -> None:
    reserved = cache_propensity_script._reserve_temp(tmp_path / "cache.npy")
    mapped = cache_propensity_script._open_reserved_memmap(reserved, (2, 3))
    mapped[:] = 0.5
    mapped.flush()
    del mapped
    assert np.all(np.load(reserved, allow_pickle=False) == 0.5)

    replaced = cache_propensity_script._reserve_temp(tmp_path / "replace.npy")
    replaced.unlink()
    replaced.symlink_to(tmp_path / "target")
    with pytest.raises(OSError):
        cache_propensity_script._open_reserved_memmap(replaced, (2, 3))


def _tiny_strict_training_case(tmp_path: Path, run_name: str):
    from torch.utils.data import DataLoader, TensorDataset

    from wnn.propensity import PropensityNet

    source_root = tmp_path / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    source_file = source_root / "source.py"
    source_file.write_text("strict fixture source\n")
    source_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
    split_path = tmp_path / "split.json"
    split_payload = json.dumps({"sources": {"valid": "fixture"}}).encode()
    split_path.write_bytes(split_payload)
    split = CampaignSplit(
        path=split_path,
        sha256=hashlib.sha256(split_payload).hexdigest(),
        fold=0,
        n_folds=2,
        train_groups=(0, 1),
        valid_groups=(2, 3),
        evaluation_groups=(4,),
        components=(
            ("train", 0, 0),
            ("train", 1, 1),
            ("valid", 2, 2),
            ("valid", 3, 3),
            ("valid", 4, 4),
        ),
    )
    features = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16
    masks = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float32)
    dataset = TensorDataset(features, torch.zeros(2), masks)
    dataset.indices = [2, 3]
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = PropensityNet(n_in=2, hidden=3)
    model.mean.copy_(torch.tensor([0.25, 0.5]))
    model.std.copy_(torch.tensor([0.75, 0.5]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    args: dict[str, object] = {
        "hidden": 3,
        "epochs": 2,
        "lr": 0.01,
        "seed": 17,
        "checkpoint_epoch": 1,
        "resume": False,
        "split_manifest": str(split_path),
        "loader_workers": 0,
    }
    normalization = {
        "method": "fixture_fit_only",
        "count": 16,
        "accumulator_dtype": "float64",
        "buffer_dtype": "float32",
    }
    provenance: dict[str, object] = {
        "n_in": 2,
        "source_sha256": {"source.py": source_sha},
        "checkpoint_sha256": {},
    }
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return model, optimizer, loader, args, provenance, normalization, split, source_root, run_dir


def _run_tiny_strict_fit(case, *, resume: bool, build_inputs=None):
    from wnn.propensity_fit import fit_strict_propensity

    model, optimizer, loader, args, provenance, normalization, split, source_root, run_dir = case
    call_args = dict(args)
    call_args["resume"] = resume
    return fit_strict_propensity(
        model=model,
        loader=loader,
        optimizer=optimizer,
        build_inputs=build_inputs or (lambda values: values),
        args=call_args,
        provenance=provenance,
        normalization=None if resume else normalization,
        split=split,
        data_root=source_root,
        source_root=source_root,
        device=torch.device("cpu"),
        warmup=0,
        chunk_rows=2,
        resume=resume,
        run_dir=run_dir,
    )


def test_strict_fit_resume_replays_only_uncommitted_epoch(tmp_path: Path, monkeypatch) -> None:
    import random

    from wnn.propensity_failure import create_only_json
    from wnn.propensity_progress import ProgressJournal
    from wnn.propensity_recovery import load_resume_state

    monkeypatch.setattr("wnn.propensity_fit.emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("wnn.propensity_fit_update.emit", lambda *args, **kwargs: None)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        random.seed(17)
        torch.manual_seed(17)
        interrupted = _tiny_strict_training_case(tmp_path / "interrupted", "run")
        (tmp_path / "interrupted").mkdir(exist_ok=True)
        calls = 0

        def fail_once(values: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("synthetic interruption")
            return values

        with pytest.raises(RuntimeError, match="synthetic interruption"):
            _run_tiny_strict_fit(interrupted, resume=False, build_inputs=fail_once)
        run_dir = interrupted[-1]
        failures = list(run_dir.glob("failure-*.json"))
        assert len(failures) == 1
        first_failure_path = failures[0]
        first_failure_bytes = first_failure_path.read_bytes()
        failure = json.loads(first_failure_bytes)
        assert failure["committed_next_epoch"] == 1
        assert failure["optimizer_attempts_this_attempt"] == 6
        with pytest.raises(FileExistsError):
            create_only_json(first_failure_path, {"status": "overwritten"})
        assert first_failure_path.read_bytes() == first_failure_bytes
        saved = load_resume_state(
            run_dir / "training_state.pt",
            json.loads((run_dir / "provenance.json").read_text())["resume_contract"]["static"],
        )
        journal = ProgressJournal(run_dir / "progress.jsonl", resume_after=saved["event_sequence"])
        recovery = journal.recovery_summary(saved["event_sequence"])
        assert recovery["rolled_back_optimizer_attempts"] == 2
        assert recovery["rolled_back_completed_updates"] == 2

        torch.manual_seed(17)
        resumed = _tiny_strict_training_case(tmp_path / "interrupted", "run")

        calls = 0

        def fail_second_attempt(values: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second synthetic interruption")
            return values

        with pytest.raises(RuntimeError, match="second synthetic interruption"):
            _run_tiny_strict_fit(resumed, resume=True, build_inputs=fail_second_attempt)
        failures = sorted(run_dir.glob("failure-*.json"))
        assert len(failures) == 2
        failure_bytes = {path.name: path.read_bytes() for path in failures}

        torch.manual_seed(17)
        resumed = _tiny_strict_training_case(tmp_path / "interrupted", "run")
        _run_tiny_strict_fit(resumed, resume=True)
        resumed_provenance = json.loads((run_dir / "provenance.json").read_text())
        progress = resumed_provenance["optimizer_progress"]
        assert progress["optimizer_attempts"] == 8
        assert progress["optimizer_step_calls"] == 8
        assert progress["optimizer_completed_updates"] == 8
        assert progress["nonfinite_updates"] == 0
        assert progress["learning_rate_first"] == 0.01
        assert progress["learning_rate_min"] == 0.01
        assert progress["preclip_grad_norm_steps"] == 8
        assert resumed_provenance["recovery_history"][-1]["rolled_back_completed_updates"] == 2
        assert resumed_provenance["fit_status"] == "completed"
        assert resumed_provenance["current_attempt_status"] == "completed"
        assert (
            resumed_provenance["current_fit_attempt_id"]
            == resumed_provenance["training_attempts"][-1]["attempt_id"]
        )
        assert [item["status"] for item in resumed_provenance["training_attempts"]] == [
            "failed",
            "failed",
            "completed",
        ]
        assert len(resumed_provenance["failure_history"]) == 2
        assert {path.name: path.read_bytes() for path in failures} == failure_bytes

        random.seed(17)
        torch.manual_seed(17)
        baseline = _tiny_strict_training_case(tmp_path / "baseline", "run")
        (tmp_path / "baseline").mkdir(exist_ok=True)
        _run_tiny_strict_fit(baseline, resume=False)
        resumed_weights = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)[
            "model"
        ]
        baseline_weights = torch.load(
            baseline[-1] / "model.pt", map_location="cpu", weights_only=False
        )["model"]
        for name in resumed_weights:
            torch.testing.assert_close(
                resumed_weights[name], baseline_weights[name], rtol=0, atol=0
            )
    finally:
        torch.set_num_threads(prior_threads)


def test_strict_resume_binds_runtime_and_ordered_fit_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wnn.propensity_recovery import build_static_contract

    case = _tiny_strict_training_case(tmp_path, "run")
    model, optimizer, loader, args, provenance, _, split, source_root, _ = case
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    contract = build_static_contract(
        args=args,
        provenance=provenance,
        split=split,
        data_root=source_root,
        source_root=source_root,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert (
        contract["split"]["loader_fit_valid_ordered_int64_le_sha256"]
        == contract["split"]["fit_valid_ordered_int64_le_sha256"]
    )
    assert contract["runtime"]["cublas_workspace_config"] == ":4096:8"
    assert "cudnn_version" in contract["runtime"]
    assert "cudnn_deterministic" in contract["runtime"]
    assert "cudnn_benchmark" in contract["runtime"]

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    changed_runtime = build_static_contract(
        args=args,
        provenance=provenance,
        split=split,
        data_root=source_root,
        source_root=source_root,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert contract["runtime"] != changed_runtime["runtime"]

    loader.dataset.indices = [3, 2]
    with pytest.raises(ValueError, match="ordered split.valid_groups"):
        build_static_contract(
            args=args,
            provenance=provenance,
            split=split,
            data_root=source_root,
            source_root=source_root,
            loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )


def test_strict_terminal_checkpoint_can_resume_finalize_only(tmp_path: Path, monkeypatch) -> None:
    import wnn.propensity_recovery as recovery_module

    monkeypatch.setattr("wnn.propensity_fit.emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("wnn.propensity_fit_update.emit", lambda *args, **kwargs: None)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        case = _tiny_strict_training_case(tmp_path, "run")
        state_hash_calls = 0
        original_file_sha256 = recovery_module.file_sha256

        def fail_terminal_state_digest(path: Path) -> str:
            nonlocal state_hash_calls
            if path.name == "training_state.pt":
                state_hash_calls += 1
                if state_hash_calls == 3:
                    raise OSError("synthetic post-save checkpoint digest failure")
            return original_file_sha256(path)

        monkeypatch.setattr(recovery_module, "file_sha256", fail_terminal_state_digest)
        with pytest.raises(OSError, match="post-save checkpoint digest"):
            _run_tiny_strict_fit(case, resume=False)
        run_dir = case[-1]
        failure = json.loads(next(run_dir.glob("failure-*.json")).read_text())
        assert failure["committed_next_epoch"] == 2

        monkeypatch.setattr(recovery_module, "file_sha256", original_file_sha256)
        torch.manual_seed(17)
        resumed = _tiny_strict_training_case(tmp_path, "run")
        _run_tiny_strict_fit(resumed, resume=True)
        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert provenance["fit_status"] == "completed"
        assert provenance["optimizer_progress"]["optimizer_completed_updates"] == 8
        assert len(list(run_dir.glob("failure-*.json"))) == 1
    finally:
        torch.set_num_threads(prior_threads)


def test_strict_save_failure_reports_last_durable_epoch(tmp_path: Path, monkeypatch) -> None:
    import wnn.propensity_fit_checkpoint as checkpoint_module

    monkeypatch.setattr("wnn.propensity_fit.emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("wnn.propensity_fit_update.emit", lambda *args, **kwargs: None)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        case = _tiny_strict_training_case(tmp_path, "run")
        original_save = checkpoint_module.save_resume_state

        def fail_before_checkpoint(payload: dict[str, object], path: Path, *, on_saved=None) -> str:
            if path.name == "training_state.pt":
                raise OSError("synthetic checkpoint-save failure")
            return original_save(payload, path, on_saved=on_saved)

        monkeypatch.setattr(checkpoint_module, "save_resume_state", fail_before_checkpoint)
        with pytest.raises(OSError, match="checkpoint-save failure"):
            _run_tiny_strict_fit(case, resume=False)
        failure = json.loads(next(case[-1].glob("failure-*.json")).read_text())
        assert failure["committed_next_epoch"] == 0
    finally:
        torch.set_num_threads(prior_threads)


def test_strict_resume_marks_hard_interruption_without_exception_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("wnn.propensity_fit.emit", lambda *args, **kwargs: None)
    monkeypatch.setattr("wnn.propensity_fit_update.emit", lambda *args, **kwargs: None)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        interrupted = _tiny_strict_training_case(tmp_path, "run")
        optimizer = interrupted[1]
        original_step = optimizer.step

        def interrupt_after_step() -> None:
            original_step()
            raise KeyboardInterrupt("synthetic hard interruption after optimizer invocation")

        monkeypatch.setattr(optimizer, "step", interrupt_after_step)

        with pytest.raises(KeyboardInterrupt, match="hard interruption"):
            _run_tiny_strict_fit(interrupted, resume=False)
        run_dir = interrupted[-1]
        assert not list(run_dir.glob("failure-*.json"))

        torch.manual_seed(17)
        resumed = _tiny_strict_training_case(tmp_path, "run")
        _run_tiny_strict_fit(resumed, resume=True)
        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert [attempt["status"] for attempt in provenance["training_attempts"]] == [
            "interrupted_recovered",
            "completed",
        ]
        recovery = provenance["recovery_history"][-1]
        assert recovery["rolled_back_optimizer_attempts"] == 1
        assert recovery["known_rolled_back_optimizer_step_calls"] == 0
        assert recovery["unknown_inflight_optimizer_step_calls"] == 1
        assert provenance["failure_history"] == []
        assert provenance["fit_status"] == "completed"
    finally:
        torch.set_num_threads(prior_threads)


def test_strict_update_counts_and_fails_on_nonfinite_input(tmp_path: Path, monkeypatch) -> None:
    from wnn.propensity import PropensityNet
    from wnn.propensity_fit_update import _fit_update
    from wnn.propensity_progress import ProgressJournal, UpdateProgress

    monkeypatch.setattr("wnn.propensity_fit_update.emit", lambda *args, **kwargs: None)
    model = PropensityNet(n_in=2, hidden=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    journal = ProgressJournal(tmp_path / "progress.jsonl")
    cumulative, epoch = UpdateProgress(), UpdateProgress()
    with pytest.raises(FloatingPointError, match="input batch"):
        _fit_update(
            model=model,
            optimizer=optimizer,
            inputs=torch.full((1, 2, 2), float("nan")),
            target=torch.zeros((1, 2)),
            hidden=None,
            chunk_start=0,
            warmup=0,
            epoch=0,
            batch=0,
            attempt_id="tiny",
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch,
        )
    assert cumulative.optimizer_attempts == 1
    assert cumulative.optimizer_completed_updates == 0
    assert cumulative.nonfinite_inputs == 1
    assert cumulative.snapshot()["optimizer_rejected_nonfinite"] == 1
    events = ProgressJournal(tmp_path / "progress.jsonl", resume_after=journal.sequence)
    assert events.recovery_summary(0)["rolled_back_nonfinite_updates"] == 1


def test_strict_update_rejects_overflowed_norm_before_step(tmp_path: Path, monkeypatch) -> None:
    from wnn.propensity import PropensityNet
    from wnn.propensity_fit_update import _fit_update
    from wnn.propensity_progress import ProgressJournal, UpdateProgress

    model = PropensityNet(n_in=2, hidden=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    step_calls = 0
    original_step = optimizer.step

    def tracked_step():
        nonlocal step_calls
        step_calls += 1
        return original_step()

    monkeypatch.setattr(optimizer, "step", tracked_step)
    monkeypatch.setattr(
        torch.linalg,
        "vector_norm",
        lambda *_args, **_kwargs: torch.tensor(float("inf")),
    )
    journal = ProgressJournal(tmp_path / "progress.jsonl")
    cumulative, epoch = UpdateProgress(), UpdateProgress()
    with pytest.raises(FloatingPointError, match="gradient norm"):
        _fit_update(
            model=model,
            optimizer=optimizer,
            inputs=torch.ones((1, 2, 2)),
            target=torch.zeros((1, 2)),
            hidden=None,
            chunk_start=0,
            warmup=0,
            epoch=0,
            batch=0,
            attempt_id="tiny-norm",
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch,
        )
    assert step_calls == 0
    assert cumulative.optimizer_step_calls == 0
    assert cumulative.nonfinite_gradient_norms == 1


def test_strict_update_marks_post_step_optimizer_state_failure(tmp_path: Path, monkeypatch) -> None:
    from wnn.propensity import PropensityNet
    from wnn.propensity_fit_update import _fit_update
    from wnn.propensity_progress import ProgressJournal, UpdateProgress

    monkeypatch.setattr("wnn.propensity_fit_update.emit", lambda *args, **kwargs: None)
    model = PropensityNet(n_in=2, hidden=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original_step = optimizer.step

    def corrupt_state():
        result = original_step()
        for state in optimizer.state.values():
            state["exp_avg"].fill_(float("nan"))
        return result

    monkeypatch.setattr(optimizer, "step", corrupt_state)
    journal = ProgressJournal(tmp_path / "progress.jsonl")
    cumulative, epoch = UpdateProgress(), UpdateProgress()
    with pytest.raises(FloatingPointError, match="optimizer state"):
        _fit_update(
            model=model,
            optimizer=optimizer,
            inputs=torch.ones((1, 2, 2)),
            target=torch.zeros((1, 2)),
            hidden=None,
            chunk_start=0,
            warmup=0,
            epoch=0,
            batch=0,
            attempt_id="tiny-state",
            journal=journal,
            cumulative=cumulative,
            epoch_progress=epoch,
        )
    assert cumulative.optimizer_step_calls == 1
    assert cumulative.optimizer_completed_updates == 0
    assert cumulative.nonfinite_optimizer_states == 1
    rejected = [event for event in journal._events if event["event"] == "optimizer_update_rejected"]
    assert rejected[0]["optimizer_step_called"] is True
