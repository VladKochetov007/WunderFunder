from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from archive_safe_state import archive_once
from migrate_monitor_checkpoint import clone_checkpoint, prepare_output
from prepare_gru_archive_source import validator_source
from test_transition_constraints import identities, safe_state
from transition_support import copy_prefix, same_payload, validate_transition

from wnn.training.sequence_monitor_events import MonitorEventWriter
from wnn.training.sequence_provenance import canonical_sha256, file_sha256, recorder_for_run


def recipe_fixture() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    """Provide a nonnumerical config amendment over tiny tensor-only checkpoint fixtures."""
    old, new, config = identities()
    validator = "src/wnn/training/sequence_config_validation.py"
    old["source_files"][validator] = "0" * 64
    new["source_files"][validator] = "9" * 64
    for identity in (old, new):
        identity["source_sha256"] = canonical_sha256(identity["source_files"])
    config |= {"batch_size": 16, "model": {"architecture": "gru"}, "monitoring": {}}
    amended = config | {"checkpoint_retention": "final_only"}
    old["config_sha256"] = canonical_sha256(config)
    new["config_sha256"] = canonical_sha256(amended)
    return old, new, config, amended


def source_run(folder: Path, identity: dict[str, object], config: dict[str, object]) -> str:
    """Create complete native monitor prefixes plus intentionally unsafe attempt tails."""
    folder.mkdir()
    recorder_for_run(folder, config, identity)
    for name in ("rank0.jsonl", "metrics.jsonl"):
        (folder / name).write_bytes(b'{"step": 2}\n')
    writer = MonitorEventWriter(folder / "monitor_events.jsonl", resume=False)
    writer.append(
        "evaluation",
        2,
        1,
        {
            "valid/sequence_count": 1873,
            "valid/selected_rows": 3528273,
            "valid/wp": 0.1,
        },
        {"run_identity_sha256": canonical_sha256(identity)},
    )
    state = safe_state(identity)
    state["rank_log_offsets"] = [(folder / "rank0.jsonl").stat().st_size]
    state["metrics_offset"] = (folder / "metrics.jsonl").stat().st_size
    state["monitor_offsets"] = writer.offsets()
    torch.save(state, folder / "training_state.pt")
    for name in (
        "rank0.jsonl",
        "metrics.jsonl",
        "monitor_events.jsonl",
        "monitor_evaluations.jsonl",
    ):
        with (folder / name).open("ab") as stream:
            stream.write(b"incomplete unsafe next-epoch tail")
    return file_sha256(folder / "training_state.pt")


def test_retention_transition_is_exact_and_scientific_drift_refused() -> None:
    old, new, config, amended = recipe_fixture()
    validate_transition(config, amended, old, new)
    for field, value in (("seed", 101002), ("batch_size", 32), ("epochs", 13), ("lr", 0.1)):
        with pytest.raises(ValueError, match="configuration changed"):
            validate_transition(config, amended | {field: value}, old, new)
    for field in ("runtime", "data", "world_size"):
        changed = copy.deepcopy(new)
        changed[field] = "changed"
        with pytest.raises(ValueError, match="outside whitelist"):
            validate_transition(config, amended, old, changed)


def test_additional_source_edit_refused() -> None:
    old, new, config, amended = recipe_fixture()
    new["source_files"]["scripts/train_sequence.py"] = "f" * 64
    new["source_sha256"] = canonical_sha256(new["source_files"])
    with pytest.raises(ValueError, match="diagnostic whitelist"):
        validate_transition(config, amended, old, new)


def test_canonical_config_admission_for_both_actual_recipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the S3 validator through canonical load_config without initializing CUDA."""
    import wnn.training.sequence_config as canonical
    import wnn.training.sequence_config_validation as validation

    namespace: dict[str, object] = {}
    source = Path(validation.__file__)
    exec(compile(validator_source(source.read_text()), str(source), "exec"), namespace)
    monkeypatch.setattr(
        canonical, "validate_monitoring_config", namespace["validate_monitoring_config"]
    )
    root = Path(__file__).parent / "continuation-s3-r3/configs"
    for seed in (101001, 101002):
        path = root / f"gru_s{seed}.json"
        expected = json.loads(path.read_bytes())
        expected["model"]["precision_mode"] = "fp32"
        assert canonical.load_config(path) == expected
    monitor = expected["monitoring"] | {
        "early_stopping": {"min_epoch": 5, "patience": 4, "min_delta": 0.0005}
    }
    with pytest.raises(ValueError, match="checkpoint_retention=all_epochs"):
        namespace["validate_monitoring_config"](monitor, expected["model"], 12, "final_only")
    namespace["validate_monitoring_config"](monitor, expected["model"], 12, "all_epochs")


def test_frozen_s1_validator_refuses_final_only_extended_monitoring() -> None:
    """Keep the original route refusal reproducible; S1/S2 source was never edited."""
    from wnn.training.sequence_config_validation import validate_monitoring_config

    config = json.loads(
        (Path(__file__).parent / "continuation-s3-r3/configs/gru_s101001.json").read_bytes()
    )
    with pytest.raises(ValueError, match="checkpoint_retention=all_epochs"):
        validate_monitoring_config(config["monitoring"], config["model"], 12, "final_only")


def test_prepare_canonical_output_preserves_old_rng_tensors_and_byte_prefixes(
    tmp_path: Path,
) -> None:
    old, new, config, amended = recipe_fixture()
    before, after = tmp_path / "old-run", tmp_path / "continued-run"
    sha = source_run(before, old, config)
    old_files = {p.name: p.read_bytes() for p in before.iterdir()}
    proof = prepare_output(before, after, amended, old, new, sha)
    assert proof["next_epoch"] == 1 and proof["global_step"] == 2
    original = torch.load(before / "training_state.pt", weights_only=False)
    cloned = torch.load(after / "training_state.pt", weights_only=False)
    for key in original.keys() - {"contract"}:
        assert same_payload(original[key], cloned[key])
    for name, item in proof["logs"].items():
        assert (after / name).read_bytes() == old_files[name][: item["bytes"]]
    assert {p.name: p.read_bytes() for p in before.iterdir()} == old_files
    # This invokes the unchanged canonical immutable-metadata equality check.
    recorder_for_run(after, amended, new)
    assert not (after / "epoch_index.json").exists()
    assert not (after / "epoch_checkpoints").exists()
    with pytest.raises(FileExistsError):
        prepare_output(before, after, amended, old, new, sha)


@pytest.mark.parametrize("offset", [-1, True, 4, 1 << 31])
def test_prefix_copy_refuses_invalid_or_partial_offsets(tmp_path: Path, offset: int) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"a": 1}\n')
    with pytest.raises(ValueError):
        copy_prefix(source, tmp_path / "copy.jsonl", offset)
    assert source.read_bytes() == b'{"a": 1}\n'


def test_checkpoint_parent_hash_refusal_leaves_original_unchanged(tmp_path: Path) -> None:
    old, new, _config, _amended = recipe_fixture()
    path = tmp_path / "source.pt"
    torch.save(safe_state(old), path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="stopped-boundary pin"):
        clone_checkpoint(path, tmp_path / "new.pt", old, new, "0" * 64)
    assert path.read_bytes() == before and not (tmp_path / "new.pt").exists()


def test_archive_is_immutable_and_refuses_missed_epoch(tmp_path: Path) -> None:
    _old, new, _config, _amended = recipe_fixture()
    output = tmp_path / "run"
    output.mkdir()
    path = output / "training_state.pt"
    state = safe_state(new)
    torch.save(state, path)
    archive = output / "safe_state_archive"
    assert archive_once(output, archive, new, "1" * 64, 1) == (1, 2)
    proof = json.loads((archive / "epoch-000001.json").read_bytes())
    checkpoint = archive / proof["checkpoint_path"]
    checkpoint_bytes = checkpoint.read_bytes()
    assert archive_once(output, archive, new, "1" * 64, 1) == (1, 2)
    state["next_epoch"] = 3
    state["global_step"] = 6
    state["history"] = [{"epoch": e, "complete": True, "updates": 2} for e in range(3)]
    torch.save(state, path)
    with pytest.raises(ValueError, match="missed or reordered"):
        archive_once(output, archive, new, "1" * 64, 2)
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert file_sha256(checkpoint) == proof["checkpoint_sha256"]
