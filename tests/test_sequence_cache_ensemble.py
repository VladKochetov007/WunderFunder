from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wnn.splits import CampaignSplit, file_sha256
from wnn.training import sequence_cache_ensemble
from wnn.training.sequence_cache import ENSEMBLE_KIND, TeacherCache
from wnn.training.sequence_cache_ensemble import merge_teacher_caches

SPLIT_HASH = "a" * 64
MEMBERS = [("train", 0), ("valid", 1)]
SPLIT = CampaignSplit(
    Path("split.json"),
    SPLIT_HASH,
    0,
    3,
    (0,),
    (1,),
    (2,),
    (("train", 0, 0), ("valid", 1, 1), ("valid", 2, 2)),
)


def _canonical_sha(value: object) -> str:
    """Hash fixture provenance using the production canonical JSON encoding."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _bf16_bf16x9_identity(attention_backend: str) -> dict[str, object]:
    """Represent PR40's BF16x9 precision identity without importing training code."""
    return {
        "mode": "bf16_bf16x9",
        "attention_backend": attention_backend,
        "cublas_emulate_single_precision": "1",
        "parameter_dtype": "float32",
        "gradient_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "activation_cache_dtype": "float32",
        "amp_scaler": False,
        "model_compute": "BF16 autocast plus process-wide cuBLAS single-precision emulation",
        "model_forward": "FP32 tensor interface with BF16 autocast and cuBLAS emulation",
        "loss": "FP32 after prediction and hint casts",
        "optimizer": "process-wide cuBLAS emulation may affect Muon operations",
    }


def _producer(
    seed: int,
    checkpoint: str,
    members: list[tuple[str, int]],
    recipe_change: dict[str, Any] | None = None,
    precision_mode: str | None = None,
    compile_blocks: bool = False,
    attention_backend: str = "math",
) -> dict[str, Any]:
    """Build a complete source-bound fixed-fit producer with seed-local device UUIDs."""
    fit_members = [list(member) for member in members]
    config: dict[str, Any] = {"seed": seed, "epochs": 30, "model": {"hint_dim": 0}}
    if precision_mode is not None:
        config["precision_mode"] = precision_mode
        config["model"].update(
            {
                "precision_mode": precision_mode,
                "attention_backend": attention_backend,
                "compile_blocks": compile_blocks,
            }
        )
    if recipe_change:
        config.update(recipe_change)
    data = {
        "split_sha256": SPLIT_HASH,
        "fit_members": fit_members,
        "input_schema": "raw112-v1",
        "mask": "is_scored AND row_index>=99",
        "training_mask": "is_scored",
        "normalization": {
            "scope": "all_fit",
            "sequence_count": len(members),
            "source_fingerprints": {
                "datasets/train.parquet": {"sha256": "2" * 64},
                "datasets/valid.parquet": {"sha256": "3" * 64},
            },
        },
        "sources": {
            "datasets/train.parquet": {"sha256": "2" * 64},
            "datasets/valid.parquet": {"sha256": "3" * 64},
        },
        "teacher": None,
    }
    devices = [
        {
            "rank": rank,
            "local_rank": rank,
            "name": "NVIDIA B300",
            "capability": [10, 0],
            "uuid": f"GPU-{seed}-{rank}",
            "total_memory_bytes": 100,
        }
        for rank in range(8)
    ]
    source_files = {"src/wnn/training/sequence_fit.py": "4" * 64}
    runtime = {
        "python": "3.12.0",
        "platform": "Linux-test",
        "torch": "2.8.0",
        "cuda_runtime": "12.8",
        "cudnn": 90000,
        "nvidia_driver": "580.0",
        "visible_devices": 8,
        "world_size": 8,
        "matmul_precision": "highest",
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast": False,
        "amp_scaler": False,
        "attention_backend": "math",
        "cublas_workspace_config": ":4096:8",
        "devices": devices,
    }
    if precision_mode == "bf16_bf16x9":
        precision = _bf16_bf16x9_identity(attention_backend)
        runtime.update(
            {
                "attention_backend": attention_backend,
                "precision": precision,
                "compiler": {
                    "enabled": compile_blocks,
                    "backend": "inductor" if compile_blocks else None,
                    "mode": "default" if compile_blocks else None,
                    "fullgraph": False,
                    "dynamic": None,
                    "recompile_limit": 64 if compile_blocks else None,
                },
            }
        )
    else:
        precision = "strict_fp32_no_autocast_no_tf32_math_sdpa"
    identity = {
        "schema": "wnn.sequence-run-identity.v1",
        "config_sha256": _canonical_sha(config),
        "data_sha256": _canonical_sha(data),
        "data": data,
        "source_sha256": _canonical_sha(source_files),
        "source_files": source_files,
        "runtime": runtime,
        "world_size": 8,
        "model_parameters": {"total": 1, "trainable": 1},
        "optimizer": {"name": "adamw", "schedule_updates": 2},
    }
    provenance = {
        "schema": "wnn.sequence-provenance.v1",
        "identity": identity,
        "normalization": {"mean": [0.0] * 112, "std": [1.0] * 112},
        "profile_receipt": {"path": f"/seed-{seed}/profile.json", "sha256": "5" * 64},
        "history": [{"seed": seed}],
        "fixed_checkpoint_policy": "configured-final-epoch",
        "completed_epochs": 30,
        "global_updates": 60,
        "precision": precision,
    }
    return {
        "checkpoint_sha256": checkpoint,
        "split_sha256": SPLIT_HASH,
        "fit_members": fit_members,
        "teacher": None,
        "config": config,
        "provenance": provenance,
    }


def _write_cache(
    root: Path,
    seed: int,
    checkpoint: str,
    values: np.ndarray,
    *,
    members: list[tuple[str, int]] = MEMBERS,
    recipe_change: dict[str, Any] | None = None,
    precision_mode: str | None = None,
    compile_blocks: bool = False,
    attention_backend: str = "math",
) -> Path:
    """Write a complete minimal canonical fit cache for receiver checks."""
    root.mkdir()
    producer = _producer(
        seed,
        checkpoint,
        members,
        recipe_change,
        precision_mode,
        compile_blocks,
        attention_backend,
    )
    entries = []
    for source, group in MEMBERS:
        path = root / f"{source}-{group}.npy"
        np.save(path, values, allow_pickle=False)
        entries.append(
            {"split": source, "row_group": group, "file": path.name, "sha256": file_sha256(path)}
        )
    manifest = {
        "schema": "sequence-teacher-cache-v1",
        "complete": True,
        "input_schema": "raw112-v1",
        "membership": "fit",
        "split_sha256": SPLIT_HASH,
        "hint_dim": 0,
        "sequence_length": 20_000,
        "producer": producer,
        "sequences": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    change: tuple[int, dict[str, Any]] | None = None,
    normalizer_change: bool = False,
    duplicate_checkpoint: bool = False,
    precision_mode: str | None = None,
    compile_blocks: bool = False,
    attention_backend: str = "math",
) -> list[Path]:
    """Create three cache roots and bind the canonical split loader for this fixture."""
    monkeypatch.setattr(
        "wnn.training.sequence_cache_ensemble.load_campaign_split",
        lambda _manifest, _root: SPLIT,
    )
    base = np.zeros((20_000, 2), dtype=np.float32)
    roots = []
    for index, seed in enumerate((11, 12, 13)):
        checkpoint = "1" * 64 if duplicate_checkpoint and index == 2 else f"{index + 1:x}" * 64
        recipe_change = change[1] if change is not None and change[0] == index else None
        roots.append(
            _write_cache(
                tmp_path / f"cache-{index}",
                seed,
                checkpoint,
                base,
                recipe_change=recipe_change,
                precision_mode=precision_mode,
                compile_blocks=compile_blocks,
                attention_backend=attention_backend,
            )
        )
    if normalizer_change:
        manifest_path = roots[2] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["producer"]["provenance"]["normalization"]["mean"][0] = 1.0
        manifest_path.write_text(json.dumps(manifest))
    return roots


def test_merge_streams_fp32_mean_and_publishes_ensemble_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    values_by_seed = (3.0, 3.0, -3.0)
    for index, root in enumerate(roots):
        manifest = json.loads((root / "manifest.json").read_text())
        for entry in manifest["sequences"]:
            np.save(
                root / entry["file"],
                np.full((20_000, 2), values_by_seed[index], dtype=np.float32),
            )
            entry["sha256"] = file_sha256(root / entry["file"])
        (root / "manifest.json").write_text(json.dumps(manifest))
    output = merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")

    result = np.load(output / "train-0.npy", allow_pickle=False)
    assert result.dtype == np.float32 and np.all(result == 1.0)
    manifest = json.loads((output / "manifest.json").read_text())
    producer = manifest["producer"]
    assert producer["kind"] == ENSEMBLE_KIND
    assert "checkpoint_sha256" not in producer
    assert len(producer["parents"]) == 3
    assert producer["shared_provenance_sha256"]
    assert (
        len(
            {
                parent["provenance"]["identity"]["runtime"]["devices"][0]["uuid"]
                for parent in producer["parents"]
            }
        )
        == 3
    )
    assert (
        producer["composition_sha256"]
        == hashlib.sha256(
            json.dumps(
                producer["composition"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    cache = TeacherCache(output, SPLIT, MEMBERS)
    assert "composition_sha256" in cache.identity
    assert "checkpoint_sha256" not in cache.identity


def test_accepts_pr40_bf16_bf16x9_provenance_and_keeps_mean_fp32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(
        tmp_path,
        monkeypatch,
        precision_mode="bf16_bf16x9",
        compile_blocks=True,
        attention_backend="efficient",
    )
    output = merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")

    values = np.load(output / "train-0.npy", allow_pickle=False)
    assert values.dtype == np.float32 and np.all(values == 0.0)
    producer = json.loads((output / "manifest.json").read_text())["producer"]
    for parent in producer["parents"]:
        provenance = parent["provenance"]
        runtime = provenance["identity"]["runtime"]
        assert provenance["precision"]["mode"] == "bf16_bf16x9"
        assert runtime["precision"] == provenance["precision"]
        assert runtime["compiler"] == {
            "enabled": True,
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": None,
            "recompile_limit": 64,
        }


@pytest.mark.parametrize("tamper", ("precision", "compiler", "config", "model_config"))
def test_rejects_bf16_precision_detached_from_config_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    roots = _caches(
        tmp_path,
        monkeypatch,
        precision_mode="bf16_bf16x9",
        compile_blocks=True,
    )
    if tamper in ("config", "model_config"):
        for root in roots:
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            producer = manifest["producer"]
            config = producer["config"]
            if tamper == "config":
                config["precision_mode"] = "fp32"
            else:
                config["model"]["precision_mode"] = "fp32"
            producer["provenance"]["identity"]["config_sha256"] = _canonical_sha(config)
            manifest_path.write_text(json.dumps(manifest))
    else:
        manifest_path = roots[0] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        provenance = manifest["producer"]["provenance"]
        if tamper == "precision":
            provenance["precision"]["mode"] = "bf16"
        else:
            provenance["identity"]["runtime"]["compiler"]["recompile_limit"] = 63
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="precision|compiler"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")
    assert not (tmp_path / "mean").exists()


def test_rejects_parent_outer_fold_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = _caches(tmp_path, monkeypatch)
    manifest_path = roots[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["producer"]["fit_members"] = [["train", 0], ["valid", 2]]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="membership"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")


@pytest.mark.parametrize("change", [{"lr": 0.5}, {"seed": 11}])
def test_rejects_mixed_recipe_or_repeated_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    roots = _caches(tmp_path, monkeypatch, change=(2, change))
    with pytest.raises(ValueError, match="seed|recipe"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")


def test_rejects_duplicate_checkpoint_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch, duplicate_checkpoint=True)
    with pytest.raises(ValueError, match="distinct checkpoints"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")


def test_rejects_two_members_pointing_to_one_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    path = roots[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    first, second = manifest["sequences"]
    second["file"], second["sha256"] = first["file"], first["sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="distinct array files"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")


def test_rejects_parent_provenance_copy_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    output = merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["producer"]["parents"][0]["provenance"]["profile_receipt"]["path"] = (
        "/detached/profile.json"
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="provenance reference"):
        TeacherCache(output, SPLIT, MEMBERS)


def test_refuses_manifest_larger_than_consumer_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequence_cache_ensemble, "MAX_CACHE_MANIFEST_BYTES", 64)
    with pytest.raises(ValueError, match="consumer size limit"):
        sequence_cache_ensemble._write_manifest(tmp_path, {"payload": "x" * 128})
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "manifest.json.partial").exists()


def test_accepts_distinct_b300_uuids_but_rejects_different_normalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch, normalizer_change=True)
    with pytest.raises(ValueError, match="shared fit provenance"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")

    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    roots = _caches(valid_root, monkeypatch)
    output = merge_teacher_caches(
        roots, valid_root, tmp_path / "split.json", valid_root / "valid-mean"
    )
    producer = json.loads((output / "manifest.json").read_text())["producer"]
    uuids = {
        device["uuid"]
        for parent in producer["parents"]
        for device in parent["provenance"]["identity"]["runtime"]["devices"]
    }
    assert len(uuids) == 24


def test_rechecks_parent_cache_after_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    original = sequence_cache_ensemble._mean_sequence
    state = {"changed": False}

    def mutate_after_first_sequence(
        caches: list[TeacherCache], member: tuple[str, int]
    ) -> np.ndarray:
        values = original(caches, member)
        if not state["changed"]:
            path = roots[0] / "train-0.npy"
            with path.open("ab") as stream:
                stream.write(b"mutation")
            state["changed"] = True
        return values

    monkeypatch.setattr(sequence_cache_ensemble, "_mean_sequence", mutate_after_first_sequence)
    with pytest.raises(ValueError, match="changed during merge"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")
    assert not (tmp_path / "mean").exists()


@pytest.mark.parametrize("corruption", ["truncated", "nonfinite", "pathescape", "dtype"])
def test_rejects_malformed_parent_arrays_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    root = roots[0]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["sequences"][0]
    array_path = root / entry["file"]
    if corruption == "truncated":
        array_path.write_bytes(array_path.read_bytes()[:100])
        entry["sha256"] = file_sha256(array_path)
    elif corruption == "nonfinite":
        array = np.load(array_path)
        array[0, 0] = np.nan
        np.save(array_path, array, allow_pickle=False)
        entry["sha256"] = file_sha256(array_path)
    elif corruption == "dtype":
        np.save(array_path, np.zeros((20_000, 2), dtype=np.float64), allow_pickle=False)
        entry["sha256"] = file_sha256(array_path)
    else:
        outside = tmp_path / "outside.npy"
        np.save(outside, np.zeros((20_000, 2), dtype=np.float32), allow_pickle=False)
        entry["file"] = "../outside.npy"
        entry["sha256"] = file_sha256(outside)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises((ValueError, OSError, EOFError)):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")
    assert not (tmp_path / "mean").exists()


def test_rejects_mixed_ensemble_and_single_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    path = roots[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["producer"]["kind"] = ENSEMBLE_KIND
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ensemble|producer"):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", tmp_path / "mean")


def test_rejects_existing_output_and_requires_exactly_three_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _caches(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly three"):
        merge_teacher_caches(roots[:2], tmp_path, tmp_path / "split.json", tmp_path / "mean")
    output = tmp_path / "mean"
    output.mkdir()
    with pytest.raises(FileExistsError):
        merge_teacher_caches(roots, tmp_path, tmp_path / "split.json", output)
