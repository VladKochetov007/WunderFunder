from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts import build_distill_cache as builder
from wnn.cleanfold_t3_cache import combine_cleanfold_t3_chunk, fit_prediction_positions
from wnn.distillation import TeacherSource
from wnn.distillation_cache import SEQUENCE_ROWS, validate_distillation_cache
from wnn.splits import CampaignSplit


def test_t3_combiner_scales_raw_fp32_mean_once_without_clipping() -> None:
    values = [np.full((1, 2, 2), 3.0, dtype=np.float32) for _ in range(3)]

    result = combine_cleanfold_t3_chunk(values)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.full_like(result, 2.4375))
    assert np.all(result > 2.0)
    assert all(np.all(value == 3.0) for value in values)


def test_t3_combiner_uses_explicit_source_order_for_fp32_arithmetic() -> None:
    values = [
        np.full((1, 1, 2), np.float32(2**24), dtype=np.float32),
        np.full((1, 1, 2), np.float32(-(2**24)), dtype=np.float32),
        np.full((1, 1, 2), np.float32(1.0), dtype=np.float32),
    ]

    result = combine_cleanfold_t3_chunk(values)

    expected = np.float32(np.float32(np.float32(1.0) / np.float32(3.0)) * np.float32(0.8125))
    np.testing.assert_array_equal(result, np.full_like(result, expected))


def test_t3_combiner_rejects_fp32_overflow_before_returning_chunk() -> None:
    maximum = np.finfo(np.float32).max
    predictions = [np.full((1, 1, 2), maximum, dtype=np.float32) for _ in range(3)]

    with pytest.raises(ValueError, match="produced nonfinite values"):
        combine_cleanfold_t3_chunk(predictions)


def test_fit_prediction_positions_reorders_only_the_exact_fit_membership() -> None:
    groups, positions = fit_prediction_positions([10, 12], [12, 99, 11, 10], [10, 12])

    assert groups == (12, 10)
    np.testing.assert_array_equal(positions, [1, 0])


def test_target_fit_order_digest_matches_permuted_prediction_mapping() -> None:
    target_groups = [12, 99, 11, 10]
    expected_fit_groups = (10, 12)
    prediction_groups = [10, 12]

    target_fit_order = builder._ordered_fit_groups(target_groups, frozenset(expected_fit_groups))
    prediction_fit_order, positions = fit_prediction_positions(
        prediction_groups, target_groups, expected_fit_groups
    )

    assert target_fit_order == prediction_fit_order == (12, 10)
    np.testing.assert_array_equal(positions, [1, 0])
    assert builder.row_groups_sha256(list(target_fit_order)) == (
        "64ac415e7e58f26540f04791fc97156f1ca27b21989c3c73d1f96c4e0305e765"
    )


@pytest.mark.parametrize(
    ("predicted", "message"),
    [
        ([10], "missing"),
        ([10, 12, 99], "outer"),
        ([10, 10, 12], "repeats"),
    ],
)
def test_fit_prediction_positions_rejects_bad_membership(
    predicted: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_prediction_positions(predicted, [10, 11, 12, 99], [10, 12])


@pytest.mark.parametrize(
    ("identity_list", "position", "invalid"),
    [
        ("prediction", 0, True),
        ("prediction", 0, 10.0),
        ("prediction", 0, "10"),
        ("target", 1, True),
        ("target", 1, 11.0),
        ("target", 1, "11"),
        ("expected", 0, True),
        ("expected", 0, 10.0),
        ("expected", 0, "10"),
    ],
)
def test_fit_prediction_positions_rejects_identity_coercion(
    identity_list: str, position: int, invalid: object
) -> None:
    identities = {
        "prediction": [10, 12],
        "target": [10, 11, 12, 99],
        "expected": [10, 12],
    }
    identities[identity_list][position] = invalid

    with pytest.raises(ValueError, match="integer row-group identities"):
        fit_prediction_positions(
            identities["prediction"], identities["target"], identities["expected"]
        )


@pytest.mark.parametrize("row_groups", [[True], [10.0], ["10"], None])
def test_fit_sidecar_groups_requires_json_integer_values(row_groups: object) -> None:
    with pytest.raises(ValueError, match="must be JSON integers"):
        builder._fit_sidecar_groups({"row_groups": row_groups}, "run", "train")


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _synthetic_t3_contract(
    tmp_path: Path,
) -> tuple[Path, str, Path, Path, CampaignSplit]:
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    acceptance_root = tmp_path / "campaign"
    checkpoint_root = tmp_path / "runs"
    teachers = []
    for seed in (3, 4, 5):
        run = f"fg19_t3_h384x2_e3_s{seed}_v1"
        acceptance_path = Path("records") / f"teacher_s{seed}_v1" / "TERMINAL_ACCEPTANCE.json"
        teacher = {
            "run": run,
            "seed": seed,
            "epoch": 2,
            "acceptance_path": str(acceptance_path),
            "acceptance_sha256": _write(
                acceptance_root / acceptance_path, f"acceptance-{seed}".encode()
            ),
            "model_sha256": _write(checkpoint_root / run / "model.pt", f"model-{seed}".encode()),
            "checkpoint_sha256": _write(
                checkpoint_root / run / "model_ep2.pt", f"checkpoint-{seed}".encode()
            ),
            "provenance_sha256": _write(
                checkpoint_root / run / "provenance.json", f"provenance-{seed}".encode()
            ),
            "tensor_fingerprint": f"{seed:064x}",
        }
        teachers.append(teacher)
    contract = {
        "schema": "fg19.teacher-cache-execution-contract.v1",
        "source_teacher_commit": "af21d6d8409c46b36830f0bdddd50e67ee17025f",
        "split_sha256": split.sha256,
        "teacher_order": [3, 4, 5],
        "arithmetic": {
            "mean": "Unweighted arithmetic mean of S3,S4,S5 in FP32, fixed source order.",
            "scale": 0.8125,
            "scaling_count": 1,
            "cache_clipping": "none",
            "consumer_cache_scale": 1.0,
        },
        "membership": {"train": 1, "valid": 1, "evaluation_inferred": 0},
        "teachers": teachers,
    }
    contract_path = tmp_path / "execution-contract.json"
    raw = json.dumps(contract).encode()
    contract_path.write_bytes(raw)
    return (
        contract_path,
        hashlib.sha256(raw).hexdigest(),
        acceptance_root,
        checkpoint_root,
        split,
    )


def test_root_t3_contract_binds_acceptance_and_selected_checkpoint_bytes(tmp_path: Path) -> None:
    contract_path, contract_sha, acceptance_root, checkpoint_root, split = _synthetic_t3_contract(
        tmp_path
    )

    pins, contract = builder._cleanfold_t3_pins(
        contract_path, contract_sha, acceptance_root, checkpoint_root, split
    )

    assert [item["run"] for item in pins] == [
        "fg19_t3_h384x2_e3_s3_v1",
        "fg19_t3_h384x2_e3_s4_v1",
        "fg19_t3_h384x2_e3_s5_v1",
    ]
    assert contract["schema"] == "fg19.teacher-cache-execution-contract.v1"
    provenance_path = checkpoint_root / pins[0]["run"] / "provenance.json"
    original_provenance = provenance_path.read_bytes()
    oversized_provenance = b"{}" + b" " * (builder.CLEANFOLD_T3_PROVENANCE_LIMIT + 1 - len(b"{}"))
    provenance_path.write_bytes(oversized_provenance)
    with pytest.raises(ValueError, match="provenance exceeds its bounded read size"):
        builder._cleanfold_t3_pins(
            contract_path, contract_sha, acceptance_root, checkpoint_root, split
        )
    provenance_path.write_bytes(original_provenance)
    (checkpoint_root / pins[0]["run"] / "model_ep2.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint or provenance bytes disagree"):
        builder._cleanfold_t3_pins(
            contract_path, contract_sha, acceptance_root, checkpoint_root, split
        )


@pytest.mark.parametrize(
    "provenance_size",
    [
        0,
        builder.CLEANFOLD_T3_CONTRACT_LIMIT + 1,
        166_192,
        builder.CLEANFOLD_T3_PROVENANCE_LIMIT,
    ],
)
def test_t3_inspection_uses_root_pins_without_deserializing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provenance_size: int
) -> None:
    run = builder.T3_TEACHER_ORDER[0]
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    run_root = tmp_path / "runs" / run
    run_root.mkdir(parents=True)
    model_sha = _write(run_root / "model.pt", b"selected model")
    checkpoint_sha = _write(run_root / "model_ep2.pt", b"selected epoch two")
    provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": [10], "valid": [20]},
        "checkpoint_sha256": {"model.pt": model_sha, "model_ep2.pt": checkpoint_sha},
        "args": {
            "hidden": 384,
            "layers": 2,
            "cell": "gru",
            "seed": 3,
            "split_manifest_sha256": split.sha256,
        },
    }
    provenance_raw = json.dumps(provenance).encode()
    if provenance_size:
        assert provenance_size >= len(provenance_raw)
        provenance_raw += b" " * (provenance_size - len(provenance_raw))
    provenance_sha = _write(run_root / "provenance.json", provenance_raw)
    cache_root = tmp_path / "predictions"
    cache_root.mkdir()
    target_orders = {"train": [10, 11], "valid": [20, 21]}
    for split_name, group in (("train", 10), ("valid", 20)):
        prediction_path = cache_root / f"pred_{run}_{split_name}.npy"
        np.save(
            prediction_path,
            np.full((1, SEQUENCE_ROWS, 2), np.float32(group), dtype=np.float32),
            allow_pickle=False,
        )
        sidecar = {
            "run": run,
            "split": split_name,
            "epoch": 2,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": builder.file_sha256(prediction_path),
            "split_manifest_sha256": split.sha256,
            "campaign_split_manifest_sha256": split.sha256,
            "campaign_export_scope": "fit",
            "feature_only": True,
            "row_groups": [group],
        }
        prediction_path.with_suffix(".json").write_text(json.dumps(sidecar))
    monkeypatch.setattr(
        builder.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "T3 cache inspection must not deserialize checkpoints"
        ),
    )
    pin = {
        "model_sha256": model_sha,
        "checkpoint_sha256": checkpoint_sha,
        "provenance_sha256": provenance_sha,
        "acceptance_sha256": "b" * 64,
        "tensor_fingerprint": "c" * 64,
    }

    record, arrays, sidecars = builder._inspect_teacher(
        run, split, cache_root, tmp_path / "runs", target_orders, 2, fit_only=True, root_pin=pin
    )

    assert record["checkpoint_sha256"] == checkpoint_sha
    assert record["fit_group_counts"] == {"train": 1, "valid": 1}
    assert record["acceptance_sha256"] == "b" * 64
    assert arrays["train"].shape == (1, SEQUENCE_ROWS, 2)
    assert sidecars["train"]["_row_groups"] == [10]


def test_t3_inspection_rejects_provenance_above_one_megabyte_without_checkpoint_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = builder.T3_TEACHER_ORDER[0]
    run_root = tmp_path / "runs" / run
    run_root.mkdir(parents=True)
    oversized = b"{}" + b" " * (builder.CLEANFOLD_T3_PROVENANCE_LIMIT - 1)
    _write(run_root / "provenance.json", oversized)
    (run_root / "model_ep2.pt").write_bytes(b"pinned checkpoint")
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    monkeypatch.setattr(
        builder.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "root-pinned T3 inspection must not deserialize checkpoints"
        ),
    )

    with pytest.raises(ValueError, match="provenance exceeds its bounded read size"):
        builder._inspect_teacher(
            run,
            split,
            tmp_path / "predictions",
            tmp_path / "runs",
            {},
            2,
            fit_only=True,
            root_pin={"provenance_sha256": "a" * 64},
        )


def test_t3_inspection_rejects_invalid_bounded_provenance_json_without_checkpoint_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = builder.T3_TEACHER_ORDER[0]
    run_root = tmp_path / "runs" / run
    run_root.mkdir(parents=True)
    invalid = b"{not-json"
    provenance_sha = _write(run_root / "provenance.json", invalid)
    (run_root / "model_ep2.pt").write_bytes(b"pinned checkpoint")
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    monkeypatch.setattr(
        builder.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "root-pinned T3 inspection must not deserialize checkpoints"
        ),
    )

    with pytest.raises(json.JSONDecodeError):
        builder._inspect_teacher(
            run,
            split,
            tmp_path / "predictions",
            tmp_path / "runs",
            {},
            2,
            fit_only=True,
            root_pin={"provenance_sha256": provenance_sha},
        )


def test_legacy_ancestor_audit_still_verifies_checkpoint_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    run = "fg19_t3_h384x2_e3_s3_v1"
    run_root = tmp_path / "runs" / run
    fit_row_groups = {"train": [10], "valid": [20]}
    provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": fit_row_groups,
        "args": {"init": None, "propensity": None},
    }
    provenance_path = run_root / "provenance.json"
    _write(provenance_path, json.dumps(provenance).encode())
    checkpoint_path = run_root / "model.pt"
    _write(checkpoint_path, b"synthetic checkpoint bytes")
    checkpoint = {
        "epoch": 2,
        "args": {
            "checkpoint_epoch": 2,
            "checkpoint_policy": "fixed",
            "no_eval_during_fit": True,
            "split_manifest_sha256": split.sha256,
        },
        "provenance": {
            "split_manifest_sha256": split.sha256,
            "fit_row_groups": fit_row_groups,
        },
    }
    monkeypatch.setattr(builder.torch, "load", lambda *_args, **_kwargs: checkpoint)

    closure = builder._audit_ancestor_closure(run, split, tmp_path / "runs")

    assert len(closure) == 1
    assert closure[0]["run"] == run
    assert closure[0]["checkpoint_sha256"] == builder.file_sha256(checkpoint_path)
    assert closure[0]["fit_group_counts"] == {"train": 1, "valid": 1}


def test_t3_builder_rejects_wrong_order_before_source_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    monkeypatch.setattr(builder, "load_campaign_split", lambda *_: split)
    args = Namespace(
        data_root=tmp_path,
        teacher_cache_root=tmp_path / "predictions",
        checkpoint_root=tmp_path / "runs",
        split_manifest=split.path,
        output=tmp_path / "cache",
        cache_profile=builder.CLEANFOLD_T3_PROFILE,
        teacher=[
            TeacherSource("strong", "fg19_t3_h384x2_e3_s4_v1", 0.1, 2),
            TeacherSource("strong", "fg19_t3_h384x2_e3_s3_v1", 0.1, 2),
            TeacherSource("strong", "fg19_t3_h384x2_e3_s5_v1", 0.1, 2),
        ],
        output_profile="homogeneous-only",
        homogeneous_family="strong",
        expected_coefficient_sum=0.8125,
    )

    with pytest.raises(ValueError, match="ordered strong S3/S4/S5"):
        builder.build_cache(args)


def test_t3_builder_rejects_dangling_output_symlink_before_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    output = tmp_path / "cache-output"
    output.symlink_to(tmp_path / "missing-cache-target")
    monkeypatch.setattr(builder, "load_campaign_split", lambda *_: split)
    monkeypatch.setattr(
        builder, "_target_order", lambda _root, name: [10] if name == "train" else [20]
    )
    monkeypatch.setattr(
        builder,
        "_cleanfold_t3_pins",
        lambda *_args: pytest.fail("output path must be checked before reading source pins"),
    )
    args = Namespace(
        data_root=tmp_path,
        teacher_cache_root=tmp_path / "predictions",
        checkpoint_root=tmp_path / "runs",
        split_manifest=split.path,
        output=output,
        cache_profile=builder.CLEANFOLD_T3_PROFILE,
        cleanfold_contract=tmp_path / "contract.json",
        cleanfold_contract_sha256="a" * 64,
        acceptance_root=tmp_path / "campaign",
        teacher=[TeacherSource("strong", run, 0.1, 2) for run in builder.T3_TEACHER_ORDER],
        expected_coefficient_sum=0.8125,
        homogeneous_family="strong",
        chunk_sequences=1,
        min_free_gb=0.0,
        output_budget_gb=0.1,
        output_profile="homogeneous-only",
    )

    with pytest.raises(FileExistsError, match="including symlinks"):
        builder.build_cache(args)

    assert output.is_symlink()
    assert not output.resolve().exists()


def test_t3_builder_writes_validator_compatible_fit_only_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    (data_root / "cache").mkdir(parents=True)
    np.savez(data_root / "cache/train_targets.npz", row_group=np.asarray([12, 10, 11]))
    np.savez(data_root / "cache/valid_targets.npz", row_group=np.asarray([21, 20]))
    split = CampaignSplit(
        tmp_path / "fold0.json",
        "split-sha",
        0,
        4,
        (10, 12),
        (20,),
        (21,),
        (("train", 10, 0), ("train", 12, 1), ("valid", 20, 2), ("valid", 21, 3)),
    )
    sources = [TeacherSource("strong", run, 0.1, 2) for run in builder.T3_TEACHER_ORDER]
    sidecar_groups = {"train": [12, 10], "valid": [20]}

    def inspect_teacher(
        run: str, *_args: object, fit_only: bool = False, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, dict[str, object]]]:
        assert fit_only
        run_offset = builder.T3_TEACHER_ORDER.index(run)
        arrays = {}
        records = {}
        for split_name, groups in sidecar_groups.items():
            values = np.empty((len(groups), SEQUENCE_ROWS, 2), dtype=np.float32)
            for index, group in enumerate(groups):
                values[index] = np.float32(group + run_offset)
            arrays[split_name] = values
            records[split_name] = {"_row_groups": groups.copy()}
        return {"run": run, "epoch": 2, "predictions": records}, arrays, records

    pins = [
        {
            "run": run,
            "acceptance_path": f"records/teacher_s{seed}_v1/TERMINAL_ACCEPTANCE.json",
            "acceptance_sha256": f"{seed:064x}",
            "model_sha256": f"{seed + 1:064x}",
            "checkpoint_sha256": f"{seed + 2:064x}",
            "provenance_sha256": f"{seed + 3:064x}",
            "tensor_fingerprint": f"{seed + 4:064x}",
        }
        for seed, run in enumerate(builder.T3_TEACHER_ORDER, start=3)
    ]
    monkeypatch.setattr(builder, "load_campaign_split", lambda *_: split)
    monkeypatch.setattr(builder, "_inspect_teacher", inspect_teacher)
    monkeypatch.setattr(builder, "_audit_ancestor_closure", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        builder,
        "_cleanfold_t3_pins",
        lambda *_args: (pins, {"schema": "fg19.teacher-cache-execution-contract.v1"}),
    )
    args = Namespace(
        data_root=data_root,
        teacher_cache_root=tmp_path / "predictions",
        checkpoint_root=tmp_path / "runs",
        split_manifest=split.path,
        output=tmp_path / "cache-output",
        cache_profile=builder.CLEANFOLD_T3_PROFILE,
        cleanfold_contract=tmp_path / "contract.json",
        cleanfold_contract_sha256="a" * 64,
        acceptance_root=tmp_path / "campaign",
        teacher=sources,
        expected_coefficient_sum=0.8125,
        homogeneous_family="strong",
        chunk_sequences=1,
        min_free_gb=0.0,
        output_budget_gb=0.1,
        output_profile="homogeneous-only",
    )

    cache_path = builder.build_cache(args)

    metadata = json.loads((cache_path / "metadata.json").read_text())
    assert metadata["cache_profile"] == builder.CLEANFOLD_T3_PROFILE
    assert metadata["coefficient_sum"] == 0.3
    assert metadata["source_coefficients"] == [
        {"family": "strong", "run": run, "epoch": 2, "coefficient": 0.1}
        for run in builder.T3_TEACHER_ORDER
    ]
    assert metadata["stored_targets"] == ["main_homogeneous"]
    assert metadata["root_acceptance_pins"] == pins
    train = np.load(cache_path / "main_homogeneous_train.npy", allow_pickle=False)
    valid = np.load(cache_path / "main_homogeneous_valid.npy", allow_pickle=False)
    np.testing.assert_array_equal(train[:, 0, 0], np.asarray([10.5625, 8.9375], dtype=np.float32))
    np.testing.assert_array_equal(valid[:, 0, 0], np.asarray([17.0625], dtype=np.float32))
    result = validate_distillation_cache(
        cache_path, data_root, split, "homogeneous", require_families=False
    )
    assert set(result["verified_files"]) == {"main_homogeneous_train", "main_homogeneous_valid"}
