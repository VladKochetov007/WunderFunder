"""Focused source-identity and checkpoint producer tests for GRU capacity V3."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKET = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    / "gru512_768_muon_capacity_v3_profile_20261008/packet"
)
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
sys.path.insert(0, str(PACKET))
_WORKER_SPEC = importlib.util.spec_from_file_location(
    "capacity_worker_v3_test", PACKET / "capacity_worker.py"
)
assert _WORKER_SPEC is not None and _WORKER_SPEC.loader is not None
worker = importlib.util.module_from_spec(_WORKER_SPEC)
_WORKER_SPEC.loader.exec_module(worker)


def _actual_split() -> dict[str, object]:
    path = PACKET / "fold0.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SPLIT_SHA256
    return json.loads(path.read_text(encoding="utf-8"))


def _nvidia_row(index: int, uuid: str, free: int) -> str:
    return f"{index}, NVIDIA T4, {uuid}, 7.5, 15360, {free}\n"


def test_gpu_hardware_pins_physical_zero_and_records_two_t4_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _nvidia_row(0, "GPU-physical-zero", 12000) + _nvidia_row(
        1, "GPU-physical-one", 11800
    )
    selected = _nvidia_row(0, "GPU-physical-zero", 12000)
    replies = iter((inventory, selected))
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(stdout=next(replies), stderr="", returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    hardware = worker._gpu_hardware()

    assert len(hardware["gpu_inventory"]) == 2
    assert hardware["selected_physical_gpu_index"] == 0
    assert hardware["selected_physical_gpu"]["uuid"] == "GPU-physical-zero"
    assert hardware["uuid"] == "GPU-physical-zero"
    assert calls[1][1] == "--id=0"
    assert any(
        "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.free" in arg
        for arg in calls[1]
    )


def test_gpu_row_parser_fails_closed_without_device_or_on_wrong_selected_id() -> None:
    with pytest.raises(ValueError, match="no accelerator rows"):
        worker._parse_nvidia_smi_rows("\n")
    with pytest.raises(ValueError, match="different physical accelerator"):
        worker._parse_selected_nvidia_smi_row(
            _nvidia_row(1, "GPU-physical-one", 11800), expected_index=0
        )
    with pytest.raises(ValueError, match="field count"):
        worker._parse_nvidia_smi_rows("0, NVIDIA T4, only-three-fields\n")


def test_gpu_sampler_uses_the_selected_physical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(stdout="0, 321, 15000, 15360\n", stderr="", returncode=0)

    class StopAfterOneSample:
        checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks > 1

        def wait(self, interval: float) -> bool:
            return True

    sampler = worker.GpuMemorySampler()
    sampler._stop = StopAfterOneSample()
    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    sampler._sample()

    assert calls[0][1] == "--id=0"
    assert sampler.samples == [(321, 15000, 15360)]


def test_kaggle_driver_selects_named_source_dataset_and_materializes_exact_packet(
    tmp_path: Path,
) -> None:
    driver_spec = importlib.util.spec_from_file_location(
        "gru_capacity_v3_kaggle_driver_test", PACKET / "kernel_upload/kernel.py"
    )
    assert driver_spec is not None and driver_spec.loader is not None
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)
    input_root = tmp_path / "input"
    source_root = (
        input_root / "datasets/vladkochetov07/e529-gru512-768-muon-capacity-v3-source-20261008"
    )
    source_root.parent.mkdir(parents=True)
    shutil.copytree(PACKET / "source_dataset_upload", source_root)
    unrelated = input_root / "datasets/another-owner/unrelated-dataset"
    unrelated.mkdir(parents=True)
    (unrelated / "assets-manifest.json").write_text(
        json.dumps({"dataset": "another-owner/unrelated-dataset", "files": []}),
        encoding="utf-8",
    )

    assert driver.source_dataset_root(input_root) == source_root
    materialized = tmp_path / "working/packet"
    driver.materialize_packet(input_root, materialized)
    recipe, authorization = worker._verify_packet(materialized)
    assert recipe["packet_id"] == worker.PACKET_ID
    assert authorization["profile_authorized"] is False
    assert authorization["conditional_fit_authorized"] is False


def test_actual_fold_uses_source_qualified_groups_and_components() -> None:
    split = _actual_split()
    receipt = worker._validate_source_qualified_fit_membership(
        split,
        list(map(int, split["fit"]["train"])),
        list(map(int, split["fit"]["valid"])),
    )
    assert receipt["status"] == "pass"
    assert receipt["cross_source_alias_overlap_counts"] == {
        "train_fit_vs_valid_evaluation": 463,
        "valid_fit_vs_train_excluded": 9,
    }
    assert receipt["fit_component_count"] == 3148
    assert receipt["evaluation_component_count"] == 141


@pytest.mark.parametrize("source", ["train", "valid"])
def test_same_source_excluded_membership_overlap_is_rejected(source: str) -> None:
    split = _actual_split()
    bad_split = copy.deepcopy(split)
    if source == "train":
        bad_split["fit"]["train"].append(bad_split["excluded"]["train"][0])
        observed_train = list(map(int, bad_split["fit"]["train"]))
        observed_valid = list(map(int, bad_split["fit"]["valid"]))
    else:
        bad_split["fit"]["valid"].append(bad_split["excluded"]["valid"][0])
        observed_train = list(map(int, bad_split["fit"]["train"]))
        observed_valid = list(map(int, bad_split["fit"]["valid"]))
    with pytest.raises(ValueError, match="own source namespace"):
        worker._validate_source_qualified_fit_membership(bad_split, observed_train, observed_valid)


def test_shared_component_between_fit_and_evaluation_is_rejected() -> None:
    split = _actual_split()
    bad_split = copy.deepcopy(split)
    fit_train_group = int(bad_split["fit"]["train"][0])
    evaluation_group = int(bad_split["evaluation"]["valid"][0])
    evaluation_component = next(
        int(row["component"])
        for row in bad_split["sequences"]["valid"]
        if int(row["row_group"]) == evaluation_group
    )
    for row in bad_split["sequences"]["train"]:
        if int(row["row_group"]) == fit_train_group:
            row["component"] = evaluation_component
            break
    with pytest.raises(ValueError, match="connected component"):
        worker._validate_source_qualified_fit_membership(
            bad_split,
            list(map(int, bad_split["fit"]["train"])),
            list(map(int, bad_split["fit"]["valid"])),
        )


def _producer_saved_fit_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, str]]:
    import torch

    split = _actual_split()
    manifest = json.loads(
        (PACKET / "source/source_archive_manifest.json").read_text(encoding="utf-8")
    )
    source_hashes = {item["path"]: item["sha256"] for item in manifest["members"]}
    run_name = "gru_v3_fixture_muon512_b8"
    run_root = tmp_path / "artifacts/runs" / run_name
    run_root.mkdir(parents=True)
    args = {
        "name": run_name,
        "seed": 9,
        "hidden": 512,
        "layers": 2,
        "batch": 8,
        "batch_order": "seeded",
        "optim": "muon",
        "precision": "fp16",
        "amp_init_scale": 4096.0,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "valid_fold": 0,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "split_manifest_sha256": SPLIT_SHA256,
    }
    checkpoint = {
        "model": {
            "weight": torch.tensor([[1.0, -0.0], [2.5, 3.0]], dtype=torch.float32),
            "integer_buffer": torch.tensor([3], dtype=torch.int64),
        },
        "args": args,
        "n_in": 211,
        "epoch": 3,
        "readout": None,
        "readout_tilted": False,
        "provenance": {"split_manifest_sha256": SPLIT_SHA256},
    }
    epoch_path = run_root / "model_ep3.pt"
    alias_path = run_root / "model.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, alias_path)
    epoch_sha = worker.sha256_file(epoch_path)
    alias_sha = worker.sha256_file(alias_path)
    assert epoch_sha != alias_sha, "torch.save should produce path-distinct container bytes"

    expected_batches = 1492
    update = {
        "optimizer_attempts": expected_batches * worker.CHUNK_UPDATES_PER_BATCH,
        "optimizer_completed_updates": expected_batches * worker.CHUNK_UPDATES_PER_BATCH,
        "amp_skipped_updates": 0,
        "completed_batches": expected_batches,
    }
    normalization = {"fixture": "producer-style"}
    provenance_epochs = []
    history = []
    for epoch in range(4):
        order = {
            "sampler_class": "EpochOrderSampler",
            "batch_size": 8,
            "drop_last": True,
            "seeded_order": {
                "seed": 9,
                "epoch": epoch,
                "antithetic": False,
                "shared_dropped_tail": True,
            },
        }
        provenance_epochs.append(
            {
                "epoch_index": epoch,
                "epoch": update,
                "sample_order_identity": order,
                "normalization_identity": normalization,
            }
        )
        history.append({"epoch": epoch, "update_diagnostics": {"epoch": update}})
    provenance = {
        "fit_row_groups": split["fit"],
        "split_manifest_sha256": SPLIT_SHA256,
        "trainer_source_sha256": source_hashes["scripts/train_rnn.py"],
        "model_source_sha256": source_hashes["src/wnn/nn.py"],
        "update_diagnostics": provenance_epochs,
        "selected_epoch": 3,
        "checkpoint_sha256": {"model_ep3.pt": epoch_sha, "model.pt": alias_sha},
        "normalization_identity": normalization,
        "orthogonalization_precision": {
            "policy": "native_bf16_if_supported_else_fp32_v1",
            "selected_dtype": "float32",
            "torch_cuda_bf16_supported": False,
            "groups": ["weight_ih_l0"],
        },
    }
    (run_root / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (run_root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return tmp_path, split, source_hashes


def test_real_torch_save_alias_passes_with_separate_producer_hashes(tmp_path: Path) -> None:
    project_root, split, source_hashes = _producer_saved_fit_fixture(tmp_path)
    result = worker._verify_fit_artifacts(
        project_root,
        {"optimizer": "muon", "hidden": 512},
        "gru_v3_fixture_muon512_b8",
        8,
        split,
        source_hashes,
    )
    assert result["status"] == "pass"
    assert result["checkpoint_sha256"] != result["model_alias_sha256"]
    assert result["checkpoint_pair_state_identity"] == "bitwise_tensor_state_and_config_match"


@pytest.mark.parametrize("mutation", ["tensor", "args"])
def test_real_torch_save_alias_rejects_state_or_config_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    import torch

    project_root, split, source_hashes = _producer_saved_fit_fixture(tmp_path)
    run_root = project_root / "artifacts/runs/gru_v3_fixture_muon512_b8"
    alias_path = run_root / "model.pt"
    alias = torch.load(alias_path, map_location="cpu", weights_only=False)
    if mutation == "tensor":
        alias["model"]["weight"][0, 0] += 1.0
        message = "tensor state differs"
    else:
        alias["args"]["batch"] = 4
        message = "differ in args"
    torch.save(alias, alias_path)
    provenance_path = run_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["checkpoint_sha256"]["model.pt"] = worker.sha256_file(alias_path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        worker._verify_fit_artifacts(
            project_root,
            {"optimizer": "muon", "hidden": 512},
            "gru_v3_fixture_muon512_b8",
            8,
            split,
            source_hashes,
        )
