from __future__ import annotations

import json
from pathlib import Path

import torch

from wnn.training.sequence_initialization import initialize_from_parent_receipt
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_pretraining_contract import FUTURE_INNOVATION_KIND
from wnn.training.sequence_pretraining_receipt import (
    BACKBONE_SOURCE_FILES,
    write_pretraining_parent_receipt,
)
from wnn.training.sequence_provenance import file_sha256


def test_parent_receipt_copies_backbone_and_keeps_child_heads_fresh(tmp_path: Path) -> None:
    pretraining_config = {
        "seed": 203201,
        "epochs": 2,
        "pretraining": {"kind": FUTURE_INNOVATION_KIND, "horizons": [16, 64, 256]},
        "model": {
            "architecture": "gru",
            "hidden": 256,
            "layers": 2,
            "heads": 4,
            "context_rows": 512,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 336,
            "hint_projection": "learned",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }
    child_config = {
        **pretraining_config,
        "epochs": 4,
        "pretraining": None,
        "parent_initializer": FUTURE_INNOVATION_KIND,
        "objective_mode": "standalone_labels_wmse_v2",
        "supervised_target": None,
        "hint_weight": 0.0,
        "model": pretraining_config["model"] | {"hint_dim": 0},
    }
    normalization = {"mean": [0.25] * 112, "std": [2.0] * 112}
    source_files = {path: "a" * 64 for path in BACKBONE_SOURCE_FILES}
    identity = {"source_files": source_files, "seed": 203201}
    data = {
        "split_sha256": "b" * 64,
        "fit_members": [["train", 0]],
        "sources": {"datasets/train.parquet": {"sha256": "c" * 64}},
        "future_input_innovations": {
            "kind": FUTURE_INNOVATION_KIND,
            "normalization_identity_sha256": "d" * 64,
            "moments_sha256": "e" * 64,
        },
    }
    parent = build_sequence_model(pretraining_config["model"] | {"precision_mode": "fp32"})
    with torch.no_grad():
        parent.mean.fill_(0.25)
        parent.std.fill_(2.0)
        parent.head[0].weight.fill_(7.0)
        parent.hint_projector.linear.weight.fill_(9.0)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "checkpoint_schema": "wnn.sequence-fit.v1",
            "completed_epochs": 2,
            "model": parent.state_dict(),
            "config": pretraining_config,
            "data": data,
            "provenance": {"identity": identity, "normalization": normalization},
        },
        checkpoint_path,
    )
    status = {
        "status": "completed",
        "completed_epochs": 2,
        "model_sha256": file_sha256(checkpoint_path),
        "identity": identity,
    }
    (tmp_path / "terminal_status.json").write_text(json.dumps(status), encoding="utf-8")
    assert write_pretraining_parent_receipt(
        tmp_path, identity, pretraining_config, data, normalization
    )

    child = build_sequence_model(child_config["model"] | {"precision_mode": "fp32"})
    fresh_head = child.head[0].weight.detach().clone()
    child_data = {key: value for key, value in data.items() if key != "future_input_innovations"}
    result = initialize_from_parent_receipt(
        child,
        tmp_path / "future_innovation_parent_receipt.json",
        child_config,
        child_data,
        normalization,
        source_files,
    )
    assert result["optimizer_state"] == result["scheduler_state"] == "reset"
    assert result["rng_state"] == "fresh child seed and counters"
    torch.testing.assert_close(child.mean, parent.mean)
    torch.testing.assert_close(child.std, parent.std)
    torch.testing.assert_close(child.inp.weight, parent.inp.weight)
    torch.testing.assert_close(child.gru.weight_ih_l0, parent.gru.weight_ih_l0)
    torch.testing.assert_close(child.head[0].weight, fresh_head)
