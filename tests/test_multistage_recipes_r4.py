from __future__ import annotations

import json
from pathlib import Path

from scripts.multistage_recipes_r4 import (
    AUXILIARY_CONTROL_SEQUENCES,
    AUXILIARY_FIT_SEQUENCES,
    AUXILIARY_MANIFEST_SHA256,
    L4_GALAM_PRESET,
    _search_spaces,
    write_artifacts,
)
from wnn.multistage_search_space import apply_stage1_config
from wnn.training.sequence_config import load_config


def test_r4_family_docs_load_and_all_anchor_configs_match_their_search_params(
    tmp_path: Path,
) -> None:
    receipt = write_artifacts(tmp_path, L4_GALAM_PRESET)
    recipe = json.loads((tmp_path / "recipes" / "stage1-family-extension-r4.json").read_text())
    family_specs = recipe["stage1"]["families"]
    configs = recipe["nodes"]

    assert recipe["data_binding"]["auxiliary_manifest_sha256"] == AUXILIARY_MANIFEST_SHA256
    assert recipe["data_binding"]["auxiliary_fit_sequences"] == AUXILIARY_FIT_SEQUENCES == 8260
    assert (
        recipe["data_binding"]["auxiliary_control_sequences"] == AUXILIARY_CONTROL_SEQUENCES == 2065
    )
    assert len(configs) == 2
    assert {node["family"] for node in configs} == {"galam", "causal_transformer"}

    for family, spec in family_specs.items():
        config_path = Path(tmp_path) / spec["base_config"]["path"]
        config = load_config(config_path)
        assert config["precision_mode"] == "fp32"
        assert config["training_mask"] == "all_after_warmup"
        assert config["normalization_scope"] == "prefix"
        assert config["required_world_size"] == 1
        assert family in _search_spaces()

    for anchor in recipe["stage1"]["anchors"]:
        family = anchor["family"]
        config = json.loads((tmp_path / anchor["config"]["path"]).read_text())
        realized = apply_stage1_config(config, family, anchor["params"], family_specs[family])
        assert realized == config

    assert set(receipt["config_sha256"]) == {
        "recipes/configs/r4/gru-base-r4.json",
        "recipes/configs/r4/lstm-base-r4.json",
        "recipes/configs/r4/galam-g1-adapt-r4.json",
        "recipes/configs/r4/causal-transformer-base-r4.json",
    }


def test_r4_search_spaces_vary_model_capacity_regularization_and_horizon() -> None:
    spaces = _search_spaces()
    for family in ("gru", "lstm"):
        assert spaces[family]["hidden"]["choices"] == [128, 256, 384]
        assert spaces[family]["layers"]["choices"] == [2, 3]
        assert spaces[family]["dropout"]["choices"] == [0.0, 0.1]
        assert spaces[family]["epochs"]["choices"] == [2, 4, 6]
    assert "width" not in spaces["galam"]
    assert "feed_forward_width" not in spaces["galam"]
    assert spaces["galam"]["global_blocks"]["config_paths"] == [
        "model.galam.global_blocks",
        "model.layers",
    ]
    assert spaces["causal_transformer"]["context_rows"]["choices"] == [128, 256, 512]
    assert spaces["causal_transformer"]["chunk_rows"]["choices"] == [128, 256]
