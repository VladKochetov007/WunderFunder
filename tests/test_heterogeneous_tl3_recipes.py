from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.heterogeneous_tl3_recipes import (
    CONTROL_TEACHERS,
    HETEROGENEOUS_TEACHERS,
    SPLIT_SHA256,
    build_nodes,
    materialize_plan,
)
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.checkpoints import configure_checkpoint_policy
from wnn.training.cli import build_parser
from wnn.training.setup_lineage import require_chronological_primary_training


def _parsed_nodes() -> list[tuple[dict[str, object], dict[str, object]]]:
    result = []
    split = SimpleNamespace(schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, valid_groups=())
    for node in build_nodes(Path("split.json"), Path("runtime")):
        parser = build_parser()
        args = parser.parse_args(node["args"])
        configure_checkpoint_policy(args, parser)
        require_chronological_primary_training(parser, args, split)
        result.append((node, vars(args)))
    return result


def test_six_fits_have_fixed_chronological_no_valid_ancestry_contract() -> None:
    nodes = _parsed_nodes()
    assert len(nodes) == 6
    assert len({node["run"] for node, _ in nodes}) == 6
    assert sum(node["expected_updates"] for node, _ in nodes) == 154560
    assert sum(node["role"] == "teacher" for node, _ in nodes) == 2
    for node, args in nodes:
        assert args["precision"] == "fp32"
        assert args["checkpoint_policy"] == "fixed" and args["checkpoint_epoch"] == 3
        assert node["endpoint"] == 3 and args["epochs"] == 4
        assert args["train_source"] == "train" and args["train_seqs"] == 10325
        assert args["full_valid"]
        assert not any(
            args[key] for key in ("init", "propensity", "prop_input", "mask_weight", "prop_cache")
        )
        assert args["distill_boot_h"] == 0 and args["distill_family_weight"] == 0
        assert args["optim"] == "adamw" and args["lr"] == 0.001
        assert args["eval_ema"] and args["ema_teacher"] == 1 and args["ema_decay"] == 0.999
        assert args["aux_next"] == args["aux_struct"] == 0.3
        assert args["since_weight"] == 32 and args["since_floor"] == 0.25


def test_paired_students_differ_only_by_run_and_target_cache() -> None:
    nodes = _parsed_nodes()
    for seed in (1301, 1302):
        arms = [args for node, args in nodes if node["role"] == "student" and args["seed"] == seed]
        assert len(arms) == 2
        different = {key for key in arms[0] if arms[0][key] != arms[1][key]}
        assert different == {"name", "distill_cache_root"}
        for args in arms:
            assert args["hidden"] == 256 and args["layers"] == 2
            assert args["arch"] == args["cell"] == "gru"
            assert args["distill_alpha"] == 0.5 and args["distill_cache_variant"] == "source"
            assert args["distill_cache_scale"] == 1 and not args["distill"]


def test_family_weights_and_teacher_endpoints_are_predeclared() -> None:
    assert sum(record[2] for record in CONTROL_TEACHERS) == 1
    assert sum(record[2] for record in HETEROGENEOUS_TEACHERS) == pytest.approx(1)
    assert HETEROGENEOUS_TEACHERS[:2] == (("tl3c_fin256_s3", 3, 0.35), ("tl3c_fin256_s4", 3, 0.35))
    assert HETEROGENEOUS_TEACHERS[2] == ("g2-galam-lstm128", 2, 0.1)
    teachers = {node["id"]: args for node, args in _parsed_nodes() if node["role"] == "teacher"}
    assert teachers["lstm_s1201"]["cell"] == "lstm"
    assert teachers["tcn_s1202"]["arch"] == "tcn"
    assert teachers["tcn_s1202"]["layers"] == 12 and teachers["tcn_s1202"]["hidden"] == 128
    assert teachers["tcn_s1202"]["tcn_mix"] == 0
    plan = materialize_plan(Path("split.json"), Path("runtime"))
    assert plan["expected_split_sha256"] == SPLIT_SHA256
    assert plan["configured_epochs"] == 24 and plan["new_fits"] == 6


def test_shared_target_cache_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        build_nodes(Path("split.json"), Path("runtime"), Path("same"), Path("same"))
