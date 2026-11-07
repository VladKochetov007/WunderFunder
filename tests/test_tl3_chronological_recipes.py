from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.score_tl3_chronological import (
    fixed_blend,
    grade,
    load_prediction,
    validate_final_recipe,
)
from scripts.tl3_chronological_recipes import W18_COEFFICIENTS, build_nodes
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, file_sha256
from wnn.training.checkpoints import configure_checkpoint_policy
from wnn.training.cli import build_parser
from wnn.training.setup_lineage import require_chronological_primary_training


def test_all_recovered_recipes_pass_the_chronological_gate_with_fixed_endpoints() -> None:
    nodes = build_nodes(Path("manifest.json"), Path("runtime"))
    assert len(nodes) == len({node["id"] for node in nodes}) == 17
    assert sum(node["expected_updates"] for node in nodes) == 257740
    assert sum(node["epochs"] for node in nodes) == 54
    split = SimpleNamespace(schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, valid_groups=())
    for node in nodes:
        parser = build_parser()
        args = parser.parse_args(node["args"])
        configure_checkpoint_policy(args, parser)
        require_chronological_primary_training(parser, args, split)
        assert args.checkpoint_epoch == node["endpoint"]
        assert args.precision == "fp32"
        assert not (args.propensity or args.prop_input or args.mask_weight or args.prop_cache)
        assert args.since_weight == (0 if node["endpoint"] == 1 else 32)
        assert args.since_floor == 0.25
        if node["id"].startswith("fin256_s"):
            assert args.optim == "adamw"  # Historical optimizer absent; explicit fresh choice.
        if "tlead" in node["id"] or node["id"] == "fbc1k_s5":
            assert args.distill_alpha == 0.5 and args.distill_boot_h == 8
            assert args.eval_ema and args.ema_teacher == 1
    assert sum(value[1] for value in W18_COEFFICIENTS.values()) == 0.8125


def test_fixed_blend_happens_before_official_prediction_clipping() -> None:
    values = [
        np.array([[10, -10]], dtype=np.float32),
        np.array([[-1, 1]], dtype=np.float32),
        np.array([[-1, 1]], dtype=np.float32),
    ]
    result = fixed_blend(values)
    expected = np.array([[2.74, -2.74]], dtype=np.float32)
    # Separate FP32 multiply/add operations can differ from rounded decimal by one ULP.
    np.testing.assert_allclose(result, expected, atol=5e-7, rtol=0)
    assert result.dtype == np.float32
    assert not np.allclose(
        result, sum(w * np.clip(p, -2, 2) for w, p in zip((0.34, 0.33, 0.33), values, strict=True))
    )


def test_fixed_blend_rejects_missing_or_misaligned_members() -> None:
    with pytest.raises(ValueError, match="three aligned"):
        fixed_blend([np.ones((2, 2), dtype=np.float32)])
    with pytest.raises(ValueError, match="three aligned"):
        fixed_blend([np.ones((n, 2), dtype=np.float32) for n in (2, 3, 2)])


def test_final_recipe_gate_catches_changed_since_weight() -> None:
    node = next(
        node
        for node in build_nodes(Path("manifest.json"), Path("runtime"))
        if node["id"] == "fttall_tlead_s5"
    )
    parser = build_parser()
    args = parser.parse_args(node["args"])
    configure_checkpoint_policy(args, parser)
    actual = vars(args)
    actual.update(eval_source="vfold", valid_fold=0, n_valid_folds=1)
    validate_final_recipe(actual, "fttall_tlead_s5", Path("manifest.json"), Path("runtime"))
    actual["since_weight"] = 0
    with pytest.raises(ValueError, match="since_weight"):
        validate_final_recipe(actual, "fttall_tlead_s5", Path("manifest.json"), Path("runtime"))


def test_grade_rejects_alternative_same_role_manifest_before_data_access(tmp_path: Path) -> None:
    manifest = tmp_path / "another.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="frozen TL3 manifest"):
        grade(tmp_path, manifest, tmp_path / "score.json")


def test_prediction_order_cannot_be_rewritten_by_editing_the_sidecar(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "valid_index.npz", row_group=np.array([0, 1]))
    path = cache / "pred_probe_valid.npy"
    np.save(path, np.zeros((2, 20000, 2), dtype=np.float32))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "run": "probe",
                "split": "valid",
                "split_manifest_sha256": "split",
                "inference_precision": "fp32",
                "row_groups": [1, 0],
                "prediction_sha256": file_sha256(path),
            }
        )
    )
    with pytest.raises(ValueError, match="full-valid coverage"):
        load_prediction(tmp_path, "probe", [0, 1], "split")
