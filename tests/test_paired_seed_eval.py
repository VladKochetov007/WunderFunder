from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_seed_eval as evaluator  # noqa: E402
from fold_eval import pooled, sums  # noqa: E402


def manifest() -> dict:
    return {
        "status": "preregistered",
        "evidence_role": "discovery",
        "folds": [0, 1],
        "pairs": [
            {"seed": 3, "candidate": "wide3", "reference": "small3"},
            {"seed": 4, "candidate": "wide4", "reference": "small4"},
        ],
        "bootstrap_repeats": 100,
        "bootstrap_seed": 27,
    }


def direct_score(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Centered weighted Pearson on pooled rows, independent of moment aggregation."""
    y = np.clip(y[mask].astype(np.float64), -2, 2)
    pred = np.clip(pred[mask].astype(np.float64), -2, 2)
    scores = []
    for target, prediction in zip(y.T, pred.T, strict=True):
        weight = np.abs(target)
        total = weight.sum()
        if total < 1e-8:
            scores.append(0.0)
            continue
        yc = target - np.dot(weight, target) / total
        pc = prediction - np.dot(weight, prediction) / total
        sy = np.sqrt(np.dot(weight, yc * yc) / total)
        sp = np.sqrt(np.dot(weight, pc * pc) / total)
        scores.append(0.0 if min(sy, sp) <= 1e-8
                      else np.clip(np.dot(weight, yc * pc) / total / sy / sp, -1, 1))
    return float(np.mean(scores))


def sample_stats() -> tuple:
    rng = np.random.default_rng(9)
    y = rng.uniform(-3, 3, (4, 9, 2))
    mask = rng.random((4, 9)) > 0.15
    candidates = [y * 0.8 + rng.normal(size=y.shape), y * 0.3 + rng.normal(size=y.shape)]
    references = [y * 0.4 + rng.normal(size=y.shape), y * 0.7 + rng.normal(size=y.shape)]
    sc = np.stack([sums(y, p, mask) for p in candidates + [np.mean(candidates, axis=0)]])
    sr = np.stack([sums(y, p, mask) for p in references + [np.mean(references, axis=0)]])
    return y, mask, candidates, references, sc, sr


def test_seed_mean_is_mean_of_paired_effects_and_not_ensemble_score():
    y, mask, candidates, references, sc, sr = sample_stats()
    report = evaluator.summarize(sc, sr, np.array([0, 0, 1, 1]), np.arange(4), 150, 8)
    effects = np.array([direct_score(y, c, mask) - direct_score(y, r, mask)
                        for c, r in zip(candidates, references, strict=True)])
    ensemble_effect = (direct_score(y, np.mean(candidates, axis=0), mask)
                       - direct_score(y, np.mean(references, axis=0), mask))
    summary = report["training_seed_summary"]
    assert summary["matched_delta_mean"] == pytest.approx(effects.mean())
    assert summary["matched_delta_sample_sd"] == pytest.approx(effects.std(ddof=1))
    assert [r["delta"] for r in report["matched_pairs"]] == pytest.approx(effects)
    assert report["mean_prediction_ensemble"]["delta"] == pytest.approx(ensemble_effect)
    assert abs(ensemble_effect - effects.mean()) > 0.001
    assert report["confirmation_established"] is False


def test_bootstrap_moves_whole_chains_and_preserves_fold_strata():
    *_, sc, sr = sample_stats()
    # One complete chain per fold: every stratified chain resample is identical.
    report = evaluator.summarize(sc, sr, np.array([0, 0, 1, 1]),
                                 np.array([10, 10, 20, 20]), 120, 3)
    assert report["chain_count"] == 2
    for item in report["matched_pairs"] + [report["mean_prediction_ensemble"]]:
        assert item["conditional_chain_sd"] < 1e-14
        assert item["conditional_chain_ci95"] == pytest.approx([item["delta"]] * 2)


def test_rejects_chain_split_across_folds():
    *_, sc, sr = sample_stats()
    with pytest.raises(ValueError, match="chain cannot be split"):
        evaluator.summarize(sc, sr, np.array([0, 0, 1, 1]),
                            np.array([10, 10, 10, 20]), 100, 3)


@pytest.mark.parametrize("case", ["zero_weight", "constant_prediction", "constant_target"])
def test_degenerate_target_contributes_zero(case):
    y = np.array([[[-1.0, -1.0], [0.5, 0.5], [1.5, 1.5]]])
    pred = y.copy()
    if case == "zero_weight":
        y[..., 0] = 0
    elif case == "constant_target":
        y[..., 0] = 1
    else:
        pred[..., 0] = 0.25
    mask = np.ones(y.shape[:2], dtype=bool)
    assert pooled(sums(y, pred, mask).sum(0)) == pytest.approx(0.5)
    assert pooled(sums(y, pred, mask).sum(0)) == pytest.approx(direct_score(y, pred, mask))


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(status="completed"),
    lambda m: m.update(evidence_role="confirmation"),
    lambda m: m.update(pairs=m["pairs"][:1]),
    lambda m: m["pairs"][1].update(seed=3),
    lambda m: m["pairs"][1].update(seed=True),
    lambda m: m["pairs"][1].update(candidate="wide3"),
    lambda m: m["pairs"][1].update(reference="small3"),
    lambda m: m["pairs"][0].update(candidate="wide3+wide4"),
    lambda m: m["pairs"][0].update(reference="../outside"),
    lambda m: m.update(folds=[]),
    lambda m: m.update(folds=[0, 0]),
    lambda m: m.update(folds=["0", 1]),
    lambda m: m.update(bootstrap_repeats=99),
    lambda m: m.update(bootstrap_repeats=100.5),
    lambda m: m.update(bootstrap_seed=-1),
])
def test_invalid_manifest_is_rejected(mutation):
    spec = manifest()
    mutation(spec)
    with pytest.raises(ValueError):
        evaluator.validate_manifest(spec)


def test_main_uses_pure_oof_rows_warmup_mask_and_average_before_clip(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/fold_eval.py").write_text("synthetic scorer provenance\n")
    rng = np.random.default_rng(17)
    shape = (5, evaluator.WARMUP + 8, 2)
    y = rng.uniform(-1.9, 1.9, shape).astype(np.float32)
    mask = np.ones(shape[:2], dtype=bool)
    mask[:, -2] = False
    np.savez(cache / "valid_targets.npz", row_group=np.arange(5), y=y, mask=mask)
    folds = np.array([0, 0, 1, 1, 0])
    np.savez(cache / "chain_folds.npz", split=np.array(["valid"] * 5),
             row_group=np.arange(5), fold=folds, chain=np.array([10, 10, 20, 21, 99]),
             pure=np.array([True, True, True, True, False]))
    spec = manifest()
    candidates = [y * 4 + rng.normal(0, 0.4, shape), -y + rng.normal(0, 0.3, shape)]
    references = [y + rng.normal(0, 1, shape), y + rng.normal(0, 1.5, shape)]
    expected_c, expected_r = [], []
    for pair, c, r in zip(spec["pairs"], candidates, references, strict=True):
        c, r = c.astype(np.float32), r.astype(np.float32)
        expected_c.append(c.copy())
        expected_r.append(r.copy())
        for fold in spec["folds"]:
            # Non-OOF rows contain deliberately bad predictions and must be ignored.
            cp, rp = c.copy(), r.copy()
            cp[folds != fold] = -100
            rp[folds != fold] = 100
            np.save(cache / f"pred_{pair['candidate']}_f{fold}_valid.npy", cp)
            np.save(cache / f"pred_{pair['reference']}_f{fold}_valid.npy", rp)
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(spec))
    out = tmp_path / "result.json"
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator.subprocess, "check_output", lambda *a, **kw: "fixture-commit\n")
    monkeypatch.setattr(sys, "argv", ["paired_seed_eval.py", str(source), "--out", str(out)])
    evaluator.main()
    report = json.loads(out.read_text())
    use_mask = mask[:4].copy()
    use_mask[:, :evaluator.WARMUP] = False
    avg_c = np.mean(expected_c, axis=0, dtype=np.float32)[:4]
    avg_r = np.mean(expected_r, axis=0, dtype=np.float32)[:4]
    expected = direct_score(y[:4], avg_c, use_mask) - direct_score(y[:4], avg_r, use_mask)
    wrong_avg = np.mean(np.clip(expected_c, -2, 2), axis=0, dtype=np.float32)[:4]
    assert abs(direct_score(y[:4], wrong_avg, use_mask) - direct_score(y[:4], avg_c, use_mask)) > 0.01
    assert report["mean_prediction_ensemble"]["delta"] == pytest.approx(expected, abs=1e-12)
    assert report["sequence_count"] == 4
    assert report["chain_count"] == 3
    assert len(report["provenance"]["inputs"]) == 10
    original_output = out.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator.main()
    assert out.read_bytes() == original_output
    # Manifest remains the immutable pre-result specification.
    assert json.loads(source.read_text()) == copy.deepcopy(spec)
