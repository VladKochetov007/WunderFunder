from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import conditional_teacher_probe as probe  # noqa: E402
from fold_eval import sums  # noqa: E402


def manifest() -> dict:
    return {"status": "preregistered", "evidence_role": "discovery", "screen_id": "I1",
            "components": probe.COMPONENTS.copy(), "folds": [0, 1], "block_sequences": 2,
            "bootstrap_repeats": 100, "bootstrap_seed": 13,
            "gate": {"pooled_delta_min": 0.0004, "fold_delta_positive": True,
                     "fold_delta_originalC_min": 0.0, "weight_coverage_min": 0.05}}


def direct_wp(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """Centered reference calculation independent of moment aggregation."""
    yy = np.clip(y[mask].astype(np.float64), -2, 2)
    pp = np.clip(p[mask].astype(np.float64), -2, 2)
    rhos = []
    for target, pred in zip(yy.T, pp.T, strict=True):
        w = np.abs(target)
        if w.sum() < 1e-8:
            rhos.append(0.0)
            continue
        yc, pc = target - np.average(target, weights=w), pred - np.average(pred, weights=w)
        sy, sp = np.sqrt(np.average(yc ** 2, weights=w)), np.sqrt(np.average(pc ** 2, weights=w))
        rhos.append(0.0 if min(sy, sp) <= 1e-8 else np.average(yc * pc, weights=w) / sy / sp)
    return float(np.mean(rhos))


def test_gate_is_prediction_only_and_ensemble_is_formed_before_clipping():
    plain = np.array([[[6, -6], [6, 2]]], dtype=np.float32)
    distilled = np.array([[[0, 0], [5, 1]]], dtype=np.float32)
    parts = {name: np.zeros_like(plain) for name in probe.COMPONENTS}
    parts.update(plain3=plain, plain4=plain, distilled3=distilled, distilled4=distilled)
    before = {name: a.copy() for name, a in parts.items()}
    pred, gate = probe.mixtures(parts)
    np.testing.assert_array_equal(gate, [[True, False]])
    np.testing.assert_allclose(pred[3, 0], [[0.75, -0.75], [1.25, 0.25]])
    np.testing.assert_allclose(pred[2, 0, 1], [1.375, 0.375])
    assert pred[3, 0, 0, 0] != pytest.approx(0.25)  # Wrong answer if teachers were clipped first.
    for name in parts:
        np.testing.assert_array_equal(parts[name], before[name])


def test_summary_uses_pooled_wp_and_resamples_whole_chains_within_folds():
    rng = np.random.default_rng(35)
    y = rng.uniform(-3, 3, (4, 20, 2)).astype(np.float32)
    mask = rng.random(y.shape[:2]) > 0.2
    parts = {name: (y + rng.normal(size=y.shape)).astype(np.float32) for name in probe.COMPONENTS}
    pred, gate = probe.mixtures(parts)
    stats = np.stack([sums(y, p, mask) for p in pred])
    coverage = probe.coverage_sums(y, mask, gate)
    report = probe.summarize(stats, np.array([0, 0, 1, 1]), np.array([10, 10, 20, 20]),
                             np.array([0, 1, 0, 1]), coverage, 100, 13)
    for scope, selection in {"fold_0": slice(0, 2), "fold_1": slice(2, 4), "pooled": slice(None)}.items():
        scores = [direct_wp(y[selection], p[selection], mask[selection]) for p in pred]
        result = report["scores"][scope]
        assert list(result["wp"].values()) == pytest.approx(scores, abs=1e-12)
        assert result["comparisons"]["G_minus_U"]["delta"] == pytest.approx(scores[3] - scores[2], abs=1e-12)
        for comparison in result["comparisons"].values():
            # One whole chain in each stratum: every draw must reproduce the fixed comparison.
            assert comparison["conditional_chain_sd"] < 1e-14
        w = np.abs(np.clip(y[selection].astype(np.float64), -2, 2)) * mask[selection, ..., None]
        expected = (w * gate[selection, ..., None]).sum() / w.sum()
        assert result["coverage"]["metric_weight_fraction"] == pytest.approx(expected)
    assert report["confirmation_established"] is False
    with pytest.raises(ValueError, match="chain cannot be split"):
        probe.summarize(stats, np.array([0, 0, 1, 1]), np.array([10, 10, 10, 20]),
                        np.zeros(4), coverage, 100, 13)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(evidence_role="confirmation"),
    lambda m: m.update(folds=[1, 0]),
    lambda m: m.update(block_sequences=17),
    lambda m: m.update(block_sequences=True),
    lambda m: m["components"].update(plain3="substitute"),
    lambda m: m["gate"].update(pooled_delta_min=0.0),
    lambda m: m.update(bootstrap_seed=-1),
])
def test_unregistered_variant_rejected(mutation):
    spec = manifest()
    mutation(spec)
    with pytest.raises(ValueError):
        probe.validate_manifest(spec)


def test_main_hashes_before_scoring_uses_pure_oof_and_keeps_outputs_immutable(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(39)
    y = rng.uniform(-3, 3, (5, 20000, 2)).astype(np.float32)
    mask = np.ones(y.shape[:2], dtype=bool)
    mask[:, -2:] = False
    # Noncanonical row-group order verifies that classes and folds follow explicit cache ordering.
    rg = np.array([9, 3, 12, 6, 20])
    classes = np.array([0, 1, 2, 3, 0])
    folds = np.array([0, 0, 1, 1, 0])
    np.savez(cache / "valid_targets.npz", row_group=rg, y=y, mask=mask, market_class=classes)
    np.savez(cache / "chain_folds.npz", split=np.array(["valid"] * 5), row_group=rg,
             fold=folds, chain=np.array([10, 10, 20, 21, 99]), pure=np.array([True] * 4 + [False]))
    parts = {name: (rng.normal(0, 3, y.shape) + y).astype(np.float32) for name in probe.COMPONENTS}
    for name, run in probe.COMPONENTS.items():
        for fold in (0, 1):
            values = parts[name].copy()
            values[folds != fold] = -100  # Non-OOF outputs must never enter scoring.
            values[4] = 100              # Nor the impure chain.
            np.save(cache / f"pred_{run}_f{fold}_valid.npy", values)
    spec = manifest()
    source = tmp_path / "prereg.json"
    source.write_text(json.dumps(spec))
    out = tmp_path / "result.json"
    inventory = out.with_suffix(".inputs.json")
    original_mixtures = probe.mixtures

    def checked_mixtures(block):
        assert inventory.exists() and not out.exists()
        assert len(next(iter(block.values()))) <= 2
        return original_mixtures(block)

    monkeypatch.setattr(probe, "mixtures", checked_mixtures)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **kw: "synthetic-fixture\n")
    monkeypatch.setattr(sys, "argv", ["conditional_teacher_probe.py", str(source), "--out", str(out)])
    probe.main()
    result = json.loads(out.read_text())
    expected_pred, _ = original_mixtures({name: p[:4] for name, p in parts.items()})
    use_mask = mask[:4].copy()
    use_mask[:, :probe.WARMUP] = False
    expected = [direct_wp(y[:4], p, use_mask) for p in expected_pred]
    assert list(result["scores"]["pooled"]["wp"].values()) == pytest.approx(expected, abs=1e-12)
    assert result["scores"]["pooled"]["sequences"] == 4
    assert result["scores"]["pooled"]["chains"] == 3
    assert len(json.loads(inventory.read_text())["inputs"]) == 16
    frozen = out.read_bytes(), inventory.read_bytes(), source.read_bytes()
    with pytest.raises(FileExistsError):
        probe.main()
    assert frozen == (out.read_bytes(), inventory.read_bytes(), source.read_bytes())
