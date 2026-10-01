from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import utils
from wnn import evaluation, gmetric

RESULTS: list[dict] = []


def oracle(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    """Apply the published weighted centered formula in extended precision."""
    y = np.asarray(y, dtype=np.float32)[mask].astype(np.longdouble)
    p = np.asarray(p, dtype=np.float32)[mask].astype(np.longdouble)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    y, p = np.clip(y, -2, 2), np.clip(p, -2, 2)
    result = []
    for column in range(2):
        weights = np.abs(y[:, column])
        total = weights.sum()
        if total < np.longdouble("1e-8"):
            result.append(0.0)
            continue
        yc = y[:, column] - (weights * y[:, column]).sum() / total
        pc = p[:, column] - (weights * p[:, column]).sum() / total
        vy = (weights * yc * yc).sum() / total
        vp = (weights * pc * pc).sum() / total
        if np.sqrt(vy) <= np.longdouble("1e-8") or np.sqrt(vp) <= np.longdouble("1e-8"):
            result.append(0.0)
        else:
            result.append(float((weights * yc * pc).sum() / total / np.sqrt(vy * vp)))
    return float(np.mean(result))


def check(name: str, action: object) -> None:
    """Run an independent assertion and preserve every failure's traceback."""
    started = time.perf_counter()
    try:
        details = action()
        record = {"name": name, "status": "pass", "details": details}
    except Exception as error:
        record = {
            "name": name,
            "status": "fail",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    record["elapsed_seconds"] = time.perf_counter() - started
    RESULTS.append(record)
    print(json.dumps(record), flush=True)


def parity() -> dict:
    """Check masks, clipping, groups and arbitrary streaming partitions."""
    rng = np.random.default_rng(909731)
    maximum = 0.0
    for trial in range(80):
        shape = (2 + trial % 4, 131 + trial, 2)
        y = rng.normal(size=shape).astype(np.float32) * (1 + trial % 3)
        p = (y * 0.6 + rng.normal(size=shape)).astype(np.float32)
        mask = rng.random(shape[:-1]) < 0.4
        y[~mask] = np.nan
        p[~mask] = np.inf
        expected = oracle(y, p, mask)
        research = gmetric.global_wp(y, p, mask)["wp"]
        standalone = utils.GlobalAccumulator()
        optimized = gmetric.GlobalAccumulator()
        for group in range(len(y)):
            for start in range(0, y.shape[1], 17):
                standalone.add(
                    y[group, start : start + 17],
                    p[group, start : start + 17],
                    mask[group, start : start + 17],
                )
                optimized.add(
                    y[group, start : start + 17],
                    p[group, start : start + 17],
                    mask[group, start : start + 17],
                    group=group,
                )
        values = (research, standalone.result()["wp"], optimized.result()["wp"])
        maximum = max(maximum, *(abs(value - expected) for value in values))
        assert maximum < 2e-12, (trial, expected, values)
        assert optimized.selected_rows == int(mask.sum())
        assert np.allclose(sum(optimized.groups.values()), optimized.sums)
    return {"trials": 80, "max_absolute_oracle_error": maximum}


def boundaries() -> dict:
    y = np.array([[1, -1], [2, -2], [-2, 2], [-1, 1]], dtype=np.float32)
    mask = np.ones(4, dtype=bool)
    for bad in (np.nan, np.inf, -np.inf):
        p = y.copy()
        p[0, 0] = bad
        for function in (gmetric.global_wp, gmetric.reference_global_wp):
            try:
                function(y, p, mask)
            except ValueError:
                pass
            else:
                raise AssertionError((function.__name__, bad))
        for accumulator in (utils.GlobalAccumulator(), gmetric.GlobalAccumulator()):
            try:
                accumulator.add(y, p, mask)
            except ValueError:
                pass
            else:
                raise AssertionError((type(accumulator), bad))
    for p in (np.zeros((4, 3)), np.zeros((2, 2))):
        try:
            gmetric.global_wp(y, p, mask)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong shape accepted")
    for targets, predictions in ((np.zeros((4, 2)), y), (y, np.ones((4, 2)))):
        assert gmetric.global_wp(targets, predictions, mask)["wp"] == 0.0
    try:
        gmetric.global_wp(y, y, np.zeros(4, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("empty selection accepted")
    return {"raw_nonfinite_rejected": True, "shape_zero_weight_constant_empty_checked": True}


def near_constant() -> dict:
    lower = np.nextafter(np.float32(1), np.float32(0))
    y = np.tile(np.array([[lower, lower], [1, 1]], dtype=np.float32), (1024, 1))
    mask = np.ones(len(y), dtype=bool)
    expected = oracle(y, y, mask)
    assert expected > 0.999999999
    values = []
    for width in (1, 3, 17, 2048):
        for factory in (gmetric.GlobalAccumulator, utils.GlobalAccumulator):
            accumulator = factory()
            for start in range(0, len(y), width):
                accumulator.add(
                    y[start : start + width], y[start : start + width], mask[start : start + width]
                )
            values.append(accumulator.result()["wp"])
    assert max(abs(value - expected) for value in values) < 1e-6, values
    return {"expected": expected, "stream_results": values}


def near_constant_bootstrap() -> dict:
    lower = np.nextafter(np.float32(1), np.float32(0))
    y = np.tile(np.array([[lower, lower], [1, 1]], dtype=np.float32), (64, 1))
    p = np.ones_like(y)
    mask = np.ones(len(y), dtype=bool)
    a = np.stack([gmetric.wp_sums(y[:, column], y[:, column], mask) for column in range(2)])
    b = np.stack([gmetric.wp_sums(y[:, column], p[:, column], mask) for column in range(2)])
    try:
        evaluation.paired_component_bootstrap(
            np.stack([a, a]), np.stack([b, b]), np.array([0, 1]), n_boot=30
        )
    except ValueError as error:
        assert "precision" in str(error)
    else:
        raise AssertionError("ambiguous raw bootstrap moments were accepted")
    centered_a, centered_b = evaluation.paired_component_moments(
        np.stack([y, y]), np.stack([y, y]), np.stack([p, p]), np.stack([mask, mask])
    )
    result = evaluation.paired_component_bootstrap(
        centered_a, centered_b, np.array([0, 1]), n_boot=30, centered=True
    )
    expected = oracle(y, y, mask) - oracle(y, p, mask)
    assert abs(result["mean_diff"] - expected) < 1e-6, {"expected": expected, "actual": result}
    return {"expected_delta": expected, "actual": result}


def near_constant_component_subset() -> dict:
    """Bootstrap must stay accurate when samples move far from the pooled anchor."""
    upper = np.nextafter(np.float32(1.5), np.float32(0))
    one = np.tile(np.array([[upper, upper], [1.5, 1.5]], dtype=np.float32), (64, 1))
    y = np.stack([one, -one])
    p = np.ones_like(y)
    mask = np.ones(y.shape[:-1], dtype=bool)
    a, b = evaluation.paired_component_moments(y, y, p, mask)
    result = evaluation.paired_component_bootstrap(
        a, b, np.array([0, 1]), n_boot=200, centered=True
    )
    assert all(
        abs(
            oracle(y[indices], y[indices], mask[indices])
            - oracle(y[indices], p[indices], mask[indices])
            - 1
        )
        < 1e-9
        for indices in ([0, 0], [0, 1], [1, 1])
    )
    assert result["bootstrap_sd"] < 1e-10 and np.allclose(result["ci95"], [1, 1]), result
    return result


def raw_resample_precision_refusal() -> dict:
    """Well-conditioned pooled data must not hide ambiguous raw resamples."""
    center = np.float32(0.35)
    lower = np.nextafter(center, np.float32(0))
    one = np.tile(np.array([[lower, lower], [center, center]], dtype=np.float32), (16, 1))
    y = np.stack([one, -one])
    mask = np.ones(y.shape[:-1], dtype=bool)
    a = np.stack(
        [gmetric.wp_sums(y[..., column], y[..., column], mask) for column in range(2)], axis=1
    )
    b = np.stack(
        [gmetric.wp_sums(y[..., column], -y[..., column], mask) for column in range(2)], axis=1
    )
    try:
        evaluation.paired_component_bootstrap(a, b, np.array([0, 1]), n_boot=50)
    except ValueError as error:
        assert "precision" in str(error)
    else:
        raise AssertionError("precision-ambiguous raw resamples were accepted")
    return {"ambiguous_subsample_refused": True, "pooled_candidate_oracle": oracle(y, y, mask)}


def bootstrap_exact_replay() -> dict:
    rng = np.random.default_rng(606)
    y = rng.normal(size=(7, 200, 2)).astype(np.float32)
    candidate = y + rng.normal(size=y.shape).astype(np.float32) * 0.4
    reference = y + rng.normal(size=y.shape).astype(np.float32) * 0.9
    mask = rng.random(y.shape[:-1]) < 0.6
    component = np.array([10, 10, 20, 30, 30, 30, 40])

    def sums(p: np.ndarray) -> np.ndarray:
        return np.stack(
            [gmetric.wp_sums(y[..., column], p[..., column], mask) for column in range(2)], axis=1
        )

    actual = evaluation.paired_component_bootstrap(
        sums(candidate), sums(reference), component, n_boot=60, seed=801
    )
    moments_a, moments_b = evaluation.paired_component_moments(y, candidate, reference, mask)
    centered = evaluation.paired_component_bootstrap(
        moments_a, moments_b, component, n_boot=60, seed=801, centered=True
    )
    resample = np.random.default_rng(801)
    ids = np.unique(component)
    deltas = []
    for _ in range(60):
        sampled = resample.integers(0, len(ids), len(ids))
        rows = np.concatenate([np.flatnonzero(component == ids[index]) for index in sampled])
        deltas.append(
            oracle(y[rows], candidate[rows], mask[rows])
            - oracle(y[rows], reference[rows], mask[rows])
        )
    for result in (actual, centered):
        assert np.allclose(result["ci95"], np.percentile(deltas, [2.5, 97.5]), atol=1e-12)
        assert abs(result["bootstrap_sd"] - np.std(deltas, ddof=1)) < 1e-12
    return {"raw": actual, "centered": centered}


def research_evaluation_no_block_eligibility() -> dict:
    trainer = importlib.import_module("train_rnn")
    y = np.zeros((2, 20_000, 2), dtype=np.float32)
    y[0] = -1
    y[1] = 1
    mask = np.ones(y.shape[:-1], dtype=bool)
    mask[:, :99] = False
    loader = [(torch.zeros((2, 20_000, 112)), torch.from_numpy(y), torch.from_numpy(mask))]
    with (
        patch.object(trainer, "predict", return_value=y),
        patch.object(trainer, "market_class", return_value=3),
    ):
        result = trainer.evaluate(None, loader, 4000, "cpu")
    assert result["weighted_pearson"] == 1.0
    assert result["by_class"][3] == 1.0
    return result


def submission_multisequence() -> dict:
    verifier = importlib.import_module("verify_submission")
    with tempfile.TemporaryDirectory(prefix="wnn-independent-metric-") as directory:
        base = Path(directory)
        data = base / "fixture.parquet"
        with pq.ParquetWriter(
            data,
            schema=pa.schema(
                [
                    pa.field("seq_ix", pa.int64()),
                    pa.field("step_in_seq", pa.int64()),
                    pa.field("need_prediction", pa.bool_()),
                    pa.field("is_scored", pa.bool_()),
                    *[pa.field(name, pa.float32()) for name in utils.FEATURE_COLUMNS],
                    *[pa.field(name, pa.float32()) for name in utils.TARGET_COLUMNS],
                ]
            ),
        ) as writer:
            for group, value in enumerate((-1.0, -0.5, 1.0)):
                values = {
                    "seq_ix": np.full(20_000, 610 + group),
                    "step_in_seq": np.arange(20_000),
                    "need_prediction": np.arange(20_000) >= 99,
                    "is_scored": (np.arange(20_000) >= 99) & (np.arange(20_000) % 17 == 0),
                }
                values.update(
                    {
                        name: np.full(20_000, value if column < 2 else 0, dtype=np.float32)
                        for column, name in enumerate(utils.FEATURE_COLUMNS)
                    }
                )
                values.update(
                    {
                        name: np.full(20_000, value, dtype=np.float32)
                        for name in utils.TARGET_COLUMNS
                    }
                )
                writer.write_table(pa.table(values), row_group_size=20_000)
        package = base / "fixture.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(
                "solution.py",
                "class PredictionModel:\n    def predict(self, point):\n        return point.state[:2].copy() if point.need_prediction else None\n",
            )
        outputs = []
        for suffix in ("npz", "npy"):
            output = base / f"pred.{suffix}"
            result = verifier.verify_submission(package, data, output)
            assert result["global_wp"] == 1.0 and result["block_wp"] is None
            assert result["sequences"] == 3 and result["rows"] == 60_000
            if suffix == "npz":
                with np.load(output) as arrays:
                    prediction = arrays["pred"]
                    assert prediction.shape == (3, 20_000, 2)
                    assert prediction[:, 100, 0].tolist() == [-1.0, -0.5, 1.0]
            else:
                assert np.load(output).shape == (3, 20_000, 2)
            outputs.append(result)
        return {"outputs": outputs}


def main() -> None:
    started = time.perf_counter()
    names = (
        "src/wnn/metric.py",
        "src/wnn/gmetric.py",
        "src/wnn/evaluation.py",
        "utils.py",
        "scripts/train_rnn.py",
        "scripts/verify_submission.py",
    )
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    for name, action in (
        ("extended_precision_random_stream_parity", parity),
        ("raw_boundaries", boundaries),
        ("near_constant_stream", near_constant),
        ("component_bootstrap_replay", bootstrap_exact_replay),
        ("near_constant_bootstrap", near_constant_bootstrap),
        ("near_constant_component_subset", near_constant_component_subset),
        ("raw_resample_precision_refusal", raw_resample_precision_refusal),
        ("research_evaluation_no_block_eligibility", research_evaluation_no_block_eligibility),
        ("submission_multisequence_npz_npy", submission_multisequence),
    ):
        check(name, action)
    report = {
        "status": "pass" if all(row["status"] == "pass" for row in RESULTS) else "fail",
        "elapsed_seconds": time.perf_counter() - started,
        "checks": RESULTS,
        "hashes": hashes,
        "hashes_after": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
        },
        "official_source_sha256": hashlib.sha256(
            (
                ROOT / "research/campaign_2026-09-29/sources/connectome_data_overview.html"
            ).read_bytes()
        ).hexdigest(),
    }
    (REPORT / "independent_metric_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "failures": [row["name"] for row in RESULTS if row["status"] == "fail"],
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
