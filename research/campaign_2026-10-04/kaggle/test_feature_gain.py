from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from feature_gain import (  # noqa: E402
    FUTURE_GAP,
    LABEL_END,
    MATURITY_EVENTS,
    episode_anchor,
    gain_heads,
    input_summary,
    load_cache,
    realised_gain,
    run,
)
from feature_screen import digest_file  # noqa: E402

from wnn.research_features import ENDPOINT  # noqa: E402


def fixture(path: Path, first: int) -> None:
    rng = np.random.default_rng(10 + first)
    count = 128
    state = rng.normal(size=(count, 8)).astype(np.float32)
    gain = (1 + 0.3 * np.tanh(state[:, :2])).astype(np.float32)
    sequence = np.arange(first, first + count)
    anchor = np.full(count, 4200)
    prediction = rng.normal(scale=0.3, size=(count * 8, 2)).astype(np.float32)
    np.savez(
        path,
        state=state,
        summary=rng.normal(size=(count, 336)).astype(np.float32),
        gain=gain,
        seq_id=sequence,
        seq_component=sequence // 8,
        anchor=anchor,
        row=np.tile(4200 + FUTURE_GAP + np.arange(8), count),
        sequence=np.repeat(sequence, 8),
        component=np.repeat(sequence // 8, 8),
        pred=prediction,
        y=prediction * np.repeat(gain, 8, axis=0),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                version=1,
                sha256=digest_file(path),
                rows=count * 8,
                sequences=count,
                checkpoint_sha256="a" * 64,
                component_graph_sha256="b" * 64,
                outer_fold=0,
                fit_components=[9999],
                label_episode=[1024, LABEL_END],
                maturity_events=MATURITY_EVENTS,
                future_gap_rows=FUTURE_GAP,
            )
        )
    )


def test_maturity_buffer_and_input_summary_use_only_past_observations() -> None:
    raw = np.ones((5000, 112), dtype=np.float32)
    raw[LABEL_END + np.arange(MATURITY_EVENTS + 5) * 2, 110] = ENDPOINT
    anchor = episode_anchor(raw)
    assert anchor == LABEL_END + (MATURITY_EVENTS - 1) * 2
    before = input_summary(raw, anchor)
    raw[anchor + 1 :] *= 20
    assert episode_anchor(raw) == anchor
    np.testing.assert_array_equal(input_summary(raw, anchor), before)
    assert episode_anchor(raw[:anchor]) is None
    p = np.full((64, 2), 0.4, np.float32)
    gain = realised_gain(p * 2, p)
    assert np.all((gain > 1.9) & (gain < 2))
    with pytest.raises(ValueError, match="at least32"):
        realised_gain(p[:20], p[:20])


def test_planted_gain_recoverability_and_shuffled_negative_control(tmp_path: Path) -> None:
    fit_path, eval_path = tmp_path / "fit.npz", tmp_path / "evaluation.npz"
    fixture(fit_path, 0)
    fixture(eval_path, 1000)
    fit, _ = load_cache(fit_path)
    evaluation, _ = load_cache(eval_path)
    gains, _, transforms = gain_heads(fit, evaluation, "numpy")
    # Changing evaluation gain labels cannot change predictor normalization or coefficients.
    altered = dict(evaluation, gain=evaluation["gain"] + 1)
    repeated, _, repeated_transforms = gain_heads(fit, altered, "numpy")
    for key in gains:
        np.testing.assert_array_equal(gains[key], repeated[key])
    for key in transforms:
        np.testing.assert_array_equal(transforms[key], repeated_transforms[key])
    output = tmp_path / "results"
    run(fit_path, eval_path, output, bootstrap=100)
    report = json.loads((output / "report.json").read_text())
    recoverability = report["recoverability"]
    assert min(recoverability["candidate"]["r2_vs_fit_mean_per_target"]) > 0.7
    assert max(recoverability["shuffled_gain"]["r2_vs_fit_mean_per_target"]) < 0.15
    assert "candidate_minus_unchanged" in report["reports"]
    assert "candidate_minus_input_summary" in report["reports"]
    assert "earlier_label_oracle" in report["reports"]
    assert report["reports"]["candidate_minus_unchanged"]["n_sequences"] == 128
    assert "does not implement" in report["selection"]


def test_overlapping_components_or_episodes_are_rejected(tmp_path: Path) -> None:
    fit_path = tmp_path / "fit.npz"
    fixture(fit_path, 0)
    with pytest.raises(ValueError, match="separation"):
        run(fit_path, fit_path, tmp_path / "bad", bootstrap=100)
    data, metadata = load_cache(fit_path)
    data["row"][0] = LABEL_END - 1
    np.savez(fit_path, **data)
    metadata["sha256"] = digest_file(fit_path)
    fit_path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Overlapping episodes"):
        load_cache(fit_path)


def test_actual_parquet_episode_cache_skips_empty_earlier_labels(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import feature_gain as module
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from wnn import data
    from wnn.nn import RNNModel

    length = 5000
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", length)
    (tmp_path / "datasets").mkdir()
    columns, masks = [], []
    rng = np.random.default_rng(43)
    for group in range(5):
        raw = rng.normal(size=(length, 112)).astype(np.float32)
        raw[::2, 110] = ENDPOINT
        values = {name: raw[:, index] for index, name in enumerate(data.FEATURE_COLUMNS)}
        values.update(
            seq_ix=np.full(length, group),
            t0=rng.normal(size=length).astype(np.float32),
            t1=rng.normal(size=length).astype(np.float32),
        )
        columns.append(pa.table(values))
        scored = np.ones(length, dtype=bool)
        if group == 2:
            scored[1024:LABEL_END] = False
        masks.append(
            pa.table(
                dict(seq_ix=np.full(length, group), step_in_seq=np.arange(length), is_scored=scored)
            )
        )
    for path, tables in (
        (tmp_path / "datasets/valid.parquet", columns),
        (tmp_path / "datasets/valid_mask.parquet", masks),
    ):
        with pq.ParquetWriter(path, tables[0].schema) as writer:
            for table in tables:
                writer.write_table(table)
    model = RNNModel(211, hidden=4, layers=1)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        dict(
            epoch=3,
            n_in=211,
            model=model.state_dict(),
            args=dict(
                hidden=4,
                layers=1,
                checkpoint_epoch=3,
                checkpoint_policy="fixed",
                no_eval_during_fit=True,
            ),
            provenance=dict(fit_row_groups=dict(train=[0])),
        ),
        checkpoint,
    )
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    split = SimpleNamespace(
        fold=0,
        evaluation_groups=tuple(range(5)),
        components=[("valid", group, group) for group in range(5)] + [("train", 0, 100)],
    )
    monkeypatch.setattr(module, "load_campaign_split", lambda *args: split)

    def binding(path: Path) -> dict:
        return dict(path=path.relative_to(tmp_path).as_posix(), sha256=digest_file(path))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                source="valid",
                row_groups=list(range(5)),
                ancestors=[],
                checkpoint=binding(checkpoint),
                campaign_split=binding(split_path),
                mask=binding(tmp_path / "datasets/valid_mask.parquet"),
            )
        )
    )
    output = tmp_path / "cache.npz"
    import weakref

    original_frozen_state = module.frozen_state
    state_references = []

    def bounded_frozen_state(*args, **kwargs):
        # At most the previous sequence may remain in the extractor's local
        # variable. Anchors must not keep earlier full-sequence buffers alive.
        if len(state_references) >= 2:
            assert state_references[-2]() is None
        state, prediction = original_frozen_state(*args, **kwargs)
        state_references.append(weakref.ref(state))
        return state, prediction

    monkeypatch.setattr(module, "frozen_state", bounded_frozen_state)
    module.extract(tmp_path, manifest, output)
    assert len(state_references) >= 3
    assert all(reference() is None for reference in state_references)
    cache, metadata = load_cache(output)
    assert metadata["sequences"] == 4
    assert cache["seq_id"].tolist() == [0, 1, 3, 4]
    assert metadata["sampling_report"][2]["status"] == "insufficient_prespecified_episode_samples"
    assert np.all(cache["anchor"] == 4190)
