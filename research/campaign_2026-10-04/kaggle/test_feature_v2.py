from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from feature_screen import digest_file, load_cache  # noqa: E402
from feature_v2_cache import train_router, training_sample  # noqa: E402
from feature_v2_filters import (  # noqa: E402
    ema,
    fit_router,
    histories,
    route,
    trailing_mean,
    warmup_summary,
)
from feature_v2_recipes import recipe  # noqa: E402
from feature_v2_screen import FAMILIES, build_designs, load_histories, run  # noqa: E402


def fixture(path: Path, first: int) -> Path:
    rng = np.random.default_rng(first + 31)
    # Eight sequences, four components: the bootstrap must not mislabel them.
    sequence = np.repeat(np.arange(first, first + 8), 16)
    raw = rng.normal(size=(512, 112)).astype(np.float32)
    # Create event gaps, so row-clock and event-clock features differ.
    raw[:, :22] = raw[np.arange(512) // 4 * 4, :22]
    values = histories(raw)
    selected = np.arange(250, 378)
    target = values["clock_event"][selected, :2]
    base = dict(
        state=rng.normal(size=(128, 8)).astype(np.float32),
        pred=rng.normal(scale=0.1, size=(128, 2)).astype(np.float32),
        y=target,
        fit_weight=np.ones(128),
        sequence=sequence,
        component=sequence // 2,
        row=selected,
        model_id=np.full(128, "a" * 64),
    )
    np.savez(path, **base)
    metadata = dict(
        sha256=digest_file(path),
        rows=128,
        state_width=8,
        outer_fold=0,
        component_graph_sha256="b" * 64,
        members={"a" * 64: dict(checkpoint_sha256="a" * 64, fit_components=[100])},
    )
    path.with_suffix(".json").write_text(json.dumps(metadata))
    extra_path = path.with_name(path.stem + "_history.npz")
    np.savez(
        extra_path,
        **{key: value[selected] for key, value in values.items()},
        regime=(sequence % 4)[:, None],
        row=selected,
        sequence=sequence,
        component=sequence // 2,
    )
    extra_path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                version=2,
                sha256=digest_file(extra_path),
                base_cache_sha256=metadata["sha256"],
                component_graph_sha256="b" * 64,
                router_sha256="c" * 64,
                router_fit_components=[100],
            )
        )
    )
    return extra_path


def test_filters_are_causal_and_have_exact_clock_cycle_controls() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5000, 112)).astype(np.float32)
    raw[:, :22] = raw[np.arange(len(raw)) // 8 * 8, :22]
    before = histories(raw)
    altered = raw.copy()
    altered[200:] += 50
    after = histories(altered)
    for key in before:
        np.testing.assert_array_equal(before[key][:200], after[key][:200])
    np.testing.assert_array_equal(warmup_summary(raw), warmup_summary(altered))
    assert not np.array_equal(before["clock_row"], before["clock_event"])
    np.testing.assert_array_equal(before["clock_stale_event"][:4096], 0)
    np.testing.assert_array_equal(before["clock_stale_event"][4096:], before["clock_event"][:-4096])
    np.testing.assert_array_equal(before["cycle8"][:, :112], ema(trailing_mean(raw, 8), 1 / 32))
    np.testing.assert_array_equal(before["row_matched"][:, :112], ema(raw, 1 / 35.5))
    np.testing.assert_array_equal(
        before["cycle7_matched"][:, :112], ema(trailing_mean(raw, 7), 1 / 32.5)
    )
    ramp = np.arange(20)[:, None].astype(np.float32)
    assert trailing_mean(ramp, 8)[10, 0] == 6.5


def test_router_balancing_validation_and_training_sample() -> None:
    rng = np.random.default_rng(21)
    labels = np.tile(np.arange(4), 10)
    features = np.eye(4)[labels] + rng.normal(scale=0.01, size=(40, 4))
    parameters = fit_router(features, labels)
    np.testing.assert_array_equal(route(features, parameters), labels)
    with pytest.raises(ValueError, match="four TRAIN"):
        fit_router(features, labels.astype(float))
    groups = tuple(range(100))
    assert training_sample(groups, 20) == training_sample(tuple(reversed(groups)), 20)


def test_router_uses_train_row_group_mapping_and_warmup_only(tmp_path: Path, monkeypatch) -> None:
    import feature_v2_cache as module

    (tmp_path / "cache").mkdir()
    labels_path = tmp_path / "cache/train_targets.npz"
    # Excluded row group1 models the corrupt TRAIN row-group hole in real caches.
    groups = np.array([0, 2, 3, 4])
    np.savez(labels_path, row_group=groups, seq_ix=groups + 10, market_class=np.arange(4))
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"campaign_split": {"path": "split.json", "sha256": digest_file(split_path)}})
    )
    fitting = []
    split = SimpleNamespace(
        train_groups=tuple(groups),
        components=[("train", int(g), int(g)) for g in groups],
        assert_fit_groups=lambda train, valid: fitting.append((train, valid)),
    )
    monkeypatch.setattr(module, "load_campaign_split", lambda *args: split)
    accessed = []

    class Sequences:
        def __init__(self, path):
            assert path.name == "train.parquet"

        def __len__(self):
            return 5

        def read(self, group, *, targets, mask):
            assert not targets and not mask
            accessed.append(group)
            raw = np.full((120, 112), group, np.float32)
            raw[99:] = 1000
            return SimpleNamespace(seq_ix=group + 10, features=raw)

    monkeypatch.setattr(module, "SequenceFile", Sequences)
    output = tmp_path / "router.npz"
    train_router(tmp_path, manifest, labels_path, output, limit=4)
    info = json.loads(output.with_suffix(".json").read_text())
    assert info["class_counts"] == [1, 1, 1, 1]
    assert accessed == groups.tolist() and fitting == [(groups.tolist(), [])]
    assert info["feature_rows"] == [0, 98] and info["fit_components"] == groups.tolist()
    with pytest.raises(ValueError, match="TRAIN target"):
        train_router(
            tmp_path, manifest, tmp_path / "cache/valid_targets.npz", tmp_path / "bad.npz", 4
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_every_exact_family_reports_untouched_affine_and_matched_controls(
    tmp_path: Path, family: str
) -> None:
    fit_path, eval_path = tmp_path / "fit.npz", tmp_path / "eval.npz"
    fit_extra, eval_extra = fixture(fit_path, 0), fixture(eval_path, 8)
    fit, fm = load_cache(fit_path)
    evaluation, em = load_cache(eval_path)
    fh, _ = load_histories(fit_extra, fit, fm)
    eh, _ = load_histories(eval_extra, evaluation, em)
    designs, transforms = build_designs(fit, evaluation, fh, eh, family)
    changed = dict(evaluation, y=evaluation["y"] * 100, pred=evaluation["pred"])
    _, repeated = build_designs(fit, changed, fh, eh, family)
    for key in transforms:
        np.testing.assert_array_equal(transforms[key], repeated[key])
    assert designs["candidate"][0][0].shape[1] == designs["capacity"][0][0].shape[1]
    output = tmp_path / "result"
    run(fit_path, eval_path, fit_extra, eval_extra, family, "numpy", output, bootstrap=100)
    report = json.loads((output / "report.json").read_text())
    assert report["evaluation_sequences"] == 8 and report["evaluation_components"] == 4
    assert report["reports"]["candidate_minus_unchanged"]["n_sequences"] == 8
    assert "candidate_minus_global_affine" in report["reports"]
    assert report["selection"].startswith("none")
    assert (output / "transforms.npz").exists()


def test_history_leakage_and_content_mismatch_fail_closed(tmp_path: Path) -> None:
    fit_path, eval_path = tmp_path / "fit.npz", tmp_path / "eval.npz"
    fit_extra, eval_extra = fixture(fit_path, 0), fixture(eval_path, 8)
    metadata = json.loads(eval_extra.with_suffix(".json").read_text())
    metadata["router_fit_components"] = [4]
    eval_extra.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Router fit/evaluation"):
        run(fit_path, eval_path, fit_extra, eval_extra, "cycle8", "numpy", tmp_path / "bad", 100)
    fit_extra.write_bytes(b"tamper")
    fit, fm = load_cache(fit_path)
    with pytest.raises(ValueError, match="content identity"):
        load_histories(fit_extra, fit, fm)


def test_history_extraction_reuses_exact_base_rows(tmp_path: Path, monkeypatch) -> None:
    import feature_v2_cache as module

    base_path = tmp_path / "fit.npz"
    fixture(base_path, 0)
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            dict(
                campaign_split=dict(path="split.json", sha256=digest_file(split_path)),
                source="valid",
                row_groups=list(range(8)),
            )
        )
    )
    base_meta = json.loads(base_path.with_suffix(".json").read_text())
    base_meta["selection_manifest_sha256"] = digest_file(manifest_path)
    base_meta["component_graph_sha256"] = digest_file(split_path)
    base_path.with_suffix(".json").write_text(json.dumps(base_meta))
    router_path = tmp_path / "router.npz"
    np.savez(router_path, mean=np.zeros(336), scale=np.ones(336), beta=np.zeros((337, 4)))
    router_path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                sha256=digest_file(router_path),
                component_graph_sha256=digest_file(split_path),
                label_source="TRAIN market_class only",
                feature_rows=[0, 98],
                fit_components=[100],
                fitted_row_groups=[10],
            )
        )
    )
    split = SimpleNamespace(
        components=[("valid", group, group // 2) for group in range(8)],
        assert_fit_groups=lambda *args: None,
    )
    monkeypatch.setattr(module, "load_campaign_split", lambda *args: split)
    raw = np.random.default_rng(2).normal(size=(400, 112)).astype(np.float32)

    class Sequences:
        def __init__(self, path):
            assert path.name == "valid.parquet"

        def read(self, group, *, targets, mask):
            assert not targets and not mask
            return SimpleNamespace(seq_ix=group, features=raw)

    monkeypatch.setattr(module, "SequenceFile", Sequences)
    output = tmp_path / "exact.npz"
    module.extract(tmp_path, manifest_path, base_path, router_path, output)
    base, metadata = load_cache(base_path)
    extra, _ = load_histories(output, base, metadata)
    np.testing.assert_array_equal(extra["cycle8"], histories(raw)["cycle8"][base["row"]])
    np.testing.assert_array_equal(extra["regime"], 0)
    np.testing.assert_array_equal(extra["sequence"], base["sequence"])


def test_sealed_cpu_gpu_specs_and_missing_artifact_refusal(tmp_path: Path) -> None:
    for directory in ("scripts", "src"):
        (tmp_path / directory).symlink_to(ROOT / directory, target_is_directory=True)
    fixture(tmp_path / "fit.npz", 0)
    fixture(tmp_path / "eval.npz", 8)
    for backend in ("cpu", "gpu"):
        result = recipe(
            tmp_path,
            name="fixture",
            operation="screen",
            backend=backend,
            fit="fit.npz",
            evaluation="eval.npz",
            fit_histories="fit_history.npz",
            eval_histories="eval_history.npz",
            family="cycle8",
            seal=True,
        )
        assert len(result["inputs"]) == 8 and len(result["code_sha256"]) == 64
        assert result["kind"] == backend and result["limits"]["cpus"] == 1
    with pytest.raises(FileNotFoundError):
        recipe(
            tmp_path,
            name="missing",
            operation="screen",
            fit="missing.npz",
            evaluation="eval.npz",
            fit_histories="fit_history.npz",
            eval_histories="eval_history.npz",
            family="cycle8",
            seal=True,
        )
