"""CPU tests of the H-350 out-of-sample path distillation (synthetic tiny GRU, no data, no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = SOURCE_ROOT.parent / "h230_source_v8_arch"
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
sys.path.insert(0, str(SOURCE_ROOT / "src"))  # this snapshot's wnn, not the editable root install

import predict_run  # noqa: E402
import train_rnn  # noqa: E402

from wnn import ood_student as ood  # noqa: E402
from wnn.data import N_FEATURES, WARMUP  # noqa: E402
from wnn.training_views import origin_windows  # noqa: E402

CPU = torch.device("cpu")
ARGS = {"hidden": 16, "layers": 1, "dropout": 0.0}
SEQ_ROWS = 3000


def fake_sequence(ident: int, rows: int = SEQ_ROWS) -> np.ndarray:
    """Column 111 identifies (sequence, row); price/volume columns hold per element on 40% and wholly on 30% of rows."""
    g = np.random.default_rng(ident)
    x = g.standard_normal((rows, N_FEATURES)).astype(np.float32)
    hold = g.random((rows, N_FEATURES)) < 0.4
    hold[0] = False
    for t in range(1, rows):
        x[t] = np.where(hold[t], x[t - 1], x[t])
    for t in range(1, rows):
        if g.random() < 0.3:
            x[t, :104] = x[t - 1, :104]
    x[:, 111] = ident * 100_000 + np.arange(rows)
    return x


class Reader:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def __call__(self, split: str, group: int) -> np.ndarray:
        self.calls.append((split, group))
        return fake_sequence(group + (7000 if split == "valid" else 0))


FIT = [("train", g) for g in range(40)] + [("valid", g) for g in range(10, 20)]
EVAL = frozenset(("valid", g) for g in range(0, 10))


def sampler(augs="splice", rows=2000, **kw):
    reader = Reader()
    s = ood.PathSampler(FIT, reader, ood.parse_ood_aug(augs), seed=5, rows=rows, sources=8, forbidden=EVAL, **kw)
    return s, reader


# --- parsing and schedule ---------------------------------------------------------------------------------------

def test_parsing_and_rejections():
    assert ood.parse_ood_aug("splice,warp:1/0.5/2,noise:0.1,mixup_paths") == {
        "splice": (), "warp": (1.0, 0.5, 2.0), "noise": (0.1,), "mixup_paths": (0.5,)}
    for bad in ("", "splice,splice", "teleport", "warp:2/1/1", "noise:0.9", "sign_flip_bidask"):
        with pytest.raises(ValueError):
            ood.parse_ood_aug(bad)
    assert ood.parse_teacher_specs("a@3,b@ep4") == [("a", 3), ("b", 4)]
    with pytest.raises(ValueError):
        ood.parse_teacher_specs("a")


def test_batch_schedule_is_deterministic_and_has_the_requested_share():
    pattern = [ood.is_synthetic_batch("mix", 0.3, 9, 0, b) for b in range(2000)]
    assert pattern == [ood.is_synthetic_batch("mix", 0.3, 9, 0, b) for b in range(2000)]
    assert abs(np.mean(pattern) - 0.3) < 0.04
    assert [ood.is_synthetic_batch("alternate", 0.5, 0, 0, b) for b in range(6)] == [False, True] * 3
    assert all(ood.is_synthetic_batch("only", 0.1, 0, 0, b) for b in range(5))


# --- sampler: no leakage, causality -------------------------------------------------------------------------------

def test_sampler_reads_only_the_fit_split():
    s, reader = sampler("splice,warp,noise,mixup_paths")
    for step in range(6):
        s.batch(3, 0, step)
    assert set(reader.calls) <= set(FIT)
    assert s.used <= set(FIT) and not (s.used & EVAL)
    with pytest.raises(ValueError):  # fit/evaluation overlap is refused at construction
        ood.PathSampler(FIT + [("valid", 3)], Reader(), ood.parse_ood_aug("splice"), seed=1, forbidden=EVAL)
    with pytest.raises(AssertionError):  # a read outside the fit set is refused
        s._load(("valid", 3))


def test_splice_rows_are_contiguous_source_segments_without_state_reset():
    s, _ = sampler("splice")
    x, kept = s.batch(2, 0, 0)
    assert x.shape == (2, 2000, N_FEATURES) and kept.tolist() == [2000, 2000]
    for b in range(2):
        segs = s.last_trace[b]["segments"]
        assert sum(seg["length"] for seg in segs) == 2000
        assert all(ood.SPLICE_MIN <= seg["length"] <= ood.SPLICE_MAX for seg in segs[:-1])
        assert len({seg["id"] for seg in segs}) > 1  # several sequences, joined without any reset marker
        for seg in segs:
            ident = seg["id"]
            expected = fake_sequence(ident[1] + (7000 if ident[0] == "valid" else 0))
            out = x[b, seg["at"]:seg["at"] + seg["length"]].numpy()
            assert np.array_equal(out, expected[seg["start"]:seg["start"] + seg["length"]])


def test_batches_are_deterministic_in_seed_epoch_step():
    a, _ = sampler("splice,warp,noise,mixup_paths")
    b, _ = sampler("splice,warp,noise,mixup_paths")
    assert torch.equal(a.batch(2, 1, 3)[0], b.batch(2, 1, 3)[0])
    assert not torch.equal(a.batch(2, 1, 3)[0], a.batch(2, 1, 4)[0])
    state = torch.random.get_rng_state()
    a.batch(2, 0, 0)
    assert torch.equal(state, torch.random.get_rng_state())  # global torch RNG untouched (real-batch parity)


def test_noise_is_causal_keeps_static_rows_and_is_identity_at_zero_sigma():
    x = fake_sequence(3, 1500)
    table = np.quantile(x, np.linspace(0, 1, 257), axis=0).T.astype(np.float32)
    s, _ = sampler("noise:0.0", rows=1500, quantile_table=table)
    ident = s._noise(x, 0.0, np.random.default_rng(0), {})
    assert np.abs(ident - x).max() < 1e-4
    same_in = np.concatenate([np.ones((1, N_FEATURES), bool), x[1:] == x[:-1]])
    noisy = s._noise(x, 0.2, np.random.default_rng(0), {})
    same_in = np.concatenate([np.ones((1, N_FEATURES), bool), x[1:] == x[:-1]])
    assert np.array_equal(noisy[1:][same_in[1:]], noisy[:-1][same_in[1:]])  # held source value stays held
    assert (noisy != x).any()
    # causality: changing source rows after t0 cannot alter output rows up to t0
    t0 = 700
    altered = x.copy()
    altered[t0 + 1:] += 5.0
    again = s._noise(altered, 0.2, np.random.default_rng(0), {})
    assert np.array_equal(noisy[:t0 + 1], again[:t0 + 1])


def test_mixup_blends_book_columns_and_keeps_event_columns_from_one_path():
    s, _ = sampler("mixup_paths:1")
    a, b = fake_sequence(1, 800), fake_sequence(2, 800)
    trace: dict = {}
    out = s._mixup(a, b, np.random.default_rng(0), trace)
    lam = trace["mixup_lambda"]
    cols = ood.CONTINUOUS_COLUMNS
    assert np.allclose(out[:, cols], lam * a[:, cols] + (1 - lam) * b[:, cols], atol=1e-4)
    rest = np.setdiff1d(np.arange(N_FEATURES), cols)
    assert np.array_equal(out[:, rest], a[:, rest])


def test_warp_repeats_only_static_rows_in_order_and_masks_padding():
    s, _ = sampler("warp:1/0.5/0.5", rows=1500)
    x, kept = s.batch(3, 0, 0)
    assert (kept > WARMUP).all() and (kept <= 1500).all()
    assert (kept < 1500).any()  # factor 0.5 drops static rows, so the tail is padded and masked
    for b in range(3):
        rows = x[b, :kept[b]].numpy()
        # monotone time: the identifying column never decreases within a source segment (single sequence here)
        assert (np.diff(rows[:, 111]) >= 0).all()


@pytest.mark.parametrize("augs", ["splice", "splice,warp,noise,mixup_paths", "noise", "mixup_paths:1"])
def test_event_features_of_generated_paths_are_causal(augs):
    s, _ = sampler(augs, rows=1200)
    x, _ = s.batch(2, 0, 0)
    t0 = 500
    changed = x.clone()
    changed[:, t0 + 1:] = changed[:, t0 + 1:] * 1.7 + 3.0
    a, b = train_rnn.build_inputs(x), train_rnn.build_inputs(changed)
    assert torch.equal(a[:, :t0 + 1], b[:, :t0 + 1])


# --- teachers -----------------------------------------------------------------------------------------------------

def tiny_models(seed_offsets=(0, 1)):
    x = torch.randn(2, 150, N_FEATURES, generator=torch.Generator().manual_seed(4))
    n_in = train_rnn.build_inputs(x).shape[-1]
    models = []
    for k in seed_offsets:
        torch.manual_seed(10 + k)
        model = train_rnn.make_model(ARGS, n_in, 2)
        train_rnn.fit_normalization(model, [(x,)], CPU, batches=1)
        models.append(model.eval())
    return models, x, n_in


def online(models, chunk):
    return ood.OnlineTeachers(models, train_rnn.forward_chunk, chunk, lambda: torch.autocast("cpu", enabled=False))


def test_online_teacher_targets_equal_offline_predict_run_outputs():
    models, x, _ = tiny_models()
    offline = np.mean([train_rnn.predict(m, x, 150, CPU) for m in models], axis=0)
    for chunk in (150, 40):
        got = online(models, chunk)(train_rnn.build_inputs(x)).numpy()
        assert np.allclose(got, offline, atol=1e-5)
    assert all(not p.requires_grad for m in models for p in m.parameters())


def test_checkpoint_round_trip_and_loader_ancestry(tmp_path, monkeypatch):
    models, x, n_in = tiny_models()
    for k, model in enumerate(models):
        run = tmp_path / "artifacts" / "runs" / f"t{k}"
        run.mkdir(parents=True)
        provenance = {"split_manifest_sha256": "abc", "fit_row_groups": {"train": [1], "valid": [10, 11]}}
        torch.save({"model": model.state_dict(), "args": ARGS, "n_in": n_in, "epoch": 3, "readout": None,
                    "provenance": provenance}, run / "model_ep3.pt")
    monkeypatch.setattr(train_rnn, "ROOT", tmp_path)
    split = SimpleNamespace(sha256="abc")
    args = SimpleNamespace(ood_teachers="t0@3,t1@ep3", ood_aug="splice,warp", ood_sources=4, chunk=40, seed=2,
                           ood_quantile_table="", ood_mode="mix", ood_frac=0.5, ood_weight="teacher",
                           ood_scale=1.0, **ARGS)
    prov = {"evaluation_row_groups": [0, 1, 2], "fit_row_groups": {"train": [1, 2, 3], "valid": [10]}}
    teachers, smp = train_rnn.build_ood(args, split, prov, CPU, False, torch.float16)
    expected = np.mean([train_rnn.predict(m, x, 150, CPU) for m in models], axis=0)
    assert np.allclose(teachers(train_rnn.build_inputs(x)).numpy(), expected, atol=1e-5)
    assert prov["ood"]["teachers"][0]["checkpoint_sha256"] and prov["ood"]["fit_sequences"] == 4
    ck = torch.load(tmp_path / "artifacts/runs/t0/model_ep3.pt", map_location="cpu", weights_only=False)
    assert np.array_equal(train_rnn.predict(predict_run.load_model(ck, CPU), x, 40, CPU),
                          train_rnn.predict(models[0], x, 40, CPU))
    assert ("valid", 0) not in smp.fit_ids and set(smp.fit_ids) == {("train", 1), ("train", 2), ("train", 3), ("valid", 10)}
    # ancestry: another fold manifest, an evaluation sequence in the teacher fit, or differing inputs are refused
    with pytest.raises(ValueError):
        train_rnn.build_ood(args, SimpleNamespace(sha256="zzz"), dict(prov), CPU, False, torch.float16)
    with pytest.raises(ValueError):
        train_rnn.build_ood(args, split, {**prov, "evaluation_row_groups": [10]}, CPU, False, torch.float16)
    with pytest.raises(ValueError):
        ood.check_teacher_compat({"online_norm": True}, ARGS, "x")


# --- loss ---------------------------------------------------------------------------------------------------------

def test_ood_loss_weights():
    out = torch.tensor([[[1.0, -1.0], [0.0, 0.0], [5.0, 5.0]]])
    teacher = torch.tensor([[[0.5, -3.0], [0.0, 0.2], [4.0, 0.0]]])
    valid = torch.tensor([[True, True, False]])
    t = teacher.clamp(-2, 2)
    w = t.abs() * valid[..., None]
    manual = (w * (out - t) ** 2).sum() / w.sum()
    assert torch.allclose(ood.ood_loss(out, teacher, valid), manual)
    uniform = ood.ood_loss(out, teacher, valid, "uniform")
    row = valid[..., None].float().expand(1, 3, 2)
    assert torch.allclose(uniform, (row * (out - t) ** 2).sum() / row.sum())
    # rows outside the mask (warm-up, padding) contribute nothing
    changed = out.clone()
    changed[0, 2] = 100.0
    assert torch.equal(ood.ood_loss(changed, teacher, valid), ood.ood_loss(out, teacher, valid))


def run_batch(model, teachers, x, kept, step, scale=1.0, weight="teacher"):
    return ood.run_synthetic_batch(
        x, kept, teachers=teachers, build_inputs=train_rnn.build_inputs, forward=train_rnn.forward_chunk, model=model,
        windows=origin_windows(x.shape[1], 50, 50, "off", 0, 0, 0),
        autocast=lambda: torch.autocast("cpu", enabled=False), detach=train_rnn.detach_state, step=step,
        weight=weight, scale=scale)


def test_synthetic_batch_trains_student_toward_teachers_and_respects_kept_mask():
    models, x, _ = tiny_models()
    teachers = online(models, 50)
    torch.manual_seed(99)
    student = train_rnn.make_model(ARGS, train_rnn.build_inputs(x).shape[-1], 2)
    train_rnn.fit_normalization(student, [(x,)], CPU, batches=1)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-2)
    steps = []

    def step(loss):
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps.append(float(loss.detach()))

    kept = torch.tensor([150, 150])
    first = float(run_batch(student, teachers, x, kept, step))
    for _ in range(40):
        last = float(run_batch(student, teachers, x, kept, step))
    assert len(steps) == 41 * 3 and last < first * 0.7  # one optimizer step per training window
    # a student identical to a single teacher, scale 1, has zero loss; padding rows beyond `kept` carry no gradient
    single = online([models[0]], 50)
    noop = []
    assert float(run_batch(models[0], single, x, kept, noop.append)) < 1e-10
    grads = []
    shorty = torch.tensor([WARMUP + 1, WARMUP + 1])
    clone = train_rnn.make_model(ARGS, train_rnn.build_inputs(x).shape[-1], 2)
    clone.load_state_dict(student.state_dict())

    def grab(loss):
        clone.zero_grad()
        loss.backward()
        grads.append(sum(float(p.grad.abs().sum()) for p in clone.parameters() if p.grad is not None))

    run_batch(clone, teachers, x, shorty, grab)
    assert grads[1] > 0 and grads[0] == 0 == grads[2]  # only row 99 (window 2) lies in warm-up..kept


# --- flag-off parity ----------------------------------------------------------------------------------------------

def test_flag_off_is_bit_identical_to_v8_and_flags_are_registered(tmp_path):
    results = []
    for name, root in (("v8", V8_ROOT), ("v9", SOURCE_ROOT)):
        out = tmp_path / f"{name}.json"
        subprocess.run([sys.executable, str(SOURCE_ROOT / "tests" / "parity_fit.py"), str(root), str(out)],
                       check=True, env={"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin"},
                       timeout=300)
        results.append(json.loads(out.read_text()))
    assert results[0] == results[1]
    help_text = subprocess.run([sys.executable, str(SOURCE_ROOT / "scripts" / "train_rnn.py"), "--help"],
                               check=True, capture_output=True, text=True,
                               env={"PYTHONPATH": str(SOURCE_ROOT / "src"), "PATH": "/usr/bin"}).stdout
    for flag in ("--ood-teachers", "--ood-mode", "--ood-frac", "--ood-aug"):
        assert flag in help_text
    # the training loop only branches when a sampler exists; without --ood-teachers none is built
    text = (SOURCE_ROOT / "scripts" / "train_rnn.py").read_text()
    assert "ood_teachers = ood_sampler = None" in text and "if ood_sampler is not None and" in text
