from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from wnn.kd_loss import (KDConfig, kd_distill_loss, parse_channel_weights, run_ids,  # noqa: E402
                         runlevel_loss, weighted_pearson_loss)


def data(seed=0, b=2, t=40):
    g = torch.Generator().manual_seed(seed)
    out = torch.randn(b, t, 2, generator=g)
    tgt = torch.randn(b, t, 2, generator=g)
    w = torch.rand(b, t, 1, generator=g).expand(b, t, 2).contiguous() * (torch.rand(b, t, 1, generator=g) > 0.2)
    return out, tgt, w


def v9(out, tgt, w):
    return (w * (out - tgt) ** 2).sum() / w.sum().clamp_min(1e-8)


def test_default_is_bit_identical_to_v9_expression():
    out, tgt, w = data()
    assert KDConfig().is_default
    assert kd_distill_loss(out, tgt, w, KDConfig()).item().hex() == v9(out, tgt, w).item().hex()


def test_trainer_flag_off_source_matches_v9_snapshot():
    # the only difference from v9 on the default path is the dispatch `if KD_CONFIG.is_default` keeping the v9 expression
    text = (SOURCE_ROOT / "scripts/train_rnn.py").read_text()
    assert "if KD_CONFIG.is_default:\n                            loss = (w * (out[..., :2].float() - tgt) ** 2).sum()" in text


def test_channel_weight_identity_and_scaling():
    out, tgt, w = data()
    base = v9(out, tgt, w)
    # force the split path with (1+0, 1) style equality: compute with weights 1,1 through the s/d formula manually
    e = out - tgt
    es, ed = (e[..., 0] + e[..., 1]) / 2, (e[..., 0] - e[..., 1]) / 2
    manual = (2 * w.mean(-1) * (es ** 2 + ed ** 2)).sum() / w.sum()
    assert torch.allclose(manual, base, rtol=1e-5)
    split = kd_distill_loss(out, tgt, w, KDConfig(1.0 + 1e-12, 1.0))  # nonidentical floats? 1+1e-12 is a distinct double
    assert torch.allclose(split, base, rtol=1e-5)
    only_s = kd_distill_loss(out, tgt, w, KDConfig(1.0, 0.0)).item()
    only_d = kd_distill_loss(out, tgt, w, KDConfig(0.0, 1.0)).item()
    assert only_s + only_d == pytest.approx(base.item(), rel=1e-5)
    half2 = kd_distill_loss(out, tgt, w, KDConfig(0.5, 2.0)).item()
    assert half2 == pytest.approx(0.5 * only_s + 2.0 * only_d, rel=1e-5)


def test_pearson_matches_numpy_and_gradient_finite():
    out, tgt, w = data(3)
    out.requires_grad_(True)
    loss = weighted_pearson_loss(out, tgt, w)
    o, t, ww = out.detach().double().numpy(), tgt.double().numpy(), w.double().numpy()
    ref = []
    for c in range(2):
        wc = ww[..., c].ravel()
        a, b = o[..., c].ravel(), t[..., c].ravel()
        ma, mb = (wc * a).sum() / wc.sum(), (wc * b).sum() / wc.sum()
        cov = (wc * (a - ma) * (b - mb)).sum()
        ref.append(1 - cov / np.sqrt((wc * (a - ma) ** 2).sum() * (wc * (b - mb) ** 2).sum()))
    assert loss.item() == pytest.approx(float(np.mean(ref)), abs=1e-5)
    loss.backward()
    assert torch.isfinite(out.grad).all() and out.grad.abs().sum() > 0
    perfect = weighted_pearson_loss(tgt * 3 + 1, tgt, w)
    assert perfect.item() == pytest.approx(0.0, abs=1e-5)
    mix = kd_distill_loss(out.detach(), tgt, w, KDConfig(loss="mse+pearson", pearson_weight=0.5))
    assert mix.item() == pytest.approx(v9(out.detach(), tgt, w).item() + 0.5 * loss.item(), rel=1e-5)


def test_runlevel_hand_example():
    # one sequence, 6 rows; events at rows 2 and 4 (row 0 always starts a run) -> runs [0,1] [2,3] [4,5]
    events = torch.tensor([[False, False, True, False, True, False]])
    assert run_ids(events).tolist() == [[0, 0, 1, 1, 2, 2]]
    out = torch.tensor([[[1., 0], [3, 0], [2, 0], [2, 0], [5, 0], [7, 0]]])
    tgt = torch.zeros(1, 6, 2)
    w = torch.ones(1, 6, 2)
    w[0, 1] = 0  # row 1 unscored: run 0 mean of student is just row 0 -> 1
    # run means: 1, 2, 6 ; target 0 ; run weights (col0): 1, 2, 2 ; col1 all zero error
    expected = (1 * 1 ** 2 + 2 * 2 ** 2 + 2 * 6 ** 2 + 0) / (w.sum() / 1.0)
    assert runlevel_loss(out, tgt, w, events).item() == pytest.approx(float(expected), rel=1e-6)
    total = kd_distill_loss(out, tgt, w, KDConfig(runlevel_weight=2.0), events)
    assert total.item() == pytest.approx(v9(out, tgt, w).item() + 2.0 * float(expected), rel=1e-5)


def test_argparse_validation():
    assert parse_channel_weights("0.5,2") == (0.5, 2.0)
    for bad in ("1", "1,2,3", "-1,1", "nan,1", "a,b"):
        with pytest.raises(ValueError):
            parse_channel_weights(bad)
    import os
    env = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT / "src"), "OMP_NUM_THREADS": "1"}
    for flags, message in ((["--distill-channel-weights", "1,x"], "could not convert"),
                           (["--distill-loss", "pearson"], "require --distill-cache-root"),
                           (["--distill-runlevel-weight", "-1"], "nonnegative")):
        result = subprocess.run([sys.executable, str(SOURCE_ROOT / "scripts/train_rnn.py"), "--name", "t", *flags],
                                capture_output=True, text=True, timeout=120, env=env)
        assert result.returncode == 2 and message in result.stderr
