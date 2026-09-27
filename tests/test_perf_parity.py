"""Parity of the throughput optimizations against the reference code paths (CPU, no GPU needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rnn as T  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def _weights(b=6, t=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    valid = torch.arange(t)[None].expand(b, t) >= 99
    main = valid.float() * (torch.rand(b, t, generator=g) ** 2 + 0.03)
    return main, valid, g


@pytest.mark.parametrize("masked", ["none", "all", "mixed", "one_row"])
def test_true_mask_weights_branchless_matches_reference(masked):
    main, valid, g = _weights()
    scored = torch.rand(main.shape, generator=g) < 0.1
    if masked == "none":
        scored[:] = False
    elif masked == "mixed":
        scored[::2] = False
    elif masked == "one_row":
        scored[:] = False
        scored[3, 150] = True
    ref = T.true_mask_weights(main, valid, scored, 0.03, branchless=False)
    new = T.true_mask_weights(main, valid, scored, 0.03, branchless=True)
    assert torch.allclose(ref, new, rtol=1e-6, atol=0)
    if masked == "none":
        assert torch.equal(ref, main) and torch.equal(new, main)


def test_fast_ema_matches_averaged_model():
    torch.manual_seed(0)
    model = RNNModel(211, 16, 2, n_out=4)
    ref = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))
    fast = T.FastEMA(model, 0.99)
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ref.update_parameters(model)
        fast.update_parameters(model)
    for (k, a), b in zip(ref.module.named_parameters(), fast.module.parameters()):
        assert torch.equal(a, b), k


def test_loader_reads_identical_features():
    from wnn.data import FEATURE_COLUMNS, SequenceFile

    path = ROOT / "datasets" / "valid.parquet"
    if not path.exists():
        pytest.skip("dataset not present")
    sf = SequenceFile(path)
    for g in (0, 7):
        seq = sf.read(g)
        table = sf._pf.read_row_group(g, columns=list(FEATURE_COLUMNS), use_threads=False)
        ref = np.empty_like(seq.features)
        for j, name in enumerate(FEATURE_COLUMNS):
            ref[:, j] = table.column(name).to_numpy()
        assert np.array_equal(seq.features, ref, equal_nan=True)
