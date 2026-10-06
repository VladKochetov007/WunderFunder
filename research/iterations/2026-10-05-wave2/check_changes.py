"""Causality and matched-control checks before training."""
import numpy as np
import torch
import train_rnn as t

def main():
    torch.manual_seed(7)
    x = torch.randn(2, 23, 112)
    x[:, :3, 109:112] = -5.199
    x[:, 7:12, 109:112] = -5.199
    original = x.clone()
    raw, extra = t.sentinel_features(x, "age")
    assert torch.equal(x, original)
    for end in (1, 3, 8, 11, 23):
        r, e = t.sentinel_features(x[:, :end], "age")
        torch.testing.assert_close(r, raw[:, :end])
        torch.testing.assert_close(e, extra[:, :end])
    torch.testing.assert_close(extra[0, :3, 3], torch.log1p(torch.arange(1, 4).float()))
    assert (extra[:, 3, 3:] == 0).all()
    r, e = t.sentinel_features(x, "zero")
    assert torch.equal(r, x) and (e == 0).all()
    t.RAW_ONLY = True
    t.SENTINEL_MODE = "none"
    legacy = t.build_inputs(x)
    t.SENTINEL_MODE = "zero"
    padded = t.build_inputs(x)
    torch.testing.assert_close(padded[..., :legacy.shape[-1]], legacy)
    assert padded.shape[-1] == legacy.shape[-1] + 6
    assert (padded[..., -6:] == 0).all()
    # Complement gradient equals direct residual-target regression, without clipping residual.
    o = torch.randn(2, 5, 2, requires_grad=True)
    y, b, c = torch.randn_like(o).clamp(-2,2), torch.randn_like(o), .2
    w = y.abs()
    loss = (w * ((c*o + (1-c)*b-y)/c)**2).sum()/w.sum()
    residual = (y-(1-c)*b)/c
    direct = (w*(o-residual)**2).sum()/w.sum()
    torch.testing.assert_close(torch.autograd.grad(loss,o)[0], torch.autograd.grad(direct,o)[0])
    print("PASS: causal sentinel features; unchanged padded control; complement gradient")

if __name__ == "__main__":
    main()
