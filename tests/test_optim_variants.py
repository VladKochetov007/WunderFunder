import torch

from wnn.optim_variants import AdEMAMix, CautiousAdamW


def _pair(seed: int = 0) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    torch.manual_seed(seed)
    w = torch.randn(7, 5)
    return torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())


def _loss(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((x @ p.T) ** 2).mean()


def test_cautious_first_step_equals_adamw() -> None:
    a, b = _pair()
    x = torch.randn(11, 5)
    ref = torch.optim.AdamW([a], lr=1e-2, weight_decay=0.1, foreach=False)
    cau = CautiousAdamW([b], lr=1e-2, weight_decay=0.1)
    for opt, p in ((ref, a), (cau, b)):
        opt.zero_grad()
        _loss(p, x).backward()
        opt.step()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_cautious_masks_disagreeing_coordinates() -> None:
    p = torch.nn.Parameter(torch.zeros(4))
    opt = CautiousAdamW([p], lr=1.0, weight_decay=0.0, betas=(0.9, 0.999))
    p.grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
    opt.step()
    p.grad = torch.tensor([-10.0, 1.0, 1.0, 1.0])  # momentum still positive on coord 0
    before = p.detach().clone()
    opt.step()
    assert (p.detach() - before)[1:].lt(0).all()


def test_ademamix_alpha_zero_equals_adamw() -> None:
    a, b = _pair(1)
    ref = torch.optim.AdamW([a], lr=3e-3, weight_decay=0.05, foreach=False)
    mix = AdEMAMix([b], lr=3e-3, weight_decay=0.05, alpha=0.0, beta3=0.999)
    for step in range(5):
        x = torch.randn(9, 5, generator=torch.Generator().manual_seed(step))
        for opt, p in ((ref, a), (mix, b)):
            opt.zero_grad()
            _loss(p, x).backward()
            opt.step()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_ademamix_schedule_endpoints() -> None:
    group = {"alpha": 5.0, "beta3": 0.999, "warmup": 100, "betas": (0.9, 0.999)}
    a0, b0 = AdEMAMix._schedules(group, 0)
    assert a0 == 0.0 and abs(b0 - 0.9) < 1e-12
    assert AdEMAMix._schedules(group, 100) == (5.0, 0.999)
    assert AdEMAMix._schedules(group, 50)[1] < 0.999
