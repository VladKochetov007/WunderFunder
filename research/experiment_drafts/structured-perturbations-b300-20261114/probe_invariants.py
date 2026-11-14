"""Independent tensor invariants for the server-only perturbation admission."""

from __future__ import annotations

import torch
from torch import Tensor
from wnn.training.sequence_perturbations import (
    history_drop_mask,
    perturb_history_attention,
    perturb_input,
)

SEED = 208401


def require(condition: bool, message: str) -> None:
    """Stop admission on an observed numerical contract violation."""
    if not condition:
        raise ValueError(message)


def check_input(device: torch.device) -> dict[str, object]:
    """Recover the imputed source row without reusing the implementation's mask."""
    rows = torch.arange(20_000, device=device, dtype=torch.float32)
    x = rows[None, :, None].expand(2, -1, 112).clone()
    modified = x.clone()
    modified[:, 10_000:] += 100_000
    settings = {"seed": SEED, "epoch": 2, "rank": 0, "batch_start": 128}
    outcomes = {}
    for mode in ("none", "iid", "burst"):
        before = torch.cuda.get_rng_state(device)
        out = perturb_input(x, mode, **settings)
        prefix = perturb_input(modified, mode, **settings)
        require(torch.equal(before, torch.cuda.get_rng_state(device)), "input RNG leaked")
        require(torch.equal(out[:, :10_000], prefix[:, :10_000]), "future input affected prefix")
        require(torch.equal(out[:, :, :52], x[:, :, :52]), "i0 changed")
        require(torch.equal(out[:, :, 104:], x[:, :, 104:]), "auxiliary channels changed")
        source = out[:, :, 52].long()
        position = rows.long()[None, :].expand_as(source)
        missing = source != position
        expected_count = 0 if mode == "none" else 4992
        require(bool((missing.sum(dim=1) == expected_count).all()), "wrong stale row count")
        require(bool((source <= position).all()), "future imputation")
        require(bool((source >= 0).all()), "imputation crossed sequence start")
        expected_source = torch.where(missing, 0, position).cummax(dim=1).values
        require(torch.equal(source, expected_source), "not last-observation carry")
        require(
            torch.equal(out[:, :, 52:104], source[:, :, None].expand(-1, -1, 52).float()),
            "i1 group is incoherent",
        )
        require(not bool(missing[:, 0].any() | missing[:, 19969:].any()), "protected rows changed")
        if mode == "burst":
            blocks = missing[:, 1:19969].reshape(2, 156, 128)
            require(bool((blocks == blocks[:, :, :1]).all()), "burst length contract failed")
        outcomes[mode] = {"masked_rows": expected_count, "prefix_exact": True}
    require(torch.equal(x, rows[None, :, None].expand_as(x)), "input tensor mutated")
    return outcomes


def check_history(device: torch.device) -> dict[str, object]:
    """Check cache eligibility, exact counts, ring topology and numerical attention."""
    counts = {}
    for past in (0, 16, 32, 33, 96, 1023):
        eligible = max(0, past - 32)
        for mode in ("none", "iid", "burst"):
            generator = torch.Generator(device=device).manual_seed(SEED)
            mask = history_drop_mask(3, past, mode, device, generator=generator)
            count = 0 if mode == "none" else eligible // 4
            require(mask.shape == (3, past), "history mask shape")
            require(bool((mask.sum(dim=1) == count).all()), "history mask count")
            require(not bool(mask[:, eligible:].any()), "recent cache protection failed")
            if mode == "burst" and eligible:
                ring = mask[:, :eligible]
                transitions = (ring != ring.roll(1, dims=1)).sum(dim=1)
                require(bool((transitions <= 2).all()), "not one circular block")
            counts[f"{mode}_{past}"] = count
    measurements = {}
    for mode in ("iid", "burst"):
        torch.manual_seed(SEED)
        q = torch.randn(2, 12, 64, 64, device=device, dtype=torch.bfloat16)
        k = torch.randn(2, 12, 160, 64, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        original_k, original_v = k.clone(), v.clone()
        torch.cuda.synchronize()
        rng = torch.cuda.get_rng_state(device)
        dropped = history_drop_mask(2, 96, mode, device)

        def attention(
            backend: str,
            key: Tensor,
            value: Tensor,
            *,
            saved_rng: Tensor = rng,
            query: Tensor = q,
            corruption: str = mode,
        ) -> Tensor:
            """Replay an independently inspectable mask on unchanged query tensors."""
            torch.cuda.set_rng_state(saved_rng, device)
            return perturb_history_attention(
                query, key, value, 96, 0, 128, 0.0, corruption, backend
            )

        efficient = attention("efficient", k, v)
        reference = attention("math", k, v)
        error = float((efficient.float() - reference.float()).abs().max())
        require(error <= 0.03125, "efficient/math attention disagreement")
        hidden_k, hidden_v = k.clone(), v.clone()
        hidden_k[:, :, :96] += dropped[:, None, :, None] * 5
        hidden_v[:, :, :96] -= dropped[:, None, :, None] * 5
        require(
            torch.equal(efficient, attention("efficient", hidden_k, hidden_v)),
            "chosen dropped keys remain visible",
        )
        retained_v = v.clone()
        retained_v[:, :, 95] += 5
        require(
            bool((efficient != attention("efficient", k, retained_v)).any()),
            "protected recent key has no effect",
        )
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 144:] += 2
        changed_v[:, :, 144:] -= 2
        altered = attention("efficient", changed_k, changed_v)
        require(torch.equal(efficient[:, :, :48], altered[:, :, :48]), "future key leakage")
        require(torch.equal(k, original_k) and torch.equal(v, original_v), "cache mutated")
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        value = attention("efficient", k, v)
        value.float().square().mean().backward()
        require(
            all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in (q, k, v)),
            "nonfinite or missing attention gradient",
        )
        measurements[mode] = {
            "math_max_abs": error,
            "future_prefix_exact": True,
            "dropped_key_invariance": True,
            "retained_key_sensitivity": True,
        }
    return {"counts": counts, "attention": measurements}
