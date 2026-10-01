"""Frozen synthetic proof; no competition files, labels, masks, or scores."""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import time
from pathlib import Path

import torch
from model import ARMS, ENDPOINT, EventContentModel, walsh_table
from reference import sequential
from torch import Tensor

ROOT = Path(__file__).resolve().parent


def error(a: Tensor, b: Tensor) -> float:
    return float((a.detach().double() - b.detach().double()).abs().max())


def state_error(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> list[float]:
    return [error(u, v) for u, v in zip(a, b, strict=True)]


def inputs(length: int, dtype: torch.dtype = torch.float32) -> Tensor:
    x = torch.randn(2, length, 211, dtype=dtype) * 0.2
    # Plateaus, first event, row64/128/129 and long quiet spans.
    x[..., :22] = 0
    x[..., 110] = 0
    for t in (0, 1, 63, 64, 127, 128, 129):
        if t < length:
            x[:, t, 110] = ENDPOINT
            if t > 0:
                x[:, t:, :22] += 0.1
    return x


def objective(out: Tensor, state: tuple[Tensor, ...]) -> Tensor:
    return out.square().mean() + 0.001 * (
        state[0].square().mean() + state[1].square().mean() + state[2].square().mean()
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(3)
    started = time.perf_counter()
    report: dict = {
        "source_hashes": {
            n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
            for n in ("model.py", "reference.py", "proof.py", "preregistration.json")
        },
        "cases": [],
    }

    def record(name: str, passed: bool, **metrics: object) -> None:
        entry = {"name": name, "pass": bool(passed), **metrics}
        report["cases"].append(entry)
        print(json.dumps(entry), flush=True)

    table = walsh_table()
    rank = int(torch.linalg.matrix_rank(table))
    gram = error(table.T @ table, 128 * torch.eye(96))
    record(
        "walsh_fullrank",
        rank == 96 and gram == 0,
        rank=rank,
        gram_max_error=gram,
        aliases="n and n+128 identical; synthetic exposure only, TRAIN exposure remains independent prefit gate",
    )
    for dtype in (torch.float64, torch.float32):
        for arm in ARMS:
            torch.manual_seed(3)
            m = EventContentModel(211, 23, arm).to(dtype)
            ref = copy.deepcopy(m)
            x = inputs(133, dtype).requires_grad_()
            xr = x.detach().clone().requires_grad_()
            # Nonzero positive consistent S/Z carried state; differentiable incoming floating states.
            state = list(m.initial_state(x))
            state[0] = torch.randn_like(state[0]) * 0.05
            state[1] = torch.randn_like(state[1]) * 0.01
            state[2] = torch.ones_like(state[2]) * 0.1
            state[3] += 7
            state[4] += 2
            for j in (0, 1, 2):
                state[j].requires_grad_()
            sr = tuple(v.detach().clone().requires_grad_(v.requires_grad) for v in state)
            out, end = m(x, tuple(state))
            outr, endr = sequential(ref, xr, sr)
            objective(out, end).backward()
            objective(outr, endr).backward()
            pg = max(
                error(p.grad, r.grad) for p, r in zip(m.parameters(), ref.parameters(), strict=True)
            )
            ig = error(x.grad, xr.grad)
            sg = [error(state[j].grad, sr[j].grad) for j in (0, 1, 2)]
            se = state_error(end, endr)
            oe = error(out, outr)
            otol, gtol, stol = (
                (1e-9, 1e-8, 1e-9) if dtype == torch.float64 else (3e-5, 0.0003, 0.003)
            )
            record(
                f"parallel_reference_{arm}_{dtype}",
                oe <= otol and max(se) <= stol and max(pg, ig, *sg) <= gtol,
                output_max=oe,
                state_max=se,
                input_gradient_max=ig,
                state_gradient_max=sg,
                param_gradient_max=pg,
            )
            with torch.no_grad():
                baseline, basestate = m(x.detach())
                partitions = {}
                for chunk in (1, 17, 64, 2000):
                    carry = None
                    parts = []
                    for start in range(0, x.shape[1], chunk):
                        y, carry = m(x.detach()[:, start : start + chunk], carry)
                        parts.append(y)
                    partitions[str(chunk)] = {
                        "output": error(torch.cat(parts, 1), baseline),
                        "state": state_error(carry, basestate),
                    }
                record(
                    f"partition_{arm}_{dtype}",
                    all(
                        d["output"] <= otol and max(d["state"]) <= stol for d in partitions.values()
                    ),
                    partitions=partitions,
                )
    torch.manual_seed(3)
    m = EventContentModel(211, 23)
    x = inputs(193)
    with torch.no_grad():
        y, s = m(x)
        altered = x.clone()
        altered[:, 130:] = torch.randn_like(altered[:, 130:]) * 5
        yf, _ = m(altered)
        record(
            "future_causality",
            error(y[:, :130], yf[:, :130]) <= 3e-5,
            max_error=error(y[:, :130], yf[:, :130]),
        )
        reset, _ = m(x)
        record("reset", torch.equal(y, reset))
        a, _ = m(x[:1])
        record("batch_isolation", error(a, y[:1]) <= 3e-5, max_error=error(a, y[:1]))
        quiet = torch.zeros(1, 20000, 211)
        for arm in ARMS:
            m.arm = arm
            q, qs = m(quiet)
            finite = all(torch.isfinite(t).all().item() for t in (q, *qs))
            record(
                f"quiet20k_{arm}",
                finite
                and torch.count_nonzero(qs[1][:, 2:]) == 0
                and torch.count_nonzero(qs[2][:, 2:]) == 0,
                event_s_max=float(qs[1][:, 2:].abs().max()),
                event_z_max=float(qs[2][:, 2:].abs().max()),
                row_z_max=float(qs[2][:, :2].max()),
            )
        m.arm = "endpoint-content"
        noevent = torch.zeros(1, 4, 211)
        _, s0 = m(noevent[:, :3])
        first = noevent[:, 3:]
        first[..., 110] = ENDPOINT
        _, sf = m(first, s0)
        record(
            "first_actual_write",
            sf[4].item() == 1 and sf[2][:, 2:].min() > 0 and s0[2][:, 2:].max() == 0,
            writecount=int(sf[4].item()),
        )
        old = copy.deepcopy(m)
        m.query.weight.mul_(1e8)
        m.key.weight.mul_(1e8)
        extreme, es = m(x)
        record("extreme_finite_keys", all(torch.isfinite(t).all().item() for t in (extreme, *es)))
        m = old
        # Constant +/-bounded values: retrieval must remain in convex hull [-1,1].
        m.value.weight.zero_()
        m.value.bias.fill_(3)
        constant = torch.tanh(torch.tensor(3.0))
        _, cs = m(x)
        ratios = cs[1] / cs[2][..., None].clamp_min(1e-6)
        record(
            "convex_hull_constant_values",
            ratios.min() >= 0 and ratios.max() <= constant + 1e-6,
            minimum=float(ratios.min()),
            maximum=float(ratios.max()),
            constant=float(constant),
        )
        m = old
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ay, as_ = m(x)
        record(
            "outer_autocast_fp32_state",
            as_[1].dtype == torch.float32
            and as_[2].dtype == torch.float32
            and torch.isfinite(ay).all(),
            state_dtypes=[str(t.dtype) for t in as_],
        )
    # Fixed stronger longcases added before v4 checks, preserving original tolerances.
    with torch.no_grad():
        for arm in ARMS:
            for policy in ("quiet_after_one_event", "event_each_row"):
                torch.manual_seed(534)
                longmodel = EventContentModel(211, 23, arm)
                longx = torch.zeros(1, 20000, 211)
                if policy == "quiet_after_one_event":
                    longx[:, 1, 110] = ENDPOINT
                    longx[:, 1:, :22] = 0.2
                else:
                    longx[:, 1:, 110] = ENDPOINT
                    longx[..., :22] = (torch.arange(20000) % 2)[None, :, None] * 0.2
                py, ps = longmodel(longx)
                sy, ss = sequential(longmodel, longx)
                oe = error(py, sy)
                se = state_error(ps, ss)
                record(
                    f"long20k_{arm}_{policy}",
                    oe <= 3e-5 and max(se) <= 0.003,
                    output_max=oe,
                    state_max=se,
                    writecount=int(ps[4].item()),
                )
    for arm in ARMS:
        torch.manual_seed(3)
        g = EventContentModel(211, 23, arm)
        xx = torch.randn(2, 256, 211) * 0.2
        xx[..., 110] = ENDPOINT
        yy, ss = g(xx)
        objective(yy, ss).backward()
        qmin = float(g.query.weight.grad.abs().sum(0).min())
        kmin = float(g.key.weight.grad.abs().sum(0).min())
        perhead = [
            float(g.query.weight.grad[j * 8 : (j + 1) * 8].abs().sum(0).min()) for j in range(4)
        ]
        kper = [float(g.key.weight.grad[j * 8 : (j + 1) * 8].abs().sum(0).min()) for j in range(4)]
        other = {
            n: float(p.grad.abs().sum())
            for n, p in g.named_parameters()
            if n.startswith(("value", "read_gate", "head"))
        }
        record(
            f"active_gradients_{arm}",
            min(qmin, kmin, *perhead, *kper) > 1e-12 and min(other.values()) > 1e-12,
            query_column_min=qmin,
            key_column_min=kmin,
            query_head_column_min=perhead,
            key_head_column_min=kper,
            other_gradient_l1=other,
        )
    torch.manual_seed(3)
    cost = EventContentModel(211, 23)
    bx = inputs(2000)
    tick = time.perf_counter()
    by, bs = cost(bx)
    forward_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    objective(by, bs).backward()
    backward_seconds = time.perf_counter() - tick
    report.update(
        parameters=sum(p.numel() for p in cost.parameters()),
        state_bytes=sum(t.numel() * t.element_size() for t in cost.initial_state(bx)) // 2,
        cpu_batch2_chunk2000_forward_seconds=forward_seconds,
        cpu_backward_seconds=backward_seconds,
        duration_seconds=time.perf_counter() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        cuda_initialized=torch.cuda.is_initialized(),
        all_pass=all(c["pass"] for c in report["cases"]),
    )
    (ROOT / "results_v4.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    if not report["all_pass"] or report["cuda_initialized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
