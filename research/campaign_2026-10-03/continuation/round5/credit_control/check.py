"""Source-bound tiny common-macrostep credit controls; never a training fit."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from wnn.nn import DualClockModel, RNNModel  # noqa: E402
from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402
from wnn.outcomes import N_SP_AUX  # noqa: E402
from wnn.spread_aux import structural_loss  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts/train_rnn.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_source() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Compile original trainer functions and four selected original loss bodies."""
    tree = ast.parse(SOURCE.read_text())
    names = {"FastEMA", "next_label_change", "next_change_labels", "forward_chunk", "detach_state"}
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert len(nodes) == len(names)
    ns = {
        "torch": torch,
        "F": F,
        "DualClockModel": DualClockModel,
        "structural_loss": structural_loss,
        "N_SP_AUX": N_SP_AUX,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    clauses = {}
    for key, test, marker in [
        ("main", "teach is not None", "yc ="),
        ("struct", "args.aux_struct > 0", "structural_loss("),
        ("next", "args.aux_next > 0", "wv = nw"),
        ("ema", "args.ema_teacher > 0", "consistency_scale ="),
    ]:
        found = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == test
            and marker in ast.unparse(ast.Module(body=n.body, type_ignores=[]))
        ]
        assert len(found) == 1, (key, len(found))
        clauses[key] = found[0]
    body = sum([clauses[k].body for k in ("main", "struct", "next", "ema")], [])
    code = compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec")
    identities = {
        k: {"line": n.lineno, "ast_sha256": hashlib.sha256(ast.dump(n).encode()).hexdigest()}
        for k, n in clauses.items()
    }
    return ns, code, identities


def fixture(ns: dict[str, Any]) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261003)
    x = torch.randn(1, 20, 3, generator=generator)
    # Pair change at 14 preserves D; difference changes at 9 and 17.
    y = torch.tensor(
        [[[3.0, -0.5]] * 9 + [[0.25, 1.5]] * 5 + [[0.75, 2.0]] * 3 + [[-1.0, 0.5]] * 3]
    )
    teach = 3 * torch.randn(1, 20, 2, generator=generator)
    valid = torch.ones(1, 20, dtype=torch.bool)
    nx_rows, nx_up, nx_ok = ns["next_change_labels"](y, valid)
    return {
        "x": x,
        "y": y,
        "teach": teach.detach(),
        "main_valid": torch.linspace(0.2, 1.8, 20)[None],
        "nxt_label": ns["next_label_change"](y),
        "nx_rows": nx_rows,
        "nx_up": nx_up,
        "nx_ok": nx_ok,
        "sp_lab": torch.arange(20)[None] % N_SP_AUX,
        "br_lab": torch.arange(20)[None] % 3,
        "cl_lab": torch.arange(20)[None] % 4,
    }


def bound_loss(
    ns: dict[str, Any],
    code: Any,
    data: dict[str, torch.Tensor],
    out: torch.Tensor,
    ema_out: torch.Tensor,
    horizon: int,
    mask: torch.Tensor,
) -> dict[str, Any]:
    v = data["main_valid"] * mask
    env = dict(ns, **data)
    env.update(
        out=out,
        out_ema=ema_out.detach(),
        main_valid=v,
        sw=v,
        nw=v * data["nx_ok"],
        sl=slice(0, 20),
        struct_at=2,
        next_at=21,
        spread_encoded=None,
        kth=[],
        args=SimpleNamespace(
            distill_alpha_far=None,
            distill_alpha=0.5,
            distill_boot_h=horizon,
            aux_struct=0.3,
            aux_next=0.3,
            ema_teacher=1.0,
            ema_metric_weight=False,
        ),
    )
    exec(code, env)
    return env


def independent_oracle(
    data: dict[str, torch.Tensor],
    out: torch.Tensor,
    ema_out: torch.Tensor,
    horizon: int,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form output coefficients; explicit loops independently reconstruct targets."""
    y, teacher = data["y"], data["teach"].clamp(-2, 2)
    clipped = y.clamp(-2, 2)
    label = clipped.clone()
    for t in range(20):
        change = next((j for j in range(t + 1, 20) if not torch.equal(y[0, j], y[0, j - 1])), 20)
        if horizon and t + horizon < 20 and change > t + horizon:
            label[0, t] = teacher[0, t + horizon]
    target = 0.5 * label + 0.5 * teacher
    v = data["main_valid"] * mask
    w = v[..., None] * clipped.abs()
    derivative = torch.zeros_like(out)
    derivative[..., :2] = 2 * w / w.sum() * (out[..., :2] - target)
    derivative[..., :2] += 2 * v[..., None] / v.sum() * (out[..., :2] - ema_out[..., :2])
    for start, width, labels in [
        (2, N_SP_AUX, data["sp_lab"]),
        (14, 3, data["br_lab"]),
        (17, 4, data["cl_lab"]),
    ]:
        probs = out[..., start : start + width].softmax(-1)
        onehot = F.one_hot(labels, width)
        derivative[..., start : start + width] = 0.3 * v[..., None] / v.sum() * (probs - onehot)
    nw = v * data["nx_ok"]
    derivative[..., 21] = (
        0.6 * nw / nw.sum().clamp_min(1e-8) * (out[..., 21] - (data["nx_rows"] - 3) / 2)
    )
    derivative[..., 22] = (
        0.3 * nw / nw.sum().clamp_min(1e-8) * (out[..., 22].sigmoid() - data["nx_up"])
    )
    return derivative.detach(), target


def forward(
    ns: dict[str, Any], model: RNNModel, x: torch.Tensor, h: torch.Tensor, detach: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, mid = ns["forward_chunk"](model, x, slice(0, 10), h)
    if mid.requires_grad:
        mid.retain_grad()
    b, end = ns["forward_chunk"](
        model, x, slice(10, 20), ns["detach_state"](mid) if detach else mid
    )
    return torch.cat([a, b], 1), mid, end


def make_system(ns: dict[str, Any]) -> tuple[Any, ...]:
    model = RNNModel(3, hidden=6, layers=2, n_out=23)
    opt = MuonAdamW(muon_param_groups(model, "rnn"), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.001, total_steps=40, pct_start=0.05, final_div_factor=100
    )
    ema = ns["FastEMA"](model, 0.999)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, opt, sched, ema, scaler


def step(system: tuple[Any, ...], loss: torch.Tensor) -> float:
    model, opt, sched, ema, scaler = system
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    ema.update_parameters(model)
    return float(norm)


def equal_tree(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main() -> None:
    started, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    torch.manual_seed(20261003)
    ns, code, clauses = bind_source()
    data = fixture(ns)
    ones = torch.ones(1, 20)
    system = make_system(ns)
    model, opt, sched, ema, scaler = system
    h = torch.zeros(2, 1, 6)
    he = h.clone()
    for _ in range(3):
        out, _, end = forward(ns, model, data["x"], h, False)
        with torch.no_grad():
            eo, _, ee = forward(ns, ema.module, data["x"], he, False)
        step(system, bound_loss(ns, code, data, out, eo, 0, ones)["loss"])
        h, he = end.detach(), ee.detach()
    capture = {
        "raw": copy.deepcopy(model.state_dict()),
        "ema": copy.deepcopy(ema.module.state_dict()),
        "optimizer": copy.deepcopy(opt.state_dict()),
        "scheduler": copy.deepcopy(sched.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "ema_n": ema.n,
        "raw_h": h.clone(),
        "ema_h": he.clone(),
        "data": data,
        "provenance": "synthetic warm state; not an exported scientific checkpoint",
    }
    torch.save(capture, HERE / "synthetic_capture.pt")
    restored = torch.load(HERE / "synthetic_capture.pt", map_location="cpu", weights_only=True)
    checks: dict[str, bool] = {}
    checks["capture_roundtrip"] = equal_tree(capture, restored)
    capture = restored
    checks["lagged_ema_differs_from_raw"] = not equal_tree(capture["raw"], capture["ema"])
    checks["raw_and_ema_incoming_states_distinct"] = not torch.equal(h, he)
    checks["warm_state_all_parameters"] = len(opt.state) == len(list(model.parameters()))
    records = []
    replays = []
    for horizon in (0, 8):
        for control in ("all", "second", "first"):
            mask = ones.clone()
            if control == "second":
                mask[:, :10] = 0
            if control == "first":
                mask[:, 10:] = 0
            arms = []
            for detached in (False, True):
                arm = make_system(ns)
                m, o, s, e, sc = arm
                m.load_state_dict(capture["raw"])
                e.module.load_state_dict(capture["ema"])
                e.n = capture["ema_n"]
                o.load_state_dict(copy.deepcopy(capture["optimizer"]))
                s.load_state_dict(copy.deepcopy(capture["scheduler"]))
                sc.load_state_dict(capture["scaler"])
                checks[f"h{horizon}_{control}_{detached}_restored_state"] = equal_tree(
                    o.state_dict(), capture["optimizer"]
                ) and equal_tree(s.state_dict(), capture["scheduler"])
                x = data["x"].clone().requires_grad_()
                output, mid, _ = forward(ns, m, x, h.clone(), detached)
                with torch.no_grad():
                    eo, _, _ = forward(ns, e.module, x.detach(), he.clone(), False)
                env = bound_loss(ns, code, data, output, eo, horizon, mask)
                oracle, target = independent_oracle(data, output.detach(), eo, horizon, mask)
                params = tuple(m.parameters())
                actual = torch.autograd.grad(env["loss"], params, retain_graph=True)
                expected = torch.autograd.grad(
                    output, params, grad_outputs=oracle, retain_graph=True
                )
                oracle_error = max(
                    float((a - b).abs().max()) for a, b in zip(actual, expected, strict=True)
                )
                checks[f"h{horizon}_{control}_{detached}_oracle"] = (
                    oracle_error <= 2e-6 and torch.equal(target, env["tgt"])
                )
                dx, dh = torch.autograd.grad(
                    env["loss"], (x, mid), retain_graph=True, allow_unused=True
                )
                # Clear retain_grad side effects before the actual single update.
                mid.grad = None
                clip_norm = step(arm, env["loss"])
                replays.append(
                    {
                        "horizon": horizon,
                        "control": control,
                        "detached": detached,
                        "pre_clip_gradients": dict(
                            zip((n for n, _ in m.named_parameters()), actual, strict=True)
                        ),
                        "post_clip_gradients": {n: p.grad.clone() for n, p in m.named_parameters()},
                        "post_optimizer": copy.deepcopy(o.state_dict()),
                        "post_raw": copy.deepcopy(m.state_dict()),
                        "post_ema": copy.deepcopy(e.module.state_dict()),
                        "post_scheduler": copy.deepcopy(s.state_dict()),
                        "post_scaler": copy.deepcopy(sc.state_dict()),
                        "post_ema_n": e.n,
                    }
                )
                clock = {
                    "scheduler_last_epoch": s.last_epoch,
                    "ema_n": e.n,
                    "adam_steps": [int(st["step"]) for st in o.state.values() if "step" in st],
                    "scaler_enabled": sc.is_enabled(),
                    "scaler_state": sc.state_dict(),
                    "optimizer_calls": 1,
                    "scheduler_calls": 1,
                    "ema_calls": 1,
                    "scaler_step_calls": 1,
                    "scaler_update_calls": 1,
                }
                arms.append(
                    {
                        "out": output.detach(),
                        "ema_out": eo,
                        "target": env["tgt"],
                        "loss": env["loss"].detach(),
                        "grads": actual,
                        "input_first_norm": float(dx[:, :10].norm()),
                        "mid_norm": 0.0 if dh is None else float(dh.norm()),
                        "oracle_error": oracle_error,
                        "clock": clock,
                        "clip_norm": clip_norm,
                        "post_raw": copy.deepcopy(m.state_dict()),
                        "post_optimizer": copy.deepcopy(o.state_dict()),
                    }
                )
            a, b = arms
            prefix = f"h{horizon}_{control}"
            checks[prefix + "_bitexact_forward_target_loss"] = all(
                torch.equal(a[k], b[k]) for k in ("out", "ema_out", "target", "loss")
            )
            checks[prefix + "_finite"] = all(
                torch.isfinite(arm[k]).all().item()
                for arm in arms
                for k in ("out", "ema_out", "target", "loss")
            )
            checks[prefix + "_clock"] = (
                a["clock"] == b["clock"]
                and a["clock"]["scheduler_last_epoch"] == 4
                and a["clock"]["ema_n"] == 4
                and set(a["clock"]["adam_steps"]) == {4}
            )
            grad_delta = max(
                float((ga - gb).abs().max()) for ga, gb in zip(a["grads"], b["grads"], strict=True)
            )
            if control == "second":
                checks[prefix + "_planted_credit"] = (
                    a["input_first_norm"] > 1e-9
                    and a["mid_norm"] > 1e-9
                    and b["input_first_norm"] == 0
                    and b["mid_norm"] == 0
                    and grad_delta > 1e-9
                )
                checks[prefix + "_update_mismatch"] = not equal_tree(
                    a["post_raw"], b["post_raw"]
                ) and not equal_tree(a["post_optimizer"], b["post_optimizer"])
            if control == "first":
                checks[prefix + "_equal_gradient_null"] = all(
                    torch.equal(ga, gb) for ga, gb in zip(a["grads"], b["grads"], strict=True)
                )
                checks[prefix + "_equal_update_null"] = equal_tree(
                    a["post_raw"], b["post_raw"]
                ) and equal_tree(a["post_optimizer"], b["post_optimizer"])
            if control == "all":
                left = ones.clone()
                left[:, 10:] = 0
                right = 1 - left
                wrong = 0.5 * (
                    bound_loss(ns, code, data, a["out"], a["ema_out"], horizon, left)["loss"]
                    + bound_loss(ns, code, data, a["out"], a["ema_out"], horizon, right)["loss"]
                )
                checks[prefix + "_reject_mean_normalized_halves"] = (
                    abs(float(wrong - a["loss"])) > 1e-5
                )
            records.append(
                {
                    "horizon": horizon,
                    "control": control,
                    "loss": float(a["loss"]),
                    "gradient_max_difference": grad_delta,
                    "arms": [
                        {
                            k: arm[k]
                            for k in (
                                "input_first_norm",
                                "mid_norm",
                                "oracle_error",
                                "clock",
                                "clip_norm",
                            )
                        }
                        for arm in arms
                    ],
                }
            )
    # Strict pair-change and sequence-end gates are bound and independently reconstructed.
    dummy = torch.zeros(1, 20, 23)
    env = bound_loss(ns, code, data, dummy, dummy, 8, ones)
    checks["h8_strict_boundary"] = (
        bool(env["alive"][0, 0]) and not bool(env["alive"][0, 1]) and not bool(env["alive"][0, 12])
    )
    checks["pair_vs_difference_change"] = int(data["nxt_label"][0, 9]) == 14 and float(
        data["nx_rows"][0, 9]
    ) == float(torch.log1p(torch.tensor(8.0)))
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            SOURCE,
            ROOT / "src/wnn/nn.py",
            ROOT / "src/wnn/optim.py",
            ROOT / "src/wnn/spread_aux.py",
            ROOT / "src/wnn/outcomes.py",
        ]
    }
    torch.save(replays, HERE / "synthetic_replays.pt")
    names = {id(p): n for n, p in model.named_parameters()}
    group_names = [
        {
            "muon": g.get("muon", False),
            "muon_blocks": g.get("muon_blocks", 1),
            "parameters": [names[id(p)] for p in g["params"]],
        }
        for g in opt.param_groups
    ]
    result = {
        "checks": checks,
        "passed": all(checks.values()),
        "records": records,
        "source_hashes": sources,
        "loss_ast": clauses,
        "optimizer_group_names": group_names,
        "torch": torch.__version__,
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": len(checks),
                "failed": [k for k, v in checks.items() if not v],
                "cost": result["cost"],
            },
            indent=2,
        )
    )
    assert result["passed"], "control falsified; see results.json"


if __name__ == "__main__":
    main()
