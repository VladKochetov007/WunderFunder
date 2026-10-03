"""Independent adversaries using actual tiny RNN/Muon and source update block."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import logging
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
sys.path.insert(0, str(ROOT / "src"))
from wnn.nn import RNNModel, weighted_mse_loss
from wnn.optim import MuonAdamW, muon_param_groups
from wnn.run_identity import EpochUpdateCounter

LOGGER = logging.getLogger(__name__)
TRAINER = ROOT / "scripts/train_rnn.py"
TEXT = TRAINER.read_text()
TREE = ast.parse(TEXT)
EMA_NODE = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "FastEMA"
)
EMA_ENV = {"torch": torch}
exec(  # noqa: S102 -- trusted source AST
    compile(ast.Module(body=[EMA_NODE], type_ignores=[]), str(TRAINER), "exec"), EMA_ENV
)
FastEMA = EMA_ENV["FastEMA"]


def call_name(node: ast.AST) -> str:
    """Extract call syntax for selecting the original update block."""
    return (
        ast.unparse(node.value.func)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        else ""
    )


def original_update_block() -> tuple[object, list[int]]:
    """Compile exact trainer statements, not a reimplemented ordering oracle."""
    for node in ast.walk(TREE):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        starts = [
            i for i, item in enumerate(body) if call_name(item) == "opt.zero_grad"
        ]
        for start in starts:
            end = next(
                (
                    i
                    for i in range(start + 1, len(body))
                    if isinstance(body[i], ast.If)
                    and ast.unparse(body[i].test) == "duty is not None"
                ),
                None,
            )
            if end is not None:
                selected = body[start:end]
                return compile(
                    ast.Module(body=selected, type_ignores=[]), str(TRAINER), "exec"
                ), [selected[0].lineno, selected[-1].end_lineno]
    raise RuntimeError("Original trainer update boundary not found")


UPDATE, UPDATE_LINES = original_update_block()


class NullTimer:
    """Ignore only profiling bookkeeping."""

    def mark(self, _name: str) -> None:
        pass


def bundle() -> tuple:
    """Actual source implementations, small synthetic dimensions, CPU FP32."""
    raw = RNNModel(3, hidden=4, layers=2, dropout=0.35, n_out=2).train()
    ema = FastEMA(raw, 0.7)
    opt = MuonAdamW(muon_param_groups(raw), lr=0.025)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=2)
    return raw, ema, opt, sched, scaler


def source_step(raw, ema, opt, sched, scaler, loss: torch.Tensor, count: int) -> dict:
    """Execute unchanged actual backward/unscale/clip/scaler/scheduler/EMA order."""
    counter = EpochUpdateCounter(torch.device("cpu"))
    env = {
        "torch": torch,
        "model": raw,
        "ema_model": ema,
        "opt": opt,
        "sched": sched,
        "scaler": scaler,
        "loss": loss,
        "epoch_counter": counter,
        "grad_stats": None,
        "args": SimpleNamespace(clip=0.4, parity_dump=False),
        "timer": NullTimer(),
        "step_count": count,
        "swa_model": None,
        "swa_start": 0,
    }
    exec(UPDATE, env)  # noqa: S102 -- exact trusted trainer statements
    return {"counter": counter.finish({})["epoch"], "step_count": env["step_count"]}


def names(raw, opt) -> list[list[str]]:
    by_id = {id(p): name for name, p in raw.named_parameters()}
    return [[by_id[id(p)] for p in group["params"]] for group in opt.param_groups]


def capture() -> dict:
    torch.manual_seed(1907)
    raw, ema, opt, sched, scaler = bundle()
    h = he = None
    for step in range(4):
        x = torch.randn(2, 5, 3)
        pred, h = raw(x, h)
        h = h.detach()
        with torch.no_grad():
            target, he = ema.module(x, he)
            he = he.detach()
        loss = pred.square().mean() + (pred - target).square().mean()
        source_step(raw, ema, opt, sched, scaler, loss, step)
    # Generating the exact chunk happens BEFORE the pre-forward RNG capture.
    x, y = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
    weight = torch.linspace(0.2, 1, 10).reshape(2, 5)
    state = copy.deepcopy(
        {
            "raw": raw.state_dict(),
            "ema": ema.module.state_dict(),
            "ema_n": ema.n,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "h": h,
            "he": he,
            "x": x,
            "y": y,
            "weight": weight,
            "rng": torch.get_rng_state(),
            "count": 4,
        }
    )
    state["group_names"] = names(raw, opt)
    state["param_id_names"] = {
        pid: name
        for group, ns in zip(
            state["opt"]["param_groups"], state["group_names"], strict=True
        )
        for pid, name in zip(group["params"], ns, strict=True)
    }
    return state


def replay(state: dict, *, skip: bool = False, fresh_teacher: bool = False) -> dict:
    """Restore full local snapshot independently and measure one update."""
    raw, ema, opt, sched, scaler = bundle()
    if names(raw, opt) != state["group_names"]:
        raise ValueError("Named optimizer groups differ")
    for group, ns in zip(
        state["opt"]["param_groups"], state["group_names"], strict=True
    ):
        if [state["param_id_names"][pid] for pid in group["params"]] != ns:
            raise ValueError("Optimizer ID/name order differs")
    raw.load_state_dict(state["raw"])
    ema.module.load_state_dict(state["ema"])
    ema.n = state["ema_n"]
    opt.load_state_dict(copy.deepcopy(state["opt"]))
    sched.load_state_dict(copy.deepcopy(state["sched"]))
    scaler.load_state_dict(copy.deepcopy(state["scaler"]))
    torch.set_rng_state(state["rng"])
    pred, h = raw(state["x"], state["h"])
    post_forward_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        teacher, he = ema.module(state["x"], state["he"])
    main = weighted_mse_loss(pred, state["y"], state["weight"])
    consistency = (
        state["weight"][..., None] * (pred.float() - teacher.float()).square()
    ).sum() / state["weight"].sum()
    loss = main + consistency
    if skip:
        loss = loss * torch.tensor(float("inf"))
    incoming_teacher = teacher.detach().clone()
    clocks = source_step(raw, ema, opt, sched, scaler, loss, state["count"])
    if fresh_teacher:
        with torch.no_grad():
            teacher, _ = ema.module(state["x"], state["he"])
    return copy.deepcopy(
        {
            "pred": pred.detach(),
            "teacher": teacher.detach(),
            "frozen_teacher": incoming_teacher,
            "teacher_detached": not teacher.requires_grad,
            "h": h.detach(),
            "he": he.detach(),
            "main": main.detach(),
            "consistency": consistency.detach(),
            "raw": raw.state_dict(),
            "ema": ema.module.state_dict(),
            "ema_n": ema.n,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "gradients": {
                name: p.grad.detach().clone() for name, p in raw.named_parameters()
            },
            "clocks": clocks,
            "post_forward_rng": post_forward_rng,
        }
    )


def differences(left: object, right: object, path: str = "") -> list[str]:
    """Require exact deterministic CPU identity, including state layout."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        identical = (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        )
        return [] if identical else [path]
    if isinstance(left, dict) and isinstance(right, dict):
        if list(left) != list(right):
            return [path + "/keys"]
        return [
            p
            for key in left
            for p in differences(left[key], right[key], f"{path}/{key}")
        ]
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return [path + "/length"]
        return [
            p
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for p in differences(a, b, f"{path}/{index}")
        ]
    return [] if left == right else [path]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    state = capture()
    torch.save(state, OWN / "synthetic_capture.pt")
    baseline = replay(state)
    checks = {}
    checks["five_pristine_repeats_bitwise"] = all(
        not differences(baseline, replay(state)) for _ in range(5)
    )
    memory = io.BytesIO()
    torch.save(state, memory)
    memory.seek(0)
    restored = torch.load(memory, weights_only=False)
    checks["serialized_state_bitwise"] = not differences(baseline, replay(restored))
    adversaries = {}
    for tag in (
        "wrong_raw_hidden",
        "wrong_ema_hidden",
        "swapped_hidden",
        "alter_muon_moment",
        "alter_adam_moment",
        "post_forward_rng",
    ):
        corrupt = copy.deepcopy(state)
        if tag == "wrong_raw_hidden":
            corrupt["h"] = torch.zeros_like(state["h"])
        elif tag == "wrong_ema_hidden":
            corrupt["he"] = torch.zeros_like(state["he"])
        elif tag == "swapped_hidden":
            corrupt["h"], corrupt["he"] = state["he"].clone(), state["h"].clone()
        elif tag.startswith("alter_"):
            moment = "momentum_buffer" if "muon" in tag else "exp_avg"
            parameter = next(
                p for p, values in corrupt["opt"]["state"].items() if moment in values
            )
            corrupt["opt"]["state"][parameter][moment].add_(0.3)
        else:
            corrupt["rng"] = baseline["post_forward_rng"]
        observed = replay(corrupt)
        mismatch = differences(baseline, observed)
        adversaries[tag] = {
            "rejected": bool(mismatch),
            "different_fields": mismatch,
            "aggregate_clocks_identical": not differences(
                baseline["clocks"], observed["clocks"]
            ),
            "gradient_identical": not differences(
                baseline["gradients"], observed["gradients"]
            ),
        }
    wrong_order = copy.deepcopy(state)
    wrong_order["opt"]["param_groups"][1]["params"].reverse()
    try:
        replay(wrong_order)
        adversaries["optimizer_name_order"] = {"rejected": False}
    except ValueError as error:
        adversaries["optimizer_name_order"] = {"rejected": True, "reason": str(error)}
    fresh = replay(state, fresh_teacher=True)
    adversaries["updated_ema_used_as_teacher"] = {
        "rejected": bool(differences(baseline, fresh)),
        "teacher_max_error": float(
            (fresh["teacher"] - baseline["teacher"]).abs().max()
        ),
        "frozen_target_unchanged": torch.equal(
            fresh["frozen_teacher"], baseline["teacher"]
        ),
        "aggregate_clocks_identical": not differences(
            baseline["clocks"], fresh["clocks"]
        ),
    }
    skipped = replay(state, skip=True)
    counter = skipped["clocks"]["counter"]
    checks["actual_cpu_scaler_optimizer_skip"] = (
        counter["optimizer_completed_updates"] == 0
        and counter["amp_skipped_updates"] == 1
    )
    checks["skip_preserves_raw_weights_and_moments"] = not differences(
        state["raw"], skipped["raw"]
    ) and not differences(state["opt"]["state"], skipped["opt"]["state"])
    checks["skip_advances_scheduler"] = (
        skipped["sched"]["last_epoch"] == state["sched"]["last_epoch"] + 1
    )
    checks["skip_advances_ema_and_attempt_clock"] = (
        skipped["ema_n"] == state["ema_n"] + 1
        and skipped["clocks"]["step_count"] == state["count"] + 1
    )
    checks["skip_changes_lagged_ema_weights"] = bool(
        differences(state["ema"], skipped["ema"])
    )
    checks["skip_halves_scaler"] = (
        skipped["scaler"]["scale"] == state["scaler"]["scale"] / 2
    )
    normalized = copy.deepcopy(skipped)
    normalized["clocks"]["counter"]["optimizer_completed_updates"] = 1
    adversaries["normalize_skipped_clock_to_attempt"] = {
        "rejected": bool(differences(skipped, normalized))
    }
    halted_ema = copy.deepcopy(skipped)
    halted_ema["ema"], halted_ema["ema_n"], halted_ema["sched"] = (
        state["ema"],
        state["ema_n"],
        state["sched"],
    )
    adversaries["hold_ema_scheduler_on_skip"] = {
        "rejected": bool(differences(skipped, halted_ema)),
        "aggregate_clocks_identical": not differences(
            skipped["clocks"], halted_ema["clocks"]
        ),
    }
    # Equal current weights alone do not erase recurrent trajectory mismatch.
    equal = copy.deepcopy(state)
    equal["ema"] = copy.deepcopy(equal["raw"])
    equal["he"] = equal["h"].clone()
    # EMA eval vs raw train dropout intentionally differs; disable by using raw eval
    # in a direct actual-model control, leaving the replay fixture's modes intact.
    raw, ema, _, _, _ = bundle()
    raw.load_state_dict(equal["raw"])
    ema.module.load_state_dict(equal["ema"])
    raw.eval()
    with torch.no_grad():
        first, _ = raw(equal["x"], equal["h"])
        second, _ = ema.module(equal["x"], equal["he"])
        wrong, _ = ema.module(equal["x"], state["he"])
    checks["equal_weights_hidden_modes_zero_consistency"] = torch.equal(first, second)
    checks["equal_weights_alone_not_zero"] = not torch.equal(first, wrong)
    # Actual FastEMA n=0 copies parameters but DOES NOT update buffers.
    raw.mean.add_(2)
    ema.n = 0
    ema.update_parameters(raw)
    checks["fastema_first_update_copies_parameters"] = all(
        torch.equal(a, b)
        for a, b in zip(raw.parameters(), ema.module.parameters(), strict=True)
    )
    checks["fastema_does_not_refresh_buffers"] = not torch.equal(
        raw.mean, ema.module.mean
    )
    # A teacher forward must not attach gradients to EMA, even when current values match.
    for parameter in ema.module.parameters():
        parameter.requires_grad_(True)
    attached_teacher, _ = ema.module(state["x"], state["he"])
    adversaries["attached_ema_prediction_target"] = {
        "rejected": attached_teacher.requires_grad,
        "reason": "Target tensor has autograd graph rather than frozen no_grad forward",
    }
    # Verify byte comparator sensitivity beyond numerical equality.
    checks["bitwise_comparator_detects_signed_zero"] = bool(
        differences(torch.tensor(0.0), torch.tensor(-0.0))
    )
    results = {
        "checks": checks,
        "adversaries": adversaries,
        "source_update_lines": UPDATE_LINES,
        "backend": "CPU FP32 actual RNNModel/MuonAdamW/FastEMA, CPU GradScaler enabled; synthetic inputs",
        "excluded": "CUDA fp16/cuDNN actual combination, complete combo objective, live capture; no tierB admission",
        "skip_clocks": skipped["clocks"],
        "prereg_sha256": hashlib.sha256((OWN / "TESTS.json").read_bytes()).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                TRAINER,
                ROOT / "src/wnn/nn.py",
                ROOT / "src/wnn/optim.py",
                ROOT / "src/wnn/run_identity.py",
            )
        },
        "fixture_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "torch_version": torch.__version__,
        "cost": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    results["passed"] = all(checks.values()) and all(
        a["rejected"] for a in adversaries.values()
    )
    (OWN / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    state_report = json.loads((OWN / "STATE.json").read_text())
    state_report.update(
        phase="fixtures_complete",
        passed=results["passed"],
        cost=results["cost"],
        pending=["freeze source findings and report"],
    )
    (OWN / "STATE.json").write_text(json.dumps(state_report, indent=2) + "\n")
    LOGGER.info("RESULTS %s", json.dumps(results))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "failure.log").write_text(failure)
        raise
