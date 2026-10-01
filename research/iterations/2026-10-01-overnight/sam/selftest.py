"""Synthetic tests only. No competition dataset, checkpoint, fit or WP readout."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from sam_step import sam_backward, skip_evaluation_rng
from wnn.nn import RNNModel


def load_trainer():
    spec = importlib.util.spec_from_file_location("sam_trainer_test", HERE / "train_sam.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_recipe(module, argv):
    tree = ast.parse((HERE / "train_sam.py").read_text())
    func = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main")
    end = next(i for i, x in enumerate(func.body) if isinstance(x, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "args" for t in x.targets))
    func.body = func.body[:end + 1] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    func.name = "parse_only"
    code = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    ns = dict(module.__dict__)
    exec(compile(code, "synthetic_parser", "exec"), ns)
    old = sys.argv
    try:
        sys.argv = ["synthetic_parser", *argv]
        return ns["parse_only"]()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    opts = ap.parse_args()
    device = torch.device(opts.device)
    torch.set_num_threads(2)
    torch.manual_seed(93571)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    module = load_trainer()
    checks = []

    # Bind the materialized source exactly to the declared transform.
    receipt = json.loads((HERE / "trainer-source.json").read_text())
    src = (ROOT / receipt["base"]).read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == receipt["base_sha256"]
    for before, after in receipt["changes"]:
        assert src.count(before) == 1
        src = src.replace(before, after)
    assert src == (HERE / "train_sam.py").read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == receipt["implementation_sha256"]
    checks.append("exact_declared_source_transformation")

    # Source inspection is exact for the original complete loss expression.
    base = (ROOT / receipt["base"]).read_text()
    start = base.index('                if moments is not None:\n                    ce = F.cross_entropy',
                       base.index('            for s in range(0, inputs.shape[1], args.chunk):'))
    end = base.index('                timer.mark("loss")', start)
    loss_block = base[start:end]
    assert ''.join('    '+x if x.strip() else x for x in loss_block.splitlines(True)) in src
    checks.append("full_original_chunk_loss_preserved_verbatim")

    # Three chunks, AdamW state, clipped gradients, EMA and carried recurrence.
    original = RNNModel(5, hidden=8, layers=2, n_out=3).to(device)
    state0 = torch.randn(2, 3, 8, device=device)
    xs = [torch.randn(3, 11, 5, device=device) for _ in range(3)]
    ys = [torch.randn(3, 11, 3, device=device) for _ in range(3)]

    def run(two_pass):
        m = copy.deepcopy(original)
        ema = copy.deepcopy(original)
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01)
        scaler = torch.amp.GradScaler(device.type, init_scale=16., enabled=True)
        h = state0.clone()
        for x, y in zip(xs, ys):
            h_in = h
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out, h_plain = m(x, h_in)
            with torch.no_grad():
                teacher = ema(x, h_in)[0]
            objective = lambda z: (z.float()-y).square().mean() + (z.float()-teacher.float()).square().mean()
            loss = objective(out)
            opt.zero_grad(set_to_none=True)
            calls = []
            def closure():
                calls.append(h_in.detach().clone())
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    z, _ = m(x, h_in)
                return objective(z)
            sam_backward(loss, closure, m.parameters(), scaler, 0., force_two_pass=two_pass)
            assert len(calls) == int(two_pass)
            if calls:
                assert torch.equal(calls[0], h_in)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                for p, q in zip(ema.parameters(), m.parameters()):
                    p.lerp_(q, .1)
            h = h_plain.detach()
        return m, ema, h, opt.state_dict(), scaler.state_dict()

    baseline = run(False)
    paired = run(True)
    for left, right in zip(baseline[:2], paired[:2]):
        assert all(torch.equal(v, right.state_dict()[k]) for k,v in left.state_dict().items())
    assert torch.equal(baseline[2], paired[2])
    assert baseline[4] == paired[4]
    for key, val in baseline[3]["state"].items():
        for k, v in val.items():
            assert torch.equal(v, paired[3]["state"][key][k])
    checks.append("rho_zero_three_chunk_exact_weights_hidden_AdamW_EMA_scaler_parity")

    # Nonzero rho: analytic independent gradient and exact restoration.
    p = torch.nn.Parameter(torch.tensor([.5, -1., 2.], device=device))
    old = p.detach().clone()
    disabled = torch.amp.GradScaler(device.type, enabled=False)
    target = torch.tensor([1., 1., -1.], device=device)
    g = 2*(old-target)
    expected_e = .05*g/g.double().norm().to(g.dtype)
    seen = []
    def quadratic():
        seen.append(p.detach().clone())
        return (p-target).square().sum()
    sam_backward((p-target).square().sum(), quadratic, [p], disabled, .05)
    assert torch.equal(p, old)
    assert torch.allclose(seen[0], old+expected_e, atol=1e-7, rtol=0)
    assert torch.allclose(p.grad, 2*(old+expected_e-target), atol=1e-7, rtol=0)
    checks.append("rho005_analytic_adversarial_gradient_and_exact_restore")

    p.grad = None
    def fail():
        raise RuntimeError("planted second-pass failure")
    try:
        sam_backward((p-target).square().sum(), fail, [p], disabled, .05)
    except RuntimeError as error:
        assert "planted" in str(error)
    else:
        raise AssertionError("exception must propagate")
    assert torch.equal(p, old)
    checks.append("second_pass_exception_restores_parameters_exactly")

    # Enabled scaling must skip the entire first-overflow update, with no closure.
    q = torch.nn.Parameter(old.clone())
    opt = torch.optim.AdamW([q], lr=.01, weight_decay=.01)
    scaler = torch.amp.GradScaler(device.type, init_scale=16., enabled=True)
    calls = []
    result = sam_backward((q*float("inf")).sum(), lambda: calls.append(1), [q], scaler, .05)
    assert result["first_overflow"] and not calls
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([q], 1.)
    scaler.step(opt)
    scaler.update()
    assert torch.equal(q, old) and not opt.state and scaler.get_scale() == 8.
    checks.append("first_overflow_no_perturbation_AdamW_skip_scale_backoff")

    # Nonfinite second loss also restores then delegates skip to GradScaler.
    q.grad = None
    result = sam_backward(q.square().sum(), lambda: (q*float("inf")).sum(), [q], scaler, .05)
    assert result["two_pass"] and torch.equal(q, old)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([q], 1.)
    scaler.step(opt)
    scaler.update()
    assert torch.equal(q, old) and not opt.state and scaler.get_scale() == 4.
    checks.append("second_overflow_exact_restore_AdamW_skip_scale_backoff")

    # Sequential evaluation advances exactly one CPU base-seed, preserving epochs.
    for workers in [0, 2]:
        def order(skipped):
            torch.manual_seed(182)
            train = DataLoader(TensorDataset(torch.arange(41)), batch_size=8, shuffle=True, num_workers=0)
            valid = DataLoader(TensorDataset(torch.zeros(19, 2)), batch_size=16, num_workers=workers)
            ans = []
            for _ in range(4):
                ans.append(torch.cat([b[0] for b in train]))
                if skipped:
                    skip_evaluation_rng(valid)
                else:
                    list(valid)
            return ans, torch.get_rng_state()
        actual, actual_rng = order(False)
        skipped, skipped_rng = order(True)
        assert all(torch.equal(a,b) for a,b in zip(actual, skipped))
        assert torch.equal(actual_rng, skipped_rng)
    checks.append("four_epoch_sampler_order_eval_RNG_parity_workers0_and2")

    # Check every immutable execution command without entering main/data loaders.
    for path in sorted(HERE.glob("execution-plan*.json")):
        plan = json.loads(path.read_text())
        for job in plan["jobs"]:
            argv = job["train_argv"]
            args = parse_recipe(module, argv[2:])
            module.validate_sam_recipe(args, argparse.ArgumentParser())
    checks.append("all_registered_training_argv_pass_strict_recipe_guard")
    print(json.dumps({"status":"PASS", "device":str(device), "torch":torch.__version__,
                      "checks":checks, "trainer_sha256":receipt["implementation_sha256"],
                      "sam_step_sha256":hashlib.sha256((HERE/"sam_step.py").read_bytes()).hexdigest(),
                      "competition_data_reads":0, "forecasting_fits":0, "WP_readouts":0}, indent=2))


if __name__ == "__main__":
    main()
