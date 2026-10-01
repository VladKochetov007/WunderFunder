"""CPU-only interface regression proof for private combined trainer sources."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parser(module: ModuleType, path: Path):
    main = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = []
    for statement in main.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "args" for target in statement.targets):
            break
        body.append(statement)
    body.append(ast.Return(value=ast.Name(id="ap", ctx=ast.Load())))
    main.name, main.body = "private_parser", body
    ast.fix_missing_locations(main)
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.private_parser()


def original_loss(output, labels, weights):
    o, wv, ce = output.float(), weights.reshape(-1), 0.0
    for a0, a1, lab in [(2, 14, labels[0]), (14, 17, labels[1]), (17, 21, labels[2])]:
        ce = ce + F.cross_entropy(o[..., a0:a1].reshape(-1, a1 - a0), lab.reshape(-1), reduction="none")
    return (ce * wv).sum() / wv.sum().clamp_min(1e-8)


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT / "src"))
    closure = json.loads((HERE / "source_closure.json").read_text())
    for relative, expected in closure["local_runtime_and_artifact_closure"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    base = load("private_base_trainer", HERE / "base/scripts/train_rnn.py")
    parent = torch.load(ROOT / "artifacts/runs/c20260929_since_s3_f0/model_ep3.pt", map_location="cpu", weights_only=False)
    assert parent["model"]["head.2.weight"].shape[0] == 23
    results = {}
    for label in ["ssm_spread", "all_three"]:
        own = HERE / label
        for filename in closure["outputs"][label]["modules"]:
            load("wnn." + filename[:-3], own / "src/wnn" / filename)
        trainer = load("private_" + label, own / "scripts/train_rnn.py")
        ap = parser(trainer, own / "scripts/train_rnn.py")
        defaults = ap.parse_args(["--name", "synthetic"])
        assert defaults.arch == "gru" and defaults.spread_aux_mode == "original" and defaults.spread_cache_root == ""
        legacy = []
        for arch in ["gru", "dual", "gated"]:
            args = {**parent["args"], "arch": arch}
            torch.manual_seed(3)
            old = base.make_model(args, 211, 23)
            torch.manual_seed(3)
            new = trainer.make_model(args, 211, 23)
            assert old.state_dict().keys() == new.state_dict().keys()
            assert all(torch.equal(old.state_dict()[key], new.state_dict()[key]) for key in old.state_dict())
            x = torch.randn(2, 17, 211)
            a, _ = base.forward_chunk(old, x, slice(None), None)
            b, _ = trainer.forward_chunk(new, x, slice(None), None)
            assert torch.equal(a, b)
            a.square().mean().backward()
            b.square().mean().backward()
            assert all((p.grad is None and q.grad is None) or (p.grad is not None and q.grad is not None and torch.equal(p.grad, q.grad)) for p, q in zip(old.parameters(), new.parameters(), strict=True))
            legacy.append(arch)
        actual_parent = trainer.make_model(parent["args"], parent["n_in"], 23)
        actual_parent.load_state_dict(parent["model"], strict=True)
        variants = [("structured", "decay_init", value, 2) for value in ["multi", "single"]]
        if label == "all_three":
            variants.extend(("event-content", "memory_arm", value, 1) for value in ["endpoint-content", "endpoint-position", "raw-code-content"])
        new_checks = []
        for arch, field, value, layers in variants:
            args = {**parent["args"], "arch": arch, "hidden": 96, "layers": layers, field: value, "freeze": ""}
            model = trainer.make_model(args, 211, 23)
            restored = trainer.make_model(args, 211, 23)
            restored.load_state_dict(model.state_dict(), strict=True)
            x = torch.randn(2, 34, 211)
            out, state = trainer.forward_chunk(model, x, slice(0, 17), None)
            other, other_state = trainer.forward_chunk(restored, x, slice(0, 17), None)
            assert torch.equal(out, other) and isinstance(state, tuple)
            assert all(torch.equal(a, b) for a, b in zip(state, other_state, strict=True))
            out.square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            detached = trainer.detach_state(state)
            teacher = trainer.FastEMA(restored, 0.999)
            teacher.update_parameters(restored)
            with torch.no_grad():
                continuation, _ = trainer.forward_chunk(teacher.module, x, slice(17, 34), detached)
            assert continuation.shape == (2, 17, 23)
            child = ap.parse_args(["--name", "synthetic"])
            trainer.inherit_init_arguments(child, {"args": args}, [])
            assert getattr(child, field) == value and child.hidden == 96 and child.layers == layers
            new_checks.append({"arch": arch, field: value, "flat_state_tensors": len(state), "strict_load_forward_backward_ema_inheritance": True})
        loss_checks = []
        for zero in [False, True]:
            torch.manual_seed(3)
            logits = torch.randn(2, 17, 23)
            labels = [torch.randint(0, count, (2, 17)) for count in [12, 3, 4]]
            weights = torch.zeros(2, 17) if zero else torch.rand(2, 17)
            a, b = logits.clone().requires_grad_(), logits.clone().requires_grad_()
            old_loss = original_loss(a, labels, weights)
            new_loss = trainer.structural_loss(b, *labels, weights, 2, 12)
            old_loss.backward()
            new_loss.backward()
            assert torch.equal(old_loss, new_loss) and torch.equal(a.grad, b.grad)
            loss_checks.append({"zero_weights": zero, "default_loss_and_gradient_bitwise_equal": True})
        recipes = json.loads((HERE / "launch_recipes_proposed.json").read_text())
        configurations = []
        for job in recipes["training_jobs_in_order"]:
            options = job["argv"][2:]
            args = ap.parse_args(options)
            source = parent if job["stage"] == "continuation" else {"args": {**parent["args"], "spread_aux_mode": job["mode"], "spread_cache_root": "must-not-inherit"}}
            trainer.inherit_init_arguments(args, source, options)
            assert args.aux_struct == args.aux_next == 0.3
            assert args.hidden == 128 and args.layers == 2 and args.arch == "gru"
            assert args.seed == 3 and args.epochs == 1 and args.checkpoint_epoch == 0 and args.no_eval_during_fit
            trainer.configure_checkpoint_policy(args, ap)
            if job["stage"] == "continuation":
                assert args.batch == 32 and args.chunk == 2000 and args.spread_aux_mode == job["mode"]
                assert args.ema_teacher == 1 and args.eval_ema and args.since_weight == 32
            else:
                assert args.batch == 16 and args.spread_aux_mode == "original" and args.spread_cache_root == ""
                assert args.ema_teacher == 0 and not args.eval_ema and args.since_weight == 0 and args.propensity == ""
            n_out = 2 + (12 + 3 + 4 if args.aux_struct > 0 else 0) + (2 if args.aux_next > 0 else 0)
            assert n_out == 23
            configurations.append({"stage": job["stage"], "mode": job["mode"], "n_out": n_out, "aux_struct": args.aux_struct, "aux_next": args.aux_next, "effective_spread_mode": args.spread_aux_mode, "eval_ema": args.eval_ema})
        results[label] = {"default_args_preserved": True, "legacy_factory_output_gradients_bitwise_equal": legacy, "actual_parent_strict_load": True, "new_factory_checks": new_checks, "default_original_loss": loss_checks, "proposed_cli_inheritance_checks": configurations}
    assert not torch.cuda.is_initialized()
    assert hashlib.sha256((ROOT / "scripts/train_rnn.py").read_bytes()).hexdigest() == closure["base_trainer_sha256"]
    cgroup = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    scope = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    limits = {name: (scope / name).read_text().strip() for name in ["memory.max", "memory.swap.max", "cpu.max"]}
    assert limits["memory.max"] == "2147483648" and limits["memory.swap.max"] == "0"
    quota, period = map(int, limits["cpu.max"].split())
    assert quota <= period
    result = {"passed": True, "synthetic_only": True, "real_data_read": False, "actual_parent_weights_loaded_for_strict_factory_check": True, "CUDA_initialized": False, "threads": torch.get_num_threads(), "cgroup": cgroup, "verified_limits": limits, "runtime_seconds": time.monotonic() - started, "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "python": sys.version, "torch": torch.__version__, "proof_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "closure_sha256": hashlib.sha256((HERE / "source_closure.json").read_bytes()).hexdigest(), "results": results}
    (HERE / "cpu_proof_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "runtime_seconds": result["runtime_seconds"], "peak_rss_KiB": result["peak_rss_KiB"]}))


if __name__ == "__main__":
    main()
