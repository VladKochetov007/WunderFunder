"""Independent original-source observer and preregistered backend-floor policy."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import torch
from live_capture import ROOT, Capture, compare, restore, sha, trainer_module

REPEATS = 6
ABS_CEILING = 1e-5
REL_CEILING = 1e-4
FLOOR_MULTIPLIER = 2.0


def independent_original_nodes(cpu: bool) -> list[ast.stmt]:
    """Extract original trainer independently of the online instrumenter."""
    tree = ast.parse((ROOT / "scripts/train_rnn.py").read_text())
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "s"
        and ast.unparse(node.iter) == "range(0, inputs.shape[1], args.chunk)"
    ]
    if len(loops) != 1:
        raise ValueError("original baseline source structure drift")
    result = []
    for source_node in loops[0].body:
        node = copy.deepcopy(source_node)
        text = ast.unparse(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "duty is not None":
            break
        if isinstance(node, ast.If) and text.startswith("if moments is not None:"):
            result.append(node)
            result.extend(ast.parse("_baseline.component('main', loss)").body)
            continue
        if isinstance(node, ast.If):
            arm = {
                "args.aux_struct > 0": "struct",
                "args.aux_next > 0": "next",
                "args.ema_teacher > 0": "ema",
            }.get(ast.unparse(node.test))
            if arm:
                for assignment in node.body:
                    if (
                        isinstance(assignment, ast.Assign)
                        and ast.unparse(assignment.targets[0]) == "loss"
                        and isinstance(assignment.value, ast.BinOp)
                    ):
                        assignment.value.right = ast.Call(
                            func=ast.parse("_baseline.component", mode="eval").body,
                            args=[ast.Constant(arm), assignment.value.right],
                            keywords=[],
                        )
        if text == "scaler.unscale_(opt)":
            result.extend(ast.parse("_baseline.inject(locals())").body)
        if text.startswith("torch.nn.utils.clip_grad_norm_"):
            node.value = ast.Call(
                func=ast.parse("_baseline.record_clip", mode="eval").body,
                args=[node.value],
                keywords=[],
            )
        result.append(node)
        if text == "epoch_counter.observe_unscaled(scaler, opt)":
            result.extend(ast.parse("_baseline.gradients(locals(), False)").body)
        if text.startswith("torch.nn.utils.clip_grad_norm_"):
            result.extend(ast.parse("_baseline.gradients(locals(), True)").body)
        if isinstance(node, ast.If) and text.startswith("if ema_model is not None:"):
            result.extend(ast.parse("_baseline.after(locals())").body)
        if cpu:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and ast.unparse(child.func) == "torch.autocast":
                    child.args[0] = ast.Constant("cpu")
                    child.keywords.append(ast.keyword(arg="enabled", value=ast.Constant(False)))
    return result


def original_default(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["execution"]["device"] == "cuda":
        from runner import runtime_guard

        runtime_guard()
    env = restore(payload, trainer_module())
    observer = Capture(True, force_skip=payload["force_skip"])
    observer.pre = payload
    env["_baseline"] = observer
    code = compile(
        ast.fix_missing_locations(
            ast.Module(
                body=independent_original_nodes(payload["execution"]["device"] == "cpu"),
                type_ignores=[],
            )
        ),
        "independent_original_default",
        "exec",
    )
    exec(code, env)
    return observer.post


def floating_leaves(value: Any, path: str = "root") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return {path: value}
    if isinstance(value, dict):
        return {
            key: tensor
            for name, item in value.items()
            for key, tensor in floating_leaves(item, path + "." + str(name)).items()
        }
    if isinstance(value, (list, tuple)):
        return {
            key: tensor
            for index, item in enumerate(value)
            for key, tensor in floating_leaves(item, path + "." + str(index)).items()
        }
    return {}


def compare_policy(
    left: Any, right: Any, policy: dict[str, Any], path: str = "root"
) -> dict[str, float]:
    if isinstance(left, torch.Tensor) and left.is_floating_point():
        if (
            not isinstance(right, torch.Tensor)
            or left.shape != right.shape
            or left.dtype != right.dtype
            or left.device != right.device
        ):
            raise AssertionError(path + ": tensor metadata differs")
        for mask in (torch.isnan, torch.isposinf, torch.isneginf):
            if not torch.equal(mask(left), mask(right)):
                raise AssertionError(path + ": nonfinite pattern differs")
        if path not in policy["limits"]:
            raise AssertionError(path + ": no independently frozen baseline floor")
        limit = policy["limits"][path]
        if limit == 0:
            return compare(left, right, path=path)
        finite = torch.isfinite(left)
        discrepancy = float((left[finite] - right[finite]).abs().max()) if finite.any() else 0.0
        if discrepancy > limit:
            raise AssertionError(path + ": exceeds independently frozen floor; no relaxation")
        return {path: discrepancy}
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(path + ": keys differ")
        out = {}
        for key in left:
            out.update(compare_policy(left[key], right[key], policy, path + "." + str(key)))
        return out
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(path + ": sequence differs")
        out = {}
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.update(compare_policy(a, b, policy, path + "." + str(index)))
        return out
    return compare(left, right, path=path)


def derive_policy(samples: list[dict[str, Any]], device: str) -> dict[str, Any]:
    if len(samples) != REPEATS:
        raise ValueError("exactly six independent original/default baselines required")
    leaves = [floating_leaves(sample) for sample in samples]
    limits, floors, ceilings = {}, {}, {}
    for path, reference in leaves[0].items():
        maximum, floor = 0.0, 0.0
        for index, leaf in enumerate(leaves):
            if set(leaf) != set(leaves[0]):
                raise AssertionError("baseline floating paths differ")
            value = leaf[path]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise AssertionError(path + ": baseline metadata changed")
            for mask in (torch.isnan, torch.isposinf, torch.isneginf):
                if not torch.equal(mask(value), mask(reference)):
                    raise AssertionError(path + ": unstable baseline nonfinite masks")
            finite = torch.isfinite(value)
            maximum = max(maximum, float(value[finite].abs().max()) if finite.any() else 0.0)
            for previous in leaves[:index]:
                floor = max(
                    floor,
                    float((value[finite] - previous[path][finite]).abs().max())
                    if finite.any()
                    else 0.0,
                )
        limit = FLOOR_MULTIPLIER * floor
        ceiling = ABS_CEILING + REL_CEILING * maximum
        if limit > ceiling or (device == "cpu" and floor != 0):
            raise AssertionError(path + ": baseline floor violates fixed ceiling/exact CPU policy")
        floors[path], limits[path], ceilings[path] = floor, limit, ceiling
    policy = {
        "version": 1,
        "device": device,
        "repeats": REPEATS,
        "floor_source": "independent original/defaults only; first reconstructed descriptor included",
        "cold_backend_limit": "process already warmed; globally cold CUDA process/cache UNSUPPORTED",
        "limits": limits,
        "empirical_floors": floors,
        "fixed_ceilings": ceilings,
        "ceiling_rule": {"abs": ABS_CEILING, "rel": REL_CEILING},
        "multiplier": FLOOR_MULTIPLIER,
        "automatic_relaxation": False,
    }
    for sample in samples[1:]:
        compare_policy(samples[0], sample, policy)
    return policy


def calibrate(payload: dict[str, Any], directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory.mkdir(exist_ok=False)
    samples = []
    for index in range(REPEATS):
        sample = original_default(payload)
        torch.save(sample, directory / f"original_default_{index}.pt")
        samples.append(sample)
    policy = derive_policy(samples, payload["execution"]["device"])
    policy["source"] = payload["source"]
    policy["sample_hashes"] = {
        f"original_default_{index}.pt": sha(directory / f"original_default_{index}.pt")
        for index in range(REPEATS)
    }
    (directory / "frozen_floor.json").write_text(json.dumps(policy, indent=2) + "\n")
    return samples[0], policy
