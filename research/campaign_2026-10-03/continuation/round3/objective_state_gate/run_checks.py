"""Frozen small state-gate fixtures and real export metadata inspection."""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
from capture_replay import (
    TinyGRU,
    analytic_output_gradient,
    capture_toy,
    detached_copy,
    replay,
    terms,
    validate_capture,
)
from torch._subclasses.fake_tensor import FakeTensorMode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def make_batch(slot: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(141 + slot)
    raw_y = torch.randn(2, 10, 2, generator=generator) * 2
    rows = torch.arange(10) + (10 if slot == 12 else 0)
    next_pair = (rows + 8)[None].expand(2, -1).clone()
    next_pair[:, 1::2] += 1  # strict boundary vs greater; teacher length also bounds eligibility
    next_pair.clamp_(max=20)
    return {
        "x": torch.randn(2, 10, 3, generator=generator),
        "raw_y": raw_y,
        "v": torch.rand(2, 10, generator=generator) + 0.03,
        "teachers": torch.randn(2, 2, 20, 2, generator=generator) * 4,
        "teacher_scale": torch.tensor([0.6, 1.7]),
        "next_pair": next_pair,
        "rows": rows,
        "struct_labels": torch.stack(
            [torch.randint(k, (2, 10), generator=generator) for k in (2, 3, 4)], -1
        ),
        "next_distance": torch.randint(1, 40, (2, 10), generator=generator).float(),
        "next_up": torch.randint(2, (2, 10), generator=generator).float(),
        "next_known": torch.randint(2, (2, 10), generator=generator).float(),
    }


def max_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(float((left[k] - right[k]).abs().max()) for k in left)


def export_metadata() -> dict[str, Any]:
    checkpoint = ROOT / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"
    provenance_path = checkpoint.parent / "provenance.json"
    with FakeTensorMode():
        exported = torch.load(checkpoint, map_location="cpu", weights_only=False)
        shapes = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in exported["model"].items()
        }
    provenance = json.loads(provenance_path.read_text())
    gaps = [
        "raw student paired with lagged EMA at same pre-forward instant",
        "optimizer implementation/groups/current learning rates/moments/step ages",
        "detached incoming h and h_ema with TBPTT history",
        "exact chunk inputs/labels/weights/cached teacher values and identities",
        "RNG, scaler, scheduler phase and attempted/completed step clocks at that instant",
    ]
    return {
        "inspection": "FakeTensorMode metadata-only; no model forward or tensor-value reads",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "provenance_sha256": sha256(provenance_path),
        "checkpoint_keys": sorted(exported),
        "model_tensor_schema": shapes,
        "epoch": exported["epoch"],
        "n_in": exported["n_in"],
        "args_selected": {
            key: exported["args"].get(key)
            for key in (
                "eval_ema",
                "ema_teacher",
                "ema_decay",
                "optim",
                "fast",
                "loss",
                "loss_norm",
                "distill_alpha",
                "distill_boot_h",
                "aux_struct",
                "aux_next",
                "clip",
            )
        },
        "provenance_keys": sorted(provenance),
        "exact_replay_admitted": False,
        "missing_inputs": gaps,
        "selection_clause": "train_rnn.checkpoint_model returns ema_model.module when eval_ema; otherwise SWA/raw",
        "caveat": "export may contain evaluation EMA weights, but not paired raw/lagged EMA and state history; parity-dump separately can store raw+EMA weights, still insufficient alone",
    }


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(194)
    student = TinyGRU()
    ema = copy.deepcopy(student).requires_grad_(False)
    opt = torch.optim.AdamW(
        student.parameters(), lr=0.03, weight_decay=0.02, foreach=False, fused=False
    )
    h, h_ema = torch.zeros(1, 2, 5), torch.zeros(1, 2, 5)
    # Real detached EMA sequence: each warmup teacher uses preceding averaged weights.
    for step in range(4):
        batch = make_batch(step)
        out, h = student(batch["x"], h.detach())
        with torch.no_grad():
            ema_out, h_ema = ema(batch["x"], h_ema.detach())
        loss = sum(terms(out, ema_out, batch).values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.12)
        opt.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), student.parameters(), strict=True):
                average.mul_(0.999).add_(current, alpha=0.001)
        h, h_ema = h.detach(), h_ema.detach()
    opt.zero_grad(set_to_none=True)
    identity = sha256(HERE / "capture_replay.py")
    captures = [
        capture_toy(student, ema, opt, h, h_ema, make_batch(10 + i), 4, identity) for i in range(3)
    ]
    # Vary weight mass while keeping each arm's finite multiplier multiset identical.
    for capture, scale in zip(captures, (0.25, 1.0, 4.0), strict=True):
        capture["batch"]["v"] *= scale
    masses = torch.tensor(
        [
            float((c["batch"]["v"][..., None] * c["batch"]["raw_y"].clamp(-2, 2).abs()).sum())
            for c in captures
        ]
    )
    scalars = masses / masses.mean()
    permutation = [2, 0, 1]
    reference = replay(captures[0], float(scalars[0]))
    repeated = replay(captures[0], float(scalars[0]))
    repeat_error = max_difference(reference, repeated)
    tolerance = max(1e-7, 10 * repeat_error)
    tests: dict[str, Any] = {}

    def record(name: str, passed: bool, **measurements: Any) -> None:
        tests[name] = {"pass": bool(passed), **measurements}
        print(json.dumps({"test": name, **tests[name]}), flush=True)

    record("repeat_identity", repeat_error <= tolerance, max_abs=repeat_error, tolerance=tolerance)
    rows = []
    for i, capture in enumerate(captures):
        aligned = replay(capture, float(scalars[i]))
        permuted = replay(capture, float(scalars[permutation[i]]))
        reconstruct = (
            float(scalars[i]) * aligned["g_main"]
            + aligned["g_ema"]
            + aligned["g_struct"]
            + aligned["g_next"]
        )
        component_error = float((reconstruct - aligned["gradient"]).abs().max())
        model = TinyGRU()
        model.load_state_dict(capture["student"])
        out, _ = model(capture["batch"]["x"], capture["h"])
        derivative = analytic_output_gradient(
            out.detach(), aligned["ema_output"], capture["batch"], float(scalars[i])
        )
        expected = torch.cat(
            [
                g.flatten()
                for g in torch.autograd.grad(
                    out, tuple(model.parameters()), grad_outputs=derivative
                )
            ]
        )
        oracle_error = float((expected - aligned["gradient"]).abs().max())
        arithmetic_tolerance = max(
            tolerance,
            4 * torch.finfo(torch.float32).eps * (1 + float(aligned["gradient"].abs().max())),
        )
        record(
            f"independent_derivative_slot_{i}",
            max(component_error, oracle_error) <= arithmetic_tolerance,
            component_error=component_error,
            oracle_error=oracle_error,
            arithmetic_tolerance=arithmetic_tolerance,
        )
        rows.append(
            {
                "slot": i,
                "mass": float(masses[i]),
                "aligned_scalar": float(scalars[i]),
                "permuted_scalar": float(scalars[permutation[i]]),
                "gradient_difference": float((aligned["gradient"] - permuted["gradient"]).norm()),
                "update_difference": float((aligned["delta"] - permuted["delta"]).norm()),
                "preclip_norm": float(aligned["preclip_norm"]),
                "postclip_norm": float(aligned["clipped"].norm()),
            }
        )
    record(
        "aligned_vs_permuted_positive",
        all(r["update_difference"] > tolerance for r in rows),
        slots=rows,
    )
    null = replay(captures[0], 1.0)
    equal_mass_captures = [detached_copy(captures[0]) for _ in range(3)]
    for i, capture in enumerate(equal_mass_captures):
        capture["batch"]["x"] = captures[i]["batch"]["x"].clone()
    equal_masses = [
        (c["batch"]["v"][..., None] * c["batch"]["raw_y"].clamp(-2, 2).abs()).sum()
        for c in equal_mass_captures
    ]
    equal_scalars = torch.stack(equal_masses) / torch.stack(equal_masses).mean()
    null_error = max(
        max_difference(
            replay(c, float(equal_scalars[i])), replay(c, float(equal_scalars[permutation[i]]))
        )
        for i, c in enumerate(equal_mass_captures)
    )
    record(
        "equal_mass_multiset_null",
        null_error <= tolerance,
        max_abs=null_error,
        scalars=equal_scalars.tolist(),
        explanation="distinct causal inputs, equal target/row masses; permuted scalars identical",
    )
    equal = detached_copy(captures[0])
    equal["ema"] = detached_copy(equal["student"])
    equal["h_ema"] = equal["h"].clone()
    zero_ema = replay(equal, 1.0)
    record(
        "equal_weight_and_hidden_ema_null",
        float(zero_ema["g_ema"].abs().max()) <= tolerance,
        ema_grad_max=float(zero_ema["g_ema"].abs().max()),
        main_grad_norm=float(zero_ema["g_main"].norm()),
    )
    record(
        "lagged_ema_is_nonzero",
        float(null["g_ema"].norm()) > tolerance,
        ema_grad_norm=float(null["g_ema"].norm()),
    )
    record(
        "changed_ema_detected",
        float((null["gradient"] - zero_ema["gradient"]).norm()) > tolerance,
        gradient_difference=float((null["gradient"] - zero_ema["gradient"]).norm()),
    )
    weight_only = detached_copy(equal)
    weight_only["h_ema"] = captures[0]["h_ema"].clone()
    weight_only_result = replay(weight_only, 1.0)
    record(
        "equal_weights_without_hidden_is_not_null",
        float(weight_only_result["g_ema"].norm()) > tolerance,
        ema_grad_norm=float(weight_only_result["g_ema"].norm()),
    )
    empty_moments = detached_copy(captures[0])
    for entry in empty_moments["optimizer"]["state"].values():
        entry["exp_avg"].zero_()
        entry["exp_avg_sq"].zero_()
    reset = replay(empty_moments, 1.0)
    record(
        "optimizer_moments_detected",
        float((null["delta"] - reset["delta"]).norm()) > tolerance
        and float((null["gradient"] - reset["gradient"]).abs().max()) <= tolerance,
        gradient_max_difference=float((null["gradient"] - reset["gradient"]).abs().max()),
        update_difference=float((null["delta"] - reset["delta"]).norm()),
    )
    for name in ("missing_hidden", "attached_ema", "unsupported_optimizer", "wrong_group_order"):
        invalid = detached_copy(captures[0])
        if name == "missing_hidden":
            del invalid["h_ema"]
        elif name == "attached_ema":
            invalid["ema"]["head.weight"].requires_grad_(True)
        elif name == "unsupported_optimizer":
            invalid["optimizer_class"] = "Muon"
        else:
            invalid["optimizer"]["param_groups"][0]["params"].reverse()
        try:
            validate_capture(invalid)
        except ValueError as error:
            record(name + "_rejected", True, reason=str(error))
        else:
            record(name + "_rejected", False)
    torch.save(
        {"captures": captures, "scalars": scalars, "permutation": permutation},
        HERE / "toy_capture.pt",
    )
    restored = torch.load(HERE / "toy_capture.pt", weights_only=False)
    record(
        "serialization_replay",
        max_difference(reference, replay(restored["captures"][0], float(restored["scalars"][0])))
        <= tolerance,
    )
    metadata = export_metadata()
    (HERE / "export_schema.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {
        "scope": "synthetic instrument validation only",
        "tests": tests,
        "all_pass": all(t["pass"] for t in tests.values()),
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "gpu_used": False,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_pass": result["all_pass"], "cost": result["cost"]}), flush=True)
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
