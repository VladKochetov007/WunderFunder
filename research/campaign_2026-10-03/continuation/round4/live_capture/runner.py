"""Bounded synthetic actual-code smoke; real-fit mode intentionally fails closed."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from live_capture import (
    HERE,
    ROOT,
    Capture,
    cloned,
    compare,
    compiled_chunk,
    generate_copy,
    identities,
    make_environment,
    replay,
    sha,
    tensor_bytes,
    trainer_module,
    validate,
)


def verify_review(path: Path | None) -> None:
    """Controller supplies an independent review receipt; actor never issues it."""
    if path is None:
        raise ValueError("CUDA launch requires independent controller review receipt")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("status") != "approved"
        or receipt.get("queue_finished") is not True
        or receipt.get("reviewer") in (None, "", "round4-capture")
    ):
        raise ValueError("independent review and queue completion required")
    names = (
        "runner.py",
        "live_capture.py",
        "fit_panel.py",
        "fit_panel.json",
        "PREREG.txt",
        "source_binding.json",
        "trainer.patch",
    )
    if (
        receipt.get("candidate_sha256") != {name: sha(HERE / name) for name in names}
        or receipt.get("source") != identities()
    ):
        raise ValueError("review receipt does not bind this candidate/source")


def check_gpu_memory() -> int:
    free, total = torch.cuda.mem_get_info()
    observed = total - free
    if observed > 5_500_000_000:
        raise RuntimeError("observed total device memory exceeds 5.5 decimal GB")
    return observed


def synthetic_environment(device: str = "cpu", scaler_enabled: bool = False) -> dict[str, Any]:
    """Actual architecture on CUDA, tiny same-code architecture on CPU."""
    module = trainer_module()
    provenance = json.loads(
        (ROOT / "artifacts/runs/c20261003_combo_s3_f0/provenance.json").read_text()
    )
    args = provenance["args"].copy()
    args["precision"] = "fp16" if device == "cuda" else "cpu-protocol"
    args["parity_dump"] = ""
    args["chunk"] = 1000 if device == "cuda" else 10
    cfg = {
        "n_in": 211 if device == "cuda" else 3,
        "hidden": 256 if device == "cuda" else 4,
        "layers": 2,
        "n_out": 23,
    }
    total = (provenance["fit_membership_identity"]["dataset_length"] // 32) * 20 * args["epochs"]
    env = make_environment(module, args, cfg, device, total, scaler_enabled or device == "cuda")
    b, t = (32, 6000) if device == "cuda" else (2, 60)
    env["inputs"] = torch.randn(b, t, cfg["n_in"], device=device) * 0.1
    # Explicitly synthetic labels, weights and teachers. No mature teacher claim.
    block = max(12, args["chunk"] // 2)
    labels = torch.randn(b, (t + block - 1) // block, 2, device=device)
    env["y"] = labels.repeat_interleave(block, dim=1)[:, :t]
    env["teach"] = torch.randn(b, t, 2, device=device) * 0.3
    env["main_valid"] = torch.rand(b, t, device=device) * 0.4 + 0.03
    env["sp_lab"] = torch.randint(12, (b, t), device=device)
    env["br_lab"] = torch.randint(3, (b, t), device=device)
    env["cl_lab"] = torch.randint(4, (b, t), device=device)
    env["sw"] = env["main_valid"].clone()
    valid = torch.ones(b, t, dtype=torch.bool, device=device)
    env["nx_rows"], env["nx_up"], nx_ok = module.next_change_labels(env["y"], valid)
    env["nw"] = env["main_valid"] * nx_ok
    env["nxt_label"] = module.next_label_change(env["y"])
    env["capture_metadata"] = {
        "evidence_kind": "synthetic_cuda_actual_architecture"
        if device == "cuda"
        else "synthetic_cpu_actual_code",
        "row_ids": [],
        "input_origin": "seeded random; synthetic labels/weights/teachers/aux; NOT real recipe data",
        "full_sequence_length": t,
        "scaler_protocol_cpu": scaler_enabled,
    }
    code = compiled_chunk(device == "cpu")
    for index in range(4):
        env["s"] = index * args["chunk"]
        exec(code, env)
        if device == "cuda":
            check_gpu_memory()
    env["s"] = 4 * args["chunk"]
    return env


def capture_step(
    env: dict[str, Any], directory: Path | None = None, force_skip: bool = False
) -> Capture:
    observer = Capture(True, directory, force_skip)
    env["_capture"] = observer
    exec(compiled_chunk(env["device"].type == "cpu"), env)
    validate(observer.pre)
    return observer


def run_probe(output: Path, device: str = "cpu", mode: str = "synthetic") -> dict[str, Any]:
    torch.manual_seed(734019)
    if device == "cuda":
        # Exactly the trainer's configure + fast toggles (no deterministic override).
        module = trainer_module()
        module.configure_torch()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.7, 4_800_000_000 / total))
    if mode == "real":
        from fit_panel import real_environment

        if device != "cuda":
            raise ValueError("real capture requires CUDA; no CPU architecture proxy")
        torch.manual_seed(3)
        env = real_environment()
        code = compiled_chunk()
        for index in range(4):
            env["s"] = index * 1000
            exec(code, env)
            check_gpu_memory()
        env["s"] = 4000
    else:
        env = synthetic_environment(device)
    observed = capture_step(env, output / "normal")
    # Avoid holding online model + replay model activation graphs simultaneously.
    del env
    module = trainer_module()
    atol, rtol = (1e-5, 1e-4) if device == "cuda" else (1e-7, 1e-6)
    default = replay(observed.pre, module)
    deltas = compare(observed.post, default, atol, rtol)
    # Independent uninstrumented original statements: no component hooks in the
    # online math/backward/update. Observe their output only after completion.
    from live_capture import restore

    bare_env = restore(observed.pre, module)
    exec(compiled_chunk(device == "cpu", instrumented=False), bare_env)
    bare_observer = Capture(True)
    bare_observer.pre = observed.pre
    bare_observer.gradients(bare_env, True)
    bare_observer.after(bare_env)
    comparable = set(observed.post) - {"losses", "preclip"}
    bare_deltas = compare(
        {key: observed.post[key] for key in comparable},
        {key: bare_observer.post[key] for key in comparable},
        atol,
        rtol,
    )
    del bare_env
    repeated = replay(observed.pre, module)
    repeat_deltas = compare(default, repeated, atol, rtol)
    decomposition = replay(observed.pre, module, components=True)
    compare(observed.post["losses"], decomposition["losses"], atol, rtol)
    torch.save(decomposition, output / "normal" / "components.pt")
    # Separate genuine warm state for deliberate overflow; compare unchanged default.
    if device == "cuda":
        env = restore(observed.pre, module)
    else:
        torch.manual_seed(734019)
        env = synthetic_environment(device, scaler_enabled=True)
    skipped = capture_step(env, output / "forced_skip", True)
    del env
    skip_replay = replay(skipped.pre, module)
    skip_deltas = compare(skipped.post, skip_replay, atol, rtol)
    compare(skipped.pre["student"], skipped.post["student"], 0, 0)
    assert skipped.post["clock"]["attempts"] == skipped.pre["clock"]["attempts"] + 1
    assert skipped.post["clock"]["skips"].item() == skipped.pre["clock"]["skips"].item() + 1
    assert skipped.post["ema_n"] == skipped.pre["ema_n"] + 1
    assert skipped.post["scheduler"]["last_epoch"] == skipped.pre["scheduler"]["last_epoch"] + 1
    result = {
        "evidence_kind": observed.pre["metadata"]["evidence_kind"],
        "source": identities(),
        "normal_max_delta": max(deltas.values(), default=0),
        "repeat_max_delta": max(repeat_deltas.values(), default=0),
        "skip_max_delta": max(skip_deltas.values(), default=0),
        "normal_deltas": deltas,
        "skip_deltas": skip_deltas,
        "repeat_deltas": repeat_deltas,
        "four_losses": {key: float(value) for key, value in observed.post["losses"].items()},
        "payload_tensor_bytes": tensor_bytes(observed.pre),
        "component_tensor_bytes": tensor_bytes(decomposition),
        "skip_clock": cloned(skipped.post["clock"]),
        "all_four_components": sorted(decomposition["component_gradients"]),
        "gpu_parity_established": device == "cuda",
        "real_fit_parity_established": device == "cuda" and mode == "real",
    }
    result["skip_clock"]["skips"] = int(result["skip_clock"]["skips"])
    result["uninstrumented_default_max_delta"] = max(bare_deltas.values(), default=0)
    result["uninstrumented_default_deltas"] = bare_deltas
    result["tolerances"] = {"absolute": atol, "relative": rtol}
    if device == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["observed_total_device_bytes_at_end"] = check_gpu_memory()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("synthetic", "real", "all"), default="synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path)
    options = parser.parse_args()
    if options.mode == "real":
        raise ValueError(
            "synthetic actual-architecture gate must run first in same job: use --mode all"
        )
    if options.device == "cuda":
        verify_review(options.review_receipt)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
    options.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        generate_copy()
        if options.mode == "all":
            if options.device != "cuda":
                raise ValueError(
                    "combined actual-architecture/fit panel mode requires reviewed CUDA launch"
                )
            synthetic = options.output / "synthetic_actual_architecture"
            synthetic.mkdir()
            result = run_probe(synthetic, options.device)
            (synthetic / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            real = options.output / "real_fit_panel"
            real.mkdir()
            result = run_probe(real, options.device, "real")
        else:
            result = run_probe(options.output, options.device)
        result["seconds"] = time.monotonic() - started
        (options.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {key: value for key, value in result.items() if not key.endswith("deltas")},
                indent=2,
            )
        )
    except BaseException:
        (options.output / "failure.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
