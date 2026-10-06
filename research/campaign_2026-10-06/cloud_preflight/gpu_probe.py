from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

OUTPUT = Path("/kaggle/working/cloud_preflight_gpu")
CHECKPOINT = Path(
    "/kaggle/input/wnn-jax-realdata-e463-tpu-20261004-v1/"
    "source_bundle/artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"
)
CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
ROOT_NN_PY_SHA256 = ""
N_IN, HIDDEN, LAYERS, N_OUT = 211, 256, 2, 23
CHUNK_ROWS = 2000
STARTED = time.monotonic()
TERMINAL: dict[str, object] = {"status": "starting"}


def write_json(path: Path, value: object) -> None:
    """Atomically persist each phase and the terminal result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Hash checkpoint and export bytes without loading them into host memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_snapshot() -> dict[str, object]:
    """Capture host capacity, availability and this process RSS."""
    entries = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        entries[key] = int(raw.strip().split()[0]) * 1024
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    return {
        "host_total_bytes": entries["MemTotal"],
        "host_available_bytes": entries["MemAvailable"],
        "process_rss_bytes": resident_pages * page_bytes,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "cpu_count": os.cpu_count(),
    }


def checkpoint_state(torch: ModuleType) -> dict[str, object]:
    """Load the hash-bound private epoch-3 model with a safe Torch unpickler."""
    digest = sha256_file(CHECKPOINT)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"epoch-3 checkpoint hash differs: {digest}")
    safe_class = getattr(torch.torch_version, "TorchVersion", None)
    if safe_class is None:
        loaded = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    else:
        with torch.serialization.safe_globals([safe_class]):
            loaded = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("model"), dict):
        raise TypeError("epoch-3 checkpoint has no model state mapping")
    return loaded


def check_finite(torch: ModuleType, value: object, name: str) -> None:
    """Fail the preflight when a model output or gradient contains NaN/Inf."""
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"nonfinite {name}")


def run_probe() -> dict[str, object]:
    """Run root-GRU parity and one production-chunk update on one T4."""
    global TERMINAL
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(
        OUTPUT / "startup.json",
        {
            "phase": "python_started_before_framework_import",
            "time_epoch_seconds": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "packet": "wnn-gru-device-preflight-20261006-v1",
        },
    )
    signal.signal(
        signal.SIGALRM,
        lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("payload exceeded 360 seconds")),
    )
    signal.alarm(360)
    phase = "host_admission"
    try:
        host_start = memory_snapshot()
        host_reserve = max(2 * 1024**3, int(host_start["host_total_bytes"] * 0.20))
        if host_start["host_available_bytes"] < host_reserve:
            raise MemoryError("host RAM lacks the required 20%/2GiB headroom at startup")
        write_json(OUTPUT / "host_start.json", host_start)

        phase = "framework_import"
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model_source import RNNModel

        torch.set_num_threads(2)
        versions = {}
        for name in ("torch", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "absent"
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("requested T4 kernel has no visible CUDA device")
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            reserve_bytes = max(2 * 1024**3, int(total_bytes * 0.20))
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_bytes": total_bytes,
                    "free_bytes_at_start": free_bytes,
                    "required_headroom_bytes": reserve_bytes,
                }
            )
            if free_bytes < reserve_bytes:
                raise MemoryError(f"GPU {index} lacks the required 20%/2GiB free headroom")
        phase = "checkpoint_validation"
        loaded = checkpoint_state(torch)
        cpu_model = RNNModel(N_IN, hidden=HIDDEN, layers=LAYERS, n_out=N_OUT)
        cpu_model.load_state_dict(loaded["model"], strict=True)
        cpu_model.eval()

        phase = "device_parity"
        torch.manual_seed(314159)
        short_noise = torch.randn((1, 8, N_IN), dtype=torch.float32)
        short_input = cpu_model.mean + short_noise * cpu_model.std
        with torch.inference_mode():
            cpu_output, cpu_state = cpu_model(short_input)
        check_finite(torch, cpu_output, "CPU parity output")
        gpu_model = RNNModel(N_IN, hidden=HIDDEN, layers=LAYERS, n_out=N_OUT)
        gpu_model.load_state_dict(loaded["model"], strict=True)
        gpu_model = gpu_model.to("cuda:0").eval()
        with torch.inference_mode():
            gpu_output, gpu_state = gpu_model(short_input.to("cuda:0"))
        torch.cuda.synchronize(0)
        check_finite(torch, gpu_output, "GPU parity output")
        check_finite(torch, gpu_state, "GPU parity state")
        parity_max_abs = float((gpu_output.cpu() - cpu_output).abs().max().item())
        torch.testing.assert_close(gpu_output.cpu(), cpu_output, rtol=1e-3, atol=1e-4)
        parity = {
            "input_shape": list(short_input.shape),
            "output_shape": list(gpu_output.shape),
            "cpu_state_shape": list(cpu_state.shape),
            "state_shape": list(gpu_state.shape),
            "max_abs_error": parity_max_abs,
            "finite": True,
        }
        write_json(
            OUTPUT / "heartbeat.json",
            {
                "phase": phase,
                "elapsed_seconds": time.monotonic() - STARTED,
                "memory": memory_snapshot(),
            },
        )

        phase = "production_chunk_forward_backward"
        torch.manual_seed(271828)
        noise = torch.randn((1, CHUNK_ROWS, N_IN), dtype=torch.float32)
        cpu_chunk = cpu_model.mean + noise * cpu_model.std
        gpu_chunk = cpu_chunk.to("cuda:0")
        gpu_model.train()
        optimizer = torch.optim.AdamW(gpu_model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        output, state = gpu_model(gpu_chunk)
        loss = output.float().square().mean()
        check_finite(torch, output, "production-chunk output")
        check_finite(torch, state, "production-chunk recurrent state")
        check_finite(torch, loss, "diagnostic loss")
        loss.backward()
        gradient_tensors = [
            parameter.grad for parameter in gpu_model.parameters() if parameter.grad is not None
        ]
        if not gradient_tensors:
            raise RuntimeError("root GRU produced no gradients")
        for gradient in gradient_tensors:
            check_finite(torch, gradient, "gradient")
        optimizer.step()
        torch.cuda.synchronize(0)
        for parameter in gpu_model.parameters():
            check_finite(torch, parameter, "updated parameter")
        write_json(
            OUTPUT / "heartbeat.json",
            {
                "phase": phase,
                "elapsed_seconds": time.monotonic() - STARTED,
                "memory": memory_snapshot(),
            },
        )

        phase = "export_reload"
        gpu_model.eval()
        with torch.inference_mode():
            post_step_gpu, _ = gpu_model(short_input.to("cuda:0"))
        export_path = OUTPUT / "root_gru_one_step_export.pt"
        torch.save(
            {name: value.detach().cpu() for name, value in gpu_model.state_dict().items()},
            export_path,
        )
        export_model = RNNModel(N_IN, hidden=HIDDEN, layers=LAYERS, n_out=N_OUT)
        export_model.load_state_dict(
            torch.load(export_path, map_location="cpu", weights_only=True), strict=True
        )
        export_model.eval()
        with torch.inference_mode():
            post_step_cpu, _ = export_model(short_input)
        torch.testing.assert_close(post_step_gpu.cpu(), post_step_cpu, rtol=1e-3, atol=1e-4)
        torch.cuda.synchronize(0)
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        free_after, total_after = torch.cuda.mem_get_info(0)
        reserve = max(2 * 1024**3, int(total_after * 0.20))
        if peak_reserved > total_after - reserve or free_after < reserve:
            raise MemoryError("T4 peak or terminal free memory violated required headroom")
        host_after = memory_snapshot()
        if host_after["host_available_bytes"] < max(
            2 * 1024**3, int(host_after["host_total_bytes"] * 0.20)
        ):
            raise MemoryError("host RAM fell below the required 20%/2GiB headroom")
        phase = "complete"
        TERMINAL = {"status": "passed"}
        return {
            "status": "passed",
            "classification": "actual T4 root-GRU numerical/resource preflight; one synthetic diagnostic update; no scientific fit or score",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "root_nn_source_sha256": ROOT_NN_PY_SHA256,
            "dataset_mounts": {
                "verified_e463_source_and_checkpoint": CHECKPOINT.parent.parent.parent.parent.parent.exists(),
                "private_wunder_funder": Path("/kaggle/input/wunder-funder").exists(),
            },
            "model_dimensions": {"n_in": N_IN, "hidden": HIDDEN, "layers": LAYERS, "n_out": N_OUT},
            "actual_gpus": devices,
            "versions": versions,
            "parity": parity,
            "production_chunk": {
                "input_shape": list(gpu_chunk.shape),
                "output_shape": list(output.shape),
                "state_shape": list(state.shape),
                "finite_loss": float(loss.detach().cpu().item()),
                "finite_gradients": len(gradient_tensors),
                "optimizer_steps": 1,
                "chunk_rows": CHUNK_ROWS,
            },
            "export": {
                "path": export_path.name,
                "sha256": sha256_file(export_path),
                "bytes": export_path.stat().st_size,
                "post_reload_max_abs_error": float(
                    (post_step_gpu.cpu() - post_step_cpu).abs().max().item()
                ),
            },
            "gpu_memory": {
                "peak_reserved_bytes": peak_reserved,
                "peak_allocated_bytes": peak_allocated,
                "free_bytes_after": free_after,
                "total_bytes": total_after,
                "required_headroom_bytes": reserve,
            },
            "host_memory": host_after,
            "elapsed_seconds": time.monotonic() - STARTED,
        }
    except BaseException as error:
        TERMINAL = {
            "status": "failed",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({"event": "preflight_failed", **TERMINAL}), flush=True)
        raise
    finally:
        signal.alarm(0)
        TERMINAL["elapsed_seconds"] = time.monotonic() - STARTED
        TERMINAL["last_phase"] = phase
        TERMINAL["terminal_memory"] = memory_snapshot()
        write_json(OUTPUT / "terminal.json", TERMINAL)
        write_json(OUTPUT / "complete.json", {"completed": time.time(), **TERMINAL})


def main() -> None:
    """Run the bounded device probe and preserve a failure status on exceptions."""
    try:
        result = run_probe()
    except BaseException:
        sys.exit(1)
    else:
        write_json(OUTPUT / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
