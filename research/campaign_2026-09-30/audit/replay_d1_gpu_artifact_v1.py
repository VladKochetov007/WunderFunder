"""Frozen D1 literal replay and separate SSM failure-coordinate diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch

from grade_event_content_v2 import serial
from grade_structured_state import reference
from wnn.event_content import EventContentModel
from wnn.structured_state import StructuredStateModel

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    limits = dict(line.split("=", 1) for line in subprocess.check_output(
        ["systemctl", "--user", "show", os.environ["WNN_SCOPE_UNIT"] + ".scope",
         "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "CPUQuotaPerSecUSec", "-p", "Slice"], text=True).splitlines())
    assert limits == {"MemoryMax": "2147483648", "MemorySwapMax": "0", "CPUQuotaPerSecUSec": "1s", "Slice": "wnn-campaign.slice"}
    ssm_path = AUDIT.parent / "integration/gpu_probe_multi_v2.cpu_replay.pt"
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ssm = torch.load(ssm_path, weights_only=True, map_location="cpu")
    ssm_model = StructuredStateModel(ssm["n_in"], ssm["n_out"], decay_init="multi")
    ssm_model.load_state_dict(ssm["model"], strict=True)
    with torch.no_grad():
        _, state = reference(ssm_model, ssm["input"], ssm_model.reset(1))
    coordinates = []
    for block, (gpu, cpu) in enumerate(zip(ssm["state"], state, strict=True)):
        error = (gpu - cpu).abs()
        bound = 2e-5 + 2e-4 * cpu.abs()
        for batch, index in (error > bound).nonzero().tolist():
            coordinates.append({"block": block, "batch": batch, "state_index": index,
                                "mode": "scalar" if index < 64 else "rotation",
                                "mode_index": index if index < 64 else (index - 64) // 2,
                                "rotation_coordinate": None if index < 64 else (index - 64) % 2,
                                "gpu_value": float(gpu[batch, index]), "cpu_literal_value": float(cpu[batch, index]),
                                "absolute_error": float(error[batch, index]), "bound": float(bound[batch, index])})
    diagnostic = {"replay_sha256": sha(ssm_path), "coordinates": coordinates,
                  "cpu_coefficients": [{"scalar_rho": b.transition_coefficients()[0].tolist(),
                                        "rotation_a": b.transition_coefficients()[1].tolist(),
                                        "rotation_b": b.transition_coefficients()[2].tolist(),
                                        "injection_scale": b.dynamics()[1].tolist()} for b in ssm_model.blocks],
                  "limits": "Coordinates and CPU coefficients only; GPU coefficients not captured, no cause attribution."}
    with (AUDIT / "ssm_literal_failure_coordinates_v1.json").open("x") as stream:
        json.dump(diagnostic, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"ssm_failure_coordinates": coordinates}), flush=True)
    profile_path = AUDIT.parent / "event_content/gpu_probe_content_v2.json"
    profile = json.loads(profile_path.read_text())
    checkpoint_path = profile_path.with_suffix(".pt")
    assert sha(checkpoint_path) == profile["checkpoint_sha256"]
    assert sha(ROOT / "src/wnn/event_content.py") == profile["source_sha256"]["model.py"]
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    assert data["synthetic_only"] is True and data["optimizer_updates"] == 10
    assert data["source_sha256"] == profile["source_sha256"]
    parity = data["parity"]
    assert parity["backend_flags"] == profile["parity_artifact"]["backend_flags"]
    assert parity["backend_flags"]["float32_matmul_precision"] == "highest"
    assert parity["backend_flags"]["cuda_matmul_allow_tf32"] is False
    assert parity["tolerances"] == {"output_max_absolute": 3e-5, "floating_state_max_absolute": .003,
                                    "integer_counts_and_previous_rawprices_exact": True}
    model = EventContentModel(data["n_in"], data["n_out"], data["arm"])
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    x = parity["inputs"]
    assert x.shape == (1, 2000, 211) and x.dtype == torch.float32 and torch.isfinite(x).all()
    with torch.no_grad():
        literal, state = serial(model, x)
        parallel, parallel_state = model(x, None)
    def compare(a: torch.Tensor, b: torch.Tensor, tolerance: float) -> dict:
        maximum = float((a - b).abs().max())
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        return {"max_abs": maximum, "finite": finite, "passed": finite and maximum <= tolerance}
    checks = {"gpu_output_vs_literal": compare(parity["output"], literal, 3e-5),
              "cpu_parallel_vs_literal": compare(parallel, literal, 3e-5)}
    for index in range(3):
        checks[f"gpu_state{index}_vs_literal"] = compare(parity["state"][index], state[index], .003)
        checks[f"cpu_parallel_state{index}_vs_literal"] = compare(parallel_state[index], state[index], .003)
    for index in range(3, 6):
        checks[f"state{index}_exact"] = {"passed": torch.equal(parity["state"][index], state[index]) and torch.equal(parallel_state[index], state[index])}
    amp = data["amp_parity_diagnostic"]
    assert all(torch.isfinite(t).all() for t in (amp["output"], *amp["state"]))
    assert all(torch.equal(amp["state"][i], parity["state"][i]) for i in range(3, 6))
    assert amp["state"][1].dtype == torch.float32 and amp["state"][2].dtype == torch.float32
    result = {"passed": all(c["passed"] for c in checks.values()), "arm": data["arm"], "checks": checks,
              "profile_sha256": sha(profile_path), "checkpoint_sha256": sha(checkpoint_path),
              "model_source_sha256": sha(ROOT / "src/wnn/event_content.py"),
              "literal_reference_source_sha256": sha(AUDIT / "grade_event_content_v2.py"),
              "script_sha256": sha(Path(__file__)), "scope_properties": limits,
              "cuda_initialized": torch.cuda.is_initialized(), "competition_data_read": False,
              "duration_seconds": time.monotonic() - start, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "amp_diagnostic": {"finite": True, "rawcounts_previous_exact": True, "numerical_parity_claim": False},
              "limits": "Final synthetic content student/reset B1T2000 only; position/raw/trained checkpoints and AMP accuracy ungraded."}
    assert result["cuda_initialized"] is False
    with (AUDIT / "d1_gpu_literal_replay_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
