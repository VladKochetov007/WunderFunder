"""Replay synthetic GPU profiles with the independent literal CPU recurrence."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch

from grade_structured_state import difference, reference
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
    model_source = ROOT / "src/wnn/structured_state.py"
    cases = []
    for arm in ["multi", "single"]:
        base = AUDIT.parent / f"integration/gpu_probe_{arm}_v2"
        profile_path = base.with_suffix(".json")
        profile = json.loads(profile_path.read_text())
        replay_path = base.with_suffix(".cpu_replay.pt")
        assert sha(replay_path) == profile["cpu_replay_sha256"]
        assert sha(model_source) == profile["source_sha256"]["model.py"]
        # Author serialized torch.__version__ as this inert str subclass.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            data = torch.load(replay_path, map_location="cpu", weights_only=True)
        assert data["synthetic_only"] is True and data["competition_data_read"] is False
        assert data["reset"] is True and data["autocast_enabled"] is False
        assert data["decay_init"] == arm and data["optimizer_updates"] == 10 and data["ema_updates"] == 10
        assert data["source_sha256"] == profile["source_sha256"]
        assert data["backend_flags"] == profile["literal_replay_backend_flags"]
        assert data["backend_flags"]["float32_matmul_precision"] == "highest"
        assert data["backend_flags"]["cuda_matmul_allow_tf32"] is False
        checkpoint_path = base.with_suffix(".pt")
        assert sha(checkpoint_path) == profile["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert all(torch.equal(value, checkpoint["model"][name]) for name, value in data["model"].items())
        model = StructuredStateModel(data["n_in"], data["n_out"], decay_init=arm)
        model.load_state_dict(data["model"], strict=True)
        model.eval()
        x = data["input"]
        assert x.shape == (1, 2000, 211) and x.dtype == torch.float32
        assert torch.isfinite(x).all() and torch.isfinite(data["prediction"]).all()
        assert all(s.dtype == torch.float32 and torch.isfinite(s).all() for s in data["state"])
        with torch.no_grad():
            literal, states = reference(model, x, model.reset(1))
            parallel, parallel_states = model(x, None)
        checks = {
            "gpu_prediction_vs_cpu_literal": difference(data["prediction"], literal, 2e-5, 2e-4),
            "cpu_parallel_vs_cpu_literal": difference(parallel, literal, 2e-5, 2e-4),
            **{f"gpu_state{i}_vs_cpu_literal": difference(s, target, 2e-5, 2e-4) for i, (s, target) in enumerate(zip(data["state"], states, strict=True))},
            **{f"cpu_parallel_state{i}_vs_cpu_literal": difference(s, target, 2e-5, 2e-4) for i, (s, target) in enumerate(zip(parallel_states, states, strict=True))},
        }
        cases.append({"arm": arm, "passed": all(c["finite"] and c["violations"] == 0 for c in checks.values()),
                      "profile_sha256": sha(profile_path), "replay_sha256": sha(replay_path),
                      "checkpoint_sha256": sha(checkpoint_path), "backend_flags": data["backend_flags"], "checks": checks})
        print(json.dumps(cases[-1]), flush=True)
    result = {"passed": all(c["passed"] for c in cases), "cases": cases,
              "script_sha256": sha(Path(__file__)), "literal_reference_source_sha256": sha(AUDIT / "grade_structured_state.py"),
              "model_source_sha256": sha(model_source), "tolerances": {"atol": 2e-5, "rtol": 2e-4},
              "scope_properties": limits, "cuda_initialized": torch.cuda.is_initialized(),
              "duration_seconds": time.monotonic() - start, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "competition_data_read": False,
              "limits": "Final synthetic student/reset B1T2000 FP32 only; not teacher/trained checkpoints/AMP deployment or public quality."}
    assert result["cuda_initialized"] is False
    with (AUDIT / "ssm_gpu_literal_replay_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
