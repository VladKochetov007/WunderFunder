"""Prepare fixed-case metadata and source-inferred trajectory without gradient reads."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
CONT = HERE.parent
CAPTURE = CONT / "round4/gpu_capture_v4_default_2120r"
V4 = CONT / "round4/live_capture_v4"
sys.path.insert(0, str(HERE / "amp_linearity_audit_v2"))
from metadata import canonical_metadata  # noqa: E402


def sha(path: Path) -> str:
    """Hash bounded file blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value: torch.Tensor) -> str:
    """Preserve actor hash grammar without copying the entire input tensor."""
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("Expected contiguous CPU mmap tensor")
    digest = hashlib.sha256(
        json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode()
    )
    view = memoryview(value.detach().view(torch.uint8).reshape(-1).numpy())
    for start in range(0, len(view), 1024**2):
        digest.update(view[start : start + 1024**2])
    return digest.hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Preserve any previous metadata preparation."""
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def main() -> None:
    """Require landed independent measurement PASS before trusted local mmap."""
    review = CONT / "round6/capture_measurement_review/STATE.json"
    verdict = json.loads(review.read_text())
    if verdict["verdict"] != "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY":
        raise ValueError("Independent default measurement gate missing")
    manifest = json.loads((CAPTURE / "output_hashes.json").read_text())["files"]
    torch.set_num_threads(1)
    cases: dict[str, Any] = {}
    for name, prefix in (
        ("synthetic_actual_architecture_normal", "synthetic_actual_architecture/normal"),
        ("real_panel_normal", "real_fit_panel/normal"),
    ):
        paths = {kind: CAPTURE / prefix / f"{kind}.pt" for kind in ("pre", "post", "components")}
        pre = torch.load(paths["pre"], map_location="cpu", mmap=True, weights_only=False)
        canonical = {str(key): value for key, value in pre["optimizer_id_names"].items()}
        cases[name] = {
            "default_ordinary_passed": True,
            "forced_skip": False,
            "evidence_kind": pre["metadata"]["evidence_kind"],
            "sha256": {kind: manifest[f"{prefix}/{kind}.pt"] for kind in paths},
            "paths": {kind: str(path) for kind, path in paths.items()},
            "source": pre["source"], "execution": pre["execution"],
            "metadata_identity": canonical_metadata(pre["metadata"]), "optimizer_id_names": canonical,
            "optimizer_groups": [
                {"ids": group["params"], "names": names, "muon": group.get("muon", False)}
                for group, names in zip(pre["optimizer"]["param_groups"], pre["optimizer_names"], strict=True)
            ],
            "tensor_identity": {
                item: {"shape": list(pre["student"][item].shape), "dtype": str(pre["student"][item].dtype)}
                for item in canonical.values()
            },
        }
        if name == "real_panel_normal" and not (HERE / "real_trajectory.json").exists():
            if int(pre["clock"]["skips"]) != 0 or pre["clock"]["attempts"] != 4:
                raise ValueError("Fixed original warm4/no-skip trajectory missing")
            trace = {
                "initial_student_h": None, "initial_ema_h": None,
                "warm_starts": [0, 1000, 2000, 3000],
                "completed_warm_updates": 4, "warm_skips": 0,
                "entrypoint": "runner.run_probe(mode='real')",
                "capture_runner_sha256": sha(V4 / "runner.py"),
                "fit_panel_source_sha256": sha(V4 / "fit_panel.py"),
                "adapter_sha256": sha(V4 / "live_capture.py"),
                "source": pre["source"],
                "inputs_identity": identity(pre["batch"]["inputs"]),
                "mean_identity": identity(pre["student"]["mean"]),
                "std_identity": identity(pre["student"]["std"]),
                "panel_sha256": pre["metadata"].get("panel_sha256"),
                "provenance_limit": "Source-inferred initial states and loop; no saved intermediate weight trace",
                "measured_capture_review_sha256": sha(review),
            }
            write_new(HERE / "real_trajectory.json", trace)
        del pre
    write_new(HERE / "fixed_case_metadata_v2.json", {
        "cases": cases, "approval_status": "METADATA_ONLY_NOT_ANALYSIS_AUTHORIZATION",
        "gradients_read": False, "CUDA_used": False,
        "measurement_review_sha256": sha(review),
    })
    print("Fixed metadata and source-inferred warm4/no-skip trajectory saved; no gradient reads")


if __name__ == "__main__":
    main()
