"""Package a native blend submission.

    .venv/bin/python scripts/package_submission.py NAME RUN:COEF [RUN:COEF ...]
RUN is a training run in artifacts/runs (head sliced to the two targets) or 'baseline' (exact
RNNModel rewrite of the organizer ONNX; COEF should include its 0.25 output scale). Writes
submissions/NAME/pkg and submissions/NAME/submission.zip using the verified native builds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE_FROM = ROOT / "submissions" / "ab18_wmse_struct" / "pkg"


def export(run: str, path: Path):
    if run == "baseline":
        src = ROOT / "submissions" / "_parts" / "baseline_rnnmodel.npz"
        if not src.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_baseline.py"), str(src)], check=True)
        shutil.copy(src, path)
        return
    import torch

    ck = torch.load(ROOT / "artifacts" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
    sd = {k: v.float().numpy() for k, v in ck["model"].items()}
    sd["head.2.weight"] = np.ascontiguousarray(sd["head.2.weight"][:2])
    sd["head.2.bias"] = np.ascontiguousarray(sd["head.2.bias"][:2])
    np.savez(path, **sd)


def main():
    name, members = sys.argv[1], sys.argv[2:]
    pkg = ROOT / "submissions" / name / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    for lvl in ("v2", "v3"):
        shutil.copytree(NATIVE_FROM / lvl, pkg / lvl)
    shutil.copy(ROOT / "submission" / "solution_multi.py", pkg / "solution.py")
    spec = []
    for i, m in enumerate(members):
        run, _, coef = m.rpartition(":")
        fname = f"weights_{i}.npz"
        export(run, pkg / fname)
        spec.append({"weights": fname, "coef": float(coef), "source": run})
    (pkg / "members.json").write_text(json.dumps(spec, indent=1))
    zp = ROOT / "submissions" / name / "submission.zip"
    zp.unlink(missing_ok=True)
    files = ["solution.py", "members.json", "v2", "v3"] + [s["weights"] for s in spec]
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    print(zp, zp.stat().st_size, spec)


if __name__ == "__main__":
    main()
