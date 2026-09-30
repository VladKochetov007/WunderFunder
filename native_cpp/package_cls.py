"""Package a class-scaled C++ submission (submission/solution_cls.py) by reusing an existing C++ package.

    .venv/bin/python native_cpp/package_cls.py NAME BASE_PKG DET_RUN --scale 3=1.4 [--scale 0=0 ...]

BASE_PKG is submissions/<name>/pkg of a package_cpp.py build (builds, members.json, weights are copied).
DET_RUN is an aux-struct GRU run whose head rows [17, 21) are the market-class logits; it is exported
twice with rows (17, 18) and (19, 20) in the two target slots.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_ZIP_BYTES = 20 * 1024 * 1024
LO = 17


def export_rows(run: str, rows: tuple[int, int], path: Path) -> None:
    import torch

    run_name, _, epoch = run.partition("@ep")
    ckpt = f"model_ep{int(epoch)}.pt" if epoch else "model.pt"
    ck = torch.load(ROOT / "artifacts" / "runs" / run_name / ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.float().numpy() for k, v in ck["model"].items()}
    sd["head.2.weight"] = np.ascontiguousarray(sd["head.2.weight"][list(rows)])
    sd["head.2.bias"] = np.ascontiguousarray(sd["head.2.bias"][list(rows)])
    np.savez(path, **sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("base_pkg", type=Path)
    ap.add_argument("det_run")
    ap.add_argument("--scale", action="append", default=[], help="CLASS=FACTOR")
    a = ap.parse_args()
    pkg = ROOT / "submissions" / a.name / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.copytree(a.base_pkg, pkg)
    shutil.copy(ROOT / "submission" / "solution_cls.py", pkg / "solution.py")
    export_rows(a.det_run, (LO, LO + 1), pkg / "det_a.npz")
    export_rows(a.det_run, (LO + 2, LO + 3), pkg / "det_b.npz")
    scale = {c: float(f) for c, f in (s.split("=") for s in a.scale)}
    (pkg / "classes.json").write_text(json.dumps(
        {"detector": ["det_a.npz", "det_b.npz"], "from_row": 50, "decide_row": 99, "scale": scale,
         "source": a.det_run}, indent=1))
    zp = ROOT / "submissions" / a.name / f"{a.name}.zip"
    zp.unlink(missing_ok=True)
    files = sorted(p.name for p in pkg.iterdir())
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    size = zp.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise SystemExit(f"{zp} is {size} bytes, over the 20 MB limit")
    print(zp, size, scale)


if __name__ == "__main__":
    main()
