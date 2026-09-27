"""Package a native blend submission.

    .venv/bin/python scripts/package_submission.py NAME RUN:COEF [RUN:COEF ...] [--native PKGDIR]
RUN is a training run in artifacts/runs (head sliced to the two targets) or 'baseline' (exact
RNNModel rewrite of the organizer ONNX; COEF should include its 0.25 output scale). Runs trained
with --online-norm (n_in 323) are flagged "online_norm" in members.json and need a native build
with that capability (scripts/build_native.sh); PKGDIR holds the v2/ and v3/ builds to ship.
Writes submissions/NAME/pkg and submissions/NAME/submission.zip.
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
NATIVE_FROM = ROOT / "submissions" / "ab18_wmse_struct" / "pkg"
N_RAW, N_EVENT = 112, 99


def export(run: str, path: Path) -> dict:
    """Write the member's weights to path; return the extra members.json fields it needs."""
    if run == "baseline":
        src = ROOT / "submissions" / "_parts" / "baseline_rnnmodel.npz"
        if not src.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_baseline.py"), str(src)], check=True)
        shutil.copy(src, path)
        return {}
    import torch

    ck = torch.load(ROOT / "artifacts" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    online = bool(a.get("online_norm"))
    unsupported = (a.get("arch", "gru") != "gru" or a.get("cell", "gru") != "gru"
                   or a.get("enc") not in (None, "linear") or a.get("raw_only") or ck.get("readout") is not None)
    if unsupported or ck["n_in"] != N_RAW + N_EVENT + N_RAW * online:
        raise SystemExit(f"{run}: the native engine runs linear-encoder GRUs on raw + event (+ online-norm) "
                         f"inputs with a regression head; got n_in {ck['n_in']}, args {a}")
    sd = {k: v.float().numpy() for k, v in ck["model"].items()}
    sd["head.2.weight"] = np.ascontiguousarray(sd["head.2.weight"][:2])
    sd["head.2.bias"] = np.ascontiguousarray(sd["head.2.bias"][:2])
    np.savez(path, **sd)
    return {"online_norm": True, "n_in": ck["n_in"]} if online else {}


def native_capabilities(pkg: Path, level: str) -> set:
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import wnn_native; "
            "print(' '.join(getattr(wnn_native, 'CAPABILITIES', ())))")
    out = subprocess.run([sys.executable, "-c", code, str(pkg / level)], check=True, capture_output=True, text=True)
    return set(out.stdout.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("members", nargs="+", help="RUN:COEF")
    ap.add_argument("--native", type=Path, default=NATIVE_FROM, help="directory with the v2/ and v3/ builds")
    args = ap.parse_args()
    pkg = ROOT / "submissions" / args.name / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    for lvl in ("v2", "v3"):
        shutil.copytree(args.native / lvl, pkg / lvl)
    shutil.copy(ROOT / "submission" / "solution_multi.py", pkg / "solution.py")
    spec = []
    for i, m in enumerate(args.members):
        run, _, coef = m.rpartition(":")
        fname = f"weights_{i}.npz"
        extra = export(run, pkg / fname)
        spec.append({"weights": fname, "coef": float(coef), "source": run, **extra})
    # Probe v2 (v3 cannot be imported on a CPU without AVX2); both are built from the same source.
    if any(s.get("online_norm") for s in spec) and "online_norm" not in native_capabilities(pkg, "v2"):
        raise SystemExit(f"{args.native}: native build lacks online_norm; build one with scripts/build_native.sh")
    (pkg / "members.json").write_text(json.dumps(spec, indent=1))
    zp = ROOT / "submissions" / args.name / "submission.zip"
    zp.unlink(missing_ok=True)
    files = ["solution.py", "members.json", "v2", "v3"] + [s["weights"] for s in spec]
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    print(zp, zp.stat().st_size, spec)


if __name__ == "__main__":
    main()
