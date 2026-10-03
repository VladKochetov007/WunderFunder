"""Package a native blend submission.

    .venv/bin/python scripts/package_submission.py NAME RUN:COEF [RUN:COEF ...] [--native DIR] [--out DIR]
RUN is a training run in artifacts/runs (or a path to its model.pt; head sliced to the two targets)
or 'baseline' (exact RNNModel rewrite of the organizer ONNX; COEF should include its 0.25 output
scale). Writes OUT/pkg and OUT/submission.zip (OUT defaults to submissions/NAME) with the native
builds from DIR/v2 and DIR/v3 (default: the verified linear-only builds). Each member's encoder
(enc="linear" | "conv") is recorded in members.json and must be supported by both builds.
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


def checkpoint(run: str) -> Path:
    p = Path(run)
    return p if p.suffix == ".pt" else ROOT / "artifacts" / "runs" / run / "model.pt"


def export(run: str, path: Path) -> str:
    """Write the member's weights npz; returns its encoder name."""
    if run == "baseline":
        src = ROOT / "submissions" / "_parts" / "baseline_rnnmodel.npz"
        if not src.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_baseline.py"), str(src)], check=True)
        shutil.copy(src, path)
        return "linear"
    import torch

    ck = torch.load(checkpoint(run), map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    if a.get("arch", "gru") != "gru" or a.get("cell", "gru") != "gru":
        raise SystemExit(f"{run}: arch={a.get('arch')} cell={a.get('cell')} has no native engine")
    sd = {k: v.float().numpy() for k, v in ck["model"].items()}
    sd["head.2.weight"] = np.ascontiguousarray(sd["head.2.weight"][:2])
    sd["head.2.bias"] = np.ascontiguousarray(sd["head.2.bias"][:2])
    np.savez(path, **sd)
    return a.get("enc", "linear")


def native_encoders(level_dir: Path) -> set[str]:
    """Encoders a native build supports (builds before enc="conv" only run the linear encoder)."""
    code = "import wnn_native; print(','.join(getattr(wnn_native, 'ENCODERS', ['linear'])))"
    out = subprocess.run([sys.executable, "-c", code], cwd=level_dir, capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(level_dir), "PATH": "/usr/bin:/bin"})
    return set(out.stdout.strip().split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("members", nargs="+", metavar="RUN:COEF")
    ap.add_argument("--native", type=Path, default=NATIVE_FROM, help="directory holding v2/ and v3/ builds")
    ap.add_argument("--out", type=Path, default=None, help="default submissions/NAME")
    args = ap.parse_args()
    out = args.out or ROOT / "submissions" / args.name
    pkg = out / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    for lvl in ("v2", "v3"):
        shutil.copytree(args.native / lvl, pkg / lvl)
    shutil.copy(ROOT / "submission" / "solution_multi.py", pkg / "solution.py")
    spec = []
    for i, m in enumerate(args.members):
        run, _, coef = m.rpartition(":")
        fname = f"weights_{i}.npz"
        enc = export(run, pkg / fname)
        spec.append({"weights": fname, "coef": float(coef), "source": run, "enc": enc})
    need = {s["enc"] for s in spec}
    for lvl in ("v2", "v3"):
        missing = need - native_encoders(pkg / lvl)
        if missing:
            raise SystemExit(f"{args.native / lvl} does not support enc={sorted(missing)}; "
                             "build current native/ with scripts/build_native.sh and pass --native")
    (pkg / "members.json").write_text(json.dumps(spec, indent=1))
    zp = out / "submission.zip"
    zp.unlink(missing_ok=True)
    files = ["solution.py", "members.json", "v2", "v3"] + [s["weights"] for s in spec]
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    print(zp, zp.stat().st_size, spec)


if __name__ == "__main__":
    main()
