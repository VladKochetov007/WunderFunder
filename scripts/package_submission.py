"""Package a native blend submission.

    .venv/bin/python scripts/package_submission.py NAME RUN:COEF [RUN:COEF ...]
RUN is a training run in artifacts/runs (head sliced to the two targets; RUN@epN takes
model_epN.pt instead of model.pt), 'A@ep3|B@ep0' (one shared trunk with the average of the runs' heads merged
into a single wider head; the runs must be head-only fine-tunes of one parent) or 'baseline' (exact
RNNModel rewrite of the organizer ONNX; COEF should include its 0.25 output scale). Writes
submissions/NAME/pkg and submissions/NAME/submission.zip using the verified native builds.
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


def load_state(run: str) -> dict[str, np.ndarray]:
    import torch

    run_name, _, epoch = run.partition("@ep")
    ckpt = f"model_ep{int(epoch)}.pt" if epoch else "model.pt"
    ck = torch.load(ROOT / "artifacts" / "runs" / run_name / ckpt, map_location="cpu", weights_only=False)
    return {k: v.float().numpy() for k, v in ck["model"].items()}


def merged_heads(runs: list[str]) -> dict[str, np.ndarray]:
    """One trunk carrying the average of k heads, as a single head of hidden width k*H.

    head(x) = W2 GELU(W1 x + b1) + b2 per head; stacking the W1/b1 rows and scaling the stacked W2 columns by 1/k
    gives exactly mean_k head_k(x). Every run must share the trunk bit for bit (head-only fine-tunes of one parent).
    """
    states = [load_state(r) for r in runs]
    base = states[0]
    for r, st in zip(runs[1:], states[1:]):
        for k, v in base.items():
            if not k.startswith("head.") and not np.array_equal(v, st[k]):
                raise ValueError(f"{r}: trunk tensor {k} differs from {runs[0]}")
    k = len(states)
    out = {key: v for key, v in base.items() if not key.startswith("head.")}
    out["head.0.weight"] = np.concatenate([st["head.0.weight"] for st in states], axis=0)
    out["head.0.bias"] = np.concatenate([st["head.0.bias"] for st in states])
    out["head.2.weight"] = np.concatenate([st["head.2.weight"][:2] for st in states], axis=1) / k
    out["head.2.bias"] = sum(st["head.2.bias"][:2] for st in states) / k
    return {key: np.ascontiguousarray(v, dtype=np.float32) for key, v in out.items()}


def export(run: str, path: Path):
    if "|" in run:
        np.savez(path, **merged_heads(run.split("|")))
        return
    if run == "baseline":
        src = ROOT / "submissions" / "_parts" / "baseline_rnnmodel.npz"
        if not src.exists():
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "convert_baseline.py"), str(src)],
                check=True,
            )
        shutil.copy(src, path)
        return
    import torch

    run_name, _, epoch = run.partition("@ep")
    ckpt = f"model_ep{int(epoch)}.pt" if epoch else "model.pt"
    ck = torch.load(
        ROOT / "artifacts" / "runs" / run_name / ckpt, map_location="cpu", weights_only=False
    )
    sd = {k: v.float().numpy() for k, v in ck["model"].items()}
    sd["head.2.weight"] = np.ascontiguousarray(sd["head.2.weight"][:2])
    sd["head.2.bias"] = np.ascontiguousarray(sd["head.2.bias"][:2])
    np.savez(path, **sd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("members", nargs="+")
    parser.add_argument("--native-from", type=Path, default=NATIVE_FROM)
    parser.add_argument("--solution", type=Path, default=ROOT / "submission" / "solution_multi.py")
    parser.add_argument("--precision", choices=("f32", "f16", "i16"), default=None,
                        help="write precision.txt for the native solutions (default: solution's own default)")
    args = parser.parse_args()
    name, members = args.name, args.members
    pkg = ROOT / "submissions" / name / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    levels = [lvl for lvl in ("v2", "avx2", "v3") if (args.native_from / lvl).is_dir()]
    if not {"v2", "v3"} <= set(levels):
        raise SystemExit(f"{args.native_from} needs at least v2 and v3 builds")
    for lvl in levels:
        shutil.copytree(args.native_from / lvl, pkg / lvl)
    shutil.copy(args.solution, pkg / "solution.py")
    extra = []
    if args.precision:
        (pkg / "precision.txt").write_text(args.precision + "\n")
        extra.append("precision.txt")
    spec = []
    for i, m in enumerate(members):
        run, _, coef = m.rpartition(":")
        fname = f"weights_{i}.npz"
        export(run, pkg / fname)
        spec.append({"weights": fname, "coef": float(coef), "source": run})
    (pkg / "members.json").write_text(json.dumps(spec, indent=1))
    zp = ROOT / "submissions" / name / "submission.zip"
    zp.unlink(missing_ok=True)
    files = ["solution.py", "members.json", *extra, *levels] + [s["weights"] for s in spec]
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    print(zp, zp.stat().st_size, spec)


if __name__ == "__main__":
    main()
