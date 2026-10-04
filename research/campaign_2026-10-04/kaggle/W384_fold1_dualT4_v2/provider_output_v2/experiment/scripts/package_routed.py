"""Package a market-class-routed native ensemble (submission/solution_routed.py).

    .venv/bin/python scripts/package_routed.py NAME --native-from submissions/_native_ens3 \
        --group a=RUN,RUN,RUN,RUN --group b=RUN,RUN,RUN,RUN --route 0=a,1=a,2=a,3=b \
        --detector RUN

Members of a group get equal weights; a run used by several groups is stored once. The detector
is exported with its full head (the auxiliary market-class logits sit at [17, 21) for runs with
--aux-struct and the default target layout).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from package_submission import ROOT, export

LEVELS = ("v2", "avx2", "v3")


def export_full(run: str, path: Path) -> int:
    import torch

    name, _, epoch = run.partition("@ep")
    ckpt = f"model_ep{int(epoch)}.pt" if epoch else "model.pt"
    ck = torch.load(ROOT / "artifacts" / "runs" / name / ckpt, map_location="cpu", weights_only=False)
    state = {k: v.float().numpy() for k, v in ck["model"].items()}
    np.savez(path, **state)
    return int(state["head.2.weight"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--group", action="append", required=True, help="NAME=RUN,RUN,...")
    parser.add_argument("--route", required=True, help="0=a,1=a,2=a,3=b")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--class-slice", default="17,21")
    parser.add_argument("--from-row", type=int, default=50)
    parser.add_argument("--native-from", type=Path, required=True)
    parser.add_argument("--solution", type=Path, default=ROOT / "submission" / "solution_routed.py")
    parser.add_argument("--precision", choices=("f32", "f16", "i16"), default=None)
    args = parser.parse_args()

    pkg = ROOT / "submissions" / args.name / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    levels = [lvl for lvl in LEVELS if (args.native_from / lvl).is_dir()]
    if not {"v2", "v3"} <= set(levels):
        raise SystemExit(f"{args.native_from} needs at least v2 and v3 builds")
    for lvl in levels:
        shutil.copytree(args.native_from / lvl, pkg / lvl)
    shutil.copy(args.solution, pkg / "solution.py")

    files: dict[str, str] = {}
    groups = {}
    for spec in args.group:
        gname, _, runs = spec.partition("=")
        members = []
        runs_list = runs.split(",")
        for run in runs_list:
            if run not in files:
                files[run] = f"w{len(files)}.npz"
                export(run, pkg / files[run])
            members.append({"weights": files[run], "coef": 1.0 / len(runs_list), "source": run})
        groups[gname] = members
    route = dict(part.split("=") for part in args.route.split(","))
    if set(route.values()) - set(groups):
        raise SystemExit(f"route names unknown groups: {route}")
    lo, hi = (int(v) for v in args.class_slice.split(","))
    n_out = export_full(args.detector, pkg / "detector.npz")
    if hi > n_out:
        raise SystemExit(f"detector head has {n_out} outputs, class slice {lo}:{hi} out of range")
    cfg = {
        "groups": groups,
        "class_group": route,
        "detector": {"weights": "detector.npz", "lo": lo, "hi": hi, "from_row": args.from_row,
                     "source": args.detector},
    }
    (pkg / "routing.json").write_text(json.dumps(cfg, indent=1))
    zp = ROOT / "submissions" / args.name / "submission.zip"
    zp.unlink(missing_ok=True)
    content = ["solution.py", "routing.json", "detector.npz", *levels, *files.values()]
    if args.precision:
        (pkg / "precision.txt").write_text(args.precision + "\n")
        content.append("precision.txt")
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *content], cwd=pkg, check=True)
    print(zp, zp.stat().st_size, json.dumps(route), {g: len(m) for g, m in groups.items()})


if __name__ == "__main__":
    main()
