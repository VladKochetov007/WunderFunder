"""Package a submission ZIP that runs the C++ engine (native_cpp, module wnn_cpp).

    .venv/bin/python native_cpp/package_cpp.py NAME RUN:COEF [RUN:COEF ...] [--python PY311]

RUN is a training run in artifacts/runs (model.pt; RUN@epN for model_epN.pt). Weights are exported
exactly like scripts/package_submission.py (head sliced to the two targets). The cp311 builds for the
v2 / avx2 / v3 CPU levels are (re)built with native_cpp/build.sh and copied next to
submission/solution_cpp.py, which picks the level at import time and runs every member with the
runtime governor. Writes submissions/NAME/pkg and submissions/NAME/NAME.zip.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from package_submission import export  # noqa: E402

LEVELS = ("v2", "avx2", "v3")
MAX_ZIP_BYTES = 20 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("members", nargs="+")
    ap.add_argument("--python", default=str(Path.home() / "venv311" / "bin" / "python"),
                    help="Python 3.11 with pybind11 (the scorer's interpreter version)")
    args = ap.parse_args()

    subprocess.run([str(ROOT / "native_cpp" / "build.sh"), args.python, *LEVELS], check=True)
    pkg = ROOT / "submissions" / args.name / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    for level in LEVELS:
        built = list((ROOT / "native_cpp" / "build" / level).glob("wnn_cpp.cpython-311-*.so"))
        if len(built) != 1:
            raise SystemExit(f"expected one cp311 build for {level}, found {built}")
        (pkg / level).mkdir()
        shutil.copy(built[0], pkg / level / built[0].name)
    shutil.copy(ROOT / "submission" / "solution_cpp.py", pkg / "solution.py")
    spec = []
    for i, member in enumerate(args.members):
        run, _, coef = member.rpartition(":")
        fname = f"weights_{i}.npz"
        export(run, pkg / fname)
        spec.append({"weights": fname, "coef": float(coef), "source": run})
    (pkg / "members.json").write_text(json.dumps(spec, indent=1))
    zp = ROOT / "submissions" / args.name / f"{args.name}.zip"
    zp.unlink(missing_ok=True)
    files = ["solution.py", "members.json", *LEVELS, *(s["weights"] for s in spec)]
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    size = zp.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise SystemExit(f"{zp} is {size} bytes, over the 20 MB limit")
    print(zp, size, spec)


if __name__ == "__main__":
    main()
