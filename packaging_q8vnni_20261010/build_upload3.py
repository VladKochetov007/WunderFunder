"""Assemble the 2026-10-10b candidate ZIPs from existing audited weights and the new native libraries.

Run from the worktree root.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

W = Path.cwd()
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
S = MAIN / "submissions"
NATIVE = W / "packaging_q8vnni_20261010" / "native2"
OUT = W / "submissions" / "upload_2026-10-11"
SOL = W / "submission" / "solution_native.py"


def build(name, members, precision, solution=SOL, native=True):
    pkg = OUT / name / "pkg"
    shutil.rmtree(OUT / name, ignore_errors=True)
    pkg.mkdir(parents=True)
    files = ["solution.py"]
    shutil.copy(solution, pkg / "solution.py")
    if members:
        spec = []
        for i, (src, coef, label) in enumerate(members):
            shutil.copy(src, pkg / f"weights_{i}.npz")
            spec.append({"weights": f"weights_{i}.npz", "coef": coef, "source": label})
            files.append(f"weights_{i}.npz")
        (pkg / "members.json").write_text(json.dumps(spec, indent=1))
        (pkg / "precision.txt").write_text(precision + "\n")
        files += ["members.json", "precision.txt"]
    if native:
        for lvl in ("v2", "avx2", "v3"):
            shutil.copytree(NATIVE / lvl, pkg / lvl)
            files.append(lvl)
    zp = OUT / name / "submission.zip"
    subprocess.run([sys.executable, "-m", "zipfile", "-c", str(zp), *files], cwd=pkg, check=True)
    print(name, zp.stat().st_size, hashlib.sha256(zp.read_bytes()).hexdigest())


S = MAIN / "submissions"
z1 = S / "c20261008_z1c_combo_i16/pkg/weights_0.npz"
z4 = S / "c20261009_z4_combo_i16/pkg/weights_0.npz"
l1 = json.loads((S / "c20261008_z1c_combo_i16/pkg/members.json").read_text())[0]["source"]
l4 = json.loads((S / "c20261009_z4_combo_i16/pkg/members.json").read_text())[0]["source"]
build("z1c_z4_pair_q8vnni", [(z1, 0.5, l1), (z4, 0.5, l4)], "q8vnni")
build("z1c_q8vnni_single", [(z1, 1.0, l1)], "q8vnni")
build("z1c_z4_pair_i16", [(z1, 0.5, l1), (z4, 0.5, l4)], "i16")
build("z1c_i16_single", [(z1, 1.0, l1)], "i16")
