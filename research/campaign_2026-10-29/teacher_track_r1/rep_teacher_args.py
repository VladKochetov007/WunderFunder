"""Print exact PR49 recipe args for one node, renamed tl3c_ -> tt29rep_ (local reproduction).

Usage: rep_teacher_args.py NODE_ID   (fin256_s3 | fin256_s4 | fbc1k_s5)
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/vlad/development/wunder/research_tmp/pr49_wt/scripts")
from tl3_chronological_recipes import build_nodes  # noqa: E402

SPLIT = Path(__file__).resolve().parent / "chronological_primary.json"
node = {n["id"]: n for n in build_nodes(SPLIT, Path("artifacts/runs"))}[sys.argv[1]]
args = [a.replace("tl3c_", "tt29rep_") for a in node["args"]]
print(" ".join(args))
