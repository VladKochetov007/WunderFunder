"""Derive a family-head-free 'core' run from a distilled student checkpoint.

    python make_core_checkpoint.py SRC_RUN EPOCH DST_RUN

The H230 students append 8 teacher-family rows to head.2 (31 = 23 base + 8 family).
Serving and head-only fine-tunes use only the base rows, so the core checkpoint keeps
head.2[:23] and clears the family-head arguments; every other tensor is copied bitwise.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE_ROWS = 23

src, epoch, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
src_dir, dst_dir = ROOT / "artifacts/runs" / src, ROOT / "artifacts/runs" / dst
if dst_dir.exists():
    raise FileExistsError(dst_dir)
ck = torch.load(src_dir / f"model_ep{epoch}.pt", map_location="cpu", weights_only=False)
state = ck["model"]
rows = state["head.2.weight"].shape[0]
if rows != BASE_ROWS + 8:
    raise ValueError(f"expected {BASE_ROWS + 8} head rows, found {rows}")
state["head.2.weight"] = state["head.2.weight"][:BASE_ROWS].clone()
state["head.2.bias"] = state["head.2.bias"][:BASE_ROWS].clone()
for holder in (ck.get("args"), (ck.get("provenance") or {}).get("args")):
    if isinstance(holder, dict):
        holder["distill_family_head"] = False
        holder["distill_family_weight"] = 0.0
ck["core_derivation"] = {"source_run": src, "source_epoch": epoch, "dropped_family_rows": 8}
dst_dir.mkdir(parents=True)
torch.save(ck, dst_dir / f"model_ep{epoch}.pt")
prov = json.loads((src_dir / "provenance.json").read_text())
prov["core_derivation"] = ck["core_derivation"]
if isinstance(prov.get("args"), dict):
    prov["args"]["distill_family_head"] = False
    prov["args"]["distill_family_weight"] = 0.0
(dst_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
print(json.dumps({"dst": dst, "head_rows": BASE_ROWS, "source_rows": rows}))
