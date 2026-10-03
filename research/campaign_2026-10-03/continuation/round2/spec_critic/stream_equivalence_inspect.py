from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
OWN = Path(__file__).resolve().parent
checkpoint = ROOT / "artifacts/runs/c20261001_d256_s3_f0/model_ep3.pt"
ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
manifest_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
manifest = json.loads(manifest_path.read_text())
summary = {
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "checkpoint_keys": list(ck),
    "epoch": ck.get("epoch"),
    "n_in": ck.get("n_in"),
    "args": ck["args"],
    "provenance_keys": list(ck.get("provenance", {})),
    "provenance": ck.get("provenance", {}),
    "manifest_keys": list(manifest),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "readout": ck.get("readout") is not None,
}
(OWN / "stream_equivalence_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
print(
    "fit/eval provenance:",
    ck.get("provenance", {}).get("split_manifest_sha256"),
    {k: len(v) for k, v in ck.get("provenance", {}).get("fit_row_groups", {}).items()},
)
print(
    "manifest membership:",
    {k: manifest[k] for k in manifest if k not in {"sequences", "sources", "fit", "excluded"}},
)
