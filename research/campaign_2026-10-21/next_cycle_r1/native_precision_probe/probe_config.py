"""Pinned paths and identities for the bounded native precision probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRATCH = Path("/home/vlad/development/wunder/research_tmp")
REPLAY = ROOT / "research/campaign_2026-10-18/online_features_r1/tl3_replay"
ORIGINAL_ZIP = Path(
    "/home/vlad/development/wunder/wnn_connectome_starterpack/submissions/team_tl3_original_20261018.zip"
)
CP5_ZIP = SCRATCH / "h512-tl3-endpoint-screen-r1-package-cp5/submission.zip"
ORIGINAL_ZIP_SHA256 = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
CP5_ZIP_SHA256 = "ff5d5bdade2d024e0c2bf98237024cef4569664bad0fa3c99d4686e9621f54e2"
NATIVE_GRADE = ROOT / "research/campaign_2026-10-21/next_cycle_r1/NATIVE_GRADE_R3.json"
NATIVE_GRADE_SHA256 = "872ae52f0c83611d390e778d9666f910773e8d24e6494c12cb56fd8384788e35"
ORIGINAL_CACHE = ROOT / "cache/pred_tl3_unknown_ancestry_f0_valid.npy"
ORIGINAL_CACHE_SHA256 = "3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3"
ORIGINAL_SIDECAR_SHA256 = "b9d8b7371dbff016d48e155da154cd87d85c887cb08921e86caabba082059910"
CP5_CACHE = SCRATCH / "h512_tl3_endpoint_cp5_f0.npy"
CP5_CACHE_SHA256 = "2a4d95a332f12f7787a75fbe0acc66811d938d614c203fee28dcc7735611cfe2"
CP5_SIDECAR_SHA256 = "0055bbfc4fc8c9bf4537746b1e2f86b82b8e6e8838b787ebc906705d96b34696"
TORCH_CACHE = ROOT / "cache/pred_r01_h512_f0_s1003_local_cp5_r1_valid.npy"
TORCH_CACHE_SHA256 = "14cf538ee27dd44420f3b682dc449766bda1fe37bcc377f21ff81d10aa59f240"
TORCH_CHECKPOINT_SHA256 = "4311fbb71d1ba121a285fefe06b431f9fa2c16f7fabd23f18b72135180bffe2f"
TARGET_CACHE = ROOT / "cache/valid_targets.npz"
TARGET_CACHE_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Create deterministic JSON evidence without replacement."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
