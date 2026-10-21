"""Frozen package and prediction artifact identities for the F2 comparison."""

from __future__ import annotations

CANDIDATES: dict[str, dict[str, str]] = {
    "tl3": {
        "zip": "submissions/team_tl3_original_20261018.zip",
        "zip_sha256": "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569",
        "ancestry": "unknown; package metadata calls its parents all-data",
        "prediction": "tl3_f2.npy",
        "sidecar": "tl3_f2.sidecar.json",
        "receipt": "tl3_f2.replay.json",
    },
    "r3": {
        "zip": "submissions/geometry_r3_reverse_half_q16x_20261018_v2.zip",
        "zip_sha256": "27f01f53ba32f7256ce1663e8b163b382e673fc96e7870fdebdf310d4886f98e",
        "ancestry": "inference-only derivative of the R2 checkpoint, adaptively selected on exposed F0",
        "prediction": "r3_f2.npy",
        "sidecar": "r3_f2.sidecar.json",
        "receipt": "r3_f2.replay.json",
    },
}
