from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import runpy
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET = Path("research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2")
PROPOSAL = PACKET / "H512_F1_CONDITIONAL_PROFILE_FIT_PROPOSAL_20261021_R1.json"
ADMISSION = PACKET / "root_admission.json"
ORIGINAL = Path("research/campaign_2026-09-29/artifacts/splits/fold1.json")
DERIVED = PACKET / "fold1_profile_view.json"
REGISTRY = PACKET / "fold1_portable_inputs.json"
PROFILE_NAME = "r01_h512_f1_s1004_kaggle_profile_r2"
FIT_NAME = "r01_h512_f1_s1004_kaggle_fit_r2"
SWA_NAME = "r01_h512_f1_s1004_kaggle_swa345_r2"
PROFILE_UPDATES = 40
FIT_BATCHES = 4_476
EPOCH_CHUNK_UPDATES = 7_460
TOTAL_CHUNK_UPDATES = 44_760
PROFILE_SLOWDOWN = 1.25
EXPORT_RESERVE_SECONDS = 2_700
FINALIZER_RESERVE_SECONDS = 300
PROVIDER_SAFETY_SECONDS = 1_200
OUTER_WALL_CAP_SECONDS = 15_500
WORKER_WALL_CAP_SECONDS = OUTER_WALL_CAP_SECONDS - FINALIZER_RESERVE_SECONDS
EXPECTED = {
    "original_manifest_sha256": "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae",
    "derived_manifest_sha256": "548b42f06710a0f7c06377b8f2aaf7241469d75f0dad9de05bd4fbc29448d0c3",
    "registry_sha256": "2dcd5cee727d1299f1fdab86b71116dc4a70008160c9495be8bf6f5c00b9dc67",
    "chains_sha256": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
    "chain_folds_sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    "proposal_sha256": "dc519df5c7f6a182bfe3d34c1d580580b3b8dad55e4144a8117f24e16208ad09",
}
CACHE_IDENTITIES = {
    Path("cache/chains.npz"): (650_142, EXPECTED["chains_sha256"]),
    Path("cache/chain_folds.npz"): (550_072, EXPECTED["chain_folds_sha256"]),
}
RAW_IDENTITIES = {
    "datasets/train.parquet": (
        29_240_219_011,
        "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    ),
    "datasets/valid.parquet": (
        5_615_509_128,
        "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    ),
}
RECEIPT_PATH = PACKET / "conditional_worker_receipt_r2.json"


def sha256(path: Path) -> str:
    """Hash a file in bounded memory without retaining its payload."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(event: str, started: float, **fields: object) -> None:
    """Write a flushed machine-readable phase record."""
    print(
        json.dumps(
            {
                "event": event,
                "observed_at_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                **fields,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def read_object(path: Path) -> dict[str, Any]:
    """Read a metadata object and reject other JSON roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


class ProfileCapture:
    """Tee trainer logs while retaining only the two bounded profile records."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.pending = ""
        self.records: dict[str, dict[str, Any]] = {}

    def write(self, value: str) -> int:
        """Forward a write and parse complete profile lines."""
        written = self.target.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._record(line)
        return written

    def flush(self) -> None:
        """Flush the original worker log stream."""
        self.target.flush()

    def _record(self, line: str) -> None:
        for prefix, key in (("PROFILE ", "profile"), ("PROFILE_UPDATE_EVIDENCE ", "updates")):
            if line.startswith(prefix):
                payload = json.loads(line[len(prefix) :])
                if not isinstance(payload, dict):
                    raise ValueError(f"Malformed trainer event: {key}")
                self.records[key] = payload


def validate_small_inputs(root: Path, started: float) -> dict[str, Any]:
    """Verify manifest/cache identities before the one canonical raw preflight."""
    observed = {
        "original_manifest_sha256": sha256(root / ORIGINAL),
        "derived_manifest_sha256": sha256(root / DERIVED),
        "registry_sha256": sha256(root / REGISTRY),
        "proposal_sha256": sha256(root / PROPOSAL),
    }
    for name, expected in EXPECTED.items():
        if name in observed and expected and observed[name] != expected:
            raise ValueError(f"Sealed packet identity changed: {name}")
    original = read_object(root / ORIGINAL)
    derived = read_object(root / DERIVED)
    if {**original, "sources": derived.get("sources")} != derived:
        raise ValueError("Derived manifest changed a non-source canonical field")
    if derived.get("sources") != {
        "datasets/train.parquet": original["sources"]["datasets/train.parquet"],
        "datasets/valid.parquet": original["sources"]["datasets/valid.parquet"],
    }:
        raise ValueError("Derived view raw sources differ from canonical F1 identities")
    registry = read_object(root / REGISTRY)
    if registry.get("split_manifest_sha256") != EXPECTED["original_manifest_sha256"]:
        raise ValueError("Portable registry is not pinned to canonical F1")
    for path, (size, digest) in CACHE_IDENTITIES.items():
        cache = root / path
        if cache.stat().st_size != size or sha256(cache) != digest:
            raise ValueError(f"Structural cache mismatch: {path}")
    portable = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if portable != REGISTRY.as_posix():
        raise ValueError("Portable registry environment differs from the F1 packet")
    mounts = {}
    for name, (size, digest) in RAW_IDENTITIES.items():
        path = root / name
        stat = path.stat()
        if not path.is_symlink() or stat.st_size != size:
            raise ValueError(f"Raw dataset mount size or symlink mismatch: {name}")
        entry = registry["sources"][name]
        if entry.get("sha256") != digest or entry["original_identity"].get("bytes") != size:
            raise ValueError(f"Raw dataset registry identity mismatch: {name}")
        mounts[name] = {"bytes": size, "sha256": digest, "sha256_check": "pending canonical loader"}
    cache_files = {
        str(path): {"bytes": size, "sha256": digest}
        for path, (size, digest) in CACHE_IDENTITIES.items()
    }
    emit(
        "sealed_inputs_verified",
        started,
        identities=observed,
        caches={str(p): d for p, (_, d) in CACHE_IDENTITIES.items()},
        raw_mounts=mounts,
    )
    return {**observed, "cache_files": cache_files, "raw_sources": mounts}


def run_cli(
    root: Path, argv: list[str], *, capture_profile: bool = False
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Run an existing canonical CLI in this process and return elapsed time/events."""
    previous_argv = sys.argv
    sys.argv = list(argv)
    capture = ProfileCapture(sys.stdout) if capture_profile else None
    began = time.monotonic()
    try:
        if capture is None:
            runpy.run_path(str(root / argv[0]), run_name="__main__")
        else:
            with contextlib.redirect_stdout(capture):
                runpy.run_path(str(root / argv[0]), run_name="__main__")
    finally:
        sys.argv = previous_argv
    return time.monotonic() - began, {} if capture is None else capture.records


def require_profile(records: dict[str, dict[str, Any]]) -> float:
    """Require the canonical three-batch profile and all 40 finite FP32 updates."""
    profile = records.get("profile")
    updates = records.get("updates")
    if profile is None or updates is None:
        raise ValueError("Canonical trainer omitted PROFILE or PROFILE_UPDATE_EVIDENCE")
    step_wall = profile.get("step_wall")
    if not isinstance(step_wall, (int, float)) or not math.isfinite(step_wall) or step_wall <= 0:
        raise ValueError("Profile step_wall is not finite and positive")
    required = {
        "valid_for_throughput": True,
        "optimizer_attempts": PROFILE_UPDATES,
        "optimizer_completed_updates": PROFILE_UPDATES,
        "amp_skipped_updates": 0,
        "loss_checks": PROFILE_UPDATES,
        "loss_finite": True,
        "gradient_finite": True,
        "gradient_stats_steps": PROFILE_UPDATES,
        "gradient_stats_nonfinite_steps": 0,
    }
    if any(updates.get(key) != value for key, value in required.items()):
        raise ValueError(f"Profile update evidence failed: {updates}")
    return float(step_wall) / 1000.0


def write_receipt(root: Path, receipt: dict[str, Any]) -> None:
    """Preserve one immutable worker receipt for this packet attempt."""
    path = root / RECEIPT_PATH
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite worker evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
