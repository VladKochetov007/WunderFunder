"""Source and physical-storage binding for bounded geometry experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

SCRATCH = Path("/home/vlad/development/wunder/research_tmp")
REFERENCE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")


def validate_storage(output: Path, required_bytes: int = 3_000_000_000) -> None:
    """Require physical approved scratch and enough free disk for the staged pilot."""
    resolved = output.resolve()
    if not resolved.is_relative_to(SCRATCH) or not SCRATCH.is_dir():
        raise ValueError("staging must use approved physical research scratch")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1])
        if resolved.is_relative_to(mount):
            mounts.append((len(str(mount)), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("staging filesystem is RAM-backed or unverified")
    if shutil.disk_usage(SCRATCH).free < required_bytes:
        raise ValueError("insufficient physical staging space")


def bind_sources(root: Path, membership: dict[str, object]) -> dict[str, object]:
    """Check trusted footer/index identities and bind every selected sequence id."""
    reference = root / REFERENCE
    manifest = json.loads(reference.read_text())
    sources = {}
    ids = {}
    for split in ("train", "valid"):
        name = f"datasets/{split}.parquet"
        path, identity = root / name, manifest["sources"][name]
        stat = path.stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"changed immutable source: {name}")
        remaining = identity["footer_bytes"] + 8
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(-remaining, 2)
            while remaining:
                chunk = stream.read(min(1 << 20, remaining))
                if not chunk:
                    raise ValueError("truncated parquet footer")
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != identity["footer_sha256"]:
            raise ValueError(f"footer identity mismatch: {name}")
        sources[name] = identity
        cache_name = f"cache/{split}_targets.npz"
        cache = root / cache_name
        expected = manifest["sources"][cache_name]
        if (
            cache.stat().st_size != expected["bytes"]
            or cache.stat().st_mtime_ns != expected["mtime_ns"]
        ):
            raise ValueError("changed target index source")
        with np.load(cache, allow_pickle=False) as archive:
            group, seq_ix = archive["row_group"], archive["seq_ix"]
        actual = hashlib.sha256(
            np.asarray(group, dtype="<i8").tobytes() + np.asarray(seq_ix, dtype="<i8").tobytes()
        ).hexdigest()
        if actual != expected["index_sha256"]:
            raise ValueError("sequence identity index changed")
        ids[split] = dict(zip(map(int, group), map(int, seq_ix), strict=True))
    for name, expected in membership["cache_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("membership cache changed")
        if expected != manifest["sources"][name]["sha256"]:
            raise ValueError("chain graph does not match trusted source revision")
    mapping = {
        role: [
            {"source": s, "row_group": g, "seq_ix": ids[s][g]} for s, g, _, _ in membership[role]
        ]
        for role in ("fit", "eval")
    }
    return {
        "reference": str(REFERENCE),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "sources": sources,
        "selected_sequences": mapping,
    }


def verify_staged(root: Path, output: Path, binding: dict[str, object]) -> None:
    """Independently compare staged identity, a raw channel, both targets and masks to sources."""
    import pyarrow.parquet as pq

    for split in ("train", "valid"):
        reader = pq.ParquetFile(root / "datasets" / f"{split}.parquet")
        columns = ["seq_ix", "i0_p0", "t0", "t1", "need_prediction"]
        if split == "valid":
            columns.append("is_scored")
        for role in ("fit", "eval"):
            raw = np.load(output / f"{role}_raw.npy", mmap_mode="r")
            targets = np.load(output / f"{role}_y.npy", mmap_mode="r")
            masks = np.load(output / f"{role}_mask.npy", mmap_mode="r")
            for index, identity in enumerate(binding["selected_sequences"][role]):
                if identity["source"] != split:
                    continue
                table = reader.read_row_group(
                    identity["row_group"], columns=columns, use_threads=False
                )
                if not np.all(table["seq_ix"].to_numpy() == identity["seq_ix"]):
                    raise ValueError("actual source sequence id differs from trusted index")
                if not np.array_equal(raw[index, :, 0], table["i0_p0"].to_numpy()):
                    raise ValueError("staged raw channel differs")
                if not np.array_equal(
                    targets[index], np.column_stack([table[c].to_numpy() for c in ("t0", "t1")])
                ):
                    raise ValueError("staged targets differ")
                mask = table["need_prediction"].to_numpy().copy()
                if role == "eval":
                    mask &= table["is_scored"].to_numpy()
                if not np.array_equal(masks[index], mask):
                    raise ValueError("staged prediction mask differs")
        reader.close()
        del reader
