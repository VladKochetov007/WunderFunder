"""Precompute a frozen propensity net over every sequence once (train and valid), for --prop-cache.

    scripts/limited .venv/bin/python scripts/cache_propensity.py prop_v1 [--check 16]
Writes cache/propcache_<name>_<split>.npy of shape (row groups, 20000) float32 plus a .json sidecar
with the sha256 of the propensity weights; train_rnn.py refuses a cache whose sha differs. The
legacy profile preserves fp16 autocast; strict_fp32_v1 follows the checkpoint profile and records
split, fit-normalization, source, output-order and cache-byte identities. --check N recomputes N
random sequences in a training-shaped batch and reports the maximum absolute difference.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", SOURCE_ROOT)).resolve()
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
import train_rnn as T  # noqa: E402

from wnn.data import N_FEATURES, SequenceFile  # noqa: E402
from wnn.features import N_EVENT_FEATURES  # noqa: E402
from wnn.propensity import (  # noqa: E402
    LEGACY_PROFILE,
    STRICT_FP32_PROFILE,
    load_propensity,
    ordered_groups_sha256,
    propensity,
)
from wnn.resources import configure_strict_fp32, configure_torch  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402


class FeatureOnlySequences(Dataset):
    """Read input columns only, so strict inference never loads validation targets or masks."""

    def __init__(self, split: str, indices: np.ndarray) -> None:
        self.path = ROOT / "datasets" / f"{split}.parquet"
        self.indices = list(map(int, indices))
        self._file = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._file is None:
            self._file = SequenceFile(self.path)
        sequence = self._file.read(self.indices[index], targets=False, mask=False)
        return torch.from_numpy(sequence.features)


def _strict_output_paths(root: Path, name: str) -> list[tuple[str, Path, Path]]:
    paths = []
    for split in ("train", "valid"):
        output = root / "cache" / f"propcache_{name}_{split}.npy"
        paths.append((split, output, output.with_suffix(".json")))
    collisions = [
        path
        for _, output, metadata in paths
        for path in (output, metadata)
        if os.path.lexists(path)
    ]
    if collisions:
        raise FileExistsError(f"strict propensity cache output already exists: {collisions[0]}")
    return paths


def _reserve_temp(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _open_reserved_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:
    """Initialize and map only the regular file reserved by ``mkstemp``."""
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        os.close(descriptor)
        raise RuntimeError(f"reserved cache temporary changed before opening: {path}")
    with os.fdopen(descriptor, "r+b") as stream:
        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": shape,
        }
        np.lib.format.write_array_header_2_0(stream, header)
        offset = stream.tell()
        stream.truncate(offset + int(np.prod(shape)) * np.dtype(np.float32).itemsize)
        stream.flush()
        return np.memmap(stream, dtype=np.float32, mode="r+", offset=offset, shape=shape)


def _write_reserved_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_pair(temp_array: Path, temp_metadata: Path, output: Path, metadata: Path) -> None:
    """Publish both reserved files without replacing either final path."""
    with temp_array.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temp_array, output)
    try:
        os.link(temp_metadata, metadata)
    except Exception as exc:
        raise RuntimeError(
            f"strict cache array published at {output}, but sidecar publication failed; "
            f"preserved temporary evidence at {temp_array} and {temp_metadata}"
        ) from exc
    temp_array.unlink()
    temp_metadata.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--check", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split-manifest", type=Path, default=None)
    ap.add_argument("--loader-workers", type=int, default=None)
    a = ap.parse_args()
    if a.batch < 8:
        # cuDNN picks an fp32-like GRU path for tiny batches (batch 1-4 differ from batch >= 8 by up to 0.09);
        # training runs at batch >= 8, so the cache must be built in the same regime (measured 2026-09-27)
        ap.error("--batch must be >= 8 to match the training-time propensity values")
    configure_torch()
    dev = torch.device("cuda")
    wpath = ROOT / "artifacts" / "runs" / a.name / "model.pt"
    sha = file_sha256(wpath)
    model = load_propensity(wpath, dev)
    profile = getattr(model, "_propensity_profile", LEGACY_PROFILE)
    if a.loader_workers is None:
        a.loader_workers = 0 if profile == STRICT_FP32_PROFILE else 4
    if a.loader_workers < 0:
        ap.error("--loader-workers must be nonnegative")
    campaign_split = None
    checkpoint_provenance = getattr(model, "_propensity_provenance", {})
    if profile == STRICT_FP32_PROFILE:
        if a.split_manifest is None:
            ap.error("strict_fp32_v1 cache production requires --split-manifest")
        campaign_split = load_campaign_split(a.split_manifest, ROOT)
        if checkpoint_provenance.get("split_manifest_sha256") != campaign_split.sha256:
            ap.error("cache split manifest differs from the strict propensity checkpoint")
        if checkpoint_provenance.get("fit_row_groups", {}).get("valid") != list(
            campaign_split.valid_groups
        ):
            ap.error("strict propensity checkpoint fit membership differs from the manifest")
        if checkpoint_provenance.get("evaluation_row_groups") != list(
            campaign_split.evaluation_groups
        ):
            ap.error("strict propensity checkpoint evaluation membership differs from the manifest")
        configure_strict_fp32()
    elif a.split_manifest is not None:
        campaign_split = load_campaign_split(a.split_manifest, ROOT)
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if profile == STRICT_FP32_PROFILE:
        strict_paths = _strict_output_paths(ROOT, a.name)
        strict_work = {}
        for split, output, metadata in strict_paths:
            strict_work[split] = (
                output,
                metadata,
                _reserve_temp(output),
                _reserve_temp(metadata),
            )
    else:
        strict_work = {}
    for split in ("train", "valid"):
        n = len(SequenceFile(ROOT / "datasets" / f"{split}.parquet"))
        out_path = ROOT / "cache" / f"propcache_{a.name}_{split}.npy"
        metadata_path = out_path.with_suffix(".json")
        if profile == STRICT_FP32_PROFILE:
            _, _, work_path, metadata_temp = strict_work[split]
        else:
            work_path, metadata_temp = out_path, metadata_path
        if profile == STRICT_FP32_PROFILE:
            out = _open_reserved_memmap(work_path, (n, 20000))
        else:
            out = np.lib.format.open_memmap(
                work_path, mode="w+", dtype=np.float32, shape=(n, 20000)
            )
        if profile == STRICT_FP32_PROFILE:
            out[:] = 0.0
        idx = np.arange(n)
        if split == "train":
            idx = np.array([g for g in idx if g not in T.CORRUPT_TRAIN_ROW_GROUPS])
        ds = (
            FeatureOnlySequences(split, idx)
            if profile == STRICT_FP32_PROFILE
            else T.Sequences(split, idx)
        )
        loader_options = {"num_workers": a.loader_workers, "pin_memory": True}
        if a.loader_workers:
            loader_options["prefetch_factor"] = 2
        loader = DataLoader(ds, batch_size=a.batch, **loader_options)
        t0, pos = time.time(), 0
        for batch in loader:
            x = batch if profile == STRICT_FP32_PROFILE else batch[0]
            inputs = T.build_inputs(x.to(dev, non_blocking=True))
            p = propensity(model, inputs[..., : N_FEATURES + N_EVENT_FEATURES]).cpu().numpy()
            if profile == STRICT_FP32_PROFILE and (
                not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            ):
                raise FloatingPointError(
                    "strict propensity cache batch is not finite probability data"
                )
            out[idx[pos : pos + len(p)]] = p
            pos += len(p)
        out.flush()
        del out
        metadata = {
            "model": a.name,
            "model_sha256": sha,
            "split": split,
            "rows": int(n),
            "n_filled": int(len(idx)),
            "seconds": round(time.time() - t0, 1),
        }
        if profile == STRICT_FP32_PROFILE:
            metadata.update(
                {
                    "schema": "wnn.propensity_cache.strict_fp32.v1",
                    "precision_profile": STRICT_FP32_PROFILE,
                    "inference_precision": "float32_no_autocast_tf32_disabled",
                    "split_manifest_sha256": campaign_split.sha256,
                    "fit_row_groups_sha256": checkpoint_provenance["fit_row_groups_sha256"][
                        "valid"
                    ],
                    "fit_row_groups_count": len(checkpoint_provenance["fit_row_groups"]["valid"]),
                    "ordered_output_row_groups_sha256": ordered_groups_sha256(range(n)),
                    "populated_row_groups_sha256": ordered_groups_sha256(idx),
                    "checkpoint_normalization": checkpoint_provenance["normalization"],
                    "cache_shape": [int(n), 20000],
                    "cache_dtype": "float32",
                    "cache_sha256": file_sha256(work_path),
                    "source_sha256": {
                        **checkpoint_provenance["source_sha256"],
                        "scripts/cache_propensity.py": file_sha256(
                            SOURCE_ROOT / "scripts/cache_propensity.py"
                        ),
                    },
                }
            )
        if profile == STRICT_FP32_PROFILE:
            _write_reserved_json(metadata_temp, metadata)
            _publish_pair(work_path, metadata_temp, out_path, metadata_path)
        else:
            with metadata_path.open("w") as stream:
                json.dump(metadata, stream, sort_keys=True)
        print(split, "rows", n, "filled", len(idx), f"{time.time() - t0:.0f}s", flush=True)
        # parity: recompute random sequences in a fresh batch of a different composition
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, min(a.check, len(idx)), replace=False)
        cache = np.load(out_path, mmap_mode="r")
        if profile == STRICT_FP32_PROFILE:
            x = torch.stack([FeatureOnlySequences(split, np.asarray([g]))[0] for g in pick])
        else:
            x = torch.stack([torch.from_numpy(T.Sequences(split, [g])[0][0].numpy()) for g in pick])
        fresh = (
            propensity(model, T.build_inputs(x.to(dev))[..., : N_FEATURES + N_EVENT_FEATURES])
            .cpu()
            .numpy()
        )
        cached = np.asarray(cache[pick])
        if profile == STRICT_FP32_PROFILE and (
            not np.isfinite(fresh).all()
            or not np.isfinite(cached).all()
            or np.any((fresh < 0) | (fresh > 1))
            or np.any((cached < 0) | (cached > 1))
        ):
            raise FloatingPointError("strict propensity cache parity values are invalid")
        d = np.abs(fresh - cached)
        print(
            f"{split} parity on {len(pick)} random sequences: max abs diff {d.max():.3g}, mean {d.mean():.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
