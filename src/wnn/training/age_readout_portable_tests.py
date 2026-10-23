"""Synthetic byte-order and mmap-release checks for the portable cache builder."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .age_readout_artifacts import file_sha256
from .age_readout_cohort import ROWS_PER_SEQUENCE, WIDTH
from .age_readout_portable_builder import (
    read_npy_header,
    rebuild_cache_file,
    release_sequence_pages,
)


def check_portable_rebuild_and_page_release(tmp_root: Path) -> None:
    """Check exact fit/eval bytes and safe page release using one synthetic sequence."""
    with TemporaryDirectory(dir=tmp_root) as directory:
        root = Path(directory)
        raw = np.zeros((1, ROWS_PER_SEQUENCE, 112), dtype="<f4")
        raw[:, :, 0] = np.arange(ROWS_PER_SEQUENCE, dtype=np.float32)
        suffix = np.ones((1, ROWS_PER_SEQUENCE, 5), dtype="<f4")
        reference = np.concatenate((raw, suffix), axis=-1)
        original_path = root / "original.npy"
        np.save(original_path, reference, allow_pickle=False)
        header, _, _ = read_npy_header(original_path)
        original_sha = file_sha256(original_path)
        suffix_path = root / "suffix.npy"
        np.save(suffix_path, suffix, allow_pickle=False)
        suffix_header, suffix_shape, _ = read_npy_header(suffix_path)
        destination = root / "rebuilt.npy"
        rebuild_cache_file(
            destination,
            header,
            [{"split": "train", "row_group": 3}],
            lambda _split, _row_group: raw[0],
            suffix_path,
            file_sha256(suffix_path),
            suffix_shape,
            len(suffix_header),
            original_sha,
            evaluation=False,
        )
        if file_sha256(destination) != original_sha:
            raise AssertionError("portable fit cache reconstruction changed exact NPY bytes")
        _check_evaluation_assembly(root, raw)
        _check_page_release(root)


def _check_evaluation_assembly(root: Path, raw: np.ndarray) -> None:
    parent = np.full((1, ROWS_PER_SEQUENCE, 2), np.float32(0.25), dtype="<f4")
    expected = np.zeros((1, ROWS_PER_SEQUENCE, WIDTH), dtype="<f4")
    expected[:, :, :112] = raw
    expected[:, :, 115:117] = parent
    expected_path = root / "eval_original.npy"
    suffix_path = root / "parent.npy"
    np.save(expected_path, expected, allow_pickle=False)
    np.save(suffix_path, parent, allow_pickle=False)
    header, _, _ = read_npy_header(expected_path)
    suffix_header, suffix_shape, _ = read_npy_header(suffix_path)
    destination = root / "eval_rebuilt.npy"
    rebuild_cache_file(
        destination,
        header,
        [{"split": "valid", "row_group": 9}],
        lambda _split, _row_group: raw[0],
        suffix_path,
        file_sha256(suffix_path),
        suffix_shape,
        len(suffix_header),
        file_sha256(expected_path),
        evaluation=True,
    )
    rebuilt = np.load(destination, mmap_mode="r", allow_pickle=False)
    if np.any(rebuilt[0, :, 112:115]) or not np.array_equal(rebuilt[0, :, 115:117], parent[0]):
        raise AssertionError("portable evaluation cache did not zero labels or preserve parent")
    rebuilt._mmap.close()


def _check_page_release(root: Path) -> None:
    mapped = np.lib.format.open_memmap(
        root / "mapped.npy", mode="w+", dtype="<f4", shape=(1, ROWS_PER_SEQUENCE, WIDTH)
    )
    mapped[0, 10, 0] = np.float32(0.75)
    mapped.flush()
    release_sequence_pages(mapped, 0)
    if mapped[0, 10, 0] != np.float32(0.75):
        raise AssertionError("mmap page release changed cache values")
    mapped._mmap.close()
