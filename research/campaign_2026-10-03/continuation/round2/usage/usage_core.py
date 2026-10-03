from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
QUERIES = np.arange(511, 4096, 64)
LAGS = np.array([0, 1, 4, 16, 32])
N_ROWS = int(QUERIES[-1] + 33)
NAMES = ["identity", "i0_past256", "i1_past256", "i0_current", "i1_current", "i0_past32"]
I0 = np.r_[0:52, 112:156, 178:183, 186:188, 189:211]
I1 = np.r_[52:104, 156:178, 183:186, 188:189]


def update_state(phase: str, detail: dict | None = None) -> None:
    """Persist the instrument phase within its own write scope."""
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = phase
    if detail:
        state.update(detail)
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))


def intervention_window(inputs: np.ndarray, query: int, name: str) -> tuple[int, np.ndarray]:
    """Create a prefix-compatible instrument-space replay window.

    Donors always precede the replaced row. History changes stop strictly before
    the query; its whole original current input is preserved. Future rows return
    to original inputs, while the altered recurrent state propagates normally.
    """
    if name not in NAMES or query < 320 or query + 33 > len(inputs):
        raise ValueError("Unknown intervention or insufficient prefix/window")
    length = 32 if name == "i0_past32" else 256
    start = query if name.endswith("current") else query - length
    window = inputs[start : query + 33].copy()
    if name == "identity":
        return start, window
    group = I0 if name.startswith("i0") else I1
    if name.endswith("current"):
        window[0, group] = inputs[query - 64, group]
    else:
        window[:length, group] = inputs[start - 64 : query - 64, group]
        if not np.array_equal(window[length], inputs[query]):
            raise AssertionError("History intervention changed current input")
    return start, window


def cached_array_prefix(field: str, position: int, rows: int) -> np.ndarray:
    """Read one stored NPZ member's sequence prefix without materializing its file."""
    path = REPO / "cache/valid_targets.npz"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{field}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Bounded cache reader requires ZIP_STORED NPZ")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            header_bytes = stream.tell()
    if fortran or shape[1] != 20000:
        raise ValueError("Unexpected cached sequence layout")
    width = int(np.prod(shape[2:]))
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        local = stream.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local zip header")
        name_length, extra_length = struct.unpack_from("<HH", local, 26)
        payload = info.header_offset + 30 + name_length + extra_length + header_bytes
        stream.seek(payload + position * 20000 * width * dtype.itemsize)
        values = np.frombuffer(stream.read(rows * width * dtype.itemsize), dtype=dtype).copy()
    return values.reshape((rows,) + shape[2:])


def quiet_ages(raw: np.ndarray) -> np.ndarray:
    """Count elapsed rows since any of the 22 i0 price identifiers changed."""
    changed = np.r_[False, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
    rows = np.arange(len(raw))
    return rows - np.maximum.accumulate(np.where(changed, rows, 0))
