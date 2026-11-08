from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from collect_l4_readouts import collect as collect_r1
from collect_l4_readouts_r2 import CELLS, MAX_FILE, MAX_RESPONSE, collect
from l4_readout_mirror_r2 import apply_readouts


def test_large_receipts_are_batched_with_exact_bytes_under_response_bound(tmp_path: Path) -> None:
    """The real indented ancestry exceeds1MiB; receive several immutable receipts over deltas."""
    native, local = tmp_path / "native", tmp_path / "mirror"
    raw = (json.dumps({"complete": True, "ancestry_fixture": "x" * 1_500_000}) + "\n").encode()
    for cell in CELLS[:5]:
        path = native / cell / "readouts/epoch6/readout.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
    with pytest.raises(ValueError, match="bounded"):
        collect_r1(str(native), {})
    known, batches = {}, 0
    while len(known) < 5:
        value = collect(str(native), known)
        assert len(json.dumps(value).encode()) <= MAX_RESPONSE
        assert value["files"]
        known.update(apply_readouts(value, local))
        batches += 1
        assert batches <= 3
    assert batches > 1
    assert set(known.values()) == {hashlib.sha256(raw).hexdigest()}
    assert all((local / name).read_bytes() == raw for name in known)


def test_four_mib_limit_and_eight_mib_total_remain_finite(tmp_path: Path) -> None:
    assert MAX_FILE == 4 * 1_048_576
    assert MAX_RESPONSE == 8 * 1_048_576
    path = tmp_path / CELLS[0] / "readouts/swa3456/readout.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (MAX_FILE + 1))
    with pytest.raises(ValueError, match="bounded"):
        collect(str(tmp_path), {})
