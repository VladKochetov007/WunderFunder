from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
FROZEN = OWN.parents[1] / "round2" / "first_passage"
PRICE_COLUMNS = [f"i0_p{index}" for index in range(22)]
QUANTA = (0.12543, 0.24133, 0.46854, 0.65687)
ENDPOINT = np.float32(-5.199337482452393)
TOLERANCE = 0.002


def digest(path: Path) -> str:
    """Hash small permitted artifacts without scanning whole data files."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def save_json(name: str, data: object) -> None:
    """Write strict JSON within the independent reproducer's write scope."""
    (OWN / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def phase(name: str, **fields: object) -> None:
    """Preserve command/sample state while recording a phase transition."""
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(fields)
    state["phase"] = name
    save_json("STATE.json", state)


def price_code(prices: np.ndarray) -> str:
    """Implement the documented rounding/byte grammar independently."""
    numeric = np.asarray(prices, dtype=np.float32)
    rounded = np.around(numeric, decimals=4)
    return np.asarray(rounded, dtype="<f4", order="C").tobytes().hex()
