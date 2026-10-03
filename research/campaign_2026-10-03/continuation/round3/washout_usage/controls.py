from __future__ import annotations

import json

import numpy as np
import torch
from common import ROOT, donor, raw_window, rms, train


def advance(state: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = state.copy()
    values = []
    for row in raw:
        h[:-1] = h[1:]
        h[-1] = row[0]
        values.append(h[0])
    return np.array(values), h


def main() -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(614304)
    raw = rng.normal(size=(1600, 112)).astype(np.float32)
    packed = train.build_inputs(torch.from_numpy(raw)[None])[0].numpy()
    planted = []
    sham = []
    identity = []
    for t in [512, 768, 1024]:
        d = donor(raw, t, np.ones(211, dtype=np.float32))
        altered, window, gate = raw_window(raw, packed, d)
        assert gate["accepted"] and gate["suffix_max_abs"] == 0
        _, state = advance(np.zeros(161, dtype=np.float32), raw[: d["start"]])
        old, _ = advance(state, raw[d["start"] : t + 1])
        changed, _ = advance(state, altered[d["start"] : t + 1])
        original, _ = advance(np.zeros(161, dtype=np.float32), raw[: t + 1])
        planted.append(changed[-1] - old[-1])
        identity.append(old[-1] - original[-1])
        sham.append(altered[t, 0] - raw[t, 0])
        canary = raw.copy()
        canary[t + 1 :] = 0
        assert np.array_equal(
            train.build_inputs(torch.from_numpy(canary)[None])[0, : t + 1].numpy(), packed[: t + 1]
        )
    assert rms(planted) > 0.01 and rms(sham) == 0 and rms(identity) == 0
    result = dict(
        passed=True,
        seed=614304,
        accepted_toy_origins=3,
        feature_tol_atol=0,
        feature_tol_rtol=0,
        planted161state_history_response=rms(planted),
        memoryless_sham_response=rms(sham),
        identity_response=rms(identity),
        future_feature_canary=True,
    )
    (ROOT / "controls.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
