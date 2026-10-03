from __future__ import annotations

import json

import numpy as np
from common import dump, ema
from fit_proxy import experiment


def main() -> None:
    rng = np.random.default_rng(614404)
    component = np.repeat(np.arange(24), 12)
    latent = rng.normal(size=(288, 12))
    load = rng.normal(size=(12, 726))
    base = latent @ load + 0.05 * rng.normal(size=(288, 726))
    peer = []
    shift = []
    for _ in range(24):
        state = ema(rng.normal(size=4096), 16)
        q = 512 + 128 * np.arange(12)
        peer.extend(state[q])
        shift.extend(state[q - 256])
    peer = np.array(peer)
    shift = np.array(shift)
    age = np.tile(np.arange(12) % 2, 24)
    planted = 3 * peer + 0.2 * base[:, 0] + 0.1 * rng.normal(size=288)
    null = 0.5 * base[:, 0] + 0.1 * rng.normal(size=288)
    positive, _ = experiment(
        base, peer, shift, planted, component, age, list(range(16)), list(range(16, 24))
    )
    negative, _ = experiment(
        base, peer, shift, null, component, age, list(range(16)), list(range(16, 24))
    )
    result = dict(
        passed=positive["gate_passed"] and not negative["gate_passed"],
        seed=614404,
        positive=positive,
        negative=negative,
        scope="same726dimension block-balanced fit protocol; peer from causal EMA16 and256rowshift; known observable-proxy signal/absent signal, not market realism",
    )
    dump("controls.json", result)
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                positive_gate=positive["gate_passed"],
                null_gate=negative["gate_passed"],
                positive_mse=positive["arm_test_component_mse"],
                null_mse=negative["arm_test_component_mse"],
            )
        )
    )


if __name__ == "__main__":
    main()
