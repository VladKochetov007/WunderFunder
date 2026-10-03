from __future__ import annotations

import json
import logging

import numpy as np
from usage_core import NAMES, ROOT, intervention_window, update_state


def main() -> None:
    """Discriminate a planted memory reader from a current-only sham."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    update_state("synthetic_controls_running")
    rng = np.random.default_rng(614103)
    inputs = rng.standard_normal((1600, 211)).astype(np.float32)
    queries = np.arange(511, 1408, 64)
    effects = {name: {"planted": [], "sham": []} for name in NAMES}
    for query in queries:
        original = float(0.4 * inputs[query - 80, 0] + 0.2 * inputs[query, 52])
        sham_original = float(0.2 * inputs[query, 52])
        for name in NAMES:
            start, window = intervention_window(inputs, int(query), name)
            # The toy reader still receives unchanged earlier prefix when its
            # planted lag lies outside the instrument's alteration window.
            restored = inputs[: query + 33].copy()
            restored[start:] = window
            prediction = float(0.4 * restored[query - 80, 0] + 0.2 * restored[query, 52])
            sham_prediction = float(0.2 * restored[query, 52])
            effects[name]["planted"].append(prediction - original)
            effects[name]["sham"].append(sham_prediction - sham_original)
            changed_suffix = inputs.copy()
            changed_suffix[query + 33 :] *= -17
            _, test_window = intervention_window(changed_suffix, int(query), name)
            if not np.array_equal(window, test_window):
                raise AssertionError("Future canary changed replay window")
    summary = {
        name: {
            model: float(np.sqrt(np.mean(np.square(values)))) for model, values in models.items()
        }
        for name, models in effects.items()
    }
    assert summary["i0_past256"]["planted"] > 0.1
    assert summary["i1_current"]["planted"] > 0.1
    for name in ["identity", "i1_past256", "i0_current", "i0_past32"]:
        assert summary[name]["planted"] == 0
    for name in ["identity", "i0_past256", "i1_past256", "i0_current", "i0_past32"]:
        assert summary[name]["sham"] == 0
    output = {
        "passed": True,
        "rms_responses": summary,
        "future_canary_passed": True,
        "queries": queries.tolist(),
    }
    (ROOT / "synthetic_results.json").write_text(json.dumps(output, indent=2))
    logging.info(json.dumps(output, indent=2))
    update_state("synthetic_passed_ready_model_identity")


if __name__ == "__main__":
    main()
