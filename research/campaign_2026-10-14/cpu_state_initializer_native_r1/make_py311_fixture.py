"""Export a tiny deterministic native-callback fixture from the Torch environment."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("PY311_CALLBACK_FIXTURE_R1.npz")
MODES = ("zero", "constant", "first_row", "prefix")


def main() -> None:
    helpers = runpy.run_path(str(ROOT / "tests/test_state_initialization_native.py"))
    model_factory = helpers["_model"]
    raw_factory = helpers["_raw_rows"]
    feature_factory = helpers["_features"]
    export_initializer = helpers["export_native_initializer"]

    raw = raw_factory()
    features = torch.from_numpy(feature_factory(raw))[None]
    arrays: dict[str, np.ndarray] = {
        "raw": raw,
        "modes": np.asarray(MODES),
    }
    for mode in MODES:
        wrapper = model_factory(mode)
        with torch.no_grad():
            predictions, final_state = wrapper(features)
        prefix = f"{mode}__"
        arrays[prefix + "predictions"] = predictions[0].cpu().numpy().astype(np.float32)
        arrays[prefix + "hidden"] = final_state.hidden[:, 0].cpu().numpy().astype(np.float32)
        for name, tensor in wrapper.backbone.state_dict().items():
            arrays[prefix + "weight__" + name] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(), dtype=np.float32
            )
        config = export_initializer(wrapper)
        for name, value in config.items():
            if isinstance(value, np.ndarray):
                arrays[prefix + "initializer__" + name] = np.ascontiguousarray(
                    value, dtype=np.float32
                )
            else:
                arrays[prefix + "initializer__" + name] = np.asarray(value)

    np.savez_compressed(OUTPUT, **arrays)
    print(f"fixture={OUTPUT}")
    print(f"bytes={OUTPUT.stat().st_size}")
    print(f"modes={','.join(MODES)} rows={raw.shape[0]} raw_width={raw.shape[1]}")


if __name__ == "__main__":
    main()
