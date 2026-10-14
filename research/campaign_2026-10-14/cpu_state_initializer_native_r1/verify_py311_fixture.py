"""Load the NumPy-only fixture and exercise the abi3 callback under Python 3.11."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import wnn_native

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "PY311_CALLBACK_FIXTURE_R1.npz"
RECEIPT = HERE / "PY311_ABI3_NATIVE_RECEIPT_R1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(model: object, rows: np.ndarray, seq_ix: int) -> np.ndarray:
    predictions = []
    for row in rows:
        point = SimpleNamespace(seq_ix=seq_ix, state=row, need_prediction=True)
        prediction = model.predict(point)
        if prediction is None:
            raise AssertionError("prediction unexpectedly missing")
        predictions.append(np.asarray(prediction, dtype=np.float32))
    return np.stack(predictions)


def model_weights(archive: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    marker = prefix + "weight__"
    return {
        key[len(marker) :]: np.ascontiguousarray(archive[key], dtype=np.float32)
        for key in archive.files
        if key.startswith(marker)
    }


def initializer_config(archive: np.lib.npyio.NpzFile, prefix: str) -> dict[str, object]:
    marker = prefix + "initializer__"
    config: dict[str, object] = {}
    for key in archive.files:
        if not key.startswith(marker):
            continue
        name = key[len(marker) :]
        value = archive[key]
        if name == "mode":
            config[name] = str(value.item())
        elif name == "residual_bound":
            config[name] = float(value.item())
        else:
            config[name] = np.ascontiguousarray(value, dtype=np.float32)
    return config


def main() -> None:
    mode_errors: dict[str, dict[str, float]] = {}
    with np.load(FIXTURE, allow_pickle=False) as archive:
        raw = np.ascontiguousarray(archive["raw"], dtype=np.float32)
        modes = [str(value) for value in archive["modes"].tolist()]
        for index, mode in enumerate(modes):
            prefix = f"{mode}__"
            weights = model_weights(archive, prefix)
            config = initializer_config(archive, prefix)
            model = wnn_native.EnsemblePredictor(
                [weights],
                [1.0],
                fast_activations=False,
                precision="f32",
                state_initializers=[config],
            )
            actual = run(model, raw, seq_ix=100 + index)
            expected = archive[prefix + "predictions"]
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
            actual_hidden = model.hidden_states[0]
            expected_hidden = archive[prefix + "hidden"]
            np.testing.assert_allclose(actual_hidden, expected_hidden, rtol=4e-4, atol=4e-5)
            final_hidden = actual_hidden.copy()
            replay = run(model, raw, seq_ix=200 + index)
            np.testing.assert_array_equal(replay, actual)
            np.testing.assert_array_equal(model.hidden_states[0], final_hidden)
            mode_errors[mode] = {
                "prediction_max_abs_error": float(np.max(np.abs(actual - expected))),
                "hidden_max_abs_error": float(np.max(np.abs(actual_hidden - expected_hidden))),
                "reset_replay_bitwise_equal": True,
            }

            if mode == "zero":
                baseline = wnn_native.EnsemblePredictor(
                    [weights], [1.0], fast_activations=False, precision="f32"
                )
                baseline_output = run(baseline, raw, seq_ix=100 + index)
                np.testing.assert_array_equal(baseline_output, actual)
                np.testing.assert_array_equal(baseline.hidden_states[0], final_hidden)

    native_path = Path(wnn_native.__file__).resolve()
    receipt = {
        "schema": "wnn.state-initializer-native-py311.v1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "extension_path": str(native_path),
        "extension_sha256": sha256(native_path),
        "fixture_path": FIXTURE.name,
        "fixture_sha256": sha256(FIXTURE),
        "fixture_reference_environment": "Python 3.13.9, deterministic synthetic Torch wrapper",
        "modes": mode_errors,
        "zero_vs_omitted": "bitwise identical predictions and final hidden state",
        "scope": "Python 3.11 abi3 import and native serving callback/transport only; wrapper/export generation used Python 3.13 because Torch is absent from the profile environment",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
