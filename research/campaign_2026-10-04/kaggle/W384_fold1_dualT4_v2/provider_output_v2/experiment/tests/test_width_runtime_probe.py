from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts/probe_width_runtime.py"
LOADER = importlib.util.spec_from_file_location("width_runtime_probe", PROBE_PATH)
assert LOADER is not None and LOADER.loader is not None
probe = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(probe)


def weights(hidden: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    return {
        "inp.bias": np.zeros(hidden, np.float32),
        "head.0.weight": rng.normal(size=(hidden, 2 * hidden)).astype(np.float32),
        "head.0.bias": rng.normal(size=hidden).astype(np.float32),
        "head.2.weight": rng.normal(size=(2, hidden)).astype(np.float32),
        "head.2.bias": rng.normal(size=2).astype(np.float32),
    }


def test_duplicate_head_changes_workload_without_replacing_trunk() -> None:
    original = weights(8)
    copied = {key: value.copy() for key, value in original.items()}
    duplicate = probe.duplicate_head(original)
    for name in original:
        np.testing.assert_array_equal(original[name], copied[name])
    assert duplicate["head.0.weight"].shape == (16, 16)
    assert duplicate["head.2.weight"].shape == (2, 16)
    np.testing.assert_array_equal(duplicate["inp.bias"], original["inp.bias"])
    x = np.arange(16, dtype=np.float32) / 16
    activation = np.tanh(original["head.0.weight"] @ x + original["head.0.bias"])
    repeated_activation = np.tanh(duplicate["head.0.weight"] @ x + duplicate["head.0.bias"])
    first = original["head.2.weight"] @ activation + original["head.2.bias"]
    second = duplicate["head.2.weight"] @ repeated_activation + duplicate["head.2.bias"]
    np.testing.assert_allclose(first, second, atol=2e-6, rtol=1e-6)


def test_prepare_binds_sources_and_trained_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    build = tmp_path / "build"
    for name in (
        "rnn_fast.rs",
        "half.rs",
        "approx.rs",
        "python.rs",
        "kernels.rs",
        "stream.rs",
        "rnn.rs",
    ):
        for parent in (repository / "native/src", build / "src/src"):
            parent.mkdir(parents=True, exist_ok=True)
            (parent / name).write_text(f"source for {name}")
    for level in ("v2", "avx2", "v3"):
        (build / level).mkdir()
        (build / level / "wnn_native.abi3.so").write_bytes(f"{level} library".encode())
    d256, w384, data = (tmp_path / name for name in ("d256.pt", "w384.pt", "data.parquet"))
    for path in (d256, w384, data):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(
        probe, "export_checkpoint", lambda path: weights(256 if path == d256 else 384)
    )
    args = argparse.Namespace(
        repository=repository,
        build=build,
        d256=d256,
        w384=w384,
        data=data,
        out=tmp_path / "output",
        seqs=3,
    )
    probe.prepare(args)
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["row_groups"] == [0, 1, 2]
    assert manifest["cases"]["w384_head1"]["heads"] == 1
    assert "duplicate" in manifest["cases"]["w384_head2"]["head2_source"]
    assert len(manifest["libraries"]["v3"]) == 64
    with pytest.raises(FileExistsError):
        probe.prepare(args)
    args.out = tmp_path / "other_output"
    (repository / "native/src/half.rs").write_text("changed current source")
    with pytest.raises(ValueError, match="differs from current"):
        probe.prepare(args)
