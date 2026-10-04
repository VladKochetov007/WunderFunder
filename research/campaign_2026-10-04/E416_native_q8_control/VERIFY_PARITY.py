"""Independent artifact completeness/content audit, without rerunning inference."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from wnn.gmetric import GlobalAccumulator

out = Path(sys.argv[1])
level = sys.argv[2]
manifest = json.loads((out / "manifest.json").read_text())
expected = {f"{model}_{mode}" for model in ("d256", "w384") for mode in ("q16x", "q8r", "q8x")}
assert set(manifest["cases"]) == expected
assert len(manifest["sequences"]) == 3
assert manifest["gates"] == {"max_abs": 3e-4, "absolute_wp_delta": 5e-6, "portable_bitwise": True, "v2_v3_bitwise": True, "repeat_reset_bitwise": True}

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

for path, expected_hash in manifest["source_sha256"].items():
    assert sha(Path(path)) == expected_hash, path
for name, expected_hash in manifest["files_sha256"].items():
    assert sha(out / name) == expected_hash, name
build = out.with_name("E416_native_q8_build")
assert sha(build / level / "wnn_native.abi3.so") == manifest["library_sha256"][level]
record = json.loads((out / f"parity_{level}.json").read_text())
assert set(record["cases"]) == expected
features = np.load(out / "features.npy", mmap_mode="r")
target, mask, need = [np.load(out / f"{name}.npy") for name in ("targets", "mask", "need")]
assert features.shape == (3, 20000, 112)
assert target.shape == (3, 20000, 2)
assert mask.shape == need.shape == (3, 20000)
checked = {}
with np.load(out / f"parity_{level}.npz") as predictions, np.load(out / "torch_references.npz") as reference:
    assert set(predictions.files) == set(reference.files) == expected
    for case in sorted(expected):
        prediction, ref = predictions[case], reference[case]
        assert prediction.shape == ref.shape == target.shape
        assert np.isfinite(prediction[need]).all() and np.isfinite(ref).all()
        error = float(np.abs(prediction[need] - ref[need]).max())
        p, r = GlobalAccumulator(), GlobalAccumulator()
        p.add(target, prediction, mask)
        r.add(target, ref, mask)
        delta = float(p.result()["wp"] - r.result()["wp"])
        saved = record["cases"][case]
        assert error == saved["max_abs"] and delta == saved["wp_delta"], case
        assert error <= 3e-4 and abs(delta) <= 5e-6, case
        assert saved["passed"] is True and saved["portable_bitwise"] is True
        checked[case] = {"max_abs": error, "wp_delta": delta}
report = {"level": level, "checked_cases": checked, "source_and_prepared_hashes_verified": True,
          "parity_json_sha256": sha(out / f"parity_{level}.json"),
          "parity_npz_sha256": sha(out / f"parity_{level}.npz"),
          "limits": "Artifact audit independently recomputes reference errors and verifies completeness; portable/reset claims remain evidence from the completed scientific process."}
print(json.dumps(report, indent=2))
