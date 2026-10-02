"""engine_fp16_check torch-parity with q16x added:  torch_parity_q16x.py LIB OUT.json REF [REF ...]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import engine_fp16_check as e  # noqa: E402

lib, out, *refs = sys.argv[1:]
native = e.load_native(Path(lib))
report = {"lib": lib, "cases": {}}
for path in refs:
    with np.load(path) as ref:
        x, y64, y32 = ref["X"], ref["y64"], ref["y32"]
        weights = {k[2:]: np.ascontiguousarray(ref[k]) for k in ref.files if k.startswith("w/")}
    case = {"rows": int(len(x)), "torch_f32_vs_f64_max": float(np.abs(y32 - y64).max())}
    for prec in ("f32", "i16", "q16x"):
        model = native.ModelStepper(weights, fast_activations=True, precision=prec)
        y = model.run(x)
        model.reset()
        again = model.run(x)
        case[f"{prec}_fast"] = {
            "effective": model.precision,
            "max_abs_vs_torch64": float(np.abs(y - y64).max()),
            "mean_abs_vs_torch64": float(np.abs(y - y64).mean()),
            "reset_bitwise_repeat": bool(np.array_equal(y.view(np.uint32), again.view(np.uint32))),
        }
    report["cases"][Path(path).stem] = case
    print(Path(path).stem, json.dumps(case), flush=True)
Path(out).write_text(json.dumps(report, indent=1))
