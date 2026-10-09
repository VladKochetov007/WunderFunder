"""Write waves 3-4 as refill job files (one mechanism each, fold1 from scratch, combo recipe otherwise)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bundle as B  # noqa: E402

WAVES = {
    "w3": {
        "vt_f1_adamw": {"--optim": "adamw"},
        "vt_f1_wmsegcorr": {"--loss": "wmse_gcorr"},
        "vt_f1_encmlp": {"--enc": "mlp"},
        "vt_f1_dual": {"--arch": "dual"},
    },
    "w4": {
        "vt_f1_twoclock": {"--arch": "twoclock"},
        "vt_f1_gated": {"--arch": "gated"},
        "vt_f1_gru384_fixmix": {"--hidden": "384", "+": ["--readout-activation", "fixed_mix"]},
        "vt_f1_seed5": {"--seed": "5"},
    },
}


def main() -> None:
    out = Path(__file__).resolve().parent / "refill_jobs"
    out.mkdir(exist_ok=True)
    for wave, variants in WAVES.items():
        for index, (name, change) in enumerate(variants.items()):
            job = {"name": name, "steps": [
                {"label": "fit", "timeout": 4 * 3600, "argv": B.argv_for(name, change)},
                {"label": "export", "timeout": 1800, "argv": B.export_argv(name)},
            ]}
            (out / f"{wave}_{index}_{name}.json").write_text(json.dumps(job, indent=1))
    print(sorted(p.name for p in out.glob("*.json")))


if __name__ == "__main__":
    main()
