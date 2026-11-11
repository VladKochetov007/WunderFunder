"""Build Kaggle specs for the TabM-GRU and NorMuon screens on the fast cold Muon recipe.

Same recipe, seeds, folds, hardware class and strict-FP32 settings as the Kaggle R3 Muon
controls (`krepro_{f0_s2001,f2_s2002}_muon_r3`, reused, not refit). Four new fits:
TabM-mini k=8 (Muon) and NorMuon, each on F0/seed 2001 and F2/seed 2002.
Usage: build_specs.py OUT_DIR
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R3 = ROOT / "research/campaign_2026-10-25/kaggle_muon_repro_r1/specs_r3"
sys.path.insert(0, str(R3.parent))
from build_specs import canonical, descriptor  # noqa: E402

ARMS = {
    "tabm8": (["--ensemble-k", "8"], 11_000_000_000, 13_000_000_000, 12_600),
    "normuon": (["--muon-update", "normuon"], 6_000_000_000, 8_000_000_000, 5_400),
}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for cell in ("f0_s2001", "f2_s2002"):
        base = json.loads((R3 / f"krepro_{cell}_muon_r3.json").read_text())
        code = sorted(
            (descriptor(d["path"]) for d in base["code"]), key=lambda d: d["path"]
        )
        extra = sorted({"src/wnn/tabm_gru.py"} - {d["path"] for d in code})
        code = sorted(code + [descriptor(p) for p in extra], key=lambda d: d["path"])
        for arm, (flags, gpu, stop, timeout) in ARMS.items():
            name = f"tabm29_{cell}_{arm}_r1"
            argv = list(base["argv"])
            argv[argv.index("--name") + 1] = name
            argv += flags
            spec = dict(base)
            spec.update(
                name=name,
                argv=argv,
                code=code,
                code_sha256=canonical(code),
                artifacts=[f"artifacts/runs/{name}/{f}" for f in ("model_ep2.pt", "provenance.json", "history.json")],
                limits={**base["limits"], "gpu_memory_bytes": gpu, "gpu_stop_bytes": stop, "timeout_seconds": timeout},
            )
            (out / f"{name}.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
            names.append(name)
    print(json.dumps({"specs": names, "code_files": len(code)}))


if __name__ == "__main__":
    main()
