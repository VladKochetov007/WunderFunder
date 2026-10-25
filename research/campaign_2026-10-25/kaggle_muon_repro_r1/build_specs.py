"""Build Kaggle specs for the independent Muon-vs-AdamW reproduction (fresh seeds, true FP32, T4).

Four fits of the exact fast cold recipe used locally (H512x1, 10% frozen memberships, 3 epochs,
fixed CP2): {AdamW, Muon(rnn)} x {F0 seed 2001, F2 seed 2002}, plus --strict-fp32-kernels.
Kaggle returns only run directories; export and official scoring happen locally.
Usage: build_specs.py OUT_DIR
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = "research/campaign_2026-10-25/kaggle_muon_repro_r1"
LEAD = ROOT / "research/campaign_2026-10-25/i1_lead_panel_r1/PLANS_R1.json"
CACHES = ("cache/chains.npz", "cache/chain_folds.npz", "cache/train_targets.npz", "cache/valid_targets.npz")
PARQUET = {
    "datasets/train.parquet": ("4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81", 29240219011),
    "datasets/valid.parquet": ("2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782", 5615509128),
}
CELLS = {"f0": ("f0_s1001", 2001, "0"), "f2": ("f2_s1002", 2002, "2")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(rel: str) -> dict:
    path = ROOT / rel
    return {"path": rel, "size": path.stat().st_size, "sha256": sha256(path)}


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    files = subprocess.run(["git", "ls-files", "src/wnn"], cwd=ROOT, capture_output=True, text=True, check=True)
    code_paths = sorted(p for p in files.stdout.split() if p.endswith(".py"))
    code_paths += ["scripts/train_rnn.py", "scripts/kaggle_gpu_runner.py", "scripts/kaggle_common.py",
                   "scripts/kaggle_kernel.py", "utils.py"]
    code = sorted((descriptor(p) for p in code_paths), key=lambda d: d["path"])
    registry = {"version": 1, "sources": {}}
    for fold in ("0", "2"):
        manifest = json.loads((ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json").read_text())
        for name, identity in manifest["sources"].items():
            entry = {"original_identity": identity, "sha256": sha256(ROOT / name)}
            if registry["sources"].setdefault(name, entry) != entry:
                raise ValueError(f"conflicting source identity across folds: {name}")
    registry_rel = f"{CAMPAIGN}/portable_inputs.json"
    (ROOT / registry_rel).write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    lead = json.loads(LEAD.read_text())
    names = []
    for key, (cell, seed, fold) in CELLS.items():
        base = lead[f"i1lead_{cell}_control_r1"]["argv"]
        for arm in ("adamw", "muon"):
            argv = list(base)
            name = f"krepro_{key}_s{seed}_{arm}_r1"
            argv[argv.index("--name") + 1] = name
            argv[argv.index("--seed") + 1] = str(seed)
            if arm == "muon":
                argv[argv.index("--optim") + 1] = "muon"
                argv += ["--muon-scope", "rnn"]
            argv.append("--strict-fp32-kernels")
            subset = argv[argv.index("--fit-subset-manifest") + 1]
            split = argv[argv.index("--split-manifest") + 1]
            inputs = [{"path": p, "sha256": s, "size": n} for p, (s, n) in PARQUET.items()]
            inputs += [descriptor(p) for p in (*CACHES, split, subset, registry_rel)]
            spec = {
                "version": 1,
                "name": name,
                "kind": "fit_only",
                "argv": ["{python}", *argv[1:]],
                "code": code,
                "code_sha256": canonical(code),
                "inputs": inputs,
                "artifacts": [f"artifacts/runs/{name}/{f}" for f in ("model_ep2.pt", "provenance.json", "history.json")],
                "limits": {"cpus": 2, "gpu_memory_bytes": 6_000_000_000, "gpu_stop_bytes": 8_000_000_000,
                           "ram_bytes": 7_000_000_000, "timeout_seconds": 5400},
                "environment": {"NVIDIA_TF32_OVERRIDE": "0", "PYTHONHASHSEED": "0",
                                "WNN_DEFERRED_IDENTITY_PATHS": "datasets/train.parquet,datasets/valid.parquet",
                                "WNN_PORTABLE_DATA_MANIFEST": registry_rel},
            }
            (out / f"{name}.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
            names.append(name)
    print(json.dumps({"specs": names, "code_files": len(code), "registry": registry_rel}))


if __name__ == "__main__":
    main()
