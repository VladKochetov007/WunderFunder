"""Build the Kaggle long-run packet: bundle dataset (code snapshot, split, registry, propensity model, job list) and kernel script.

Reuses the Vast job builders (vast_teachers_r1/build_bundle.py: BASE args, argv_for, export_argv) so Kaggle arms are matched to the Vast ladder.
Kaggle mounts the big inputs (wunder-funder, wnn-c20261004-fold0-assets) read-only; the kernel symlinks them into the layout the trainer expects.
"""
from __future__ import annotations
import hashlib, io, json, os, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1"))
os.environ.pop("VT_LANE", None)
import build_bundle as vb  # noqa: E402

SNAP = vb.E529 / "h230_source_v6_lead"
WSD = ["--wsd-decay", "0.25", "--wsd-final", "0.0", "--wsd-shape", "linear"]
ARMS = {
    "kg_stable_swa_e12": {"--epochs": "12", "--checkpoint-epoch": "11", "--lr": "0.0004", "--wd": "0.03", "--sched": "wsd", "--dropout": "0.0",
                          "+": ["--intra-snapshots", "--wsd-decay", "0.1", "--wsd-final", "0.025", "--wsd-shape", "sqrt", "--swa-frac", "0.6"]},
    "kg_thin_muonrnn_e12": {"--epochs": "12", "--checkpoint-epoch": "11", "--lr": "0.0005", "--wd": "0.05", "--sched": "wsd", "--dropout": "0.1",
                            "+": ["--intra-snapshots", *WSD, "--thin-prob", "0.7", "--thin-range", "0.15,0.35"]},
}


def add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name); info.size = len(payload); info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def main() -> None:
    W = vb.W
    jobs = {
        "env": {"WNN_PORTABLE_DATA_MANIFEST": f"{W}/{vb.REGISTRY}", "WNN_LOADER_WORKERS": "1"},
        "prepare": [{"label": "propcache", "timeout": 3600, "argv": [vb.PY, f"{W}/scripts/cache_propensity.py", "c20260929_prop_f1"]}],
        "slots": [{"gpu": 0, "threads": 4}, {"gpu": 1, "threads": 4}], "stagger_seconds": 30,
        "jobs": [{"name": n, "steps": [
            {"label": "fit", "timeout": 30600, "argv": vb.argv_for(n, c)},
            {"label": "export", "timeout": 3000, "argv": vb.export_argv(n, epoch=c["--checkpoint-epoch"])}]} for n, c in ARMS.items()],
    }
    out = HERE / "dataset"
    tar_path = out / "bundle.wnn"  # Kaggle unpacks .tar.gz uploads; an opaque extension keeps the bundle intact
    with tarfile.open(tar_path, "w:gz") as archive:
        for sub in ("src", "scripts"):
            for path in sorted((SNAP / sub).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=f"{sub}/{path.relative_to(SNAP / sub)}")
        for script in ("predict_run.py", "cache_propensity.py", "snapshot_curve.py", "grade_prediction_bank.py"):
            archive.add(ROOT / "scripts" / script, arcname=f"scripts/{script}")
        for relative in (vb.SPLIT, vb.REGISTRY):
            archive.add(ROOT / relative, arcname=relative)
        for path in sorted((ROOT / vb.PROP).iterdir()):
            archive.add(path, arcname=f"{vb.PROP}/{path.name}")
        add_bytes(archive, "kaggle_jobs.json", json.dumps(jobs, indent=1).encode())
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out / "dataset-metadata.json").write_text(json.dumps({"title": "wnn kaggle long bundle 20261010", "id": "vladkochetov07/wnn-kaggle-long-bundle-20261010",
                                                          "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    assets = json.loads((ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json").read_text())["files"]
    amap = {f["path"]: f["filename"] for f in assets if f["path"] in ("cache/chain_folds.npz", "cache/chains.npz", "cache/train_targets.npz", "cache/valid_targets.npz")}
    (HERE / "kernel/asset_map.json").write_text(json.dumps(amap, indent=1))
    print("bundle", digest, tar_path.stat().st_size, "arms", list(ARMS))


if __name__ == "__main__":
    main()
