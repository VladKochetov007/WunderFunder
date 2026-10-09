"""Build the Vast teacher bundle (code, split, registry, propensity model, job list) for a private Kaggle dataset.

The bundle is downloaded by the rented host through a signed URL, so nothing large is uploaded from
the laptop.  Jobs are matched from-scratch fold1 teachers on the h230_source_v3_lineard2z snapshot,
varying one axis each against the onecycle GRU256x2 Muon combo recipe.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
E529 = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
SNAP = E529 / "h230_source_v3_lineard2z"
MULTIFOLD = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1"
SPLIT = "research/campaign_2026-09-29/artifacts/splits/fold1.json"
REGISTRY = f"{MULTIFOLD}/portable_inputs_fold1.json"
PROP = "artifacts/runs/c20260929_prop_f1"
W = "/root/wnn"
PY = "python"

BASE = [
    "--device", "cuda", "--arch", "gru", "--cell", "gru", "--enc", "linear", "--hidden", "256",
    "--layers", "2", "--dropout", "0", "--seed", "3", "--split-manifest", f"{W}/{SPLIT}", "--valid-fold", "1",
    "--eval-source", "vfold", "--train-source", "train+valid", "--train-seqs", "10607", "--epochs", "4",
    "--checkpoint-policy", "fixed", "--checkpoint-epoch", "3", "--no-eval-during-fit", "--batch", "32",
    "--chunk", "1000", "--loss", "wmse", "--loss-norm", "chunk", "--lr", "0.001", "--wd", "0.01",
    "--optim", "muon", "--muon-scope", "rnn", "--sched", "onecycle", "--precision", "fp16",
    "--propensity", "c20260929_prop_f1", "--prop-power", "2", "--prop-eps", "0.03", "--valid-true-mask", "0.03",
    "--aux-struct", "0.3", "--aux-next", "0.3", "--since-weight", "32", "--fast", "--save-training-state",
]

VARIANTS = {
    "vt_f1_gru384": {"--hidden": "384"},
    "vt_f1_gru512": {"--hidden": "512"},
    "vt_f1_lstm256": {"--cell": "lstm"},
    "vt_f1_gru256x3": {"--layers": "3"},
    "vt_f1_lineard2z": {"--sched": "wsd", "+": ["--wsd-decay", "1.0", "--wsd-final", "0.0", "--wsd-shape", "linear"]},
    "vt_f1_trapezoid": {"--sched": "wsd", "+": ["--wsd-decay", "0.2", "--wsd-final", "0.0", "--wsd-shape", "linear"]},
    "vt_f1_fixmix": {"+": ["--readout-activation", "fixed_mix"]},
    "vt_f1_seed4": {"--seed": "4"},
}


def argv_for(name: str, change: dict) -> list[str]:
    args = list(BASE)
    for flag, value in change.items():
        if flag == "+":
            continue
        if flag == "-":
            args = [a for a in args if a not in value]
            continue
        args[args.index(flag) + 1] = value
    extra = change.get("+", [])
    position = args.index("--fast")
    args[position:position] = extra
    return [PY, f"{W}/scripts/train_rnn.py", "--name", name, *args]


def export_argv(name: str, epoch: str = "3", output_name: str | None = None) -> list[str]:
    runner = (
        "import runpy,sys; from pathlib import Path; run_script=sys.argv[1]; trainer=sys.argv[2]; "
        "sys.path.insert(0,str(Path(trainer).resolve().parent)); sys.argv=[run_script,*sys.argv[3:]]; "
        "runpy.run_path(run_script,run_name='__main__')"
    )
    return [PY, "-c", runner, f"{W}/scripts/predict_run.py", f"{W}/scripts/train_rnn.py", name, "valid", "--epoch", epoch,
            *([] if output_name is None else ["--output-name", output_name])]


def jobs() -> dict:
    return {
        "pip": ["pyarrow", "scipy", "numba", "matplotlib"],
        "env": {"WNN_PORTABLE_DATA_MANIFEST": f"{W}/{REGISTRY}"},
        "prepare": [
            {"label": "propcache", "timeout": 1800,
             "argv": [PY, f"{W}/scripts/cache_propensity.py", "c20260929_prop_f1"]},
        ],
        "slots": [{"gpu": 0, "threads": 6}, {"gpu": 1, "threads": 6}, {"gpu": 0, "threads": 6}, {"gpu": 1, "threads": 6}],
        "stagger_seconds": 60,
        "jobs": [
            {"name": name, "steps": [
                {"label": "fit", "timeout": 4 * 3600, "argv": argv_for(name, change)},
                {"label": "export", "timeout": 1800, "argv": export_argv(name)},
            ]}
            for name, change in VARIANTS.items()
        ],
    }


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def main() -> None:
    out = HERE / "bundle"
    out.mkdir(exist_ok=True)
    tar_path = out / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for sub in ("src", "scripts"):
            for path in sorted((SNAP / sub).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=f"{sub}/{path.relative_to(SNAP / sub)}")
        for script in ("predict_run.py", "cache_propensity.py"):
            archive.add(ROOT / "scripts" / script, arcname=f"scripts/{script}")
        for relative in (SPLIT, REGISTRY):
            archive.add(ROOT / relative, arcname=relative)
        for path in sorted((ROOT / PROP).iterdir()):
            archive.add(path, arcname=f"{PROP}/{path.name}")
        add_bytes(archive, "vast_jobs.json", json.dumps(jobs(), indent=1).encode())
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out / "BUNDLE_RECEIPT.json").write_text(json.dumps({
        "bundle": "bundle.tar.gz", "sha256": digest, "bytes": tar_path.stat().st_size,
        "jobs": list(VARIANTS), "snapshot": str(SNAP.relative_to(ROOT)),
    }, indent=1))
    print(digest, tar_path.stat().st_size)


if __name__ == "__main__":
    main()
