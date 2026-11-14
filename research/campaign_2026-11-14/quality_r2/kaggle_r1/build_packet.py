"""Build the four-fit private Kaggle packet from the frozen PR49 producer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[3]
PR49 = Path("/home/vlad/development/wunder/research_tmp/pr49_wt")
DATA = Path("/home/vlad/development/wunder/research_tmp/chrono_root")
SCRATCH = Path("/home/vlad/development/wunder/research_tmp/q14_kaggle")
STAGE = SCRATCH / "project_root"
UPLOAD = SCRATCH / "upload_dataset"
CAMPAIGN = "research/campaign_2026-11-14/quality_r2/kaggle_r1"
SPLIT_PATH = "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
RAW_SHA = {
    "datasets/train.parquet": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    "datasets/valid.parquet": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
}
ASSET = "vladkochetov07/wnn-c20261004-fold0-assets"
VALID_TARGET = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn"
TEACHER = "tt29rep_fbc1k_s5"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(root: Path, name: str) -> dict:
    path = root / name
    return {"path": name, "size": path.stat().st_size, "sha256": sha(path)}


def link(root: Path, name: str, source: Path) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(source.resolve())


def main() -> None:
    if STAGE.exists() or UPLOAD.exists():
        raise FileExistsError("Refusing to replace an existing scratch packet")
    if PR49.resolve() != Path("/home/vlad/development/wunder/research_tmp/pr49_wt"):
        raise ValueError("Unexpected frozen source root")
    split = json.loads(
        (
            REPO / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
        ).read_text()
    )
    split_sha = sha(
        REPO / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
    )
    if split_sha != "6d23a1b76f66d8001dac805d2b218a3b139e4eebe555f0c9b96fe1263ce186f0":
        raise ValueError("Chronological split bytes changed")
    prior_registry = json.loads(
        (
            REPO / "research/campaign_2026-10-26/kaggle_parent_numerics_r1/portable_inputs.json"
        ).read_text()
    )
    registry_sources = {}
    for name, identity in split["sources"].items():
        digest = identity.get("sha256") or prior_registry["sources"][name]["sha256"]
        registry_sources[name] = {"original_identity": identity, "sha256": digest}
    for name, expected in RAW_SHA.items():
        if registry_sources[name]["sha256"] != expected:
            raise ValueError(f"Raw source digest differs from prior verified registry: {name}")
    registry_path = PACKET / "portable_inputs.json"
    registry_path.write_text(
        json.dumps({"version": 1, "sources": registry_sources}, indent=1, sort_keys=True) + "\n"
    )

    STAGE.mkdir(parents=True)
    tracked = subprocess.run(
        ["git", "ls-files", "src/wnn"], cwd=PR49, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    code_names = sorted(name for name in tracked if name.endswith(".py"))
    code_names.extend(
        [
            "scripts/train_rnn.py",
            "scripts/kaggle_gpu_runner.py",
            "scripts/kaggle_common.py",
            "scripts/kaggle_kernel.py",
            "scripts/kaggle_traced_train.py",
            "scripts/kaggle_strict_traced_train.py",
            "utils.py",
        ]
    )
    for name in code_names:
        source = (
            PACKET / "kaggle_strict_traced_train.py"
            if name.endswith("kaggle_strict_traced_train.py")
            else (REPO / name if name == "scripts/kaggle_traced_train.py" else PR49 / name)
        )
        link(STAGE, name, source)

    local_inputs = [
        "cache/chronological_chains.npz",
        "cache/source_audit.json",
        "cache/train_index.npz",
        "cache/train_index.identity.json",
        "cache/valid_index.npz",
        "cache/valid_index.identity.json",
        "cache/train_targets.npz",
        f"cache/pred_{TEACHER}_train.npy",
        f"cache/pred_{TEACHER}_train.json",
        f"artifacts/runs/{TEACHER}/model.pt",
        f"artifacts/runs/{TEACHER}/provenance.json",
    ]
    for name in local_inputs:
        link(STAGE, name, DATA / name)
    split_dest = STAGE / SPLIT_PATH
    split_dest.parent.mkdir(parents=True, exist_ok=True)
    link(STAGE, SPLIT_PATH, REPO / SPLIT_PATH)
    reg_dest = STAGE / f"{CAMPAIGN}/portable_inputs.json"
    reg_dest.parent.mkdir(parents=True, exist_ok=True)
    link(STAGE, f"{CAMPAIGN}/portable_inputs.json", registry_path)

    source_code = sorted(
        (descriptor(STAGE, name) for name in set(code_names)), key=lambda item: item["path"]
    )
    canonical_code = hashlib.sha256(
        json.dumps(source_code, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    inputs = [descriptor(STAGE, name) for name in local_inputs]
    inputs.extend(
        [descriptor(STAGE, SPLIT_PATH), descriptor(STAGE, f"{CAMPAIGN}/portable_inputs.json")]
    )
    inputs.extend(
        [
            {
                "path": "datasets/train.parquet",
                "size": split["sources"]["datasets/train.parquet"]["bytes"],
                "sha256": RAW_SHA["datasets/train.parquet"],
            },
            {
                "path": "datasets/valid.parquet",
                "size": split["sources"]["datasets/valid.parquet"]["bytes"],
                "sha256": RAW_SHA["datasets/valid.parquet"],
            },
            {
                "path": "cache/valid_targets.npz",
                "size": 337186198,
                "sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
            },
        ]
    )
    base = [
        "--hidden",
        "256",
        "--layers",
        "2",
        "--dropout",
        "0",
        "--train-source",
        "train",
        "--train-seqs",
        "10325",
        "--eval-source",
        "dev",
        "--full-valid",
        "--split-manifest",
        SPLIT_PATH,
        "--precision",
        "fp32",
        "--checkpoint-policy",
        "fixed",
        "--loss-norm",
        "chunk",
        "--sched",
        "onecycle",
        "--warmup-frac",
        "0.05",
        "--wd",
        "0.01",
        "--clip",
        "1",
        "--beta2",
        "0.999",
        "--grad-stats",
        "--save-training-state",
        "--train-eval-seqs",
        "0",
        "--loss",
        "wmse",
        "--optim",
        "muon",
        "--muon-scope",
        "rnn",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--batch",
        "32",
        "--chunk",
        "1000",
        "--epochs",
        "4",
        "--lr",
        "0.001",
        "--aux-next",
        "0.3",
        "--aux-struct",
        "0.3",
        "--aux-weight",
        "0.5",
        "--distill",
        TEACHER,
        "--distill-alpha",
        ".9",
    ]
    specs = []
    for seed in (5, 6):
        for arm, extra in (("control", []), ("boot8", ["--distill-boot-h", "8"])):
            name = f"q14_kd90_{arm}_s{seed}"
            argv = [
                "{python}",
                "-u",
                "scripts/kaggle_strict_traced_train.py",
                *base,
                "--name",
                name,
                "--seed",
                str(seed),
                *extra,
            ]
            specs.append(
                {
                    "version": 1,
                    "name": name,
                    "kind": "fit_only",
                    "argv": argv,
                    "code": source_code,
                    "code_sha256": canonical_code,
                    "inputs": inputs,
                    "artifacts": [
                        f"artifacts/runs/{name}/{file}"
                        for file in (
                            "model_ep3.pt",
                            "model.pt",
                            "provenance.json",
                            "history.json",
                            "training_state.pt",
                        )
                    ]
                    + [f"artifacts/runtime/{name}.json"],
                    "limits": {
                        "cpus": 2,
                        "gpu_memory_bytes": 3_000_000_000,
                        "gpu_stop_bytes": 5_000_000_000,
                        "ram_bytes": 8_000_000_000,
                        "timeout_seconds": 7_200,
                    },
                    "environment": {
                        "NVIDIA_TF32_OVERRIDE": "0",
                        "PYTHONHASHSEED": "0",
                        "WNN_PROJECT_ROOT": "/kaggle/working/experiment",
                        "WNN_DEFERRED_IDENTITY_PATHS": "datasets/train.parquet,datasets/valid.parquet",
                        "WNN_PORTABLE_DATA_MANIFEST": f"{CAMPAIGN}/portable_inputs.json",
                        "WNN_SHARED_DIGEST_CACHE": "/kaggle/working/portable_digest_cache.json",
                        "WNN_THREADS": "2",
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    },
                }
            )
    spec_paths = []
    spec_dir = PACKET / "specs"
    spec_dir.mkdir(exist_ok=True)
    for spec in specs:
        path = spec_dir / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
        spec_paths.append(path)
    command = [
        sys.executable,
        str(PR49 / "scripts/kaggle_bundle.py"),
        "--root",
        str(STAGE),
        "--destination",
        str(UPLOAD),
        "--dataset",
        "vladkochetov07/wnn-q14-kd90-boot8-20261114-r1",
        "--external",
        "datasets/train.parquet=train.parquet",
        "--external",
        "datasets/valid.parquet=valid.parquet",
        "--external",
        f"cache/valid_targets.npz={ASSET}:{VALID_TARGET}",
        "--asset-dataset",
        ASSET,
    ]
    for path in spec_paths:
        command.extend(["--spec", str(path)])
    subprocess.run(command, check=True, cwd=PR49)

    kernel = UPLOAD / "kernel/kernel.py"
    source = kernel.read_text()
    source = source.replace(
        "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200", "KERNEL_PROCESS_WALL_SECONDS_MAX = 13_500"
    )
    if "KERNEL_PROCESS_WALL_SECONDS_MAX = 13_500" not in source:
        raise ValueError("Could not bind the 3h45 kernel deadline")
    kernel.write_text(source)
    receipt = {
        "status": "built_not_pushed",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PR49, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "split_sha256": split_sha,
        "specs": [spec["name"] for spec in specs],
        "portable_registry_sha256": sha(registry_path),
        "bundle_sha256": sha(UPLOAD / "bundle.wnn"),
        "bundle_manifest_sha256": sha(UPLOAD / "bundle-manifest.json"),
        "kernel_sha256": sha(kernel),
        "kernel_wall_hard_seconds": 13_500,
        "worker_timeout_seconds": 7_200,
        "gpu_operating_bytes": 3_000_000_000,
        "gpu_stop_bytes": 5_000_000_000,
        "max_gpu_device_hours_at_two_t4": 7.5,
        "upload_bytes": (UPLOAD / "bundle.wnn").stat().st_size,
        "expanded_bytes": json.loads((UPLOAD / "bundle-manifest.json").read_text())[
            "expanded_bytes"
        ],
    }
    (PACKET / "BUILD_RECEIPT_R1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
