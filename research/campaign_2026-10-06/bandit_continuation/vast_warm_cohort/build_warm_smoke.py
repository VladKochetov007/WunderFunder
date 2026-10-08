"""Build the warm-queue smoke packet, its frozen worker inputs and the root intent draft.

Run from the repository that owns the frozen inputs (the main checkout):

    scripts/limited .venv/bin/python <this file> --main-root <main> --out <dir>

The packet is a plain uncompressed tar with one root-level run.sh. Large inputs
are never placed in the packet; they are listed as frozen worker inputs and are
staged by the admitted Vast supervisor. Parquet inputs use the signed prefetch
path and carry the identities from the V7 intent, which are re-checked at
admission by vast_campaign.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "packet_templates"
RUN_NAME = "smoke_c1v2_s3_f0"
QUEUE_RUN_ID = "smoke-c1v2-s3-f0-v1"
ATTEMPT_ID = "e529-vast-warm-smoke-c1v2-f0-20261008"
CAMPAIGN_LABEL = "wnn-e529-warm-smoke-c1v2-20261008"
IMAGE = "vastai/pytorch@sha256:450710d89268fcd55632f28fdd80508ee920a53075b3f8e9f828bdc154e3d856"
ACCOUNT_SCOPE_SHA256 = "5edaeb435fffdc8d36029206229fcf1ee9a930816996519f3c4fdeeb3d35858a"
V7_INTENT = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/"
    "h230_seed54354_v1/H230_SEED54354_V7_PARALLEL_INTENT_20261008_R6.json"
)
OFFER_SEARCH = HERE / "OFFER_SNAPSHOT_RTX3090_20261008_R1.json"
OFFER_ID = 47997542
PARQUET = ("datasets/train.parquet", "datasets/valid.parquet")
SPLIT = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PROP = "c20260929_prop_f0"
PARENT_S3 = "c20261003_combo_s3_f0"
PARENT_S4 = "c20261003_combo_s4_f0"
PY = "/venv/main/bin/python"
PY_WRAPPER = (
    "import pyarrow,runpy,sys;"
    "pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1);"
    "sys.path.insert(0,'scripts');"
    "script=sys.argv[1];sys.argv=[script,*sys.argv[2:]];"
    "runpy.run_path(script,run_name='__main__')"
)
SOURCE_FILES = (
    "scripts/train_rnn.py",
    "scripts/predict_run.py",
    "scripts/score_fold0_panel.py",
    "scripts/evaluate_bandit_continuation.py",
    "scripts/build_distill_cache.py",
    "scripts/warm_queue_worker.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def frozen_inputs(main: Path) -> list[dict[str, object]]:
    """Return the worker input set; hashes for parquet come from the reviewed V7 intent."""
    v7 = json.loads((main / V7_INTENT).read_text(encoding="utf-8"))
    v7_rows = {row["remote_path"]: row for row in v7["worker_inputs"]}
    rows: list[dict[str, object]] = []
    for remote in PARQUET:
        row = v7_rows[remote]
        rows.append({"remote_path": remote, "source_path": row["source_path"],
                     "source_sha256": row["source_sha256"], "size_bytes": row["size_bytes"]})
    local = [
        f"artifacts/runs/{PROP}/model.pt",
        f"artifacts/runs/{PROP}/provenance.json",
        f"artifacts/runs/{PARENT_S3}/model_ep3.pt",
        f"artifacts/runs/{PARENT_S3}/provenance.json",
        f"artifacts/runs/{PARENT_S3}/history.json",
        SPLIT,
        "cache/chains.npz",
        "cache/chain_folds.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    ]
    for name in (PARENT_S3, PARENT_S4):
        for split in ("train", "valid"):
            local += [f"cache/pred_{name}_{split}.npy", f"cache/pred_{name}_{split}.json"]
    for relative in local:
        source = main / relative
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"frozen input is absent or linked: {relative}")
        rows.append({"remote_path": relative, "source_path": relative,
                     "source_sha256": sha256_file(source), "size_bytes": source.stat().st_size})
    return sorted(rows, key=lambda row: str(row["remote_path"]))


def train_argv() -> list[str]:
    return [PY, "-c", PY_WRAPPER, "scripts/train_rnn.py",
            "--name", RUN_NAME, "--device", "cuda", "--arch", "gru", "--cell", "gru",
            "--hidden", "256", "--layers", "2", "--seed", "3",
            "--split-manifest", SPLIT, "--valid-fold", "0", "--train-source", "train+valid",
            "--train-seqs", "1000", "--init", f"{PARENT_S3}@ep3", "--epochs", "1",
            "--checkpoint-policy", "fixed", "--checkpoint-epoch", "0", "--no-eval-during-fit",
            "--batch", "16", "--chunk", "2000", "--loss", "wmse", "--lr", "0.0001", "--wd", "0.01",
            "--optim", "adamw", "--sched", "onecycle", "--propensity", PROP, "--prop-power", "2",
            "--prop-eps", "0.03", "--valid-true-mask", "0.03", "--aux-struct", "0.3",
            "--aux-next", "0.3", "--distill-boot-h", "0", "--fast", "--save-training-state",
            "--distill", f"{PARENT_S3}*0.40625,{PARENT_S4}*0.40625", "--distill-alpha", "0.7",
            "--distill-family-weight", "0.0"]


def queue_spec() -> dict[str, object]:
    env = {"OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    return {
        "run_id": QUEUE_RUN_ID,
        "arm": "warm_smoke_c1v2_fold0",
        "hypothesis": "transport smoke: fold-0 C1v2 continuation, prediction and official panel grade; no scientific claim",
        "priority": 10,
        "steps": [
            {"name": "train_fit_ep0", "argv": train_argv(), "timeout_seconds": 3600, "env": env},
            {"name": "predict_valid", "argv": [PY, "-c", PY_WRAPPER, "scripts/predict_run.py",
                                              RUN_NAME, "valid", "--epoch", "0",
                                              "--inference-precision", "fp32"],
             "timeout_seconds": 1200, "env": env},
            {"name": "grade_fold0_panel", "argv": [PY, "-c", PY_WRAPPER,
                                                  "scripts/score_fold0_panel.py",
                                                  f"{RUN_NAME}@ep0", "--out",
                                                  "artifacts/smoke/grade_smoke_c1v2_s3_f0.json"],
             "timeout_seconds": 900, "env": env},
        ],
        "expected_outputs": [
            f"artifacts/runs/{RUN_NAME}/model_ep0.pt",
            f"artifacts/runs/{RUN_NAME}/provenance.json",
            f"cache/pred_{RUN_NAME}_valid.npy",
            "artifacts/smoke/grade_smoke_c1v2_s3_f0.json",
        ],
    }


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def build(main: Path, out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {}
    worker_inputs = frozen_inputs(main)
    spec = {
        "schema_version": 1,
        "attempt_id": ATTEMPT_ID,
        "campaign_label": CAMPAIGN_LABEL,
        "expected_image": IMAGE,
        "expected_input_count": len(worker_inputs),
        "inputs": [{"path": row["remote_path"], "sha256": row["source_sha256"],
                    "size_bytes": row["size_bytes"]} for row in worker_inputs],
        "queue_run_id": QUEUE_RUN_ID,
        "run_name": RUN_NAME,
    }
    files["worker_spec.json"] = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    files[f"queue/pending/{QUEUE_RUN_ID}.json"] = (
        json.dumps(queue_spec(), indent=2, sort_keys=True) + "\n").encode()
    files["queue/done/.keep"] = b""
    files["queue/failed/.keep"] = b""
    files["project/artifacts/smoke/.keep"] = b""
    for name in ("run.sh", "stage_inputs.py", "finalize.py"):
        files[name] = (TEMPLATES / name).read_bytes()
    for relative in SOURCE_FILES:
        files[f"project/{relative}"] = (main / relative).read_bytes()
    files["project/research/campaign_2026-10-04/portable_inputs.json"] = (
        main / "research/campaign_2026-10-04/portable_inputs.json").read_bytes()
    for path in sorted((main / "src" / "wnn").rglob("*.py")):
        if "__pycache__" not in path.parts:
            files[f"project/src/wnn/{path.relative_to(main / 'src' / 'wnn')}"] = path.read_bytes()
    # The grader and trainer import the same wnn package; ensure all packages are present.
    tar_path = out / "warm_smoke_v1.tar"
    names = sorted(files)
    with tarfile.open(tar_path, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in names:
            mode = 0o755 if name == "run.sh" else 0o644
            add_bytes(tar, name, files[name], mode)
    packet_sha = sha256_file(tar_path)
    packet_bytes = tar_path.stat().st_size
    upload_total = packet_bytes + sum(int(row["size_bytes"]) for row in worker_inputs)
    offer_rows = json.loads(OFFER_SEARCH.read_text(encoding="utf-8"))
    offer = next(row["offer"] for row in offer_rows if row["offer"]["id"] == OFFER_ID)
    intent = {
        "schema_version": 1,
        "attempt_id": ATTEMPT_ID,
        "campaign_label": CAMPAIGN_LABEL,
        "account_scope_sha256": ACCOUNT_SCOPE_SHA256,
        "packet_path": str((out / "warm_smoke_v1.tar").relative_to(main)),
        "packet_sha256": packet_sha,
        "create_payload": {
            "cancel_unavail": True, "client_id": "me", "disk": 120, "image": IMAGE,
            "label": CAMPAIGN_LABEL, "onstart": "sleep infinity", "runtype": "ssh_direc ssh_proxy",
        },
        "offer": offer,
        "min_offer_reliability": "0.97",
        "worker_inputs": [{"source_path": row["source_path"], "source_sha256": row["source_sha256"],
                           "size_bytes": row["size_bytes"], "remote_path": row["remote_path"]}
                          for row in worker_inputs],
        "max_upload_bytes": upload_total + 1_000_000_000,
        "upload_timeout_seconds": 5400,
        "expected_artifacts": ["artifact_bundle.tar", "result.json", "worker.log", "image_preflight.json"],
        "max_runtime_seconds": 10800,
        "worker_setup_timeout_seconds": 450,
        "max_artifact_bytes": 3_500_000_000,
        "max_worker_host_ram_gb": "28",
        "remote_output_dir": "/workspace/results",
        "reservation_usd": {"compute_storage_usd": "1.20", "upload_usd": "0.40",
                            "download_usd": "0.10", "cleanup_usd": "0.30"},
        "reservation_total_usd": "2.00",
        "purpose": "warm-cohort transport smoke: one fold-0 C1v2 continuation, prediction and official panel grade; no scientific claim",
    }
    (out / "warm_smoke_intent.json").write_text(json.dumps(intent, indent=2, sort_keys=True) + "\n")
    return {"packet": str(tar_path), "packet_sha256": packet_sha, "packet_bytes": packet_bytes,
            "worker_input_count": len(worker_inputs), "worker_input_bytes": upload_total - packet_bytes,
            "upload_total_bytes": upload_total, "offer_id": OFFER_ID,
            "files_in_packet": len(names)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.main_root.resolve(), args.out.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
