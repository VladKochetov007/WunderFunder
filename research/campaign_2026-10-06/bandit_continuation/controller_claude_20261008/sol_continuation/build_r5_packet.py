from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OUT = Path(__file__).resolve().parent
DEST = OUT / "r5_packet"
BASE = Path(
    "/home/vlad/development/wunder/worktrees/wnn-gru512-768-muon-fit-v5-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/gru512_768_muon_fit_v5r3_skip_accounting_20261008/packet"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name in (
        "capacity_worker.py",
        "capacity_fit_worker.py",
        "verified_digest_handoff.py",
        "train_entry.py",
        "fold0.json",
        "portable_inputs.json",
        "GRU512_768_MUON_FIT_RECIPE_V5R3.json",
    ):
        shutil.copyfile(BASE / name, DEST / name)
    for directory in ("source", "producer_evidence/v4"):
        shutil.copytree(BASE / directory, DEST / directory, dirs_exist_ok=True)
    base = json.loads((BASE / "GRU512_768_MUON_FIT_RECIPE_V5R3.json").read_text())
    recipe = {
        "packet_id": "E529_CAPACITY_VAST_R5_SEQUENTIAL_SEED9_FOLD0_V1",
        "base_recipe_sha256": sha(BASE / "GRU512_768_MUON_FIT_RECIPE_V5R3.json"),
        "science": "unchanged sequential fresh Muon512/Muon768 seed9 B8/four epochs/fixedep3, source/target/data/split/sampler unchanged from V5R3",
        "device_adaptation": "3090/4090 actual-device 1warmup+2timed-batch smoke per arm, resolved native BF16 orthogonalization policy unchanged; no smoke weights carry into fit",
        "profile_reuse": "successful 230-update T4 V4 profiles retained as producer evidence; no full profile repeat",
        "delivery": "fixedep3 checkpoint+provenance+history; exact468 FP32 feature-only validation predictions; complete train and valid FP32 teacher predictions in actual parquet row-group order; raw receipts/logs/resources; independent grader follows",
        "runtime": "10h hardstop from root create, bounded worker deadline; sequential arm budget from actual-device timing, preserve partials and terminate on empty useful queue",
        "warm_data": "one verified physical input set; per-arm independent subprocess/model/optimizer; parsed loader not shared; OS cache reuse not claimed",
        "export_reserve_seconds": 5400,
        "max_artifact_bytes": 5000000000,
        "export_fitting_change": False,
        "inputs": base["inputs"]["source_members"],
        "exposure": "fold0 exposed development; no public forecast",
        "authority": "root pending exact source/recipe/spec/admission, no provider mutation by builder",
    }
    (DEST / "CAPACITY_R5_RECIPE.json").write_text(json.dumps(recipe, indent=2) + "\n")
    packet_files = {
        str(path.relative_to(DEST)): sha(path)
        for path in sorted(DEST.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name not in {"worker_spec.json", "ROOT_AUTHORIZATION.json"}
    }
    spec = {
        "schema_version": 1,
        "packet_id": recipe["packet_id"],
        "inputs": base["inputs"]["source_members"],
        "packet_files": packet_files,
        "expected_image": "vastai/pytorch@sha256:450710d89268fcd55632f28fdd80508ee920a53075b3f8e9f828bdc154e3d856",
        "root_staging_guard_sha256": sha(ROOT / "scripts/vast_campaign.py"),
        "root_prefetch_helper_sha256": sha(ROOT / "scripts/kaggle_vast_prefetch.py"),
        "root_prefetch_remote_sha256": sha(ROOT / "scripts/kaggle_vast_prefetch_remote.py"),
        "max_laptop_upload_bytes": 99999999,
        "max_artifact_bytes": 5000000000,
        "max_worker_host_ram_gb": 28,
    }
    (DEST / "worker_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (DEST / "ROOT_AUTHORIZATION.json").write_text(
        json.dumps(
            {
                "authorized": False,
                "attempt_id": "pending_root",
                "recipe_sha256": sha(DEST / "CAPACITY_R5_RECIPE.json"),
                "worker_spec_sha256": sha(DEST / "worker_spec.json"),
            },
            indent=2,
        )
        + "\n"
    )
    archive_path = OUT / "r5_capacity_packet_v3.tar"
    with tarfile.open(archive_path, "w") as archive:
        for path in sorted(DEST.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.add(path, arcname=str(path.relative_to(DEST)), recursive=False)
    small_bytes = archive_path.stat().st_size + sum(
        row["bytes"]
        for path, row in spec["inputs"].items()
        if not path.startswith("datasets/")
        and path not in {"cache/train_targets.npz", "cache/valid_targets.npz"}
    )
    assert small_bytes < 99999999
    report = {
        "status": "sealed_pre_admission_packet",
        "archive_sha256": sha(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "recipe_sha256": sha(DEST / "CAPACITY_R5_RECIPE.json"),
        "worker_spec_sha256": sha(DEST / "worker_spec.json"),
        "laptop_closure_bytes": small_bytes,
        "root_guard_sha256": spec["root_staging_guard_sha256"],
        "root_prefetch_remote_sha256": spec["root_prefetch_remote_sha256"],
        "authority_pending": True,
        "archive_path": str(archive_path),
    }
    (OUT / "R5_PACKET_BUILD_R3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
