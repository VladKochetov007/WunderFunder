"""Run bounded, offline integrity checks for the frozen H382 upload packet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRODUCER_PREDICTION = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/kaggle_long_20261010/"
    "output_v6/out/pred_kg_mu256_prop0_s5_valid.npy"
)
EXPECTED_JOBS = {
    "kg_h382_f0_muonall_prop2_e3_s3": (2.0, 3, 2),
    "kg_h382_f0_muonall_prop0_e3_s3": (0.0, 3, 2),
    "kg_h382_f0_muonall_prop0_e4_s3": (0.0, 4, 3),
}


def file_sha256(path: Path) -> str:
    """Hash a file with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_npy_header(path: Path) -> tuple[tuple[int, ...], str, bool, int]:
    """Read an NPY header without loading its potentially large array payload."""
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"not an NPY file: {path}")
        version = stream.read(2)
        if version == b"\x01\x00":
            header_size = int.from_bytes(stream.read(2), "little")
        elif version in (b"\x02\x00", b"\x03\x00"):
            header_size = int.from_bytes(stream.read(4), "little")
        else:
            raise ValueError(f"unsupported NPY version {version!r}: {path}")
        if header_size > 1_000_000:
            raise ValueError(f"unreasonable NPY header size {header_size}: {path}")
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"truncated NPY header: {path}")
        header = ast.literal_eval(header_bytes.decode("latin1").strip())
        payload_offset = stream.tell()
    return tuple(header["shape"]), str(header["descr"]), bool(header["fortran_order"]), payload_offset


def require(condition: bool, message: str) -> None:
    """Raise a descriptive validation error for an invalid packet invariant."""
    if not condition:
        raise ValueError(message)


def validate_prediction(path: Path) -> dict[str, object]:
    """Verify a completed producer prediction against its real sidecar receipt."""
    sidecar = path.with_suffix(".json")
    require(path.is_file(), f"actual producer prediction is missing: {path}")
    require(sidecar.is_file(), f"actual producer sidecar is missing: {sidecar}")
    shape, dtype, fortran_order, offset = read_npy_header(path)
    require(shape == (1873, 20_000, 2), f"producer shape differs: {shape}")
    require(dtype in ("<f4", "=f4"), f"producer dtype differs: {dtype}")
    require(not fortran_order, "producer NPY is not C-order")
    require(path.stat().st_size == offset + 1873 * 20_000 * 2 * 4, "producer NPY size differs")
    prediction_sha = file_sha256(path)
    metadata: dict[str, Any] = json.loads(sidecar.read_text())
    require(metadata.get("split") == "valid", "producer is not a validation export")
    require(metadata.get("prediction_sha256") == prediction_sha, "producer sidecar prediction hash differs")
    require(metadata.get("row_groups") == list(range(1873)), "producer sidecar does not cover all valid groups")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": prediction_sha,
        "shape": list(shape),
        "dtype": dtype,
        "sidecar_sha256": file_sha256(sidecar),
        "producer_run": metadata.get("run"),
        "producer_epoch": metadata.get("epoch"),
        "producer_split_manifest_sha256": metadata.get("split_manifest_sha256"),
    }


def parse_flags(argv: list[str]) -> tuple[dict[str, str], set[str]]:
    """Parse the simple, frozen flag/value argv used by the H382 jobs."""
    values: dict[str, str] = {}
    switches: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        require(token.startswith("--"), f"unexpected non-flag in trainer argv: {token}")
        if token in {"--no-eval-during-fit", "--fast", "--save-training-state"}:
            switches.add(token)
            index += 1
            continue
        require(index + 1 < len(argv), f"flag lacks a value: {token}")
        require(token not in values, f"duplicate flag: {token}")
        values[token] = argv[index + 1]
        index += 2
    return values, switches


def validate_jobs(jobs: dict[str, Any], packet: dict[str, Any]) -> None:
    """Check exact cohort recipes, data membership and execution ceilings."""
    require(jobs.get("packet_id") == packet["packet_id"], "job packet ID differs")
    require(jobs.get("bundle_id") == packet["bundle_id"], "job bundle ID differs")
    require(jobs.get("split_manifest_sha256") == packet["split_manifest_sha256"], "split hash differs")
    require(jobs.get("portable_registry_sha256") == packet["portable_registry_sha256"], "registry hash differs")
    require(jobs.get("asset_manifest_sha256") == packet["asset_manifest_sha256"], "asset manifest hash differs")
    require(jobs.get("fit_membership") == packet["fit_membership"], "fit membership differs")
    require(len(jobs.get("slots", [])) == 2, "H382 expects two execution slots")
    require(jobs.get("cohort_runtime_seconds") == 14_400, "cohort hard wall differs")
    observed_names: set[str] = set()
    for job in jobs["jobs"]:
        name = job["name"]
        require(name in EXPECTED_JOBS and name not in observed_names, f"unexpected/duplicate job: {name}")
        observed_names.add(name)
        expected_prop, expected_epochs, expected_epoch = EXPECTED_JOBS[name]
        require(job["prop_power"] == expected_prop, f"{name}: propensity power differs")
        require(job["epochs"] == expected_epochs, f"{name}: epoch count differs")
        require(job["checkpoint_epoch"] == expected_epoch, f"{name}: checkpoint epoch differs")
        require(len(job["steps"]) == 2, f"{name}: expected fit and export steps")
        fit, export = job["steps"]
        require(fit["label"] == "fit" and fit["timeout"] == 5400, f"{name}: fit cap differs")
        require(export["label"] == "export" and export["timeout"] == 1800, f"{name}: export cap differs")
        fit_argv = fit["argv"]
        require(fit_argv[:2] == ["python", "/root/wnn/scripts/train_rnn.py"], f"{name}: trainer entrypoint differs")
        require(fit_argv[2] == "--name" and fit_argv[3] == name, f"{name}: fit run name differs")
        values, switches = parse_flags(fit_argv[4:])
        expected_values = {
            "--device": "cuda",
            "--arch": "gru",
            "--cell": "gru",
            "--enc": "linear",
            "--hidden": "256",
            "--layers": "2",
            "--dropout": "0",
            "--seed": "3",
            "--split-manifest": f"/root/wnn/{packet['split_manifest_path']}",
            "--valid-fold": "0",
            "--eval-source": "vfold",
            "--train-source": "train+valid",
            "--train-seqs": "10607",
            "--epochs": str(expected_epochs),
            "--checkpoint-policy": "fixed",
            "--checkpoint-epoch": str(expected_epoch),
            "--batch": "32",
            "--chunk": "1000",
            "--loss": "wmse",
            "--loss-norm": "chunk",
            "--lr": "0.001",
            "--wd": "0.01",
            "--optim": "muon",
            "--muon-scope": "all",
            "--sched": "onecycle",
            "--precision": "fp16",
            "--propensity": "c20260929_prop_f0",
            "--prop-power": str(int(expected_prop)),
            "--prop-eps": "0.03",
            "--valid-true-mask": "0.03",
            "--aux-struct": "0.3",
            "--aux-next": "0.3",
            "--since-weight": "32",
        }
        require(values == expected_values, f"{name}: fit args differ from preregistered recipe")
        require(switches == {"--no-eval-during-fit", "--fast", "--save-training-state"}, f"{name}: fit switches differ")
        require(export["argv"] == [
            "python", "/root/wnn/scripts/predict_run.py", name, "valid", "--epoch", str(expected_epoch)
        ], f"{name}: export argv differs")
    require(observed_names == set(EXPECTED_JOBS), "H382 cohort job set differs")
    require(len(packet["cohort"]) == len(EXPECTED_JOBS), "packet cohort count differs")
    require(packet["fit_sequence_count"] == 11_940, "fit population differs")
    require(packet["fit_membership"]["train"]["count"] == 10_535, "train fit membership count differs")
    require(packet["fit_membership"]["valid"]["count"] == 1_405, "valid fit membership count differs")


def validate_packet(producer_prediction: Path | None) -> dict[str, object]:
    """Validate generated metadata, bundled source and exact runnable recipes."""
    packet: dict[str, Any] = json.loads((HERE / "H382_PACKET_R1.json").read_text())
    metadata: dict[str, Any] = json.loads((HERE / "kernel/kernel-metadata.json").read_text())
    bundle_path = HERE / "dataset/bundle.wnn"
    dataset_metadata: dict[str, Any] = json.loads((HERE / "dataset/dataset-metadata.json").read_text())
    kernel_path = HERE / "kernel/kernel.py"
    require(packet["schema"] == "wnn.kaggle-h382-packet.v1", "unknown packet schema")
    require(metadata["id"] == packet["packet_id"], "Kaggle kernel ID differs")
    require(metadata["title"] == packet["kernel_title"], "Kaggle kernel title differs")
    require(metadata["code_file"] == "kernel.py", "Kaggle kernel code file differs")
    slug = metadata["title"].lower().replace(" ", "-")
    require(slug == packet["kernel_slug"], "Kaggle title-derived slug differs")
    require(dataset_metadata["id"] == packet["bundle_id"], "bundle dataset ID differs")
    require(file_sha256(bundle_path) == packet["bundle_sha256"], "bundle content hash differs")
    require(bundle_path.stat().st_size == packet["bundle_bytes"], "bundle byte count differs")
    require(file_sha256(kernel_path) == packet["kernel_sha256"], "kernel source hash differs")
    require(kernel_path.stat().st_size == packet["kernel_bytes"], "kernel source size differs")
    require(file_sha256(HERE / "kernel/kernel-metadata.json") == packet["kernel_metadata_sha256"], "kernel metadata hash differs")
    require(file_sha256(HERE / "dataset/dataset-metadata.json") == packet["dataset_metadata_sha256"], "dataset metadata hash differs")
    require(packet["kernel_bytes"] < packet["limits"]["source_bootstrap_bytes_max"], "kernel exceeds source envelope")
    with tarfile.open(bundle_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "bundle contains duplicate members")
        require(all(member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts for member in members), "bundle contains unsafe members")
        source_bytes = archive.extractfile("h382_source_manifest.json").read()  # type: ignore[union-attr]
        jobs_bytes = archive.extractfile("kaggle_jobs.json").read()  # type: ignore[union-attr]
        source_manifest: dict[str, Any] = json.loads(source_bytes)
        jobs: dict[str, Any] = json.loads(jobs_bytes)
        require(hashlib.sha256(source_bytes).hexdigest() == packet["source_manifest_sha256"], "bundled source manifest hash differs")
        require(jobs["source_manifest_sha256"] == packet["source_manifest_sha256"], "job/source manifest binding differs")
        for entry in source_manifest["files"]:
            member = archive.extractfile(entry["path"])
            require(member is not None, f"source file missing from bundle: {entry['path']}")
            payload = member.read()
            require(len(payload) == entry["bytes"], f"source file size differs: {entry['path']}")
            require(hashlib.sha256(payload).hexdigest() == entry["sha256"], f"source file hash differs: {entry['path']}")
        require(len(source_manifest["files"]) == packet["source_file_count"], "source file count differs")
        by_path = {entry["path"]: entry for entry in source_manifest["files"]}
        for path_key, sha_key in (
            ("existing_recipe_audit_path", "existing_recipe_audit_sha256"),
            ("preregistration_path", "preregistration_sha256"),
        ):
            path = packet[path_key]
            require(path in by_path, f"bound campaign receipt missing from source manifest: {path}")
            entry = by_path[path]
            require(entry["sha256"] == packet[sha_key], f"campaign receipt hash differs: {path}")
            payload = archive.extractfile(path)
            require(payload is not None, f"campaign receipt missing from bundle: {path}")
            require(hashlib.sha256(payload.read()).hexdigest() == packet[sha_key], f"campaign receipt bytes differ: {path}")
        validate_jobs(jobs, packet)
    producer = validate_prediction(producer_prediction) if producer_prediction else None
    return {
        "packet_id": packet["packet_id"],
        "bundle_sha256": packet["bundle_sha256"],
        "kernel_sha256": packet["kernel_sha256"],
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "job_count": len(EXPECTED_JOBS),
        "producer_prediction": producer,
        "result": "pass",
    }


def main() -> None:
    """Validate a frozen packet, optionally including an actual producer export."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-prediction", type=Path, default=PRODUCER_PREDICTION)
    parser.add_argument("--skip-producer", action="store_true")
    args = parser.parse_args()
    result = validate_packet(None if args.skip_producer else args.producer_prediction)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
