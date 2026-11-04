from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from attrs import define, field

from scripts.tl3_finite17_results_validation import _verify_args, _verify_history
from wnn.heterogeneous_pool_sources import groups_hash, read_object, validate_source
from wnn.splits import CampaignSplit, file_sha256
from wnn.training.chronological_teacher import validate_chronological_teacher_membership

# Owned-worker/source/producer receipts are heterogeneous JSON, validated before use.
JsonObject = dict[str, Any]


def write_json(path: Path, value: object, *, replace: bool = False) -> None:
    """Create a receipt, or atomically update this controller's one status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if not replace:
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"preserved receipt differs: {path}")
            return
        with path.open("x", encoding="utf-8") as stream:
            stream.write(raw)
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pin(path: Path) -> JsonObject:
    """Bind original bytes without replacing the producer's source schema."""
    return {"path": str(path.resolve(strict=True)), "sha256": file_sha256(path)}


def ticks(pid: int) -> str | None:
    """Return Linux process birth identity, never infer ownership from PID alone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def source_binding(source: Path) -> JsonObject:
    """Verify the actual immutable all-Python source inventory before launching."""
    identity_path = source.with_name(source.name + ".identity.json")
    identity = read_object(identity_path)
    files = identity.get("files", {})
    if identity.get("schema") != "wnn.heterogeneous-source.v1" or not files:
        raise ValueError("missing immutable source inventory")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != identity.get("sha256"):
        raise ValueError("immutable source inventory digest differs")
    for relative, expected in files.items():
        path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source.resolve()) or file_sha256(path) != expected:
            raise ValueError(f"immutable source file differs: {relative}")
    return {"path": str(identity_path), "sha256": digest}


def gpu_snapshot() -> dict[int, JsonObject]:
    """Observe card identities and foreign occupancy; never terminate anything."""
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    cards = {}
    for line in text.splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(",")]
        cards[int(index)] = {
            "uuid": uuid,
            "memory_mib": int(memory),
            "utilization": int(utilization),
            "pids": [],
        }
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=20,
    )
    by_uuid = {card["uuid"]: card for card in cards.values()}
    for line in text.splitlines():
        if line.strip():
            uuid, pid = [value.strip() for value in line.split(",")]
            if uuid in by_uuid:
                by_uuid[uuid]["pids"].append(int(pid))
    return cards


@define
class Runtime:
    """Resource bindings for one fixed six-fit packet, not a generic scheduler."""

    root: Path
    source: Path
    python: Path
    binding: JsonObject
    deadline: float
    jobs: dict[str, JsonObject] = field(factory=dict)
    handles: dict[str, subprocess.Popen[bytes] | None] = field(factory=dict)
    helper_sha256: str = ""

    def job(
        self,
        key: str,
        script: str,
        args: list[str],
        *,
        gpu: int | None,
        phase: str,
        role: str,
        seconds: int,
    ) -> str:
        """Admit one explicit bounded phase once; preserve failures without retries."""
        status_path = self.root / "status" / f"{key}.json"
        request_path = self.root / "requests" / f"{key}.json"
        if key not in self.jobs and (request_path.exists() or status_path.exists()):
            if __package__:
                from scripts.heterogeneous_six_evidence import adopt_owned_job
            else:
                from heterogeneous_six_evidence import adopt_owned_job
            self.jobs[key] = adopt_owned_job(self, key, script, args, gpu, phase, role)
            self.handles[key] = None
        if key not in self.jobs:
            if time.monotonic() >= self.deadline:
                return "admission_closed"
            card = None if gpu is None else gpu_snapshot()[gpu]
            if card is not None and (
                card["pids"] or card["memory_mib"] > 128 or card["utilization"] > 5
            ):
                return "waiting_free_gpu"
            affinity = "56-63" if gpu is None else f"{gpu * 8}-{gpu * 8 + 7}"
            environment = {
                "CUDA_VISIBLE_DEVICES": "" if card is None else card["uuid"],
                "WNN_PROJECT_ROOT": str(self.root),
                "PYTHONPATH": f"{self.source / 'src'}:{self.source}",
                "WNN_THREADS": "4",
                "WNN_LOADER_WORKERS": "2",
                "WNN_LOADER_RAM_GB": "8",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "1",
                "ARROW_NUM_THREADS": "1",
                "CUBLAS_EMULATE_SINGLE_PRECISION": "0",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                "MALLOC_ARENA_MAX": "2",
            }
            command = [
                "taskset",
                "-c",
                affinity,
                str(self.python),
                "-u",
                str(self.source / "scripts" / script),
                *args,
            ]
            request = {
                "command": command,
                "cwd": str(self.source),
                "log": str(self.root / "logs" / f"{key}.log"),
                "env": environment,
                "wall_seconds": min(seconds, max(1, self.deadline - time.monotonic())),
                "phase": phase,
                "role": role,
                "source_identity": self.binding,
                "gpu_uuid": None if card is None else card["uuid"],
            }
            helper = self.root / "tools-r1/heterogeneous_owned_command.py"
            if file_sha256(helper) != self.helper_sha256:
                raise ValueError("owned-command runner changed")
            write_json(request_path, request)
            handle = subprocess.Popen(
                [
                    str(self.python),
                    str(helper),
                    "--request",
                    str(request_path),
                    "--output",
                    str(status_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.handles[key] = handle
            self.jobs[key] = {
                "status_path": str(status_path),
                "request_path": str(request_path),
                "runner_pid": handle.pid,
                "runner_start_ticks": ticks(handle.pid),
                "phase": phase,
                "role": role,
                "gpu": gpu,
            }
        if not status_path.exists():
            handle = self.handles[key]
            if handle is None or handle.poll() is not None:
                raise RuntimeError(f"owned runner exited without phase status: {key}")
            return "launching"
        status = read_object(status_path)
        if status.get("request_sha256") != file_sha256(request_path) or status.get(
            "source_identity"
        ) not in (self.binding, self.binding["sha256"]):
            raise ValueError(f"phase request/source binding changed: {key}")
        if (
            status.get("runner_pid") != self.jobs[key]["runner_pid"]
            or status.get("runner_start_ticks") != self.jobs[key]["runner_start_ticks"]
        ):
            raise ValueError(f"phase runner birth identity changed: {key}")
        if status.get("status") == "running":
            actual = ticks(status["pid"])
            if actual is not None and actual != status.get("start_ticks"):
                raise ValueError(f"worker PID was reused: {key}")
        self.jobs[key]["observed"] = status
        if status.get("status") in {"failed", "timed_out", "interrupted"}:
            raise RuntimeError(
                f"owned phase failed without retry: {key}: {status.get('error', status.get('return_code'))}"
            )
        return status["status"]


def legacy_record(
    root: Path, output: Path, run: str, family: str, coefficient: float, split: CampaignSplit
) -> JsonObject:
    """Qualify actual legacy export/provenance, retaining its native schema."""
    directory = root / "artifacts/runs" / run
    prediction = root / "cache" / f"pred_{run}_train.npy"
    sidecar = prediction.with_suffix(".json")
    provenance = read_object(directory / "provenance.json")
    record = {
        "run": run,
        "family": family,
        "epoch": 3,
        "coefficient": coefficient,
        "producer": "legacy_rnn_chronological",
        "prediction": pin(prediction),
        "sidecar": pin(sidecar),
        "checkpoint": pin(directory / "model.pt"),
        "provenance": pin(directory / "provenance.json"),
    }
    qualification = {
        "schema": "wnn.teacher-source-qualification.v1",
        "run": run,
        "producer": record["producer"],
        "epoch": 3,
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "prediction_sha256": record["prediction"]["sha256"],
        "sidecar_sha256": record["sidecar"]["sha256"],
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": []},
        "evaluation_valid_groups": list(split.evaluation_groups),
        "source_sha256": provenance["trainer_implementation"]["sha256"],
        "normalization_fit_groups_sha256": groups_hash(list(split.train_groups)),
    }
    path = output / f"{run}.json"
    write_json(path, qualification)
    record["qualification"] = pin(path)
    validate_source(record, split)
    return record


def verify_fit(root: Path, source: Path, node: JsonObject, split: CampaignSplit) -> JsonObject:
    """Verify the actual fixed endpoint and full four-epoch update/validation history."""
    directory = root / "artifacts/runs" / node["run"]
    provenance = read_object(directory / "provenance.json")
    _verify_args({}, node, provenance.get("args"))
    history = json.loads((directory / "history.json").read_text())
    result = _verify_history(history, node, list(split.train_groups))
    implementation = provenance["trainer_implementation"]
    if any(
        file_sha256(source / name) != expected for name, expected in implementation["files"].items()
    ):
        raise ValueError("fit producer source differs from its immutable source")
    artifacts = {
        name: pin(directory / name)
        for name in ("model.pt", "model_ep3.pt", "history.json", "provenance.json")
    }
    if any(
        provenance["checkpoint_sha256"].get(name) != artifacts[name]["sha256"]
        for name in ("model.pt", "model_ep3.pt")
    ):
        raise ValueError("fixed endpoint bytes disagree with actual fit provenance")
    validate_chronological_teacher_membership(provenance, split, node["run"])
    return {
        "artifacts": artifacts,
        "metrics": result,
        "training_source_sha256": implementation["sha256"],
    }
