"""Build the frozen V5R2 project archive from the reviewed V4 source snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_fit_v5r2_20261009/packet"
)
V4_PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_capacity_v4_manifest_profile_20261008/packet"
)
BASE_SOURCE_COMMIT = "ead2eb7ce4108bd1ae4fa1fd04b2d5469cfe656c"
TRAINER_MARKER = '    timer = StageTimer(args.profile_steps > 0, args.profile_warmup)\n'
DATA_READY_MARKER = (
    '    print("CAPACITY_DATA_READY " + json.dumps('
    '{"train_sequences": len(train_idx), "train_batches": len(train_loader), '
    '"eval_sequences": n_valid}), flush=True)\n'
)


def sha256_file(path: Path) -> str:
    """Hash a source file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_once(content: str, old: str, new: str, label: str) -> str:
    if content.count(old) != 1:
        raise ValueError(f"expected one stable source anchor for {label}")
    return content.replace(old, new, 1)


def _frozen_source(repo_root: Path, relative_path: str) -> str:
    """Read one source member from the declared immutable base commit."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{BASE_SOURCE_COMMIT}:{relative_path}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout


def _patch_trainer(content: str) -> str:
    if DATA_READY_MARKER in content:
        raise ValueError("trainer already has V5R2 data-ready instrumentation")
    return _replace_once(
        content,
        TRAINER_MARKER,
        DATA_READY_MARKER + TRAINER_MARKER,
        "trainer data-ready marker",
    )


def _patch_runner(content: str) -> str:
    # V5R2: V5R1 refused admission because cgroup memory.current still held ~28 GB of reclaimable page
    # cache from hashing the 35 GB raw inputs.  Count only non-file (unreclaimable) cgroup usage.
    content = _replace_once(
        content,
        "            available = min(available, capacity - int(current.read_text().strip()))\n",
        "            used = int(current.read_text().strip())\n"
        "            stat_path = Path(\"/sys/fs/cgroup/memory.stat\")\n"
        "            if stat_path.exists():\n"
        "                stat = dict(line.split() for line in stat_path.read_text().splitlines())\n"
        "                used -= min(used, int(stat.get(\"file\", 0)))\n"
        "            available = min(available, capacity - used)\n",
        "page-cache-aware cgroup availability",
    )
    content = _replace_once(
        content,
        "WORKER_TIMEOUT_CEILING_SECONDS = 5_200",
        "WORKER_TIMEOUT_CEILING_SECONDS = 26_000",
        "fit worker deadline ceiling",
    )
    content = _replace_once(
        content,
        "def admit_specs(specs: list[dict], host: dict, devices: list[dict]) -> int:",
        "def admit_specs(\n"
        "    specs: list[dict],\n"
        "    host: dict,\n"
        "    devices: list[dict],\n"
        "    *,\n"
        "    maximum_preexisting_gpu_used_bytes: int = 128 * 1024**2,\n"
        ") -> int:",
        "explicit V4 T4 baseline memory bound",
    )
    content = _replace_once(
        content,
        'if device["used_bytes"] > 128 * 1024**2:',
        'if device["used_bytes"] > maximum_preexisting_gpu_used_bytes:',
        "explicit measured GPU baseline gate",
    )
    content = _replace_once(
        content,
        '                        "CUDA_VISIBLE_DEVICES": device["uuid"],\n',
        '                        "CUDA_VISIBLE_DEVICES": device["uuid"],\n'
        '                        "WNN_PHYSICAL_GPU_UUID": str(device["uuid"]),\n'
        '                        "WNN_PHYSICAL_GPU_INDEX": str(device["index"]),\n',
        "physical GPU identity propagation to arm worker",
    )
    content = _replace_once(
        content,
        "def run(root: Path, spec_paths: list[Path], output: Path, stagger_seconds: float = 30) -> None:",
        "def run(\n"
        "    root: Path,\n"
        "    spec_paths: list[Path],\n"
        "    output: Path,\n"
        "    stagger_seconds: float = 30,\n"
        "    *,\n"
        "    required_concurrency: int | None = None,\n"
        "    maximum_preexisting_gpu_used_bytes: int = 128 * 1024**2,\n"
        "    stop_on_worker_failure: bool = True,\n"
        ") -> None:",
        "fit concurrency and baseline controls",
    )
    content = _replace_once(
        content,
        "concurrency = admit_specs(specs, host, devices)",
        "concurrency = admit_specs(\n"
        "        specs,\n"
        "        host,\n"
        "        devices,\n"
        "        maximum_preexisting_gpu_used_bytes=maximum_preexisting_gpu_used_bytes,\n"
        "    )\n"
        "    if required_concurrency is not None and (\n"
        "        isinstance(required_concurrency, bool)\n"
        "        or required_concurrency < 1\n"
        "        or concurrency < required_concurrency\n"
        "    ):\n"
        "        raise ValueError(\"measured host/device capacity cannot admit required concurrency\")",
        "required concurrency admission gate",
    )
    content = _replace_once(
        content,
        'spec["limits"]["timeout_seconds"] = resolved_timeout',
        'spec["limits"]["timeout_seconds"] = resolved_timeout\n'
        '                env["WNN_ARM_DEADLINE_MONOTONIC"] = repr(\n'
        "                    launch_monotonic + resolved_timeout\n"
        "                )",
        "per-arm watchdog deadline binding",
    )
    content = _replace_once(
        content,
        '"peak_gpu_bytes": 0,\n                    }',
        '"peak_gpu_bytes": 0,\n'
        '                        "phase_offset": 0,\n'
        '                        "latest_phase": None,\n'
        '                    }',
        "safe phase relay state",
    )
    content = _replace_once(
        content,
        '"peak_gpu_bytes": job["peak_gpu_bytes"],\n'
        '                        "artifacts": export_artifacts(root, spec, job["output"]),',
        '"peak_gpu_bytes": job["peak_gpu_bytes"],\n'
        '                        "worker_phase": job.get("latest_phase"),\n'
        '                        "artifacts": export_artifacts(root, spec, job["output"]),',
        "terminal phase record",
    )
    content = _replace_once(
        content,
        '                    if record["status"] == "failed":\n'
        '                        raise RuntimeError(f"Worker failed: {spec[\'name\']}")',
        '                    if record["status"] == "failed":\n'
        '                        incident = incident or f"Worker failed: {spec[\'name\']}"\n'
        '                        if stop_on_worker_failure:\n'
        '                            raise RuntimeError(incident)',
        "preserve independent arm on sibling failure",
    )
    content = _replace_once(
        content,
        '            if now - last_heartbeat >= 30:\n'
        '                heartbeat = resource_heartbeat(active, finished, queue, now)',
        '            if now - last_heartbeat >= 30:\n'
        '                for job in active:\n'
        '                    job["phase_offset"], job["latest_phase"] = read_worker_phase(\n'
        '                        job["log"].name, job["phase_offset"], job["spec"]["name"]\n'
        '                    )\n'
        '                heartbeat = resource_heartbeat(active, finished, queue, now)',
        "sanitized fit phase heartbeat",
    )
    content = _replace_once(
        content,
        '                "peak_gpu_bytes": job["peak_gpu_bytes"],\n'
        '            }\n'
        '            for job in active',
        '                "peak_gpu_bytes": job["peak_gpu_bytes"],\n'
        '                "worker_phase": job.get("latest_phase"),\n'
        '            }\n'
        '            for job in active',
        "heartbeat phase fields",
    )
    helper = '''\n\ndef read_worker_phase(log_path: str, offset: int, expected_arm: str) -> tuple[int, dict | None]:
    """Read only the latest allowlisted, sanitized capacity phase marker."""
    phases = {"bootstrap", "source_verify", "input_mount", "input_verify", "profile", "profile_gate", "fit", "checkpoint_validate", "prediction_export", "prediction_validate", "model_import", "data_load", "terminal"}
    statuses = {"running", "pass", "fail", "skipped", "timeout", "profile_only"}
    arms = {"muon512", "muon768", "cohort"}
    numeric = {"elapsed_s", "remaining_s", "batch", "step_wall_ms", "optimizer_attempts", "completed_updates", "amp_skips", "gpu_peak_used_mb", "gpu_min_free_mb", "fit_projection_s", "export_reserve_s", "worker_cap_s", "exit_code", "epoch", "train_sequences", "train_batches", "eval_sequences"}
    latest = None
    current = offset
    with open(log_path, "rb") as stream:
        stream.seek(offset)
        chunk = stream.read()
    lines = chunk.splitlines(keepends=True)
    for line in lines:
        if not line.endswith(b"\\n"):
            break
        current += len(line)
        prefix = b"GRU_CAPACITY_PHASE "
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix):].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not {"phase", "arm", "status", "elapsed_s"}.issubset(payload):
            continue
        if set(payload) - ({"phase", "arm", "status"} | numeric):
            continue
        if payload.get("phase") not in phases or payload.get("status") not in statuses:
            continue
        if payload.get("arm") not in arms or payload.get("arm") not in {expected_arm, "cohort"}:
            continue
        valid = True
        for key in set(payload) & numeric:
            value = payload[key]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                valid = False
                break
        if not valid:
            continue
        latest = payload
    return current, latest
'''
    return content + helper


def _archive_source(packet_root: Path, repo_root: Path) -> tuple[Path, Path]:
    """Extract V4, apply instrumentation-only runner changes and seal a V5R2 archive."""
    v4_root = repo_root / V4_PACKET_RELATIVE
    v4_archive = v4_root / "source/source_archive.tar"
    v5_source_root = packet_root / "source"
    archive_path = v5_source_root / "source_archive.tar"
    manifest_path = v5_source_root / "source_archive_manifest.json"
    base_members = {}
    with tempfile.TemporaryDirectory(prefix="gru-capacity-v5-archive-") as temporary:
        extracted = Path(temporary)
        with tarfile.open(v4_archive, "r:") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    raise ValueError("V4 source archive contains a nonregular member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("V4 source member cannot be read")
                target = extracted / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.read())
                base_members[member.name] = target

        trainer_path = extracted / "scripts/train_rnn.py"
        trainer = trainer_path.read_text(encoding="utf-8")
        trainer_path.write_text(_patch_trainer(trainer), encoding="utf-8")

        files_to_add = {
            "scripts/kaggle_common.py": _frozen_source(repo_root, "scripts/kaggle_common.py"),
            "scripts/kaggle_gpu_runner.py": _patch_runner(
                _frozen_source(repo_root, "scripts/kaggle_gpu_runner.py")
            ),
            "scripts/capacity_arm_worker.py": packet_root / "capacity_arm_worker.py",
        }
        for name, source in files_to_add.items():
            target = extracted / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8") if isinstance(source, str) else target.write_bytes(source.read_bytes())
            base_members[name] = target

        members = []
        for name in sorted(base_members):
            path = base_members[name]
            members.append(
                {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        manifest = {
            "schema": "wnn-gru-capacity-source-archive-v1",
            "parent_packet": "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V4_20261008",
            "parent_source_commit": BASE_SOURCE_COMMIT,
            "instrumentation_only_changes": [
                "CAPACITY_DATA_READY marker in archived scripts/train_rnn.py",
                "fit deadline, measured T4 baseline, required-concurrency and allowlisted phase relay in scripts/kaggle_gpu_runner.py",
                "new independent process entrypoint scripts/capacity_arm_worker.py",
            ],
            "member_count": len(members),
            "members": members,
        }
        v5_source_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(base_members):
                path = base_members[name]
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return archive_path, manifest_path


def main() -> None:
    """Rebuild the V5R2 source archive without reading mounted training arrays."""
    script_path = Path(__file__).resolve()
    packet_root = script_path.parent
    repo_root = next(
        parent for parent in script_path.parents if (parent / "AGENTS.md").is_file()
    )
    _archive_source(packet_root, repo_root)


if __name__ == "__main__":
    main()
