"""Run the exact clean F0 propensity fit after shared preflight gates."""

from __future__ import annotations

import datetime as dt
import shutil

# Executed after the preflight module in one shared namespace.
# ruff: noqa: F821


def main() -> None:
    """Run fresh source, data, resource, and unit admission gates before launch."""
    if PAYLOAD.get("source_commit") != COMMIT or PAYLOAD["fit_arguments"] != FIT_ARGS:
        raise ValueError("fit source or literal CLI differs from the frozen prerequisite")
    if (
        os.path.lexists(OUTPUT)
        or os.path.lexists(RUN_DIR)
        or (B / "runtime/STOP_ADMISSION.json").exists()
    ):
        raise RuntimeError("output/run exists or campaign admission is stopped")
    if (
        _run(["systemctl", "--user", "show", "--all", UNIT, "-p", "LoadState", "--value"]).strip()
        != "not-found"
    ):
        raise RuntimeError("exact clean-fit unit already exists")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise FileNotFoundError("pinned project's Python interpreter is unavailable")
    if any(
        path.is_symlink() or not path.is_dir()
        for path in (B, SOURCE, DATA, B / "research_tmp", RUN_DIR.parent)
    ):
        raise RuntimeError("expected real working directory is unavailable")
    observed, profile = _pins(), None
    profile = _profile(observed)
    if _sha(SPLIT) != SPLIT_SHA or shutil.disk_usage(B / "research_tmp").free < 20_000_000_000:
        raise RuntimeError("fold split changed or scratch has less than 20 GB available")
    fs = _run(
        ["findmnt", "-T", str(B / "research_tmp"), "-n", "-o", "FSTYPE,SOURCE,TARGET"]
    ).strip()
    if fs.split()[0] in {"tmpfs", "ramfs"}:
        raise RuntimeError("training scratch is not on a physical filesystem")
    data, parent, populated = _data(), *_parent()
    gpu, foreign = _gpu()
    if os.path.lexists(OUTPUT) or os.path.lexists(RUN_DIR):
        raise FileExistsError("fit output path appeared during preflight")
    if (
        _run(["systemctl", "--user", "show", "--all", UNIT, "-p", "LoadState", "--value"]).strip()
        != "not-found"
    ):
        raise RuntimeError("exact clean-fit unit appeared during preflight")
    OUTPUT.mkdir(mode=0o700)
    admission = dict(
        schema="clean_propensity_fit_admission.v1",
        created_utc=dt.datetime.now(dt.UTC).isoformat(),
        unit=UNIT,
        run_name=RUN_NAME,
    )
    admission.update(
        dict(
            source_commit=COMMIT,
            source_file_count=len(observed),
            source_hashes=observed,
            profile_result_sha256=PAYLOAD["profile_result_sha256"],
        )
    )
    admission.update(
        dict(
            profile_attempt_id=profile["attempt_id"],
            source_manifest_sha256=PAYLOAD["source_manifest_sha256"],
            profile_acceptance_sha256=PAYLOAD["local_profile_acceptance_sha256"],
            profile_review_sha256=PAYLOAD["review_sha256"],
        )
    )
    admission.update(
        dict(
            split_manifest_sha256=SPLIT_SHA,
            portable_source_registry_sha256=REGISTRY_SHA,
            data_identity_stats_only=data,
            parent=parent,
            populated_campaign_descendants=populated,
            gpu=gpu,
            gpu_processes_preserved=foreign,
            scratch_filesystem=fs,
        )
    )
    admission.update(
        dict(
            fit_reservation={
                "cpu_quota": "100%",
                "memory_high": 7_500_000_000,
                "memory_max": FIT_MEMORY,
                "swap_max": 0,
                "runtime_max_seconds": 86_400,
            },
            scope="one fresh F0 propensity prerequisite; no teacher/student admission or campaign completion claim",
        )
    )
    _write(OUTPUT / "ADMISSION.json", admission)
    env = dict(
        item.split("=", 1)
        for item in f"""PYTHONPATH={SOURCE}/src:{SOURCE}/scripts|PYTHONDONTWRITEBYTECODE=1|WNN_PROJECT_ROOT={DATA}|WNN_PORTABLE_DATA_MANIFEST={REGISTRY}|WNN_GPU_MEMORY_BYTES=3000000000|WNN_LOADER_WORKERS=0|WNN_LOADER_RAM_GB=1|WNN_THREADS=1|OMP_NUM_THREADS=1|OPENBLAS_NUM_THREADS=1|MKL_NUM_THREADS=1|CUDA_VISIBLE_DEVICES={GPU}|NVIDIA_TF32_OVERRIDE=0|CUBLAS_WORKSPACE_CONFIG=:4096:8|TMPDIR={B}/research_tmp|WNN_TMP_ROOT={B}/research_tmp""".split(
            "|"
        )
    )
    command = [
        "systemd-run",
        "--user",
        f"--unit={UNIT}",
        *[
            "--property=Type=exec",
            "--property=RemainAfterExit=yes",
            "--property=Slice=wnn-campaign.slice",
            "--property=CPUQuota=100%",
            "--property=MemoryHigh=7500000000",
            "--property=MemoryMax=8000000000",
            "--property=MemorySwapMax=0",
            "--property=RuntimeMaxSec=86400",
            "--property=TimeoutStopSec=60",
        ],
        f"--property=StandardOutput=append:{OUTPUT}/unit.log",
        f"--property=StandardError=append:{OUTPUT}/unit.log",
        f"--working-directory={SOURCE}",
    ]
    command.extend("--setenv=" + key + "=" + value for key, value in env.items())
    command += [str(PYTHON), "-u", str(SOURCE / "scripts/train_propensity.py"), *FIT_ARGS]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        submitted = dict(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            outcome="submitted" if result.returncode == 0 else "unknown_or_failed",
        )
    except subprocess.TimeoutExpired as error:
        submitted = dict(
            returncode=None,
            stdout=_text(error.stdout),
            stderr=_text(error.stderr),
            outcome="unknown",
        )
    props = ["InvocationID", "ExecStart", "ActiveState", "SubState", "MainPID", "ControlGroup"]
    show = subprocess.run(
        ["systemctl", "--user", "show", "--all", UNIT, *sum((["-p", key] for key in props), [])],
        capture_output=True,
        text=True,
        timeout=10,
    )
    launch = dict(
        schema="clean_propensity_fit_launch.v1",
        created_utc=dt.datetime.now(dt.UTC).isoformat(),
        unit=UNIT,
        command=command,
        submission=submitted,
        systemctl_after_submission=dict(
            returncode=show.returncode, stdout=show.stdout, stderr=show.stderr
        ),
        unknown_outcome_policy="never resubmit; inspect this unit and run directory first",
    )
    _write(OUTPUT / "LAUNCH.json", launch)
    print(
        json.dumps(
            {
                name: base64.b64encode((OUTPUT / name).read_bytes()).decode("ascii")
                for name in ("ADMISSION.json", "LAUNCH.json")
            }
        )
    )
    if submitted["returncode"] != 0:
        raise SystemExit(2)
