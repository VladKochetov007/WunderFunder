from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

# Reviewed h217_packet_v5/recipe.json bytes. This literal is the guard pin.
# Do not replace it with a hash computed from the recipe file or a manifest.
EXPECTED_RECIPE_SHA256 = "c18b96fda798ead5cffbb089bd79589c5ce3aa5149cc81a49d2b4eb8985c9eac"
ARMS = ("tcn_full128", "tcn_full256")
ARM_HIDDEN = {"tcn_full128": "128", "tcn_full256": "256"}
RUN_NAMES = {
    "tcn_full128": "bd26_h217_tcn_full128_s7_f0_v5_amp4096",
    "tcn_full256": "bd26_h217_tcn_full256_s7_f0_v5_amp4096",
}
PROFILE_RUN_NAMES = {arm: f"{name}_profile_b8" for arm, name in RUN_NAMES.items()}
OUTPUT_NAMES = {
    "tcn_full128": "h217_v5_tcn_full128_s7_f0_ep3",
    "tcn_full256": "h217_v5_tcn_full256_s7_f0_ep3",
}
PROFILE_SECONDS = 900
FIT_SECONDS = 7200
VALID_EXPORT_SECONDS = 1800
AMP_INITIAL_SCALE = 4096.0
FITTED_GROUPS = 11940
EXCLUDED_EVALUATION_GROUPS = 468
FIT_TRAIN_GROUPS = 10535
FIT_VALID_GROUPS = 1405
FEATURE_VIEW = {
    "event_channels": 99,
    "heuristics": False,
    "input_width": 211,
    "online_norm": False,
    "raw_columns": 112,
    "raw_only": False,
    "research_features": "none",
}
MEMBERSHIP = {
    "excluded_evaluation_groups": EXCLUDED_EVALUATION_GROUPS,
    "fit_train": FIT_TRAIN_GROUPS,
    "fit_valid": FIT_VALID_GROUPS,
    "fitted_groups": FITTED_GROUPS,
    "fold": 0,
    "split_manifest": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
}
REQUIRED_SOURCE_HASHES = {
    "research/campaign_2026-09-29/artifacts/splits/fold0.json": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "research/campaign_2026-10-04/portable_inputs.json": "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
    "scripts/predict_run.py": "de4da30aebba33cb72a535b516839e4e627d00f6d6943cb4b2acbca7440b7fee",
    "scripts/train_rnn.py": "affae686fb97e39a2bfe49a96c2124ee2a66d30374a24ab1fe0a8ede9824bf5b",
    "src/wnn/amp.py": "190688ecc856fedf9698ef82d79864f33566bb5f6f1f7a92ce77ae27fcd4ed53",
    "src/wnn/data.py": "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00",
    "src/wnn/features.py": "284c119d150b74bff0ccc2628bc2a195ca61c351a628b085b7d9dd85512bfeee",
    "src/wnn/gmetric.py": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
    "src/wnn/nn.py": "9293f8f4e2a2f3bd9f0157f8f83f12a6a758608dd03c7e141c290d82d4d5aeca",
}
SEALED_INPUT_HASHES = {
    "cache/chain_folds.npz": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
    "cache/chains.npz": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
    "cache/train_targets.npz": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
    "cache/valid_targets.npz": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
    "datasets/train.parquet": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    "datasets/valid.parquet": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
}
FORBIDDEN_ARGV = (
    "--raw-only",
    "--cell",
    "--heuristics",
    "--online-norm",
    "--research-features",
    "--include-heldout-valid",
    "--fit-groups-manifest",
)


def validate_incoming_runtime(value: str, cohort_ceiling_seconds: int) -> int:
    """Accept a positive controller runtime no greater than the reviewed ceiling."""
    try:
        requested_seconds = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("incoming worker runtime must be a positive integer") from error
    if requested_seconds <= 0:
        raise ValueError("incoming worker runtime must be positive")
    if cohort_ceiling_seconds <= 0 or requested_seconds > cohort_ceiling_seconds:
        raise ValueError("incoming worker runtime exceeds the reviewed cohort ceiling")
    return requested_seconds


def cohort_phase_deadline(
    started_at: float,
    incoming_runtime_seconds: int,
    setup_reserve_seconds: int,
    packaging_reserve_seconds: int,
) -> float:
    """Reserve setup and terminal packaging from the actual controller deadline."""
    if setup_reserve_seconds < 0 or packaging_reserve_seconds < 0:
        raise ValueError("runtime reserves cannot be negative")
    return started_at + incoming_runtime_seconds - setup_reserve_seconds - packaging_reserve_seconds


def projected_delivery_budget_seconds(profile: dict[str, object]) -> int:
    """Return the profile-projected fit time plus the bounded valid-export phase."""
    projected_seconds = float(profile["projected_fit_and_export_seconds"])
    if not math.isfinite(projected_seconds) or projected_seconds <= 0:
        raise ValueError("profile delivery projection must be finite and positive")
    return math.ceil(projected_seconds) + VALID_EXPORT_SECONDS


def digest(path: Path) -> str:
    """Hash a file incrementally so large prediction exports stay out of RAM."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically persist a bounded worker receipt."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def stop(child: subprocess.Popen[bytes]) -> None:
    """Terminate only the process group created for one phase."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def run_phase(
    argv: list[str],
    name: str,
    timeout_seconds: int,
    project: Path,
    output: Path,
    deadline: float,
) -> dict[str, object]:
    """Run one phase with worker RSS, host RAM, VRAM, and deadline guards."""
    import psutil

    original_argv = list(argv)
    if len(argv) > 1 and argv[1].startswith("scripts/"):
        wrapper = (
            "import sys,runpy,pyarrow;pyarrow.set_cpu_count(1);"
            "pyarrow.set_io_thread_count(1);sys.argv=sys.argv[1:];"
            "sys.path.insert(0,'scripts');runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        argv = [argv[0], "-c", wrapper, *argv[1:]]

    started = time.monotonic()
    phase_deadline = min(deadline, started + timeout_seconds)
    ram_cap = int(os.environ["WNN_WORKER_RAM_BYTES"])
    vram_cap = int(os.environ["WNN_WORKER_VRAM_BYTES"])
    peak_rss = 0
    peak_vram = 0
    child: subprocess.Popen[bytes] | None = None
    error_text: str | None = None
    log_path = output / f"{name}.log"
    try:
        with log_path.open("wb") as log:
            child = subprocess.Popen(
                argv,
                cwd=project,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            while child.poll() is None:
                if time.monotonic() >= phase_deadline:
                    raise TimeoutError(f"{name} exceeded its phase or cohort deadline")
                try:
                    processes = [psutil.Process(child.pid)]
                    processes.extend(processes[0].children(recursive=True))
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                vram = sum(int(value) * 1024**2 for value in gpu.stdout.splitlines())
                peak_rss = max(peak_rss, rss)
                peak_vram = max(peak_vram, vram)
                heartbeat = {
                    "phase": name,
                    "elapsed_seconds": time.monotonic() - started,
                    "rss_bytes": rss,
                    "peak_rss_bytes": peak_rss,
                    "vram_bytes": vram,
                    "peak_vram_bytes": peak_vram,
                    "host_available_bytes": psutil.virtual_memory().available,
                }
                write_json(output / "heartbeat.json", heartbeat)
                if (
                    rss > ram_cap
                    or heartbeat["host_available_bytes"] < 2_000_000_000
                    or vram > vram_cap
                ):
                    raise MemoryError(f"resource guard stopped {name}: {heartbeat}")
                time.sleep(2)
    except Exception as error:  # Preserve phase errors so another arm can still run.
        error_text = f"{type(error).__name__}: {error}"
    finally:
        if child is not None and child.poll() is None:
            stop(child)

    exit_code = None if child is None else child.returncode
    if exit_code not in (0, None) and error_text is None:
        error_text = f"phase exited {exit_code}; see {log_path.name}"
    if child is None and error_text is None:
        error_text = "phase process did not start"
    return {
        "phase": name,
        "argv": original_argv,
        "subprocess_argv": argv,
        "exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "peak_vram_bytes": peak_vram,
        "log_sha256": digest(log_path),
        "error": error_text,
    }


def profile_gate(path: Path, fit: dict[str, object]) -> dict[str, object]:
    """Require exact finite AMP evidence and estimate fit time in outer batches."""
    rows = path.read_text().splitlines()
    config = [json.loads(row) for row in rows if row.startswith("{") and "amp_initial_scale" in row]
    evidence = [
        json.loads(row.split(" ", 1)[1])
        for row in rows
        if row.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    timings = [json.loads(row.split(" ", 1)[1]) for row in rows if row.startswith("PROFILE ")]
    if len(config) != 1 or len(evidence) != 1 or len(timings) != 1:
        raise ValueError("profile needs one config, timing, and update-evidence receipt")
    if (
        config[0].get("precision") != "fp16"
        or float(config[0].get("amp_initial_scale", -1)) != AMP_INITIAL_SCALE
    ):
        raise ValueError("profile did not use the declared FP16 AMP scale 4096")
    item = evidence[0]
    expected_counts = {
        "optimizer_attempts": 230,
        "optimizer_completed_updates": 230,
        "amp_skipped_updates": 0,
        "loss_checks": 230,
    }
    for name, expected in expected_counts.items():
        value = item.get(name)
        if type(value) is not int or value != expected:
            raise ValueError(f"profile update counter evidence requires {name}={expected}")
    required_true = ("loss_finite", "gradient_finite", "counts_consistent", "valid_for_throughput")
    if any(item.get(key) is not True for key in required_true):
        raise ValueError("profile failed strict finite-update evidence")
    if float(item.get("scaler_initial_scale", -1)) != AMP_INITIAL_SCALE:
        raise ValueError("effective GradScaler initial scale differs from the declared 4096")
    step_seconds = float(timings[0]["step_wall"]) / 1000.0
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("profile outer-batch time is invalid")
    estimate = math.ceil(FITTED_GROUPS / int(fit["batch_size"])) * 4 * step_seconds * 1.5 + 600
    if estimate > FIT_SECONDS:
        raise TimeoutError(f"projected four-epoch fit/export exceeds fit budget: {estimate:.1f}s")
    return {
        "status": "passed",
        "update_evidence": item,
        "outer_batch_step_wall_ms": timings[0]["step_wall"],
        "timing_unit": "one outer DataLoader batch, including its internal chunk loop",
        "projected_fit_and_export_seconds": round(estimate, 1),
    }


def _option(argv: list[str], key: str) -> str | None:
    """Return the value following one exact CLI flag."""
    try:
        return argv[argv.index(key) + 1]
    except (ValueError, IndexError):
        return None


def verify_source_provenance(project: Path, recipe: dict[str, object]) -> dict[str, str]:
    """Hash sealed trainer, model, feature, metric, split and registry files.

    Args:
        project: Packet project root copied from the sealed V4 source.
        recipe: Versioned V5 recipe whose declared hashes must match this worker.

    Returns:
        Relative path to observed SHA256 for every required sealed file.

    Raises:
        ValueError: A declared hash, file byte, or sealed input identity drifted.
    """
    declared = recipe.get("source_provenance")
    if not isinstance(declared, dict) or declared.get("required_sha256") != REQUIRED_SOURCE_HASHES:
        raise ValueError("recipe source provenance drifted from the sealed V4 project hashes")
    if declared.get("sealed_input_sha256") != SEALED_INPUT_HASHES:
        raise ValueError("recipe input provenance drifted from the sealed portable registry")
    observed: dict[str, str] = {}
    for relative, expected in REQUIRED_SOURCE_HASHES.items():
        path = project / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"sealed source is missing: {relative}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"sealed source hash mismatch: {relative}")
        observed[relative] = actual
    registry = json.loads(
        (project / "research/campaign_2026-10-04/portable_inputs.json").read_text()
    )
    if registry.get("version") != 1:
        raise ValueError("unsupported portable input registry version")
    if (
        registry.get("split_manifest_sha256")
        != REQUIRED_SOURCE_HASHES["research/campaign_2026-09-29/artifacts/splits/fold0.json"]
    ):
        raise ValueError("portable registry split hash differs from sealed fold0")
    sources = registry.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("portable registry has no sources")
    for relative, expected in SEALED_INPUT_HASHES.items():
        entry = sources.get(relative)
        if not isinstance(entry, dict) or entry.get("sha256") != expected:
            raise ValueError(f"sealed input hash missing from portable registry: {relative}")
    return observed


def verify_split_membership(project: Path, recipe: dict[str, object]) -> dict[str, int]:
    """Check frozen fold0 fit and evaluation groups before any phase starts.

    Args:
        project: Packet project root containing the sealed fold0 manifest.
        recipe: Versioned recipe whose membership block must match fold0.

    Returns:
        Observed fit and excluded group counts.

    Raises:
        ValueError: Membership, fold, overlap, or recipe declaration drifted.
    """
    if recipe.get("membership") != MEMBERSHIP:
        raise ValueError("recipe membership drifted from the sealed fold0 contract")
    path = project / MEMBERSHIP["split_manifest"]
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("fold") != 0:
        raise ValueError("sealed fold0 manifest identity drifted")
    fit_train = [int(group) for group in manifest["fit"]["train"]]
    fit_valid = [int(group) for group in manifest["fit"]["valid"]]
    excluded = [int(group) for group in manifest["evaluation"]["valid"]]
    counts = manifest["counts"]
    if (
        len(fit_train) != FIT_TRAIN_GROUPS
        or len(set(fit_train)) != FIT_TRAIN_GROUPS
        or len(fit_valid) != FIT_VALID_GROUPS
        or len(set(fit_valid)) != FIT_VALID_GROUPS
        or len(excluded) != EXCLUDED_EVALUATION_GROUPS
        or len(set(excluded)) != EXCLUDED_EVALUATION_GROUPS
        or set(fit_valid) & set(excluded)
        or int(counts["fit_train"]) != FIT_TRAIN_GROUPS
        or int(counts["fit_valid"]) != FIT_VALID_GROUPS
        or int(counts["evaluation_valid"]) != EXCLUDED_EVALUATION_GROUPS
        or FIT_TRAIN_GROUPS + FIT_VALID_GROUPS != FITTED_GROUPS
    ):
        raise ValueError("sealed fold0 membership is not 11940 fitted and 468 excluded groups")
    return {
        "excluded_evaluation_groups": len(excluded),
        "fit_train": len(fit_train),
        "fit_valid": len(fit_valid),
        "fitted_groups": len(fit_train) + len(fit_valid),
    }


def verify_staged_inputs(project: Path, registry: dict[str, object]) -> list[dict[str, object]]:
    """Hash staged inputs against the sealed registry and link them into the project.

    Args:
        project: Packet project root that receives input symlinks.
        registry: Parsed portable input registry.

    Returns:
        One size and SHA256 record per sealed input, in path order.

    Raises:
        ValueError: The supervisor manifest, size, or content hash disagrees.
    """
    if registry.get("version") != 1:
        raise ValueError("unsupported portable input registry version")
    sources = registry.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SEALED_INPUT_HASHES):
        raise ValueError("staged registry source set drifted from the sealed input identities")
    raw_items = json.loads(os.environ["WNN_INPUT_MANIFEST"])
    if not isinstance(raw_items, list):
        raise ValueError("input manifest must be a list")
    items = {str(item["path"]): item for item in raw_items}
    if set(items) != set(sources):
        raise ValueError("staged input set differs from the sealed portable registry")
    input_dir = Path(os.environ["WNN_INPUT_DIR"])
    observed: list[dict[str, object]] = []
    for path in sorted(sources):
        entry = sources[path]
        if not isinstance(entry, dict):
            raise ValueError(f"portable registry entry is invalid: {path}")
        expected_hash = str(entry.get("sha256") or "")
        identity = entry.get("original_identity")
        if not isinstance(identity, dict) or expected_hash != SEALED_INPUT_HASHES[path]:
            raise ValueError(f"portable registry lacks the sealed content hash: {path}")
        expected_size = int(identity["bytes"])
        if int(items[path]["size_bytes"]) != expected_size:
            raise ValueError(f"supervisor manifest size differs from sealed input: {path}")
        source = input_dir / path
        if not source.is_file() or source.stat().st_size != expected_size:
            raise ValueError(f"missing or truncated staged input: {path}")
        actual_hash = digest(source)
        if actual_hash != expected_hash:
            raise ValueError(f"staged input hash differs from sealed registry: {path}")
        destination = project / path
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"refusing to replace an existing project path: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
        observed.append({"path": path, "sha256": actual_hash, "size_bytes": expected_size})
    return observed


def load_bound_recipe(path: Path) -> dict[str, object]:
    """Parse a recipe only after its raw bytes match the pinned V5 SHA256.

    Args:
        path: Recipe file that will be executed. Its bytes are the authority.

    Returns:
        The parsed recipe object from those exact bytes.

    Raises:
        FileNotFoundError: The recipe file is absent. There is no substitute file.
        ValueError: The bytes differ from the reviewed recipe. They are not parsed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"sealed recipe is missing: {path}")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_RECIPE_SHA256:
        raise ValueError(
            "recipe bytes differ from the reviewed V5 recipe; refusing to parse or execute"
        )
    recipe = json.loads(raw)
    if not isinstance(recipe, dict):
        raise ValueError("sealed recipe must be a JSON object")
    return recipe


def validate_recipe(recipe: dict[str, object], runtime: dict[str, object]) -> None:
    """Fail closed unless the recipe is the two scratch full-feature V5 TCNs."""
    if recipe.get("recipe_id") != "E529-H217-TCN-FULLFEATURE-CAPACITY-V5-AMP4096":
        raise ValueError("recipe id is not the V5 full-feature capacity ladder")
    if recipe.get("feature_view") != FEATURE_VIEW or recipe.get("membership") != MEMBERSHIP:
        raise ValueError("recipe feature view or fold membership drifted")
    historical = recipe.get("historical_raw_only_reference")
    if not isinstance(historical, dict) or historical.get("paired_factor") is not False:
        raise ValueError("historical raw-only TCN must stay outside the paired factor")
    if (
        historical.get("quality_graded") is not False
        or historical.get("admits_v5_fit") is not False
    ):
        raise ValueError("historical raw-only timing cannot admit or grade a V5 fit")
    authorization = recipe.get("authorization")
    if not isinstance(authorization, dict) or any(authorization.values()):
        raise ValueError("packet authorization must remain false pending root review")
    pending = recipe.get("pending_root_review")
    if not isinstance(pending, dict) or pending.get("status") != "pending_root_review":
        raise ValueError("recipe is missing the pending root review block")
    if any(
        pending.get(field) is not False
        for field in (
            "fit_authorized",
            "provider_create_authorized",
            "push_authorized",
            "grade_authorized",
            "profile_authorized",
        )
    ):
        raise ValueError(
            "pending root review still authorizes create, push, profile, fit, or grade"
        )
    profiles = {str(row["arm"]): row for row in recipe["profile_sequence"]}
    fits = {str(row["arm"]): row for row in recipe["fit_branches"]["batch8"]}
    if (
        tuple(str(row["arm"]) for row in recipe["profile_sequence"]) != ARMS
        or tuple(str(row["arm"]) for row in recipe["fit_branches"]["batch8"]) != ARMS
        or set(profiles) != set(ARMS)
        or set(fits) != set(ARMS)
    ):
        raise ValueError("recipe must contain exactly tcn_full128 and tcn_full256, in that order")
    shared = {
        "--amp-initial-scale": str(int(AMP_INITIAL_SCALE)),
        "--batch": "8",
        "--checkpoint-epoch": "3",
        "--checkpoint-policy": "fixed",
        "--chunk": "2000",
        "--distill": "",
        "--distill-boot-h": "0",
        "--dropout": "0",
        "--epochs": "4",
        "--init": "",
        "--loss": "wmse",
        "--loss-norm": "chunk",
        "--lr": "0.001",
        "--optim": "adamw",
        "--precision": "fp16",
        "--propensity": "",
        "--prop-input": "",
        "--sched": "onecycle",
        "--seed": "7",
        "--train-seqs": "10607",
        "--train-source": "train+valid",
        "--valid-fold": "0",
        "--wd": "0.01",
    }
    for arm in ARMS:
        for phase, branch in (("profile", profiles[arm]), ("fit", fits[arm])):
            argv = [str(item) for item in branch["argv"]]
            if any(flag in argv for flag in FORBIDDEN_ARGV) or "gru" in argv:
                raise ValueError(f"{arm} {phase} retains a stale raw-only, GRU, or all-data flag")
            if any(_option(argv, key) != value for key, value in shared.items()):
                raise ValueError(f"{arm} {phase} argv differs from the V5 full-feature recipe")
            if _option(argv, "--arch") != "tcn" or "--no-eval-during-fit" not in argv:
                raise ValueError(f"{arm} {phase} must be a TCN fit with evaluation disabled")
            expected_name = PROFILE_RUN_NAMES[arm] if phase == "profile" else RUN_NAMES[arm]
            if _option(argv, "--name") != expected_name or branch.get("run_name") != expected_name:
                raise ValueError(f"{arm} {phase} run name is not the new V5 name")
            if "raw" in expected_name or "gru" in expected_name:
                raise ValueError(f"{arm} {phase} run name collides with a stale arm")
            dimensions = {"--hidden": ARM_HIDDEN[arm], "--layers": "12", "--tcn-mix": "1"}
            if any(_option(argv, key) != value for key, value in dimensions.items()):
                raise ValueError(f"{arm} {phase} architecture dimensions differ")
            if _option(argv, "--split-manifest") != MEMBERSHIP["split_manifest"]:
                raise ValueError(f"{arm} {phase} split manifest differs from sealed fold0")
            if phase == "profile" and (
                _option(argv, "--profile-warmup") != "3" or _option(argv, "--profile-steps") != "20"
            ):
                raise ValueError(f"{arm} profile must stay at warmup 3 and 20 timed steps")
            if phase == "fit" and ("--profile-warmup" in argv or "--profile-steps" in argv):
                raise ValueError(f"{arm} fit command must not be a profile")
        export = fits[arm].get("prediction_export", {})
        if (
            export.get("epoch") != 3
            or export.get("split") != "valid"
            or export.get("inference_precision") != "fp32"
            or export.get("output_name") != OUTPUT_NAMES[arm]
        ):
            raise ValueError(f"{arm} export must be a new valid-only fixed-epoch-3 FP32 name")
    if int(runtime["cohort_max_runtime_seconds"]) != 21600:
        raise ValueError("cohort ceiling differs from the reviewed 21600-second maximum")
    for field, expected in {
        "profile_timeout_seconds": PROFILE_SECONDS,
        "fit_timeout_seconds": FIT_SECONDS,
        "valid_export_timeout_seconds": VALID_EXPORT_SECONDS,
        "setup_timeout_seconds": 450,
        "terminal_packaging_reserve_seconds": 600,
    }.items():
        if int(runtime[field]) != expected:
            raise ValueError(f"runtime cap drifted: {field}")
    phase_sum = len(ARMS) * int(runtime["profile_timeout_seconds"]) + len(ARMS) * (
        int(runtime["fit_timeout_seconds"]) + int(runtime["valid_export_timeout_seconds"])
    )
    reserves = int(runtime["setup_timeout_seconds"]) + int(
        runtime["terminal_packaging_reserve_seconds"]
    )
    if phase_sum + reserves > int(runtime["cohort_max_runtime_seconds"]):
        raise ValueError("phase budgets plus setup/package reserves exceed the worker lifetime")
    if phase_sum != 19800 or reserves != 1050:
        raise ValueError("V5 cohort accounting drifted from 19800 phase seconds and 1050 reserves")


def validate_prediction(
    project: Path, export: dict[str, object], checkpoint: Path
) -> dict[str, object]:
    """Verify fixed-epoch FP32 validation output with a bounded memmap scan."""
    import numpy as np

    cache = project / "cache"
    name = str(export["output_name"])
    prediction = cache / f"pred_{name}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    if not prediction.is_file() or not sidecar.is_file():
        raise FileNotFoundError("fixed-epoch valid prediction or sidecar is missing")
    metadata = json.loads(sidecar.read_text())
    expected_checkpoint_hash = digest(checkpoint)
    if metadata.get("epoch") != 3 or metadata.get("inference_precision") != "fp32":
        raise ValueError("prediction metadata does not bind fixed epoch 3 FP32 inference")
    if metadata.get("checkpoint_sha256") != expected_checkpoint_hash:
        raise ValueError("prediction sidecar checkpoint hash differs from delivered checkpoint")
    expected_groups = np.load(cache / "valid_targets.npz")["row_group"]
    actual_groups = np.asarray(metadata.get("row_groups", []), dtype=np.int64)
    if not np.array_equal(actual_groups, expected_groups):
        raise ValueError("prediction row groups differ from the valid target membership")
    values = np.load(prediction, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(expected_groups), 20_000, 2)
    if values.shape != expected_shape or values.dtype != np.float32:
        raise ValueError(f"prediction shape/dtype mismatch: {values.shape} {values.dtype}")
    for start in range(0, len(values), 16):
        if not np.isfinite(values[start : start + 16]).all():
            raise ValueError("prediction contains a nonfinite value")
    prediction_hash = digest(prediction)
    if metadata.get("prediction_sha256") != prediction_hash:
        raise ValueError("prediction sidecar hash mismatch")
    return {
        "path": str(prediction.relative_to(project)),
        "sidecar": str(sidecar.relative_to(project)),
        "prediction_sha256": prediction_hash,
        "checkpoint_sha256": expected_checkpoint_hash,
        "row_group_count": len(expected_groups),
        "row_groups_sha256": hashlib.sha256(actual_groups.astype("<i8").tobytes()).hexdigest(),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": True,
        "inference_precision": "fp32",
        "epoch": 3,
    }


PhaseRunner = Callable[[list[str], str, int, Path, Path, float], dict[str, object]]
ProfileGate = Callable[[Path, dict[str, object]], dict[str, object]]


def execute_arms(
    recipe: dict[str, object],
    project: Path,
    output: Path,
    deadline: float,
    result: dict[str, object],
    phase_runner: PhaseRunner,
    gate_runner: ProfileGate,
) -> None:
    """Run both profiles, then each eligible fit/export without cross-arm aborts."""
    profiles = {str(row["arm"]): row for row in recipe["profile_sequence"]}
    fits = {str(row["arm"]): row for row in recipe["fit_branches"]["batch8"]}
    arms: dict[str, dict[str, object]] = {
        arm: {
            "profile_status": "pending",
            "fit_status": "blocked_profile",
            "export_status": "not_run",
        }
        for arm in ARMS
    }
    result["arms"] = arms

    for arm in ARMS:
        profile = profiles.get(arm)
        fit = fits.get(arm)
        if time.monotonic() >= deadline:
            arms[arm]["profile_status"] = "deferred_cohort_runtime"
            arms[arm]["fit_status"] = "deferred_profile_runtime"
            arms[arm]["error"] = "actual incoming runtime left no time for this arm's profile"
            write_json(output / "result.json", result)
            continue
        if profile is None or fit is None:
            arms[arm]["profile_status"] = "invalid_recipe"
            arms[arm]["error"] = "profile or fit branch is missing"
            write_json(output / "result.json", result)
            continue
        try:
            phase = phase_runner(
                [sys.executable, *profile["argv"]],
                f"profile_{arm}",
                PROFILE_SECONDS,
                project,
                output,
                deadline,
            )
            arms[arm]["profile_phase"] = phase
            if phase.get("exit_code") != 0 or phase.get("error"):
                phase_error = str(phase.get("error") or f"profile exit {phase.get('exit_code')}")
                if "cohort deadline" in phase_error.lower():
                    arms[arm]["profile_status"] = "deferred_cohort_runtime"
                    arms[arm]["fit_status"] = "deferred_profile_runtime"
                else:
                    arms[arm]["profile_status"] = "failed"
                arms[arm]["error"] = phase_error
            else:
                gate = gate_runner(output / f"profile_{arm}.log", fit)
                arms[arm]["profile_gate"] = gate
                arms[arm]["profile_status"] = "passed"
                arms[arm]["fit_status"] = "pending"
        except Exception as error:
            arms[arm]["profile_status"] = "failed"
            arms[arm]["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "result.json", result)

    eligible = [arm for arm in ARMS if arms[arm]["profile_status"] == "passed"]
    for index, arm in enumerate(eligible):
        fit = fits[arm]
        remaining = max(0.0, deadline - time.monotonic())
        current_profile = arms[arm].get("profile_gate", {})
        current_budget = projected_delivery_budget_seconds(current_profile)
        later_arms = eligible[index + 1 :]
        later_budget = sum(
            projected_delivery_budget_seconds(arms[next_arm].get("profile_gate", {}))
            for next_arm in later_arms
        )
        if remaining < current_budget:
            arms[arm]["fit_status"] = "deferred_cohort_runtime"
            arms[arm]["error"] = (
                "actual remaining runtime is below this arm's profile-projected fit "
                "plus valid-export budget; no fit was started"
            )
            arms[arm]["runtime_admission"] = {
                "remaining_seconds": round(remaining, 1),
                "required_projected_delivery_seconds": current_budget,
                "later_arm_projected_delivery_seconds": later_budget,
            }
            write_json(output / "result.json", result)
            continue
        if later_arms and remaining < current_budget + later_budget:
            arms[arm]["runtime_admission"] = {
                "remaining_seconds": round(remaining, 1),
                "required_projected_delivery_seconds": current_budget,
                "later_arm_ids_rechecked_after_this_arm": later_arms,
                "later_arm_projected_delivery_seconds": later_budget,
                "policy": "Admit this arm when its profile-projected fit and bounded valid export fit; recheck each later arm against the actual remaining deadline instead of reserving every later arm's worst case.",
            }
        try:
            fit_phase = phase_runner(
                [sys.executable, *fit["argv"]],
                f"fit_{arm}",
                FIT_SECONDS,
                project,
                output,
                deadline,
            )
            arms[arm]["fit_phase"] = fit_phase
            if fit_phase.get("exit_code") != 0 or fit_phase.get("error"):
                arms[arm]["fit_status"] = "failed"
                arms[arm]["error"] = (
                    fit_phase.get("error") or f"fit exit {fit_phase.get('exit_code')}"
                )
                write_json(output / "result.json", result)
                continue
            checkpoint = project / str(fit["output_path"]) / "model_ep3.pt"
            if not checkpoint.is_file():
                arms[arm]["fit_status"] = "failed_missing_checkpoint"
                arms[arm]["error"] = (
                    "fit returned success without the declared fixed epoch-3 checkpoint"
                )
                write_json(output / "result.json", result)
                continue
            arms[arm]["checkpoint_sha256"] = digest(checkpoint)
            arms[arm]["fit_status"] = "passed"
            write_json(output / "result.json", result)
            export = fit["prediction_export"]
            export_argv = [
                sys.executable,
                "scripts/predict_run.py",
                str(fit["run_name"]),
                "valid",
                "--epoch",
                "3",
                "--output-name",
                str(export["output_name"]),
                "--inference-precision",
                "fp32",
            ]
            export_phase = phase_runner(
                export_argv,
                f"export_{arm}_valid",
                VALID_EXPORT_SECONDS,
                project,
                output,
                deadline,
            )
            arms[arm]["export_phase"] = export_phase
            if export_phase.get("exit_code") != 0 or export_phase.get("error"):
                arms[arm]["export_status"] = "failed"
                arms[arm]["error"] = (
                    export_phase.get("error") or f"export exit {export_phase.get('exit_code')}"
                )
                write_json(output / "result.json", result)
                continue
            arms[arm]["prediction"] = validate_prediction(project, export, checkpoint)
            arms[arm]["export_status"] = "passed"
            arms[arm]["delivery_status"] = "delivered_ungraded"
        except Exception as error:
            arms[arm]["error"] = f"{type(error).__name__}: {error}"
            if arms[arm]["fit_status"] == "pending":
                arms[arm]["fit_status"] = "failed"
            elif arms[arm]["fit_status"] == "passed":
                arms[arm]["export_status"] = "failed"
        write_json(output / "result.json", result)


def main() -> int:
    """Run the V5 cohort after the exact recipe-byte guard and retain evidence."""
    packet = Path(__file__).resolve().parent
    project = packet / "project"
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    worker_started_at = time.monotonic()
    deadline: float | None = None
    result: dict[str, object] = {
        "status": "starting",
        "attempt_id": os.environ.get("WNN_ATTEMPT_ID"),
        "packet_sha256": os.environ.get("WNN_PACKET_SHA256"),
        "scientific_score": None,
        "grade_performed": False,
        "fit_authorized_by_packet": False,
        "provider_create_authorized_by_packet": False,
        "historical_raw_only_tcn_is_paired_factor": False,
    }
    try:
        recipe = load_bound_recipe(packet / "recipe.json")
        result["recipe_sha256"] = EXPECTED_RECIPE_SHA256
        result["exact_recipe_guard"] = "passed"
        runtime = json.loads((packet / "runtime_resources.json").read_text())
        for environment, field in {
            "WNN_THREADS": "cpu_threads",
            "WNN_LOADER_WORKERS": "loader_workers",
            "WNN_WORKER_RAM_BYTES": "worker_memory_max_bytes",
            "WNN_WORKER_VRAM_BYTES": "observed_total_vram_max_bytes",
            "WNN_GPU_MEMORY_BYTES": "cuda_allocator_max_bytes",
        }.items():
            if int(os.environ[environment]) != runtime[field]:
                raise ValueError(f"runtime resource mismatch: {environment}")
        cohort_ceiling = int(runtime["cohort_max_runtime_seconds"])
        incoming_runtime = validate_incoming_runtime(
            os.environ["WNN_MAX_RUNTIME_SECONDS"], cohort_ceiling
        )
        setup_reserve = int(runtime["setup_timeout_seconds"])
        packaging_reserve = int(runtime["terminal_packaging_reserve_seconds"])
        deadline = cohort_phase_deadline(
            worker_started_at, incoming_runtime, setup_reserve, packaging_reserve
        )
        operational_runtime = incoming_runtime - setup_reserve - packaging_reserve
        result["runtime_budget"] = {
            "incoming_runtime_seconds": incoming_runtime,
            "reviewed_cohort_ceiling_seconds": cohort_ceiling,
            "setup_reserve_seconds": setup_reserve,
            "terminal_packaging_reserve_seconds": packaging_reserve,
            "phase_deadline_offset_seconds": operational_runtime,
            "deadline_basis": "worker start plus actual incoming runtime minus declared setup and terminal packaging reserves",
        }
        write_json(output / "runtime_resources.json", runtime)
        validate_recipe(recipe, runtime)
        result["recipe_id"] = recipe["recipe_id"]
        result["feature_view"] = recipe["feature_view"]
        result["source_provenance"] = verify_source_provenance(project, recipe)
        result["split_membership"] = verify_split_membership(project, recipe)
        if operational_runtime <= 0:
            result["status"] = "deferred_insufficient_runtime"
            result["deferred_reason"] = (
                "incoming runtime is positive but does not exceed the declared "
                "setup and terminal packaging reserves; no profile or fit was started"
            )
            result["arms"] = {
                arm: {
                    "profile_status": "deferred_cohort_runtime",
                    "fit_status": "deferred_cohort_runtime",
                    "export_status": "not_run",
                    "error": result["deferred_reason"],
                }
                for arm in ARMS
            }
            result["worker_finished_at_unix_seconds"] = time.time()
            return 0
        os.environ["PYTHONPATH"] = str(project / "src")
        os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(
            project / "research/campaign_2026-10-04/portable_inputs.json"
        )
        registry = json.loads(
            (project / "research/campaign_2026-10-04/portable_inputs.json").read_text()
        )
        inputs = verify_staged_inputs(project, registry)
        import torch

        if torch.__version__ != "2.14.0+cu130" or not torch.cuda.is_available():
            raise RuntimeError(f"unexpected Torch/device environment: {torch.__version__}")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("the packet declares exactly one GPU")
        if time.monotonic() >= deadline:
            raise TimeoutError("cohort deadline expired during input preflight")
        result["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "input_manifest": inputs,
        }
        write_json(output / "environment.json", result["environment"])
        result["runtime_resources"] = runtime
        result["status"] = "running_profiles"
        write_json(output / "result.json", result)
        if deadline is None:
            raise RuntimeError("cohort deadline was not established")
        execute_arms(recipe, project, output, deadline, result, run_phase, profile_gate)
        arms = result["arms"]
        delivered = [
            arm for arm in arms.values() if arm.get("delivery_status") == "delivered_ungraded"
        ]
        if len(delivered) == len(ARMS):
            result["status"] = "delivered_ungraded"
        elif delivered:
            result["status"] = "partial_delivery_ungraded"
        elif any(
            str(arm.get("fit_status", "")).startswith("deferred")
            or str(arm.get("profile_status", "")).startswith("deferred")
            for arm in arms.values()
        ):
            result["status"] = "deferred_insufficient_runtime"
        else:
            result["status"] = "failed"
        result["worker_finished_at_unix_seconds"] = time.time()
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(output / "result.json", result)
        artifact_path = output / "artifacts.tar"
        with tarfile.open(artifact_path, "w") as archive:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name not in {
                    "artifacts.tar",
                    "artifact_manifest.json",
                    "worker.log",
                }:
                    archive.add(path, arcname=f"diagnostics/{path.name}", recursive=False)
            for directory in (project / "artifacts/runs", project / "cache"):
                if directory.exists():
                    for path in sorted(directory.rglob("*")):
                        if (
                            path.is_file()
                            and not path.is_symlink()
                            and (directory.name == "runs" or path.name.startswith("pred_"))
                        ):
                            archive.add(
                                path, arcname=str(path.relative_to(project)), recursive=False
                            )
    return (
        0
        if result["status"]
        in {
            "delivered_ungraded",
            "partial_delivery_ungraded",
            "deferred_insufficient_runtime",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
