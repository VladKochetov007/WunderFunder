from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAPSULE = Path(__file__).resolve().parent
NIGHT = ROOT / "research/campaign_2026-10-05/continuation_night"
CADENCE = NIGHT / "cadence"
sys.path.insert(0, str(ROOT / "scripts"))
import continuation_recipes as historical  # noqa: E402

_cadence_spec = importlib.util.spec_from_file_location(
    "registered_cadence_materializer", CADENCE / "materialize_specs.py"
)
if _cadence_spec is None or _cadence_spec.loader is None:
    raise RuntimeError("cannot load the registered cadence source gate")
cadence_source = importlib.util.module_from_spec(_cadence_spec)
_cadence_spec.loader.exec_module(cadence_source)

BLOCKS = ((3, 0), (4, 0), (3, 1))
ARMS = {
    "update1000": (1000, 1000),
    "pooled2000": (2000, 1000),
    "connected2000": (2000, 2000),
}
COMPARISONS = [
    ["pooled2000", "update1000"],
    ["connected2000", "pooled2000"],
    ["connected2000", "update1000"],
]
CPU_LIMIT = 2
RAM_BYTES = 12_000_000_000
GPU_BYTES = 2_500_000_000
GPU_STOP_BYTES = 2_800_000_000
PROFILE_TIMEOUT_SECONDS = 600
FIT_TIMEOUT_SECONDS = 14_400
PROTOCOL_PATHS = [
    "research/campaign_2026-10-05/continuation_night/CONTRACT.md",
    "research/campaign_2026-10-05/continuation_night/ACTIVE_LOOP.md",
    "research/campaign_2026-10-05/continuation_night/cadence/DESIGN.md",
    "research/campaign_2026-10-05/continuation_night/cadence/materialize_specs.py",
    "research/campaign_2026-10-05/continuation_night/supervision/NEXT_BLOCKS.md",
    "research/campaign_2026-10-04/E430_optimizer_cadence/CRITIQUE_AND_PREREG.md",
    str(Path(__file__).relative_to(ROOT)),
    str((CAPSULE / "DESIGN.md").relative_to(ROOT)),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _set_flag(argv: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one {flag} in registered recipe")
    argv[positions[0] + 1] = value


def _run_name(arm: str, seed: int, fold: int) -> str:
    return f"c20261006_cap3g_cadence_{arm}_b16_s{seed}_f{fold}"


def _outputs(run: str, epoch: int) -> list[str]:
    return [
        f"artifacts/runs/{run}/{filename}"
        for filename in (
            f"model_ep{epoch}.pt",
            "provenance.json",
            "history.json",
            "training_state.pt",
        )
    ]


def _code_paths(*, sequence_runner: bool = False) -> list[str]:
    paths = historical.source_paths(ROOT) + [
        "research/campaign_2026-10-05/continuation_night/cadence/materialize_specs.py",
        str(Path(__file__).relative_to(ROOT)),
    ]
    if sequence_runner:
        paths.extend(["scripts/kaggle_sequence.py", "scripts/kaggle_common.py"])
    return sorted(set(paths))


def _spec(
    name: str,
    argv: list[str],
    inputs: list[str],
    artifacts: list[str],
    *,
    sequence_runner: bool = False,
    timeout_seconds: int = FIT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    source_manifest = str((CAPSULE / "source_manifest.json").relative_to(ROOT))
    return {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": argv,
        "code": _code_paths(sequence_runner=sequence_runner),
        "inputs": sorted(set(inputs + PROTOCOL_PATHS + [source_manifest])),
        "artifacts": artifacts,
        "limits": {
            "cpus": CPU_LIMIT,
            "ram_bytes": RAM_BYTES,
            "timeout_seconds": timeout_seconds,
            "gpu_memory_bytes": GPU_BYTES,
            "gpu_stop_bytes": GPU_STOP_BYTES,
        },
        "environment": {
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "3",
            "WNN_PORTABLE_DATA_MANIFEST": "research/campaign_2026-10-04/portable_inputs.json",
        },
    }


def _parent_argv(arm: str, seed: int, fold: int, run: str) -> list[str]:
    update_rows, detach_rows = ARMS[arm]
    argv = cadence_source.cadence_args(run, seed, fold, arm)
    _set_flag(argv, "--batch", "16")
    if argv[argv.index("--update-rows") + 1] != str(update_rows):
        raise ValueError(f"{arm}: registered update width changed")
    if argv[argv.index("--detach-rows") + 1] != str(detach_rows):
        raise ValueError(f"{arm}: registered detach width changed")
    return argv


def _prediction(run: str, epoch: int) -> dict[str, object]:
    return _spec(
        f"predict_{run}_ep{epoch}",
        [
            "{python}",
            "scripts/predict_run.py",
            run,
            "valid",
            "--epoch",
            str(epoch),
            "--inference-precision",
            "fp32",
        ],
        [
            "datasets/valid.parquet",
            "cache/valid_targets.npz",
            "research/campaign_2026-10-04/portable_inputs.json",
            f"artifacts/runs/{run}/model_ep{epoch}.pt",
            f"artifacts/runs/{run}/provenance.json",
        ],
        [f"cache/pred_{run}_valid.{extension}" for extension in ("npy", "json")],
        timeout_seconds=3_600,
    )


def _profile(
    arm: str, seed: int, fold: int, parent_argv: list[str], inputs: list[str]
) -> tuple[dict[str, object], dict[str, object]]:
    run = _run_name(arm, seed, fold)
    probe_name = f"{run}_resource_probe"
    probe_argv = copy.deepcopy(parent_argv)
    _set_flag(probe_argv, "--name", probe_name)
    probe_argv.remove("--save-training-state")
    probe_argv += ["--profile-steps", "2", "--profile-warmup", "1"]
    report = str((CAPSULE / "probes" / f"{probe_name}_steps.json").relative_to(ROOT))
    plan_path = CAPSULE / "plans" / f"{probe_name}.plan.json"
    plan_relative = str(plan_path.relative_to(ROOT))
    plan = {
        "version": 1,
        "steps": [{"name": probe_name, "argv": probe_argv}],
        "report": report,
    }
    probe = _spec(
        probe_name,
        ["{python}", "scripts/kaggle_sequence.py", "--plan", plan_relative],
        inputs + [plan_relative],
        [report],
        sequence_runner=True,
        timeout_seconds=PROFILE_TIMEOUT_SECONDS,
    )
    return probe, plan


def _endpoint(seed: int, fold: int, endpoint: str) -> dict[str, object]:
    mixtures = {}
    for arm in ARMS:
        run = _run_name(arm, seed, fold)
        mixtures[arm] = {run: 0.5, f"{run}_head": 0.5} if endpoint == "complete" else {run: 1.0}
    return {
        "folds": [fold],
        "n_boot": 2000,
        "arms": mixtures,
        "comparisons": COMPARISONS,
        "report_requirements": {
            "metric": "global_wp_clip2_v1",
            "targets": ["t0", "t1"],
            "market_class": [0, 1, 2, 3],
            "event_age": "book touch-event age, bins [0,4), [4,16), [16,64), [64,inf)",
            "paired_component_bootstrap_seed": 271828,
        },
        "registered_seed": seed,
        "endpoint": endpoint,
    }


def _content_manifest(paths: list[str]) -> list[dict[str, object]]:
    entries = []
    for name in sorted(set(paths)):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"source or protocol file is missing: {name}")
        entries.append({"path": name, "size": path.stat().st_size, "sha256": _sha256(path)})
    return entries


def _inventory(
    specification: dict[str, object], producers: dict[str, str]
) -> list[dict[str, object]]:
    entries = []
    for name in specification["inputs"]:
        path = ROOT / name
        present = path.is_file()
        entries.append(
            {
                "path": name,
                "present_in_checkout": present,
                "size_bytes": path.stat().st_size if present else None,
                "producer_job": producers.get(name),
            }
        )
    return entries


def materialize() -> None:
    """Emit hash-bound cadence specs only after the registered source gate passes."""
    generated = (
        "source_manifest.json",
        "draft_bindings.json",
        "dependencies.json",
        "RECEIPT.json",
        "specs",
        "plans",
        "endpoints",
        "probes",
    )
    collisions = [name for name in generated if (CAPSULE / name).exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite cap-3G cadence content: {collisions}")
    cadence_source.verify_supported_semantics(ROOT)
    CAPSULE.mkdir(parents=True, exist_ok=True)

    source_manifest = {
        "version": 1,
        "status": "source_closed_at_materialization_pending_critic_and_root_recheck",
        "code": _content_manifest(_code_paths(sequence_runner=True)),
        "protocols": _content_manifest(PROTOCOL_PATHS),
        "not_launched": True,
    }
    source_manifest["code_sha256"] = _canonical_sha256(source_manifest["code"])
    _write_new(CAPSULE / "source_manifest.json", source_manifest)

    jobs: dict[str, dict[str, object]] = {}
    plans: dict[str, dict[str, object]] = {}
    endpoint_paths = []
    for seed, fold in BLOCKS:
        for arm in ARMS:
            run = _run_name(arm, seed, fold)
            parent_argv = _parent_argv(arm, seed, fold, run)
            parent_inputs = historical.input_paths(fold)
            jobs[f"{arm}_s{seed}_f{fold}_parent"] = _spec(
                run, parent_argv, parent_inputs, _outputs(run, 3)
            )

            head = run + "_head"
            head_argv = historical.head_args(head, run + "@ep3", fold) + [
                "--batch-order",
                "seeded",
            ]
            jobs[f"{arm}_s{seed}_f{fold}_head"] = _spec(
                head,
                head_argv,
                historical.input_paths(fold, parent=run + "@ep3"),
                _outputs(head, 0),
            )
            jobs[f"{arm}_s{seed}_f{fold}_parent_prediction"] = _prediction(run, 3)
            jobs[f"{arm}_s{seed}_f{fold}_head_prediction"] = _prediction(head, 0)

            if (seed, fold) == (3, 0):
                profile, plan = _profile(arm, seed, fold, parent_argv, parent_inputs)
                jobs[f"{arm}_s{seed}_f{fold}_resource_probe"] = profile
                plans[str(profile["name"])] = plan

        for endpoint in ("parent", "complete"):
            path = CAPSULE / "endpoints" / f"e495_s{seed}_f{fold}_{endpoint}.json"
            endpoint_paths.append(path)
            _write_new(path, _endpoint(seed, fold, endpoint))

    draft_paths = {}
    for key, specification in jobs.items():
        path = CAPSULE / "specs" / f"{key}.draft.json"
        _write_new(path, specification)
        draft_paths[key] = str(path.relative_to(ROOT))
    for probe_name, plan in plans.items():
        _write_new(CAPSULE / "plans" / f"{probe_name}.plan.json", plan)

    job_bindings = {}
    for key, specification in jobs.items():
        entries = _content_manifest(specification["code"])
        job_bindings[key] = {
            "draft": draft_paths[key],
            "draft_sha256": _sha256(ROOT / draft_paths[key]),
            "code_sha256": _canonical_sha256(entries),
            "code": entries,
        }
        plan_inputs = [name for name in specification["inputs"] if name.endswith(".plan.json")]
        if plan_inputs:
            if len(plan_inputs) != 1 or not (ROOT / plan_inputs[0]).is_file():
                raise ValueError(f"{key}: expected one materialized profile plan")
            job_bindings[key]["plan"] = {
                "path": plan_inputs[0],
                "sha256": _sha256(ROOT / plan_inputs[0]),
            }
    _write_new(
        CAPSULE / "draft_bindings.json",
        {
            "version": 1,
            "source_manifest_sha256": _sha256(CAPSULE / "source_manifest.json"),
            "drafts": job_bindings,
            "not_launched": True,
        },
    )

    producers = {
        artifact: key
        for key, specification in jobs.items()
        if not key.endswith("_resource_probe")
        for artifact in specification["artifacts"]
    }
    dependencies = []
    for key, specification in jobs.items():
        inputs = _inventory(specification, producers)
        dependencies.append(
            {
                "job": key,
                "draft": draft_paths[key],
                "inputs": inputs,
                "missing_inputs": [item for item in inputs if not item["present_in_checkout"]],
                "artifacts": specification["artifacts"],
            }
        )
    _write_new(
        CAPSULE / "dependencies.json",
        {"version": 1, "jobs": dependencies, "not_launched": True},
    )
    _write_new(
        CAPSULE / "RECEIPT.json",
        {
            "status": "prepared_unsealed_no_profiles_or_fits_launched",
            "parent_batch": 16,
            "parent_hidden": 256,
            "parent_layers": 2,
            "microchunk_rows": 1000,
            "registered_update_detach_rows": ARMS,
            "resource_profile_count": 3,
            "parent_fit_count": sum(key.endswith("_parent") for key in jobs),
            "head_fit_count": sum(key.endswith("_head") for key in jobs),
            "prediction_job_count": sum("_prediction" in key for key in jobs),
            "parent_complete_endpoint_count": len(endpoint_paths),
            "profile_limits": {
                "cpus": CPU_LIMIT,
                "host_ram_bytes": RAM_BYTES,
                "allocator_bytes": GPU_BYTES,
                "observed_device_stop_bytes": GPU_STOP_BYTES,
                "timeout_seconds": PROFILE_TIMEOUT_SECONDS,
            },
            "source_manifest_sha256": _sha256(CAPSULE / "source_manifest.json"),
            "root_sealing_and_admission": "pending",
        },
    )


if __name__ == "__main__":
    materialize()
