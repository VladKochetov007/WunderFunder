from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAPSULE = Path(__file__).resolve().parent
NIGHT = ROOT / "research/campaign_2026-10-05/continuation_night"
sys.path.insert(0, str(ROOT / "scripts"))
import continuation_recipes as historical  # noqa: E402

BLOCKS = ((3, 0), (4, 0), (3, 1))
GPU_BYTES = 2_500_000_000
GPU_STOP_BYTES = 2_800_000_000
RAM_BYTES = 12_000_000_000
PROBE_TIMEOUT_SECONDS = 600
FIT_TIMEOUT_SECONDS = 14_400
CLASS_MIX = {"pub143_scenario": [0.183, 0.160, 0.198, 0.459]}
ARM_OPTIONS = {
    "control": ["--readout-activation", "gelu", "--distill-reanchor", "0"],
    "fixed_mix": ["--readout-activation", "fixed_mix", "--distill-reanchor", "0"],
    "reanchor010": ["--readout-activation", "gelu", "--distill-reanchor", "0.1"],
    "bootstrap_off": ["--readout-activation", "gelu", "--distill-reanchor", "0"],
}
RUN_LABELS = {
    "control": "common_b16",
    "fixed_mix": "fixed_mix_b16",
    "reanchor010": "reanchor010_b16",
    "bootstrap_off": "bootstrap_off_b16",
}
PROTOCOL_PATHS = [
    "research/campaign_2026-10-05/continuation_night/CONTRACT.md",
    "research/campaign_2026-10-05/continuation_night/ACTIVE_LOOP.md",
    "research/campaign_2026-10-05/continuation_night/activation_transfer/DESIGN.md",
    "research/campaign_2026-10-05/continuation_night/supervision/DESIGN.md",
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
    positions = [index for index, entry in enumerate(argv) if entry == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one {flag} in recipe")
    argv[positions[0] + 1] = value


def _run_name(arm: str, seed: int, fold: int) -> str:
    return f"c20261006_cap3g_{RUN_LABELS[arm]}_s{seed}_f{fold}"


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
    paths = historical.source_paths(ROOT) + [str(Path(__file__).relative_to(ROOT))]
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
    return {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": argv,
        "code": _code_paths(sequence_runner=sequence_runner),
        "inputs": sorted(
            set(
                inputs
                + PROTOCOL_PATHS
                + [str((CAPSULE / "source_manifest.json").relative_to(ROOT))]
            )
        ),
        "artifacts": artifacts,
        "limits": {
            "cpus": 2,
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
    argv = historical.training_args(run, fold, seed)
    _set_flag(argv, "--batch", "16")
    argv += ["--batch-order", "seeded"] + ARM_OPTIONS[arm]
    if arm == "bootstrap_off":
        _set_flag(argv, "--distill-boot-h", "0")
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
        timeout_seconds=PROBE_TIMEOUT_SECONDS,
    )
    return probe, plan


def _endpoint(seed: int, fold: int, endpoint: str, arms: tuple[str, ...]) -> dict[str, object]:
    mixtures = {
        "control": {"control": 0.5},
        "fixed_mix": {"fixed_mix": 0.5},
        "reanchor010": {"reanchor010": 0.5},
        "bootstrap_off": {"bootstrap_off": 0.5},
    }
    for arm in arms:
        run = _run_name(arm, seed, fold)
        if endpoint == "complete":
            mixtures[arm] = {run: 0.5, f"{run}_head": 0.5}
        else:
            mixtures[arm] = {run: 1.0}
    if arms == ("control", "fixed_mix"):
        comparisons = [["fixed_mix", "control"]]
    else:
        comparisons = [
            ["reanchor010", "control"],
            ["bootstrap_off", "control"],
            ["reanchor010", "bootstrap_off"],
        ]
    return {
        "folds": [fold],
        "n_boot": 2000,
        "arms": {arm: mixtures[arm] for arm in arms},
        "comparisons": comparisons,
        "class_mix": CLASS_MIX,
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


def materialize() -> None:
    """Write one hash-bound, unlaunched cap-3G transfer capsule."""
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
        raise FileExistsError(f"refusing to overwrite capsule content: {collisions}")
    CAPSULE.mkdir(parents=True, exist_ok=True)

    source_manifest = {
        "version": 1,
        "status": "source_closed_at_materialization_pending_critic_and_root_recheck",
        "code": _content_manifest(_code_paths(sequence_runner=True)),
        "protocols": _content_manifest(PROTOCOL_PATHS),
        "not_launched": True,
    }
    code_manifest = source_manifest["code"]
    source_manifest["code_sha256"] = _canonical_sha256(code_manifest)
    _write_new(CAPSULE / "source_manifest.json", source_manifest)

    jobs: dict[str, dict[str, object]] = {}
    plans: dict[str, dict[str, object]] = {}
    endpoint_paths: list[Path] = []
    arm_names = tuple(ARM_OPTIONS)
    for seed, fold in BLOCKS:
        for arm in arm_names:
            run = _run_name(arm, seed, fold)
            parent_argv = _parent_argv(arm, seed, fold, run)
            parent_inputs = historical.input_paths(fold)
            parent = _spec(run, parent_argv, parent_inputs, _outputs(run, 3))
            jobs[f"{arm}_s{seed}_f{fold}_parent"] = parent

            head = run + "_head"
            head_argv = historical.head_args(head, run + "@ep3", fold) + [
                "--batch-order",
                "seeded",
            ]
            head = _spec(
                head,
                head_argv,
                historical.input_paths(fold, parent=run + "@ep3"),
                _outputs(head, 0),
            )
            jobs[f"{arm}_s{seed}_f{fold}_head"] = head
            jobs[f"{arm}_s{seed}_f{fold}_parent_prediction"] = _prediction(run, 3)
            jobs[f"{arm}_s{seed}_f{fold}_head_prediction"] = _prediction(run + "_head", 0)

            if (seed, fold) == (3, 0):
                profile, plan = _profile(arm, seed, fold, parent_argv, parent_inputs)
                probe_name = str(profile["name"])
                jobs[f"{arm}_s{seed}_f{fold}_resource_probe"] = profile
                plans[probe_name] = plan

        for endpoint in ("parent", "complete"):
            activation_path = CAPSULE / "endpoints" / f"e493_s{seed}_f{fold}_{endpoint}.json"
            supervision_path = CAPSULE / "endpoints" / f"e494_s{seed}_f{fold}_{endpoint}.json"
            endpoint_paths.extend([activation_path, supervision_path])
            _write_new(
                activation_path,
                _endpoint(seed, fold, endpoint, ("control", "fixed_mix")),
            )
            _write_new(
                supervision_path,
                _endpoint(
                    seed,
                    fold,
                    endpoint,
                    ("control", "reanchor010", "bootstrap_off"),
                ),
            )

    draft_paths: dict[str, str] = {}
    for key, specification in jobs.items():
        path = CAPSULE / "specs" / f"{key}.draft.json"
        _write_new(path, specification)
        draft_paths[key] = str(path.relative_to(ROOT))

    for probe_name, plan in plans.items():
        path = CAPSULE / "plans" / f"{probe_name}.plan.json"
        _write_new(path, plan)

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
                raise ValueError(f"{key}: expected one materialized resource plan")
            job_bindings[key]["plan"] = {
                "path": plan_inputs[0],
                "sha256": _sha256(ROOT / plan_inputs[0]),
            }
    bindings = {
        "version": 1,
        "source_manifest_sha256": _sha256(CAPSULE / "source_manifest.json"),
        "drafts": job_bindings,
        "not_launched": True,
    }
    _write_new(CAPSULE / "draft_bindings.json", bindings)

    producers = {
        artifact: key
        for key, specification in jobs.items()
        if not key.endswith("_resource_probe")
        for artifact in specification["artifacts"]
    }
    dependencies = []
    for key, specification in jobs.items():
        input_inventory = []
        for name in specification["inputs"]:
            path = ROOT / name
            present = path.is_file()
            entry = {
                "path": name,
                "present_in_checkout": present,
                "size_bytes": path.stat().st_size if present else None,
                "producer_job": producers.get(name),
            }
            input_inventory.append(entry)
        dependencies.append(
            {
                "job": key,
                "draft": draft_paths[key],
                "inputs": input_inventory,
                "missing_inputs": [
                    item for item in input_inventory if not item["present_in_checkout"]
                ],
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
            "shared_parent_batch": 16,
            "parent_hidden": 256,
            "parent_layers": 2,
            "parent_chunk_rows": 1000,
            "resource_profile_count": 4,
            "parent_fit_count": sum(key.endswith("_parent") for key in jobs),
            "head_fit_count": sum(key.endswith("_head") for key in jobs),
            "prediction_job_count": sum("_prediction" in key for key in jobs),
            "complete_and_parent_endpoint_count": len(endpoint_paths),
            "profile_limits": {
                "cpus": 2,
                "host_ram_bytes": RAM_BYTES,
                "allocator_bytes": GPU_BYTES,
                "observed_device_stop_bytes": GPU_STOP_BYTES,
                "timeout_seconds": PROBE_TIMEOUT_SECONDS,
            },
            "source_manifest_sha256": _sha256(CAPSULE / "source_manifest.json"),
            "critic_review": "pending",
            "root_sealing_and_admission": "pending",
        },
    )


if __name__ == "__main__":
    materialize()
