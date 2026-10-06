from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NIGHT = Path(__file__).resolve().parents[1]
CADENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import continuation_recipes as historical  # noqa: E402

BLOCKS = ((3, 0), (4, 0), (3, 1))
ARM_ROWS = {
    "update1000": (1000, 1000),
    "pooled2000": (2000, 1000),
    "connected2000": (2000, 2000),
}
COMPARISONS = [
    ["pooled2000", "update1000"],
    ["connected2000", "pooled2000"],
    ["connected2000", "update1000"],
]


def write_json(path: Path, value: object) -> None:
    """Write one deterministic, human-readable draft artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def set_flag(argv: list[str], flag: str, value: str) -> None:
    """Replace one existing option value, failing if the base recipe drifted."""
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one value for {flag}")
    argv[positions[0] + 1] = value


def verify_supported_semantics(root: Path = ROOT) -> None:
    """Fail closed if trainer flags or row-clock/pooling implementation changed."""
    trainer = (root / "scripts/train_rnn.py").read_text()
    controls = (root / "src/wnn/training_controls.py").read_text()
    dynamics = (root / "src/wnn/training_dynamics.py").read_text()
    parser_flags = (
        '"--chunk"',
        '"--update-rows"',
        '"--detach-rows"',
        '"--ema-reference-rows"',
        '"--profile-steps"',
        '"--profile-warmup"',
    )
    trainer_contract = (
        "controlled_loss = bool(args.update_rows",
        "update_rows = args.update_rows or args.chunk",
        "ema_update_decay = ema_decay_per_update(args.ema_decay, update_rows, args.ema_reference_rows)",
        "SEQUENCE_LENGTH % update_rows",
        "detach_rows=detach_rows",
        "add_dynamics_arguments(ap)",
    )
    control_contract = (
        "main_mass = main_weights.sum().clamp_min(1e-8)",
        'main_divisor = reference_mass if denominator == "fixed" else main_mass',
        "for start in range(rows.start, rows.stop, micro_rows)",
        "cut = (",
        "yield partial, state, ema_state",
        "return decay if reference_rows == 0 else decay ** (update_rows / reference_rows)",
    )
    missing = [token for token in parser_flags + trainer_contract if token not in trainer]
    missing.extend(token for token in control_contract if token not in controls)
    if '"--batch-order"' not in dynamics:
        missing.append('"--batch-order" in src/wnn/training_dynamics.py')
    if missing:
        raise RuntimeError("unsupported cadence trainer semantics; missing: " + ", ".join(missing))


def cadence_args(run: str, seed: int, fold: int, arm: str) -> list[str]:
    """Return the existing strong recipe plus one preregistered cadence arm."""
    if arm not in ARM_ROWS:
        raise ValueError(f"unknown cadence arm: {arm}")
    update_rows, detach_rows = ARM_ROWS[arm]
    argv = historical.training_args(run, fold, seed)
    set_flag(argv, "--chunk", "1000")
    argv += [
        "--update-rows",
        str(update_rows),
        "--detach-rows",
        str(detach_rows),
        "--ema-reference-rows",
        "1000",
        "--batch-order",
        "seeded",
    ]
    return argv


def spec(name: str, argv: list[str], inputs: list[str], outputs: list[str]) -> dict:
    """Build an unsealed Kaggle draft using the existing resource contract."""
    design_inputs = [
        str((NIGHT / path).relative_to(ROOT))
        for path in ("CONTRACT.md", "supervision/NEXT_BLOCKS.md", "cadence/DESIGN.md")
    ] + ["research/campaign_2026-10-04/E430_optimizer_cadence/CRITIQUE_AND_PREREG.md"]
    return {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": argv,
        "code": sorted(
            set(historical.source_paths(ROOT) + [str(Path(__file__).relative_to(ROOT))])
        ),
        "inputs": sorted(set(inputs + design_inputs)),
        "artifacts": outputs,
        "limits": {
            "cpus": 2,
            "ram_bytes": 18_000_000_000,
            "timeout_seconds": 14_400,
            "gpu_memory_bytes": 4_200_000_000,
            "gpu_stop_bytes": 4_800_000_000,
        },
        "environment": {
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "3",
            "WNN_PORTABLE_DATA_MANIFEST": historical.CAMPAIGN + "/portable_inputs.json",
        },
    }


def run_outputs(run: str, epoch: int) -> list[str]:
    """List the fixed checkpoint and provenance outputs required by downstream jobs."""
    return [
        f"artifacts/runs/{run}/{filename}"
        for filename in (
            f"model_ep{epoch}.pt",
            "provenance.json",
            "history.json",
            "training_state.pt",
        )
    ]


def prediction_spec(run: str, epoch: int) -> dict:
    """Build a common-FP32 validation export bound to its fixed checkpoint."""
    return spec(
        "predict_" + run,
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
            historical.CAMPAIGN + "/portable_inputs.json",
            f"artifacts/runs/{run}/model_ep{epoch}.pt",
            f"artifacts/runs/{run}/provenance.json",
        ],
        [f"cache/pred_{run}_valid.{extension}" for extension in ("npy", "json")],
    )


def probe_spec(parent: dict, directory: str, run: str) -> tuple[dict, dict]:
    """Wrap a no-checkpoint production-shape profile in the supervised worker."""
    profile = copy.deepcopy(parent)
    probe_run = run + "_resource_probe"
    set_flag(profile["argv"], "--name", probe_run)
    profile["argv"].remove("--save-training-state")
    profile["argv"] += ["--profile-steps", "2", "--profile-warmup", "1"]
    report = f"{directory}/probes/{probe_run}_steps.json"
    plan_path = f"{directory}/specs/{probe_run}.plan.json"
    plan = {"version": 1, "steps": [{"name": probe_run, "argv": profile["argv"]}], "report": report}
    profile["argv"] = ["{python}", "scripts/kaggle_sequence.py", "--plan", plan_path]
    profile["code"] = sorted(
        set(profile["code"] + ["scripts/kaggle_sequence.py", "scripts/kaggle_common.py"])
    )
    profile["inputs"] = sorted(set(profile["inputs"] + [plan_path]))
    profile["artifacts"] = [report]
    profile["limits"]["timeout_seconds"] = 900
    profile["name"] = probe_run
    return profile, plan


def materialize() -> None:
    """Write unsealed specs and a zero-fit packet; never submit or launch a process."""
    verify_supported_semantics()
    deps: list[dict] = []
    for seed, fold in BLOCKS:
        parent_mixtures: dict[str, dict[str, float]] = {}
        complete_mixtures: dict[str, dict[str, float]] = {}
        for arm in ARM_ROWS:
            run = f"c20261005_cn_cadence_{arm}_s{seed}_f{fold}"
            head = run + "_head"
            parent = spec(
                run,
                cadence_args(run, seed, fold, arm),
                historical.input_paths(fold),
                run_outputs(run, 3),
            )
            head_job = spec(
                head,
                historical.head_args(head, run + "@ep3", fold) + ["--batch-order", "seeded"],
                historical.input_paths(fold, parent=run + "@ep3"),
                run_outputs(head, 0),
            )
            for phase, draft in (
                ("parent", parent),
                ("head", head_job),
                ("parent_prediction", prediction_spec(run, 3)),
                ("head_prediction", prediction_spec(head, 0)),
            ):
                path = CADENCE / "specs" / f"{run}_{phase}.draft.json"
                write_json(path, draft)
                deps.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "fold": fold,
                        "phase": phase,
                        "spec": str(path.relative_to(ROOT)),
                        "missing_inputs": [
                            path for path in draft["inputs"] if not (ROOT / path).exists()
                        ],
                    }
                )
            if (seed, fold) == (3, 0):
                probe, plan = probe_spec(
                    parent, "research/campaign_2026-10-05/continuation_night/cadence", run
                )
                plan_path = CADENCE / "specs" / f"{probe['name']}.plan.json"
                write_json(plan_path, plan)
                write_json(CADENCE / "specs" / f"{probe['name']}.draft.json", probe)
                deps.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "fold": fold,
                        "phase": "resource_probe",
                        "spec": str(
                            (CADENCE / "specs" / f"{probe['name']}.draft.json").relative_to(ROOT)
                        ),
                        "missing_inputs": [
                            path for path in probe["inputs"] if not (ROOT / path).exists()
                        ],
                    }
                )
            parent_mixtures[arm] = {run: 1.0}
            complete_mixtures[arm] = {run: 0.5, head: 0.5}
        for endpoint, mixtures in (("parent", parent_mixtures), ("complete", complete_mixtures)):
            write_json(
                CADENCE / f"screen_s{seed}_f{fold}_{endpoint}.json",
                {
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
                },
            )
    write_json(CADENCE / "dependencies.json", deps)
    receipt_path = CADENCE / "cpu_tests.xml"
    suite = ET.parse(receipt_path).getroot().find("testsuite")
    if suite is None or any(int(suite.attrib[field]) for field in ("failures", "errors")):
        raise ValueError("a passing cadence CPU test receipt is required")
    bindings = [
        "scripts/train_rnn.py",
        "src/wnn/training_controls.py",
        "src/wnn/training_dynamics.py",
        "scripts/continuation_recipes.py",
        "scripts/kaggle_sequence.py",
        "scripts/kaggle_common.py",
        str(Path(__file__).relative_to(ROOT)),
        str((CADENCE / "test_cadence.py").relative_to(ROOT)),
        str((CADENCE / "DESIGN.md").relative_to(ROOT)),
        str((NIGHT / "CONTRACT.md").relative_to(ROOT)),
        str((NIGHT / "supervision/NEXT_BLOCKS.md").relative_to(ROOT)),
        "research/campaign_2026-10-04/E430_optimizer_cadence/CRITIQUE_AND_PREREG.md",
    ]
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    write_json(
        CADENCE / "ledger_packet.json",
        {
            "experiment": {
                "id": "E-495",
                "date": "2026-10-05",
                "hypotheses": ["H-155"],
                "kind": "fresh strong-recipe cadence and recurrent-credit comparison",
                "status": "implemented_cpu_validated_pending_independent_review_and_resource_admission",
                "design": str((CADENCE / "DESIGN.md").relative_to(ROOT)),
                "arms": {
                    arm: {"update_rows": values[0], "detach_rows": values[1]}
                    for arm, values in ARM_ROWS.items()
                },
                "screen": {
                    "seed": 3,
                    "fold": 0,
                    "parent_epoch": 3,
                    "head_epoch": 0,
                    "head_seed": 3,
                },
                "confirmations": [{"seed": 4, "fold": 0}, {"seed": 3, "fold": 1}],
                "endpoint": "fixed equal prediction mixture of parent ep3 and head ep0; parent scored separately",
                "metric": "global_wp_clip2_v1",
                "bootstrap_draws": 2000,
                "scientific_fits_launched": 0,
                "resource_probes_launched": 0,
                "public_forecast": None,
                "cpu_validation": {
                    "tests": int(suite.attrib["tests"]),
                    "seconds": float(suite.attrib["time"]),
                    "receipt": str(receipt_path.relative_to(ROOT)),
                    "receipt_sha256": receipt_sha,
                    "limits": {"cpus": 1, "ram_bytes": 2_147_483_648, "gpu": False},
                },
            },
            "source_sha256": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in bindings
            },
            "root_ledger_written_by_actor": False,
            "admission_status": "unsealed drafts only; no science or score claim",
        },
    )


def main() -> None:
    """Materialize the preregistered cadence drafts after CPU validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="verify existing trainer semantics"
    )
    args = parser.parse_args()
    verify_supported_semantics()
    if not args.check_only:
        materialize()


if __name__ == "__main__":
    main()
