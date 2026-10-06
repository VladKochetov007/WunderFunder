from __future__ import annotations

import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NIGHT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import continuation_recipes as historical  # noqa: E402

ARMS = {
    "control": ["--readout-activation", "gelu", "--distill-reanchor", "0"],
    "fixed_mix": ["--readout-activation", "fixed_mix", "--distill-reanchor", "0"],
    "reanchor010": ["--readout-activation", "gelu", "--distill-reanchor", "0.1"],
    "bootstrap_off": ["--readout-activation", "gelu", "--distill-reanchor", "0"],
}
BLOCKS = [(3, 0), (4, 0), (3, 1)]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _replace(argv: list[str], flag: str, value: str) -> None:
    argv[argv.index(flag) + 1] = value


def _spec(name: str, argv: list[str], inputs: list[str], artifacts: list[str]) -> dict:
    return {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": argv,
        "code": historical.source_paths(ROOT) + [str(Path(__file__).relative_to(ROOT))],
        "inputs": sorted(
            set(
                inputs
                + [
                    str((NIGHT / relative).relative_to(ROOT))
                    for relative in (
                        "CONTRACT.md",
                        "activation_transfer/DESIGN.md",
                        "supervision/DESIGN.md",
                    )
                ]
            )
        ),
        "artifacts": artifacts,
        "limits": {
            "cpus": 2,
            "ram_bytes": 18_000_000_000,
            "timeout_seconds": 14400,
            "gpu_memory_bytes": 4_200_000_000,
            "gpu_stop_bytes": 4_800_000_000,
        },
        "environment": {
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "3",
            "WNN_PORTABLE_DATA_MANIFEST": historical.CAMPAIGN + "/portable_inputs.json",
        },
    }


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


def _prediction(run: str, epoch: int) -> dict:
    return _spec(
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


def materialize() -> None:
    """Write unsealed drafts only; this function never submits or starts a process."""
    dependencies = []
    for seed, fold in BLOCKS:
        complete, parents = {}, {}
        for arm, options in ARMS.items():
            directory = NIGHT / (
                "activation_transfer" if arm in ("control", "fixed_mix") else "supervision"
            )
            run = f"c20261005_cn_{arm}_s{seed}_f{fold}"
            head = run + "_head"
            argv = historical.training_args(run, fold, seed) + ["--batch-order", "seeded"] + options
            if arm == "bootstrap_off":
                _replace(argv, "--distill-boot-h", "0")
            parent = _spec(run, argv, historical.input_paths(fold), _outputs(run, 3))
            head_job = _spec(
                head,
                historical.head_args(head, run + "@ep3", fold) + ["--batch-order", "seeded"],
                historical.input_paths(fold, parent=run + "@ep3"),
                _outputs(head, 0),
            )
            for phase, specification in (
                ("parent", parent),
                ("head", head_job),
                ("parent_prediction", _prediction(run, 3)),
                ("head_prediction", _prediction(head, 0)),
            ):
                path = directory / "specs" / f"{run}_{phase}.draft.json"
                _write(path, specification)
                dependencies.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "fold": fold,
                        "phase": phase,
                        "spec": str(path.relative_to(ROOT)),
                        "missing_inputs": [
                            entry
                            for entry in specification["inputs"]
                            if not (ROOT / entry).exists()
                        ],
                    }
                )
            if (seed, fold) == (3, 0):
                probe = copy.deepcopy(parent)
                probe_run = run + "_resource_probe"
                probe["name"] = probe_run
                _replace(probe["argv"], "--name", probe_run)
                probe["argv"].remove("--save-training-state")
                probe["argv"] += ["--profile-steps", "2", "--profile-warmup", "1"]
                report = str((directory / "probes" / f"{probe_run}_steps.json").relative_to(ROOT))
                plan_path = directory / "specs" / f"{probe_run}.plan.json"
                _write(
                    plan_path,
                    {
                        "version": 1,
                        "steps": [{"name": probe_run, "argv": probe["argv"]}],
                        "report": report,
                    },
                )
                probe["argv"] = [
                    "{python}",
                    "scripts/kaggle_sequence.py",
                    "--plan",
                    str(plan_path.relative_to(ROOT)),
                ]
                probe["code"] += ["scripts/kaggle_sequence.py", "scripts/kaggle_common.py"]
                probe["inputs"] += [str(plan_path.relative_to(ROOT))]
                probe["artifacts"] = [report]
                probe["limits"]["timeout_seconds"] = 900
                _write(directory / "specs" / f"{probe_run}.draft.json", probe)
            complete[arm] = {run: 0.5, head: 0.5}
            parents[arm] = {run: 1.0}
        for endpoint, arms in (("complete", complete), ("parent", parents)):
            for family, names, comparisons in (
                ("activation_transfer", ("control", "fixed_mix"), [["fixed_mix", "control"]]),
                (
                    "supervision",
                    ("control", "reanchor010", "bootstrap_off"),
                    [
                        ["reanchor010", "control"],
                        ["bootstrap_off", "control"],
                        ["reanchor010", "bootstrap_off"],
                    ],
                ),
            ):
                _write(
                    NIGHT / family / f"screen_s{seed}_f{fold}_{endpoint}.json",
                    {
                        "folds": [fold],
                        "n_boot": 2000,
                        "arms": {name: arms[name] for name in names},
                        "comparisons": comparisons,
                        "class_mix": {"pub143_scenario": [0.183, 0.160, 0.198, 0.459]},
                    },
                )
    _write(NIGHT / "activation_transfer/dependencies.json", dependencies)
    sources = [
        "artifacts/runs/c20261004_combo_s4_final/provenance.json",
        "artifacts/runs/c20261004_combo_head_s4_final/provenance.json",
        "scripts/continuation_recipes.py",
    ]
    evidence = {
        source: {"sha256": hashlib.sha256((ROOT / source).read_bytes()).hexdigest()}
        for source in sources
    }
    for source in sources[:2]:
        evidence[source]["args"] = json.loads((ROOT / source).read_text())["args"]
    _write(
        NIGHT / "activation_transfer/recipe_provenance.json",
        {
            "sources": evidence,
            "shared_control": "One fresh control per seed/fold supplies both predeclared experiments; no selection from cached controls.",
            "intentional_common_changes": [
                "component-excluded fold lineages",
                "no evaluation during fitting",
                "seeded ordering and fixed normalization sample membership",
                "synchronous loader workers0",
            ],
            "endpoint": "Equal prediction mixture of fixed parent ep3 and fixed head ep0; no weight search.",
            "not_launched": True,
        },
    )
    binding_paths = [
        "src/wnn/nn.py",
        "src/wnn/training_controls.py",
        "scripts/train_rnn.py",
        "scripts/export_onnx.py",
        "scripts/package_submission.py",
        "tests/test_training_controls.py",
        "tests/test_continuation_treatments.py",
        str(Path(__file__).relative_to(ROOT)),
    ]
    bindings = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in binding_paths
    }
    receipt = NIGHT / "activation_transfer/cpu_tests_v2.xml"
    suite = ET.parse(receipt).getroot().find("testsuite")
    if suite is None or any(int(suite.attrib[field]) for field in ("failures", "errors")):
        raise ValueError("a passing CPU test receipt is required before emitting validated packets")
    for directory, identifier, hypothesis, treatment, controls in (
        ("activation_transfer", "E-493", "H-121", "fixed_mix", ["control"]),
        ("supervision", "E-494", "H-124", "reanchor010", ["control", "bootstrap_off"]),
    ):
        _write(
            NIGHT / directory / "ledger_packet.json",
            {
                "experiment": {
                    "id": identifier,
                    "date": "2026-10-05",
                    "hypotheses": [hypothesis],
                    "kind": "complete strong recipe " + treatment,
                    "status": "implemented_cpu_validated_pending_independent_review_and_resource_admission",
                    "design": str((NIGHT / directory / "DESIGN.md").relative_to(ROOT)),
                    "treatment": treatment,
                    "controls": controls,
                    "screen": {
                        "seed": 3,
                        "fold": 0,
                        "parent_epoch": 3,
                        "head_epoch": 0,
                        "head_seed": 3,
                    },
                    "confirmations": [{"seed": 4, "fold": 0}, {"seed": 3, "fold": 1}],
                    "shared_control": "predeclared one fresh control per seed/fold across E-493 and E-494",
                    "metric": "global_wp_clip2_v1",
                    "bootstrap_draws": 2000,
                    "scientific_fits_launched": 0,
                    "resource_probes_launched": 0,
                    "public_forecast": None,
                    "cpu_validation": {
                        "tests": int(suite.attrib["tests"]),
                        "seconds": float(suite.attrib["time"]),
                        "receipt": str(receipt.relative_to(ROOT)),
                        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        "limits": {"cpus": 1, "ram_bytes": 2147483648, "gpu": False},
                    },
                },
                "source_sha256": bindings,
                "root_ledger_written_by_actor": False,
                "admission_status": "unsealed drafts; no science, transfer or deployment claim",
            },
        )


if __name__ == "__main__":
    materialize()
