from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.continuation_recipes import input_paths  # noqa: E402
from scripts.kaggle_common import canonical_sha256, sha256_file  # noqa: E402


HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE / "specs"
ASSET_MANIFEST = ROOT / "artifacts/kaggle_assets/wnn-c20261004-fold1-assets/assets-manifest.json"
E413_MANIFEST = ROOT / "research/campaign_2026-10-04/kaggle/E413_volume_v1/bundle/bundle-manifest.json"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold1-assets"
RAW_DATASET = "vladkochetov07/wunder-funder"
SLUG = "wnn-w384-fold1-dualt4-20261004-v1"
CAMPAIGN_PATH = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v1"
PORTABLE_PATH = f"{CAMPAIGN_PATH}/portable_inputs_f1.json"
SCREEN_PATH = f"{CAMPAIGN_PATH}/paired_fold1_screen.json"
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold1.json"
WORKER_TIMEOUT = 15_600
ALLOCATOR_BYTES = 11 * 1024**3
STOP_BYTES = 13 * 1024**3
RAM_BYTES = 10 * 1024**3


def descriptor(path: str) -> dict:
    absolute = ROOT / path
    return {"path": path, "size": absolute.stat().st_size, "sha256": sha256_file(absolute)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_closure() -> list[str]:
    files = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "scripts").glob("*.py"))
    ]
    files.extend(
        [
            "research/campaign_2026-09-30/audit/campaign_score.py",
            f"{CAMPAIGN_PATH}/phase_runner.py",
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/test_trainer_resume_integration.py",
            "tests/test_training_state.py",
            "tests/test_width_runtime_probe.py",
            "tests/training_dynamics_helpers.py",
        ]
    )
    files.extend(path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "src/wnn").glob("*.py")))
    return sorted(set(files))


def all_inputs() -> dict[str, dict]:
    assets = json.loads(ASSET_MANIFEST.read_text())
    by_path = {item["path"]: item for item in assets["files"]}
    raw_manifest = json.loads(E413_MANIFEST.read_text())
    raw_paths = {"datasets/train.parquet", "datasets/valid.parquet", "datasets/valid_mask.parquet"}
    raw = {item["path"]: item for item in raw_manifest["files"] if item["path"] in raw_paths}
    paths = set(input_paths(1))
    paths.discard("research/campaign_2026-10-04/portable_inputs.json")
    paths.add(PORTABLE_PATH)
    paths.update((SCREEN_PATH,))
    result = {}
    for name in sorted(paths):
        if name in raw:
            result[name] = {key: raw[name][key] for key in ("path", "size", "sha256")}
        elif name in by_path:
            item = by_path[name]
            result[name] = {key: item[key] for key in ("path", "size", "sha256")}
        else:
            result[name] = descriptor(name)
    return result


def fit_args(name: str, hidden: int, *, probe: bool) -> list[str]:
    args = [
        "{python}",
        "scripts/train_rnn.py",
        "--name",
        name,
        "--train-source",
        "train+valid",
        "--full-valid",
        "--loss",
        "wmse",
        "--aux-struct",
        "0.3",
        "--aux-next",
        "0.3",
        "--propensity",
        "c20260929_prop_f1",
        "--prop-power",
        "2",
        "--prop-eps",
        "0.03",
        "--valid-true-mask",
        "0.03",
        "--hidden",
        str(hidden),
        "--layers",
        "2",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "1" if probe else "4",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--seed",
        "3",
        "--optim",
        "adamw",
        "--ema-teacher",
        "1",
        "--ema-decay",
        "0.999",
        "--eval-ema",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--distill",
        "c20260929_since_s3_f1,c20260929_since_s4_f1",
        "--distill-alpha",
        "0.5",
        "--distill-boot-h",
        "0",
        "--fast",
        "--precision",
        "fp16",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "0" if probe else "3",
        "--save-training-state",
        "--split-manifest",
        SPLIT_PATH,
        "--no-eval-during-fit",
    ]
    if probe:
        args += ["--max-batches", "1"]
    return args


def sequence_plan(hidden: int, *, probe: bool) -> tuple[str, dict]:
    suffix = "probe" if probe else "fit"
    run_name = f"c20261004_w{hidden}_adamw_s3_f1_t4v1_{suffix}"
    report = f"{CAMPAIGN_PATH}/{run_name}_steps.json"
    steps = [{
        "name": "one_batch_production_shape_forward_backward" if probe else "fixed_four_epoch_training",
        "argv": fit_args(run_name, hidden, probe=probe),
    }]
    if not probe:
        steps.append({
            "name": "fixed_epoch3_full_valid_fp32_prediction",
            "argv": ["{python}", "scripts/predict_run.py", run_name, "valid", "--epoch", "3"],
        })
    return run_name, {"version": 1, "report": report, "steps": steps}


def make_spec(name: str, plan_path: str, code: list[dict], inputs: list[dict], artifacts: list[str], timeout: int) -> dict:
    return {
        "version": 1,
        "name": name,
        "kind": "gpu",
        "argv": ["{python}", "scripts/kaggle_sequence.py", "--plan", plan_path],
        "code": code,
        "code_sha256": canonical_sha256(sorted(code, key=lambda item: item["path"])),
        "inputs": inputs,
        "artifacts": artifacts,
        "limits": {
            "cpus": 2,
            "ram_bytes": RAM_BYTES,
            "timeout_seconds": timeout,
            "gpu_memory_bytes": ALLOCATOR_BYTES,
            "gpu_stop_bytes": STOP_BYTES,
        },
        "environment": {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "WNN_LOADER_WORKERS": "1",
            "WNN_LOADER_RAM_GB": "3",
            "WNN_PORTABLE_DATA_MANIFEST": PORTABLE_PATH,
        },
    }


def main() -> None:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    run_names = {}
    plans = {}
    for hidden in (256, 384):
        for is_probe in (True, False):
            run_name, plan = sequence_plan(hidden, probe=is_probe)
            plans[run_name] = plan
            plan_path = f"{CAMPAIGN_PATH}/plans/{run_name}_plan.json"
            write_json(ROOT / plan_path, plan)
            run_names[(hidden, is_probe)] = run_name

    screen = {
        "folds": [1],
        "arms": {
            "W256": {"c20261004_w256_adamw_s3_f1_t4v1_fit": 1.0},
            "W384": {"c20261004_w384_adamw_s3_f1_t4v1_fit": 1.0},
        },
        "comparisons": [["W384", "W256"]],
        "n_boot": 2000,
    }
    write_json(ROOT / SCREEN_PATH, screen)

    base_code_paths = source_closure()
    input_map = all_inputs()
    for run_name, plan in plans.items():
        input_map[f"{CAMPAIGN_PATH}/plans/{run_name}_plan.json"] = descriptor(
            f"{CAMPAIGN_PATH}/plans/{run_name}_plan.json"
        )
    for hidden in (256, 384):
        for is_probe in (True, False):
            run_name = run_names[(hidden, is_probe)]
            plan_path = f"{CAMPAIGN_PATH}/plans/{run_name}_plan.json"
            code = [descriptor(path) for path in base_code_paths]
            inputs = [input_map[name] for name in sorted(input_map) if name != SCREEN_PATH or not is_probe]
            artifacts = [
                f"artifacts/runs/{run_name}",
                f"{CAMPAIGN_PATH}/plans/{run_name}_steps.json",
            ]
            if not is_probe:
                artifacts += [
                    f"cache/pred_{run_name}_valid.npy",
                    f"cache/pred_{run_name}_valid.json",
                ]
            spec = make_spec(run_name, plan_path, code, inputs, artifacts, 300 if is_probe else WORKER_TIMEOUT)
            write_json(SPEC_DIR / f"{run_name}.sealed.json", spec)

    print(json.dumps({
        "dataset_slug": SLUG,
        "asset_dataset": ASSET_DATASET,
        "raw_dataset": RAW_DATASET,
        "specs": [p.name for p in sorted(SPEC_DIR.glob("*.sealed.json"))],
        "plans": [p.name for p in sorted((ROOT / CAMPAIGN_PATH / "plans").glob("*.json"))],
        "screen": SCREEN_PATH,
        "worker_timeout_seconds": WORKER_TIMEOUT,
        "provider_timeout_seconds": 16_200,
        "maximum_dual_gpu_charge_hours": 9.0,
    }, indent=2))


if __name__ == "__main__":
    main()
