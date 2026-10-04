"""Run preregistered specialist and prefix-adapter arms one at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
DEADLINE = datetime.fromisoformat("2026-10-04T17:00:00+00:00").timestamp()
FIT_START_CUTOFF = datetime.fromisoformat("2026-10-04T16:10:00+00:00").timestamp()
ARMS = (
    "h093_learned",
    "h093_gru_budget",
    "h094_conditional",
    "h094_gru_budget",
    "h093_hash",
    "h093_universal",
    "h094_unconditional",
    "h094_diagonal",
)
CORE = ARMS[:4]
OUTPUT = CAMPAIGN / "checks/e424_sequence_v2.json"
BASELINE = ("c20261004_combo_s3_final_r1", "c20261004_combo_head_s3_final")


def stop_if_paused() -> None:
    """Stop at a guard pause; do not acknowledge or relaunch automatically."""
    if (RUNNER / "PAUSE.json").exists():
        raise RuntimeError("resource guard paused; manual review is required")


def sha256_file(path: Path) -> str:
    """Hash a completed checkpoint or prediction without loading it into memory."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def wait_finished(attempt: str) -> None:
    """Wait for one exact admitted attempt and require its verified success."""
    while time.time() < DEADLINE:
        stop_if_paused()
        record_path = RUNNER / f"attempts/{attempt}.json"
        if not record_path.exists():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text())
        if record["status"] == "finished":
            time.sleep(5)
            return
        if record["status"] not in ("reserved", "running"):
            raise RuntimeError(f"execution stopped: {attempt}: {record['status']}")
        time.sleep(5)
    unit = f"wnn-c04-{attempt}.service"
    subprocess.run(
        ["systemctl", "--user", "kill", "--signal=TERM", "--kill-who=all", unit],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["systemctl", "--user", "stop", unit], cwd=ROOT, check=False, capture_output=True, text=True
    )
    raise TimeoutError("20:00 local cutoff reached while waiting for an admitted arm")


def launch(
    sealed: Path, attempt: str, completed: list[str], *, minimum_seconds: int = 45 * 60
) -> None:
    """Launch one sealed job only if a conservative finish window remains."""
    stop_if_paused()
    spec = json.loads(sealed.read_text())
    existing = RUNNER / f"attempts/{attempt}.json"
    if existing.exists():
        record = json.loads(existing.read_text())
        if record["spec"] != spec:
            raise ValueError(f"Existing attempt has a different sealed identity: {attempt}")
        if record["status"] in ("reserved", "running"):
            wait_finished(attempt)
            record = json.loads(existing.read_text())
        if record["status"] != "finished" or not record.get("teardown_verified"):
            raise RuntimeError(
                f"Refusing to reuse unsuccessful attempt {attempt}: {record['status']}"
            )
        if attempt not in completed:
            completed.append(attempt)
        OUTPUT.write_text(
            json.dumps({"status": "running", "completed": completed}, indent=2) + "\n"
        )
        return
    runtime = int(spec.get("limits", {}).get("timeout_seconds", 0))
    if runtime <= 0:
        raise ValueError(f"{sealed}: missing positive timeout_seconds")
    finish_budget = max(minimum_seconds, runtime + 180)
    is_fit = "predict-" not in attempt and "score-" not in attempt
    if time.time() + finish_budget > DEADLINE or (is_fit and time.time() >= FIT_START_CUTOFF):
        raise TimeoutError(
            f"not starting {attempt}: its {runtime}s timeout plus stop buffer does not fit before20:00 local"
        )
    command = [sys.executable, "scripts/run_campaign.py"]
    subprocess.run([*command, "launch", str(sealed), "--attempt", attempt], cwd=ROOT, check=True)
    wait_finished(attempt)
    completed.append(attempt)
    OUTPUT.write_text(json.dumps({"status": "running", "completed": completed}, indent=2) + "\n")


def score(runs: list[str], label: str, completed: list[str]) -> None:
    """Export completed fixed checkpoints and score only complete paired controls."""
    from continuation_recipes import prediction_recipe

    command = [sys.executable, "scripts/run_campaign.py"]
    pred_names: list[str] = []
    for run in runs:
        pred_name = run
        pred_names.append(pred_name)
        prediction_path = ROOT / f"cache/pred_{pred_name}_valid.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        checkpoint_path = ROOT / f"artifacts/runs/{run}/model_ep3.pt"
        if prediction_path.exists() or sidecar_path.exists():
            if not prediction_path.exists() or not sidecar_path.exists():
                raise FileExistsError(f"incomplete cached prediction pair for {run}")
            sidecar = json.loads(sidecar_path.read_text())
            if (
                sidecar.get("run") != run
                or sidecar.get("epoch") != 3
                or sidecar.get("checkpoint_sha256") != sha256_file(checkpoint_path)
                or sidecar.get("prediction_sha256") != sha256_file(prediction_path)
            ):
                raise ValueError(f"cached prediction does not match fixed checkpoint: {run}")
            continue
        draft = prediction_recipe(run, 3)
        draft["name"] = f"e424_predict_{run}"
        draft["argv"] += ["--inference-precision", "fp32", "--output-name", pred_name]
        draft["artifacts"] = [f"cache/pred_{pred_name}_valid.{ext}" for ext in ("npy", "json")]
        draft["limits"]["ram_bytes"] = 24 * 1024**3
        path = CAMPAIGN / "specs" / f"e424_predict_{run}_v2.draft.json"
        if not path.exists():
            path.write_text(json.dumps(draft, indent=2) + "\n")
        sealed = path.with_name(path.name.replace(".draft.json", ".sealed.json"))
        if not sealed.exists():
            subprocess.run([*command, "seal", str(path), str(sealed)], cwd=ROOT, check=True)
        launch(
            sealed, f"e424-predict-{run.replace('_', '-')}-v2", completed, minimum_seconds=4 * 60
        )

    prior_ctrl = "c20261004_combo_s3_final_r1"
    prior_head = "c20261004_combo_head_s3_final"
    baseline_checkpoints: dict[str, str] = {}
    for run in BASELINE:
        prediction = ROOT / f"cache/pred_{run}_valid.npy"
        sidecar = json.loads(prediction.with_suffix(".json").read_text())
        if sidecar.get("run") != run or sidecar.get("prediction_sha256") != sha256_file(prediction):
            raise ValueError(f"baseline prediction is not byte-bound to its sidecar: {run}")
        candidates = sorted((ROOT / "artifacts" / "runs" / run).glob("model*.pt"))
        matching = [p for p in candidates if sha256_file(p) == sidecar.get("checkpoint_sha256")]
        if len(matching) != 1:
            raise ValueError(f"baseline sidecar binds {len(matching)} checkpoints: {run}")
        baseline_checkpoints[run] = str(matching[0].relative_to(ROOT))
    arms: dict[str, dict[str, float]] = {
        "COMBO": {prior_ctrl: 0.5, prior_head: 0.5},
        "H093_LEARNED": {"c20261004_h093_learned_s3_final": 1.0},
        "H093_GRU": {"c20261004_h093_gru_budget_s3_final": 1.0},
        "H094_CONDITIONAL": {"c20261004_h094_conditional_s3_final": 1.0},
        "H094_GRU": {"c20261004_h094_gru_budget_s3_final": 1.0},
    }
    comparisons = [
        ["H093_LEARNED", "H093_GRU"],
        ["H094_CONDITIONAL", "H094_GRU"],
        ["H093_LEARNED", "COMBO"],
        ["H094_CONDITIONAL", "COMBO"],
    ]
    if label == "all":
        arms.update(
            {
                "H093_HASH": {"c20261004_h093_hash_s3_final": 1.0},
                "H093_UNIVERSAL": {"c20261004_h093_universal_s3_final": 1.0},
                "H094_UNCONDITIONAL": {"c20261004_h094_unconditional_s3_final": 1.0},
                "H094_DIAGONAL": {"c20261004_h094_diagonal_s3_final": 1.0},
            }
        )
        comparisons += [
            ["H093_HASH", "H093_GRU"],
            ["H093_UNIVERSAL", "H093_GRU"],
            ["H094_UNCONDITIONAL", "H094_GRU"],
            ["H094_DIAGONAL", "H094_GRU"],
            ["H093_HASH", "COMBO"],
            ["H093_UNIVERSAL", "COMBO"],
            ["H094_UNCONDITIONAL", "COMBO"],
            ["H094_DIAGONAL", "COMBO"],
        ]
    screen = {"folds": ["holdout"], "arms": arms, "comparisons": comparisons, "n_boot": 2000}
    screen_path = CAMPAIGN / "checks" / f"e424_screen_{label}.json"
    screen_path.write_text(json.dumps(screen, indent=2) + "\n")
    score_path = CAMPAIGN / "checks" / f"e424_report_{label}.json"
    if score_path.exists():
        raise FileExistsError(score_path)
    template_path = CAMPAIGN / "specs/e419_precision_score_v1.draft.json"
    template = json.loads(template_path.read_text())
    template["name"] = f"e424_score_{label}"
    template["argv"] = [
        "{python}",
        "scripts/screen_score.py",
        str(screen_path),
        "--output",
        str(score_path),
    ]
    template["code"] = sorted(
        set(template["code"]) | {str(p.relative_to(ROOT)) for p in (ROOT / "src/wnn").glob("*.py")}
    )
    template["code"] = sorted(set(template["code"]) | {"scripts/screen_score.py"})
    template["inputs"] = [
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "cache/valid_targets.npz",
        "datasets/valid.parquet",
        "cache/chains.npz",
        str(screen_path),
    ]
    for run in [*BASELINE, *pred_names]:
        template["inputs"] += [f"cache/pred_{run}_valid.{ext}" for ext in ("npy", "json")]
        template["inputs"] += [
            baseline_checkpoints.get(run, f"artifacts/runs/{run}/model_ep3.pt"),
            f"artifacts/runs/{run}/provenance.json",
        ]
    template["artifacts"] = [str(score_path)]
    template["limits"] = {
        "cpus": 1,
        "ram_bytes": 2 * 1024**3,
        "gpu_memory_bytes": 0,
        "gpu_stop_bytes": 0,
        "timeout_seconds": 900,
    }
    template["environment"] = {}
    draft = CAMPAIGN / "specs" / f"e424_score_{label}_v2.draft.json"
    draft.write_text(json.dumps(template, indent=2) + "\n")
    sealed = draft.with_name(draft.name.replace(".draft.json", ".sealed.json"))
    subprocess.run([*command, "seal", str(draft), str(sealed)], cwd=ROOT, check=True)
    launch(sealed, f"e424-score-{label}-v2", completed, minimum_seconds=16 * 60)


def main() -> None:
    """Fit four primary paired arms, score them, then continue secondary arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if OUTPUT.exists() and not arguments.resume:
        raise FileExistsError(OUTPUT)
    completed: list[str] = []
    if arguments.resume and OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text())
        completed = previous["completed"]
        for attempt in completed:
            record = json.loads((RUNNER / f"attempts/{attempt}.json").read_text())
            if record["status"] != "finished" or not record.get("teardown_verified"):
                raise RuntimeError(f"Resume list contains an unverified attempt: {attempt}")
    OUTPUT.write_text(json.dumps({"status": "running", "completed": completed}, indent=2) + "\n")
    try:
        for arm in CORE:
            sealed = CAMPAIGN / "specs" / f"e424_{arm}_v2.sealed.json"
            if not sealed.exists():
                raise FileNotFoundError(sealed)
            attempt = f"e424-{arm.replace('_', '-')}-v2"
            launch(sealed, attempt, completed)
        score([f"c20261004_{arm}_s3_final" for arm in CORE], "core", completed)
        for arm in ARMS[4:]:
            sealed = CAMPAIGN / "specs" / f"e424_{arm}_v2.sealed.json"
            if not sealed.exists():
                raise FileNotFoundError(sealed)
            attempt = f"e424-{arm.replace('_', '-')}-v2"
            launch(sealed, attempt, completed)
        score([f"c20261004_{arm}_s3_final" for arm in ARMS], "all", completed)
        OUTPUT.write_text(
            json.dumps({"status": "completed", "completed": completed}, indent=2) + "\n"
        )
    except Exception as error:
        OUTPUT.write_text(
            json.dumps(
                {
                    "status": "stopped",
                    "completed": completed,
                    "reason": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
