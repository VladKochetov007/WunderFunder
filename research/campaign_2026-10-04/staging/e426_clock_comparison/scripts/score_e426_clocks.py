"""Validate fixed E426 prediction exports, then run the frozen fold-0 scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNS = {
    "constant": ("e426_f0_constant_s3_4ep_v2", "e426-f0-constant-s3-4ep-v2"),
    "aligned": ("e426_f0_aligned_s3_4ep_v2", "e426-f0-aligned-s3-4ep-v2"),
    "shuffled": ("e426_f0_shuffled_s3_4ep_v2", "e426-f0-shuffled-s3-4ep-v2"),
}
PREDICTIONS = {clock: f"{run}_fp32" for clock, (run, _) in RUNS.items()}
SCREEN = CAMPAIGN / "staging/e426_clock_comparison/specs/e426_clock_screen_v1.json"
FOLD0 = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_fit_and_export_records() -> None:
    """Bind checkpoint and prediction bytes to the fixed successful fit outputs."""
    import torch

    import numpy as np

    manifest_sha = sha256(FOLD0)
    manifest = json.loads(FOLD0.read_text())
    evaluation = manifest["evaluation"]["valid"]
    require(len(evaluation) == 468, "fold-0 VALID evaluation domain changed")
    require(all(type(group) is int for group in evaluation), "fold-0 group IDs must be typed integers")

    # Use the frozen scorer's stored-NPZ mmap reader, so validation does not
    # unpack the 337-MB target archive or construct a second target copy.
    sys.path.insert(0, str(ROOT / "src"))
    import importlib.util

    frozen_path = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
    frozen_spec = importlib.util.spec_from_file_location("e426_frozen_score", frozen_path)
    frozen = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen)
    targets = frozen.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
    groups_array = targets["row_group"]
    require(groups_array.dtype.kind in "iu", "VALID row-group IDs must have integer dtype")
    groups = groups_array.tolist()
    require(
        len(groups) == 1873
        and all(type(group) is int for group in groups)
        and len(set(groups)) == 1873
        and groups == list(range(1873)),
        "VALID prediction domain must be typed canonical IDs 0..1872 in target-cache order",
    )
    require(targets["y"].shape == (1873, 20_000, 2), "VALID target sequence shape changed")
    require(targets["mask"].shape == (1873, 20_000), "VALID mask shape changed")
    require(set(int(value) for value in targets["market_class"]) == {0, 1, 2, 3},
            "VALID class labels must contain all four declared classes")

    shared_args = None
    shared_norm = None
    shared_closure = None
    shared_fit = None
    for clock, (run, attempt_name) in RUNS.items():
        run_dir = ROOT / "artifacts/runs" / run
        checkpoint_path = run_dir / "model_ep3.pt"
        training_state_path = run_dir / "training_state.pt"
        provenance_path = run_dir / "provenance.json"
        attempt_path = CAMPAIGN / f"runner/attempts/{attempt_name}.json"
        attempt = json.loads(attempt_path.read_text())
        require(
            attempt.get("spec", {}).get("name") == run
            and attempt.get("kind") == "gpu"
            and attempt.get("status") == "finished"
            and attempt.get("exit_code") == 0
            and attempt.get("teardown_verified") is True,
            f"{clock}: fit attempt is not terminal-successful",
        )
        fit_artifacts = {item["path"]: item["sha256"] for item in attempt["artifacts"]}
        relative_checkpoint = str(checkpoint_path.relative_to(ROOT))
        require(fit_artifacts.get(relative_checkpoint) == sha256(checkpoint_path),
                f"{clock}: checkpoint bytes do not match authoritative fit attempt")
        require(fit_artifacts.get(str(provenance_path.relative_to(ROOT))) == sha256(provenance_path),
                f"{clock}: provenance bytes do not match authoritative fit attempt")

        provenance = json.loads(provenance_path.read_text())
        args = provenance["args"]
        require(
            args.get("name") == run
            and args.get("arch") == "gru"
            and args.get("eval_ema") is True
            and args.get("no_eval_during_fit") is True
            and args.get("checkpoint_policy") == "fixed"
            and args.get("checkpoint_epoch") == 3
            and args.get("epochs") == 4
            and args.get("teacher_ema_clock") == clock
            and args.get("split_manifest_sha256") == manifest_sha
            and provenance.get("selected_epoch") == 3
            and provenance.get("split_manifest_sha256") == manifest_sha
            and provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") == sha256(checkpoint_path),
            f"{clock}: not the fixed no-evaluation epoch-3 clean-fold GRU checkpoint",
        )
        comparable_args = dict(args)
        comparable_args.pop("name", None)
        comparable_args.pop("teacher_ema_clock", None)
        if shared_args is None:
            shared_args = comparable_args
            shared_norm = provenance["normalization_identity"]
            shared_closure = provenance["code_source_closure"]
            shared_fit = provenance["fit_membership_identity"]
        else:
            require(comparable_args == shared_args, f"{clock}: training recipe changed beyond clock/run name")
            require(provenance["normalization_identity"] == shared_norm,
                    f"{clock}: saved input normalization differs across clocks")
            require(provenance["code_source_closure"] == shared_closure,
                    f"{clock}: saved training source closure differs across clocks")
            require(provenance["fit_membership_identity"] == shared_fit,
                    f"{clock}: training row membership/order differs across clocks")
        norm = provenance["normalization_identity"]
        require(norm["mean"]["shape"] == [211] and norm["std"]["shape"] == [211],
                f"{clock}: expected 211-feature normalization")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        require(checkpoint.get("epoch") == 3 and checkpoint.get("n_in") == 211,
                f"{clock}: checkpoint epoch or input width changed")
        model_state = checkpoint.get("model")
        require(isinstance(model_state, dict), f"{clock}: checkpoint lacks ordinary inference model state")
        head_weight = model_state.get("head.2.weight")
        require(head_weight is not None and tuple(head_weight.shape)[0] == 23,
                f"{clock}: checkpoint must preserve the registered 23-output head")
        evaluation_ema = state.get("evaluation_ema")
        require(isinstance(evaluation_ema, dict) and evaluation_ema.get("kind") == "fast",
                f"{clock}: saved training state lacks the separate evaluation EMA")
        ema_state = evaluation_ema.get("state")
        require(isinstance(ema_state, dict) and ema_state.keys() == model_state.keys(),
                f"{clock}: checkpoint state keys differ from fixed evaluation EMA")
        require(
            all(torch.equal(model_state[key], ema_state[key]) for key in model_state),
            f"{clock}: exported inference checkpoint is not the saved fixed evaluation EMA",
        )
        del checkpoint, state, model_state, ema_state

        prediction_path = ROOT / "cache" / f"pred_{PREDICTIONS[clock]}_valid.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        require(prediction_path.is_file() and sidecar_path.is_file(), f"{clock}: FP32 export is missing")
        sidecar = json.loads(sidecar_path.read_text())
        require(
            sidecar.get("run") == run
            and sidecar.get("split") == "valid"
            and sidecar.get("epoch") == 3
            and sidecar.get("checkpoint_sha256") == sha256(checkpoint_path)
            and sidecar.get("prediction_sha256") == sha256(prediction_path)
            and sidecar.get("split_manifest_sha256") == manifest_sha
            and sidecar.get("inference_precision") == "fp32"
            and sidecar.get("row_groups") == groups,
            f"{clock}: prediction sidecar does not bind the fixed checkpoint and canonical all-VALID rows",
        )
        predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        require(predictions.shape == (1873, 20_000, 2) and predictions.dtype == np.float32,
                f"{clock}: prediction must be FP32 [1873,20000,2]")
        del predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path = output_path.resolve()
    if not output_path.is_relative_to(ROOT):
        raise ValueError("report output must stay inside the repository")
    if output_path.exists():
        raise FileExistsError(output_path)
    validate = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--preflight-only"],
                              cwd=ROOT, check=False)
    if validate.returncode:
        raise subprocess.CalledProcessError(validate.returncode, validate.args)
    screen = subprocess.run(
        [sys.executable, "scripts/screen_score.py", str(SCREEN.relative_to(ROOT)),
         "--output", str(output_path.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
    )
    if screen.returncode:
        raise subprocess.CalledProcessError(screen.returncode, screen.args)
    report = json.loads(output_path.read_text())
    require(report.get("metric_id") == "global_wp_clip2_v1", "frozen global-WP metric changed")
    require(report.get("n_sequences") == 468, "paired screen must score exactly fold-0's 468 evaluation sequences")
    expected_screen = json.loads(SCREEN.read_text())
    require(report.get("screen") == expected_screen, "scored screen differs from the fixed preregistration")
    fold = report.get("folds", {}).get("0", {})
    require(set(fold.get("exact_global_wp", {})) == set(RUNS), "fold-0 arm set changed")
    classes = fold.get("by_class", {})
    require(set(classes) == {"0", "1", "2", "3"}, "fold-0 class guard did not report all four classes")
    comparisons = fold.get("paired_component_bootstrap", {})
    require(
        set(comparisons) == {
            "aligned_minus_constant",
            "aligned_minus_shuffled",
            "shuffled_minus_constant",
        },
        "paired component-bootstrap contrasts differ from preregistration",
    )
    require(
        all(value.get("n_boot") == 2000 for value in comparisons.values()),
        "paired component-bootstrap did not use all 2,000 declared replicates",
    )
    require(
        all(value.get("n_sequences") == 468 and value.get("n_components") == 141
            for value in comparisons.values()),
        "bootstrap domain differs from fold-0 evaluation sequences/components",
    )
    print(json.dumps({"status": "PASS", "report": str(output_path.relative_to(ROOT)), "n_sequences": 468,
                      "n_boot": 2000, "comparisons": sorted(comparisons)}, sort_keys=True))


if __name__ == "__main__":
    if "--preflight-only" in sys.argv:
        sys.argv.remove("--preflight-only")
        validate_fit_and_export_records()
        print("E426 fixed-epoch/checkpoint/export preflight passed")
    else:
        main()
