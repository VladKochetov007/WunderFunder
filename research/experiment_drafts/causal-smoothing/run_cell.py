from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from cell_config import (
    ARMS,
    TRAIN_SCRIPT,
    cell_key,
    data_root,
    run_name,
    sha256,
    source_identity,
    training_dir,
)
from cell_plan import command_for, current_bindings, paths, plan
from cell_support import (
    begin_stage,
    execute_logged,
    finish_stage,
    process_environment,
    read_json,
    shell,
    stage_digest,
    verify_fit,
)

PHASES = ("profile", "fit", "predict", "score")
SEEDS = {0: range(1001, 1009), 1: range(1001, 1009), 2: range(1001, 1008), 3: range(1001, 1008)}


def verify_prior(path: Path, expected: dict[str, Any], *, profile: bool = False) -> dict[str, Any]:
    """Require a successful, provenance-matched predecessor stage."""
    prior = read_json(path)
    if prior.get("status") != "complete":
        raise ValueError(f"required predecessor is not complete: {path}")
    if prior.get("binding_sha256") != stage_digest(prior):
        raise ValueError(f"predecessor receipt checksum differs: {path}")
    keys = ("arm", "recipe_sha256", "source_identity", "dataset_identity_sha256")
    if not profile:
        keys += ("fold", "seed", "attempt", "split", "run_name")
    for key in keys:
        if prior.get(key) != expected.get(key):
            raise ValueError(f"predecessor binding changed: {key}")
    return prior


def hardware_inventory(rows: list[str]) -> list[tuple[str, str, str]]:
    """Reduce nvidia-smi output to ordered model, memory, and driver identity."""
    result = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 4:
            raise ValueError("unrecognized nvidia-smi hardware inventory")
        result.append((parts[0], parts[2], parts[3]))
    return result


def profile_predecessors(
    root: Path, args: argparse.Namespace, bindings: dict[str, Any], runtime: dict[str, Any]
) -> None:
    """Require current representative GPU profiles for both matched feature arms."""
    for arm in ARMS:
        prior_path = (
            root
            / "artifacts/causal-smoothing-runner/profiles"
            / f"profile-{arm}-a{args.profile_attempt}"
            / "profile.json"
        )
        expected = {**bindings, "arm": arm}
        profile = verify_prior(prior_path, expected, profile=True)
        if profile.get("profile_evidence", {}).get("valid_for_throughput") is not True:
            raise ValueError(f"profile evidence is invalid for {arm}")
        profile_rows = profile.get("runtime", {}).get("nvidia_smi", [])
        fit_rows = runtime.get("nvidia_smi", [])
        if not profile_rows or hardware_inventory(profile_rows) != hardware_inventory(fit_rows):
            raise ValueError(
                f"fit GPU inventory differs from its profiled hardware envelope: {arm}"
            )


def execute(args: argparse.Namespace) -> None:
    """Run one explicit stage and persist success or failure before returning."""
    root = data_root()
    stage_dir, key = paths(root, args.phase, args.arm, args.fold, args.seed, args.attempt)
    receipt_path = stage_dir / f"{args.phase}.json"
    name = run_name(args.arm, args.fold, args.seed, args.attempt, profile=args.phase == "profile")
    receipt: dict[str, Any] = {
        "schema": "wnn.causal-smoothing-stage.v1",
        "phase": args.phase,
        "cell": key,
        "arm": args.arm,
        "fold": args.fold,
        "seed": args.seed,
        "attempt": args.attempt,
        "profile_attempt": args.profile_attempt,
        "run_name": name,
    }
    begin_stage(receipt_path, receipt)
    try:
        bindings = current_bindings(root, args)
        receipt.update({key: value for key, value in bindings.items() if key != "recipe"})
        training = bindings["recipe"]["training"]
        if args.phase == "profile" and (args.fold != 0 or args.seed != 1001):
            raise ValueError("profile only representative fold0/seed1001 cells")
        if args.phase in ("predict", "score"):
            base = (
                root
                / "artifacts/causal-smoothing-runner/cells"
                / cell_key(args.arm, args.fold, args.seed, args.attempt)
            )
            fit = verify_prior(base / "fit.json", {**receipt}, profile=False)
            if args.phase == "score":
                prediction = verify_prior(base / "predict.json", {**receipt}, profile=False)
                pred_path = root / "cache" / f"pred_{name}_valid.npy"
                if sha256(pred_path) != prediction.get("prediction", {}).get("sha256"):
                    raise ValueError("prediction bytes changed after its successful stage")
            else:
                checkpoint = Path(fit["fit_evidence"]["checkpoint"])
                if sha256(checkpoint) != fit["fit_evidence"]["checkpoint_sha256"]:
                    raise ValueError("fixed checkpoint bytes changed after fit")
        env, runtime = process_environment(
            root, require_gpu=args.phase in ("profile", "fit", "predict")
        )
        receipt["runtime"] = runtime
        if args.phase == "fit":
            profile_predecessors(root, args, bindings, runtime)
        split = bindings["split"]
        command = command_for(args, root, split, training)
        receipt["command"], receipt["command_shell"] = command, shell(command)
        if args.phase in ("profile", "fit"):
            if not TRAIN_SCRIPT.is_file():
                raise FileNotFoundError(f"packet trainer is missing: {TRAIN_SCRIPT}")
            out = training_dir(root, name)
            if out.exists():
                raise FileExistsError(f"refusing to reuse trainer output: {out}")
        if args.phase == "predict" and (root / "cache" / f"pred_{name}_valid.npy").exists():
            raise FileExistsError("refusing to overwrite existing prediction")
        code, log = execute_logged(command, env, stage_dir / f"{args.phase}.log")
        receipt["stdout_log"], receipt["exit_code"] = log, code
        if code:
            raise RuntimeError(f"{args.phase} command exited with status {code}")
        if args.phase == "profile":
            marker = "PROFILE_UPDATE_EVIDENCE "
            text = Path(log).read_text(errors="replace")
            payload = next(
                (line.split(marker, 1)[1] for line in text.splitlines() if marker in line), None
            )
            evidence = json.loads(payload) if payload else {}
            if evidence.get("valid_for_throughput") is not True:
                raise ValueError("profile did not emit complete finite update evidence")
            receipt["profile_evidence"] = evidence
        elif args.phase == "fit":
            receipt["fit_evidence"] = verify_fit(
                root,
                name,
                training,
                bindings["recipe_sha256"],
                args.arm,
                args.fold,
                args.seed,
                bindings["split"],
                bindings["source_identity"],
            )
        elif args.phase == "predict":
            pred = root / "cache" / f"pred_{name}_valid.npy"
            metadata = read_json(pred.with_suffix(".json"))
            if (
                metadata.get("epoch") != training["checkpoint_epoch"]
                or metadata.get("checkpoint_sha256") != fit["fit_evidence"]["checkpoint_sha256"]
                or metadata.get("campaign_split_manifest_sha256") != split["sha256"]
                or metadata.get("campaign_export_scope") != "evaluation"
                or metadata.get("inference_precision") != "fp32"
                or metadata.get("quantization_preset") is not None
                or metadata.get("inference_batch") != training["batch"]
                or metadata.get("inference_chunk") != training["chunk"]
                or metadata.get("row_groups")
                != json.loads(Path(split["path"]).read_text())["evaluation"]["valid"]
            ):
                raise ValueError(
                    "prediction metadata does not match the fixed fit and evaluation membership"
                )
            prediction_sha = sha256(pred)
            if metadata.get("prediction_sha256") != prediction_sha:
                raise ValueError("prediction bytes differ from the immutable sidecar hash")
            receipt["prediction"] = {
                "path": str(pred),
                "sha256": prediction_sha,
                "metadata_sha256": sha256(pred.with_suffix(".json")),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            }
        else:
            score_path = stage_dir / "grade_result.json"
            score = read_json(score_path)
            if (
                score.get("prediction_sha256") != prediction["prediction"]["sha256"]
                or score.get("checkpoint_sha256") != fit["fit_evidence"]["checkpoint_sha256"]
                or score.get("split_manifest_sha256") != split["sha256"]
            ):
                raise ValueError("grade receipt changed prediction, checkpoint, or fold binding")
            other_arm = next(arm for arm in ARMS if arm != args.arm)
            other_fit_path = (
                root
                / "artifacts/causal-smoothing-runner/cells"
                / cell_key(other_arm, args.fold, args.seed, args.attempt)
                / "fit.json"
            )
            # First completed arm stays explicitly unpaired. The later score verifies both.
            receipt["pair_order"] = {"status": "awaiting_counterpart"}
            if other_fit_path.exists() and read_json(other_fit_path).get("status") == "complete":
                other_fit = verify_prior(
                    other_fit_path,
                    {
                        **receipt,
                        "arm": other_arm,
                        "run_name": run_name(other_arm, args.fold, args.seed, args.attempt),
                    },
                )
                if other_fit["fit_evidence"].get("sampling_epochs") != fit["fit_evidence"].get(
                    "sampling_epochs"
                ):
                    raise ValueError("paired arms have different epoch sampling identities")
                receipt["pair_order"] = {
                    "status": "matched",
                    "counterpart_fit": str(other_fit_path),
                    "counterpart_binding_sha256": other_fit["binding_sha256"],
                }
            receipt["score"] = score
        if source_identity()["sha256"] != bindings["source_identity"]["sha256"]:
            raise RuntimeError("source files changed while the stage was running")
        receipt["binding_sha256"] = stage_digest(receipt)
        finish_stage(receipt_path, receipt)
    except Exception as exc:
        receipt["traceback"] = traceback.format_exc()
        finish_stage(receipt_path, receipt, f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    """Plan by default; execute only when explicitly authorized on a GPU host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--profile-attempt", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="run the planned GPU/data stage")
    args = parser.parse_args()
    if args.seed not in SEEDS[args.fold] or args.attempt < 1 or args.profile_attempt < 1:
        parser.error("seed or attempt is outside the frozen cell contract")
    try:
        execute(args) if args.execute else plan(args)
    except Exception as exc:
        parser.exit(2, f"stage failed (failure receipt persisted for --execute): {exc}\n")


if __name__ == "__main__":
    main()
