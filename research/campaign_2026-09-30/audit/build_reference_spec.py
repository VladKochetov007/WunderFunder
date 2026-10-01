"""Freeze four-fold reference design and bind completed clean-control lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaign_score import ROOT, file_sha256, mmap_stored_npz, validate_member

from wnn.splits import load_campaign_split

AUDIT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "research/campaign_2026-09-29"


def identity(path: Path) -> dict[str, str]:
    """Bind an existing artifact without changing its bytes."""
    return {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}


def publish(path: Path, value: dict) -> None:
    """Refuse to replace an earlier design or specification."""
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def reference_design() -> dict:
    """Freeze exact clean seed3/seed4 fine-tunes, equal weights, all four folds."""
    folds = []
    for fold in range(4):
        members = []
        for seed in (3, 4):
            name = f"c20260929_ft_since_s{seed}_f{fold}"
            members.append(
                {
                    "run": name,
                    "epoch": 0,
                    "weight": 0.5,
                    "fit_source": "valid",
                    "checkpoint": f"artifacts/runs/{name}/model_ep0.pt",
                    "expected_args": {
                        "seed": 3,
                        "valid_fold": fold,
                        "epochs": 1,
                        "loss": "gcorr",
                        "batch": 16,
                        "hidden": 128,
                        "layers": 2,
                        "train_source": "valid",
                        "ema_teacher": 0.0,
                        "eval_ema": False,
                        "aux_struct": 0.3,
                        "aux_next": 0.3,
                        "init": f"{name}h@ep0",
                        "checkpoint_policy": "fixed",
                        "checkpoint_epoch": 0,
                        "no_eval_during_fit": True,
                    },
                }
            )
        folds.append(
            {
                "fold": fold,
                "manifest": identity(CAMPAIGN / f"artifacts/splits/fold{fold}.json"),
                "roles": {"reference": members},
            }
        )
    return {
        "metric_id": "global_wp_clip2_v1",
        "n_boot": 2000,
        "bootstrap_seed": 271828,
        "comparisons": [],
        "derived_predictions": {},
        "folds": folds,
    }


def checkpoint_binding(run: str, epoch: int, manifest_sha: str) -> dict:
    """Validate one ancestor's checkpoint hashes and clean fixed fitting policy."""
    directory = ROOT / "artifacts/runs" / run
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checkpoint = directory / f"model_ep{epoch}.pt"
    if (
        provenance["split_manifest_sha256"] != manifest_sha
        or provenance["selected_epoch"] != epoch
        or provenance["args"]["checkpoint_policy"] != "fixed"
        or provenance["args"]["checkpoint_epoch"] != epoch
        or not provenance["args"]["no_eval_during_fit"]
        or provenance["checkpoint_sha256"].get(checkpoint.name) != file_sha256(checkpoint)
    ):
        raise ValueError(f"{run}: fixed clean ancestor checkpoint provenance mismatch")
    return {
        "run": run,
        "epoch": epoch,
        "checkpoint": identity(checkpoint),
        "provenance": identity(provenance_path),
        "seed": provenance["seed"],
        "args": provenance["args"],
        "fit_row_groups": provenance["fit_row_groups"],
        "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
        "propensity_checkpoint_sha256": provenance.get("propensity_checkpoint_sha256"),
    }


def bind(design_path: Path, design: dict) -> dict:
    """Build a scorer input after every final FT and prediction sidecar exists."""
    cache_path = ROOT / "cache/valid_targets.npz"
    cache = mmap_stored_npz(cache_path)
    row_groups = cache["row_group"].astype(int).tolist()
    artifacts, lineage = {}, {}
    for fold in design["folds"]:
        index = fold["fold"]
        split = load_campaign_split(ROOT / fold["manifest"]["path"], ROOT)
        manifest = json.loads((ROOT / fold["manifest"]["path"]).read_text())
        artifacts[str(index)], lineage[str(index)] = {}, {}
        for member in fold["roles"]["reference"]:
            name = member["run"]
            prediction = ROOT / "cache" / f"pred_{name}_valid.npy"
            bundle = {
                "prediction": identity(prediction),
                "sidecar": identity(prediction.with_suffix(".json")),
                "checkpoint": identity(ROOT / member["checkpoint"]),
                "provenance": identity(ROOT / "artifacts/runs" / name / "provenance.json"),
                "job": identity(CAMPAIGN / "jobs" / f"{name}.json"),
                "initial_checkpoint": identity(
                    ROOT / "artifacts/runs" / f"{name}h" / "model_ep0.pt"
                ),
            }
            validate_member(member, bundle, split, manifest, row_groups)
            middle = checkpoint_binding(f"{name}h", 0, split.sha256)
            parent_name = name.replace("c20260929_ft_since_", "c20260929_since_", 1)
            parent = checkpoint_binding(parent_name, 3, split.sha256)
            propensity = checkpoint_binding(f"c20260929_prop_f{index}", 3, split.sha256)
            if (
                middle["seed"] != 3
                or middle["args"]["init"] != f"{parent_name}@ep3"
                or middle["initial_checkpoint_sha256"] != parent["checkpoint"]["sha256"]
            ):
                raise ValueError(f"{name}: stage1 parent/seed binding mismatch")
            prop_model = ROOT / "artifacts/runs" / propensity["run"] / "model.pt"
            if parent["propensity_checkpoint_sha256"] != file_sha256(prop_model):
                raise ValueError(f"{name}: parent propensity binding mismatch")
            job = json.loads((CAMPAIGN / "jobs" / f"{name}.json").read_text())
            command = job["command"]
            location = command.index("scripts/ft_gclp.sh")
            if command[location + 1 : location + 5] != [
                f"{parent_name}@ep3",
                name,
                str(index),
                "3",
            ]:
                raise ValueError(f"{name}: durable job does not bind both-stage FT recipe")
            for stage in (middle, parent, propensity):
                fit = stage["fit_row_groups"]
                split.assert_fit_groups(fit.get("train", []), fit.get("valid", []))
            artifacts[str(index)][name] = bundle
            lineage[str(index)][name] = {
                "stage2_final": bundle,
                "stage1_head_warmup": middle,
                "parent_fixed_epoch3": parent,
                "fold_propensity_fixed_epoch3": propensity,
                "parent_job": identity(CAMPAIGN / "jobs" / f"{parent_name}.json"),
                "propensity_job": identity(CAMPAIGN / "jobs" / f"{propensity['run']}.json"),
                "both_ft_stages_owned_by_one_durable_job": bundle["job"],
                "seed4_parent_ft_seed_is3": True,
            }
    return {
        "scorer_freeze": identity(AUDIT / "campaign_score_freeze.json"),
        "preregistration": identity(design_path),
        "targets": identity(cache_path),
        "design": design,
        "artifacts": artifacts,
        "backward_lineage": lineage,
        "builder_source_sha256": file_sha256(Path(__file__)),
    }


def main() -> None:
    """Create design alone now, or bind all completed immutable artifacts later."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.design.exists():
        design = json.loads(args.design.read_text())["design"]
        if design != reference_design():
            raise ValueError("prior reference design differs")
    else:
        design = reference_design()
        publish(
            args.design,
            {
                "purpose": "reference-only4fold clean recipe analog, no candidate predictions",
                "design": design,
            },
        )
    if args.spec is not None:
        publish(args.spec, bind(args.design.resolve(), design))
    print(
        json.dumps(
            {"design": str(args.design), "spec": None if args.spec is None else str(args.spec)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
