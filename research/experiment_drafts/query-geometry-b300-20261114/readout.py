"""Compare the six independent fixed-E4 grades against the native PR75 recipe."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent
RUNTIME = "8626000a4b965763a70011a0291a891aa6535d7e"
BASE = Path("research/experiment_drafts/transformer-qk-normalization-20261109")
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
SEEDS = (209401, 209402)
ARMS = {"plain": "none", "angular": "angular", "radial": "radial"}
COMPARISONS = (("angular", "plain"), ("radial", "plain"), ("angular", "radial"))


def native_grader(runtime: Path) -> dict:
    """Use the same centered-component bootstrap and grade parser as the frozen baseline."""

    from prepare_runtime import verify_preparation

    verify_preparation(runtime)
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    os.environ.setdefault("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
    return runpy.run_path(str(runtime / BASE / "compare_panel.py"))


def collect(grader: dict, args: argparse.Namespace) -> dict:
    """Validate six endpoint lineages before comparing matched seeds."""
    read = grader["read"]
    grade = grader["grade"]
    canonical = grader["canonical_sha256"]
    digest = grader["file_sha256"]
    load_config = grader["load_config"]
    source = grader["source_identity"](
        args.runtime_root, args.runtime_root / "scripts/train_sequence.py"
    )
    split = grader["load_campaign_split"](args.split_manifest, args.data_root)
    if split.sha256 != SPLIT or split.valid_groups or len(split.train_groups) != 10_325:
        raise ValueError("readout requires the frozen train-only fit population")
    fitting = [("train", group) for group in split.train_groups]
    evaluation = [("valid", group) for group in split.evaluation_groups]
    if len(set(evaluation)) != 1_873 or len(evaluation) != 1_873:
        raise ValueError("readout requires the complete official evaluation population")

    qualification = read(args.run_root / "qualification.json")
    packet_commit = subprocess.check_output(
        ["git", "-C", str(PACKET), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if (
        qualification["schema"] != "wnn.query-geometry.qualification.v1"
        or qualification["runtime_commit"] != RUNTIME
        or qualification["packet_commit"] != packet_commit
        or qualification["source_sha256"] != source["sha256"]
        or qualification["split_sha256"] != SPLIT
    ):
        raise ValueError("qualification belongs to a different panel or runtime")
    # The receiver admits all cells against the same short B300 streaming probe.
    from query_probe import validate_probe

    validate_probe(read(args.run_root / "probe.json"), qualification)
    template = load_config(args.runtime_root / BASE / "configs/plain_s203301.json")
    receipts, scores, contrasts = {}, {}, []
    common = set()
    for seed in SEEDS:
        grades = {}
        pair_runtime = set()
        for arm in ARMS:
            cell = f"{arm}_s{seed}"
            root = args.run_root / cell
            fit = root / "fit"
            expected = template | {
                "seed": seed,
                "model": template["model"] | {"query_perturbation": ARMS[arm]},
            }
            config = load_config(PACKET / "configs" / f"{cell}.json")
            if config != expected or qualification["configs"].get(cell) != canonical(config):
                raise ValueError(
                    f"{cell}: recipe differs beyond the declared perturbation and seed"
                )
            hashes = {
                "checkpoint_sha256": digest(fit / "model.pt"),
                "score_sha256": digest(root / "score.json"),
                "predictions_manifest_sha256": digest(root / "predictions/manifest.json"),
                "terminal_status_sha256": digest(fit / "terminal_status.json"),
                "epoch_index_file_sha256": digest(fit / "epoch_index.json"),
                "profile_sha256": digest(root / "profile/profile_receipt.json"),
                "approved_profile_sha256": digest(root / "approved_profile.json"),
                "config_sha256": canonical(config),
                "source_sha256": source["sha256"],
                "split_sha256": SPLIT,
            }
            row = grade(root / "score.json")
            producer = row["producer"]
            grader["validate_producer"](producer, SPLIT, fitting)
            provenance = producer["provenance"]
            identity = provenance["identity"]
            terminal = read(fit / "terminal_status.json")
            manifest = read(root / "predictions/manifest.json")
            epoch_index = read(fit / "epoch_index.json")
            hashes["epoch_index_sha256"] = canonical(
                {key: value for key, value in epoch_index.items() if key != "index_sha256"}
            )
            approved = read(root / "approved_profile.json")
            if (
                approved["identity"] != identity
                or approved["approved"] is not True
                or approved["profile_sha256"] != hashes["profile_sha256"]
                or epoch_index["identity"] != identity
                or epoch_index["completed_epochs"] != 4
                or epoch_index["global_step"] != 3_280
                or epoch_index["index_sha256"] != hashes["epoch_index_sha256"]
            ):
                raise ValueError(f"{cell}: profile or epoch index does not bind this fit")
            if (
                row.get("predictions_manifest_sha256") != hashes["predictions_manifest_sha256"]
                or manifest.get("schema") != "sequence-predictions-v1"
                or manifest.get("complete") is not True
                or manifest.get("membership") != "evaluation"
                or manifest.get("input_schema") != "raw112-v1"
                or manifest.get("hint_dim") != 0
                or manifest.get("sequence_length") != 20_000
                or manifest.get("world_size") != 2
                or manifest.get("split_sha256") != SPLIT
                or manifest.get("producer") != producer
                or [(e["split"], e["row_group"]) for e in manifest["sequences"]] != evaluation
            ):
                raise ValueError(f"{cell}: official grade and complete export differ")
            if (
                terminal.get("status") != "completed"
                or terminal.get("identity") != identity
                or terminal.get("completed_epochs") != 4
                or terminal.get("global_updates") != 3_280
                or terminal.get("model_sha256") != hashes["checkpoint_sha256"]
                or producer.get("checkpoint_sha256") != hashes["checkpoint_sha256"]
                or terminal.get("epoch_index_sha256") != hashes["epoch_index_sha256"]
                or provenance.get("epoch_index_sha256") != hashes["epoch_index_sha256"]
                or provenance.get("checkpoint_purpose") != "fixed_configured_final_endpoint"
                or provenance.get("completed_epochs") != 4
                or provenance.get("global_updates") != 3_280
                or provenance["profile_receipt"]["sha256"] != hashes["approved_profile_sha256"]
                or producer.get("config") != config
                or producer.get("teacher") is not None
                or "reference" in producer
                or "kind" in producer
                or identity.get("source_files") != source["files"]
                or identity.get("source_sha256") != hashes["source_sha256"]
                or identity.get("config_sha256") != hashes["config_sha256"]
                or identity.get("world_size") != 2
                or identity["data"].get("split_sha256") != SPLIT
                or identity["data"].get("fit_members") != [list(member) for member in fitting]
                or identity["data"].get("evaluation_members")
                != [list(member) for member in evaluation]
            ):
                raise ValueError(
                    f"{cell}: checkpoint, config, source or fixed E4 provenance mismatch"
                )
            # Preserve rank, device class, capacity and runtime, permitting disjoint GPU pairs.
            devices = identity["runtime"]["devices"]
            if len(devices) != 2 or any(
                item["rank"] != rank or "B300" not in item["name"]
                for rank, item in enumerate(devices)
            ):
                raise ValueError(f"{cell}: expected two B300 ranks")
            hardware = [
                {key: val for key, val in item.items() if key != "uuid"} for item in devices
            ]
            pair_runtime.add(canonical(identity["runtime"]))
            common.add(canonical([grader["comparable_identity"](producer), hardware]))
            grades[arm] = row
            scores[cell], receipts[cell] = row["score"], hashes

        if len(pair_runtime) != 1:
            raise ValueError(f"seed {seed}: candidate and control need the same pair of GPUs")
        for candidate, control in COMPARISONS:
            # The native helper fixes sorted component IDs, 2,000 draws and seed 271828.
            # Thus all contrasts reuse identical bootstrap multiplicities.
            result = grader["contrast"](grades[candidate], grades[control])
            contrasts.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "control": control,
                    "role": "primary_screen" if control == "plain" else "diagnostic",
                    **result,
                }
            )

    if (
        len(common) != 1
        or len({item["checkpoint_sha256"] for item in receipts.values()}) != 6
        or set(receipts) != set(qualification["configs"])
    ):
        raise ValueError("panel requires six distinct checkpoints and matched fit conditions")
    summaries = []
    for candidate, control in COMPARISONS:
        matched = [
            row for row in contrasts if (row["candidate"], row["control"]) == (candidate, control)
        ]
        mean_delta = sum(row["mean_diff"] for row in matched) / len(SEEDS)
        summaries.append(
            {
                "candidate": candidate,
                "control": control,
                "mean_delta": mean_delta,
                "both_seeds_positive": all(row["mean_diff"] > 0 for row in matched),
                "followup_budget_trigger_only": control == "plain"
                and mean_delta >= 0.001
                and all(row["mean_diff"] > 0 for row in matched),
            }
        )
    return {
        "schema": "wnn.query-geometry.comparison.v1",
        "status": "complete_exposed_fixed_e4",
        "runtime_commit": RUNTIME,
        "packet_commit": packet_commit,
        "split_sha256": SPLIT,
        "qualification_sha256": digest(args.run_root / "qualification.json"),
        "probe_sha256": digest(args.run_root / "probe.json"),
        "receipts": receipts,
        "scores": scores,
        "contrasts": contrasts,
        "summaries": summaries,
        "automatic_next_fit": False,
        "uncertainty": "Paired intervals use common component multiplicities and condition on fixed fitted predictors. Three descriptive contrasts and two seeds on development-exposed data do not establish significance or public transfer. Endpoint is fixed E4; no best epoch, seed or arm promotion.",
    }


def main() -> None:
    """Write one immutable full-panel readout from receiver receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "run-root", "data-root", "split-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve(strict=True)
    output = args.output or args.run_root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    report = collect(native_grader(args.runtime_root), args)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, output)


if __name__ == "__main__":
    main()
