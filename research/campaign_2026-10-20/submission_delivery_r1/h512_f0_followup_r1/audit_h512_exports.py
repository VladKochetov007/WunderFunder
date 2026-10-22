"""Bind the two frozen H512 exports to accepted checkpoints and F0 membership."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nll_adaptation"))
from prediction_artifact_validation import GROUPS, _json, _pin, _prediction, _sha  # noqa: E402

RUN = "r01_h512_f0_s1003_local_cp5_r1"
SWA = "r01_h512_f0_s1003_local_swa345_r1"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PACKET = Path("research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1")


def audit(root: Path) -> dict[str, object]:
    """Check actual output bytes before the frozen scorer accesses targets."""
    packet = root / PACKET
    terminal = _json(packet / "FIT_TERMINAL_ACCEPTANCE_R1.json")
    if terminal.get("status") != "FIT_COMPLETE_ACCEPTED" or terminal.get("run_name") != RUN:
        raise ValueError("no accepted source fit")
    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    _pin(split_path, SPLIT_SHA)
    split = _json(split_path)
    plan = _json(packet / "H512_POSTFIT_DELIVERY_PLAN_R1.json")
    records: list[dict[str, object]] = []
    for item in plan["prediction_exports"]["existing_baselines"]:
        path = root / item["prediction"]
        records.append(_pin(path, item["prediction_sha256"], 299_680_128))
    source_provenance = root / "artifacts/runs" / RUN / "provenance.json"
    _pin(source_provenance, terminal["provenance_sha256"])
    expected_args = _json(source_provenance)["args"]
    for run in (RUN, SWA):
        directory = root / "artifacts/runs" / run
        checkpoint = directory / "model_ep5.pt"
        provenance_path = directory / "provenance.json"
        provenance = _json(provenance_path)
        checkpoint_sha = _sha(checkpoint)
        if run == RUN:
            expected_sha = terminal["checkpoint_sha256"]["model_ep5.pt"]
        else:
            derivation = _json(directory / "weight_derivation_receipt.json")
            if (
                derivation.get("method") != "equal_parameter_mean_epochs_3_4_5"
                or derivation.get("source_run_name") != RUN
                or derivation.get("destination_run_name") != SWA
            ):
                raise ValueError("derived endpoint is outside the frozen trajectory mean")
            expected_sha = derivation["outputs"]["model_ep5.pt"]["sha256"]
        if (
            checkpoint_sha != expected_sha
            or provenance.get("checkpoint_sha256", {}).get("model_ep5.pt") != expected_sha
            or provenance.get("args") != expected_args
            or provenance.get("split_manifest_sha256") != SPLIT_SHA
        ):
            raise ValueError(f"checkpoint/provenance binding differs: {run}")
        for source in ("train", "valid"):
            groups = provenance.get("fit_row_groups", {}).get(source, [])
            expected_groups = split["fit"][source]
            if len(groups) != len(set(groups)) or set(groups) != set(expected_groups):
                raise ValueError(f"fitting membership differs: {run}/{source}")
        groups = provenance.get("evaluation_row_groups", [])
        if len(groups) != len(set(groups)) or set(groups) != set(split["evaluation"]["valid"]):
            raise ValueError(f"evaluation membership differs: {run}")
        prediction = root / "cache" / f"pred_{run}_valid.npy"
        sidecar_path = prediction.with_suffix(".json")
        sidecar = _json(sidecar_path)
        prediction_sha = _prediction(prediction, scan=True)
        expected_sidecar = {
            "run": run,
            "split": "valid",
            "epoch": 5,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": prediction_sha,
            "split_manifest_sha256": SPLIT_SHA,
            "campaign_split_manifest_sha256": SPLIT_SHA,
            "inference_precision": "fp32",
            "feature_only": True,
            "campaign_export_scope": "full-valid-feature-only",
            "row_groups": GROUPS,
        }
        if any(sidecar.get(key) != value for key, value in expected_sidecar.items()):
            raise ValueError(f"export sidecar differs: {run}")
        records.append(
            {
                "run": run,
                "checkpoint_sha256": checkpoint_sha,
                "provenance_sha256": _sha(provenance_path),
                "prediction_sha256": prediction_sha,
                "sidecar_sha256": _sha(sidecar_path),
                "all_predictions_finite": True,
            }
        )
    result = {"status": "PASS", "records": records, "auditor_sha256": _sha(Path(__file__))}
    with (packet / "EXPORT_AUDIT_R1.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    """Audit the create-only exports before official evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(strict=True)), indent=2))


if __name__ == "__main__":
    main()
