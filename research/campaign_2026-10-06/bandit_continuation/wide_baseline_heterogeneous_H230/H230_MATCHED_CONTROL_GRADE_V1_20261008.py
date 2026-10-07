from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "H230_MATCHED_CONTROL_GRADE_V1_20261008.json"


def main() -> None:
    """Grade the sealed family-on/off pair using the original frozen grader."""
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / "src"))
    grader_path = HERE / "H230_SEED3_INDEPENDENT_GRADE_V1.py"
    spec = importlib.util.spec_from_file_location("h230_frozen_grader", grader_path)
    if spec is None or spec.loader is None:
        raise ImportError(grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    targets_path = ROOT / "cache/valid_targets.npz"
    hashes = {
        "grader": grader.sha256_file(grader_path),
        "metric": grader.require_hash(
            ROOT / "src/wnn/gmetric.py", grader.EXPECTED["metric"], "metric"
        ),
        "split": grader.require_hash(split_path, grader.EXPECTED["split"], "split"),
        "targets": grader.require_hash(targets_path, grader.EXPECTED["targets"], "targets"),
        "grade_freeze": grader.require_hash(
            grader.GRADE_FREEZE, grader.EXPECTED["grade_freeze"], "grade freeze"
        ),
    }
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, ROOT)
    groups = np.asarray(sorted(split.evaluation_groups), dtype=np.int64)
    if len(groups) != 468:
        raise ValueError("frozen evaluation membership differs")
    predictions = []
    run_names = ["bd26_h230_hetero_s3_f0", "bd26_h230_familyoff_s3_retry1_f0"]
    receipt_names = [
        "H230_SEED3_EXPORT_TERMINAL_RECEIPT_V1_SOURCEV2_20261008.json",
        "H230_SEED3_CONTROL_RETRY1_EXPORT_TERMINAL_RECEIPT_V1_SOURCEV2_20261008.json",
    ]
    receipt_hashes = [
        "bffaac7584fa337dc7cd0a3fb4cb61f7fca900006218fc236cb4a7d6db4616d9",
        "0570797a6e856adf97679334dc8a6e429699448f47a42d9a06f816755e115dba",
    ]
    ancestry = []
    for run_name, receipt_name, expected in zip(
        run_names, receipt_names, receipt_hashes, strict=True
    ):
        receipt_path = HERE / receipt_name
        hashes[receipt_name] = grader.require_hash(receipt_path, expected, receipt_name)
        receipt = grader.read_json(receipt_path)
        if receipt["attempt"]["exit_status"] != 0:
            raise ValueError("export was unsuccessful")
        binding = receipt["prediction"]
        prediction, sidecar, pred_hash, sidecar_hash = grader.open_prediction(
            binding["path"], binding["sha256"], run_name, hashes["split"]
        )
        if sidecar_hash != binding["sidecar_sha256"] or sidecar["epoch"] != 1:
            raise ValueError("sidecar differs from the sealed export")
        run_path = ROOT / "artifacts/runs" / run_name
        provenance = grader.read_json(run_path / "provenance.json")
        split.assert_fit_groups(
            provenance["fit_row_groups"]["train"], provenance["fit_row_groups"]["valid"]
        )
        if set(provenance["evaluation_row_groups"]) != set(groups):
            raise ValueError("evaluated groups differ from the frozen split")
        if provenance["args"]["distill_family_weight"] != (0.1 if len(predictions) == 0 else 0.0):
            raise ValueError("family treatment differs")
        if sidecar["checkpoint_sha256"] != grader.sha256_file(run_path / "model_ep1.pt"):
            raise ValueError("prediction checkpoint differs")
        hashes[run_name + "_prediction"] = pred_hash
        hashes[run_name + "_provenance"] = grader.sha256_file(run_path / "provenance.json")
        ancestry.append(
            (provenance["initial_checkpoint_sha256"], provenance["args"]["distill_cache_root"])
        )
        predictions.append(prediction[groups])
    if ancestry[0] != ancestry[1]:
        raise ValueError("initializer/cache ancestry differs")
    with np.load(targets_path, allow_pickle=False) as cache:
        if cache["row_group"].tolist() != list(range(1873)):
            raise ValueError("target row order differs")
        targets = cache["y"][groups]
        scored = cache["mask"][groups]
    mask = scored & (np.arange(20_000)[None, :] >= grader.WARMUP)
    lookup = {(part, group): component for part, group, component in split.components}
    components = np.asarray([lookup[("valid", int(group))] for group in groups])
    result = grader.compare(
        "matched_family_off_seed3", targets, predictions[0], predictions[1], mask, components
    )
    payload = {
        "owner": "root-independent-grader",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "completed_exposed_matched_comparison",
        "evaluation_groups": groups.tolist(),
        "components": int(len(np.unique(components))),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hashes": hashes,
        "result": result,
        "wall_seconds": time.monotonic() - started,
        "memory": grader.execution_memory(),
        "interpretation_scope": "Same exposed fold, paired seed and fixed epoch; component bootstrap conditions on these fitted models. No public forecast or exact teammate replication. Contaminated all-data incumbent is excluded.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(OUTPUT), "result": result, "wall_seconds": payload["wall_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
