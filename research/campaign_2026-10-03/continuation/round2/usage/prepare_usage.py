from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
from inspect_control import digest
from usage_core import I0, I1, REPO, ROOT, update_state

from wnn.splits import load_campaign_split


def main() -> None:
    """Verify component exclusions and sample six components using metadata alone."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    update_state("preparing_frozen_metadata_sample")
    control = json.loads((ROOT / "control_identity.json").read_text())
    args = control["args"]
    if any(args.get(flag) for flag in ["raw_only", "online_norm", "heuristics", "prop_input"]):
        raise ValueError("Instrument requires the inspected raw+event preprocessing")
    if args.get("arch") != "gru" or args.get("enc") != "linear" or control["readout_present"]:
        raise ValueError("Unexpected control architecture or readout")
    split = load_campaign_split(REPO / control["split_manifest"], REPO)
    trained = json.loads((REPO / "artifacts/runs/c20261001_d256_s3_f0/provenance.json").read_text())
    split.assert_fit_groups(trained["fit_row_groups"]["train"], trained["fit_row_groups"]["valid"])
    with np.load(REPO / "cache/valid_targets.npz", allow_pickle=False) as cache:
        groups, seq_ids = cache["row_group"], cache["seq_ix"]
    sidecar = json.loads((REPO / "cache/pred_c20261001_d256_s3_f0_valid.json").read_text())
    if sidecar["row_groups"] != groups.tolist():
        raise ValueError("Prediction and target cache ordering differ")
    lookup = {
        int(group): (position, int(seq_ids[position])) for position, group in enumerate(groups)
    }
    components = {
        group: component for source, group, component in split.components if source == "valid"
    }
    ranked = sorted(
        split.evaluation_groups,
        key=lambda group: hashlib.sha256(f"round2-usage-v1:{lookup[group][1]}".encode()).digest(),
    )
    samples, selected_components = [], set()
    own = json.loads(
        (REPO / "research/campaign_2026-10-03/eda/statphys/continuation_samples.json").read_text()
    )
    own_prior = set(own["possibly_exposed_row_groups"]["valid"])
    own_recent = {item["row_group"] for item in own["samples"]["valid"]}
    for group in ranked:
        component = components[group]
        if component in selected_components:
            continue
        position, seq_id = lookup[group]
        samples.append(
            {
                "row_group": group,
                "cache_position": position,
                "seq_ix": seq_id,
                "component": component,
                "own_prior_target_census_exposure": True,
                "own_reconstructed_prior_feature_exposure": group in own_prior,
                "own_cpu_continuation_feature_exposure": group in own_recent,
                "campaign_validation_exposure": True,
            }
        )
        selected_components.add(component)
        if len(samples) == 6:
            break
    if len(samples) != 6 or len(I0) != 125 or len(I1) != 78:
        raise AssertionError("Incorrect sample or stream partition")
    if set(I0) & set(I1) or set(I0) | set(I1) | set(range(104, 112)) != set(range(211)):
        raise AssertionError("Input partition overlap/missing channel")
    prediction_path = REPO / "cache/pred_c20261001_d256_s3_f0_valid.npy"
    prediction_hash = digest(prediction_path)
    if prediction_hash != sidecar["prediction_sha256"]:
        raise ValueError("Cached prediction hash mismatch")
    result = {
        "samples": samples,
        "checkpoint_sha256": control["checkpoint_sha256"],
        "split_manifest_sha256": split.sha256,
        "prediction_sha256": prediction_hash,
        "evaluation_components": len({components[group] for group in split.evaluation_groups}),
        "fit_groups_verified": {
            key: len(value) for key, value in trained["fit_row_groups"].items()
        },
        "component_exclusion_verified": True,
        "selection_used_targets_masks_features": False,
        "cache_order_verified": True,
    }
    (ROOT / "samples.json").write_text(json.dumps(result, indent=2))
    logging.info(json.dumps(result, indent=2))
    update_state(
        "metadata_sample_frozen_ready_synthetic",
        {"samples": samples, "pending": ["synthetic controls", "CPU identity/replay", "report"]},
    )


if __name__ == "__main__":
    main()
