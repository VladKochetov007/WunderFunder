"""Exercise the independent review's failure cases against actual producer metadata."""

from __future__ import annotations

import copy
import json

from grade_local import CACHE, OUT, RUNS, validate_identity


def main() -> None:
    """Accept the real control and reject incorrectly sourced or rebound artifacts."""
    run = "q14s_local_kd90_s3"
    metadata = json.loads((CACHE / f"pred_{run}_valid.json").read_text())
    provenance = json.loads((RUNS / run / "provenance.json").read_text())
    validate_identity(run, metadata, provenance)
    rejected = []
    for case in ("wrong_source", "wrong_manifest", "valid_fitted", "false_checkpoint_hash"):
        side, lineage = copy.deepcopy(metadata), copy.deepcopy(provenance)
        if case == "wrong_source":
            side["split"] = "train"
        elif case == "wrong_manifest":
            side["campaign_split_manifest_sha256"] = "0" * 64
        elif case == "valid_fitted":
            lineage["fit_row_groups"]["valid"] = [0]
        else:
            side["checkpoint_sha256"] = "0" * 64
            lineage["checkpoint_sha256"]["model_ep3.pt"] = "0" * 64
        try:
            validate_identity(run, side, lineage)
        except AssertionError:
            rejected.append(case)
        else:
            raise AssertionError(f"invalid producer fixture accepted: {case}")
    result = {"actual_control_accepted": True, "rejected_cases": rejected}
    (OUT / "IDENTITY_CHECK_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
