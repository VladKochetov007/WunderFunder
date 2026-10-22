"""Grade the fixed normalized pair on the already exposed eight-group sample."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent
SAVED = Path(
    "/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021"
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/campaign_2026-09-30/audit")]
from campaign_score import mmap_stored_npz  # noqa: E402

from wnn.gmetric import GlobalAccumulator  # noqa: E402


def digest(path: Path) -> str:
    """Hash inputs with bounded memory before opening target values."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    """Evaluate only the declared 0.5/0.5 candidate and original fixed controls."""
    output = PACKET / "FIXED_SAMPLE_GRADE_R2.json"
    assert not output.exists()
    check = json.loads(
        (ROOT / "submissions/tl3_two_parents_20261022/TECHNICAL_CHECK_R3.json").read_text()
    )
    archive = ROOT / "submissions/tl3_two_parents_20261022/tl3_two_parents_20261022.zip"
    assert digest(archive) == check["zip_sha256"]
    assert check["status"] == "TECHNICAL_PASS_EXPLORATORY_NOT_PROMOTED"
    assert check["max_abs_prior_pair_need_rows"] <= 1e-5
    predictions = SAVED / "member_predictions.npz"
    assert digest(predictions) == "3c254fd21f0c41d8ee45aeb0fdbbd38a7d3c996554304053db733887fd80cd98"
    selection = json.loads((SAVED / "selection_plan.json").read_text())
    members = selection["members"]
    assert [m["row_group"] for m in members] == [g["row_group"] for g in check["groups"]]
    assert (
        selection["selected_membership_sha256"]
        == "eacc150ce8f30bf79e749678bdd1a79d9f61c77218aaa8175fefb8224a51d184"
    )
    with np.load(predictions, allow_pickle=False) as tracks:
        raw = tracks["retained_pair"].copy()
        original = (
            0.34 * tracks["original_student"].astype(np.float64) + raw.astype(np.float64)
        ).astype(np.float32)
    normalized = raw / np.float32(0.66)
    target_path = ROOT / "cache/valid_targets.npz"
    assert digest(target_path) == "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
    targets = mmap_stored_npz(target_path)
    assert targets["y"].shape == (1873, 20000, 2)
    assert set(map(int, targets["row_group"])) == set(range(1873))
    positions = {int(group): index for index, group in enumerate(targets["row_group"])}
    accumulators = {
        name: GlobalAccumulator() for name in ("original", "raw_pair", "normalized_pair")
    }
    for index, member in enumerate(members):
        group = positions[member["row_group"]]
        mask = targets["mask"][group].astype(bool).copy()
        mask[:99] = False
        for name, values in (
            ("original", original),
            ("raw_pair", raw),
            ("normalized_pair", normalized),
        ):
            accumulators[name].add(targets["y"][group], values[index], mask)
    scores = {name: accumulator.result() for name, accumulator in accumulators.items()}
    assert abs(scores["raw_pair"]["wp"] - 0.7803073782867451) < 1e-12
    result = {
        "status": "EXPOSED_SAMPLE_DIAGNOSTIC_NOT_PROMOTION",
        "zip_sha256": check["zip_sha256"],
        "scores": scores,
        "paired_delta_vs_original": scores["normalized_pair"]["wp"] - scores["original"]["wp"],
        "normalization_effect": scores["normalized_pair"]["wp"] - scores["raw_pair"]["wp"],
        "limits": "Eight adaptively exposed components, same previously fitted/unknown-ancestry data; saved outputs validated to actual ZIP within2.384e-7. No full-fold result, independent replication, uncertainty estimate or public forecast.",
        "new_fits": 0,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
