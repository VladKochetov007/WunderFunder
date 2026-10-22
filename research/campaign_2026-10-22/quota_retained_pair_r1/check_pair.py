"""Check the actual quota-probe ZIP through complete native callbacks, without labels."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "research/campaign_2026-10-18/online_features_r1/tl3_replay"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(REPLAY)]
from check_submission_abi import inspect_artifact  # noqa: E402
from native_runtime import extract_safe, load_callback, run_sequence, sha256_file  # noqa: E402

from utils import FEATURE_COLUMNS, DataPoint, validate_sequence  # noqa: E402

OUT = Path("/home/vlad/development/wunder/research_tmp/tl3-retained-pair-quota-20261022-r1")
SAVED = Path(
    "/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021"
)


def main() -> None:
    """Bind ABI, finite/reset outputs and prior prediction parity to the exact delivered ZIP."""
    archive = OUT / "submission.zip"
    build = json.loads((OUT / "BUILD_RECEIPT_R1.json").read_text())
    assert sha256_file(archive) == build["zip_sha256"]
    report_path = OUT / "TECHNICAL_CHECK_R3.json"
    assert not report_path.exists()
    abi = inspect_artifact(archive)
    assert abi["passed"]
    assert (
        sha256_file(SAVED / "member_predictions.npz")
        == "3c254fd21f0c41d8ee45aeb0fdbbd38a7d3c996554304053db733887fd80cd98"
    )
    selection = json.loads((SAVED / "selection_plan.json").read_text())
    assert (
        selection["selected_membership_sha256"]
        == "eacc150ce8f30bf79e749678bdd1a79d9f61c77218aaa8175fefb8224a51d184"
    )
    with np.load(SAVED / "member_predictions.npz", allow_pickle=False) as saved:
        expected = saved["retained_pair"].copy() / np.float32(0.66)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    records = []
    retained = []
    with tempfile.TemporaryDirectory(prefix="quota-pair-check-", dir=OUT) as temporary:
        package = Path(temporary) / "pair"
        package.mkdir()
        with zipfile.ZipFile(archive) as stream:
            extract_safe(stream, package)
        _, callback, native_path = load_callback(package)
        assert callback.member_count == 2
        assert (
            sha256_file(native_path) == build["entry_sha256"][str(native_path.relative_to(package))]
        )
        first = None
        maximum = 0.0
        for index, member in enumerate(selection["members"]):
            table = parquet.read_row_group(
                member["row_group"],
                columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS],
                use_threads=False,
            )
            sequence, need = validate_sequence(table)
            assert sequence == member["seq_ix"]
            features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
                np.float32, copy=False
            )
            assert features.shape == (20000, 112)
            values, warmup, predicted, elapsed = run_sequence(
                callback, DataPoint, sequence, features, need
            )
            difference = float(np.max(np.abs(values[need] - expected[index][need])))
            assert difference <= 1e-5, difference
            maximum = max(maximum, difference)
            records.append(
                {
                    "row_group": member["row_group"],
                    "rows": len(values),
                    "warmup": warmup,
                    "predicted": predicted,
                    "callback_ns": elapsed,
                    "max_abs_saved_pair": difference,
                }
            )
            if index == 0:
                first = values.copy()
            if index < 2:
                retained.append((sequence, features.copy(), need.copy()))
        repeated = run_sequence(callback, DataPoint, *retained[0])[0]
        assert np.array_equal(first.view(np.uint32), repeated.view(np.uint32))
        reference_timings = json.loads(
            subprocess.check_output(
                [sys.executable, str(Path(__file__).with_name("reference_cost.py"))], text=True
            )
        )
        timings = {"pair": [], "reference": reference_timings}
        for _ in range(2):
            for item in retained:
                timings["pair"].append(run_sequence(callback, DataPoint, *item)[3])
        ratio = float(np.median(timings["pair"]) / np.median(timings["reference"]))
    assert sha256_file(archive) == build["zip_sha256"]
    report = {
        "status": "TECHNICAL_PASS_EXPLORATORY_NOT_PROMOTED",
        "zip_sha256": build["zip_sha256"],
        "abi": abi,
        "python_version": sys.version,
        "groups": records,
        "reset_aba_bitwise": True,
        "max_abs_prior_pair_need_rows": maximum,
        "same_host_callback_ratio_pair_over_TL3": ratio,
        "timings_ns": timings,
        "target_values_read": False,
        "quality_score_computed": False,
        "evidence_limit": "Eight previously exposed groups; relative local callback timing, not measured competition runtime. Selection was adaptive after prior diagnostic.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "max_abs_pair": maximum,
                "callback_ratio": ratio,
                "zip_sha256": build["zip_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
