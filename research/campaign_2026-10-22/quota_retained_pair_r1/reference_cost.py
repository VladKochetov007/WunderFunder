"""Measure the unchanged TL3 reference in its own native-module process."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "research/campaign_2026-10-18/online_features_r1/tl3_replay")]
from native_runtime import extract_safe, load_callback, run_sequence, sha256_file  # noqa: E402

from utils import FEATURE_COLUMNS, DataPoint, validate_sequence  # noqa: E402


def main() -> None:
    """Return four matched complete-sequence callback times from the frozen reference."""
    original = ROOT / "submissions/team_tl3_original_20261018.zip"
    assert (
        sha256_file(original) == "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
    )
    selection = json.loads(
        Path(
            "/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021/selection_plan.json"
        ).read_text()
    )
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    timings = []
    with tempfile.TemporaryDirectory(
        prefix="quota-reference-", dir="/home/vlad/development/wunder/research_tmp"
    ) as temporary:
        package = Path(temporary)
        with zipfile.ZipFile(original) as archive:
            extract_safe(archive, package)
        _, model, _ = load_callback(package)
        assert model.member_count == 3
        for _ in range(2):
            for member in selection["members"][:2]:
                table = parquet.read_row_group(
                    member["row_group"],
                    columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS],
                    use_threads=False,
                )
                sequence, need = validate_sequence(table)
                assert sequence == member["seq_ix"]
                features = np.column_stack(
                    [table[name].to_numpy() for name in FEATURE_COLUMNS]
                ).astype(np.float32, copy=False)
                timings.append(run_sequence(model, DataPoint, sequence, features, need)[3])
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
