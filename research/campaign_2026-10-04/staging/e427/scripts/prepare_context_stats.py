"""Stream E427's fallback mean and preregistered endpoint windows from fit-only inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
import sys

sys.path.insert(0, str(ROOT / "src"))

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, SEQUENCE_LENGTH, SequenceFile  # noqa: E402
from wnn.research_features import ENDPOINT, research_features_numpy  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402


def _ordered(groups: list[int], split: str) -> list[int]:
    return sorted(groups, key=lambda group: hashlib.sha256(
        f"E427-window-v1:{split}:{group}".encode()
    ).hexdigest())


def prepare(parent_provenance_path: Path, output_path: Path) -> dict:
    parent = json.loads(parent_provenance_path.read_text())
    fit = parent.get("fit_row_groups")
    if not isinstance(fit, dict) or set(fit) != {"train", "valid"}:
        raise ValueError("parent provenance does not expose exact TRAIN and fit-VALID membership")
    groups = {split: list(map(int, fit[split])) for split in ("train", "valid")}
    if not groups["train"] or not groups["valid"]:
        raise ValueError("E427 needs the complete parent TRAIN plus fit-VALID population")
    if set(groups["train"]) & CORRUPT_TRAIN_ROW_GROUPS:
        raise ValueError("corrupt TRAIN row group 528 is forbidden in E427 fit context")

    total = np.zeros(28, dtype=np.float64)
    row_count = 0
    samples: list[dict] = []
    input_hashes: dict[str, str] = {}
    for split in ("train", "valid"):
        path = ROOT / "datasets" / f"{split}.parquet"
        input_hashes[f"datasets/{split}.parquet"] = file_sha256(path)
        reader = SequenceFile(path)
        if max(groups[split]) >= len(reader) or min(groups[split]) < 0:
            raise ValueError(f"{split} fit row group is outside its Parquet file")
        sample_groups = set(_ordered(groups[split], split)[:32])
        for group in groups[split]:
            seq = reader.read(group, targets=False, mask=False)
            feature = research_features_numpy(seq.features, "volume28")
            total += feature.sum(axis=0, dtype=np.float64)
            row_count += feature.shape[0]
            if group in sample_groups:
                endpoints = np.flatnonzero(seq.features[:, 110] == ENDPOINT)[:4]
                windows = []
                for endpoint in endpoints.tolist():
                    start, stop = max(0, endpoint - 8), min(SEQUENCE_LENGTH, endpoint + 9)
                    windows.append({
                        "endpoint_row": int(endpoint),
                        "window_rows": [start, stop],
                        "raw_volume_22": seq.features[start:stop, 22:44].tolist(),
                    })
                samples.append({
                    "split": split,
                    "row_group": group,
                    "seq_ix": int(seq.seq_ix),
                    "sample_key_sha256": hashlib.sha256(
                        f"E427-window-v1:{split}:{group}".encode()
                    ).hexdigest(),
                    "first_four_endpoint_windows": windows,
                })
    if row_count != (len(groups["train"]) + len(groups["valid"])) * SEQUENCE_LENGTH:
        raise RuntimeError("E427 input-only context scan did not cover each fit row exactly once")
    result = {
        "format": "wnn.e427-volume28-fit-stats",
        "version": 1,
        "parent_checkpoint": "c20261004_combo_s3_final_r1@ep3",
        "parent_provenance_sha256": file_sha256(parent_provenance_path),
        "fit_row_groups": groups,
        "fit_membership": {
            split: hashlib.sha256(",".join(map(str, sorted(value))).encode()).hexdigest()
            for split, value in groups.items()
        },
        "input_sha256": input_hashes,
        "row_count": row_count,
        "mean": (total / row_count).astype(np.float32).tolist(),
        "statistics": "unweighted mean over every raw input row in exact TRAIN+fit-VALID membership; no labels read",
        "endpoint_sample_rule": "SHA256(E427-window-v1:<split>:<row_group>), first 32 per split, first four endpoints, raw volume rows [endpoint-8, endpoint+8] clipped",
        "endpoint_samples": samples,
        "scoring_rows_used": False,
        "corrupt_train_row_group_528": "rejected if present",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-provenance", default="artifacts/runs/c20261004_combo_s3_final_r1/provenance.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(Path(args.parent_provenance), Path(args.output))
    print(json.dumps({"row_count": result["row_count"], "input_sha256": result["input_sha256"],
                      "endpoint_sample_count": len(result["endpoint_samples"])}, sort_keys=True))


if __name__ == "__main__":
    main()
