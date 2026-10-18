"""Extract scored F0 rows and a complete-sequence serving panel without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PINS = {
    "fold": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
    "wide": "21af6180c3290a99f450d701daca03581762ea18cfc70ffcc9453929435ee8f0",
    "deep": "50ea5e088b60da8b31ee1796771c4c13f2940131293a51793d63ceadf324be3d",
}


def sha256(path: Path) -> str:
    """Hash a file with bounded reads."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(base: Path, output: Path) -> None:
    """Bind archived predictions to original source labels and sequence identities."""
    output.mkdir(exist_ok=False)
    snapshot = base / "result_snapshots/rr18_r01_f0_s1001_e6"
    paths = {"fold": snapshot / "evidence/fold0.json"}
    paths.update({arm: snapshot / f"predictions/{arm}.npy" for arm in ("wide", "deep")})
    for key, path in paths.items():
        if sha256(path) != PINS[key]:
            raise ValueError(f"pinned {key} hash mismatch")
    fold = json.loads(paths["fold"].read_text())
    groups = fold["evaluation"]["valid"]
    identities = {row["row_group"]: row for row in fold["sequences"]["valid"]}
    predictions = {arm: np.load(paths[arm], mmap_mode="r") for arm in ("wide", "deep")}
    for arm, prediction in predictions.items():
        if prediction.shape != (468, 20000, 2) or prediction.dtype != np.float32:
            raise ValueError(f"invalid {arm} prediction shape/dtype")
        sidecar = json.loads((snapshot / f"predictions/{arm}.json").read_text())
        # Full export-sidecar hashes and ordering were checked by archived grade.
        (output / f"{arm}_sidecar.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    parquet_path = base / "data/datasets/valid.parquet"
    parquet = pq.ParquetFile(parquet_path)
    parquet_stat = parquet_path.stat()
    if parquet_stat.st_size != 5615509128 or parquet.num_row_groups != 1873:
        raise ValueError("source parquet structural identity changed")
    columns = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]
    pieces: dict[str, list[np.ndarray]] = {
        key: [] for key in ("y", "wide", "deep", "row_group", "component")
    }
    with pq.ParquetWriter(
        output / "panel.parquet", parquet.schema_arrow, compression="zstd"
    ) as panel:
        for index, group in enumerate(groups):
            table = parquet.read_row_group(
                group, columns=None if index < 8 else columns, use_threads=False
            )
            steps = table["step_in_seq"].to_numpy()
            seq = table["seq_ix"].to_numpy()
            if not np.array_equal(steps, np.arange(20000)) or not np.all(
                seq == identities[group]["seq_ix"]
            ):
                raise ValueError(f"invalid sequence identity/order: {group}")
            mask = table["is_scored"].to_numpy().astype(bool) & table[
                "need_prediction"
            ].to_numpy().astype(bool)
            if np.any(mask[:99]):
                raise ValueError("scored warmup row")
            y = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
                np.float32
            )
            pieces["y"].append(y[mask])
            for arm in ("wide", "deep"):
                values = predictions[arm][index][mask]
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite scored prediction")
                pieces[arm].append(values)
            for key, value in (("row_group", group), ("component", identities[group]["component"])):
                pieces[key].append(np.full(int(mask.sum()), value, dtype=np.int64))
            if index < 8:
                panel.write_table(table, row_group_size=20000)
    arrays = {key: np.concatenate(parts) for key, parts in pieces.items()}
    if arrays["y"].shape != (808863, 2) or not np.isfinite(arrays["y"]).all():
        raise ValueError("unexpected scored rows or nonfinite targets")
    arrays.update(
        {
            "wide_prediction_sha256": np.asarray(PINS["wide"]),
            "deep_prediction_sha256": np.asarray(PINS["deep"]),
            "split_manifest_sha256": np.asarray(PINS["fold"]),
            "grade_sha256": np.asarray(
                "65f448989243aa575cac2e5701c55bf8ee6e874e6b01f39840bb4418dbb9fdd0"
            ),
        }
    )
    np.savez(output / "scored.npz", **arrays)
    for arm in ("wide", "deep"):
        np.save(output / f"panel_{arm}.npy", predictions[arm][:8])
    (output / "fold0.json").write_bytes(paths["fold"].read_bytes())
    receipt = {
        "status": "extracted",
        "rows": len(arrays["y"]),
        "groups": len(groups),
        "components": len(np.unique(arrays["component"])),
        "panel_groups": groups[:8],
        "source_pins": PINS,
        "source_parquet": str(parquet_path),
        "parquet_size": parquet_stat.st_size,
        "parquet_mtime_ns": parquet_stat.st_mtime_ns,
        "parquet_full_hash": "inherited from F0 grade; size/footer/group identities rechecked here",
        "target_source": "original parquet t0/t1; scored AND need_prediction",
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output.iterdir()
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "EXTRACTION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def main() -> None:
    """Run a create-only, bounded evidence extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.base, args.output)


if __name__ == "__main__":
    main()
