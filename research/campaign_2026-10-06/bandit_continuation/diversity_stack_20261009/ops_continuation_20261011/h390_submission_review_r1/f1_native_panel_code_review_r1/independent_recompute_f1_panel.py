"""Independently stream official scores for the frozen native F1 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CATALOG_SHA = "c2a0dd18526f98e23bce64937336125fc12be6680c8640815879d8f64d5c6105"
FEATURE_MANIFEST_SHA = "733c7368506b42e7893f37dc07e5841e1c4d152faa03d7a0c5405454a85ba109"
SPLIT_SHA = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
FIXTURE_SHA = "e44fd415caf13d60a09f572c12bd7e94292b6d56234ab53a8880f43e155f83a2"
GPU_MEMBERS = {
    "s3": {
        "run": "vt_f1_muonall_h384_e3",
        "file_stem": "pred_vt_f1_muonall_h384_e3",
        "checkpoint_sha256": "9c7d5e55fde7112dc691c80670fb17b62f24b9d759d2c90312a9ec89334843c6",
        "prediction_sha256_recorded": "53a67d8e7af4f8d860c7be4d0bc697c55955c69c551df57e37b19fb9769b37e8",
    },
    "s4": {
        "run": "vt_f1_muonall_h384_e3_s4",
        "file_stem": "pred_vt_f1_muonall_h384_e3_s4",
        "checkpoint_sha256": "752f462036a8e340db00de98dcb141795f3a0ae54cfffce3e95ce7ceed7f5e9e",
        "prediction_sha256_recorded": "e4f4ebb3559c0933b0d8f263f8ea6954d5ecb8e3d6d4ba26533a43f144e1ee2e",
    },
    "s5": {
        "run": "vt_f1_muonall_h384_e3_s5",
        "file_stem": "pred_vt_f1_muonall_h384_e3_s5",
        "checkpoint_sha256": "ee154f035720069ef26bf0fd328c61887cdbb426c49d51fb25172e5bb52b1f80",
        "prediction_sha256_recorded": "3225864d4f830058366d41ba31f2d72d4b7f56143a0311ed3373f54ab3059b3c",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_gpu_cache(cache: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, object]] = {}
    for seed, expected in GPU_MEMBERS.items():
        npy = cache / f"{expected['file_stem']}_valid.npy"
        sidecar = cache / f"{expected['file_stem']}_valid.json"
        metadata = json.loads(sidecar.read_text())
        if (
            metadata.get("run") != expected["run"]
            or metadata.get("split") != "valid"
            or metadata.get("epoch") != 2
            or metadata.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or metadata.get("prediction_sha256") != expected["prediction_sha256_recorded"]
            or metadata.get("split_manifest_sha256") != SPLIT_SHA
            or metadata.get("inference_precision") != "fp16"
            or metadata.get("row_groups") != list(range(1873))
        ):
            raise ValueError(f"GPU FP16 cache sidecar mismatch for {seed}")
        array = np.load(npy, mmap_mode="r", allow_pickle=False)
        if array.shape != (1873, 20_000, 2) or array.dtype != np.float32:
            raise ValueError(f"GPU FP16 cache shape/dtype mismatch for {seed}")
        arrays[seed] = array
        records[seed] = {
            "run": expected["run"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "prediction_sha256_sidecar_recorded_not_rehashed": expected[
                "prediction_sha256_recorded"
            ],
            "sidecar_sha256": sha256_file(sidecar),
            "npy_path": str(npy),
            "npy_bytes": npy.stat().st_size,
            "precision_label": "GPU torch fp16 export, distinct from Q8 native runtime",
        }
    return arrays, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-grade", type=Path)
    parser.add_argument("--gpu-cache", type=Path)
    args = parser.parse_args()

    for path in (args.catalog, args.feature_manifest, args.split, args.valid_parquet):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.catalog) != CATALOG_SHA:
        raise ValueError("candidate catalog changed from frozen R3 bytes")
    if sha256_file(args.feature_manifest) != FEATURE_MANIFEST_SHA:
        raise ValueError("feature manifest changed from frozen bytes")
    if sha256_file(args.split) != SPLIT_SHA:
        raise ValueError("fold1 split changed from frozen bytes")

    manifest = json.loads(args.feature_manifest.read_text())
    fixture = manifest["fixture"]
    selection = manifest["selection"]
    if fixture["sha256"] != FIXTURE_SHA or selection["seed"] != 572911:
        raise ValueError("feature fixture or selection does not match frozen panel")
    groups = fixture["group_records"]
    if len(groups) != 41 or len(selection["components"]) != 14:
        raise ValueError("unexpected panel size/component count")
    split = json.loads(args.split.read_text())
    eval_rows = set(split["evaluation"]["valid"])

    sys.path.insert(0, str(args.project_root.resolve(strict=True)))
    from wnn.gmetric import METRIC_ID, GlobalAccumulator  # noqa: PLC0415

    if METRIC_ID != "global_wp_clip2_v1":
        raise ValueError(f"unexpected official metric {METRIC_ID}")
    candidates = json.loads(args.catalog.read_text())["candidates"]
    expected_ids = {str(c["candidate_id"]) for c in candidates}
    if len(candidates) != 4 or len(expected_ids) != 4:
        raise ValueError("candidate pool is not the frozen four-candidate panel")

    arrays: dict[str, np.ndarray] = {}
    candidate_hashes: dict[str, dict[str, str]] = {}
    accumulators = {candidate_id: GlobalAccumulator() for candidate_id in expected_ids}
    for c in candidates:
        cid = str(c["candidate_id"])
        package = Path(c["package_path"])
        package_receipt_path = Path(c["package_receipt_path"])
        inference_path = Path(c["inference_receipt_path"])
        pred_path = Path(c["predictions_path"])
        if sha256_file(package) != c["package_sha256"]:
            raise ValueError(f"candidate ZIP drift: {cid}")
        if sha256_file(package_receipt_path) != c["package_receipt_sha256"]:
            raise ValueError(f"candidate build receipt drift: {cid}")
        build = json.loads(package_receipt_path.read_text())
        inference = json.loads(inference_path.read_text())
        if build.get("precision") != "q8vnni":
            raise ValueError(f"non-Q8 candidate in frozen panel: {cid}")
        if (
            inference.get("package", {}).get("sha256") != c["package_sha256"]
            or inference.get("package", {}).get("precision_asserted") != "q8vnni"
            or inference.get("panel", {}).get("sha256") != FIXTURE_SHA
            or inference.get("predictions", {}).get("sha256") != sha256_file(pred_path)
        ):
            raise ValueError(f"inference receipt/source mismatch: {cid}")
        if sha256_file(inference_path) == "":
            raise AssertionError("unreachable empty digest")
        pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        if pred.shape != (41, 20_000, 2) or pred.dtype != np.float32:
            raise ValueError(f"candidate prediction shape/dtype mismatch: {cid}")
        arrays[cid] = pred
        candidate_hashes[cid] = {
            "zip_sha256": str(c["package_sha256"]),
            "build_receipt_sha256": str(c["package_receipt_sha256"]),
            "inference_receipt_sha256": sha256_file(inference_path),
            "prediction_sha256": str(inference["predictions"]["sha256"]),
        }

    parquet = pq.ParquetFile(args.valid_parquet)
    if parquet.metadata.num_row_groups != 1873:
        raise ValueError("valid parquet row-group count differs from fold1 source")
    selected_rows = 0
    group_receipts: list[dict[str, object]] = []
    gpu_arrays: dict[str, np.ndarray] = {}
    gpu_records: dict[str, dict[str, object]] = {}
    gpu_pair = GlobalAccumulator()
    gpu_trio = GlobalAccumulator()
    if args.gpu_cache:
        gpu_arrays, gpu_records = load_gpu_cache(args.gpu_cache)

    for i, record in enumerate(groups):
        rg = int(record["row_group"])
        if rg not in eval_rows:
            raise ValueError(f"row group not in fold1 evaluation set: {rg}")
        row = split["sequences"]["valid"][rg]
        if (row["row_group"], row["seq_ix"], row["component"]) != (
            rg,
            record["sequence_id"],
            record["component"],
        ):
            raise ValueError(f"fold1 row mapping mismatch: {rg}")
        table = parquet.read_row_group(
            rg,
            columns=["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
            use_threads=False,
        )
        if table.num_rows != 20_000:
            raise ValueError(f"incomplete sequence in source group {rg}")
        seq = table["seq_ix"].to_numpy(zero_copy_only=False)
        if not np.all(seq == record["sequence_id"]):
            raise ValueError(f"sequence id mismatch: {rg}")
        targets = np.column_stack(
            [table[c].to_numpy(zero_copy_only=False) for c in ("t0", "t1")]
        ).astype(np.float32, copy=False)
        need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        mask = need & scored
        if sha256_array(need) != record["need_prediction_sha256"]:
            raise ValueError(f"need_prediction mismatch: {rg}")
        if mask[:99].any() or not np.isfinite(targets).all():
            raise ValueError(f"invalid labels or warmup mask: {rg}")
        selected_rows += int(mask.sum())
        pred_slice_hashes: dict[str, str] = {}
        for cid, pred in arrays.items():
            group_pred = np.asarray(pred[i])
            if not np.isfinite(group_pred).all():
                raise ValueError(f"non-finite prediction panel: {cid}/{rg}")
            pred_slice_hashes[cid] = sha256_array(group_pred)
            accumulators[cid].add(targets, group_pred, mask)

        gpu_group_hashes: dict[str, str] = {}
        if gpu_arrays:
            members = {seed: np.asarray(arr[rg]) for seed, arr in gpu_arrays.items()}
            if not all(np.isfinite(x).all() for x in members.values()):
                raise ValueError(f"non-finite selected GPU reference: {rg}")
            gpu_group_hashes = {seed: sha256_array(x) for seed, x in members.items()}
            gpu_pair.add(targets, (members["s4"] + members["s5"]) * 0.5, mask)
            gpu_trio.add(targets, (members["s3"] + members["s4"] + members["s5"]) / 3.0, mask)
        group_receipts.append(
            {
                "row_group": rg,
                "sequence_id": int(record["sequence_id"]),
                "component": int(record["component"]),
                "selected_rows": int(mask.sum()),
                "target_sha256": sha256_array(targets),
                "mask_sha256": sha256_array(mask),
                "candidate_prediction_slice_sha256": pred_slice_hashes,
                "gpu_fp16_member_prediction_slice_sha256": gpu_group_hashes,
            }
        )

    result_scores = {cid: acc.result() for cid, acc in accumulators.items()}
    comparison: dict[str, object] = {"official_grade_comparison": "not_supplied"}
    if args.official_grade:
        official = json.loads(args.official_grade.read_text())
        ref_scores = official.get("candidate_scores", {})
        comparison = {"official_grade_path": str(args.official_grade.resolve()), "matches": {}}
        for cid, independent in result_scores.items():
            if cid not in ref_scores:
                comparison["matches"][cid] = False
                continue
            comparison["matches"][cid] = all(
                abs(float(independent[field]) - float(ref_scores[cid][field])) <= 1e-12
                for field in ("t0", "t1", "wp", "selected_rows")
            )

    gpu_result = None
    if gpu_arrays:
        gpu_result = {
            "scope": "Matched CP GPU torch-fp16 prediction export diagnostic; backend/export difference included, not a pure Q8 quantization delta.",
            "member_receipts": gpu_records,
            "selected_prediction_slices_hashed": True,
            "full_prediction_arrays_rehashed": False,
            "pair_s4_s5_equal": gpu_pair.result(),
            "trio_s3_s4_s5_equal": gpu_trio.result(),
        }

    receipt = {
        "schema": "wnn.independent_native_f1_metric_recompute.r1",
        "verdict": "PASS",
        "metric_id": METRIC_ID,
        "metric_source_sha256": sha256_file(args.project_root / "src/wnn/gmetric.py"),
        "candidate_catalog_sha256": CATALOG_SHA,
        "feature_manifest_sha256": FEATURE_MANIFEST_SHA,
        "feature_fixture_sha256": FIXTURE_SHA,
        "split_sha256": SPLIT_SHA,
        "source_data": {
            "path": str(args.valid_parquet.resolve()),
            "bytes": args.valid_parquet.stat().st_size,
            "mtime_ns": args.valid_parquet.stat().st_mtime_ns,
            "full_valid_parquet_sha256": None,
            "read_columns_only": ["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
        },
        "panel": {
            "groups": 41,
            "whole_components": 14,
            "selected_rows": selected_rows,
            "row_groups": [int(r["row_group"]) for r in groups],
            "sequence_ids": [int(r["sequence_id"]) for r in groups],
            "exposure": "historically exposed fold1 subset; not fresh CV confirmation",
        },
        "candidate_artifacts": candidate_hashes,
        "candidate_scores": result_scores,
        "official_grade_reconciliation": comparison,
        "matched_gpu_fp16_diagnostic": gpu_result,
        "group_receipts": group_receipts,
        "claims_excluded": [
            "no public-score forecast",
            "no fresh held-out/CV claim",
            "GPU torch-fp16 to native-Q8 difference is not a pure quantization delta",
            "no explicit repeated bitwise reset replay from the one-pass native panel",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"scores": result_scores, "selected_rows": selected_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
