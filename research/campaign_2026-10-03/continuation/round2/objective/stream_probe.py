from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from objective_math import bootstrap_target, next_pair_change

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()


def since_from_inputs(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Reconstruct trainer's capped float32 expm1(log1p(since_i0px))."""
    event = np.r_[False, (bid[1:] != bid[:-1]) & (ask[1:] != ask[:-1])]
    index = np.arange(len(bid))
    last = np.maximum.accumulate(np.where(event, index, 0))
    return np.expm1(np.log1p(np.minimum(index - last, 10000).astype(np.float32)))


def main() -> None:
    """Stream selected TRAIN groups once and retain only chunk sufficient statistics."""
    start = time.monotonic()
    protocol = json.loads((HERE / "PREREG.json").read_text())
    groups = protocol["sample"]["row_groups_in_order"]
    run = ROOT / "artifacts/runs" / protocol["recipe_run"]
    provenance = json.loads((run / "provenance.json").read_text())
    args = provenance["args"]
    if args["distill_alpha"] != 0.5 or args["distill_boot_h"] != 8 or not args["prop_cache"]:
        raise ValueError("active recipe differs from preregistration")
    metadata = np.load(ROOT / "cache/train_targets.npz")
    row_groups = metadata["row_group"]
    positions = {int(group): i for i, group in enumerate(row_groups)}
    metadata.close()
    cache_names = [
        "propcache_c20260929_prop_f0_train.npy",
        "pred_c20260929_since_s3_f0_train.npy",
        "pred_c20260929_since_s4_f0_train.npy",
    ]
    caches = [np.load(ROOT / "cache" / name, mmap_mode="r") for name in cache_names]
    hashes = {name: hashlib.sha256() for name in cache_names}
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    n = len(groups)
    chunks = 20
    masses = np.zeros((n, chunks, 4), float)  # main W, row R, next mask R, label-only W
    gradient = np.zeros(
        (n, chunks, 2, 2, 3), float
    )  # zero/teacher, target, intercept/since/prop head coordinates
    inputs = np.zeros((n, chunks, 8), float)
    labels = np.zeros((n, chunks, 6), float)
    bins = np.zeros(
        (n, chunks, 16, 2, 4), float
    )  # mass, label moment, combo moment, nonboot moment
    seq_ids = []
    data_hash = hashlib.sha256()
    for sample, group in enumerate(groups):
        columns = ["seq_ix", "t0", "t1", "i0_p0", "i0_p11", "i1_p0", "i1_p11", "a6"]
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        data = {name: table.column(name).to_numpy() for name in columns}
        seq_ids.append(int(data["seq_ix"][0]))
        y = np.column_stack([data["t0"], data["t1"]]).astype(np.float32)
        p = np.array(caches[0][group], np.float32)
        raw_teachers = [np.array(cache[positions[group]], np.float32) for cache in caches[1:]]
        for name, value in zip(cache_names, [p, *raw_teachers], strict=True):
            hashes[name].update(value.tobytes())
        for name in columns:
            data_hash.update(data[name].tobytes())
        teacher = (raw_teachers[0] + raw_teachers[1]) / 2
        clipped = np.clip(y, -2, 2)
        teacher_clipped = np.clip(teacher, -2, 2)
        since = since_from_inputs(data["i0_p0"], data["i0_p11"])
        valid = np.arange(len(y)) >= 99
        sw = np.maximum(0.25, 1 / (1 + since / 32)).astype(np.float32)
        v = valid * (p**2 + 0.03) * sw
        w = v[:, None] * np.abs(clipped)
        target = bootstrap_target(y, teacher, 8, 0.5)
        nonboot = 0.5 * clipped + 0.5 * teacher_clipped
        d = y[:, 0] - y[:, 1]
        next_d = next_pair_change(np.stack([d, d], axis=1))
        has_future = next_d < len(y)
        event0 = np.r_[
            False,
            (data["i0_p0"][1:] != data["i0_p0"][:-1]) & (data["i0_p11"][1:] != data["i0_p11"][:-1]),
        ]
        event1 = np.r_[
            False,
            (data["i1_p0"][1:] != data["i1_p0"][:-1]) | (data["i1_p11"][1:] != data["i1_p11"][:-1]),
        ]
        feature = np.column_stack([np.ones(len(y)), np.log1p(since) / 10, p]).astype(float)
        bin_id = np.searchsorted([0.25, 0.5, 0.75], p) * 4 + np.searchsorted([16, 64, 256], since)
        for chunk in range(chunks):
            sl = slice(chunk * 1000, (chunk + 1) * 1000)
            vv, ww, yy, tt = v[sl], w[sl], clipped[sl], target[sl]
            masses[sample, chunk] = [
                ww.sum(dtype=float),
                vv.sum(dtype=float),
                (vv * has_future[sl]).sum(dtype=float),
                (np.abs(yy) * valid[sl, None]).sum(dtype=float),
            ]
            for arm, prediction in enumerate((np.zeros_like(tt), teacher_clipped[sl])):
                gradient[sample, chunk, arm] = 2 * np.einsum(
                    "tk,tf->kf", ww * (prediction - tt), feature[sl]
                )
            inputs[sample, chunk] = [
                p[sl].mean(),
                (p[sl] ** 2 + 0.03).mean(),
                sw[sl].mean(),
                event0[sl].mean(),
                event1[sl].mean(),
                np.log1p(since[sl]).mean(),
                (data["i0_p11"][sl] - data["i0_p0"][sl]).mean(),
                np.abs(data["a6"][sl]).mean(),
            ]
            labels[sample, chunk] = [
                np.abs(yy).mean(),
                np.abs(yy).std(),
                yy[:, 0].mean(),
                yy[:, 1].mean(),
                np.mean(np.any(y[sl][1:] != y[sl][:-1], axis=1)),
                np.mean(np.abs(y[sl]) > 2),
            ]
            for target_index in range(2):
                for moment, value in enumerate(
                    (
                        ww[:, target_index],
                        ww[:, target_index] * yy[:, target_index],
                        ww[:, target_index] * tt[:, target_index],
                        ww[:, target_index] * nonboot[sl, target_index],
                    )
                ):
                    bins[sample, chunk, :, target_index, moment] = np.bincount(
                        bin_id[sl], weights=value, minlength=16
                    )
        if sample % 32 == 31:
            np.savez(
                HERE / "stream_partial.npz",
                completed=sample + 1,
                masses=masses[: sample + 1],
                gradient=gradient[: sample + 1],
            )
            print(f"completed={sample + 1}/{n} elapsed={time.monotonic() - start:.1f}s", flush=True)
    np.savez(
        HERE / "chunk_summaries.npz",
        groups=groups,
        seq_ids=seq_ids,
        masses=masses,
        gradient=gradient,
        inputs=inputs,
        labels=labels,
        bins=bins,
    )
    artifact = {
        "selected_sequence_ids": seq_ids,
        "selected_train_row_groups": groups,
        "cached_selected_slice_sha256": {
            name: digest.hexdigest() for name, digest in hashes.items()
        },
        "selected_parquet_columns_sha256": data_hash.hexdigest(),
        "large_cache_hash_policy": "only selected ordered slices independently hashed; complete prediction hashes taken from recipe provenance, not re-read",
        "recipe_args": args,
        "teacher_provenance": args["distill_provenance"],
        "propensity_sidecar": json.loads(
            (ROOT / "cache/propcache_c20260929_prop_f0_train.json").read_text()
        ),
        "wall_seconds": time.monotonic() - start,
        "read_columns": columns,
        "approximations": [
            "TRAIN subset omits validation-in-fit true-mask weighting",
            "float32 NumPy operations followed by float64 sums; not GPU reduction bit parity",
            "cached teacher prediction as surrogate student; no real GRU parameter/EMA auxiliary gradients",
        ],
    }
    (HERE / "sample_artifacts.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"complete": n, "wall_seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
