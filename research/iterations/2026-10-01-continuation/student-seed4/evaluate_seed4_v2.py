"""Fixed five-recipe, four-contrast I/O adapter; metric functions are the frozen fold_eval source.

No execution is authorized by this file or its draft binding manifest.
Only main() reads data. The generated source test imports helpers only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CONT = HERE.parent
OLD = ROOT / "research/iterations/2026-10-01-overnight"
MEMBERS = ("cont_sh256_s3h2", "cont_sh256_s4h2", "on_ctld256", "on_ctl128s3", "on_ctl128s4", "on_ctlc500")
RECIPES = {
    "double_shallow": (MEMBERS[0], MEMBERS[1], MEMBERS[3], MEMBERS[5]),
    "current_student4": (MEMBERS[0], MEMBERS[3], MEMBERS[4], MEMBERS[5]),
    "strong4": (MEMBERS[2], MEMBERS[3], MEMBERS[4], MEMBERS[5]),
    "single_seed4": (MEMBERS[1], MEMBERS[3], MEMBERS[4], MEMBERS[5]),
    "duplicate_seed3": (MEMBERS[0], MEMBERS[0], MEMBERS[3], MEMBERS[5]),
}
CONTRASTS = {
    "double_vs_current": ("double_shallow", "current_student4"),
    "double_vs_strong": ("double_shallow", "strong4"),
    "single_seed4_vs_current": ("single_seed4", "current_student4"),
    "double_vs_duplicate_seed3": ("double_shallow", "duplicate_seed3"),
}
METRIC_SHA256 = "d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric_module():
    path = ROOT / "scripts/fold_eval.py"
    assert sha(path) == METRIC_SHA256
    spec = importlib.util.spec_from_file_location("frozen_student_seed4_fold_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blend(parts):
    # Literal frozen load_fold accumulation order/dtype, one sequence at a time.
    return sum(parts) / len(parts)


def comparison(fe, candidate, reference, fold_ids):
    result = {
        "candidate_wp": fe.pooled(candidate.sum(0)),
        "reference_wp": fe.pooled(reference.sum(0)),
        "by_fold": {},
    }
    result["delta"] = result["candidate_wp"] - result["reference_wp"]
    for fold in (0, 1):
        selected = fold_ids == fold
        c = fe.pooled(candidate[selected].sum(0))
        r = fe.pooled(reference[selected].sum(0))
        result["by_fold"][str(fold)] = {"candidate_wp": c, "reference_wp": r, "delta": c-r}
    return result


def repository_path(relative):
    # Registered read-only inputs can traverse ROOT/cache's shared-data symlink.
    # Bind their bytes separately; reject lexical traversal rather than resolving
    # them into the creative checkout. This never grants an output destination.
    assert isinstance(relative, str) and relative
    lexical = Path(relative)
    assert not lexical.is_absolute() and ".." not in lexical.parts
    assert str(lexical) == relative, "Input name must be normalized repository-relative"
    return ROOT / lexical


def contained_output(path):
    resolved = Path(path).resolve()
    assert resolved.is_relative_to(HERE.resolve()) and resolved != HERE.resolve(), "Output escaped student-seed4"
    return resolved


def pinned(path, expected, tracked):
    assert isinstance(expected, str) and len(expected) == 64, "Incomplete immutable binding"
    actual = sha(path)
    assert actual == expected, f"Hash mismatch: {path}"
    tracked[str(path)] = expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bindings", type=Path, required=True)
    ap.add_argument("--bindings-sha256", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    assert sha(args.bindings) == args.bindings_sha256
    bindings = json.loads(args.bindings.read_text())
    assert bindings["status"] == "ROOT_ALLOCATED_FOR_FIXED_FOUR_COMPARISONS"
    assert bindings["folds"] == [0, 1]
    assert bindings["recipes"] == {k: list(v) for k, v in RECIPES.items()}
    assert bindings["contrasts"] == {k: list(v) for k, v in CONTRASTS.items()}
    assert bindings["coefficients"] == {k: [.25] * 4 for k in RECIPES}
    output = contained_output(args.output_dir)
    assert not output.exists(), "Immutable attempt output already exists"
    # Freeze source/input identities and the root's scoring allocation. The
    # controller checks completed fits; no extra review-success artifact is needed.
    tracked = {}
    assert bindings["sources"][str(Path(__file__).relative_to(ROOT))] == sha(__file__)
    for name, expected in bindings["sources"].items():
        pinned(repository_path(name), expected, tracked)
    assert bindings["data"] == json.loads((OLD / "grader/frozen-data.json").read_text())
    for name, record in bindings["data"].items():
        path = repository_path(name)
        assert path.stat().st_size == record["bytes"]
        pinned(path, record["sha256"], tracked)
    record = bindings["root_allocation"]
    pinned(repository_path(record["path"]), record["sha256"], tracked)
    receipt = json.loads(repository_path(record["path"]).read_text())
    # Field/value are explicit in frozen bindings; no inference from exit0.
    assert receipt[record["required_field"]] == record["required_value"]

    # Prospective single-event replacement; the original allocation remains intact.
    amendment = bindings["prospective_amendment"]
    assert amendment == {"path": "research/iterations/2026-10-01-continuation/student-seed4/amendment-02.json", "sha256": "45d895efaf2c55e053c04037adc8033f1c0d21c4f3ad0fdd32b5747e8c08929b"}
    pinned(repository_path(amendment["path"]), amendment["sha256"], tracked)
    authority = json.loads(repository_path(amendment["path"]).read_text())
    assert authority["id"] == "STUDENT-SEED4-PROSPECTIVE-DUPLICATE-CONTROL-02"
    assert authority["authorized_by_root"] and not authority["original_score_event_executed"]
    assert authority["quality_readout_events_total"] == 1
    assert authority["total_profiles"] == 5 and authority["total_fixed_contrasts"] == 4

    fe = metric_module()
    tg = np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False)
    groups = tg["row_group"]
    assert len(groups) == 1873 and len(np.unique(groups)) == len(groups)
    # One decompression of the immutable target/mask arrays; predictions remain mmap.
    y, mask = tg["y"], tg["mask"]
    assert y.shape == (1873, 20000, 2) and mask.shape == (1873, 20000)
    cf = np.load(ROOT / "cache/chain_folds.npz", allow_pickle=False)
    valid = cf["split"] == "valid"
    info = {int(g): (int(f), int(c), bool(p)) for g, f, c, p in
            zip(cf["row_group"][valid], cf["fold"][valid], cf["chain"][valid], cf["pure"][valid], strict=True)}
    assert len(info) == len(groups) and set(info) == set(groups.tolist())
    train_chains = set(cf["chain"][cf["split"] == "train"].tolist())
    for chain in np.unique(cf["chain"][valid]):
        sel = valid & (cf["chain"] == chain)
        assert len(np.unique(cf["fold"][sel])) == 1
        if np.any(cf["pure"][sel]):
            assert int(chain) not in train_chains
    order_sha = hashlib.sha256(groups.tobytes()).hexdigest()
    arrays = {}
    for fold in (0, 1):
        expected_positions = [i for i, g in enumerate(groups) if info[int(g)][0] == fold]
        for name in MEMBERS:
            base = CONT if name in MEMBERS[:2] else OLD
            path = base / "cache" / f"pred_{name}_f{fold}_valid.npy"
            receipt_path = path.with_suffix(".inputs.json")
            record = bindings["predictions"][f"{name}_f{fold}"]
            assert record["path"] == str(path.relative_to(ROOT))
            assert record["receipt"] == str(receipt_path.relative_to(ROOT))
            pinned(path, record["sha256"], tracked)
            pinned(receipt_path, record["receipt_sha256"], tracked)
            meta = json.loads(receipt_path.read_text())
            assert meta["run"] == f"{name}_f{fold}"
            assert meta["prediction_sha256"] == record["sha256"]
            assert meta["source_row_group_order_sha256"] == order_sha
            assert meta["chain_folds_sha256"] == bindings["data"]["cache/chain_folds.npz"]["sha256"]
            assert meta["fold"] == fold and meta["epoch"] == 1
            assert meta["rows"] == expected_positions
            assert meta["row_groups"] == groups[expected_positions].tolist()
            assert meta["checkpoint_sha256"] == record["checkpoint_sha256"]
            assert Path(meta["checkpoint"]).name == "model_ep1.pt"
            assert Path(meta["checkpoint"]).parent.name == f"{name}_f{fold}"
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            assert arr.shape == (1873, 20000, 2) and arr.dtype == np.float32
            arrays[(name, fold)] = arr

    output.mkdir()
    started = {"utc": datetime.now(timezone.utc).isoformat(), "bindings_sha256": args.bindings_sha256,
               "source_sha256": sha(__file__), "metric_sha256": METRIC_SHA256,
               "data_and_prediction_identities": tracked, "recipes": bindings["recipes"]}
    (output / "started.json").write_text(json.dumps(started, indent=2) + "\n")
    stats = {name: [] for name in RECIPES}
    positions, folds, chains = [], [], []
    for fold in (0, 1):
        rows = [i for i, g in enumerate(groups) if info[int(g)][0] == fold and info[int(g)][2]]
        assert rows
        for row in rows:
            yi, mi = y[row:row+1], mask[row:row+1].copy()
            mi[:, :fe.WARMUP] = False
            assert np.isfinite(yi).all()
            current = {name: arrays[(name, fold)][row:row+1].astype(np.float32) for name in MEMBERS}
            assert all(np.isfinite(value).all() for value in current.values())
            for recipe, members in RECIPES.items():
                stats[recipe].append(fe.sums(yi, blend([current[name] for name in members]), mi)[0])
            positions.append(row)
            folds.append(fold)
            chains.append(info[int(groups[row])][1])
    stats = {name: np.asarray(value) for name, value in stats.items()}
    fold_ids = np.asarray(folds)
    result = {
        "comparisons": {name: comparison(fe, stats[c], stats[r], fold_ids)
                        for name, (c, r) in CONTRASTS.items()},
        "fixed_comparisons": 4, "contrasts": bindings["contrasts"],
        "sequences": len(positions), "pure_only": True, "folds": [0, 1],
        "recipes": bindings["recipes"], "provenance": started,
        "inference": "four fixed descriptive comparisons; no blend search or automatic promotion",
        "exposure": "previously exposed folds; no independent confirmation or hidden-test guarantee",
        "uncertainty": "point estimates here; the separately frozen bootstrap_seed4_v2.py specifies conditional paired-chain intervals, not training-seed or selection uncertainty",
    }
    # A changed input invalidates the attempt; no quiet acceptance of replaced files.
    for path, expected in tracked.items():
        assert sha(path) == expected, f"Input changed during evaluation: {path}"
    assert sha(args.bindings) == args.bindings_sha256
    result["pre_post_hash_checks"] = "PASS"
    np.savez_compressed(output / "sufficient.npz", **stats, positions=np.asarray(positions),
                        row_groups=groups[positions], fold_id=fold_ids, chains=np.asarray(chains))
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "sequences": len(positions)}))


if __name__ == "__main__":
    main()
