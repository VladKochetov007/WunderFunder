"""Guarded entry point for the unchanged, hash-frozen pooled chain evaluator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/iterations/2026-10-01-overnight"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def record(value, start=False):
    path = CAMPAIGN / "evaluation-ledger.jsonl"
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        if start:
            assert sum(row["event"] == "started" for row in rows) < 64
            assert not any(row["id"] == value["id"] for row in rows), "Use a new immutable attempt ID"
        stream.seek(0, 2)
        stream.write(json.dumps(value) + "\n")
        stream.flush()


def target_scores(stats):
    result = []
    for w, sy, sp, syy, spp, syp in np.asarray(stats):
        if w < 1e-8:
            result.append(0.0)
            continue
        my, mp = sy / w, sp / w
        vy, vp = syy / w - my * my, spp / w - mp * mp
        result.append(0.0 if vy <= 1e-16 or vp <= 1e-16 else
                      float(np.clip((syp / w - my * mp) / np.sqrt(vy * vp), -1, 1)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--folds", default="0,1")
    args = ap.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.id)
    assert args.folds in ("0,1", "2,3", "0,1,2,3")
    assert args.plan.is_file()
    sources = json.loads((Path(__file__).parent / "frozen-sources.json").read_text())
    for path, expected in sources.items():
        assert digest(ROOT / path) == expected, f"Frozen grader dependency changed: {path}"
    spec = importlib.util.spec_from_file_location("frozen_fold_eval", ROOT / "scripts/fold_eval.py")
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    predictions = {}
    allowed_old = {"ft256dH", "ft128H", "ft128s4H", "ftc500H"}

    def guarded_load(run, fold, rows):
        parts = []
        for name in run.split("+"):
            assert re.fullmatch(r"[A-Za-z0-9_]+", name)
            if name.startswith("on_"):
                path = CAMPAIGN / "cache" / f"pred_{name}_f{fold}_valid.npy"
            else:
                assert name in allowed_old, "Unregistered historical prediction namespace"
                path = ROOT / "cache" / f"pred_{name}_f{fold}_valid.npy"
            arr = np.load(path, mmap_mode="r")
            assert arr.shape == (1873, 20000, 2), (path, arr.shape)
            assert arr.dtype == np.float32, (path, arr.dtype)
            selected = arr[rows].astype(np.float32)
            assert np.isfinite(selected).all(), f"Nonfinite/unpredicted selected rows: {path}"
            if str(path) not in predictions:
                predictions[str(path)] = {"sha256": digest(path), "shape": list(arr.shape)}
            parts.append(selected)
        # Identical accumulation dtype/order to the frozen source load_fold.
        return sum(parts) / len(parts)

    fe.load_fold = guarded_load
    sufficient = []
    source_sums = fe.sums

    def saved_sums(y, pred, mask):
        assert np.isfinite(y).all() and np.isfinite(pred).all()
        result = source_sums(y, pred, mask)
        sufficient.append(result)
        return result

    fe.sums = saved_sums
    out_dir = CAMPAIGN / "evaluations"
    out_dir.mkdir(exist_ok=True)
    output = out_dir / (args.id + ".json")
    assert not output.exists()
    receipt = {"id": args.id, "utc": stamp(), "event": "started",
               "candidate": args.candidate, "reference": args.reference,
               "plan": str(args.plan), "plan_sha256": digest(args.plan),
               "grader_sources": sources, "wrapper_sha256": digest(__file__)}
    record(receipt, start=True)
    try:
        sys.argv = ["fold_eval", args.candidate, args.reference, "--folds", args.folds,
                    "--boot", "1000", "--out", str(output)]
        fe.main()
        result = json.loads(output.read_text())
        candidate = np.concatenate(sufficient[0::2])
        reference = np.concatenate(sufficient[1::2])
        result["candidate_wp_by_target"] = target_scores(candidate.sum(0))
        result["reference_wp_by_target"] = target_scores(reference.sum(0))
        assert abs(np.mean(result["candidate_wp_by_target"]) - result["candidate_wp"]) < 1e-12
        result["provenance"] = receipt
        result["predictions"] = predictions
        result["exposure"] = "exposed exploratory folds; no fresh confirmation"
        output.write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(out_dir / (args.id + "-sufficient.npz"),
                            candidate=candidate, reference=reference)
        record({"id": args.id, "event": "completed", "utc": stamp(),
                "result_sha256": digest(output), "delta": result["delta"]})
    except BaseException as exc:
        record({"id": args.id, "event": "failed", "utc": stamp(),
                "error": type(exc).__name__ + ": " + str(exc)})
        raise


if __name__ == "__main__":
    main()
