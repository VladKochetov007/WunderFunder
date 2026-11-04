from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

# Native producer/report JSON is heterogeneous; it is bounded and checked here.
JsonObject = dict[str, Any]
REPO = "VladKochetov007/WunderFunder"
SOURCE_COMMIT = "27741738f6a5c2a87d63161e80bf61bdc5d58f77"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
INITIAL_INPUT_SHA256 = "970ba293f262951900e3bc4c519ae1e242d037627544455e8d82a67bcad47cfe"
LEGACY = {
    "w18": ("tl3c_w18final_distilled_s5", 1),
    "tiedlead_s5": ("tl3c_fttall_tlead_s5", 3),
    "tiedlead_s6": ("tl3c_fttall_tlead_s6", 3),
    "fbc": ("tl3c_fbc1k_s5", 3),
    "fin256_s3": ("tl3c_fin256_s3", 3),
    "fin256_s4": ("tl3c_fin256_s4", 3),
}
NEW = {
    "lstm": "het6_lstm_s1201",
    "tcn": "het6_tcn_s1202",
    "control_s1301": "het6_control_s1301",
    "control_s1302": "het6_control_s1302",
    "hetero_s1301": "het6_heterogeneous_s1301",
    "hetero_s1302": "het6_heterogeneous_s1302",
}
ENSEMBLES = (
    "TL3",
    "PR51_replacement",
    "GRU_family",
    "heterogeneous_teacher_pool",
    "TL3_plus_GALAM",
    "TL3_plus_LSTM",
    "TL3_plus_TCN",
    "TL3_heterogeneous",
    "GRU_student_mean",
    "heterogeneous_student_mean",
    "TL3_plus_GRU_students",
    "TL3_plus_heterogeneous_students",
)


def read_json(path: Path, limit: int = 2_000_000) -> JsonObject:
    """Read bounded JSON metadata without loading model tensors or prediction banks."""
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f"metadata read bound exceeded: {path.name}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    return value


def sha256(path: Path) -> str:
    """Hash one evidence/code file; tensor/prediction hashing belongs to the grader."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Persist this report-only watcher's state after each irreversible transition."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stats(paths: list[Path]) -> list[list[object]]:
    return [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]


def ready_snapshot(root: Path, legacy: Path, galam: Path, split: Path) -> JsonObject:
    """Use small final-pointer metadata and stat signatures; never infer tensor quality."""
    ready, pending = {}, {}
    roles = [(name, legacy, run, epoch) for name, (run, epoch) in LEGACY.items()]
    roles += [(name, root, run, 3) for name, run in NEW.items()]
    for alias, base, run, epoch in roles:
        directory, bank = base / "artifacts/runs" / run, base / "cache" / f"pred_{run}_valid.npy"
        paths = [
            bank,
            bank.with_suffix(".json"),
            directory / "model.pt",
            directory / "provenance.json",
        ]
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            pending[alias] = "final checkpoint/valid export unavailable"
            continue
        try:
            side, provenance = read_json(paths[1]), read_json(paths[3])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pending[alias] = "producer metadata write incomplete"
            continue
        args = provenance.get("args", {})
        if (
            provenance.get("selected_epoch") != epoch
            or args.get("checkpoint_policy") != "fixed"
            or args.get("checkpoint_epoch") != epoch
            or side.get("epoch") != epoch
            or side.get("run") != run
            or side.get("split") != "valid"
            or side.get("checkpoint_file") != "model.pt"
            or side.get("inference_precision") != "fp32"
            or side.get("split_manifest_sha256") != SPLIT_SHA256
            or re.fullmatch("[0-9a-f]{64}", str(side.get("prediction_sha256", ""))) is None
        ):
            pending[alias] = "fixed final producer pointer not ready"
            continue
        ready[alias] = _stats(paths)
    paths = [
        galam / "g2-galam-lstm128/receipt.json",
        galam / "g2-galam-lstm128/scored-predictions.npy",
        galam / "plan.json",
        galam / "result.json",
    ]
    if all(path.is_file() for path in paths):
        try:
            endpoint = read_json(paths[0]).get("endpoint", {})
            native = [Path(str(endpoint.get(name, ""))) for name in ("checkpoint", "epoch_index")]
            if (
                endpoint.get("cell") == "g2-galam-lstm128"
                and endpoint.get("epoch") == 3
                and all(path.is_file() for path in native)
            ):
                ready["galam"] = _stats(paths + native)
            else:
                pending["galam"] = "fixed native human-epoch3 probe unavailable"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pending["galam"] = "native probe metadata write incomplete"
    else:
        pending["galam"] = "original scored-only probe unavailable"
    common = _stats([split, root / "datasets/valid.parquet", root / "cache/valid_index.npz"])
    return {"ready": ready, "pending": pending, "common": common}


def signature(snapshot: JsonObject, aliases: set[str] | None = None) -> str:
    """Ignore changing non-final training pointers when deciding whether to regrade."""
    ready = snapshot["ready"]
    payload = {
        "common": snapshot["common"],
        "ready": {
            name: ready[name] for name in sorted(ready) if aliases is None or name in aliases
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_report(
    report: JsonObject, weights: dict[str, object] | None = None
) -> dict[str, object]:
    """Require the frozen full-valid twelve-row table before exposing any score."""
    if (
        report.get("schema") != "wnn.heterogeneous-ensemble-score.v1"
        or report.get("baseline") != "TL3"
        or report.get("split_sha256") != SPLIT_SHA256
        or report.get("valid_sequences") != 1873
        or report.get("selected_rows") != 3528273
    ):
        raise ValueError("report is not the admitted complete official-valid protocol")
    rows = report.get("rows")
    if not isinstance(rows, list) or [row.get("name") for row in rows] != list(ENSEMBLES):
        raise ValueError("report changed the frozen twelve-row comparison")
    actual_weights = {row["name"]: row["weights"] for row in rows}
    if actual_weights["TL3"] != {"w18": 0.34, "tiedlead_s5": 0.33, "tiedlead_s6": 0.33} or (
        weights is not None and actual_weights != weights
    ):
        raise ValueError("report changed canonical TL3 or the fixed mixture weights")
    for row in rows:
        score = row.get("score")
        if row.get("status") == "complete":
            if not isinstance(score, dict) or any(
                type(score.get(key)) not in (float, int) or not math.isfinite(score[key])
                for key in ("t0", "t1", "wp")
            ):
                raise ValueError("completed table row lacks finite metrics")
        elif score is not None or row.get("delta_to_tl3") is not None:
            raise ValueError("unavailable table row must not contain a fabricated score")
    baseline = rows[0]["score"]
    if report.get("baseline_available") != (baseline is not None):
        raise ValueError("report baseline availability disagrees with the measured TL3 row")
    for row in rows:
        if baseline is not None and row["score"] is not None:
            delta = row.get("delta_to_tl3")
            if type(delta) not in (float, int) or not math.isclose(
                delta, row["score"]["wp"] - baseline["wp"], rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("report delta does not match the actual TL3 baseline")
    return actual_weights


def report_ready(report: JsonObject) -> set[str]:
    """Read producer membership from the already-pinned report manifest."""
    return {
        name
        for name, record in report["input_manifest"]["members"].items()
        if record.get("state") != "pending"
    }


def comment_body(
    report: JsonObject,
    report_path: Path,
    snapshot: JsonObject,
    controller: JsonObject,
    event: str,
    hourly: bool,
) -> str:
    """Render actual results and explicit pending rows, without a public/serving claim."""
    completed = sum(row["status"] == "complete" for row in report["rows"])
    heading = "Hourly progress; last measured table" if hourly else "Updated ensemble computation"
    lines = [
        f"<!-- tabdata-heterogeneous52:{event} -->",
        f"### {heading}: {completed}/12 comparisons",
        "",
        "**Computed; pending independent review.** These are exposed full-official-valid results, with fixed coefficients; no public score, serving readiness or significance claim.",
        "",
        "| Ensemble | t0 | t1 | WP | Δ vs TL3 | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        score = row["score"]
        numbers = (
            ["—"] * 4
            if score is None
            else [f"{score[key]:.9f}" for key in ("t0", "t1", "wp")]
            + ["—" if row["delta_to_tl3"] is None else f"{row['delta_to_tl3']:+.9f}"]
        )
        status = row["status"] + (
            ": " + ", ".join(row["missing_members"]) if row["missing_members"] else ""
        )
        lines.append("| " + " | ".join([row["name"], *numbers, status]) + " |")
    new_ready = len(set(snapshot["ready"]) & NEW.keys())
    verified = len(controller.get("verified_fits", {}))
    lines += [
        "",
        f"Progress: {new_ready}/6 final new-model prediction banks ready; {verified}/6 fits verified in the controller snapshot. Ready endpoints total {len(snapshot['ready'])}/13. Controller status: `{controller.get('status', 'unavailable')}`; retained phase errors: {len(controller.get('errors', {}))}.",
        "",
        "Metric: `global_wp_clip2_v1`, all1,873 valid sequences /3,528,273 scored rows. Train-only10325; no valid fitting ancestry. Existing validation exposure remains.",
        f"[Frozen source/configuration]({f'https://github.com/{REPO}/commit/{SOURCE_COMMIT}'}); unchanged twelve mixtures. Report timestamp `{report['created_utc']}`.",
        f"Input SHA256 `{report['input_manifest_sha256']}`; report SHA256 `{sha256(report_path)}`.",
        f"Retained node report: `{report_path}`. Producer memberships are checked; legacy consumed teacher-cache bytes retain their documented ancestry limitation.",
    ]
    return "\n".join(lines) + "\n"
