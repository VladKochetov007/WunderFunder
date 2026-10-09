"""Build a metadata-only inventory of runs, planned mechanisms and reviewed scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Historical provenance has several intentionally preserved heterogeneous schemas.
Record = dict[str, Any]
ARG_FIELDS = (
    "arch",
    "cell",
    "hidden",
    "layers",
    "enc",
    "readout_activation",
    "dropout",
    "optim",
    "muon_scope",
    "loss",
    "huber_delta",
    "loss_norm",
    "sched",
    "lr",
    "wd",
    "beta2",
    "warmup_frac",
    "wsd_decay",
    "wsd_final",
    "clip",
    "grad_stats",
    "batch",
    "chunk",
    "epochs",
    "precision",
    "seed",
    "valid_fold",
    "train_source",
    "train_seqs",
    "valid_seqs",
    "valid_train_fraction",
    "epoch_samples",
    "raw_only",
    "research_features",
    "thin_prob",
    "thin_mode",
    "batch_order",
    "init",
    "freeze",
    "distill",
    "distill_alpha",
    "distill_family_weight",
    "distill_cache_variant",
    "distill_cache_root",
    "aux_next",
    "aux_struct",
    "ema_teacher",
    "ema_decay",
    "checkpoint_policy",
    "checkpoint_epoch",
    "save_training_state",
    "resume_state",
    "fit_subset_manifest",
    "fit_groups_manifest",
    "split_manifest",
)


def read_json(path: Path) -> Any:
    """Read preserved historical JSON without rewriting its schema."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Hash small metadata only; never read checkpoint or prediction tensors."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_cell(value: Any) -> str:
    """Preserve lists and dictionaries in CSV cells without dropping information."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_csv(path: Path, rows: list[Record]) -> None:
    """Write the union of recorded dimensions in a filterable flat table."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: text_cell(row.get(k)) for k in fields} for row in rows)


def collect_runs(root: Path, errors: list[Record]) -> list[Record]:
    """Inventory root Challenge2 provenance and artifact existence, not validity."""
    predictions: dict[str, list[Record]] = defaultdict(list)
    for path in sorted((root / "cache").glob("pred_*.json")):
        try:
            item = read_json(path)
            if isinstance(item, dict) and item.get("run"):
                array = path.with_suffix(".npy")
                predictions[str(item["run"])].append(
                    {
                        "sidecar": str(path.relative_to(root)),
                        "array": str(array.relative_to(root)),
                        "present": array.is_file(),
                        "split": item.get("split"),
                        "epoch": item.get("epoch"),
                        "declared_sha256": item.get("prediction_sha256"),
                    }
                )
        except (OSError, ValueError, TypeError) as error:
            errors.append({"path": str(path), "error": str(error)})
    rows = []
    for path in sorted((root / "artifacts/runs").glob("*/provenance.json")):
        try:
            provenance = read_json(path)
            args = provenance.get("args", {})
            directory = path.parent
            checkpoints = sorted(directory.glob("model*.pt"))
            history_path = directory / "history.json"
            history = read_json(history_path) if history_path.is_file() else []
            history = history if isinstance(history, list) else []
            last = history[-1] if history else {}
            fitting = provenance.get("fit_row_groups", {})
            row = {
                "run_id": directory.name,
                "inventory_status": "checkpoint_present" if checkpoints else "metadata_only",
                **{field: args.get(field) for field in ARG_FIELDS},
                "fit_train_groups": len(fitting.get("train", [])) if fitting else None,
                "fit_valid_groups": len(fitting.get("valid", [])) if fitting else None,
                "split_manifest_sha256": provenance.get("split_manifest_sha256"),
                "recorded_epochs": len(history),
                "recorded_fit_seconds": sum(float(h.get("train_seconds", 0)) for h in history),
                "last_update_diagnostics": last.get("update_diagnostics", {}).get("cumulative"),
                "checkpoint_paths": [str(p.relative_to(root)) for p in checkpoints],
                "checkpoint_bytes": sum(p.stat().st_size for p in checkpoints),
                "prediction_sidecars": predictions.get(directory.name, []),
                "prediction_arrays_present": sum(
                    p["present"] for p in predictions.get(directory.name, [])
                ),
                "provenance_path": str(path.relative_to(root)),
                "provenance_sha256": sha256(path),
                "history_path": str(history_path.relative_to(root))
                if history_path.is_file()
                else None,
                "evidence_limit": "Metadata inventory; checkpoint presence is not a completed experiment, ancestry approval, or comparable score.",
            }
            rows.append(row)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return rows


def write_browser(path: Path, datasets: dict[str, list[Record]]) -> None:
    """Write a standalone searchable table; no network or external assets."""
    data = json.dumps(datasets).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>Research experiment matrix</title>
<style>body{font:14px system-ui;margin:24px;color:#182433}input,select{padding:8px;margin:6px}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #dde2e7;padding:6px;vertical-align:top;max-width:420px;overflow-wrap:anywhere}th{position:sticky;top:0;background:#edf3f8}#wrap{overflow:auto;max-height:80vh}#count{margin:10px}</style>
<h1>Research experiment matrix</h1><p>Metadata inventory and planned mechanisms. Blank means unrecorded. Only the reviewed-results tab contains comparable frozen endpoint scores. Exposed folds remain exposed.</p>
<label>Table <select id="set"></select></label><label>Search every dimension <input id="query" size="50" placeholder="muon, huber, fold1, onecycle…"></label><label>Columns <select id="columns"><option value="compact">Compact</option><option value="all">All dimensions</option></select></label><div id="count"></div><div id="wrap"><table id="table"></table></div>
<script>const datasets=DATA;const set=document.getElementById('set');for(const k of Object.keys(datasets)){const o=document.createElement('option');o.textContent=k;o.value=k;set.append(o)}
const compact=['run_id','experiment_id','hypothesis','priority','status','inventory_status','mechanism','arch','hidden','optim','loss','sched','seed','valid_fold','train_sample','teacher_policy','serving_path','wp','delta_vs_anchor','prediction_arrays_present'];
function render(){const all=datasets[set.value];const q=document.getElementById('query').value.toLowerCase();const rows=all.filter(r=>JSON.stringify(r).toLowerCase().includes(q));let keys=[...new Set(all.flatMap(r=>Object.keys(r)))];if(document.getElementById('columns').value==='compact')keys=keys.filter(k=>compact.includes(k));const table=document.getElementById('table');table.replaceChildren();let tr=document.createElement('tr');for(const k of keys){const th=document.createElement('th');th.textContent=k;tr.append(th)}table.append(tr);for(const row of rows){tr=document.createElement('tr');for(const k of keys){const td=document.createElement('td');const v=row[k];td.textContent=v==null?'':typeof v==='object'?JSON.stringify(v):String(v);tr.append(td)}table.append(tr)}document.getElementById('count').textContent=rows.length+' / '+all.length+' rows; '+keys.length+' columns'}for(const id of ['set','query','columns'])document.getElementById(id).addEventListener('input',render);render();</script>
"""
    path.write_text(page.replace("DATA", data))


def main() -> None:
    """Rebuild the research matrix using small existing metadata only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-root", type=Path, help="Separate inventory source from isolated output checkout."
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = (
        args.output_root or root
    ).resolve() / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/matrix"
    out.mkdir(parents=True, exist_ok=True)
    errors: list[Record] = []
    runs = collect_runs(root, errors)
    planned = read_json(out / "planned_experiments.json")["experiments"]
    reviewed = read_json(out / "reviewed_endpoints.json")["endpoints"]
    cloud_path = out / "verified_cloud_fits.json"
    cloud = read_json(cloud_path)["fits"] if cloud_path.is_file() else []
    datasets = {
        "Recorded runs": runs,
        "Verified cloud fits": cloud,
        "Planned experiments": planned,
        "Reviewed results": reviewed,
    }
    for name, rows in zip(
        ["runs", "cloud_fits", "planned", "reviewed"], datasets.values(), strict=True
    ):
        write_csv(out / f"{name}.csv", rows)
    write_browser(out / "experiment_matrix.html", datasets)
    snapshot = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_rows": len(runs),
        "verified_cloud_fit_rows": len(cloud),
        "plan_rows": len(planned),
        "reviewed_rows": len(reviewed),
        "optimizer_counts": dict(Counter(text_cell(r.get("optim")) for r in runs)),
        "loss_counts": dict(Counter(text_cell(r.get("loss")) for r in runs)),
        "scheduler_counts": dict(Counter(text_cell(r.get("sched")) for r in runs)),
        "errors": errors,
        "tensor_data_read": False,
        "generator_sha256": sha256(Path(__file__)),
        "input_sha256": {
            name: sha256(out / name)
            for name in (
                "planned_experiments.json",
                "reviewed_endpoints.json",
                "verified_cloud_fits.json",
            )
            if (out / name).is_file()
        },
        "inventory_source": str(root),
    }
    (out / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    lines = [
        "# Experiment matrix",
        "",
        f"Recorded runs: **{len(runs)}**. Verified cloud fits: **{len(cloud)}**. Planned mechanism rows: **{len(planned)}**. Reviewed endpoint rows: **{len(reviewed)}**.",
        "",
        "Open [the searchable table](experiment_matrix.html), or use [all run dimensions](runs.csv), [verified cloud fits](cloud_fits.csv), [planned experiments](planned.csv), and [reviewed results](reviewed.csv).",
        "",
        "Blank cells mean unrecorded. Metadata presence does not establish successful training or valid held-out performance. Historical unreviewed scores are deliberately not ranked together. The full hypothesis/experiment ledgers remain authoritative.",
        "",
        "| Priority | Experiment | Optimizer | Loss | Scheduler | Architecture / samples | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in planned:
        values = [
            row.get(k, "")
            for k in [
                "priority",
                "experiment_id",
                "optim",
                "loss",
                "sched",
                "train_sample",
                "status",
            ]
        ]
        lines.append(
            "| " + " | ".join(html.escape(str(v)).replace("|", "/") for v in values) + " |"
        )
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
