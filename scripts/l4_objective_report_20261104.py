"""Collect measured PR62 teacher endpoints, curves and checkpoint archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SOURCE = Path(os.environ.get("WNN_REPORT_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(SOURCE), str(SOURCE / "src")]

from scripts import w18_l4_quality_finalize_20261104 as publisher  # noqa: E402
from wnn.splits import (  # noqa: E402
    file_sha256,
    load_campaign_split,
)
from wnn.training.chronological_teacher import (  # noqa: E402
    validate_chronological_teacher_membership,
)


def read(path: Path) -> Any:
    """Read an existing, bounded JSON artifact."""
    if not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError(f"missing or oversized evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect(panel_path: Path, control_path: Path) -> dict[str, Any]:
    """Join completed real fits to fixed endpoint metrics and chronological ancestry."""
    panel, control = read(panel_path), read(control_path)
    runtime = Path(panel["paths"]["runtime_root"])
    project = Path(panel["paths"]["project_root"])
    split = load_campaign_split(Path(panel["paths"]["split_manifest"]), project)
    if len(split.train_groups) != 10325 or len(split.evaluation_groups) != 1873:
        raise ValueError("official fitting/evaluation membership differs")
    import torch

    runs = []
    for current, path, cells in (
        (panel, panel_path, panel["cells"]),
        (control, control_path, [c for c in control["cells"] if c["role"] == "control"]),
    ):
        for cell in cells:
            run = cell["run_name"]
            root = Path(current["paths"]["project_root"]) / "artifacts/runs" / run
            receipt_path = (
                Path(current["paths"]["runtime_root"])
                / "receipts/fits"
                / (cell["cell_id"] + ".json")
            )
            receipt, history, provenance = (
                read(receipt_path),
                read(root / "history.json"),
                read(root / "provenance.json"),
            )
            owner = receipt.get("owner", {})
            recipe_sha = hashlib.sha256(
                json.dumps(cell["fit_argv"], separators=(",", ":")).encode()
            ).hexdigest()
            if (
                receipt.get("status") != "completed"
                or receipt.get("exit_code") != 0
                or receipt.get("timed_out") is not False
                or owner.get("panel_sha256") != file_sha256(path)
                or owner.get("cell_id") != cell["cell_id"]
                or owner.get("run_name") != run
                or owner.get("recipe_sha256") != recipe_sha
            ):
                raise ValueError(f"fit not complete or not owned by frozen panel: {run}")
            validate_chronological_teacher_membership(provenance, split, run)
            argv = cell["fit_argv"]
            recipe = {"aux_anneal": "none"}
            for key, converter in (
                ("name", str),
                ("seed", int),
                ("hidden", int),
                ("layers", int),
                ("dropout", float),
                ("cell", str),
                ("batch", int),
                ("chunk", int),
                ("epochs", int),
                ("lr", float),
                ("wd", float),
                ("clip", float),
                ("optim", str),
                ("muon_scope", str),
                ("sched", str),
                ("warmup_frac", float),
                ("precision", str),
                ("loss", str),
                ("loss_norm", str),
                ("aux_next", float),
                ("aux_struct", float),
                ("aux_weight", float),
                ("batch_order", str),
                ("checkpoint_policy", str),
                ("checkpoint_epoch", int),
                ("train_source", str),
                ("train_seqs", int),
                ("eval_source", str),
                ("aux_anneal", str),
            ):
                flag = "--" + key.replace("_", "-")
                if flag in argv:
                    recipe[key] = converter(argv[argv.index(flag) + 1])
            if "--split-manifest" in argv:
                manifest = Path(argv[argv.index("--split-manifest") + 1])
                if file_sha256(manifest) != split.sha256:
                    raise ValueError(f"declared split differs: {run}")
                # setup_lineage.validate_membership canonicalizes this label;
                # exact evaluation membership was checked above, independent of label.
                recipe["eval_source"] = "vfold"
            if [row["epoch"] for row in history] != [0, 1, 2, 3]:
                raise ValueError(f"four complete epochs required: {run}")
            for row in history:
                if (
                    row.get("metric_id") != "global_wp_clip2_v1"
                    or row.get("blocks") != 1873
                    or row.get("selected_rows") != 3528273
                    or not math.isfinite(row["weighted_pearson"])
                    or not math.isfinite(row["valid_wmse"])
                ):
                    raise ValueError(f"full official validation endpoint differs: {run}")
            checkpoint = root / "model_ep3.pt"
            digest = file_sha256(checkpoint)
            if provenance["checkpoint_sha256"][checkpoint.name] != digest:
                raise ValueError(f"fixed checkpoint hash differs: {run}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("epoch") != 3:
                raise ValueError(f"saved checkpoint epoch differs: {run}")
            validate_chronological_teacher_membership(saved["provenance"], split, run)
            for args in (provenance["args"], saved["args"]):
                if any(
                    args.get(key, "none" if key == "aux_anneal" else None) != value
                    for key, value in recipe.items()
                ):
                    raise ValueError(
                        f"actual checkpoint/training arguments differ from panel: {run}"
                    )
                if (
                    args.get("split_manifest_sha256") != split.sha256
                    or args.get("precision") != "fp32"
                ):
                    raise ValueError(f"actual precision or split differs: {run}")
            if any(
                value.is_floating_point() and value.dtype != torch.float32
                for value in saved["model"].values()
            ):
                raise ValueError(f"non-FP32 checkpoint weights: {run}")
            runs.append(
                {
                    "run_name": run,
                    "configuration_id": cell["configuration_id"],
                    "seed": cell["seed"],
                    "history": history,
                    "fit_receipt": str(receipt_path),
                    "fit_receipt_sha256": file_sha256(receipt_path),
                    "history_path": str(root / "history.json"),
                    "history_sha256": file_sha256(root / "history.json"),
                    "provenance_path": str(root / "provenance.json"),
                    "provenance_sha256": file_sha256(root / "provenance.json"),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "fixed_ep3_wp": history[3]["weighted_pearson"],
                    "fixed_ep3_wmse": history[3]["valid_wmse"],
                }
            )
    control_rows = [row for row in runs if row["configuration_id"] == "gru256x2_control"]
    controls = {row["seed"]: row for row in control_rows}
    if len(control_rows) != 2 or set(controls) != {1001, 1002}:
        raise ValueError("matched controls differ")
    comparisons = []
    for name in ("clip5", "aux_late"):
        members = [row for row in runs if row["configuration_id"] == name]
        if len(members) != 2 or {row["seed"] for row in members} != {1001, 1002}:
            raise ValueError(f"matched candidate seeds differ: {name}")
        comparisons.append(
            {
                "configuration_id": name,
                "mean_wp": sum(row["fixed_ep3_wp"] for row in members) / 2,
                "mean_paired_delta": sum(
                    row["fixed_ep3_wp"] - controls[row["seed"]]["fixed_ep3_wp"] for row in members
                )
                / 2,
                "paired_deltas": {
                    str(row["seed"]): row["fixed_ep3_wp"] - controls[row["seed"]]["fixed_ep3_wp"]
                    for row in members
                },
            }
        )
    return {
        "schema": "wnn.l4-objective-teacher-report.v1",
        "status": "complete",
        "created_epoch": time.time(),
        "panel_sha256": file_sha256(panel_path),
        "control_panel_sha256": file_sha256(control_path),
        "identity": panel["identity"],
        "runtime": str(runtime),
        "collector": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "helper_source_root": str(SOURCE.resolve()),
        },
        "runs": runs,
        "comparisons": comparisons,
        "claim_scope": "Fixedep3 teacher comparison on exposed official development; two seeds. No student, public, serving or statistical-significance claim.",
    }


def plots(report: dict[str, Any], output: Path) -> list[Path]:
    """Draw actual train/valid metrics and whole-epoch gradient summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    names = ("gru256x2_control", "clip5", "aux_late")
    for row, name in enumerate(names):
        for run in report["runs"]:
            if run["configuration_id"] != name:
                continue
            h, seed = run["history"], run["seed"]
            for col, train_key, valid_key, title in (
                (0, "train_eval_wmse", "valid_wmse", "WMSE"),
                (1, "train_eval_wp", "weighted_pearson", "WP"),
            ):
                ax = axes[row, col]
                ax.plot(
                    range(1, 5), [v[valid_key] for v in h], marker="o", label=f"valid seed{seed}"
                )
                ax.plot(
                    range(1, 5),
                    [v[train_key] for v in h],
                    linestyle="--",
                    label=f"train diagnostic seed{seed}",
                )
                ax.set(title=f"{name}: {title}", xlabel="Completed epoch", xticks=range(1, 5))
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
    fig.suptitle(
        "Official valid; train diagnostic uses 32 sequences and a different row mask. Two seeds are not CIs."
    )
    paths = []
    for suffix in ("png", "svg"):
        f = output / ("teacher_train_valid_r1." + suffix)
        fig.savefig(f, dpi=150)
        paths.append(f)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for run in report["runs"]:
        gradients = [h["update_diagnostics"]["grad_stats"] for h in run["history"]]
        label = run["configuration_id"] + " s" + str(run["seed"])
        axes[0].plot(
            range(1, 5), [g["total_q05_q50_q95"][1] for g in gradients], marker="o", label=label
        )
        axes[1].plot(
            range(1, 5), [100 * g["clip_rate"] for g in gradients], marker="o", label=label
        )
    for ax, title in zip(
        axes, ("Preclip global gradient norm: median", "Clipped updates (%)"), strict=True
    ):
        ax.set(title=title, xlabel="Completed epoch", xticks=range(1, 5))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for suffix in ("png", "svg"):
        f = output / ("teacher_gradients_r1." + suffix)
        fig.savefig(f, dpi=150)
        paths.append(f)
    plt.close(fig)
    return paths


def main() -> None:
    """Collect once after terminal fits and optionally publish the measured artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--control-panel", required=True, type=Path)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    report = collect(args.panel, args.control_panel)
    runtime = Path(report["runtime"])
    output = runtime / "teacher-results-r1"
    output.mkdir(exist_ok=False)
    results = output / "teacher_report_r1.json"
    results.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure_paths = plots(report, output)
    lines = [
        "Measured fixedep3 teacher results (two seeds; exposed official validation).",
        "",
        "| Configuration | Mean WP | Paired mean delta |",
        "|---|---:|---:|",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"| {row['configuration_id']} | {row['mean_wp']:.9f} | {row['mean_paired_delta']:+.9f} |"
        )
    lines += [
        "",
        report["claim_scope"],
        "The diagnostic mask failure provided no gradient evidence; aux_late followed the frozen prior fallback.",
    ]
    body = "\n".join(lines)
    (output / "RESULTS.md").write_text(body + "\n", encoding="utf-8")
    evidence = [
        (Path(__file__).resolve(), "teacher_report_collector.py"),
        (results, results.name),
        (output / "RESULTS.md", "RESULTS.md"),
        *[(p, p.name) for p in figure_paths],
    ]
    for p in [
        *runtime.glob("control/*.json"),
        *runtime.glob("logs/fits/*.log"),
        *runtime.glob("logs/profiles/*.log"),
        *runtime.glob("receipts/fits/*.json"),
        *runtime.glob("receipts/profiles/*.json"),
    ]:
        publisher._add_file(evidence, p, str(p.relative_to(runtime)))
    weights = []
    for run in report["runs"]:
        root = Path(run["checkpoint_path"]).parent
        for name in ("history.json", "provenance.json"):
            evidence.append((root / name, f"runs/{run['run_name']}/{name}"))
        if run["configuration_id"] != "gru256x2_control":
            weights.extend(
                (p, f"weights/{run['run_name']}/{p.name}")
                for p in sorted(root.glob("model_ep*.pt"))
            )
    if shutil.disk_usage(runtime).free < 3_700_000_000:
        raise RuntimeError("preserve output; insufficient publication reserve")
    metadata = {
        "schema": "wnn.l4-objective-teacher-archive.v1",
        "panel_sha256": report["panel_sha256"],
        "claim_scope": report["claim_scope"],
    }
    assets = [output / "PR62_teacher_evidence_r1.zip", output / "PR62_teacher_weights_r1.zip"]
    publisher._write_archive(assets[0], evidence, {**metadata, "kind": "evidence"})
    publisher._write_archive(assets[1], weights, {**metadata, "kind": "all_epoch_teacher_weights"})
    if args.token_file is not None:
        client = publisher.GitHub(args.token_file.read_text().strip())
        head = client.request("GET", "/repos/VladKochetov007/WunderFunder/pulls/62")["head"]
        if head["ref"] != "experiment/l4-objective-20261104":
            raise ValueError("PR62 branch identity differs")
        release = publisher._publish_release(
            client,
            assets,
            head["sha"],
            body,
            release_tag="tabdata-pr62-teachers-20261104-r1",
            release_name="PR62 measured teacher endpoints and all epoch weights",
        )
        comment = publisher._post_comment(
            client,
            body + "\n\nArtifacts: " + release["url"],
            pr_number=62,
            idempotency_marker="<!-- pr62-teachers-result-r1 -->",
        )
        (output / "GITHUB_PUBLICATION_R1.json").write_text(
            json.dumps({"release": release, "comment": comment}, indent=2) + "\n", encoding="utf-8"
        )
    print("TEACHER_RESULTS", results, flush=True)


if __name__ == "__main__":
    main()
