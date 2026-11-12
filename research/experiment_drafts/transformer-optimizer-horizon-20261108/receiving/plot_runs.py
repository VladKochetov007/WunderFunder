from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from admission import read


def main() -> None:
    """Render actual report-only raw-z epoch metrics to CPU PNG/SVG evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plan = read(args.plan)
    runtime = Path(plan["runtime"])
    control = runtime / "control/receiving-r1"
    control.mkdir(parents=True, exist_ok=True)
    ready = control / "plots-ready.json"
    ready.write_text(
        json.dumps(
            {"status": "ready", "matplotlib": matplotlib.__version__, "cuda": "not imported"}
        ),
        encoding="utf-8",
    )
    deadline = read(Path(plan["authority"]))["deadline_epoch"]
    groups = [
        ("Official raw-z WP", ["train_eval/z/wp", "valid/z/wp", "valid/z/wp/t0", "valid/z/wp/t1"]),
        (
            "Official raw-z WMSE",
            ["train_eval/z/wmse", "valid/z/wmse", "valid/z/wmse/t0", "valid/z/wmse/t1"],
        ),
        (
            "Raw prediction means",
            [f"valid/z/raw_mean/t{i}" for i in range(2)]
            + [f"valid/z/raw_std/t{i}" for i in range(2)],
        ),
        (
            "Prediction clipping fraction",
            [f"valid/z/prediction_clipped_fraction/t{i}" for i in range(2)],
        ),
    ]
    while time.time() < deadline:
        for cell in plan["cells"]:
            fit = runtime / "runs" / cell["cell"] / "fit"
            journal = fit / "monitor_evaluations.jsonl"
            if not journal.exists():
                continue
            events = [
                json.loads(line)
                for line in journal.read_bytes().splitlines()
                if line and json.loads(line).get("kind") == "evaluation"
            ]
            for index, event in enumerate(events):
                epoch = int(event["epoch"])
                folder = fit / "plots"
                marker = folder / f"epoch-{epoch:02d}.json"
                if marker.exists():
                    continue
                folder.mkdir(exist_ok=True)
                current = events[: index + 1]
                x = [item["epoch"] for item in current]
                figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
                for axis, (title, fields) in zip(axes.flat, groups, strict=True):
                    for field in fields:
                        if all(field in item["metrics"] for item in current):
                            axis.plot(
                                x,
                                [item["metrics"][field] for item in current],
                                marker="o",
                                label=field,
                            )
                    axis.set(title=title, xlabel="completed epoch")
                    axis.grid(alpha=0.2)
                    axis.legend(fontsize=7)
                figure.suptitle(
                    f"{cell['cell']} — full valid is_scored after warmup; train_eval is all_after_warmup; fixed endpoint4E"
                )
                files = {}
                for extension in ("png", "svg"):
                    name = f"epoch-{epoch:02d}.{extension}"
                    target, temporary = folder / name, folder / (name + ".partial")
                    figure.savefig(temporary, format=extension, dpi=130)
                    temporary.replace(target)
                    files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                plt.close(figure)
                marker.write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": event["step"],
                            "files": files,
                            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                            "source_manifest_sha256": plan["source_manifest_sha256"],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        if (control / "plots-stop.json").exists():
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
