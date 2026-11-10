from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from admission import read


def main() -> None:
    """Write actual per-epoch PNG/SVG from native journals in a separate CPU-only process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plan = read(args.plan)
    runtime = Path(plan["runtime"])
    control = runtime / "control/receiving-r1"
    control.mkdir(parents=True, exist_ok=True)
    ready = control / "plots-ready.json"
    with ready.open("x") as stream:
        json.dump(
            {
                "status": "ready",
                "matplotlib": matplotlib.__version__,
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "cuda": "not imported; CPU journal-only rendering",
            },
            stream,
        )
    deadline = read(Path(plan["authority"]))["deadline_epoch"]
    while time.time() < deadline:
        for cell in plan["cells"]:
            fit = runtime / "runs" / cell["cell"] / "fit"
            journal = fit / "monitor_evaluations.jsonl"
            if not journal.exists():
                continue
            events = []
            for line in journal.read_bytes().splitlines(keepends=True):
                if line.endswith(b"\n"):
                    event = json.loads(line)
                    if event.get("kind") == "evaluation":
                        events.append(event)
            for index, event in enumerate(events):
                folder = fit / "plots"
                marker = folder / f"epoch-{int(event['epoch']):02d}.json"
                if marker.exists():
                    continue
                folder.mkdir(exist_ok=True)
                current = events[: index + 1]
                x = [item["epoch"] for item in current]
                figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
                groups = [
                    (
                        ["train_eval/wp", "valid/wp", "valid/wp/t0", "valid/wp/t1"],
                        "Fixed .9TL3+.1z official WP",
                    ),
                    (
                        ["train_eval/z/wp", "valid/z/wp", "valid/z/wp/t0", "valid/z/wp/t1"],
                        "Standalone z official WP",
                    ),
                    (
                        [
                            "train_eval/mixture/wmse",
                            "valid/mixture/wmse",
                            "valid/mixture/wmse/t0",
                            "valid/mixture/wmse/t1",
                        ],
                        "Fixed mixture unscaled WMSE",
                    ),
                    (
                        ["train_eval/z/wmse", "valid/z/wmse", "valid/z/wmse/t0", "valid/z/wmse/t1"],
                        "Standalone z unscaled WMSE",
                    ),
                ]
                for axis, (fields, title) in zip(axes.flat, groups, strict=True):
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
                    f"{cell['cell']} — actual full-valid development-exposed reports\ntrain_eval: all_after_warmup; valid: is_scored after warmup; fixed endpoint4E"
                )
                files = {}
                for extension in ("png", "svg"):
                    name = f"epoch-{int(event['epoch']):02d}.{extension}"
                    temporary = folder / (name + ".partial")
                    figure.savefig(temporary, format=extension, dpi=130)
                    target = folder / name
                    temporary.replace(target)
                    files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
                plt.close(figure)
                with marker.open("x") as stream:
                    json.dump(
                        {
                            "epoch": event["epoch"],
                            "step": event["step"],
                            "files": files,
                            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                            "source_manifest_sha256": plan["source_manifest_sha256"],
                        },
                        stream,
                        indent=2,
                    )
        if args.once or (control / "plots-stop.json").exists():
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
