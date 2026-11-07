"""Save per-epoch local PNG/SVG artifacts from actual mirrored validation journals."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from epoch_snapshot import capture, complete_events, read_json, verify_snapshot

from wnn.multistage_search import atomic_write_json, file_lock, file_sha256

LOGGER = logging.getLogger(__name__)
DEFAULT_ARTIFACTS = "C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107"


class EpochPlots:
    """Keep one CPU plotting child while the existing bridge continues independently."""

    def __init__(self, artifacts: Path, source: Path, python: Path) -> None:
        self.artifacts, self.source, self.python = artifacts, source, python
        self.jobs: dict[str, dict[str, object]] = {}
        self.errors: dict[str, str] = {}
        self.active: tuple[subprocess.Popen[bytes], dict[str, object], float] | None = None
        self.plot_sources = {
            str(source / name): file_sha256(source / name)
            for name in (
                "scripts/plot_sequence_monitor.py",
                "src/wnn/multistage_plot_data.py",
                "src/wnn/multistage_plot_fit_figures.py",
            )
        }

    def publish(self) -> None:
        """Publish watcher-only status without writing any live run file."""
        atomic_write_json(
            self.artifacts / "epoch-plot-watcher-status.json",
            {
                "schema": "wnn.epoch-plot-watcher.v1",
                "updated_at": datetime.now(UTC).isoformat(),
                "jobs": self.jobs,
                "run_errors": self.errors,
                "active_pid": self.active[0].pid if self.active else None,
                "plot_jobs_at_once": 1,
                "plot_threads": 2,
                "model_loads": 0,
                "model_evaluations": 0,
            },
        )

    def scan(self) -> None:
        """Queue every newly recorded valid epoch, including delayed multi-epoch replay."""
        mirror = self.artifacts / "live-dashboard"
        for run in sorted((mirror / "runs").iterdir()):
            if (
                not run.is_dir()
                or run.is_symlink()
                or not (run / ".sequence_mirror_owner.json").is_file()
            ):
                continue
            try:
                for event, raw in complete_events(run / "monitor_evaluations.jsonl"):
                    key = f"{run.name}/epoch{event['epoch']}"
                    digest = hashlib.sha256(raw).hexdigest()
                    if key in self.jobs:
                        if self.jobs[key]["evaluation_event_sha256"] != digest:
                            raise ValueError("previously recorded epoch changed its native bytes")
                        continue
                    snapshot = capture(
                        run, mirror / "native" / run.name, self.artifacts, event, raw
                    )
                    if snapshot is None:
                        break
                    plots = self.artifacts / "plots" / run.name / f"epoch{event['epoch']}"
                    job = {
                        "run": run.name,
                        "epoch": event["epoch"],
                        "snapshot": str(snapshot),
                        "plots": str(plots),
                        "evaluation_event_sha256": digest,
                        "status": "pending",
                    }
                    if plots.exists():
                        self.verify_plots(plots, snapshot)
                        job["status"] = "completed"
                    self.jobs[key] = job
                self.errors.pop(run.name, None)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.errors[run.name] = f"{type(error).__name__}: {error}"
                LOGGER.exception("Epoch artifact input rejected for%s", run.name)
        self.publish()

    def verify_plots(self, plots: Path, snapshot: Path) -> None:
        """Verify native plot receipts, both formats, and their exact immutable inputs."""
        receipt = read_json(plots / "plot_source.json")
        manifest = read_json(snapshot / "snapshot_manifest.json")
        if (
            receipt.get("status") != "exported"
            or receipt.get("model_evaluations_triggered") != 0
            or Path(receipt["run_directory"]).resolve() != snapshot.resolve()
        ):
            raise ValueError("plot helper receipt does not bind this model-free snapshot")
        exports = receipt["plots_sha256"]
        if not exports or receipt["plot_count"] != len(exports):
            raise ValueError("plot helper produced no complete export inventory")
        for name, digest in exports.items():
            if (
                Path(name).name != name
                or Path(name).suffix not in (".png", ".svg")
                or file_sha256(plots / name) != digest
            ):
                raise ValueError("saved PNG/SVG bytes differ from their plot receipt")
            sibling = Path(name).with_suffix(".svg" if name.endswith(".png") else ".png").name
            if sibling not in exports:
                raise ValueError("each graph must have both PNG and SVG")
        for name, digest in receipt["source_sha256"].items():
            if manifest["files_sha256"].get(name) != digest:
                raise ValueError("plot sources differ from the immutable epoch snapshot")
        verify_snapshot(snapshot, manifest["evaluation_event_sha256"])

    def start_next(self, remaining: float) -> None:
        """Invoke the existing plot helper once with two-thread CPU environment bounds."""
        if self.active is not None or remaining <= 1:
            return
        job = next((job for job in self.jobs.values() if job["status"] == "pending"), None)
        if job is None:
            return
        final = Path(job["plots"])
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError("refusing to overwrite an existing per-epoch plot directory")
        staging = Path(tempfile.mkdtemp(prefix=f".epoch{job['epoch']}-pending-", dir=final.parent))
        environment = dict(
            os.environ,
            PYTHONPATH=str(self.source / "src"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(self.artifacts / "matplotlib-cache"),
        )
        command = [
            str(self.python),
            str(self.source / "scripts/plot_sequence_monitor.py"),
            "--run-dir",
            job["snapshot"],
            "--plots-dir",
            str(staging),
        ]
        with (staging / "plot.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                cwd=self.source,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        job.update(status="plotting", pid=process.pid, staging=str(staging), command=command)
        self.active = process, job, time.monotonic() + min(3600, remaining)
        self.publish()

    def finish(self) -> None:
        """Publish successful plots once, preserving failed scratch evidence without retry."""
        if self.active is None:
            return
        process, job, deadline = self.active
        code = process.poll()
        if code is None and time.monotonic() < deadline:
            return
        if code is None:
            self.stop_child()
            return
        staging, snapshot = Path(job["staging"]), Path(job["snapshot"])
        try:
            if code != 0:
                raise RuntimeError(f"existing plot helper failed with exit{code}")
            if any(file_sha256(Path(path)) != digest for path, digest in self.plot_sources.items()):
                raise ValueError("plot helper source changed during this watcher")
            self.verify_plots(staging, snapshot)
            atomic_write_json(
                staging / "epoch_snapshot.json",
                {
                    "snapshot": str(snapshot.resolve()),
                    "snapshot_manifest_sha256": file_sha256(snapshot / "snapshot_manifest.json"),
                    "plotter_source_sha256": self.plot_sources,
                    "watcher_source_sha256": {
                        str(path): file_sha256(path) for path in Path(__file__).parent.glob("*.py")
                    },
                    "model_loads": 0,
                    "model_evaluations": 0,
                },
                replace=False,
            )
            os.rename(staging, Path(job["plots"]))
            job.update(status="completed", finished_at=datetime.now(UTC).isoformat())
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            job.update(status="failed", error=f"{type(error).__name__}: {error}")
            LOGGER.exception("Per-epoch plot artifact failed")
        self.active = None
        self.publish()

    def stop_child(self) -> None:
        """Stop only this watcher's exact Popen handle; no live trainer is signalled."""
        if self.active is None:
            return
        process, job, _ = self.active
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        job.update(status="interrupted", exit_code=process.returncode)
        self.active = None
        self.publish()


def main() -> None:
    """Watch for up to24hours, polling actual local mirrors every15seconds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(DEFAULT_ARTIFACTS))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--plot-python", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=86400)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 86400:
        parser.error("max-seconds must be between1 and86400")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with file_lock(args.artifacts / "epoch-plot-watcher.lock", blocking=False):
        previous = args.artifacts / "epoch-plot-watcher-status.json"
        if previous.exists() and read_json(previous).get("active_pid") is not None:
            raise RuntimeError(
                "prior watcher left an unclosed plotting child; preserve and inspect its ownership before restart"
            )
        watcher = EpochPlots(
            args.artifacts.resolve(), args.source_root.resolve(), args.plot_python.resolve()
        )
        deadline, next_scan = time.monotonic() + args.max_seconds, 0.0
        try:
            while time.monotonic() < deadline:
                if time.monotonic() >= next_scan:
                    watcher.scan()
                    next_scan = time.monotonic() + 15
                watcher.finish()
                watcher.start_next(deadline - time.monotonic())
                if (
                    args.once
                    and watcher.active is None
                    and not any(job["status"] == "pending" for job in watcher.jobs.values())
                ):
                    break
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        finally:
            watcher.stop_child()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
