from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

DEFAULT_SUPPORT = Path(
    "/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/operator-r26/src"
)


def read_object(path: Path) -> dict[str, object]:
    """Read only the small private lifecycle fixture's JSON receipts."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("fixture receipt must be an object")
    return value


WORKER = """import json, subprocess, sys, time
from pathlib import Path
ready = [Path(sys.argv[1] + f'.ready{index}') for index in range(2)]
children = [subprocess.Popen([sys.executable, '-c',
    'import signal,time,sys; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(sys.argv[1]).write_text("ready"); time.sleep(30)', str(marker)],
    start_new_session=True) for marker in ready]
while not all(marker.exists() for marker in ready): time.sleep(0.01)
Path(sys.argv[1]).write_text(json.dumps([child.pid for child in children]))
time.sleep(0.6 if sys.argv[2] == 'exit' else 30)
raise SystemExit(2)
"""


class TermAfterChildren:
    """Trigger one bounded external TERM only after the synthetic rank children exist."""

    def __init__(self, mode: str, children: Path, receipt: Path, helpers: SimpleNamespace) -> None:
        self.mode, self.children, self.receipt, self.helpers = mode, children, receipt, helpers
        self.stopped = False
        self.fixture_identities: dict[int, dict[str, object]] = {}

    def __call__(self) -> None:
        """Signal only the fixture's current receipt-owned leader identity."""
        if self.children.exists():
            for pid in json.loads(self.children.read_text()):
                identity = self.helpers.identity(pid)
                if identity is not None and pid not in self.fixture_identities:
                    self.fixture_identities[pid] = identity
        if self.mode == "term" and self.children.exists() and not self.stopped:
            self.stopped = True
            self.helpers.terminate(read_object(self.receipt)["child"], grace_seconds=0.5)


def alive(pid: int) -> bool:
    """Treat zombies as stopped; retain actual process birth checks for every signal."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return raw[raw.rfind(")") + 2 :].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main() -> int:
    """Run two bounded Linux CPU containment checks; never load a model or CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-src", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        raise RuntimeError("native containment check is Linux-only")
    sys.path.insert(0, str(args.receiver))
    from controller import load_helpers

    helpers = load_helpers(args.support_src)
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="b300-lifecycle-", dir=args.scratch, delete=False
    ) as temporary:
        root = Path(temporary)
        worker = root / "synthetic_worker.py"
        worker.write_text(WORKER)
        foreign = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)",
            ],
            start_new_session=True,
        )
        foreign_identity = helpers.identity(foreign.pid)
        probes: list[TermAfterChildren] = []
        try:
            for mode in ("exit", "term"):
                proof, receipt, children = (
                    root / f"{mode}-{name}.json" for name in ("proof", "owned", "children")
                )
                progress = TermAfterChildren(mode, children, receipt, helpers)
                probes.append(progress)
                environment = dict(os.environ, WNN_CONTROLLER_DESCENDANTS_RECEIPT=str(proof))
                code, timed_out, _ = helpers.run(
                    [
                        sys.executable,
                        str(args.receiver / "owned_phase.py"),
                        "--support-src",
                        str(args.support_src),
                        "--",
                        sys.executable,
                        str(worker),
                        str(children),
                        mode,
                    ],
                    cwd=Path(__file__).parent,
                    env=environment,
                    log_path=root / f"{mode}.log",
                    receipt_path=receipt,
                    owner={"controller": helpers.owner()},
                    stage=1,
                    trial_number=0,
                    mode="cpu_lifecycle_fixture",
                    gpu_key="none",
                    output_dir=root,
                    timeout_seconds=8,
                    progress=progress,
                )
                pids = json.loads(children.read_text())
                evidence = read_object(proof)
                if (
                    code == 0
                    or timed_out
                    or not evidence.get("cleanup_success")
                    or evidence.get("remaining_non_zombie") != []
                    or any(alive(pid) for pid in pids)
                    or not alive(foreign.pid)
                    or (mode == "term" and not progress.stopped)
                ):
                    raise RuntimeError(f"native {mode} cleanup/foreign-preservation check failed")
                result = {
                    "test": mode,
                    "passed": True,
                    "child_exit": code,
                    "rank_pids": pids,
                    "foreign_preserved": True,
                    "proof": str(proof),
                    "external_TERM_grace_seconds": 0.5,
                }
                (root / f"{mode}-result.json").write_text(json.dumps(result))
                sys.stdout.write(json.dumps(result) + "\n")
        finally:
            for probe in probes:
                for identity in probe.fixture_identities.values():
                    if helpers.same(identity) and alive(int(identity["pid"])):
                        helpers.terminate(identity, grace_seconds=0)
            if foreign_identity is not None:
                helpers.terminate(foreign_identity, grace_seconds=0)
            foreign.wait(timeout=3)
        sys.stdout.write(
            json.dumps({"native_evidence_directory": str(root), "retained": True}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
