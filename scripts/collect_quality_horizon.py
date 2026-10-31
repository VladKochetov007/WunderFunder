"""Collect fixed four-epoch endpoints after the five owned fits terminate."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

Q = Path("/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1")
H = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")
PYTHON = "/data/epanov_s/tabdata/.venv/bin/python"
NAMES = ["A1302", "B1301", "B1302", "C1301", "C1302"]


def read(path: Path) -> dict[str, object]:
    """Read one declared local receipt."""
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    """Bind a checkpoint or receipt using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    """Preserve collection outputs as create-only artifacts."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def wait_for_terminal(deadline: float) -> dict[str, object]:
    """Wait for actual child receipts, bounded by the independent campaign deadline."""
    while time.time() < deadline - 1200:
        receipts = {name: Q / "receipts" / f"{name}.fit.child.json" for name in NAMES}
        if all(path.exists() for path in receipts.values()):
            statuses = {name: read(path) for name, path in receipts.items()}
            if all(
                row["status"] in {"completed", "failed", "timed_out"} for row in statuses.values()
            ):
                return statuses
        time.sleep(20)
    raise RuntimeError("owned fit receipts did not become terminal before collection reserve")


def assert_owned_groups_gone(statuses: dict[str, object]) -> None:
    """Refuse scoring while any captured fit process group has a live member."""
    groups = {int(row["child"]["pgid"]) for row in statuses.values()}
    for _attempt in range(30):
        remaining = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                raw = (path / "stat").read_text()
                fields = raw[raw.rfind(")") + 2 :].split()
                if int(fields[2]) in groups and fields[0] not in {"Z", "X"}:
                    remaining.append(int(path.name))
            except (OSError, ValueError, IndexError):
                continue
        if not remaining:
            return
        time.sleep(1)
    raise RuntimeError(f"captured owned fit groups still have live members: {remaining}")


def prepare_request(
    statuses: dict[str, object], deadline: float, output: Path
) -> dict[str, object]:
    """Reuse the frozen Source6 scorer for complete endpoints only."""
    old = read(H / "reports/full-primary-refits-r1/score_request.json")
    request = {
        key: old[key] for key in ["source6", "portable_data_registry", "primary_split", "data_root"]
    }
    request.update(
        schema="wnn.full-primary-refit-score-request.v1",
        hard_deadline_epoch=deadline,
        score_deadline_epoch=deadline - 600,
        output_dir=str(output),
        members=[],
    )
    for name in NAMES:
        child = statuses[name]
        run = Q / "runs" / name
        terminal_path = run / "terminal_status.json"
        if not terminal_path.exists():
            continue
        terminal, index = read(terminal_path), read(run / "epoch_index.json")
        configured_final = (
            child.get("status") == "completed"
            and type(child.get("exit_code")) is int
            and child.get("exit_code") == 0
            and child.get("timed_out") is False
            and terminal.get("status") == "completed"
            and terminal.get("completion_kind") == "max_epochs"
            and terminal.get("completed_epochs") == terminal.get("requested_epochs") == 4
        )
        if not configured_final or index.get("completed_epochs") != 4:
            continue
        model = run / "model.pt"
        if sha(model) != terminal.get("model_sha256"):
            raise ValueError("native terminal model bytes differ from the collected endpoint")
        approved = Q / "profiles-r2" / name / "approved_profile_receipt.json"
        profile = read(approved)
        native = read(run / "run_metadata.json")
        assert profile["approved"] is True
        assert native["source"]["sha256"] == profile["identity"]["source_sha256"]
        fit_request = read(Q / "inputs" / f"{name}.fit-request.json")
        member = {
            "fit_id": name,
            "configured_final_complete": True,
            "identity": profile["identity"],
            "config": native["config"],
            "config_sha256": fit_request["config_sha256"],
            "checkpoint": {"path": str(model), "sha256": sha(model)},
            "artifacts": {"approved_profile": {"path": str(approved), "sha256": sha(approved)}},
            "gpu_uuid": fit_request["env"]["CUDA_VISIBLE_DEVICES"],
            "predictions_path": str(output / (name + ".npy")),
            "receipt_path": str(output / (name + ".score.json")),
        }
        request["members"].append(member)
    return request


def main() -> None:
    """Own final prediction exports and the downstream no-fit paired comparison."""
    sys.path.insert(0, str(H / "operator-r26/src"))
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import process_identity

    deadline = float(read(Q / "CONTRACT_R2.json")["hard_deadline_epoch"])
    owner = process_identity(os.getpid())
    write(Q / "receipts/collector.owner.json", owner)
    statuses = wait_for_terminal(deadline)
    assert_owned_groups_gone(statuses)
    output = Q / "reports/final-scores-r1"
    output.mkdir(exist_ok=False)
    request = prepare_request(statuses, deadline, output)
    request_path = output / "score_request.json"
    write(request_path, request)

    def score(index: int) -> dict[str, object]:
        member = request["members"][index]
        name = member["fit_id"]
        env = os.environ.copy()
        env.update(read(Q / "inputs" / f"{name}.fit-request.json")["env"])
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        argv = [
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "890s",
            PYTHON,
            str(H / "operator-r26/scripts/report_multistage_primary_refits_worker.py"),
            "--request",
            str(request_path),
            "--request-sha256",
            sha(request_path),
            "--mode",
            "member",
            "--member-index",
            str(index),
        ]
        code, timed, record = run_owned_command(
            argv,
            cwd=H / "operator-r26",
            env=env,
            log_path=Q / "logs" / f"{name}.score.log",
            receipt_path=Q / "receipts" / f"{name}.score.child.json",
            owner=owner,
            stage=4,
            trial_number=index,
            mode="quality_horizon_final_score",
            gpu_key=member["gpu_uuid"],
            output_dir=output,
            timeout_seconds=900,
        )
        result = {"name": name, "exit_code": code, "timed_out": timed, "status": record["status"]}
        if code == 0:
            receipt = read(Path(member["receipt_path"]))
            events = [
                json.loads(x)
                for x in (Q / "runs" / name / "monitor_evaluations.jsonl").read_text().splitlines()
                if x
            ]
            assert events[-1]["epoch"] == 4
            assert abs(receipt["metrics"]["wp"] - events[-1]["metrics"]["valid/wp"]) < 1e-6
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        outcomes = list(pool.map(score, range(len(request["members"]))))
    write(Q / "receipts/final-scoring.completed.json", {"fits": statuses, "scores": outcomes})
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (Q / "logs/final-analysis.log").open("xb") as log:
        analysis = subprocess.run(
            [
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=5s",
                "600s",
                PYTHON,
                str(Q / "analyze_quality_horizon.py"),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=610,
        )
    assert analysis.returncode == 0, "final analysis failed; inspect its retained log"
    print("QUALITY_COLLECTION_AND_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
