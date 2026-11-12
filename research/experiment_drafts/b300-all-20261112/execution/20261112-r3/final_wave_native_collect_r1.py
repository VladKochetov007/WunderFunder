from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

ROLE = "__ROLE__"
SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def read(path: Path) -> dict:
    """Read a bounded regular evidence object without following a link."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 16_000_000:
        return {}
    value = json.loads(path.read_bytes())
    return value if isinstance(value, dict) else {}


def digest(path: Path) -> str:
    """Fingerprint a small receipt or a local model without retaining its bytes."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_summary(root: Path, cell: str, execution: dict, epochs: int) -> dict:
    """Separate execution, trainer monitoring, and an exact fixed-endpoint score."""
    fit = root / "runs" / cell / "fit"
    terminal = read(fit / "terminal_status.json")
    metadata = read(fit / "run_metadata.json")
    latest = {}
    journal = fit / "monitor_evaluations.jsonl"
    if journal.is_file() and not journal.is_symlink():
        for line in journal.read_bytes().splitlines():
            if line.strip():
                latest = json.loads(line)
    metrics = latest.get("metrics", {})
    valid = (
        metrics.get("valid/selected_rows") == 3_528_273
        and metrics.get("valid/sequence_count") == 1_873
        and latest.get("context", {}).get("split_sha256") == SPLIT
    )
    score = read(fit / "score.json")
    metric = score.get("score", {})
    model = fit / "model.pt"
    fixed = (
        terminal.get("status") == "completed"
        and terminal.get("completed_epochs") == epochs
        and score.get("split_sha256") == SPLIT
        and score.get("sequences") == 1_873
        and metric.get("metric_id") == "global_wp_clip2_v1"
        and metric.get("selected_rows") == 3_528_273
        and all(
            isinstance(metric.get(k), (int, float)) and math.isfinite(metric[k])
            for k in ("wp", "t0", "t1")
        )
        and model.is_file()
        and not model.is_symlink()
        and terminal.get("model_sha256")
        == score.get("producer", {}).get("checkpoint_sha256")
        and digest(model) == terminal.get("model_sha256")
    )
    return {
        "cell": cell,
        "execution_status": execution.get("status", "unverified"),
        "trainer_status": terminal.get("status"),
        "completed_epochs": terminal.get("completed_epochs"),
        "last_monitored_epoch": latest.get("epoch"),
        "monitor_wp": metrics.get("valid/wp") if valid else None,
        "fixed_endpoint_epochs": epochs,
        "fixed_canonical_score": metric if fixed else None,
        "canonical_score_sha256": digest(fit / "score.json") if fixed else None,
        "source_sha256": metadata.get("source", {}).get("sha256"),
        "data_sha256": metadata.get("provenance", {}).get("data_sha256"),
        "model_sha256": terminal.get("model_sha256"),
    }


def collect() -> dict:
    """Collect only this wave's declared evidence; never run a model or send a signal."""
    root = Path(
        "/data/epanov_s/tabdata-all-20261112-r1"
        if ROLE == "b300"
        else "/data/lc-epanov_s/tabdata-all-20261112-r1"
    )
    proof_path = root / "control/wave-drain-r1/final.json"
    proof = read(proof_path)
    output = {
        "role": ROLE,
        "at": time.time(),
        "proof": proof,
        "proof_sha256": digest(proof_path) if proof else None,
        "cohorts": {},
    }
    if ROLE == "b300":
        queue = read(root / "control/impact81-transformer-queue-r2/status.json")
        output["queue_status_counts"] = dict(
            Counter(x.get("status", "unknown") for x in queue.get("cells", {}).values())
        )
        output["cohorts"]["transformers"] = [
            cell_summary(root, cell, state, 4)
            for cell, state in queue.get("cells", {}).items()
            if (root / "runs" / cell / "fit/monitor_evaluations.jsonl").is_file()
        ]
        partial = root / "control/partial-e3-readout-r1"
        output["partial_readout_receipts"] = {
            p.name: read(p) for p in partial.glob("*result.json")
        }
    else:
        for label, fit_root, control in (
            ("first8", root, "root-fits-r1"),
            (
                "followups4",
                Path("/data2/lc-epanov_s/tabdata-all-followups-20261112-r1"),
                "root-fits-r4",
            ),
            (
                "ema4",
                Path("/data2/lc-epanov_s/tabdata-ema-20261112-r1"),
                "root-fits-ema-r2",
            ),
        ):
            output["cohorts"][label] = [
                cell_summary(fit_root, p.name.removesuffix("-result.json"), read(p), 6)
                for p in sorted((fit_root / "control" / control).glob("*-result.json"))
            ]
    return output


print(json.dumps(collect()))
