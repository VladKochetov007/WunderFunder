from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent
END = 1794471619.060462
STOP = END + 175
HELPER = Path(
    "C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/github_publish.py"
)


def save(path: Path, value: dict) -> None:
    """Create one reviewable receipt without overwriting a previous attempt."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)


def rpc(host: str) -> dict:
    """Use the existing persistent RPC; this never opens an SSH connection."""
    name = f"final-wave-20261112-r1-{host}"
    code = (
        (ROOT / "final_wave_native_collect_r1.py").read_text().replace("__ROLE__", host)
    )
    request = ROOT / f"{host}-rpc/requests/{name}.json"
    payload = {"code": code}
    if request.exists():
        if json.loads(request.read_bytes()) != payload:
            raise ValueError("different request already owns final collection ID")
    else:
        temporary = request.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.rename(request)
    result = ROOT / f"{host}-rpc/results/{name}.json"
    while time.time() < STOP - 35:
        try:
            wrapped = json.loads(result.read_bytes())
            text = wrapped["output"]
            if f"WNN_RPC_END {name} ok" not in text:
                raise RuntimeError(f"native {host} evidence collection failed")
            return json.loads(text[text.index("\n") + 1 : text.rindex("\nWNN_RPC_END")])
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            time.sleep(0.5)
    raise TimeoutError(f"existing {host} RPC did not return before publication cutoff")


def clearance(host: str, value: dict) -> str:
    """Claim release only when the actual native proof verifies both process views."""
    proof = value.get("proof", {})
    clear = all(
        proof.get(k) is True
        for k in (
            "ownership_proof_complete",
            "owned_processes_clear",
            "owned_gpu_processes_clear",
        )
    )
    status = (
        "verified: owned processes and GPU contexts clear"
        if clear
        else "NOT fully verified; inspect native proof"
    )
    return f"- {host}: **{status}**; native proof SHA-256 `{value.get('proof_sha256')}`. Foreign GPU processes were not targeted.\n"


def render(pr: int, values: dict) -> str:
    """Keep partial monitoring and validated fixed endpoints in separate columns."""
    marker = f"<!-- tabdata-wave-20261112-final-r1:PR{pr} -->"
    text = (
        marker
        + "\nThe five-hour wave has ended (global GPU cutoff 2026-11-12 08:20:19 UTC).\n\n"
    )
    text += "".join(clearance(host, value) for host, value in values.items())
    text += "\n| Cell | Execution | Last complete monitored epoch | Full-valid monitor WP | Canonical fixed endpoint WP |\n|---|---|---:|---:|---:|\n"
    cohorts = (
        [(values["b300"], ["transformers"])]
        if pr == 81
        else [(values["l4"], ["first8"] if pr == 87 else ["followups4", "ema4"])]
    )
    for value, labels in cohorts:
        for label in labels:
            for cell in value.get("cohorts", {}).get(label, []):
                score = cell.get("fixed_canonical_score")
                monitor = cell.get("monitor_wp")
                text += (
                    f"| `{cell['cell']}` | {cell['execution_status']} | {cell.get('last_monitored_epoch') or '-'} | {monitor:.9f}"
                    if isinstance(monitor, (float, int))
                    else f"| `{cell['cell']}` | {cell['execution_status']} | {cell.get('last_monitored_epoch') or '-'} | unavailable"
                )
                text += f" | {score['wp']:.9f} |\n" if score else " | unavailable |\n"
    if pr == 81:
        text += (
            "\nFull queue outcomes (including deferred/unstarted cells): `"
            + json.dumps(values["b300"].get("queue_status_counts", {}), sort_keys=True)
            + "`. A queued cell is not a trained model. Retained-epoch report-only rescoring is separate from the fixed E4 endpoint.\n"
        )
    text += "\nOriginal logs, all completed epoch checkpoints, source/config/model identities and failed attempts remain preserved. Primary protocol is unchanged: 10,325 fitting members; official independent valid, 1,873 sequences / 3,528,273 scored rows; `global_wp_clip2_v1`. Train-eval and valid masks/populations differ. Validation is exposed; these are descriptive results, not public-score or significance claims.\n\nSource and execution records: #88, #87; followup context: #82. Local evidence receipts contain per-cell source/data/model/score hashes.\n"
    return text


def main() -> None:
    """Publish observed closure once; fail visibly instead of inventing an outcome."""
    while time.time() < END + 8:
        time.sleep(min(10, max(0.1, END + 8 - time.time())))
    values = {host: rpc(host) for host in ("b300", "l4")}
    save(ROOT / "FINAL_WAVE_EVIDENCE_R1.json", values)
    spec = importlib.util.spec_from_file_location("wave_github", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("publication helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.api("GET", "/user")["login"] != "SemyonEpanov":
        raise RuntimeError("unexpected GitHub publication account")
    for pr in (81, 82, 87):
        if time.time() >= STOP:
            raise TimeoutError("bounded final publication deadline reached")
        body = render(pr, values)
        marker = body.split("\n", 1)[0]
        comments = module.api(
            "GET", f"/repos/{module.REPO}/issues/{pr}/comments?per_page=100"
        )
        previous = next(
            (
                x
                for x in comments
                if marker in x["body"] and x["user"]["login"] == "SemyonEpanov"
            ),
            None,
        )
        result = previous or module.api(
            "POST", f"/repos/{module.REPO}/issues/{pr}/comments", {"body": body}
        )
        path = ROOT / f"FINAL_WAVE_PR{pr}_RECEIPT_R1.json"
        if not path.exists():
            save(
                path,
                {
                    "url": result["html_url"],
                    "author": result["user"]["login"],
                    "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "at": time.time(),
                },
            )
        print(json.dumps({"pr": pr, "url": result["html_url"]}), flush=True)


if __name__ == "__main__":
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    try:
        main()
    except Exception as error:
        failure = ROOT / "FINAL_WAVE_PUBLISH_FAILURE_R1.json"
        if not failure.exists():
            save(
                failure,
                {
                    "at": time.time(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
