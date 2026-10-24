"""Publish completed diagnostic checkpoint evaluations to the authorized PRs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    """Update one persistent comment per PR only when new completed results arrive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted((args.root / "evaluations").glob("*/*/RESULT.json")):
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            results.append(result)
    if not results:
        return
    digest = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()
    stamp = args.root / "GITHUB_VALIDATION.json"
    previous = json.loads(stamp.read_text()) if stamp.exists() else {}
    if previous.get("digest") == digest:
        return
    private = Path("/data/epanov_s/tabdata/runtime/github-reporting-v1")
    sys.path.insert(0, str(private / "code"))
    from github_api import GitHub, read_token

    api = GitHub(read_token(Path(json.loads((private / "CONFIG.json").read_text())["token_file"])))
    account = api.request("GET", "/user")
    if account["login"].lower() != "semyonepanov":
        raise ValueError("reporting account differs from user-authorized SemyonEpanov")
    body = "<!-- tabdata-pr32-intermediate-validation-20261024 -->\n"
    body += (
        "Intermediate checkpoint diagnostics added at user request. "
        "**The fixed epoch-30 endpoint policy is unchanged.**\n\n"
        "Full held-out F0/F2 evaluation and deterministic 64-sequence fitting probe "
        "use the same checkpoint, causal inference, original normalization and precision. "
        "Probe WMSE is a diagnostic subset estimate, not full-training-set error. "
        "Earlier epoch checkpoints were overwritten, so their validation curves cannot "
        "be reconstructed. These scores expose the outer folds and are not used for "
        "early stopping/checkpoint selection.\n\n"
    )
    body += (
        "| Cell | Epoch | Online train WMSE | Same-weight train probe WMSE | "
        "Val WMSE | Val official WP |\n|---|---:|---:|---:|---:|---:|\n"
    )
    summaries = []
    for result in sorted(results, key=lambda item: (item["cell"], item["completed_epochs"])):
        train = result["phases"]["train_probe"]
        valid = result["phases"]["heldout"]
        online = result["training_epoch_online"]["label_loss"]
        train_loss = train["weighted_mse_all_after_warmup"]
        val_loss = valid["weighted_mse_all_after_warmup"]
        wp = valid["official_wp"]
        body += (
            f"| {result['cell']} | {result['completed_epochs']} | {online:.8f} | "
            f"{train_loss:.8f} | {val_loss:.8f} | {wp:.8f} |\n"
        )
        summaries.append(
            {
                key: result[key]
                for key in (
                    "cell",
                    "completed_epochs",
                    "global_step",
                    "checkpoint_sha256",
                    "config_sha256",
                    "split_sha256",
                    "source_commit",
                )
            }
        )
        summaries[-1].update(
            train_probe_sequences=train["processed_sequences"],
            validation_sequences=valid["processed_sequences"],
            train_probe_wmse=train_loss,
            validation_wmse=val_loss,
            validation_wp=wp,
        )
    body += "\nProvenance and membership counts:\n```json\n"
    body += json.dumps(summaries, indent=2) + "\n```\n"
    for pr in ("32", "40"):
        key = f"comment_{pr}"
        if previous.get(key):
            response = api.patch(int(previous[key]), body)
        else:
            response = api.post(int(pr), body)
        previous[key] = response["id"]
        previous[f"url_{pr}"] = response["html_url"]
    previous.update(digest=digest, updated_unix=time.time(), completed=len(results))
    temporary = stamp.with_suffix(".tmp")
    temporary.write_text(json.dumps(previous, indent=2) + "\n")
    temporary.replace(stamp)


if __name__ == "__main__":
    main()
