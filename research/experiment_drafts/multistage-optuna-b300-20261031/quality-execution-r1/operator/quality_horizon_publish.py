"""Publish native journals for the bounded five-fit batch/TBPTT follow-up."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

Q = Path("/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1")
H = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")
PREFIX = "research/experiment_drafts/multistage-optuna-b300-20261031/quality-execution-r1"
REPO = "/repos/VladKochetov007/WunderFunder"
BRANCH = "results/tabdata-multistage-b300-20261031"
LOG = logging.getLogger(__name__)


def digest(raw: bytes) -> str:
    """Hash a transported record."""
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    """Serialize finite scientific records deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def snapshot() -> dict[str, object]:
    """Preserve full-primary labels and native journal sampling receipts."""
    from multistage_github_source import _journal, _preserve_metadata

    contract = json.loads((Q / "CONTRACT_R2.json").read_bytes())
    runs = []
    for name in ["A1302", "B1301", "B1302", "C1301", "C1302"]:
        output = Q / "runs" / name
        metadata_path = output / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata_raw = metadata_path.read_bytes()
        metadata = json.loads(metadata_raw)
        profile = json.loads(
            (Q / "profiles-r2" / name / "approved_profile_receipt.json").read_bytes()
        )
        provenance = metadata["provenance"]
        expected = profile["identity"]
        assert metadata["source"]["sha256"] == contract["source_sha256"]
        assert metadata["source"]["files"] == expected["source_files"]
        assert digest(canonical(metadata["config"])) == expected["config_sha256"]
        assert all(
            provenance[k] == expected[k] for k in ["data", "data_sha256", "runtime", "world_size"]
        )
        assert provenance["data"]["split_sha256"] == contract["primary_split_sha256"]
        assert len(provenance["data"]["fit_members"]) == 10325
        assert len(provenance["data"]["evaluation_members"]) == 1873
        assert (
            provenance["data"]["evaluation_source"]
            == provenance["data"]["evaluation_role"]
            == "valid"
        )
        child = json.loads((Q / "receipts" / f"{name}.fit.child.json").read_bytes())
        runs.append(
            {
                "run_id": "qh-" + name.lower(),
                "cell": name,
                "phase": "primary_batch_tbptt_ablation",
                "metadata": _preserve_metadata(metadata),
                "metadata_source_sha256": digest(metadata_raw),
                "metadata_canonical_sha256": digest(canonical(metadata)),
                "event_journal": _journal(output / "monitor_events.jsonl", evaluations_only=False),
                "evaluation_journal": _journal(
                    output / "monitor_evaluations.jsonl", evaluations_only=True
                ),
                "process_status": child["status"],
                "exit_code": child.get("exit_code"),
            }
        )
    return {
        "schema": "wnn.quality-horizon-native-snapshot.v1",
        "runs": runs,
        "contract": contract,
        "reused_control": "primary-refit-01-s1t001002-gru",
        "valid_role": "full official valid; development-exposed report-only",
    }


def commit_files(api: object, files: dict[str, bytes]) -> str:
    """Append a bounded result commit without replacing concurrent history."""
    from multistage_github_node_api import APIError

    blobs = []
    for name, raw in files.items():
        assert len(raw) < 3_000_000
        value = api.request(
            "POST",
            REPO + "/git/blobs",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"},
        )
        blobs.append(
            {"path": PREFIX + "/" + name, "type": "blob", "mode": "100644", "sha": value["sha"]}
        )
    for attempt in range(3):
        parent = api.request("GET", REPO + "/git/ref/heads/" + BRANCH)["object"]["sha"]
        base_tree = api.request("GET", REPO + "/git/commits/" + parent)["tree"]["sha"]
        tree = api.request("POST", REPO + "/git/trees", {"base_tree": base_tree, "tree": blobs})
        commit = api.request(
            "POST",
            REPO + "/git/commits",
            {
                "message": "Update quality-horizon native training journals",
                "parents": [parent],
                "tree": tree["sha"],
            },
        )
        try:
            api.request(
                "PATCH", REPO + "/git/refs/heads/" + BRANCH, {"sha": commit["sha"], "force": False}
            )
            return commit["sha"]
        except APIError as error:
            if error.status not in (409, 422) or attempt == 2:
                raise
    raise RuntimeError("unreachable commit state")


def main() -> None:
    """Publish changed source-bound views until the declared deadline."""
    sys.path.insert(0, str(H / "reporting-r3/scripts"))
    from galam_l4_quality8.publisher import read_token
    from multistage_github_node_api import APIError, GitHubAPI

    token, _ = read_token(Q / "private/github-report.token")
    api = GitHubAPI(token)
    assert api.request("GET", "/user")["login"] == "SemyonEpanov"
    state_path = Q / "receipts/live-publish-state.json"
    state = (
        json.loads(state_path.read_bytes()) if state_path.exists() else {"terminal": [], "hour": -1}
    )
    deadline = json.loads((Q / "CONTRACT_R2.json").read_bytes())["hard_deadline_epoch"]
    while time.time() < deadline:
        try:
            body = snapshot()
            raw = canonical(body)
            if digest(raw) != state.get("snapshot_sha256"):
                packed = gzip.compress(raw, mtime=0)
                manifest = {
                    "schema": "wnn.quality-horizon-snapshot-transfer.v1",
                    "compressed_sha256": digest(packed),
                    "payload_sha256": digest(raw),
                    "payload_bytes": len(raw),
                    "compressed_bytes": len(packed),
                }
                commit = commit_files(
                    api,
                    {
                        "live/native_snapshot.json.gz": packed,
                        "live/native_snapshot.manifest.json": canonical(manifest),
                    },
                )
                state.update(snapshot_sha256=digest(raw), commit=commit)
                LOG.info("published %s native views at %s", len(body["runs"]), commit[:12])
            terminal = sorted(r["cell"] for r in body["runs"] if r["process_status"] != "running")
            hour = int(time.time() // 3600)
            if terminal != state["terminal"] or hour != state["hour"]:
                statuses = ", ".join(r["cell"] + "=" + r["process_status"] for r in body["runs"])
                url = (
                    "https://github.com/VladKochetov007/WunderFunder/tree/" + BRANCH + "/" + PREFIX
                )
                api.request(
                    "POST",
                    REPO + "/issues/53/comments",
                    {
                        "body": "Quality follow-up: five fresh four-epoch GRU384x2 fits plus exact reused A/1301. "
                        "A=batch256/TBPTT256/acc4; B=64/256/4; C=64/1024/1; matched seeds1301/1302. "
                        "Full official train/valid boundary, no valid fitting or weight learning. "
                        + statuses
                        + ". [Native monitor snapshot]("
                        + url
                        + "). "
                        "NIER/MPS preserved; this is a separately bounded follow-up to the completed wave."
                    },
                )
                state.update(terminal=terminal, hour=hour)
            state_path.write_bytes(canonical(state))
            if len(body["runs"]) == 5 and len(terminal) == 5:
                LOG.info("all five fit processes terminal; final native snapshot published")
                return
        except APIError as error:
            if error.status != 429 and error.status < 500:
                raise
            LOG.warning("temporary GitHub API failure: HTTP%s", error.status)
        except (OSError, ValueError) as error:
            LOG.warning("changing input snapshot: %s", error)
        time.sleep(min(120, max(0, deadline - time.time())))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
