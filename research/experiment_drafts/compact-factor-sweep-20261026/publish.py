"""Publish verified train/validation curves to the authorized experiment PR."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

from curves import render, selection
from stage import WAVES, read

REPOSITORY = "/repos/VladKochetov007/WunderFunder"
ARTIFACT_PATH = "research/experiment_drafts/compact-factor-sweep-20261026/plots"


def report_text(root: Path, records: list[dict]) -> str:
    """Describe actual progress and selected development endpoints."""
    state = (
        read(root / "CAMPAIGN_STATE.json")
        if (root / "CAMPAIGN_STATE.json").exists()
        else {"status": "profiles_or_preparation"}
    )
    new_points = sum(not r["reused"] for r in records)
    text = (
        "<!-- tabdata-compact-factor-sweep-20261026 -->\n"
        f"Campaign status: **{state['status']}**. Complete new validation points: **{new_points}/40**; "
        "20 retained PR43 points reused for control/joint-treatment plots.\n\n"
        "Eight new five-epoch fits; four two-B300 groups per wave. Batch128/rank, global256, "
        "1150updates/fit. BF16/BF16x9, FP32 masters/states, warmup5% + cosine.\n\n"
        "| Wave | Runtime cell | Scientific arm | Epochs | Update |\n|---|---|---|---:|---:|\n"
    )
    for wave in WAVES:
        plan = read(root / wave / "plan-profile.json")
        for job in plan["jobs"]:
            fit = root / wave / "cohort-b128" / job["name"] / "fit"
            rank = fit / "rank0.jsonl"
            last = {}
            if rank.exists():
                with rank.open("rb") as stream:
                    size = stream.seek(0, 2)
                    stream.seek(max(0, size - 16384))
                    for line in reversed(stream.read().splitlines()):
                        try:
                            last = json.loads(line)
                            break
                        except (ValueError, UnicodeDecodeError):
                            continue
            epochs = len(list((fit / "epoch_indexes").glob("epoch-*.json")))
            text += f"| {wave} | {job['name']} | {job['scientific_arm']} | {epochs}/5 | {last.get('step', 0)}/1150 |\n"
    text += "\n| Arm | Selected common epoch | F0 WP | F2 WP | Mean WP | Paired mean delta |\n|---|---:|---:|---:|---:|---:|\n"
    for arm, item in selection(records)["selection"].items():
        text += f"| {arm} | {item['epoch']} | {item['wp_by_fold']['0']:.7f} | {item['wp_by_fold']['2']:.7f} | {item['mean_wp']:.7f} | {item['mean_delta']:+.7f} |\n"
    text += (
        "\nTrain loss in figures is the same-weight model.eval fixed64 fitting probe; "
        "validation loss uses the full inner holdout, both all rows>=99. Different populations: "
        "absolute gap is not an unbiased generalization-gap estimate. Validation WP uses its official scored mask. "
        "Each arm selects a common epoch by mean F0/F2 WP, earliest exact tie. One seed per fold; "
        "all splits are development-exposed. No significance, asymptotic or public-score claim.\n"
    )
    return text


def main() -> None:
    """Update the PR branch and one reconciled progress comment, without force pushes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if (root / "DELIVERY_COMPLETE.json").exists():
        return
    with (root / "PUBLISH.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        records = render(root)
        body = report_text(root, records)
        digest = hashlib.sha256(
            (json.dumps(records, sort_keys=True) + body.split("| Wave |")[0]).encode()
        ).hexdigest()
        stamp = root / "GITHUB_PROGRESS.json"
        prior = read(stamp) if stamp.exists() else {}
        if prior.get("digest") == digest and time.time() - prior.get("unix", 0) < 3600:
            return
        helper = Path("/data/epanov_s/tabdata/runtime/github-reporting-v1")
        sys.path.insert(0, str(helper / "code"))
        from github_api import GitHub, read_token

        api = GitHub(read_token(Path(read(helper / "CONFIG.json")["token_file"])))
        if api.request("GET", "/user")["login"].lower() != "semyonepanov":
            raise ValueError("unexpected publishing account")
        publication = read(root / "GITHUB_PR.json")
        branch = publication["head_ref"]
        actual_pr = api.request("GET", REPOSITORY + "/pulls/" + str(publication["number"]))
        if actual_pr["head"]["ref"] != branch:
            raise ValueError("publication branch differs from the actual experiment PR")
        prefix = (
            "https://github.com/VladKochetov007/WunderFunder/blob/" + branch + "/" + ARTIFACT_PATH
        )
        body += f"\n![Regularization train/val]({prefix}/regularization.png?raw=true)\n\n![Learning-rate train/val]({prefix}/learning_rate.png?raw=true)\n\n[Coordinates and hashes]({prefix}/coordinates.json) · [CSV]({prefix}/coordinates.csv) · [Selected endpoints]({prefix}/selection.json)\n"
        (root / "plots/README.md").write_text(body)
        ref = api.request("GET", REPOSITORY + "/git/ref/heads/" + branch)
        parent = ref["object"]["sha"]
        commit = api.request("GET", REPOSITORY + "/git/commits/" + parent)
        entries = []
        for path in sorted((root / "plots").iterdir()):
            blob = api.request(
                "POST",
                REPOSITORY + "/git/blobs",
                {"content": base64.b64encode(path.read_bytes()).decode(), "encoding": "base64"},
            )
            entries.append(
                {
                    "path": ARTIFACT_PATH + "/" + path.name,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob["sha"],
                }
            )
        tree = api.request(
            "POST", REPOSITORY + "/git/trees", {"base_tree": commit["tree"]["sha"], "tree": entries}
        )
        result = api.request(
            "POST",
            REPOSITORY + "/git/commits",
            {
                "message": f"Update verified Transformer train/val curves ({sum(not r['reused'] for r in records)}/40 new points)",
                "tree": tree["sha"],
                "parents": [parent],
            },
        )
        api.request(
            "PATCH",
            REPOSITORY + "/git/refs/heads/" + branch,
            {"sha": result["sha"], "force": False},
        )
        if not prior.get("comment_id"):
            matches = [
                c
                for c in api.comments(publication["number"])
                if "<!-- tabdata-compact-factor-sweep-20261026 -->" in c.get("body", "")
                and c["user"]["login"].lower() == "semyonepanov"
            ]
            if matches:
                prior["comment_id"] = matches[-1]["id"]
        response = (
            api.patch(prior["comment_id"], body)
            if prior.get("comment_id")
            else api.post(publication["number"], body)
        )
        stamp.write_text(
            json.dumps(
                {
                    "digest": digest,
                    "unix": time.time(),
                    "comment_id": response["id"],
                    "url": response["html_url"],
                    "commit_sha": result["sha"],
                    "new_points": sum(not r["reused"] for r in records),
                },
                indent=2,
            )
            + "\n"
        )
        state = (
            read(root / "CAMPAIGN_STATE.json") if (root / "CAMPAIGN_STATE.json").exists() else {}
        )
        if state.get("status") == "completed" and sum(not r["reused"] for r in records) == 40:
            (root / "DELIVERY_COMPLETE.json").write_text(stamp.read_text())


if __name__ == "__main__":
    main()
