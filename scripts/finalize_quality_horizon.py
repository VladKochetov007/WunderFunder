"""Publish the completed follow-up report and byte-verified configured-final weights."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

Q = Path("/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1")
H = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")
TAG = "tabdata-pr53-quality-horizon-checkpoints-20261031-r1"
REPO = "/repos/VladKochetov007/WunderFunder"


def read(path: Path) -> dict[str, object]:
    """Read a declared completed artifact."""
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    """Hash original payloads using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    """Keep delivery receipts create-only."""
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def archive_weights(request: dict[str, object], report: dict[str, object]) -> list[Path]:
    """Archive only newly completed, actually scored endpoints and their proof."""
    folder = Q / "reports/checkpoint-release-r1"
    folder.mkdir(exist_ok=False)
    payloads = [
        (Q / "CONTRACT_R2.json", "shared/CONTRACT_R2.json"),
        (Q / "inputs/final-comparison-panel-r1.json", "shared/comparison_panel.json"),
        (H / "data/chronological_primary.json", "shared/chronological_primary.json"),
    ]
    complete = {row["name"] for row in report["records"] if row["kind"] == "fresh_configured_final"}
    for member in request["members"]:
        name = member["fit_id"]
        if name not in complete:
            continue
        checkpoint = Path(member["checkpoint"]["path"])
        assert sha(checkpoint) == member["checkpoint"]["sha256"]
        payloads.append((checkpoint, name + "/model.pt"))
        payloads.append((Q / "configs-r2" / (name + ".json"), name + "/config.json"))
        for filename in [
            "run_metadata.json",
            "terminal_status.json",
            "epoch_index.json",
            "monitor_evaluations.jsonl",
        ]:
            payloads.append((Q / "runs" / name / filename, name + "/" + filename))
        for filename in [
            "profile_receipt.json",
            "engineering_review_r2.json",
            "approved_profile_receipt.json",
        ]:
            payloads.append(
                (Q / "profiles-r2" / name / filename, name + "/engineering/" + filename)
            )
        for suffix in ["fit.child.json", "score.child.json", "root-approval.json"]:
            payloads.append((Q / "receipts" / (name + "." + suffix), name + "/receipts/" + suffix))
    manifest = {
        "schema": "wnn.quality-horizon-checkpoint-archive.v1",
        "models": len(complete),
        "model_names": sorted(complete),
        "selection": "configured-final epoch4 only",
        "partial_models_excluded": True,
        "reused_A1301_not_republished": True,
        "prior_replay_sources": "tabdata-pr53-multistage-checkpoints-20261031-r1",
        "source4_sha256": "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81",
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha(path)}
            for path, name in payloads
        ],
    }
    manifest_path = folder / "MANIFEST.json"
    write(manifest_path, manifest)
    archive_path = folder / "quality-horizon-configured-final.zip"
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for path, name in payloads:
            assert not path.is_symlink()
            archive.write(path, name)
        archive.write(manifest_path, "MANIFEST.json")
    with zipfile.ZipFile(archive_path) as archive:
        for item in manifest["files"]:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
    checksums = folder / "SHA256SUMS.txt"
    checksums.write_text(
        "\n".join(sha(path) + "  " + path.name for path in [archive_path, manifest_path]) + "\n"
    )
    return [archive_path, manifest_path, checksums]


def publish_release(paths: list[Path], token: str) -> dict[str, object]:
    """Check the bytes actually downloaded from GitHub after each upload."""
    from publish_checkpoint_releases import GitHub, SafeRedirect

    api = object.__new__(GitHub)
    api.token = token
    api.opener = urllib.request.build_opener(SafeRedirect())
    try:
        release = api.request("GET", REPO + "/releases/tags/" + TAG)
    except RuntimeError as error:
        if "HTTP404" not in str(error):
            raise
        release = api.request(
            "POST",
            REPO + "/releases",
            {
                "tag_name": TAG,
                "target_commitish": "experiment/multistage-optuna-b300-20261031",
                "name": "Quality-horizon configured-final research models, 2026-10-31",
                "prerelease": True,
                "body": "Five-fit batch/TBPTT follow-up. Includes only complete scored epoch4 models; partials and reused A1301 excluded. Official train-only ancestry. Research weights; serving ABI/runtime/public transfer not established. Source4/Source6 replay capsules remain in the prior PR53 release.",
            },
        )
    existing = {item["name"]: item for item in release.get("assets", [])}
    assets = []
    for path in paths:
        asset = existing.get(path.name) or api.upload(release["id"], path)
        target = path.parent / (
            "roundtrip-" + str(asset["id"]) + "-" + str(time.time_ns()) + ".bin"
        )
        api.download(asset, target, sha(path))
        assets.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "asset_id": asset["id"],
                "roundtrip_verified": True,
            }
        )
    return {
        "status": "complete",
        "release_url": release["html_url"],
        "assets": assets,
        "author": "SemyonEpanov",
    }


def gpu_release_receipt() -> dict[str, object]:
    """Verify our GPU-producing commands ended; preserve every other process."""
    live = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (
                (path / "cmdline").read_bytes().replace(bytes([0]), b" ").decode(errors="replace")
            )
        except OSError:
            continue
        if str(Q) in command and any(
            term in command
            for term in [
                "train_sequence.py",
                "report_multistage_primary_refits_worker.py",
                "predict_run.py",
            ]
        ):
            live.append({"pid": int(path.name), "command": command})
    assert not live, "GPU-producing commands from this wave still exist"
    return {
        "schema": "wnn.quality-horizon-gpu-release.v1",
        "status": "no_owned_gpu_producing_processes",
        "observed_epoch": time.time(),
        "owned_live_commands": live,
        "foreign_process_signals": 0,
        "foreign_NIER_MPS_preserved": True,
    }


def main() -> None:
    """Wait for collection, publish evidence and weights, then remove our staged credential."""
    sys.path.insert(0, str(H / "reporting-r3/scripts"))
    sys.path.insert(0, str(Q))
    from galam_l4_quality8.publisher import read_token, remove_owned_token
    from multistage_github_node_api import GitHubAPI
    from quality_horizon_publish import commit_files

    deadline = float(read(Q / "CONTRACT_R2.json")["hard_deadline_epoch"])
    report_path = Q / "reports/final-comparison-r1/REPORT.json"
    while time.time() < deadline - 60:
        if report_path.exists() and (report_path.parent / "train_valid_curves.svg").exists():
            break
        time.sleep(20)
    else:
        raise RuntimeError("final comparison did not arrive before delivery deadline")
    report = read(report_path)
    assert report["status"] == "complete"
    token, identity = read_token(Q / "private/github-report.token")
    api = GitHubAPI(token)
    assert api.request("GET", "/user")["login"] == "SemyonEpanov"
    request = read(Q / "reports/final-scores-r1/score_request.json")
    release = publish_release(archive_weights(request, report), token)
    write(Q / "receipts/quality-checkpoints-release-r1.json", release)
    cleanup = gpu_release_receipt()
    write(Q / "receipts/final-gpu-release-r1.json", cleanup)
    files = {
        "final/" + path.name: path.read_bytes()
        for path in report_path.parent.iterdir()
        if path.suffix in {".json", ".csv", ".png", ".svg"}
    }
    files["final/WEIGHT_RELEASE.json"] = (
        Q / "receipts/quality-checkpoints-release-r1.json"
    ).read_bytes()
    files["final/GPU_RELEASE.json"] = (Q / "receipts/final-gpu-release-r1.json").read_bytes()
    commit = commit_files(api, files)
    strongest = max(report["records"], key=lambda row: row["wp"])
    pairs = report["paired_deltas"]
    url = "https://github.com/VladKochetov007/WunderFunder/tree/results/tabdata-multistage-b300-20261031/research/experiment_drafts/multistage-optuna-b300-20261031/quality-execution-r1/final"
    text = (
        "Quality follow-up complete: "
        + str(report["observed_complete_fresh_models"])
        + "/5 new configured-final models; exact reused A1301 kept separate. Best measured row **"
        + strongest["name"]
        + " WP"
        + format(strongest["wp"], ".9f")
        + "**, delta vs replayed TL3 "
        + format(strongest["delta_vs_TL3"], "+.9f")
        + ". Paired seed deltas: "
        + json.dumps(pairs)
        + ". [Full table, trajectories and receipts]("
        + url
        + "); [SHA-roundtrip verified model weights]("
        + release["release_url"]
        + "). "
        + "Official train-only fitting, full valid report-only; fixed epoch4/seed weights/alphas, development-exposed evidence, no significance or public-score claim. "
        + "Our GPU commands have exited; foreign NIER/MPS preserved. Missing cells: "
        + json.dumps(report["missing_cells"])
        + "."
    )
    comment = api.request("POST", REPO + "/issues/53/comments", {"body": text})
    write(
        Q / "receipts/final-delivery.completed.json",
        {
            "status": "complete",
            "commit": commit,
            "comment_url": comment["html_url"],
            "url": url,
            "release_url": release["release_url"],
            "author": "SemyonEpanov",
            "finished_epoch": time.time(),
        },
    )
    assert remove_owned_token(Q / "private/github-report.token", identity)
    print("QUALITY_DELIVERY_COMPLETE", comment["html_url"], flush=True)


if __name__ == "__main__":
    main()
