"""Freeze committed canonical KD source and four exact configs into a receiver capsule."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1] / "reviews/WunderFunder-pr49-standalone-kd-20261112"
COMMIT = "33542502a1404a2843359b3999fa34f3a157bf3f"
NATIVE = "/data/lc-epanov_s/tabdata-final-quality-20261112-r1"


def sha(data: bytes) -> str:
    """Hash byte-exact Git blobs and canonical JSON identities."""
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    """Write new evidence with deterministic UTF-8 bytes."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main() -> None:
    """Preserve the historical overlay manifest and bind the actual clean source."""
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORK).decode().strip() == COMMIT
    )
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "src", "scripts"], cwd=WORK)
    source = ROOT / "kd-receiver-r2/source-kd-r1"
    source.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            COMMIT,
            "src",
            "scripts",
            "utils.py",
            "pyproject.toml",
            "tests/test_sequence_standalone_kd.py",
        ],
        cwd=WORK,
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        for member in handle.getmembers():
            assert member.isdir() or member.isfile()
            if member.isdir():
                continue
            target = source / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = handle.extractfile(member).read()
            target.write_bytes(raw)
            files[member.name] = sha(raw)
    manifest = {"schema": "wnn.git-source-capsule.v1", "commit": COMMIT, "files": files}
    save(source / "SOURCE_MANIFEST.json", manifest)
    source_manifest_sha = sha((source / "SOURCE_MANIFEST.json").read_bytes())
    identity_files = {
        key: value
        for key, value in files.items()
        if (key.startswith("src/") and key.endswith(".py")) or key == "scripts/train_sequence.py"
    }
    source_identity = sha(
        json.dumps(
            identity_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    original = ROOT / "pr49-standalone-kd-r1"
    packet = source.parent / "pr49-standalone-kd-r2"
    shutil.copytree(original, packet)
    panel = json.loads((packet / "PANEL_MANIFEST_R1.json").read_bytes())
    panel["schema"] = "wnn.pr49.standalone-kd-panel.r2"
    panel["status"] = "committed source; native cache/config preflight and profiles pending"
    panel["historical_r1_overlay_sha256"] = panel.pop("source_overlay_sha256")
    panel.pop("source_overlay_files")
    panel["source_tree_status"] = "clean committed Git-blob runtime capsule"
    panel["source_commit"] = COMMIT
    panel["source_identity_sha256"] = source_identity
    panel["source_manifest_sha256"] = source_manifest_sha
    panel["allocation"] = (
        "four new same-source lambda0/0.2 fits onGPU0-3; fixedE6; shared CLOCK_R1 unchanged"
    )
    for entry in panel["cells"]:
        assert sha((packet / entry["config"]).read_bytes()) == entry["config_sha256"]
    save(packet / "PANEL_MANIFEST_R2.json", panel)
    panel_sha = sha((packet / "PANEL_MANIFEST_R2.json").read_bytes())
    controller = (ROOT / "l4_wave_r1.py").read_text(encoding="utf-8")
    replacements = {
        '"intake/pr49-distillation-r1"': '"kd-intake-r2/pr49-standalone-kd-r2"',
        '"/data2/lc-epanov_s/tabdata-all-followups-20261112-r1/source-r2"': f'"{NATIVE}/kd-intake-r2/source-kd-r1"',
        'choices=("profile-labels", "profile-kd", "fit-labels", "fit-kd")': 'choices=("profile-newkd", "fit-newkd")',
        '"PANEL_MANIFEST_R1.json"': '"PANEL_MANIFEST_R2.json"',
        "44fdf3c1d5017c744b3070a5f8dac35fdb95b0ebb8721792e2d19f943267c33d": panel_sha,
        "29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511": source_identity,
        "00782a64e1557a8af759e6f987038cabb6ec09f098c43ab61d6cfa6bb07e9e5f": source_manifest_sha,
        "390c846cbdacd8e0e06e6079dd3b04312fa6bcc5": COMMIT,
        'kd = args.phase.endswith("-kd")': "kd = True",
        "gpu = index + (4 if kd else 0)": "gpu = index",
    }
    for before, after in replacements.items():
        assert controller.count(before) == 1, before
        controller = controller.replace(before, after)
    compile(controller, "l4_standalone_kd_wave_r1.py", "exec")
    (packet / "l4_standalone_kd_wave_r1.py").write_bytes(controller.encode())
    receipt = {
        "commit": COMMIT,
        "source_sha256": source_identity,
        "source_manifest_sha256": source_manifest_sha,
        "panel_manifest_sha256": panel_sha,
        "controller_sha256": sha(controller.encode()),
        "source_files": len(files),
    }
    save(ROOT / "KD_SOURCE_CAPSULE_R2.json", receipt)
    target = ROOT / "l4-standalone-kd-receiver-r2.tar.gz"
    with tarfile.open(target, "x:gz") as handle:
        handle.add(source, arcname=source.name)
        handle.add(packet, arcname=packet.name)
    request = {
        "transfer": {"local_archive": str(target), "remote_extract_dir": NATIVE + "/kd-intake-r2"}
    }
    temporary = (
        ROOT.parent
        / "cluster-all-experiments-20261112/l4-rpc/requests/179-kd-source-transfer-r2.tmp"
    )
    save(temporary, request)
    temporary.rename(temporary.with_suffix(".json"))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
