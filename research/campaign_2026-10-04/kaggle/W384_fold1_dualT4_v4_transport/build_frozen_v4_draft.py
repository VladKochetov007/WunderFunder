from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/campaign_2026-10-04/kaggle"
V3 = BASE / "W384_fold1_dualT4_v3"
HERE = BASE / "W384_fold1_dualT4_v4_transport"
OLD_CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3"
NEW_CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4"
OLD_DATASET = "vladkochetov07/wnn-w384-fold1-dualt4-20261004-v3"
NEW_DATASET = "vladkochetov07/wnn-w384-fold1-dualt4-20261004-v4"
NEW_SLUG = "wnn-w384-fold1-dualt4-20261004-v4"
NEW_KERNEL = "vladkochetov07/wnn-w384-fold1-dualt4-20261004-v4-runner"
OLD_ARCHIVE_SHA = "cfaed9a32399348d5bb1a13b817cd219c69b0f72adb03de7496735a173094a05"
WHOLE_JOB_LIMIT_SECONDS = 25_200
WATCHDOG_LIMIT_SECONDS = 25_180
PROBE_LIMIT_SECONDS = 2_400
FIT_LIMIT_SECONDS = 12_600


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def replace_text(value: str) -> str:
    return (
        value.replace(OLD_CAMPAIGN, NEW_CAMPAIGN)
        .replace("W384_fold1_dualT4_v3", "W384_fold1_dualT4_v4")
        .replace("t4v3", "t4v4")
        .replace(OLD_DATASET, NEW_DATASET)
        .replace("dualt4-20261004-v3", "dualt4-20261004-v4")
    )


def write_archive(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            with tarfile.open(fileobj=gz, mode="w|") as tar:
                for name, data in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(data))


def load_v3_payload() -> dict[str, bytes]:
    payload = {}
    with tarfile.open(V3 / "bundle/bundle.wnn", "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError(f"Unexpected non-file archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Missing archive content: {member.name}")
            payload[member.name] = source.read()
    if len(payload) != 266:
        raise ValueError(f"Unexpected frozen v3 member count: {len(payload)}")
    return payload


def rename_payload(payload: dict[str, bytes]) -> dict[str, bytes]:
    renamed: dict[str, bytes] = {}
    for old_path, raw in payload.items():
        if old_path.endswith("/probe_reuse_evidence.json"):
            continue
        new_path = replace_text(old_path)
        if new_path in renamed:
            raise ValueError(f"Archive path collision: {new_path}")
        if old_path.startswith("kaggle_specs/") or "/plans/" in old_path or old_path.endswith("paired_fold1_screen.json"):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Expected JSON control file: {old_path}") from exc
            rendered = json_bytes(_replace_tree(value))
            if old_path.endswith(("_probe.json", "_fit.json")) and old_path.startswith("kaggle_specs/"):
                spec = json.loads(rendered)
                timeout = PROBE_LIMIT_SECONDS if spec["name"].endswith("_probe") else FIT_LIMIT_SECONDS
                spec["limits"]["timeout_seconds"] = timeout
                rendered = json_bytes(spec)
            renamed[new_path] = rendered
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                renamed[new_path] = raw
            else:
                if old_path.endswith("/phase_runner.py"):
                    text = make_phase_runner(text)
                renamed[new_path] = replace_text(text).encode()
    expected_specs = {
        f"kaggle_specs/c20261004_w{width}_adamw_s3_f1_t4v4_{phase}.json"
        for width in (256, 384)
        for phase in ("probe", "fit")
    }
    if not expected_specs.issubset(renamed):
        raise ValueError("v4 must contain exactly the four fresh fit/probe specifications")
    return renamed


def _replace_tree(value: Any) -> Any:
    if isinstance(value, str):
        return replace_text(value)
    if isinstance(value, list):
        return [_replace_tree(item) for item in value]
    if isinstance(value, dict):
        return {
            replace_text(key) if isinstance(key, str) else key: _replace_tree(item)
            for key, item in value.items()
        }
    return value


def make_phase_runner(text: str) -> str:
    text = replace_text(text)
    probe_validator = '''def validate_probe_updates(rows: dict[str, dict], output: Path) -> dict[str, dict]:
    """Require an observed finite production-batch update on each probe worker."""
    result = {}
    for name in sorted(rows):
        log_path = output / name / "worker.log"
        if not log_path.is_file():
            raise RuntimeError(f"Probe worker log is missing: {name}")
        fit_lines = [line[4:] for line in log_path.read_text().splitlines() if line.startswith("FIT ")]
        if len(fit_lines) != 1:
            raise RuntimeError(f"Probe must emit exactly one FIT record: {name}")
        event = json.loads(fit_lines[0])
        counts = event.get("update_diagnostics", {}).get("epoch", {})
        if event.get("epoch") != 0 or event.get("evaluation") != "withheld":
            raise RuntimeError(f"Probe did not finish its single no-evaluation epoch: {name}")
        if counts.get("completed_batches") != 1:
            raise RuntimeError(f"Probe did not complete exactly one production batch: {name}")
        if counts.get("optimizer_attempts", 0) < 1 or counts.get("optimizer_completed_updates", 0) < 1:
            raise RuntimeError(f"Probe has no completed optimizer update: {name}")
        if counts.get("amp_skipped_updates") != 0:
            raise RuntimeError(f"Probe had an AMP-skipped update: {name}")
        if counts.get("ema_updates") != counts.get("optimizer_completed_updates"):
            raise RuntimeError(f"Probe EMA and optimizer update counts differ: {name}")
        result[name] = counts
    return result


'''
    if "def validate_probe_updates(" not in text:
        text = text.replace("def main() -> None:\n", probe_validator + "def main() -> None:\n")
    start = text.index("        probe_evidence_path = ROOT / f\"{CAMPAIGN}/probe_reuse_evidence.json\"")
    end_marker = '        fit_summary = launch_runner(fit_names, OUTPUT / "fit_phase")'
    end = text.index(end_marker, start) + len(end_marker)
    block = '''        probe_summary = launch_runner(probe_names, OUTPUT / "probe_phase")
        probe_rows = validate_workers(probe_summary, probe=True)
        probe_updates = validate_probe_updates(probe_rows, OUTPUT / "probe_phase")
        probe_devices = {
            width: probe_rows[WORKER_NAMES[(width, True)]]["device"]["uuid"] for width in WIDTHS
        }
        if len(set(probe_devices.values())) != 2:
            raise RuntimeError("Both probe workers must use distinct physical T4 UUIDs")
        evidence["probe_phase"] = {
            "reused": False,
            "concurrency_admitted": probe_summary["concurrency_admitted"],
            "workers": {
                row["name"]: {
                    "device": row["device"],
                    "peak_gpu_bytes": row["peak_gpu_bytes"],
                    "peak_ram_bytes": row["peak_ram_bytes"],
                    "status": row["status"],
                    "returncode": row["returncode"],
                    "single_batch_update": probe_updates[row["name"]],
                }
                for row in probe_summary["finished"]
            },
            "gate": "both fresh one-batch probes complete on distinct T4s below memory caps before either fit starts",
        }
        fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]
        if devices[0]["uuid"] == probe_devices[WIDTHS[1]]:
            fit_names.reverse()
        fit_summary = launch_runner(fit_names, OUTPUT / "fit_phase")'''
    result = text[:start] + block + text[end:]
    reuse_start = result.find("def validate_reused_probe_evidence(")
    if reuse_start >= 0:
        reuse_end = result.index("def validate_probe_updates(", reuse_start)
        result = result[:reuse_start] + result[reuse_end:]
    if "probe_reuse_evidence" in result or "validate_reused_probe_evidence" in result:
        raise ValueError("Fresh v4 runner still contains a probe-reuse path")
    if 'CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4"' not in result:
        raise ValueError("Runner campaign identity did not switch to v4")
    return result


def rebuild_specs(payload: dict[str, bytes]) -> None:
    for path, raw in list(payload.items()):
        if not path.startswith("kaggle_specs/"):
            continue
        spec = json.loads(raw)
        spec["code"] = sorted(spec["code"], key=lambda item: item["path"])
        spec["inputs"] = sorted(spec["inputs"], key=lambda item: item["path"])
        for group in ("code", "inputs"):
            for descriptor in spec[group]:
                descriptor["path"] = replace_text(descriptor["path"])
                member = descriptor["path"]
                if member in payload:
                    descriptor.update({"size": len(payload[member]), "sha256": sha(payload[member])})
        spec["code_sha256"] = canonical_sha(spec["code"])
        payload[path] = json_bytes(spec)


def main() -> None:
    bundle_dir = HERE / "bundle"
    kernel_dir = HERE / "kernel_push"
    if bundle_dir.exists() or kernel_dir.exists():
        raise FileExistsError("Draft bundle/kernel outputs already exist; preserve and version a new attempt")
    manifest = json.loads((V3 / "bundle/bundle-manifest.json").read_text())
    if manifest["archive_sha256"] != OLD_ARCHIVE_SHA:
        raise ValueError("Frozen v3 archive SHA changed")
    payload = rename_payload(load_v3_payload())
    rebuild_specs(payload)

    # Update the portable source manifest reference path in every worker spec and archive file.
    new_phase = f"{NEW_CAMPAIGN}/phase_runner.py"
    phase_bytes = payload[new_phase]
    for path, raw in list(payload.items()):
        if path.startswith("kaggle_specs/"):
            spec = json.loads(raw)
            for group in ("code", "inputs"):
                for item in spec[group]:
                    if item["path"] == new_phase:
                        item.update(size=len(phase_bytes), sha256=sha(phase_bytes))
            spec["code_sha256"] = canonical_sha(sorted(spec["code"], key=lambda item: item["path"]))
            payload[path] = json_bytes(spec)

    bundle_dir.mkdir(parents=True)
    archive_path = bundle_dir / "bundle.wnn"
    write_archive(archive_path, payload)
    archive_bytes = archive_path.read_bytes()
    new_manifest = _replace_tree(manifest)
    new_manifest.update(
        dataset=NEW_DATASET,
        archive_bytes=len(archive_bytes),
        archive_sha256=sha(archive_bytes),
        expanded_bytes=sum(map(len, payload.values())),
        spec_paths=sorted(path for path in payload if path.startswith("kaggle_specs/")),
        files=sorted(
            [{"path": path, "sha256": sha(data), "size": len(data)} for path, data in payload.items()],
            key=lambda item: item["path"],
        ),
    )
    (bundle_dir / "bundle-manifest.json").write_bytes(json_bytes(new_manifest))
    dataset_meta = {"id": NEW_DATASET, "title": NEW_SLUG, "isPrivate": True, "licenses": [{"name": "other"}]}
    (bundle_dir / "dataset-metadata.json").write_bytes(json_bytes(dataset_meta))

    kernel_dir.mkdir()
    old_kernel = (V3 / "kernel_push/kernel.py").read_text()
    old_kernel = old_kernel.replace("wnn-w384-fold1-dualt4-20261004-v3", NEW_SLUG)
    old_kernel = old_kernel.replace("W384_fold1_dualT4_v3", "W384_fold1_dualT4_v4")
    old_kernel = old_kernel.replace(OLD_ARCHIVE_SHA, sha(archive_bytes))
    old_kernel = old_kernel.replace(
        "W384_fold1_dualT4_v3/phase_runner.py", "W384_fold1_dualT4_v4/phase_runner.py"
    )
    old_kernel = old_kernel.replace("def main() -> None:", "def _kernel_work() -> None:")
    old_kernel = old_kernel.replace('import sys\n', 'import sys\nimport signal\nimport time\nimport select\n')
    old_kernel = old_kernel.replace('\n\nif __name__ == "__main__":\n    main()\n', "\n")
    watchdog = (HERE / "whole_job_watchdog_draft.py").read_text()
    helper = watchdog[watchdog.index("def _proc_identity("):]
    kernel_source = old_kernel + "\n\n" + helper + "\n\n" + '''
def main() -> None:
    watchdog_pid = start_watchdog(
        timeout_seconds=25_180,
        report_path="/kaggle/working/whole_job_watchdog.json",
    )
    try:
        _kernel_work()
    finally:
        stop_watchdog(watchdog_pid)


if __name__ == "__main__":
    main()
'''
    if 'EXPECTED_ARCHIVE_SHA256 = "' + sha(archive_bytes) + '"' not in kernel_source:
        # Existing kernel uses single-quoted assignment.
        if "EXPECTED_ARCHIVE_SHA256 = '" + sha(archive_bytes) + "'" not in kernel_source:
            raise ValueError("Kernel archive pin was not replaced")
    (kernel_dir / "kernel.py").write_text(kernel_source)
    kernel_metadata = json.loads((V3 / "kernel_push/kernel-metadata.json").read_text())
    kernel_metadata.update(id=NEW_KERNEL, title=NEW_KERNEL.rsplit("/", 1)[1])
    kernel_metadata["dataset_sources"] = [NEW_DATASET, *kernel_metadata["dataset_sources"][1:]]
    (kernel_dir / "kernel-metadata.json").write_bytes(json_bytes(kernel_metadata))

    # Verify archive membership and descriptors from bytes actually emitted.
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}
    if extracted != payload:
        raise ValueError("Round-trip archive bytes differ from frozen payload map")
    if "probe_reuse_evidence.json" in "\n".join(extracted):
        raise ValueError("Probe-reuse evidence must not be carried into fresh v4")
    for path, raw in extracted.items():
        if path.startswith("kaggle_specs/"):
            spec = json.loads(raw)
            timeout = PROBE_LIMIT_SECONDS if spec["name"].endswith("_probe") else FIT_LIMIT_SECONDS
            if spec["limits"]["timeout_seconds"] != timeout:
                raise ValueError(f"Incorrect worker timeout: {path}")
    (HERE / "build_result_draft.json").write_bytes(
        json_bytes(
            {
                "status": "built_source_only_needs_root_review",
                "dataset": NEW_DATASET,
                "kernel": NEW_KERNEL,
                "archive_sha256": sha(archive_bytes),
                "archive_bytes": len(archive_bytes),
                "manifest_sha256": sha((bundle_dir / "bundle-manifest.json").read_bytes()),
                "kernel_sha256": sha((kernel_dir / "kernel.py").read_bytes()),
                "kernel_metadata_sha256": sha((kernel_dir / "kernel-metadata.json").read_bytes()),
                "whole_job_watchdog_seconds": WATCHDOG_LIMIT_SECONDS,
                "provider_hard_timeout_seconds": WHOLE_JOB_LIMIT_SECONDS,
                "probe_worker_timeout_seconds": PROBE_LIMIT_SECONDS,
                "fit_worker_timeout_seconds": FIT_LIMIT_SECONDS,
                "fit_count": 2,
                "probe_reuse": False,
                "provider_mutations": False,
            }
        )
    )


if __name__ == "__main__":
    main()
