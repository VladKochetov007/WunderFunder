"""Grade, archive, and publish the frozen W18 L4 cohort after its exports finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from http.client import HTTPMessage
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts.w18_l4_quality_20261104.contract import (  # noqa: E402
    CohortPanel,
    load_panel,
    read_object,
    sha256_file,
)
from scripts.w18_l4_quality_20261104.runtime import cell_paths, runtime_bytes  # noqa: E402
from wnn.multistage_search_owned_command import run_owned_command  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    current_process_owner,
    terminate_owned_process,
)

LOGGER = logging.getLogger("w18_l4_quality_finalize")
REPOSITORY = "VladKochetov007/WunderFunder"
LOGIN = "SemyonEpanov"
PR_NUMBER = 60
RELEASE_TAG = "tabdata-pr60-results-20261104-r1"
COMMENT_MARKER = f"<!-- w18-l4-quality-results:{RELEASE_TAG} -->"
OUTPUT_LIMIT_BYTES = 8_000_000_000
FREE_FLOOR_BYTES = 3_000_000_000


class FinalizeError(RuntimeError):
    """Raised when frozen cohort evidence or publication identity is invalid."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the API token on fixed GitHub API/upload origins only."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        _ = (req, fp, code, msg, headers, newurl)
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FinalizeError(f"unsafe or existing finalizer receipt path: {path.name}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _stable_sha(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"expected regular non-symlink evidence file: {path.name}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != key:
        raise FinalizeError(f"evidence changed while hashing: {path.name}")
    return digest, key


def _budget(panel: CohortPanel) -> dict[str, int]:
    paths = cast(dict[str, object], panel.value["paths"])
    mount = Path(str(paths["disk_mount"]))
    free = shutil.disk_usage(mount).free
    written = runtime_bytes(panel)
    if free < FREE_FLOOR_BYTES:
        raise FinalizeError("data2 free space is below the frozen 3 GB reserve")
    if written > OUTPUT_LIMIT_BYTES:
        raise FinalizeError("new runtime output exceeds the frozen 8 GB cap")
    return {"disk_free_bytes": free, "runtime_output_bytes": written}


def _deadline(panel: CohortPanel) -> float:
    runtime = Path(str(panel.value["paths"]["runtime_root"]))
    state = read_object(runtime / "COHORT_STATE_R1.json")
    value = state.get("hard_deadline_epoch")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizeError("original cohort hard deadline is missing")
    return float(value)


def _pipeline_snapshot(panel: CohortPanel) -> tuple[str | None, dict[str, object]]:
    control = Path(str(panel.value["paths"]["control_root"]))
    path = control / "WAVE_PIPELINE_R1.json"
    if path.is_symlink() or not path.exists():
        return None, {}
    state = read_object(path)
    status = state.get("status")
    if status not in {"running", "fits_complete", "exports_complete_pending_grade", "failed"}:
        raise FinalizeError("wave pipeline has an unknown status")
    return cast(str, status), state


def _prior_guard(panel: CohortPanel) -> dict[str, object] | None:
    """Require root's completed audit of all eight epoch-zero checkpoint memberships."""
    control = Path(str(panel.value["paths"]["control_root"]))
    path = control / "FIRST_CHECKPOINT_AUDIT_R1.json"
    if path.is_symlink() or not path.exists():
        return None
    audit = read_object(path)
    checks = audit.get("checked")
    if audit.get("status") != "pass" or not isinstance(checks, list) or len(checks) != 8:
        if audit.get("status") == "failed":
            raise FinalizeError("FIRST_CHECKPOINT_AUDIT_R1 reports a failure")
        return None
    cells = cast(list[dict[str, object]], panel.value["cells"])
    expected = {str(cell["cell_id"]): str(cell["configuration_id"]) for cell in cells}
    by_run = {str(cell["run_name"]): str(cell["cell_id"]) for cell in cells}
    actual: dict[str, dict[str, object]] = {}
    for item in checks:
        if not isinstance(item, dict):
            return None
        identity = next(
            (
                item[key]
                for key in ("cell_id", "cellID", "run_name", "run")
                if isinstance(item.get(key), str)
            ),
            None,
        )
        if not isinstance(identity, str):
            return None
        cell_id = identity if identity in expected else by_run.get(identity)
        if cell_id is None:
            raise FinalizeError("FIRST_CHECKPOINT_AUDIT_R1 contains an unknown cell ID")
        actual[cell_id] = cast(dict[str, object], item)
    if set(actual) != set(expected):
        raise FinalizeError("FIRST_CHECKPOINT_AUDIT_R1 cell IDs differ from the frozen panel")
    if any(item.get("status") != "pass" for item in actual.values()):
        if any(item.get("status") == "failed" for item in actual.values()):
            raise FinalizeError("FIRST_CHECKPOINT_AUDIT_R1 contains a failed cell")
        return None
    for cell_id, item in actual.items():
        if item.get("fit_train") != 10325 or item.get("fit_valid") != 0:
            raise FinalizeError(
                "FIRST_CHECKPOINT_AUDIT_R1 reports non-train-only fitting membership"
            )
        if expected[cell_id] == "innovations" and item.get("prior_train_members") != 32:
            raise FinalizeError("FIRST_CHECKPOINT_AUDIT_R1 reports invalid innovation prior membership")
    observed_panel = audit.get("panel_sha256") or audit.get("panel_sha")
    if observed_panel is None:
        return None
    if observed_panel != panel.sha256:
        raise FinalizeError("first-checkpoint audit belongs to another frozen panel")
    return {"status": "passed", "path": str(path), "sha256": sha256_file(path), "audit": audit}


def _stop_owned_fits(panel: CohortPanel) -> list[dict[str, object]]:
    runtime = Path(str(panel.value["paths"]["runtime_root"]))
    stopped: list[dict[str, object]] = []
    for cell in cast(list[dict[str, object]], panel.value["cells"]):
        receipt_path = runtime / "receipts" / "fits" / f"{cell['cell_id']}.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            continue
        receipt = read_object(receipt_path)
        owner = receipt.get("owner")
        child = receipt.get("child")
        if (
            receipt.get("status") == "running"
            and isinstance(owner, dict)
            and owner.get("cohort_panel_sha256") == panel.sha256
            and owner.get("cell_id") == cell["cell_id"]
            and isinstance(child, dict)
        ):
            stopped.append(
                {
                    "cell_id": cell["cell_id"],
                    "identity_matched_and_stopped": terminate_owned_process(
                        child, grace_seconds=10.0
                    ),
                }
            )
    return stopped


def _wait_for_exports(
    panel: CohortPanel, out: Path, poll_seconds: int
) -> tuple[str, dict[str, object], dict[str, object] | None]:
    hard_deadline = _deadline(panel)
    progress_path = out / "PROGRESS.json"
    last_pipeline: dict[str, object] = {}
    prior_audit: dict[str, object] | None = None
    while time.time() < hard_deadline:
        pipeline_status, pipeline = _pipeline_snapshot(panel)
        if pipeline_status is not None:
            last_pipeline = pipeline
        if prior_audit is None:
            try:
                prior_audit = _prior_guard(panel)
            except FinalizeError as error:
                stopped = _stop_owned_fits(panel)
                _write_json(
                    out / "PRIOR_GUARD_FAILURE.json",
                    {
                        "schema": "wnn.w18-l4-prior-guard-failure.v1",
                        "error": str(error),
                        "stopped_owned_fits": stopped,
                    },
                )
                return (
                    "prior_guard_failed",
                    last_pipeline,
                    {"status": "failed", "reason": str(error)},
                )
        _write_json(
            progress_path,
            {
                "schema": "wnn.w18-l4-finalizer-progress.v1",
                "updated_utc": _now(),
                "pipeline_status": pipeline_status or "waiting_for_pipeline",
                "prior_guard": prior_audit,
                "hard_deadline_epoch": hard_deadline,
            },
        )
        if pipeline_status == "exports_complete_pending_grade" and prior_audit is not None:
            return pipeline_status, last_pipeline, prior_audit
        if pipeline_status == "failed":
            return "failed", last_pipeline, prior_audit
        time.sleep(min(poll_seconds, max(1, hard_deadline - time.time())))
    return "deadline_expired", last_pipeline, prior_audit


def _run_grader(
    panel: CohortPanel, grader: Path, output: Path, hard_deadline: float
) -> dict[str, object]:
    paths = cast(dict[str, object], panel.value["paths"])
    python = str(paths["python"])
    runtime = Path(str(paths["runtime_root"]))
    if grader.is_symlink() or not grader.is_file():
        raise FinalizeError("canonical grader is missing or is a symlink")
    before_sha = sha256_file(grader)
    if output.exists():
        raise FinalizeError("grade-r1 output already exists; preserve it and do not rerun")
    command = [python, str(grader), "--panel", str(panel.path), "--output", str(output)]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    receipt_path = runtime / "receipts" / "finalize-r1" / "grade.json"
    log_path = runtime / "logs" / "finalize-r1" / "grade.log"
    if receipt_path.exists() or log_path.exists():
        raise FinalizeError("grader process receipt/log already exists; refusing duplicate launch")
    owner = current_process_owner()
    owner.update(
        {
            "panel_sha256": panel.sha256,
            "grader_sha256": before_sha,
            "phase": "canonical_grade",
            "cpu_threads": 8,
        }
    )
    last_check = 0.0

    def progress() -> None:
        nonlocal last_check
        if time.time() >= hard_deadline:
            child = read_object(receipt_path).get("child")
            if isinstance(child, dict):
                terminate_owned_process(child, grace_seconds=10.0)
            raise TimeoutError("original cohort hard deadline reached during canonical grading")
        now = time.monotonic()
        if now - last_check >= 30.0:
            _budget(panel)
            last_check = now

    timeout = min(1200, max(1, int(hard_deadline - time.time())))
    code, timed_out, receipt = run_owned_command(
        command,
        cwd=grader.parents[1],
        env=env,
        log_path=log_path,
        receipt_path=receipt_path,
        owner=owner,
        stage=4,
        trial_number=1,
        mode="w18_l4_canonical_grade",
        gpu_key="cpu-finalizer",
        output_dir=output,
        timeout_seconds=timeout,
        progress=progress,
    )
    if code != 0 or timed_out:
        raise FinalizeError("canonical grader exited unsuccessfully; see its owned receipt/log")
    if sha256_file(grader) != before_sha:
        raise FinalizeError("canonical grader bytes changed during execution")
    report_path = output / "canonical" / "report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise FinalizeError("canonical grader produced no report.json")
    report = read_object(report_path)
    expected = ("complete", 1873, 3_528_273)
    observed = (report.get("status"), report.get("valid_sequences"), report.get("selected_rows"))
    if observed != expected:
        raise FinalizeError(f"canonical report identity/coverage differs: {observed}")
    return {
        "status": "complete",
        "command": command,
        "grader_sha256": before_sha,
        "owned_receipt": str(receipt_path),
        "owned_log": str(log_path),
        "receipt_status": receipt.get("status"),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "member_scores": len(cast(dict[str, object], report.get("member_scores", {}))),
        "ensemble_rows": len(cast(list[object], report.get("rows", []))),
    }


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def _add_file(
    entries: list[tuple[Path, str]], source: Path, name: str, *, required: bool = False
) -> None:
    if source.is_symlink():
        raise FinalizeError(f"archive input is a symlink: {name}")
    if not source.exists():
        if required:
            raise FinalizeError(f"required archive evidence is missing: {name}")
        return
    if (
        not source.is_file()
        or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or "\\" in name
    ):
        raise FinalizeError(f"unsafe archive input: {name}")
    entries.append((source, name))


def _write_archive(
    path: Path, entries: list[tuple[Path, str]], metadata: dict[str, object]
) -> dict[str, object]:
    if path.exists() or path.is_symlink() or len({name for _, name in entries}) != len(entries):
        raise FinalizeError(f"archive exists or has duplicate entries: {path.name}")
    partial = path.with_suffix(path.suffix + ".partial")
    manifest_entries: list[dict[str, object]] = []
    with zipfile.ZipFile(
        partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for source, destination in entries:
            before = source.stat()
            digest = hashlib.sha256()
            with source.open("rb") as input_stream, archive.open(destination, "w") as output_stream:
                while block := input_stream.read(1 << 20):
                    output_stream.write(block)
                    digest.update(block)
            after = source.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise FinalizeError(f"archive source changed while reading: {destination}")
            manifest_entries.append(
                {"path": destination, "bytes": after.st_size, "sha256": digest.hexdigest()}
            )
        archive.writestr(
            "MANIFEST.json",
            json.dumps({**metadata, "files": manifest_entries}, sort_keys=True, indent=2).encode(),
        )
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise FinalizeError(f"archive CRC check failed: {path.name}")
        for entry in manifest_entries:
            digest = hashlib.sha256(archive.read(entry["path"])).hexdigest()
            if digest != entry["sha256"]:
                raise FinalizeError(f"archive member SHA mismatch: {entry['path']}")
    os.replace(partial, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _hash_file(path),
        "members": len(entries),
    }


def _result_files(
    panel: CohortPanel, grade_dir: Path, out: Path, pipeline_status: str, grade_status: str
) -> tuple[Path, Path, Path]:
    matched_json = grade_dir / "matched_per_seed.json"
    matched_md = grade_dir / "MATCHED_PER_SEED.md"
    if matched_json.is_file() and not matched_json.is_symlink():
        comparisons = json.loads(matched_json.read_text(encoding="utf-8"))
    else:
        comparisons = [
            {
                "configuration": config,
                "seed": seed,
                "candidate_wp": None,
                "control_wp": None,
                "delta_wp_vs_control": None,
                "fixed_blend_wp": None,
                "fixed_blend_delta_wp_vs_tl3": None,
                "fixed_blend_delta_wp_vs_control_blend": None,
                "status": "unavailable",
            }
            for config in ("control", "innovations", "alpha50", "family01")
            for seed in (1001, 1002)
        ]
        matched_json = out / "matched_per_seed.json"
        _write_json(matched_json, {"status": "unavailable", "comparisons": comparisons})
    if matched_md.is_file() and not matched_md.is_symlink():
        table = matched_md.read_text(encoding="utf-8")
    else:
        table = "| configuration | seed | candidate WP | control WP | delta WP | fixed blend WP | status |\n|---|---:|---:|---:|---:|---:|---|\n"
        table += "".join(
            f"| {row['configuration']} | {row['seed']} | — | — | — | — | unavailable |\n"
            for row in comparisons
        )
    exposure = (
        "Official validation is repeatedly inspected development data, not independent confirmation. "
        "The 32-sequence train-eval proxy uses `all_after_warmup`; official valid uses `is_scored`, "
        "so their losses/scores are not a like-for-like gap. The predeclared fixed endpoint is epoch index 1."
    )
    text = (
        f"# W18 L4 matched results\n\nPipeline: `{pipeline_status}`. Grade: `{grade_status}`.\n\n"
        f"{exposure}\n\n"
        "Historical exposed-valid anchors: train-only W18 WP 0.6914165059769162; fixed TL3 WP 0.6923774245652725.\n\n"
        + table
    )
    results_md = out / "RESULTS.md"
    results_md.write_text(text, encoding="utf-8")
    if matched_json.parent != out:
        copied_json = out / "matched_per_seed.json"
        if not copied_json.exists():
            copied_json.write_bytes(matched_json.read_bytes())
        matched_json = copied_json
    return results_md, matched_json, matched_md if matched_md.is_file() else results_md


def _archive(
    panel: CohortPanel, grade_dir: Path, out: Path, results_md: Path, matched_json: Path
) -> list[dict[str, object]]:
    paths = cast(dict[str, object], panel.value["paths"])
    runtime = Path(str(paths["runtime_root"]))
    project = Path(str(paths["project_root"]))
    cells = cast(list[dict[str, object]], panel.value["cells"])
    checkpoint_entries: list[tuple[Path, str]] = []
    evidence_entries: list[tuple[Path, str]] = []
    for cell in cells:
        name = str(cell["run_name"])
        run_dir = project / "artifacts" / "runs" / name
        for filename in (
            "model.pt",
            "model_ep0.pt",
            "model_ep1.pt",
            "provenance.json",
            "history.json",
        ):
            _add_file(checkpoint_entries, run_dir / filename, f"{name}/{filename}", required=True)
        cell_paths_value = cell_paths(panel, cell)
        for phase in ("profile", "fit", "export"):
            for field in (f"{phase}_receipt", f"{phase}_log"):
                source = cell_paths_value.get(field)
                if source is not None:
                    _add_file(evidence_entries, source, f"receipts-logs/{phase}/{source.name}")
    for source, destination in (
        (Path(str(panel.path)), "control/FROZEN_PANEL_R3.json"),
        (
            Path(str(paths["control_root"])) / "WAVE_PIPELINE_R1.json",
            "control/WAVE_PIPELINE_R1.json",
        ),
        (runtime / "COHORT_STATE_R1.json", "control/COHORT_STATE_R1.json"),
        (runtime / "receipts" / "PROFILE_R1.json", "control/PROFILE_R1.json"),
        (runtime / "receipts" / "FIT_R1.json", "control/FIT_R1.json"),
        (runtime / "receipts" / "EXPORT_R1.json", "control/EXPORT_R1.json"),
        (runtime / "receipts" / "resources.jsonl", "receipts/resources.jsonl"),
    ):
        _add_file(evidence_entries, source, destination)
    admission = panel.value.get("admission_receipt")
    if isinstance(admission, str):
        _add_file(evidence_entries, Path(admission), "control/admission_receipt.json")
    state_path = runtime / "COHORT_STATE_R1.json"
    if state_path.is_file() and not state_path.is_symlink():
        state = read_object(state_path)
        profile_review = state.get("profile_review_receipt")
        if isinstance(profile_review, str):
            _add_file(evidence_entries, Path(profile_review), "control/profile_review_receipt.json")
    live = runtime / "live"
    for cell in cells:
        view = live / str(cell["run_name"])
        for filename in ("run_metadata.json", "monitor_events.jsonl", "monitor_evaluations.jsonl"):
            _add_file(evidence_entries, view / filename, f"live/{view.name}/{filename}")
    for source, destination in (
        (grade_dir / "manifest.json", "reports/grade-r1/manifest.json"),
        (grade_dir / "canonical" / "report.json", "reports/grade-r1/canonical/report.json"),
        (grade_dir / "matched_per_seed.json", "reports/grade-r1/matched_per_seed.json"),
        (grade_dir / "MATCHED_PER_SEED.md", "reports/grade-r1/MATCHED_PER_SEED.md"),
        (results_md, "reports/RESULTS.md"),
        (matched_json, "reports/matched_per_seed.json"),
    ):
        _add_file(evidence_entries, source, destination)
    _budget(panel)
    manifest = {
        "schema": "wnn.w18-l4-checkpoint-archive.v1",
        "panel_sha256": panel.sha256,
        "fixed_endpoint_epoch_index": 1,
        "development_exposed_valid": True,
    }
    checkpoint = _write_archive(
        out / "w18-l4-quality-20261104-checkpoints.zip", checkpoint_entries, manifest
    )
    _budget(panel)
    evidence = _write_archive(
        out / "w18-l4-quality-20261104-evidence.zip",
        evidence_entries,
        {
            "schema": "wnn.w18-l4-results-evidence.v1",
            "panel_sha256": panel.sha256,
            "contains_credentials": False,
        },
    )
    _budget(panel)
    return [checkpoint, evidence]


def _token_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino


def _read_token(path: Path, expected_identity: tuple[int, int] | None) -> str:
    if expected_identity is None or path.is_symlink() or not path.is_file():
        raise FinalizeError("GitHub token file is missing or a symlink")
    before = path.stat()
    if (before.st_dev, before.st_ino) != expected_identity:
        raise FinalizeError("GitHub token file identity changed before read")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise FinalizeError("GitHub token file permissions must be exactly 0600")
    token = path.read_text(encoding="utf-8").strip()
    after = path.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or not token or "\n" in token:
        raise FinalizeError("GitHub token file changed or is malformed")
    return token


def _unlink_token(path: Path, identity: tuple[int, int] | None) -> bool:
    if identity is None:
        return not path.exists()
    try:
        current = path.lstat()
    except FileNotFoundError:
        return True
    if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != identity:
        return False
    path.unlink()
    return True


class GitHub:
    """Small fixed-repository publisher using only GitHub's HTTPS API."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method: str, path: str, payload: object = None) -> object:
        if not path.startswith("/") or "\r" in path or "\n" in path:
            raise FinalizeError("GitHub API path is invalid")
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "WunderFunder-W18-L4-results-finalizer",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        try:
            with self.opener.open(request, timeout=45) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            raise FinalizeError(f"GitHub {method} request returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise FinalizeError(f"GitHub {method} request failed at {path}") from None
        if len(raw) > 4 * 1024 * 1024:
            raise FinalizeError("GitHub response exceeded the 4 MiB bound")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise FinalizeError("GitHub returned invalid JSON") from None

    def upload(self, release_id: int, path: Path) -> dict[str, object]:
        url = f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release_id}/assets?"
        url += urllib.parse.urlencode({"name": path.name})
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(path.stat().st_size),
            "User-Agent": "WunderFunder-W18-L4-results-finalizer",
        }
        with path.open("rb") as stream:
            request = urllib.request.Request(url, data=stream, headers=headers, method="POST")
            try:
                with self.opener.open(request, timeout=120) as response:
                    value = json.load(response)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
                raise FinalizeError(f"GitHub asset upload failed for {path.name}") from None
        if not isinstance(value, dict):
            raise FinalizeError("GitHub returned a malformed uploaded asset")
        return cast(dict[str, object], value)

def _publish_release(
    client: GitHub, assets: list[Path], target_commitish: str, body: str
) -> dict[str, object]:
    user = client.request("GET", "/user")
    repository = client.request("GET", f"/repos/{REPOSITORY}")
    if not isinstance(user, dict) or user.get("login") != LOGIN:
        raise FinalizeError("GitHub credential does not identify SemyonEpanov")
    permissions = repository.get("permissions") if isinstance(repository, dict) else None
    if (
        not isinstance(repository, dict)
        or repository.get("full_name", "").lower() != REPOSITORY.lower()
        or repository.get("private") is not True
        or not isinstance(permissions, dict)
        or permissions.get("push") is not True
    ):
        raise FinalizeError("GitHub repository is not the expected writable private repository")
    if len(target_commitish) != 40 or any(
        char not in "0123456789abcdef" for char in target_commitish.lower()
    ):
        raise FinalizeError("target commitish must be a full hexadecimal commit SHA")
    tag_path = f"/repos/{REPOSITORY}/releases/tags/{RELEASE_TAG}"
    try:
        release = client.request("GET", tag_path)
    except FinalizeError as error:
        if "HTTP 404" not in str(error):
            raise
        release = client.request(
            "POST",
            f"/repos/{REPOSITORY}/releases",
            {
                "tag_name": RELEASE_TAG,
                "target_commitish": target_commitish,
                "name": "W18 L4 matched results — 2026-11-04",
                "body": body,
                "draft": True,
                "prerelease": True,
            },
        )
    if not isinstance(release, dict) or type(release.get("id")) is not int:
        raise FinalizeError("GitHub release identity is malformed")
    release_id = cast(int, release["id"])
    if (
        release.get("tag_name") != RELEASE_TAG
        or release.get("target_commitish") != target_commitish
    ):
        raise FinalizeError("existing GitHub release tag or target commit differs")
    author = release.get("author")
    if isinstance(author, dict) and author.get("login") not in (None, LOGIN):
        raise FinalizeError("existing release belongs to another publisher")
    assets_path = f"/repos/{REPOSITORY}/releases/{release_id}/assets?per_page=100"
    inventory = client.request("GET", assets_path)
    if not isinstance(inventory, list):
        raise FinalizeError("GitHub release assets list is malformed")
    remote = {item.get("name"): item for item in inventory if isinstance(item, dict)}
    receipts = []
    for local in assets:
        digest = _hash_file(local)
        asset = remote.get(local.name)
        if asset is None:
            asset = client.upload(release_id, local)
        if (
            asset.get("name") != local.name
            or asset.get("state") != "uploaded"
            or asset.get("size") != local.stat().st_size
        ):
            raise FinalizeError(f"release asset metadata differs: {local.name}")
        declared_digest = asset.get("digest")
        if declared_digest != f"sha256:{digest}":
            raise FinalizeError(f"release asset SHA256 metadata differs: {local.name}")
        receipts.append({"name": local.name, "bytes": local.stat().st_size, "sha256": digest, "asset_id": asset.get("id"), "url": asset.get("browser_download_url"), "server_digest_verified": True})
    if release.get("draft") is True:
        release = client.request(
            "PATCH",
            f"/repos/{REPOSITORY}/releases/{release_id}",
            {"draft": False, "prerelease": True, "body": body},
        )
    if (
        not isinstance(release, dict)
        or release.get("prerelease") is not True
        or release.get("draft") is True
    ):
        raise FinalizeError("release did not become a published prerelease")
    return {
        "status": "published",
        "tag": RELEASE_TAG,
        "url": release.get("html_url"),
        "assets": receipts,
    }


def _post_comment(client: GitHub, text: str) -> dict[str, object]:
    endpoint = f"/repos/{REPOSITORY}/issues/{PR_NUMBER}/comments"
    comments = client.request("GET", endpoint + "?per_page=100&sort=created&direction=desc")
    if not isinstance(comments, list):
        raise FinalizeError("GitHub PR comment inventory is malformed")
    existing = next(
        (
            item
            for item in comments
            if isinstance(item, dict) and COMMENT_MARKER in str(item.get("body", ""))
        ),
        None,
    )
    body = text + f"\n\n{COMMENT_MARKER}"
    if existing is not None:
        user = existing.get("user")
        if not isinstance(user, dict) or user.get("login") != LOGIN:
            raise FinalizeError("idempotency marker exists under a different GitHub account")
        result = client.request(
            "PATCH", f"/repos/{REPOSITORY}/issues/comments/{existing['id']}", {"body": body}
        )
    else:
        result = client.request("POST", endpoint, {"body": body})
    if not isinstance(result, dict) or not isinstance(result.get("html_url"), str):
        raise FinalizeError("GitHub did not return a PR comment URL")
    author = result.get("user")
    if isinstance(author, dict) and author.get("login") != LOGIN:
        raise FinalizeError("posted PR comment author differs from SemyonEpanov")
    return {"status": "posted_or_updated", "url": result["html_url"]}


def _publish_and_comment(
    token: str, target_commitish: str, assets: list[Path], results_md: Path, grade_status: str
) -> tuple[dict[str, object], dict[str, object]]:
    client = GitHub(token)
    release_result: dict[str, object]
    try:
        release_result = _publish_release(
            client, assets, target_commitish, results_md.read_text(encoding="utf-8")
        )
    except (FinalizeError, OSError, ValueError) as error:
        release_result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    urls = [
        f"[{asset['name']}]({asset['url']})"
        for asset in cast(list[dict[str, object]], release_result.get("assets", []))
        if asset.get("url")
    ]
    release_url = release_result.get("url")
    suffix = (
        f"\n\nRelease: [{RELEASE_TAG}]({release_url})"
        if release_url
        else "\n\nRelease upload: failed; no verified release link is available."
    )
    text = (
        results_md.read_text(encoding="utf-8")
        + f"\n\nPublication status: `{release_result.get('status')}`. Grade status: `{grade_status}`."
    )
    if urls:
        text += "\n\nVerified assets: " + " · ".join(urls)
    text += suffix
    try:
        comment_result = _post_comment(client, text)
    except (FinalizeError, OSError, ValueError) as error:
        comment_result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    return release_result, comment_result


def _pipeline_body(
    pipeline_status: str,
    grade_status: str,
    report: dict[str, object],
    release: dict[str, object],
    assets: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": "wnn.w18-l4-finalization.v1",
        "finished_utc": _now(),
        "pipeline_status": pipeline_status,
        "grade_status": grade_status,
        "grade": report,
        "archives": assets,
        "release": release,
    }


def _finalize_work(args: argparse.Namespace, token_identity: tuple[int, int] | None) -> int:
    panel = load_panel(args.panel)
    runtime = Path(str(panel.value["paths"]["runtime_root"]))
    out = runtime / "reports" / "finalize-r1"
    if out.exists():
        prior = out / "COMPLETION.json"
        if prior.is_file() and not prior.is_symlink():
            LOGGER.info("finalizer already has a completion receipt; preserving it")
            return 0
        raise FinalizeError("finalizer output path already exists without completion receipt")
    out.mkdir(parents=True, mode=0o700)
    grade_dir = runtime / "reports" / "grade-r1"
    deadline = _deadline(panel)
    pipeline_status, pipeline, prior_audit = _wait_for_exports(panel, out, args.poll_seconds)
    grade_result: dict[str, object] = {"status": "not_run"}
    grade_status = "not_run"
    grader = args.grader.expanduser().resolve(strict=True)
    try:
        has_any_predictions = any(
            (
                Path(str(panel.value["paths"]["project_root"]))
                / "cache"
                / f"pred_{cell['run_name']}_valid.npy"
            ).is_file()
            for cell in cast(list[dict[str, object]], panel.value["cells"])
        )
        # Never grade a merely partial wave, even when one or more prediction files
        # happen to exist. The wave controller's terminal export receipt is the gate.
        should_grade = pipeline_status == "exports_complete_pending_grade"
        if prior_audit is None or prior_audit.get("status") != "passed":
            should_grade = False
        if should_grade:
            _budget(panel)
            grade_result = _run_grader(panel, grader, grade_dir, deadline)
            grade_status = str(grade_result["status"])
        elif pipeline_status == "failed":
            grade_status = "not_run_pipeline_failed_without_exports"
    except (FinalizeError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        grade_result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        grade_status = "failed"
    results_md, matched_json, matched_md = _result_files(
        panel, grade_dir, out, pipeline_status, grade_status
    )
    archives: list[dict[str, object]] = []
    assets: list[Path] = [results_md, matched_json]
    try:
        archives = _archive(panel, grade_dir, out, results_md, matched_json)
        assets = [Path(str(item["path"])) for item in archives] + [results_md, matched_json]
    except (FinalizeError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        LOGGER.error("Archive construction failed: %s", type(error).__name__)
    release_result: dict[str, object] = {"status": "not_attempted"}
    comment_result: dict[str, object] = {"status": "not_attempted"}
    try:
        token = _read_token(args.token_file, token_identity)
        release_allowed = (
            pipeline_status == "exports_complete_pending_grade"
            and grade_status == "complete"
            and prior_audit is not None
            and prior_audit.get("status") == "passed"
            and len(archives) == 2
        )
        if args.target_commitish and release_allowed:
            release_result, comment_result = _publish_and_comment(
                token, args.target_commitish, assets, results_md, grade_status
            )
        else:
            reasons = []
            if not args.target_commitish:
                reasons.append("missing --target-commitish")
            if pipeline_status != "exports_complete_pending_grade":
                reasons.append(f"pipeline status is {pipeline_status}")
            if grade_status != "complete":
                reasons.append(f"grade status is {grade_status}")
            if prior_audit is None or prior_audit.get("status") != "passed":
                reasons.append("first-checkpoint audit did not pass")
            if len(archives) != 2:
                reasons.append("both verified result archives were not built")
            reason = "; ".join(reasons)
            release_result = {"status": "not_published", "error": reason}
            try:
                comment_result = _post_comment(
                    GitHub(token),
                    results_md.read_text(encoding="utf-8")
                    + f"\n\nRelease upload: not published ({reason}).",
                )
            except (FinalizeError, OSError, ValueError) as error:
                comment_result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    except (FinalizeError, OSError) as error:
        release_result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        comment_result = {"status": "failed", "error": "GitHub credential unavailable or invalid"}
    completion = _pipeline_body(
        pipeline_status, grade_status, grade_result, release_result, archives
    )
    completion.update(
        {
            "panel_sha256": panel.sha256,
            "prior_guard": prior_audit,
            "pipeline_state": pipeline,
            "matched_report": str(matched_md),
            "comment": comment_result,
            "token_file_removed_or_absent": None,
            "budget": _budget(panel),
        }
    )
    _write_json(out / "COMPLETION.json", completion)
    LOGGER.info(
        "W18 finalizer completed: pipeline=%s grade=%s release=%s comment=%s",
        pipeline_status,
        grade_status,
        release_result.get("status"),
        comment_result.get("status"),
    )
    return (
        0
        if grade_status == "complete"
        and release_result.get("status") == "published"
        and comment_result.get("status") == "posted_or_updated"
        else 2
    )


def finalize(args: argparse.Namespace) -> int:
    """Run one finalization and inode-check token deletion on every exit path."""
    token_identity = _token_identity(args.token_file)
    result = 2
    error: BaseException | None = None
    try:
        result = _finalize_work(args, token_identity)
    except BaseException as caught:
        error = caught
        raise
    finally:
        removed = _unlink_token(args.token_file, token_identity)
        if not removed:
            LOGGER.error("Token cleanup refused because the file identity changed")
        try:
            panel = load_panel(args.panel)
            out = Path(str(panel.value["paths"]["runtime_root"])) / "reports" / "finalize-r1"
            receipt = out / "COMPLETION.json"
            if receipt.is_file() and not receipt.is_symlink():
                value = read_object(receipt)
                value["token_file_removed_or_absent"] = removed
                if error is not None:
                    value["finalizer_error"] = f"{type(error).__name__}: {error}"
                _write_json(receipt, value)
            elif error is not None and out.is_dir():
                failure = out / "FINALIZER_FAILURE.json"
                if not failure.exists():
                    _write_json(
                        failure,
                        {
                            "schema": "wnn.w18-l4-finalizer-failure.v1",
                            "error": f"{type(error).__name__}: {error}",
                            "token_file_removed_or_absent": removed,
                            "finished_utc": _now(),
                        },
                    )
        except (FinalizeError, OSError, RuntimeError, ValueError):
            pass
    return result


def _handle_sigterm(signum: int, frame: object) -> None:
    _ = frame
    raise FinalizeError(f"finalizer received signal {signum}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument(
        "--target-commitish", default=os.environ.get("W18_RESULTS_TARGET_COMMITISH")
    )
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.poll_seconds > 60:
        parser.error("--poll-seconds must be in [1, 60]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        return finalize(args)
    except (FinalizeError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        LOGGER.error("Finalizer stopped: %s: %s", type(error).__name__, error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
