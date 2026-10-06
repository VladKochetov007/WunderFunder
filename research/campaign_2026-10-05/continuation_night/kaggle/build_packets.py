from __future__ import annotations

import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WIDTH = HERE.parent / "width"
W384 = ROOT / "research/campaign_2026-10-05/w384_v7"
E463 = ROOT / "research/campaign_2026-10-04/kaggle/jax_tpu_realdata_preflight_v1"

ENTRY = """from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import time
import zlib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logging.info(json.dumps({"event": "kernel_entry_before_payload_import", "time": time.time()}))
RUNTIME = Path("/kaggle/working/continuation_runtime")
RUNTIME.mkdir(exist_ok=False)
(RUNTIME / "entry.json").write_text(json.dumps({"time": time.time(), "pid": os.getpid(), "packet": PACKET}) + "\\n")
for name, encoded in SOURCES.items():
    data = zlib.decompress(base64.b85decode(encoded))
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError(f"embedded source mismatch: {name}")
    (RUNTIME / name).write_bytes(data)
sys.path.insert(0, str(RUNTIME))
from host_guard import supervised, write_json

allowed = sorted(os.sched_getaffinity(0))
if len(allowed) < CPU_COUNT:
    raise RuntimeError("insufficient CPU affinity")
os.sched_setaffinity(0, allowed[:CPU_COUNT])
environment = os.environ.copy()
environment.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    environment[key] = str(CPU_COUNT)
logging.info(json.dumps({"event": "payload_sources_verified", "packet": PACKET}))
try:
    result = supervised([sys.executable, str(RUNTIME / "payload.py")], Path("/kaggle/working"),
                        Path("/kaggle/working/transport_monitor"), environment,
                        timeout=TIMEOUT, rss_limit=RSS_BYTES, reserve=2 * 1024**3)
    logging.info(json.dumps({"event": "payload_finished", "result": result}))
except BaseException:
    logging.exception("private transport failed; startup and telemetry preserved")
    raise
finally:
    removed = []
    for name in CLEANUP_LINKS:
        path = Path(name)
        if path.is_symlink():
            path.unlink()
            removed.append(name)
    write_json(RUNTIME / "input_link_cleanup.json", {"removed": removed})
"""


def sha(data: bytes) -> str:
    """Return the source byte identity used inside and outside the kernel."""
    return hashlib.sha256(data).hexdigest()


def sha_path(path: Path) -> str:
    """Hash review evidence in bounded memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    """Refuse stale transport source rather than guessing another patch point."""
    if source.count(old) != 1:
        raise ValueError(f"expected one transport anchor: {old[:100]}")
    return source.replace(old, new, 1)


def gpu_payload(width: int) -> bytes:
    """Reuse the old hashed private source bundle; replace its paired launcher."""
    source = (W384 / "bundle/kernel/kernel.py").read_text()
    manifest_sha = sha((W384 / "bundle/bundle-manifest.json").read_bytes())
    source = replace_once(
        source,
        "    manifest = json.loads(matches[0].read_text())",
        "    manifest_bytes = matches[0].read_bytes()\n"
        f"    if hashlib.sha256(manifest_bytes).hexdigest() != {manifest_sha!r}:\n"
        '        raise ValueError("Original manifest identity mismatch")\n'
        "    manifest = json.loads(manifest_bytes)",
    )
    source = replace_once(
        source,
        'manifest["expanded_bytes"] + 2 * 1024**3',
        'sum(row["size"] for row in manifest["files"] if row["path"] not in manifest["external_mounts"]) + 2 * 1024**3',
    )
    source = replace_once(
        source,
        'command = [sys.executable, str(root / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4/phase_runner.py")]',
        f'command = [sys.executable, "/kaggle/working/continuation_runtime/phase.py", "--width", "{width}", "--manifest", str(matches[0])]',
    )
    source = replace_once(source, "timeout_seconds=25_180", "timeout_seconds=17_900")
    ast.parse(source)
    return source.encode()


def build_packet(
    kind: str,
    width: int | None = None,
    *,
    gpu_slug: str | None = None,
    recovery_attempt: int = 1,
) -> dict[str, object]:
    """Produce private upload-ready kernel files; never contact the provider."""
    if kind == "gpu":
        if width not in (256, 384):
            raise ValueError("width must be 256 or 384")
        if recovery_attempt not in (1, 2):
            raise ValueError("recovery_attempt must be one of the two reserved retries")
        if gpu_slug is not None and width != 384:
            raise ValueError("the isolated recovery packet is only authorized for width 384")
        slug = gpu_slug or f"wnn-night-w{width}-serial-20261005-v2"
        metadata = json.loads((W384 / "bundle/kernel/kernel-metadata.json").read_text())
        input_manifest_path = W384 / "bundle/bundle-manifest.json"
        input_payload_path = W384 / "bundle/kernel/kernel.py"
        source_files = {"payload.py": gpu_payload(width)}
        source_files.update(
            {
                name: (WIDTH / name).read_bytes()
                for name in ("bounded_loader.py", "derive_source.py", "phase.py")
            }
        )
        cpus, timeout, rss = 2, 18000, 10 * 1024**3
        provider_timeout = 18300
        manifest = json.loads((W384 / "bundle/bundle-manifest.json").read_text())
        cleanup = [f"/kaggle/working/experiment/{name}" for name in manifest["external_mounts"]]
    elif kind == "tpu":
        slug = "wnn-night-e463-tpu-20261005-v3"
        metadata = json.loads((E463 / "kernel_push/kernel-metadata.json").read_text())
        input_manifest_path = E463 / "upload_dataset/source-manifest.json"
        input_payload_path = E463 / "kernel_push/kernel.py"
        # The complete existing scientific source, HBM checks, raw masks and
        # checkpoint identities are byte-identical inside this new transport.
        source_files = {"payload.py": (E463 / "kernel_push/kernel.py").read_bytes()}
        cpus, timeout, rss = 8, 3900, 14 * 1024**3
        provider_timeout = 4200
        manifest = json.loads((E463 / "upload_dataset/source-manifest.json").read_text())
        cleanup = [
            f"/kaggle/working/jax_realdata_preflight_e463_tpu_v1/experiment/{name}"
            for name in manifest["external_inputs"]
        ]
    else:
        raise ValueError("unknown provider kind")
    source_files["host_guard.py"] = (HERE / "host_guard.py").read_bytes()
    hashes = {name: sha(data) for name, data in source_files.items()}
    encoded = {
        name: base64.b85encode(zlib.compress(data, level=9)).decode()
        for name, data in source_files.items()
    }
    constants = f"PACKET = {slug!r}\nSOURCES = {encoded!r}\nHASHES = {hashes!r}\nCPU_COUNT = {cpus}\nTIMEOUT = {timeout}\nRSS_BYTES = {rss}\nCLEANUP_LINKS = {cleanup!r}\n"
    entry = ENTRY.replace("import base64\n", constants + "\nimport base64\n", 1)
    ast.parse(entry)
    metadata.update(id=f"vladkochetov07/{slug}", title=slug, is_private=True)
    if metadata["enable_internet"] or metadata["competition_sources"]:
        raise ValueError("unexpected network or competition attachment")
    destination = HERE / "packets" / slug
    destination.mkdir(parents=True, exist_ok=False)
    kernel_bytes = entry.encode()
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
    (destination / "kernel.py").write_bytes(kernel_bytes)
    (destination / "kernel-metadata.json").write_bytes(metadata_bytes)
    receipt = {
        "status": "prepared_private_not_uploaded",
        "kind": kind,
        "width": width,
        "kernel_id": metadata["id"],
        "metadata": metadata,
        "embedded_source_sha256": hashes,
        "kernel_sha256": sha(kernel_bytes),
        "packet_files_sha256": {
            "kernel.py": sha(kernel_bytes),
            "kernel-metadata.json": sha(metadata_bytes),
        },
        "immutable_inputs": {
            "manifest": {
                "path": input_manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha(input_manifest_path.read_bytes()),
            },
            "source_payload": {
                "path": input_payload_path.relative_to(ROOT).as_posix(),
                "sha256": sha(input_payload_path.read_bytes()),
            },
        },
        "packet_path": destination.relative_to(ROOT).as_posix(),
        "push_command": f"./.venv/bin/kaggle kernels push -p {destination.relative_to(ROOT).as_posix()}",
        "provider_timeout_seconds": provider_timeout,
        "whole_payload_timeout_seconds": timeout,
        "scientific_workers_per_notebook": 1,
        "host_target_fraction": 0.70,
        "host_stop_fraction": 0.85,
        "recovery_attempt": recovery_attempt,
        "maximum_recovery_attempts": 2,
        "controller_review_and_reservation_required": True,
        "quota_or_budget_not_granted_by_this_packet": True,
        "tpu_E465_production_gate": "no E465 until actual-device E463/HBM and full-shape gates pass",
    }
    if gpu_slug == "wnn-night-w384-serial-20261006-v3":
        evidence = [
            "research/campaign_2026-10-05/continuation_night/kaggle/W384_V3_FIX.md",
            "research/campaign_2026-10-05/continuation_night/width/bounded_loader.py",
            "research/campaign_2026-10-05/continuation_night/width/derive_source.py",
            "research/campaign_2026-10-05/continuation_night/width/phase.py",
            "research/campaign_2026-10-05/continuation_night/width/test_transport.py",
            "research/campaign_2026-10-05/continuation_night/width/run_cpu_preflight.py",
            "research/campaign_2026-10-05/continuation_night/width/SOURCE_RECEIPT.json",
            "research/campaign_2026-10-05/continuation_night/width/transport_cpu_tests.xml",
            "research/campaign_2026-10-05/continuation_night/width/legacy_cpu_tests.xml",
            "research/campaign_2026-10-05/continuation_night/kaggle/build_packets.py",
            "research/campaign_2026-10-05/continuation_night/kaggle/diagnostics/wnn-night-w384-serial-20261005-v2/width_384_serial_v1/source_derivation.json",
            "research/campaign_2026-10-05/continuation_night/kaggle/diagnostics/wnn-night-w384-serial-20261005-v2/width_384_serial_v1/cuda_preflight/result.json",
        ]
        receipt["mitigation_evidence"] = [
            {"path": path, "sha256": sha_path(ROOT / path)} for path in evidence
        ]
        receipt["preflight_failure_class"] = "legacy_fixture_length_contract_mismatch"
        receipt["production_sequence_length_rows"] = 20_000
        receipt["science_argv_or_data_split_changed"] = False
        receipt["provider_action_launched"] = False
    elif gpu_slug == "wnn-night-w384-serial-20261006-v3r1":
        evidence = [
            "research/campaign_2026-10-05/continuation_night/kaggle/W384_V3_NO_GO.md",
            "research/campaign_2026-10-05/continuation_night/kaggle/W384_V3R1_FIX.md",
            "research/campaign_2026-10-05/continuation_night/width/bounded_loader.py",
            "research/campaign_2026-10-05/continuation_night/width/derive_source.py",
            "research/campaign_2026-10-05/continuation_night/width/phase.py",
            "research/campaign_2026-10-05/continuation_night/width/test_transport.py",
            "research/campaign_2026-10-05/continuation_night/width/run_cpu_preflight.py",
            "research/campaign_2026-10-05/continuation_night/width/SOURCE_RECEIPT_V3R1.json",
            "research/campaign_2026-10-05/continuation_night/width/transport_cpu_tests_v3r1.xml",
            "research/campaign_2026-10-05/continuation_night/width/legacy_cpu_tests.xml",
            "research/campaign_2026-10-05/continuation_night/kaggle/build_packets.py",
            "research/campaign_2026-10-05/continuation_night/kaggle/packets/wnn-night-w384-serial-20261006-v3/review_packet.json",
            "research/campaign_2026-10-05/continuation_night/kaggle/diagnostics/wnn-night-w384-serial-20261005-v2/width_384_serial_v1/source_derivation.json",
            "research/campaign_2026-10-05/continuation_night/kaggle/diagnostics/wnn-night-w384-serial-20261005-v2/width_384_serial_v1/cuda_preflight/result.json",
        ]
        receipt["mitigation_evidence"] = [
            {"path": path, "sha256": sha_path(ROOT / path)} for path in evidence
        ]
        receipt["preflight_failure_class"] = "legacy_fixture_length_contract_mismatch"
        receipt["independent_review_finding"] = "v3 artifact manifest used stale output glob and omitted checkpoint/provenance/history"
        receipt["production_sequence_length_rows"] = 20_000
        receipt["science_argv_or_data_split_changed"] = False
        receipt["provider_action_launched"] = False
        receipt["revision_of"] = "wnn-night-w384-serial-20261006-v3"
    review_bytes = (json.dumps(receipt, indent=2) + "\n").encode()
    (destination / "review_packet.json").write_bytes(review_bytes)
    receipt["review_packet_sha256"] = sha(review_bytes)
    return receipt


def build_w384_recovery_v3() -> dict[str, object]:
    """Build only today's isolated W384 recovery; preserve all prior packets."""
    return build_packet(
        "gpu",
        384,
        gpu_slug="wnn-night-w384-serial-20261006-v3",
        recovery_attempt=2,
    )


def build_w384_recovery_v3_revision1() -> dict[str, object]:
    """Build a new immutable packet revision without consuming another retry."""
    return build_packet(
        "gpu",
        384,
        gpu_slug="wnn-night-w384-serial-20261006-v3r1",
        recovery_attempt=2,
    )


def main() -> None:
    """Build only the revised W384 packet without touching earlier packets."""
    print(json.dumps(build_w384_recovery_v3_revision1(), indent=2))


if __name__ == "__main__":
    main()
