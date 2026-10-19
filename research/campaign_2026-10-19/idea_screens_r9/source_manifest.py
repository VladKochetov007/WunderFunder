"""Hash gates for R9's fixed design and staged transitive import closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DESIGN_SHA256 = "5fa6f4a6fda34402232444cc670e1f36263bc359059fb66e646fa22fd4bcee70"
DESIGN_REVIEW_SHA256 = "dd07f88d59414e0d1fa581d5376118f455daa9072244a001af6ff31073e4334b"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
CHECKPOINT_SHA256 = "78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f"
CHECKPOINT_BYTES = 8_905_959
R6_PREPARATION_SHA256 = "c291687c93689463133f5e4109e7adef116dc79bca8aec08b7ec30d29a28505d"
SOURCE_SNAPSHOT = "a266b036"
TORCH_VERSION = "2.14.0+cu130"
NUMPY_VERSION = "2.2.6"
R6_MANIFEST_KEY = "idea_screens_r6/PREPARATION_MANIFEST.json"
R3_SAMPLE_FIXTURE_KEY = "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json"
R3_MEMBERSHIP_FIXTURE_KEY = (
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json"
)
RUNTIME_IMPORT_PATHS = {
    "idea_screens_r1.sample_data": "idea_screens_r1/sample_data.py",
    "idea_screens_r6.execution_host": "idea_screens_r6/execution_host.py",
    "idea_screens_r6.model_contract": "idea_screens_r6/model_contract.py",
    "idea_screens_r6.sample_panel": "idea_screens_r6/sample_panel.py",
    "idea_screens_r9": "idea_screens_r9/__init__.py",
    "idea_screens_r9.execution": "idea_screens_r9/execution.py",
    "idea_screens_r9.history_math": "idea_screens_r9/history_math.py",
    "idea_screens_r9.history_probe": "idea_screens_r9/history_probe.py",
    "idea_screens_r9.run_probe": "idea_screens_r9/run_probe.py",
    "idea_screens_r9.source_manifest": "idea_screens_r9/source_manifest.py",
    "idea_screens_r9.state_replay": "idea_screens_r9/state_replay.py",
    "idea_screens_r9.window_probe": "idea_screens_r9/window_probe.py",
}
REQUIRED_FILES = {
    "idea_screens_r9/DESIGN.md",
    "idea_screens_r9/DESIGN_REVIEW.json",
    "idea_screens_r9/__init__.py",
    "idea_screens_r9/execution.py",
    "idea_screens_r9/history_math.py",
    "idea_screens_r9/history_probe.py",
    "idea_screens_r9/run_probe.py",
    "idea_screens_r9/source_manifest.py",
    "idea_screens_r9/state_replay.py",
    "idea_screens_r9/test_history_probe.py",
    "idea_screens_r9/window_probe.py",
    R6_MANIFEST_KEY,
    "idea_screens_r6/execution_host.py",
    "idea_screens_r6/model_contract.py",
    "idea_screens_r6/sample_panel.py",
    "idea_screens_r1/sample_data.py",
    R3_SAMPLE_FIXTURE_KEY,
    R3_MEMBERSHIP_FIXTURE_KEY,
}
R6_RUNTIME_KEYS = (
    "idea_screens_r6/execution_host.py",
    "idea_screens_r6/model_contract.py",
    "idea_screens_r6/sample_panel.py",
    "idea_screens_r1/sample_data.py",
)


def sha256_file(path: Path) -> str:
    """Hash one source or payload file in bounded streaming blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(key: str, bundle_root: Path, source_root: Path) -> Path:
    """Resolve a safe staged campaign or canonical source path."""
    relative_text = key.partition(":")[2] if key.startswith("canonical:") else key
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe R9 manifest path: {key}")
    if key.startswith("canonical:"):
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid canonical R9 source path: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2 or relative.parts[0] not in {
        "idea_screens_r1",
        "idea_screens_r3",
        "idea_screens_r6",
        "idea_screens_r9",
    }:
        raise ValueError(f"invalid staged R9 source path: {key}")
    return bundle_root / relative


def verify_manifest(
    manifest_path: Path,
    expected_sha256: str,
    bundle_root: Path,
    source_root: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    """Verify all declared bytes and exact frozen R6/R3 scientific identities."""
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("R9 preparation manifest SHA256 differs")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("R9 preparation manifest has no source pins")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid R9 source pin entry")
        actual = sha256_file(source_path(key, bundle_root, source_root))
        if actual != expected:
            raise ValueError(f"R9 pinned source differs: {key}")
        checked[key] = actual
    missing = REQUIRED_FILES - checked.keys()
    if missing:
        raise ValueError(f"R9 preparation manifest omits required sources: {sorted(missing)}")
    if checked.get("idea_screens_r9/DESIGN.md") != DESIGN_SHA256:
        raise ValueError("R9 source is not bound to the frozen design")
    if checked.get("idea_screens_r9/DESIGN_REVIEW.json") != DESIGN_REVIEW_SHA256:
        raise ValueError("R9 design review pin differs")
    if checked.get(R6_MANIFEST_KEY) != R6_PREPARATION_SHA256:
        raise ValueError("frozen R6 preparation manifest identity differs")
    if checked.get(R3_SAMPLE_FIXTURE_KEY) != SAMPLE_SHA256:
        raise ValueError("R3 producer sample fixture identity differs")
    if checked.get(R3_MEMBERSHIP_FIXTURE_KEY) != MEMBERSHIP_SHA256:
        raise ValueError("R3 producer membership fixture identity differs")
    for key in RUNTIME_IMPORT_PATHS.values():
        if key not in checked:
            raise ValueError(f"R9 source manifest omits runtime import source: {key}")

    r6_manifest = json.loads(source_path(R6_MANIFEST_KEY, bundle_root, source_root).read_bytes())
    r6_files = r6_manifest.get("files")
    if not isinstance(r6_files, dict) or r6_files.get("canonical:wnn/training/forward.py") is None:
        raise ValueError("frozen R6 manifest has no canonical runtime pins")
    for key in R6_RUNTIME_KEYS:
        if checked.get(key) != r6_files.get(key):
            raise ValueError(f"R9 does not preserve frozen R6 helper identity: {key}")
    for key, digest in r6_files.items():
        if key.startswith("canonical:") and checked.get(key) != digest:
            raise ValueError(f"R9 canonical pin differs from frozen R6 source: {key}")
    fixed = manifest.get("fixed_inputs", {})
    if (
        fixed.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or fixed.get("sample_sha256") != SAMPLE_SHA256
        or fixed.get("membership_sha256") != MEMBERSHIP_SHA256
        or fixed.get("design_sha256") != DESIGN_SHA256
    ):
        raise ValueError("R9 fixed-input identities differ from the reviewed design")
    return manifest, checked
