from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

DESIGN_SHA256 = "f724f1d44ab872dc5b27952ba04f7e0762faacad6f5c5cdf71e8393b3ca6a5c8"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
CHECKPOINT_SHA256 = "78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f"
SOURCE_REVISION = "a266b036"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks without deserializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve a manifest key in the portable B source layout."""
    if key.startswith("canonical:"):
        relative = Path(key.partition(":")[2])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid canonical source key {key}")
        return source_root / "src" / relative
    relative = Path(key)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
        raise ValueError(f"invalid staged source key {key}")
    return support_root / relative


def _verify_manifest(
    manifest_path: Path, expected_sha256: str, support_root: Path, source_root: Path
) -> tuple[dict[str, object], dict[str, str]]:
    """Verify all staged code, tests, design and canonical source before imports."""
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("preparation manifest identity differs")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("preparation manifest lacks frozen source pins")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid preparation source pin")
        actual = sha256_file(_source_path(key, support_root, source_root))
        if actual != expected:
            raise ValueError(f"frozen source differs: {key}")
        checked[key] = actual
    inputs = manifest.get("fixed_inputs")
    if not isinstance(inputs, dict) or inputs.get("design_sha256") != DESIGN_SHA256:
        raise ValueError("preparation manifest is not bound to reviewed R6 design")
    if (
        inputs.get("sample_sha256") != SAMPLE_SHA256
        or inputs.get("membership_sha256") != MEMBERSHIP_SHA256
    ):
        raise ValueError("preparation manifest has unexpected R3 sample pins")
    if inputs.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("preparation manifest has unexpected R01 checkpoint pin")
    return manifest, checked


def _scratch_root(sample_path: Path) -> tuple[Path, Path, Path]:
    """Resolve B, its physical scratch directory and frozen canonical source tree."""
    sample_path = sample_path.resolve()
    if sample_path.parent.name != "research_tmp":
        raise ValueError("sample must live directly below physical B/research_tmp")
    base = sample_path.parent.parent
    scratch = base / "research_tmp"
    source_root = base / f"source-{SOURCE_REVISION}"
    if not source_root.is_dir():
        raise FileNotFoundError(f"canonical source snapshot is absent: {source_root}")
    return base, scratch, source_root


def _parse_args() -> argparse.Namespace:
    """Parse the fixed R6 CLI with every scientific input identity required."""
    parser = argparse.ArgumentParser(description="No-update R6 per-target gradient diagnostic")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Validate every static identity, then import and invoke the pinned executor."""
    sample_path = args.sample.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    base, scratch, source_root = _scratch_root(sample_path)
    package_dir = Path(__file__).resolve().parent
    support_root = package_dir.parent
    if (
        args.expected_sample_sha256 != SAMPLE_SHA256
        or args.expected_checkpoint_sha256 != CHECKPOINT_SHA256
    ):
        raise ValueError("CLI sample/checkpoint pins differ from frozen R6 identities")
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    _, pins = _verify_manifest(
        manifest_path,
        args.expected_preparation_manifest_sha256,
        support_root,
        source_root,
    )
    if sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("accepted R3 sample metadata differs")
    if sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("accepted R3 membership metadata differs")
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("R01 epoch5 checkpoint full SHA differs")
    executor = importlib.import_module("probe_execution")
    executor.execute(
        sample_path=sample_path,
        checkpoint_path=checkpoint_path,
        output=output,
        base=base,
        scratch=scratch,
        source_root=source_root,
        package_dir=package_dir,
        pins=pins,
        preparation_sha256=args.expected_preparation_manifest_sha256,
        identities={
            "design": DESIGN_SHA256,
            "sample": SAMPLE_SHA256,
            "membership": MEMBERSHIP_SHA256,
            "checkpoint": CHECKPOINT_SHA256,
            "source_revision": SOURCE_REVISION,
        },
    )


def main() -> None:
    """Run the fixed bounded CPU diagnostic from explicit hash-pinned paths."""
    _run(_parse_args())


if __name__ == "__main__":
    main()
