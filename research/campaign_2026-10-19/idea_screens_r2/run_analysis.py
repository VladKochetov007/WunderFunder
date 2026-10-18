"""Create-only CLI for the fixed, source-bound R2 CPU screen."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from idea_screens_r1 import feature_views

from idea_screens_r2.ridge_cv import run as run_probe

PACKET = Path(__file__).resolve().parent
EXPECTED_SAMPLE_SHA256 = "0b03a4e2a6aab4eca36bdbb721d30e94b675c3425c0e0a7deca5ae562c2bbaa5"
EXPECTED_HASHES = {
    "r1_sample_data.py": "13a4acfeb62ace285184c8fe0f88ff7ba65a3969843eaa0e0f6175189325abc7",
    "r1_feature_views.py": "60d42c3c961001d9177c7d7d300fcd3472662cc3d024deaac74263e37fe75550",
    "canonical_data.py": "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00",
    "canonical_gmetric.py": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
}


def _digest(path: Path) -> str:
    """Hash a small pinned source or sample manifest file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Create one finite JSON evidence record without replacing old evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_manifest(sample: Path, manifest: dict[str, object]) -> dict[str, object]:
    """Verify exact imported source pins and summarize every selected sample hash."""
    paths = {
        "r1_sample_data.py": PACKET.parent / "idea_screens_r1" / "sample_data.py",
        "r1_feature_views.py": Path(feature_views.__file__).resolve(),
        "canonical_data.py": _module_path("wnn.data"),
        "canonical_gmetric.py": _module_path("wnn.gmetric"),
    }
    source_hashes = {name: _digest(path) for name, path in paths.items()}
    if source_hashes != EXPECTED_HASHES:
        raise ValueError(f"R1/canonical dependency source pins differ: {source_hashes}")
    sample_path = sample / "SAMPLE.json"
    sample_hash = _digest(sample_path)
    if sample_hash != EXPECTED_SAMPLE_SHA256:
        raise ValueError("R1 sample manifest differs from frozen R2 input")
    r1_files = {path.name: _digest(path) for path in sorted(PACKET.glob("*.py"))}
    sample_records = []
    for record in manifest["records"]:
        sample_records.append(
            {
                key: record[key]
                for key in ("role", "source", "group", "component", "seq_ix", "scored_rows")
            }
            | {"files": record["files"]}
        )
    return {
        "schema": "nf19.r2_source_manifest.v1",
        "canonical_source_snapshot": "source-a266b036",
        "sample_path": str(sample.resolve()),
        "sample_manifest_sha256": sample_hash,
        "sample_environment": {
            "sample_python": manifest["python"],
            "sample_numpy": manifest["numpy"],
            "r2_python": platform.python_version(),
            "r2_numpy": np.__version__,
        },
        "source_sha256": {
            **source_hashes,
            **{f"r2_{name}": value for name, value in r1_files.items()},
        },
        "sample_records": sample_records,
        "loader_note": "R1 load_sample verifies every selected NPY size and SHA256 before returning read-only memmaps",
        "metric_id": "global_wp_clip2_v1",
        "metric_source": str(paths["canonical_gmetric.py"]),
        "metric_source_sha256": source_hashes["canonical_gmetric.py"],
    }


def _module_path(name: str) -> Path:
    """Resolve a canonical module file without importing its data dependencies."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ImportError(f"cannot resolve pinned canonical source module {name}")
    return Path(spec.origin).resolve()


def main() -> None:
    """Load the exact R1 sample and run the declared R2 comparison once."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = time.monotonic()
    started = time.process_time()
    sample_hash = _digest(args.sample / "SAMPLE.json")
    if sample_hash != EXPECTED_SAMPLE_SHA256:
        raise ValueError("R2 must use the exact source-bound R1 sample")
    manifest = json.loads((args.sample / "SAMPLE.json").read_bytes())
    source_manifest = _source_manifest(args.sample, manifest)
    sample_data = importlib.import_module("idea_screens_r1.sample_data")
    sequences, loaded_manifest = sample_data.load_sample(args.sample)
    if loaded_manifest != manifest or _digest(args.sample / "SAMPLE.json") != sample_hash:
        raise ValueError("R1 sample manifest changed while its content hashes were checked")
    args.output.mkdir(parents=True, exist_ok=False)
    source_manifest_path = args.output / "SOURCE_MANIFEST.json"
    _write_json(source_manifest_path, source_manifest)
    _write_json(
        args.output / "START.json",
        {
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "sample_manifest_sha256": sample_hash,
            "source_manifest_sha256": _digest(source_manifest_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sequences": len(sequences),
        },
    )
    binding = {"sha256": sample_hash, "records": source_manifest["sample_records"]}
    result = run_probe(sequences, args.output / "RESULT.json", binding)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    _write_json(
        args.output / "DONE.json",
        {
            "utc": dt.datetime.now(dt.UTC).isoformat(),
            "elapsed_s": time.monotonic() - before,
            "user_cpu_s": usage.ru_utime,
            "system_cpu_s": usage.ru_stime,
            "maxrss_kib": usage.ru_maxrss,
            "result_sha256": _digest(args.output / "RESULT.json"),
            "source_manifest_sha256": _digest(source_manifest_path),
            "view_count": len(result["views"]),
            "cuda_initialized": "torch.cuda" in sys.modules,
            "process_cpu_s": time.process_time() - started,
        },
    )


if __name__ == "__main__":
    main()
