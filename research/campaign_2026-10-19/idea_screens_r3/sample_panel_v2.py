"""Component-first, source-bound R3 sample materialization."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from panel_membership_v2 import R3_DIR, SPLIT_SHA, select_membership

R1_SAMPLE_DATA = R3_DIR.parent / "idea_screens_r1" / "sample_data.py"


def _r1_helpers() -> ModuleType:
    """Load the pinned R1 sampler helpers and their canonical SequenceFile reader."""
    name = "nf19_idea_screens_r1_sample_data"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, R1_SAMPLE_DATA)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load R1 sampler helpers from {R1_SAMPLE_DATA}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError:
        sys.modules.pop(name, None)
        raise
    return module


def _r3_output_path(output: Path, base: Path) -> None:
    """Reject overwrite, non-scratch paths, RAM mounts, and low disk headroom."""
    scratch = base / "research_tmp"
    if output.exists():
        raise FileExistsError(output)
    if not output.resolve().is_relative_to(scratch.resolve()) or not scratch.is_dir():
        raise ValueError("sample output must be under existing physical research_tmp")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if output.resolve().is_relative_to(Path(fields[1])):
            mounts.append((len(fields[1]), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("sample output storage is RAM-backed or unverified")
    if shutil.disk_usage(scratch).free < 4_000_000_000:
        raise ValueError("R3 sample requires at least4GB free in physical research_tmp")


def materialize(output: Path) -> None:
    """Create membership before reading96 selected full sequences into an R1-compatible sample."""
    started = time.monotonic()
    helpers = _r1_helpers()
    base = helpers.BASE
    _r3_output_path(output, base)
    split_path = base / "splits" / "fold0.json"
    if helpers.digest(split_path) != SPLIT_SHA or helpers.SPLIT_SHA != SPLIT_SHA:
        raise ValueError("canonical F0 split differs")
    split = json.loads(split_path.read_bytes())
    panel_metadata = json.loads((R3_DIR / "PANEL_METADATA.json").read_bytes())
    membership = select_membership(split, panel_metadata)
    sources = helpers.verify_sources(split)
    output.mkdir()
    helpers.write_json(output / "MEMBERSHIP.json", membership)
    readers = {source: helpers.SequenceFile(item["path"]) for source, item in sources.items()}
    records = membership["records"]
    for item in records:
        sequence = readers[item["source"]].read(item["group"])
        if sequence.seq_ix != item["seq_ix"] or sequence.features.shape != (20_000, 112):
            raise ValueError("native sequence identity/shape differs")
        if sequence.targets is None or not np.isfinite(sequence.features).all():
            raise ValueError("missing targets or nonfinite selected inputs")
        mask = sequence.scored
        if item["role"] == "eval" and mask is None:
            raise ValueError("evaluation original scored mask unavailable")
        if mask is None:
            mask = np.arange(20_000) >= 99
            item["mask_origin"] = "fit_rows_after_99_warmup"
        else:
            mask = mask.copy()
            if mask[:99].any():
                raise ValueError("source scored mask violates documented 99-row warmup")
            item["mask_origin"] = "provided_is_scored_after_99_warmup"
        if not np.isfinite(sequence.targets[mask]).all():
            raise ValueError("nonfinite selected labels")
        item["scored_rows"] = int(mask.sum())
        item["files"] = {}
        for key, array in (("x", sequence.features), ("y", sequence.targets), ("scored", mask)):
            path = output / f"{item['source']}_{item['group']}_{key}.npy"
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            item["files"][key] = {
                "name": path.name,
                "sha256": helpers.digest(path),
                "bytes": path.stat().st_size,
            }
    for item in sources.values():
        after = Path(item["path"]).stat()
        if after.st_size != item["bytes"] or after.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("parquet source stat changed during bounded read")
    reader_module = sys.modules[helpers.SequenceFile.__module__]
    sample_binding = helpers.digest(R1_SAMPLE_DATA)
    reader_path = Path(reader_module.__file__).resolve()
    helpers.write_json(
        output / "SAMPLE.json",
        {
            "schema": "idea_screens_r3.sample.v2",
            "split_sha256": SPLIT_SHA,
            "selection": membership["selection"],
            "metadata_counts": membership["metadata_counts"],
            "sources": sources,
            "records": records,
            "membership_file": "MEMBERSHIP.json",
            "import_closure": {
                "r1_sample_data_path": str(R1_SAMPLE_DATA),
                "r1_sample_data_sha256": sample_binding,
                "panel_metadata_path": str(R3_DIR / "PANEL_METADATA.json"),
                "panel_metadata_sha256": helpers.digest(R3_DIR / "PANEL_METADATA.json"),
                "canonical_reader": "wnn.data.SequenceFile",
                "canonical_reader_path": str(reader_path),
                "canonical_reader_sha256": helpers.digest(reader_path),
                "sample_panel_v2_sha256": helpers.digest(Path(__file__).resolve()),
                "panel_membership_v2_sha256": helpers.digest(R3_DIR / "panel_membership_v2.py"),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": helpers.importlib.metadata.version("pyarrow"),
            },
            "elapsed_s": time.monotonic() - started,
        },
    )


def main() -> None:
    """Materialize only the fixed R3 panel under approved scratch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    materialize(parser.parse_args().output)


if __name__ == "__main__":
    main()
