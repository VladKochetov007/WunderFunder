from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile

SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")


def digest(path: Path) -> str:
    """Hash a bounded source or sample file without loading it all at once."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write a create-only evidence record."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def select_membership(split: dict) -> list[dict]:
    """Choose complete eligible component memberships without looking at arrays."""
    rng = random.Random(19019)
    records = []
    all_components = {}
    for field in ("fit", "evaluation"):
        all_components[field] = set()
        for source, groups in split[field].items():
            allowed = set(groups)
            if len(allowed) != len(groups):
                raise ValueError("duplicate group in split membership")
            found = [item for item in split["sequences"][source] if item["row_group"] in allowed]
            if len(found) != len(allowed) or len({item["row_group"] for item in found}) != len(
                allowed
            ):
                raise ValueError("missing or duplicate native group metadata")
            all_components[field].update(item["component"] for item in found)
    if all_components["fit"] & all_components["evaluation"]:
        raise ValueError("full fitting/evaluation component overlap before panel selection")
    for role, field, cap in (("fit", "fit", 32), ("eval", "evaluation", 16)):
        components: dict[int, list[dict]] = defaultdict(list)
        for source, groups in split[field].items():
            allowed = set(groups)
            for sequence in split["sequences"][source]:
                if sequence["row_group"] in allowed:
                    components[sequence["component"]].append(
                        {**sequence, "source": source, "group": sequence["row_group"], "role": role}
                    )
        order = sorted(components)
        rng.shuffle(order)
        used = 0
        for component in order:
            members = components[component]
            if used + len(members) <= cap:
                records.extend(sorted(members, key=lambda item: (item["source"], item["group"])))
                used += len(members)
            if used == cap:
                break
        if used < 8:
            raise ValueError(f"insufficient component-complete membership for {role}: {used}")
    fit = {item["component"] for item in records if item["role"] == "fit"}
    evaluation = {item["component"] for item in records if item["role"] == "eval"}
    if fit & evaluation:
        raise ValueError("fit/evaluation component overlap")
    return records


def verify_sources(split: dict) -> dict:
    """Bind registered identities with current size/footer checks, not a full rescan."""
    result = {}
    for source in ("train", "valid"):
        relative = f"datasets/{source}.parquet"
        path = BASE / "data" / relative
        expected = split["sources"][relative]
        before = path.stat()
        if before.st_size != expected["bytes"] or before.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"parquet stat changed: {source}")
        with path.open("rb") as stream:
            stream.seek(-expected["footer_bytes"] - 8, 2)
            footer = stream.read(expected["footer_bytes"] + 8)
        footer_sha = hashlib.sha256(footer).hexdigest()
        if footer_sha != expected["footer_sha256"]:
            raise ValueError(f"parquet footer changed: {source}")
        result[source] = {
            "path": str(path),
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "footer_sha256": footer_sha,
            "full_payload_rehashed": False,
            "registered_full_sha256": {
                "train": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
                "valid": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
            }[source],
        }
    return result


def materialize(output: Path) -> None:
    """Read each chosen whole sequence once into a private source-bound sample."""
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    scratch = BASE / "research_tmp"
    if not output.resolve().is_relative_to(scratch.resolve()):
        raise ValueError("sample output must be under physical approved scratch")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if output.resolve().is_relative_to(Path(fields[1])):
            mounts.append((len(fields[1]), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("sample output storage is RAM-backed or unverified")
    if shutil.disk_usage(scratch).free < 1_000_000_000:
        raise ValueError("insufficient physical sample disk space")
    output.mkdir()
    split_path = BASE / "splits/fold0.json"
    if digest(split_path) != SPLIT_SHA:
        raise ValueError("canonical F0 split differs")
    split = json.loads(split_path.read_bytes())
    membership = select_membership(split)
    sources = verify_sources(split)
    write_json(
        output / "MEMBERSHIP.json",
        {"seed": 19019, "split_sha256": SPLIT_SHA, "records": membership},
    )
    readers = {source: SequenceFile(item["path"]) for source, item in sources.items()}
    for item in membership:
        sequence = readers[item["source"]].read(item["group"])
        if sequence.seq_ix != item["seq_ix"] or sequence.features.shape != (20_000, 112):
            raise ValueError("native sequence identity/shape differs")
        if sequence.targets is None or not np.isfinite(sequence.features).all():
            raise ValueError("missing targets or nonfinite inputs")
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
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
    for item in sources.values():
        after = Path(item["path"]).stat()
        if after.st_size != item["bytes"] or after.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("parquet source stat changed during bounded read")
    write_json(
        output / "SAMPLE.json",
        {
            "schema": "nf19.sample.v1",
            "split_sha256": SPLIT_SHA,
            "selection": "target-blind shuffled components, greedy whole eligible membership under caps",
            "selection_bias": "cost cap changes component inclusion probabilities; exposed discovery panel",
            "sources": sources,
            "records": membership,
            "elapsed_s": time.monotonic() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
    )


def load_sample(directory: Path) -> tuple[list[dict], dict]:
    """Verify sample bytes and expose read-only memmaps to sequential analyses."""
    manifest = json.loads((directory / "SAMPLE.json").read_bytes())
    sequences = []
    for record in manifest["records"]:
        item = {key: record[key] for key in ("role", "source", "group", "component", "seq_ix")}
        for key, binding in record["files"].items():
            path = directory / binding["name"]
            if path.stat().st_size != binding["bytes"] or digest(path) != binding["sha256"]:
                raise ValueError(f"sample binding changed: {path.name}")
            item[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        sequences.append(item)
    return sequences, manifest


def main() -> None:
    """Materialize only the frozen small supplied-data panel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.output)


if __name__ == "__main__":
    main()
