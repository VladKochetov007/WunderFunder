from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    """Hash a file with bounded buffers."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        offset = 0
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
            # Content verification itself must not refill tens of GB of cache.
            # This is advice on the already-read, exact read-only FD only.
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)
    return value.hexdigest()


def derive_trainer(source: str) -> str:
    """Derive a source-bound transport variant without runtime monkey-patching."""
    old_import = "    SequenceFile,\n"
    old_boundary = "        return result\n\n\ndef thin_rows"
    if source.count(old_import) != 1 or source.count(old_boundary) != 1:
        raise ValueError("reviewed trainer loader patch points changed")
    source = source.replace(old_import, "", 1)
    source = source.replace(
        "from wnn.features import",
        "from wnn.bounded_loader import (BoundedSequenceFile as SequenceFile, "
        "release_auxiliary_mappings)\nfrom wnn.features import",
        1,
    )
    source = source.replace(
        old_boundary,
        "        release_auxiliary_mappings(self)\n" + old_boundary,
        1,
    )
    ast.parse(source)
    return source


def require_production_sequence_length(source: str) -> None:
    """Require the immutable production data contract used by private packets."""
    module = ast.parse(source)
    matches = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SEQUENCE_LENGTH"
            for target in node.targets
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SEQUENCE_LENGTH"
        ):
            value = node.value
        else:
            continue
        try:
            matches.append(ast.literal_eval(value))
        except (TypeError, ValueError):
            matches.append(None)
    if matches != [20_000]:
        raise ValueError("production data contract must pin SEQUENCE_LENGTH = 20_000")


def stage(root: Path, original_manifest: Path, runtime: Path) -> dict[str, object]:
    """Verify source bytes before producing the isolated loader-only variant."""
    manifest = json.loads(original_manifest.read_text())
    descriptors = {row["path"]: row for row in manifest["files"]}
    data_source = root / "src/wnn/data.py"
    data_descriptor = descriptors.get("src/wnn/data.py")
    if data_descriptor is None:
        raise ValueError("source manifest does not bind src/wnn/data.py")
    if digest(data_source) != data_descriptor["sha256"] or (
        data_source.stat().st_size != data_descriptor["size"]
    ):
        raise ValueError("data module differs from the pinned production source")
    require_production_sequence_length(data_source.read_text())
    trainer = root / "scripts/train_rnn.py"
    expected = descriptors["scripts/train_rnn.py"]
    original_sha = digest(trainer)
    if original_sha != expected["sha256"] or trainer.stat().st_size != expected["size"]:
        raise ValueError("original trainer differs from the pinned source manifest")
    original = trainer.read_text()
    (root / "scripts/train_rnn.original.py").write_text(original)
    trainer.write_text(derive_trainer(original))
    loader = root / "src/wnn/bounded_loader.py"
    loader.write_bytes((runtime / "bounded_loader.py").read_bytes())
    return {
        "original_trainer_sha256": original_sha,
        "derived_trainer_sha256": digest(trainer),
        "bounded_loader_sha256": digest(loader),
        "transformation": "SequenceFile import and post-copy mmap residency release",
        "science_arguments_changed": False,
        "sampler_changed": False,
        "loader_workers": 0,
    }
