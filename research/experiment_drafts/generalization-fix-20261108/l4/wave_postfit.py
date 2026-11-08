from __future__ import annotations

from pathlib import Path
from typing import Any  # The external packet controller supplies execution/status fields.

from wave_checks import full_valid

from wnn.multistage_search import read_json
from wnn.training.sequence_provenance import source_identity


def run_postfit(wave: Any, cell: dict[str, object], fit: Path) -> None:
    full_valid(fit, cell["config_sha256"])
    admission = read_json(wave.control / "ROOT-readout-admission.json")
    source = Path(admission["source"])
    if (
        admission.get("approved") is not True
        or admission.get("reviewer") != "ROOT"
        or source_identity(source, source / "scripts/train_sequence.py")["sha256"]
        != admission["source_sha256"]
    ):
        raise ValueError("post-fit code source requires its separate pinned ROOT admission")
    output = wave.runtime / "runs" / cell["name"] / "readouts"
    output.mkdir(parents=True, exist_ok=True)
    index = read_json(fit / "epoch_index.json")
    swa = output / "swa3456.pt"
    wave.execute(
        cell,
        "swa",
        [
            wave.args.python,
            "-m",
            "wnn.training.sequence_swa",
            "--index",
            str(fit / "epoch_index.json"),
            "--index-sha256",
            index["index_sha256"],
            "--output",
            str(swa),
        ],
        output,
        source,
    )
    for name, checkpoint in (("swa3456", swa), ("epoch6", fit / "model.pt")):
        wave.execute(
            cell,
            f"readout_{name}",
            [
                wave.args.python,
                "-m",
                "wnn.training.sequence_fixed_readout",
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                str(wave.runtime / "project"),
                "--split-manifest",
                str(wave.runtime / "project/chronological_primary.json"),
                "--reference-root",
                admission["reference_root"],
                "--reference-manifest-sha256",
                admission["reference_manifest_sha256"],
                "--output",
                str(output / name),
                "--readout",
                name,
            ],
            output / name,
            source,
        )
    wave.status(cell, "complete", fixed_readouts=["swa3456", "epoch6"])
