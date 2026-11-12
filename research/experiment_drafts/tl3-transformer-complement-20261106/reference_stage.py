from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from wnn.training.sequence_reference_pins import PUBLISHED_RUNS


def stage(pr49_archive: Path, pr52_archive: Path, data_root: Path) -> None:
    """Restore only17 pinned fixed checkpoints/provenance, without loading model objects."""
    with zipfile.ZipFile(pr49_archive) as first, zipfile.ZipFile(pr52_archive) as second:
        for run, pin in sorted(PUBLISHED_RUNS.items()):
            archive = second if run == "tl3c_w18final_distilled_s5" else first
            entries = {
                f"checkpoints/model_ep{pin['epoch']}.pt": pin["checkpoint_sha256"],
                "metadata/provenance.json": pin["provenance_sha256"],
            }
            for relative, expected_hash in entries.items():
                member = f"{run}/{relative}"
                info = archive.getinfo(member)
                if not 0 < info.file_size <= 32_000_000 or info.is_dir():
                    raise ValueError("reference archive member exceeds its bounded asset size")
                payload = archive.read(member)
                if hashlib.sha256(payload).hexdigest() != expected_hash:
                    raise ValueError(f"published reference asset mismatch: {member}")
                target = data_root / "artifacts" / "runs" / run / Path(relative).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or target.parent.resolve() != target.parent.absolute():
                    raise ValueError("reference asset destination may not traverse symlinks")
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
                        raise FileExistsError(
                            f"preserve a different existing reference asset: {target}"
                        )
                    continue
                with target.open("xb") as stream:
                    stream.write(payload)


def main() -> None:
    """Stage the published PR49/52 archive endpoints into a receiver-owned data root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr49-archive", type=Path, required=True)
    parser.add_argument("--pr52-archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    stage(args.pr49_archive, args.pr52_archive, args.data_root.resolve(strict=True))


if __name__ == "__main__":
    main()
