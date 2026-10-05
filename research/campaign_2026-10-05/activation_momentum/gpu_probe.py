"""Callable E483 all-cell CUDA update probe; source only until separately admitted."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"))
sys.path.insert(0, str(ROOT / "src"))

import fit_momentum as actor  # noqa: E402


def run(output: Path) -> dict:
    """Load fit-role inputs and run the complete actual B4×256 probe matrix."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe directory: {output}")
    parents = actor.check_parent_sources()
    states = actor.make_seed_states()
    manifest, x, y, need, nodes = actor.load_fit_cohort()
    result = actor.run_probe(output, manifest, x, y, need, nodes, states)
    result["parent_source_sha256"] = parents
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested without CUDA")
    run(args.output)


if __name__ == "__main__":
    main()
