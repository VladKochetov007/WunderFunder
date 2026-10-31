from __future__ import annotations

from pathlib import Path


def save_pair(figure: object, destination: Path, stem: str) -> None:
    """Save one standalone scientific figure as both PNG and SVG, then close it."""
    for suffix in ("png", "svg"):
        figure.savefig(
            destination / f"{stem}.{suffix}",
            dpi=160 if suffix == "png" else None,
            bbox_inches="tight",
        )
    import matplotlib.pyplot as plt

    plt.close(figure)
