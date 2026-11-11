"""Declared paired experiment identities; metadata only, no model imports."""

from __future__ import annotations

TREATMENTS = {
    "01": "ple",
    "02": "groups",
    "03": "target_adapters",
    "04": "deep_supervision",
    "05": "tied_depth",
    "06": "pairwise_ranking",
}
CONTROLS = {card: "copied_depth" if card == "05" else "plain" for card in TREATMENTS}
CARDS = tuple(TREATMENTS)
EXTENSION_CARDS = ("04", "05", "06")


def variant_for(card: str, arm: str) -> str:
    """Return the frozen control/treatment variant, including card05's special control."""
    if card not in TREATMENTS or arm not in ("control", "treatment"):
        raise ValueError("unsupported paired experiment card/arm")
    return (CONTROLS if arm == "control" else TREATMENTS)[card]
