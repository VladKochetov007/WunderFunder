"""Declared paired experiment identities; metadata only, no model imports."""

from __future__ import annotations

TREATMENTS = {
    "01": "ple",
    "02": "groups",
    "03": "target_adapters",
    "04": "deep_supervision",
    "05": "tied_depth",
    "06": "pairwise_ranking",
    "07": "future_heads",
    "08": "state_perturbation",
    "09": "nll_transfer",
    "10": "clip_event",
}
CONTROLS = {
    "01": "plain",
    "02": "plain",
    "03": "plain",
    "04": "plain",
    "05": "copied_depth",
    "06": "plain",
    "07": "plain",
    "08": "perturbation_control",
    "09": "gru_plain",
    "10": "gru_plain",
}
CARDS = tuple(TREATMENTS)
EXTENSION_CARDS = ("04", "05", "06", "07", "08", "09", "10")


def variant_for(card: str, arm: str) -> str:
    """Return the frozen control/treatment variant, including card05's special control."""
    if card not in TREATMENTS or arm not in ("control", "treatment"):
        raise ValueError("unsupported paired experiment card/arm")
    return (CONTROLS if arm == "control" else TREATMENTS)[card]
