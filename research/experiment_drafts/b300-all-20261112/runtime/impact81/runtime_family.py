"""Pinned runtime, resource and environment contracts by Impact81 card family."""

from __future__ import annotations

from typing import Literal

RuntimeFamily = Literal["transformer", "gru"]

TRANSFORMER_RUNTIME_COMMIT = "8626000a4b965763a70011a0291a891aa6535d7e"
GRU_RUNTIME_COMMIT = "67c3bd818ea243177eeb1dc789cd3d145eba0e81"
# The B300 wave uses one device per fit.  Its paired recipe doubles the local
# sequence batch so each optimizer update sees the same global examples as the
# historical 2-rank recipe.
TRANSFORMER_WORLD_SIZE = 1
GRU_WORLD_SIZE = 1
TRANSFORMER_ENDPOINT_EPOCHS = 4
GRU_ENDPOINT_EPOCHS = 6

CARD_FAMILIES: dict[str, RuntimeFamily] = {
    "01": "transformer",
    "02": "transformer",
    "03": "transformer",
    "04": "transformer",
    "05": "transformer",
    "06": "transformer",
    "07": "transformer",
    "08": "gru",
    "09": "gru",
    "10": "gru",
}


def family_for_card(card: str) -> RuntimeFamily:
    """Return the canonical runtime family for one frozen card number."""
    try:
        return CARD_FAMILIES[card]
    except KeyError as error:
        raise ValueError(f"unsupported Impact81 card: {card}") from error


def runtime_commit_for_card(card: str) -> str:
    """Return the immutable canonical runtime commit required for a card."""
    family = family_for_card(card)
    return TRANSFORMER_RUNTIME_COMMIT if family == "transformer" else GRU_RUNTIME_COMMIT


def world_size_for_card(card: str) -> int:
    """Return the frozen distributed world size for a card's runtime."""
    family = family_for_card(card)
    return TRANSFORMER_WORLD_SIZE if family == "transformer" else GRU_WORLD_SIZE


def epochs_for_card(card: str) -> int:
    """Return the fixed final checkpoint epoch count for a card's family."""
    family = family_for_card(card)
    return TRANSFORMER_ENDPOINT_EPOCHS if family == "transformer" else GRU_ENDPOINT_EPOCHS


def environment_for_card(card: str) -> dict[str, str | None]:
    """Return family-specific environment overrides; None removes inherited values."""
    family = family_for_card(card)
    if family == "transformer":
        return {"CUBLAS_EMULATE_SINGLE_PRECISION": "1"}
    return {"CUBLAS_EMULATE_SINGLE_PRECISION": None}
