"""Small CPU invariants for the zero-initialized relationship residual head."""

from __future__ import annotations

import torch

from wnn.training.online_relationship import OnlineRelationshipResidualHead


def test_relationship_head_starts_at_the_parent_prediction() -> None:
    """A fresh head preserves the parent function exactly before learning."""
    head = OnlineRelationshipResidualHead()
    features = torch.randn(2, 7, 419)
    parent = torch.randn(2, 7, 2)
    torch.testing.assert_close(head(features, parent), parent, rtol=0.0, atol=0.0)
