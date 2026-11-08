from __future__ import annotations

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest

from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
from wnn.training.checkpoints import configure_checkpoint_policy
from wnn.training.cli import build_parser
from wnn.training.setup_lineage import require_chronological_primary_training


def _fixed_campaign_args(*extra: str) -> tuple[ArgumentParser, Namespace]:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--name",
            "policy-check",
            "--device",
            "cpu",
            "--split-manifest",
            "unused-manifest.json",
            "--checkpoint-policy",
            "fixed",
            *extra,
        ]
    )
    configure_checkpoint_policy(args, parser)
    return parser, args


def test_chronological_campaign_uses_fixed_checkpoint_and_evaluates_during_fit() -> None:
    parser, args = _fixed_campaign_args()
    split = SimpleNamespace(schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, valid_groups=())

    require_chronological_primary_training(parser, args, split)
    assert args.no_eval_during_fit is False
    assert args.checkpoint_epoch == args.epochs - 1


def test_random_campaign_is_rejected_even_with_fixed_checkpoint() -> None:
    parser, args = _fixed_campaign_args()
    split = SimpleNamespace(schema="wnn.sequence-inner-split.v1", valid_groups=())

    with pytest.raises(SystemExit):
        require_chronological_primary_training(parser, args, split)


def test_chronological_campaign_rejects_noeval_and_nonfixed_checkpoint() -> None:
    parser, noeval_args = _fixed_campaign_args("--no-eval-during-fit")
    split = SimpleNamespace(schema=CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, valid_groups=())
    with pytest.raises(SystemExit):
        require_chronological_primary_training(parser, noeval_args, split)

    parser = build_parser()
    nonfixed_args = parser.parse_args(
        [
            "--name",
            "policy-check",
            "--device",
            "cpu",
            "--split-manifest",
            "unused-manifest.json",
        ]
    )
    with pytest.raises(SystemExit):
        configure_checkpoint_policy(nonfixed_args, parser)
