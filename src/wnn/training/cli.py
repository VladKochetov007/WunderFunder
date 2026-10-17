from __future__ import annotations

import argparse

from wnn.training.cli_data import add_fit_arguments, add_membership_arguments
from wnn.training.cli_features import add_features_arguments, add_weights_arguments
from wnn.training.cli_model import add_architecture_arguments, add_shape_arguments
from wnn.training.cli_normalization import add_normalization_arguments
from wnn.training.cli_optimizer import add_optimizer_arguments
from wnn.training.cli_performance import add_performance_arguments
from wnn.training.cli_targets import add_targets_arguments
from wnn.training.cli_views import add_views_arguments


def build_parser() -> argparse.ArgumentParser:
    """Build the unchanged training command line from reusable option groups."""
    ap = argparse.ArgumentParser()
    add_shape_arguments(ap)
    add_views_arguments(ap)
    add_fit_arguments(ap)
    add_optimizer_arguments(ap)
    add_targets_arguments(ap)
    add_architecture_arguments(ap)
    add_membership_arguments(ap)
    add_weights_arguments(ap)
    add_features_arguments(ap)
    add_normalization_arguments(ap)
    add_performance_arguments(ap)
    return ap
