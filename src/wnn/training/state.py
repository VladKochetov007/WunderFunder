from __future__ import annotations

from attrs import define, field

from wnn.training.environment import TrainingEnvironment
from wnn.training.fit_state import BatchState, OptimizationState, ProgressState
from wnn.training.setup_state import DataState, NetworkState, OptionsState


@define
class TrainingRun:
    """Own explicit configuration, datasets, network, optimizer and stream clocks."""

    environment: TrainingEnvironment
    options: OptionsState = field(factory=OptionsState)
    data: DataState = field(factory=DataState)
    network: NetworkState = field(factory=NetworkState)
    optimization: OptimizationState = field(factory=OptimizationState)
    progress: ProgressState = field(factory=ProgressState)
    batch: BatchState = field(factory=BatchState)
