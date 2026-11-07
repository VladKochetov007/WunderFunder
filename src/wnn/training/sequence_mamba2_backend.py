"""Pinned upstream Mamba-2 weights/SSD scan with exact causal chunk state.

Adapted from historical blob 7f60c07122b5284d5a8849f2c3033f6a3af1d4aa.
The historical observation remains unchanged; this copy is a fresh adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution
from json import JSONDecodeError, loads
from typing import Protocol, cast

import attrs
import torch
import torch.nn.functional as F
from torch import nn

_ScanResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
_ChunkScan = Callable[..., _ScanResult]
_MAMBA_SSM_VERSION = "2.2.4"
_MAMBA_SSM_COMMIT = "95d8aba8a8c75aedcaa6143713b11e745e7cd0d9"


class _Mamba2Module(Protocol):
    """The subset of the upstream Mamba2 module used by the streaming wrapper."""

    d_conv: int
    d_inner: int
    d_state: int
    chunk_size: int
    headdim: int
    nheads: int
    ngroups: int
    in_proj: nn.Linear
    conv1d: nn.Conv1d
    dt_bias: nn.Parameter
    A_log: nn.Parameter
    D: nn.Parameter
    norm: nn.Module
    out_proj: nn.Linear

    def __call__(self, values: torch.Tensor) -> torch.Tensor: ...

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate the upstream inference oracle's per-layer state."""
        ...

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the upstream one-row inference oracle."""
        ...


@attrs.frozen
class Mamba2LayerState:
    """Causal convolution history and SSD state for one layer."""

    conv: torch.Tensor
    ssm: torch.Tensor

    def detach(self) -> Mamba2LayerState:
        """Detach both recurrent tensors at a truncated-BPTT boundary."""
        return Mamba2LayerState(self.conv.detach(), self.ssm.detach())


@attrs.frozen
class Mamba2State:
    """Streaming state for every Mamba-2 layer in a regressor."""

    layers: tuple[Mamba2LayerState, ...]

    def detach(self) -> Mamba2State:
        """Detach every layer state at a truncated-BPTT boundary."""
        return Mamba2State(tuple(layer.detach() for layer in self.layers))


def _load_mamba2_backend() -> tuple[type[nn.Module], _ChunkScan]:
    """Load the pinned upstream implementation only when this model is built."""
    # Pinned upstream APIs:
    # https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/modules/mamba2.py
    # https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/ssd_combined.py
    try:
        installed = distribution("mamba-ssm")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Mamba2Regressor requires mamba-ssm v2.2.4 at commit "
            f"{_MAMBA_SSM_COMMIT}; see the architecture-execution-20261107 L4 packet"
        ) from exc
    if installed.version != _MAMBA_SSM_VERSION:
        raise RuntimeError(
            f"Mamba2Regressor requires mamba-ssm v{_MAMBA_SSM_VERSION}; found v{installed.version}"
        )
    direct_url = installed.read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError(
            "Mamba2Regressor requires mamba-ssm installed from the pinned Git commit; "
            "direct_url.json metadata is missing"
        )
    try:
        revision = loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (AttributeError, JSONDecodeError) as exc:
        raise RuntimeError("mamba-ssm direct_url.json metadata is invalid") from exc
    if revision != _MAMBA_SSM_COMMIT:
        raise RuntimeError(
            f"Mamba2Regressor requires mamba-ssm Git commit {_MAMBA_SSM_COMMIT}; found {revision!r}"
        )
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as exc:
        raise RuntimeError(
            "Mamba2Regressor requires the pinned Linux CUDA mamba-ssm backend"
        ) from exc
    return Mamba2, mamba_chunk_scan_combined


class _Mamba2Block(nn.Module):
    """Pre-norm residual block using upstream Mamba-2 weights and SSD scan."""

    def __init__(
        self,
        width: int,
        state_size: int,
        expand: int,
        head_dim: int,
        conv_width: int,
        chunk_size: int,
        mixer_type: type[nn.Module],
        scan: _ChunkScan,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        # Preserve upstream Mamba2 parameters and initialization. Its A_log, D
        # and dt_bias no-decay annotations are intentionally not translated
        # into optimizer groups; the first architecture screen uses the same
        # all-parameter AdamW recipe as the comparison models.
        self.mixer = cast(
            _Mamba2Module,
            mixer_type(
                d_model=width,
                d_state=state_size,
                d_conv=conv_width,
                expand=expand,
                headdim=head_dim,
                chunk_size=chunk_size,
                ngroups=1,
                rmsnorm=True,
                norm_before_gate=False,
                # The streaming wrapper supplies its own causal Conv1d. Keep
                # upstream's full-forward oracle usable without causal-conv1d.
                use_mem_eff_path=False,
            ),
        )
        self.scan = scan

    def forward(
        self, values: torch.Tensor, state: Mamba2LayerState | None
    ) -> tuple[torch.Tensor, Mamba2LayerState]:
        """Advance a chunk while carrying exact causal convolution and SSM state."""
        batch, steps, _ = values.shape
        mixer = self.mixer
        conv_channels = mixer.conv1d.in_channels
        conv_history_size = mixer.d_conv - 1
        conv_shape = (batch, conv_history_size, conv_channels)
        ssm_shape = (batch, mixer.nheads, mixer.headdim, mixer.d_state)
        if state is None:
            conv_history = values.new_zeros(conv_shape)
            ssm_history = values.new_zeros(ssm_shape)
        else:
            if state.conv.shape != conv_shape or state.ssm.shape != ssm_shape:
                raise ValueError("Mamba-2 state shape does not match this chunk or model")
            if state.conv.device != values.device or state.ssm.device != values.device:
                raise ValueError("Mamba-2 state and input must be on the same CUDA device")
            if state.conv.dtype != torch.float32 or state.ssm.dtype != torch.float32:
                raise ValueError("Mamba-2 carried state must remain float32")
            conv_history, ssm_history = state.conv, state.ssm

        normalized = self.norm(values)
        projected = mixer.in_proj(normalized)
        z, raw_conv, dt = torch.split(
            projected,
            [mixer.d_inner, conv_channels, mixer.nheads],
            dim=-1,
        )

        # The upstream Mamba-2 layer applies depthwise Conv1d with left padding.
        # Keeping the last d_conv-1 raw projected inputs makes arbitrary chunk
        # boundaries equivalent to one causal sequence, including chunks shorter
        # than the convolution width.
        conv_context = torch.cat((conv_history, raw_conv), dim=1)
        convolved = F.conv1d(
            conv_context.transpose(1, 2),
            mixer.conv1d.weight,
            mixer.conv1d.bias,
            groups=conv_channels,
        ).transpose(1, 2)
        convolved = F.silu(convolved)
        x, b_state, c_state = torch.split(
            convolved,
            [mixer.d_inner, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state],
            dim=-1,
        )

        scan_result = self.scan(
            x.reshape(batch, steps, mixer.nheads, mixer.headdim),
            dt,
            # Match upstream stability handling; activations and SSM states keep
            # the model's dtype, so the default research path remains FP32.
            -torch.exp(mixer.A_log.float()),
            b_state.reshape(batch, steps, mixer.ngroups, mixer.d_state),
            c_state.reshape(batch, steps, mixer.ngroups, mixer.d_state),
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            initial_states=ssm_history,
            return_final_states=True,
        )
        if not isinstance(scan_result, tuple):
            raise RuntimeError("upstream Mamba-2 SSD scan did not return its final state")
        scan_output, next_ssm = scan_result
        scan_output = scan_output.reshape(batch, steps, mixer.d_inner)
        mixed = mixer.out_proj(mixer.norm(scan_output, z))

        if conv_history_size == 0:
            next_conv = conv_context[:, :0].clone()
        else:
            # Clone the small persistent tail; otherwise this view keeps the
            # entire chunk's concatenated projection storage alive between calls.
            next_conv = conv_context[:, -conv_history_size:].clone()
        return values + mixed, Mamba2LayerState(next_conv, next_ssm)
