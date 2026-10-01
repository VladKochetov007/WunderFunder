from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

State = tuple[Tensor, Tensor]
WIDTH = 96
SCALAR_MODES = 64
ROTATION_MODES = 32
STATE_WIDTH = SCALAR_MODES + 2 * ROTATION_MODES


class StructuredStateBlock(nn.Module):
    """Stable scalar/rotation recurrence followed by a nonlinear residual map."""

    def __init__(self, decay_init: str = "multi") -> None:
        super().__init__()
        if decay_init not in {"multi", "single"}:
            raise ValueError("decay_init must be multi or single")
        grid = torch.tensor([2.0, 8.0, 32.0, 128.0, 512.0, 2048.0])
        half_life = grid[torch.arange(WIDTH) % len(grid)]
        if decay_init == "single":
            half_life = torch.full_like(half_life, 32.0)
        self.raw_half_life = nn.Parameter(torch.logit(half_life.log() / math.log(8192.0)))
        phase_grid = math.pi * torch.tensor([-1 / 32, -1 / 128, -1 / 512, 1 / 512, 1 / 128, 1 / 32])
        self.register_buffer("phase", phase_grid[(torch.arange(ROTATION_MODES) // 6) % 6])
        self.inject = nn.Linear(WIDTH, STATE_WIDTH, bias=False)
        self.readout = nn.Linear(STATE_WIDTH, WIDTH, bias=False)
        self.direct = nn.Linear(WIDTH, WIDTH, bias=False)
        self.gate = nn.Linear(WIDTH, WIDTH)

    def half_lives(self) -> Tensor:
        """Return learned half-lives bounded to [1,8192] rows."""
        return torch.exp(torch.sigmoid(self.raw_half_life) * math.log(8192.0))

    def dynamics(self) -> tuple[Tensor, Tensor]:
        """Return log radii and cancellation-safe normalized injection scales."""
        log_rho = -math.log(2.0) / self.half_lives()
        return log_rho, torch.sqrt(-torch.expm1(2 * log_rho))

    def _drive(self, u: Tensor, scale: Tensor) -> Tensor:
        projected = self.inject(u).to(scale.dtype)
        real_scale = torch.cat([scale[:SCALAR_MODES], scale[SCALAR_MODES:].repeat_interleave(2)])
        return projected * real_scale

    def _mix(self, u: Tensor, states: Tensor) -> Tensor:
        o = self.readout(states) + self.direct(u)
        return u + torch.sigmoid(self.gate(u)) * F.gelu(o)

    def fft_states(self, drive: Tensor, incoming: Tensor) -> Tensor:
        """Causal convolution with the analytically propagated incoming state.

        Args:
            drive: Already normalized injection, shape [batch,time,128].
            incoming: Real state immediately before this chunk, shape [batch,128].

        Returns:
            Real states after each row, shape [batch,time,128].
        """
        length = drive.shape[1]
        log_rho, _ = self.dynamics()
        row = torch.arange(length, device=drive.device, dtype=drive.dtype)
        n_fft = 1 << (2 * length - 2).bit_length()
        scalar_kernel = torch.exp(row[:, None] * log_rho[:SCALAR_MODES])
        scalar_spectrum = torch.fft.rfft(drive[..., :SCALAR_MODES], n_fft, dim=1)
        scalar_filter = torch.fft.rfft(scalar_kernel, n_fft, dim=0)
        scalar = torch.fft.irfft(scalar_spectrum * scalar_filter[None], n_fft, dim=1)[:, :length]
        scalar = (
            scalar
            + (scalar_kernel * torch.exp(log_rho[:SCALAR_MODES]))[None]
            * incoming[:, None, :SCALAR_MODES]
        )

        pair = drive[..., SCALAR_MODES:].reshape(drive.shape[0], length, ROTATION_MODES, 2)
        complex_drive = torch.complex(pair[..., 0], pair[..., 1])
        log_lambda = torch.complex(log_rho[SCALAR_MODES:], self.phase)
        rotation_kernel = torch.exp(row[:, None] * log_lambda)
        rotation = torch.fft.ifft(
            torch.fft.fft(complex_drive, n_fft, dim=1)
            * torch.fft.fft(rotation_kernel, n_fft, dim=0)[None],
            n_fft,
            dim=1,
        )[:, :length]
        initial_pair = incoming[:, SCALAR_MODES:].reshape(-1, ROTATION_MODES, 2)
        initial_complex = torch.complex(initial_pair[..., 0], initial_pair[..., 1])
        rotation = (
            rotation + (rotation_kernel * torch.exp(log_lambda))[None] * initial_complex[:, None]
        )
        real_rotation = torch.stack([rotation.real, rotation.imag], dim=-1)
        return torch.cat([scalar, real_rotation.flatten(-2)], dim=-1)

    def real_step(self, drive: Tensor, incoming: Tensor) -> Tensor:
        """Apply independent real scalar/2x2 dynamics, without FFT machinery."""
        log_rho, _ = self.dynamics()
        rho = log_rho.exp()
        scalar = rho[:SCALAR_MODES] * incoming[:, :SCALAR_MODES]
        pair = incoming[:, SCALAR_MODES:].reshape(-1, ROTATION_MODES, 2)
        cosine, sine = self.phase.cos(), self.phase.sin()
        rotated = (
            torch.stack(
                [
                    cosine * pair[..., 0] - sine * pair[..., 1],
                    sine * pair[..., 0] + cosine * pair[..., 1],
                ],
                dim=-1,
            )
            * rho[SCALAR_MODES:, None]
        )
        return torch.cat([scalar, rotated.flatten(-2)], dim=-1) + drive

    def forward(self, u: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate a training chunk using bounded O(T log T) convolution."""
        _, scale = self.dynamics()
        drive = self._drive(u, scale)
        states = self.fft_states(drive, state)
        return self._mix(u, states), states[:, -1]

    def sequential(self, u: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate a chunk with the independent recurrence for parity only."""
        _, scale = self.dynamics()
        drive = self._drive(u, scale)
        outputs: list[Tensor] = []
        for row in drive.unbind(dim=1):
            state = self.real_step(row, state)
            outputs.append(state)
        states = torch.stack(outputs, dim=1)
        return self._mix(u, states), state

    def stream_step(self, u: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Process one row with the same real recurrence used by native ports."""
        _, scale = self.dynamics()
        state = self.real_step(self._drive(u, scale), state)
        return self._mix(u, state), state


class StructuredStateModel(nn.Module):
    """Two width96 stable-state blocks with a projection skip and shared head."""

    def __init__(self, n_in: int, n_out: int = 2, decay_init: str = "multi") -> None:
        super().__init__()
        if n_in <= 0 or n_out <= 0:
            raise ValueError("n_in and n_out must be positive")
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, WIDTH)
        self.blocks = nn.ModuleList(
            [StructuredStateBlock(decay_init), StructuredStateBlock(decay_init)]
        )
        self.head = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, n_out))

    def reset(self, batch_size: int = 1) -> State:
        """Return zero state for a new independent sequence batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        prototype = self.blocks[0].raw_half_life
        return tuple(prototype.new_zeros(batch_size, STATE_WIDTH) for _ in range(2))

    def _checked_state(self, batch: int, state: State | None) -> State:
        if state is None:
            return self.reset(batch)
        if len(state) != 2 or any(s.shape != (batch, STATE_WIDTH) for s in state):
            raise ValueError("state must contain two [batch,128] real tensors")
        return state

    def encode(self, x: Tensor) -> Tensor:
        """Standardize input and produce its causal row projection."""
        return F.gelu(self.inp((x - self.mean) / self.std))

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        """Process [batch,time,input] and carry a flat tuple across TBPTT chunks."""
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[-1] != self.mean.numel():
            raise ValueError("x must have shape [batch,positive_time,n_in]")
        state = self._checked_state(x.shape[0], state)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.encode(x.to(self.mean.dtype))
            u, s1 = self.blocks[0](z, state[0])
            u, s2 = self.blocks[1](u, state[1])
            return self.head(torch.cat([u, z], dim=-1)), (s1, s2)

    def sequential(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        """Use independent real updates as an auditable full-model reference."""
        state = self._checked_state(x.shape[0], state)
        z = self.encode(x)
        u, s1 = self.blocks[0].sequential(z, state[0])
        u, s2 = self.blocks[1].sequential(u, state[1])
        return self.head(torch.cat([u, z], dim=-1)), (s1, s2)

    def stream_step(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        """Advance one [batch,input] row; resetting is explicit via reset()."""
        if x.ndim != 2 or x.shape[-1] != self.mean.numel():
            raise ValueError("stream input must have shape [batch,n_in]")
        state = self._checked_state(x.shape[0], state)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.encode(x.to(self.mean.dtype))
            u, s1 = self.blocks[0].stream_step(z, state[0])
            u, s2 = self.blocks[1].stream_step(u, state[1])
            return self.head(torch.cat([u, z], dim=-1)), (s1, s2)
