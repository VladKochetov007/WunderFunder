"""Current-row feature representations for impact experiment cards 01 and 02."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

N_FEATURES = 112
PLE_KNOTS = 16
PLE_GROUP_WIDTH = 4
GROUP_TOKEN_WIDTH = 64
# With four heads, one call expands to 64,000 batch-head entries at most.
GROUP_ATTENTION_ROW_CHUNK = 16_000


def _validate_values(values: Tensor) -> tuple[int, ...]:
    """Validate normalized FP32 model inputs and return their leading dimensions."""
    if values.ndim < 1 or values.shape[-1] != N_FEATURES:
        raise ValueError("representation inputs must end in the canonical 112 feature columns")
    if values.dtype != torch.float32:
        raise TypeError("representation inputs must remain FP32 under autocast")
    return tuple(values.shape[:-1])


class PLEInputProjection(nn.Module):
    """Add train-quantile scalar embeddings while retaining the raw linear bypass.

    ``knots`` contains 16 packed FP32 knots per canonical feature. ``counts``
    records each feature's valid, strictly increasing prefix; one valid knot
    marks a constant feature whose embedding branch is forced to zero.
    """

    def __init__(
        self,
        base: nn.Linear,
        knots: Tensor,
        counts: list[int],
        embedding_dim: int = PLE_GROUP_WIDTH,
    ) -> None:
        """Create the continuous embedding branch around an existing projection.

        Args:
            base: Canonical raw 112-to-hidden linear projection.
            knots: Train-only knot values with shape ``[112, 16]`` and FP32 dtype.
            counts: Number of valid knots in each packed feature row.
            embedding_dim: Learned vector width at every knot.

        Raises:
            TypeError: If counts, knots, or embedding width have invalid types.
            ValueError: If dimensions, counts, or valid knot order are invalid.
        """
        super().__init__()
        if base.in_features != N_FEATURES:
            raise ValueError("PLE base projection must accept 112 canonical features")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not isinstance(counts, list) or len(counts) != N_FEATURES:
            raise TypeError("counts must be a list with one integer per canonical feature")
        if any(type(count) is not int or not 1 <= count <= PLE_KNOTS for count in counts):
            raise ValueError("each feature must have between one and 16 valid knots")
        if knots.shape != (N_FEATURES, PLE_KNOTS):
            raise ValueError("knots must have shape [112, 16]")
        if knots.dtype != torch.float32:
            raise TypeError("knot values must be FP32")

        packed_knots = knots.detach().clone().contiguous()
        for feature, count in enumerate(counts):
            valid = packed_knots[feature, :count]
            if not bool(torch.isfinite(valid).all()):
                raise ValueError(f"feature {feature} has a non-finite valid knot")
            if count > 1 and not bool(torch.all(valid[1:] > valid[:-1])):
                raise ValueError(f"feature {feature} knots must be strictly increasing")
            if count < PLE_KNOTS:
                packed_knots[feature, count:] = valid[-1]

        self.base = base
        self.counts = tuple(counts)
        self.embedding_dim = embedding_dim
        self.register_buffer("knots", packed_knots)
        active = torch.tensor([count > 1 for count in counts], dtype=torch.float32)
        self.register_buffer("active_features", active, persistent=False)
        self.knot_embeddings = nn.Parameter(
            torch.empty(N_FEATURES, PLE_KNOTS, embedding_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.knot_embeddings, mean=0.0, std=0.02)
        self.output_projection = nn.Linear(N_FEATURES * embedding_dim, base.out_features)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, normalized_values: Tensor) -> Tensor:
        """Return the unchanged raw projection plus an interpolated PLE branch."""
        leading = _validate_values(normalized_values)
        base_output = self.base(normalized_values)
        rows = normalized_values.reshape(-1, N_FEATURES)
        feature_values = rows.transpose(0, 1).contiguous()
        positions = torch.searchsorted(self.knots, feature_values, right=True)
        counts = torch.tensor(self.counts, device=rows.device, dtype=torch.long).unsqueeze(1)
        lower = torch.minimum((positions - 1).clamp_min(0), counts - 1)
        upper = torch.minimum(lower + 1, counts - 1)

        knot_zero = torch.gather(self.knots, 1, lower)
        knot_one = torch.gather(self.knots, 1, upper)
        fraction = (feature_values - knot_zero) / (knot_one - knot_zero).clamp_min(1e-12)
        fraction = fraction.clamp(0.0, 1.0)

        offsets = torch.arange(N_FEATURES, device=rows.device).unsqueeze(1) * PLE_KNOTS
        embeddings = self.knot_embeddings.reshape(-1, self.embedding_dim)
        embedding_zero = F.embedding(offsets + lower, embeddings)
        embedding_one = F.embedding(offsets + upper, embeddings)
        interpolated = embedding_zero + fraction.unsqueeze(-1) * (embedding_one - embedding_zero)
        interpolated = interpolated * self.active_features.view(N_FEATURES, 1, 1)
        encoded = (
            interpolated.permute(1, 0, 2)
            .contiguous()
            .reshape(rows.shape[0], N_FEATURES * self.embedding_dim)
        )
        branch = self.output_projection(encoded).reshape(*leading, self.base.out_features)
        return base_output + branch


def _canonical_feature_groups() -> tuple[tuple[str, slice], ...]:
    """Describe groups in the exact ``wnn.data.FEATURE_COLUMNS`` order."""
    groups: list[tuple[str, slice]] = []
    for instrument, offset in (("i0", 0), ("i1", 52)):
        groups.extend(
            (
                (f"{instrument}_p22", slice(offset, offset + 22)),
                (f"{instrument}_v22", slice(offset + 22, offset + 44)),
                (f"{instrument}_dp4", slice(offset + 44, offset + 48)),
                (f"{instrument}_dv4", slice(offset + 48, offset + 52)),
            )
        )
    groups.extend((f"a{index}", slice(104 + index, 105 + index)) for index in range(8))
    return tuple(groups)


class GroupInputProjection(nn.Module):
    """Add within-row feature-group attention before the temporal backbone."""

    def __init__(self, base: nn.Linear) -> None:
        """Create 16 schema-group tokens and a zero-initialized hidden projection.

        Args:
            base: Canonical raw 112-to-hidden linear projection.

        Raises:
            ValueError: If the base projection does not accept 112 features.
        """
        super().__init__()
        if base.in_features != N_FEATURES:
            raise ValueError("group projection base must accept 112 canonical features")
        self.base = base
        groups = _canonical_feature_groups()
        if len(groups) != 16:
            raise RuntimeError("canonical feature schema must define 16 input groups")
        self.group_names = tuple(name for name, _ in groups)
        self.group_slices = tuple(group for _, group in groups)
        self.group_encoders = nn.ModuleList(
            nn.Linear(group.stop - group.start, GROUP_TOKEN_WIDTH) for group in self.group_slices
        )
        self.group_ids = nn.Embedding(len(groups), GROUP_TOKEN_WIDTH)
        self.attention_norm = nn.LayerNorm(GROUP_TOKEN_WIDTH)
        self.attention = nn.MultiheadAttention(GROUP_TOKEN_WIDTH, 4, dropout=0.0, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(GROUP_TOKEN_WIDTH)
        self.feedforward = nn.Sequential(
            nn.Linear(GROUP_TOKEN_WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, GROUP_TOKEN_WIDTH),
        )
        self.output_projection = nn.Linear(GROUP_TOKEN_WIDTH, base.out_features)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, normalized_values: Tensor) -> Tensor:
        """Mix the 16 groups independently at each row and add them to the bypass."""
        leading = _validate_values(normalized_values)
        base_output = self.base(normalized_values)
        rows = normalized_values.reshape(-1, N_FEATURES)
        tokens = torch.stack(
            [
                F.gelu(encoder(rows[:, group]))
                for encoder, group in zip(self.group_encoders, self.group_slices, strict=True)
            ],
            dim=1,
        )
        tokens = tokens + self.group_ids.weight.unsqueeze(0)
        normalized = self.attention_norm(tokens)
        attended = _row_chunked_attention(self.attention, normalized, GROUP_ATTENTION_ROW_CHUNK)
        tokens = tokens + attended
        tokens = tokens + self.feedforward(self.feedforward_norm(tokens))
        pooled = tokens.mean(dim=1)
        branch = self.output_projection(pooled).reshape(*leading, self.base.out_features)
        return base_output + branch


def _row_chunked_attention(
    attention: nn.MultiheadAttention,
    tokens: Tensor,
    max_rows: int = GROUP_ATTENTION_ROW_CHUNK,
) -> Tensor:
    """Attend over feature groups in bounded batches of independent sequence rows."""
    if tokens.ndim != 3 or tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
        raise ValueError("group attention expects nonempty [rows, groups, width] tokens")
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("group attention row chunk must be a positive integer")
    if tokens.shape[0] <= max_rows:
        return attention(tokens, tokens, tokens, need_weights=False)[0]
    attended = [
        attention(chunk, chunk, chunk, need_weights=False)[0]
        for chunk in tokens.split(max_rows, dim=0)
    ]
    return torch.cat(attended, dim=0)
