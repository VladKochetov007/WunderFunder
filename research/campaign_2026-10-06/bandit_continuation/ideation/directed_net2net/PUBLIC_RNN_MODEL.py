import torch
import torch.nn.functional as F
from torch import nn

class RNNModel(nn.Module):
    """Input projection -> stacked GRU -> head fed by GRU output and the projection skip."""

    def __init__(
        self, n_in: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0, n_out: int = 2,
        cell: str = "gru", enc: str = "linear", tok_dim: int = 32, att_layers: int = 1, conv_k: int = 16,
        readout_activation: str = "gelu",
    ):
        super().__init__()
        if readout_activation not in ("gelu", "fixed_mix"):
            raise ValueError("readout_activation must be gelu or fixed_mix")
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        # Optional per-row encoders, added as residuals to the linear projection (research screening only;
        # the native engine supports enc="linear").
        self.enc = enc
        if enc == "mlp":
            self.enc_norm = nn.LayerNorm(hidden)
            self.enc_up = nn.Linear(hidden, 4 * hidden)
            self.enc_down = nn.Linear(2 * hidden, hidden)
        elif enc == "tokens":
            self.groups = token_groups(n_in)
            self.tok_in = nn.ModuleList(nn.Linear(b - a, tok_dim) for a, b in self.groups)
            self.cls = nn.Parameter(torch.zeros(1, 1, tok_dim))
            layer = nn.TransformerEncoderLayer(tok_dim, 4, 2 * tok_dim, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.att = nn.TransformerEncoder(layer, att_layers, enable_nested_tensor=False)
            self.tok_out = nn.Linear(2 * tok_dim, hidden)
        elif enc == "conv":
            self.conv_k = conv_k
            self.conv_dw = nn.Conv1d(hidden, hidden, conv_k, groups=hidden)
            self.conv_pw = nn.Linear(hidden, hidden)
        elif enc != "linear":
            raise ValueError(enc)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        # Attribute name kept as ``gru`` so existing checkpoints and exporters stay valid.
        self.gru = rnn(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU() if readout_activation == "gelu" else FixedReadoutMixture(),
            nn.Linear(hidden, n_out),
        )

    def encode(self, xn):
        u = self.inp(xn)
        if self.enc == "tokens":
            b, t, _ = xn.shape
            flat = xn.reshape(b * t, -1)
            toks = torch.stack([lin(flat[:, a:c]) for lin, (a, c) in zip(self.tok_in, self.groups)], dim=1)
            toks = self.att(torch.cat([self.cls.expand(b * t, -1, -1), toks], dim=1))
            pooled = torch.cat([toks[:, 0], toks[:, 1:].mean(1)], dim=-1)
            u = u + self.tok_out(pooled).reshape(b, t, -1)
        z = F.gelu(u)
        if self.enc == "mlp":
            a, g = self.enc_up(self.enc_norm(z)).chunk(2, dim=-1)
            z = z + self.enc_down(F.gelu(a) * torch.sigmoid(g))
        elif self.enc == "conv":
            c = F.pad(z.transpose(1, 2), (self.conv_k - 1, 0))  # causal; zero left context at chunk starts
            z = z + F.gelu(self.conv_pw(self.conv_dw(c).transpose(1, 2)))
        return z

    def forward(self, x, h=None):
        z = self.encode((x - self.mean) / self.std)
        o, h = self.gru(z, h)
        return self.head(torch.cat([o, z], dim=-1)), h
