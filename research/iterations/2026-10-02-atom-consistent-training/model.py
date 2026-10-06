"""Only the original atom decoder changes; parameter names/shapes/order are identical."""
from __future__ import annotations
import torch
from wnn.atom_head import AtomRNNModel


def consistent_decode(pi, atoms, regression):
    c = atoms.clamp(-2, 2)
    amplitude = c.abs().mean(-1)
    # Zero-amplitude atoms are unidentifiable under weighted CE and carry no metric mass.
    safe_amplitude = torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
    u = (pi / safe_amplitude) * (amplitude > 0).to(pi.dtype)
    numerator = u @ (c * c.abs())
    denominator = u @ c.abs()
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(denominator > 0, numerator / safe_denominator, regression)


class ConsistentAtomRNNModel(AtomRNNModel):
    def __init__(self, *args, atom_decoder="original", **kwargs):
        if atom_decoder not in ("original", "corrected"):
            raise ValueError(atom_decoder)
        super().__init__(*args, **kwargs)
        self.atom_decoder = atom_decoder

    def forward(self, x, h=None):
        if self.atom_decoder == "original":
            return super().forward(x, h)
        z = self.encode((x - self.mean) / self.std)
        o, h = self.gru(z, h)
        feats = torch.cat([o, z], dim=-1)
        out = self.head(feats).float()
        logits = self.atom_head(feats).float()
        pi = torch.softmax(logits, dim=-1)
        reg = out[..., :2]
        dec = consistent_decode(pi, self.atoms, reg)
        a, b = self.class_slice
        p3 = torch.softmax(out[..., a:b], dim=-1)[..., 3:4]
        lam = torch.sigmoid(self.mix_logit)
        mixed = reg + lam * p3 * (dec - reg)
        return torch.cat([mixed, out[..., 2:], logits], dim=-1), h
