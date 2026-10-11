"""Tiny deterministic CPU fit used by the flag-off parity test; run once per source tree.

    python tests/parity_fit.py <source_root> <out.json>

Builds the model through ``make_model`` with only the arguments a v6 command line has, trains five steps on
synthetic rows and dumps hex-exact losses, outputs and parameters.
"""

import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "src"))
import train_rnn  # noqa: E402
from wnn.nn import weighted_mse_loss  # noqa: E402

torch.manual_seed(7)
torch.set_num_threads(1)
args = {"hidden": 16, "layers": 1, "dropout": 0.0}
x = torch.randn(3, 150, 112)
y = torch.randn(3, 150, 2)
valid = torch.ones(3, 150, dtype=torch.bool)
n_in = train_rnn.build_inputs(x).shape[-1]
model = train_rnn.make_model(args, n_in, 2)
train_rnn.fit_normalization(model, [(x,)], torch.device("cpu"), batches=1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
losses = []
for _ in range(5):
    out, _ = model(train_rnn.build_inputs(x), None)
    loss = weighted_mse_loss(out[..., :2], y, valid)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item().hex())
with torch.no_grad():
    out, _ = model(train_rnn.build_inputs(x), None)
state = {k: v.flatten().tolist() for k, v in model.state_dict().items()}
Path(sys.argv[2]).write_text(json.dumps({"n_in": n_in, "losses": losses, "out": out.flatten().tolist(),
                                         "state": state}))
