import json

import torch
from torch import nn

from wnn.training_trace import TrainTrace, parameter_families


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(4, 8)
        self.rnn = nn.GRU(8, 8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        h, _ = self.rnn(torch.relu(self.inp(x)))
        return self.head(h)


def test_trace_records_steps_and_state(tmp_path):
    torch.manual_seed(0)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    trace = TrainTrace(tmp_path / "trace.jsonl", model, every=2, state_every=4, flush_every=3)
    assert {"rnn_hh", "rnn_ih", "inp", "head", "bias"} <= set(parameter_families(model))
    for step in range(9):
        loss = ((model(torch.randn(3, 5, 4)) - 1.0) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        trace.observe_grad(step)
        optimizer.step()
        trace.record(step, 0, loss, {"family": loss.detach() * 0.5}, optimizer, scaler)
    trace.flush()
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    steps = [r for r in rows if r["kind"] == "step"]
    states = [r for r in rows if r["kind"] == "state"]
    assert [r["step"] for r in steps] == [0, 2, 4, 6, 8]
    assert all(r["gn_total"] > 0 and r["loss_family"] == r["loss"] * 0.5 for r in steps)
    assert steps[0]["lr"] == [1e-2] and steps[0]["beta1"] == [0.9]
    assert [r["step"] for r in states] == [0, 4, 8]
    assert "ur_rnn_hh" not in states[0] and states[1]["ur_rnn_hh"] > 0
    assert states[1]["v_rms_head"] > 0 and states[1]["m_norm_head"] > 0
