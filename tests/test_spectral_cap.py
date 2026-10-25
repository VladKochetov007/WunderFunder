import torch

from wnn.spectral_cap import cap_gate_blocks_


def test_cap_clamps_each_gate_block_and_preserves_small_ones() -> None:
    torch.manual_seed(0)
    gru = torch.nn.GRU(6, 4)
    model = torch.nn.Module()
    model.gru = gru
    with torch.no_grad():
        gru.weight_hh_l0[4:8] *= 50.0  # update-gate block of W_hh
    before = gru.weight_hh_l0.detach().clone()
    largest = cap_gate_blocks_(model, tau=2.0)
    assert largest > 2.0
    for p in (gru.weight_ih_l0, gru.weight_hh_l0):
        for g in range(3):
            assert float(torch.linalg.svdvals(p[g * 4 : (g + 1) * 4].detach())[0]) <= 2.0 + 1e-5
    r_before = torch.linalg.svdvals(before[:4])
    if float(r_before[0]) <= 2.0:
        torch.testing.assert_close(gru.weight_hh_l0[:4].detach(), before[:4])
