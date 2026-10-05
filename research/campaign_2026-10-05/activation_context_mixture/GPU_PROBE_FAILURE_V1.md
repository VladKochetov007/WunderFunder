# E484 CUDA probe v1 incident

The first E484 probe attempt, `e484-context-mixture-real-gpu-probe-v1`, failed
before the first model update. The failure came from
`check_common_parameter_bytes`: it passed CUDA model state and saved CPU
initial state directly to `torch.equal`, which rejects tensors on different
devices. This was a state-check device mismatch, not a parameter mismatch.

The preserved runner log is
`research/campaign_2026-10-04/runner/logs/e484-context-mixture-real-gpu-probe-v1.log`;
the attempt receipt is
`research/campaign_2026-10-04/runner/attempts/e484-context-mixture-real-gpu-probe-v1.json`.
It reports failure after 11.872740337 seconds, 1,318,625,280 bytes peak RAM,
364,904,448 bytes observed VRAM, no worker memory-event increments, and no
result artifact. The v1 sealed source and its files remain unchanged.

The v2 actor compares each tensor's dtype, shape, and exact byte representation
after copying both sides to contiguous CPU storage. The v2 synthetic test
checks this canonical representation (including signed-zero distinction), and
the fit-only CPU preflight passes all twelve cells. An independent source
review and renewed root admission are still required before a new CUDA probe.
