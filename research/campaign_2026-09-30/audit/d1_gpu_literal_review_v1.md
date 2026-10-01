# D1 synthetic GPU literal replay

PASS for frozen v4 endpoint-content. The actual input, GPU readout and all six carried-state tensors are embedded in `event_content/gpu_probe_content_v2.pt` under `parity`. A separate `.cpu_replay.pt` file is not required; the critic's initial filename-based lookup was corrected before numerical testing.

Strictly loaded final synthetic student weights, reset batch1/length2000 and exact FP32 input were evaluated with `grade_event_content_v2.py::serial`, independently written before the GPU profile. GPU output maximum difference2.62260e-6 passes the unchanged3e-5 gate. Floating state differences are3.01003e-6,1.37091e-6,1.31130e-6, each passing.003. Row/write counters and previous raw prices match exactly. CPU parallel versus literal passes as well.

Inherited GPU flags were checked: highest matmul precision, CUDA matmul TF32 disabled, cuDNN TF32 enabled/deterministic. The separate saved AMP diagnostic is finite, keeps S/Z in FP32 and matches integer counters/previous raw prices; no numerical AMP accuracy claim is made.

`d1_gpu_literal_replay_v1.json` binds profile, checkpoint, model and independent reference hashes. Durable `c20260930_independent_d1_gpu_literal_v1` exited0; combined D1 replay and SSM failure-coordinate extraction took1.02s with observed peak549.69MiB. Nested hard1core/2GiB/swap0 in `wnn-campaign.slice` was checked, CUDA stayed uninitialized, and no competition data were read.

This grade applies to source SHA256 `75beff387c4f9312dc3bc7e065d76e7df2ceeb666fe1bb3e93cc245dfca445cd` and synthetic content/reset FP32 only. Position/raw arms, trained checkpoints and a proposed v5 scan optimization require their own appropriate gates. It supplies no quality claim.
