# E471 recurrent candidate: independent CPU review

**Local decision: PASS for a separately reviewed production-shape GPU probe. No fit or provider launch is approved by this note.** The six-arm candidate remains a one-seed, small-model, historically exposed mechanism screen; do not report an optimizer ranking or a public-score expectation.

The final reviewed candidate uses the actual E470 `ActivationGRU` parameter/state API and copies the same one-layer PyTorch GRU tensors. The scripted candidate scan preserves PyTorch reset/update/new gate order and the recurrent candidate bias inside `reset * (W_hn h + b_hn)`. Only the post-`tanh` candidate map changes to `u + alpha*sin(pi*u)`. The fixed alpha values are `-0.1, 0, +0.1`; no learned alpha or future-dependent input is present. Six arms share the seed, E470 architecture, train/test cohort, 474-update budget, data order, and optimizer settings. E470's Muon selector verifies disjoint/exhaustive parameter assignment and sends only the two GRU matrices through three-block Muon; other parameters remain AdamW.

I independently ran the bundled adversarial CPU suite under one CPU and 2 GiB:

```bash
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
  WNN_SLICE=wnn-campaign.slice scripts/limited .venv/bin/python \
  research/campaign_2026-10-05/recurrent_activation/tests.py
```

It passed. I also ran an independent bounded synthetic fixture against the E470 model: alpha-zero maximum prediction difference `5.96e-8`, parameter-gradient difference `7.45e-9`, and uneven carried-state chunk difference `2.98e-8`. A sentinel with the recurrent candidate bias removed changed the output by `0.0582`, demonstrating that the parity fixture is sensitive to that placement. Both nonzero alpha values had finite predictions, states, and gradients; states stayed within `[-1,1]` on the fixture. The CPU suite additionally checks `nn.GRU` forward/input/weight gradients and AdamW update parity, seeded E470 initialization and one full-model update, causal prefix invariance, streaming state carry/reset, and one finite parameter-changing update for each of the six arms.

The fit-only real-data CPU preflight also passed under the same cap (`--preflight-real --device cpu`, output at `/tmp/e471_recurrent_independent_preflight`): all six arms used the same four fit row groups, had 8,235 parameters, completed one update, and retained finite changed parameters. This was only a one-update preflight; no full fit or test scoring was run.

The earlier bundled-test failure was a reference-interface mismatch: it compared the custom tensor state with an E467 tuple wrapper. That version was not accepted. The final reviewed test imports E470 `ActivationGRU`, verifies matching parameter names/initialization, and compares the direct tensor-state API; the exact final hashes below apply to this corrected version.

| Source | SHA-256 |
|---|---|
| `fit_recurrent_candidate.py` | `998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf` |
| `tests.py` | `266df5c2a7f126aecf67e37416b3ce17cc011620f3848e747b64019b03678015` |
| `SPEC_CRITIC.md` | `4744369dd6919997b5bea0f940437348f5e3ad7711594c86e39052db2114c76b` |

The TorchScript test emits a deprecation warning under the local PyTorch build; it does not fail parity or CPU fixtures. The source-only `gpu_probe.py` measures one synthetic B4×256 update per arm but does not yet compare CUDA alpha-zero outputs/state with CUDA `nn.GRU`. Before using the candidate for research fits, independently measure a production-shape GPU batch/chunk with finite loss, gradient and parameter checks, allocation cap, throughput and CUDA-vs-`nn.GRU` alpha-zero parity. The recurrent scan loops sequentially over timesteps, so CPU parity does not establish GPU fit throughput.

For a stronger comparison, use the strong two-output causal GRU checkpoint as a shared initialization and continue every alpha arm from that exact checkpoint with fresh optimizer states. Evaluate on predeclared chain-disjoint component folds with fold-specific preprocessing and paired seeds. Cross candidate-only versus embedding-only placement and AdamW/Muon; calibrate Muon-vs-AdamW recurrent update RMS on fit rows only and freeze that calibration before held-out evaluation. The current six-arm E470 screen does not test LR equivalence, placement, fresh components, multiple seeds, or strong-model continuation. Details and the proposed balanced follow-up are in [SPEC_CRITIC.md](SPEC_CRITIC.md).
