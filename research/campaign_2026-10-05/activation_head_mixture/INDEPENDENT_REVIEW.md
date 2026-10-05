# Independent review: E477 readout mixture × dropout

**Decision: GO for an actual-data B4×256 probe only.** The full twelve-fit matrix remains gated on a passing sealed probe and controller admission. I ran the bounded CPU invariants and a clean probe `--help`; I did not run CUDA or fits.

The three readout choices are genuinely distinct, parameter-free functions on the same 64 coordinates: exact GELU, a fixed 32/32 GELU/LeakyReLU split, and a same-coordinate 50/50 GELU/LeakyReLU mixture. The split and mixture have the same width, model parameters, and underlying baseline weights, so this tests coordinate assignment against a per-channel blend without adding capacity. All variants replace only the hidden head activation; the 42-channel input GELU, two-layer GRU, recurrent gates, `[recurrent, embedding]` concatenation and output layer remain unchanged. The head's `nn.Dropout` is immediately after activation; `p=0` is the matched no-dropout condition and `p=0.1` is train-mode only.

For each seed, every arm loads the same E474 state dictionary and retains its keys, values, buffers, and 61,058 trainable parameters. The trainer resets PyTorch RNG to the seed before each model construction; all variants construct the same parameter shapes before loading the shared state, so the `p=0.1` cells receive matched dropout RNG streams across activations. Fit groups are shuffled once per seed and the same order is replayed across its six cells; the full-fit receipt correctly records that shuffled order. B4 batches reset state, 79 chunks carry/detach it, and the inherited fixed telemetry indices are 0, 46, 236, and 473. The E470 objective and `need_prediction` mask are reused, and the loader excludes test groups.

The CPU invariants passed under a one-CPU/2-GiB limit. They check exact initialization and state-dict equality, exact GELU/p=0 output/state/gradient parity with E474, formula values and finite derivatives, dropout-disabled eval behavior, causal-prefix isolation, carried/reset state, and that `p=0.1` changes train-mode outputs. The source-bound fit-only CPU preflight has 12 cells with one real B4×256 update each, 61,058 parameters, finite updates, and shared fit-only group order. The GPU probe likewise uses a real B4×256 batch, exact baseline output/state/gradient parity, and one finite parameter-changing update for each seed/activation/dropout cell.

Before sealing, the controller must declare the probe's actual output artifact as `probe.json` under the supplied output directory. The source writes `probe.json`; it does not write `result.json`. This check matters because a prior E474 spec named the wrong probe artifact. The probe resets CUDA peak statistics before its actual batch/model allocations and records peak allocated and reserved device memory. Its successful result is only a shape/resource gate.

Source identities reviewed:

- `fit_activation_head_mixture.py`: `2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7`
- `gpu_probe.py`: `1275b980942943ef9982992d965587ffdd3dacc21d593e41d02fdf9bf282cdc4`
- `tests.py`: `876bd7bb397ef2f948fe0fe8f4bcd75b41bc6f67223892b0060ffe5b960cb778`
- `PREREG.md`: `ade9059c2cfa6d0c2bd9e2b8dbc0838bd4e12b294ac98c1e0f85069f290fbb38`

Scores must retain all six arms for both seeds. The sixteen evaluation components have prior exposure, so reported contrasts are conditional exploratory findings and must not be used to claim a public forecast or select an unmeasured winner.
