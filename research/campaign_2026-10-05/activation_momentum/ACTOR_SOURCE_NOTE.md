# E483 actor source and CPU preflight receipt

This actor implements the registered 18-cell design without changing E481 or
its optimizer factory: seeds 271831/271832/271833 × baseline/sine × recurrent
Muon beta1 0.8/0.9/0.95. Each seed has one generated initial state shared by its
six cells; the three states have distinct hashes. Only the single Muon group
containing `gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.weight_ih_l1`, and
`gru.weight_hh_l1` receives the varied beta. Every non-Muon parameter stays in
AdamW with `(0.9, 0.999)`. Tests verify exact disjoint trainable-parameter
coverage, LR 0.001, weight decay 0.01, Muon blocks 3, five Newton–Schulz steps,
and scale 0.2.

The synthetic optimizer check uses two non-collinear successive gradients and
an independent calculation of the frozen Muon buffer/effective-gradient
recurrence. It confirms distinct second-step Muon results at all three betas
and bit-identical AdamW-group results. The CPU parity check compares the local
clipping helper to the frozen E474 update given identical gradients. The
callable CUDA probe adds native reset-after GRU output/state/gradient parity and
clipped beta-0.9 MuonAdamW-step parity, then checks an actual B4×256 update in
all 18 cells. That GPU probe has not been run.

The real-data CPU preflight was run under one CPU and a 2 GiB memory cap:

```bash
WNN_CPU_PCT=4 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 scripts/limited .venv/bin/python research/campaign_2026-10-05/activation_momentum/fit_momentum.py --output research/campaign_2026-10-05/activation_momentum/preflight_cpu_v3 --preflight-real --device cpu
```

It passed one fit-only B4×256 update for all 18 cells, with finite loss and
parameters and changed parameters in every arm. The receipt records
`test_rows_loaded=false`, the exact E470 fit cohort, dataset/cohort hashes,
distinct initialization hashes and resolved optimizer groups. A separate
receipt audit confirmed the exact arm matrix and group betas/names. Four
synthetic tests pass. Ruff and `py_compile` pass for all three Python files.

| Object | SHA256 |
|---|---|
| Actor | `54f4fbc66f69e165a5b21ff740faa556ef368d326ebbeb24736d68a36440bbe8` |
| Synthetic tests | `d63b46acc9abae85e6ebac2bdda0c89de01832f197c89b50f926d3fe675e9a42` |
| CUDA probe source | `af3ac0fba649d0027841f07ccd551798222f57f3da569e84b7064393a4f775c4` |
| CPU preflight manifest | `d9fb30e5c809fceded545ab39ff9779def9b16ee467586723381fa868cc4cb13` |

Frozen parent identities include E470/E474/E481 actor helpers and
`src/wnn/nn.py`, `src/wnn/optim.py`, and `src/wnn/resources.py`; the exact
hashes are embedded in the preflight manifest. The design and blind-critic
digests are also embedded there. The preregistered contrast set remains 27
global contrasts: nine per seed (three sine-minus-baseline effects, four
momentum effects, and two difference-in-differences interactions), each
reported separately by seed with joint paired component resampling.

No GPU probe, full fit, checkpoint scoring, or test-row read was run. This is
source/preflight evidence for independent review, not fit admission.
