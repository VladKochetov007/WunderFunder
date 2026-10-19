# Four-arm architecture handoff

Prepared implementation source is published at [bd78ac104](https://github.com/VladKochetov007/WunderFunder/tree/bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb). This packet proposes its focused port and future four-arm comparison on the current modular trainer. **No executable current-base trainer, GPU validation, fit or grade is supplied here.** The historical one-epoch F0 screen is retained unchanged and is not CP30.

| Arm | Model | Primary contrast |
|---|---|---|
| A | GRU H256×2 | Common reference |
| B | Causal Transformer D256×4, 8 heads, FFN×4, window256 | B−A |
| C | Same Transformer, window2048 | C−B |
| D | Mamba-2 D256×4, state64, head64, expand2, conv4, chunk256 | D−A |

Future scientific design: 30 paired cells per arm, 120 endpoint fits. Parameter count and measured FLOPs differ; common data exposure is not equal compute. Training authorization, current-trainer bindings and resource profiles are pending. Use existing cell execution infrastructure; do not introduce another queue manager.

- [PR.md](PR.md) gives the 11-section scientific handoff and blockers.
- [recipe.json](recipe.json) is metadata for the existing plan printer, not a runnable trainer recipe.
- [Published source map](../../unified/ideas/PUBLISHED_GPU_PACKETS_20261018.md) links architecture modules, the frozen host runner, CPU budget checker and the separate complete smoothing implementation. It explains why neither old branch should be merged wholesale into the current trainer.
- [Original fixed recipe and commands](https://github.com/VladKochetov007/WunderFunder/blob/bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb/research/campaign_2026-10-17/architecture_gpu_wave_r1/README.md) remain historical. Do not run them as this new CP30 comparison.

The next receiving-side work is to port model/state/prediction seams, explicitly enforce strict FP32/TF32-off, verify sequence causality and streaming/backward parity on the target GPU, bind clean data/configs and profile all arms. CPU inference or GRU distillation is separate. No local tests or training were run for publication.

Independent documentation review checked the 4×30 matrix, three contrasts, historical/current-source separation and precision gates. It requested explicit Mamba chunk256, now bound above and in the recipe. This review did not execute code.
