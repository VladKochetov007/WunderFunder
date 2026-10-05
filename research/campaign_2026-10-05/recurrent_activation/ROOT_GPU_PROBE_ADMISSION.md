# Parent admission of the real CUDA probe

The independent CPU review passes equation, initialization, state, causality and optimizer partition checks. The updated probe now uses only the frozen four fit sequences and compares alpha-zero output/state and input/parameter gradients against E470 on the requested device before six one-update checks. Its CPU dry check passed. No test rows are loaded, and no full fit is included in this admission.

The production-shaped probe is bounded by two CPUs, 6 GiB worker RAM, 2 GiB allocator, 3 decimal GB observed GPU stop and 300 seconds. Parent remains 12 CPU / 30 decimal GB RAM hard / 27 GB soft / zero swap. This is a resource and numerical probe, not evidence of generalization. Initial per-arm timing includes compilation effects; any estimate must include evaluation and margin.

The independent review's old description of synthetic-only probing applies to the superseded probe, not this reviewed version. No tolerance was widened after seeing a CUDA result.

- `gpu_probe.py`: `aeac54f44ade8370845746a65c3e5d7c7285d3c4241e96c768d73f1a7c5a5b5d`
- `fit_recurrent_candidate.py`: `998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf`
- `tests.py`: `266df5c2a7f126aecf67e37416b3ce17cc011620f3848e747b64019b03678015`
