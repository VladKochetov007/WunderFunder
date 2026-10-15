# minGRU Q8 phase profile method

The sealed paired local profile measured the full 20k-row raw callback at 0.3086 s for H256×2 GRU Q8 and 0.9394 s for H384×2 minGRU Q8. Their matrix counts were 972,032 and 966,528 respectively, with the same deterministic source-qualified train37 sequence and clean-F0 normalization. Each used `fast_activations=true` and asserted `precision="q8vnni"`; minGRU's `Q8Vnni` selects `Isa::Vnni512`. The GRU's selected `Q8Vnni` wrapper also selects VNNI512 but calls its unfused `step_dyn_impl` specialization (`FUSED=false`). These are cost comparisons for synthetic weights, not quality evidence.

The paths still perform different surrounding work. The GRU encoder is I16, then the serving step quantizes its encoder state and each recurrent layer's updated state. Its dynamic path invokes the AVX2 quantizer, vector fast GELU, and vector GRU update when supported. The minGRU prototype stores Q8 weights in its encoder, both input-only cell projections, and first readout projection; each Q8 linear independently quantizes its current activation, so a two-layer callback quantizes widths 211, 384, 384, and 768 each row (1,747 input values), while the paired GRU dynamic state path quantizes three 256-wide vectors (768 values). minGRU also uses scalar fast-activation loops for encoder/readout GELU and the candidate/write-sigmoid state update. This source accounting motivates phase timing but does not establish the share of elapsed runtime.

The profile feature records the following spans in `vnni::prof::CYCLES`; other cargo builds do not include counters or the Python counter accessors.

| Slot | Span | Interpretation |
|---:|---|---|
| 13 | 211-value normalization | Per-row FP32 preprocessing before the encoder |
| 14 | Q8 activation quantization | Sum across every Q8 linear's scalar quantizer |
| 15 | Q8 matvec | Sum across every VNNI512 dynamic-Q8 matrix operation |
| 16 | FP32 matvec | Includes minGRU's final two-output head |
| 17 | Encoder activation | Current selected fast GELU implementation |
| 18 | Candidate, write gate, and state update | Sum across recurrent layers; scientific recurrence is unchanged |
| 19 | Skip-state/projection concatenation | Recurrent state plus encoder activation |
| 20 | Head activation | Current selected fast GELU implementation |
| 21 | Whole model step, inclusive | Includes nested phase spans |
| 22 | Raw sanitize and causal feature builder | 112 raw columns to 211 model columns |
| 23 | Raw callback body, inclusive | Includes feature building and the whole model step |

The runner first reproduces the sealed uninstrumented Q8 prediction and final-state hashes, then executes three complete reset-separated profile callbacks and requires bitwise equality to that baseline. It reports cumulative RDTSC ticks per raw row and each phase's ratio to the inclusive callback-body ticks. Inclusive totals overlap the component spans. Timer and atomic counter overhead perturbs the instrumented worker, so timing interpretation is limited to within-build phase shares; paired uninstrumented wall time remains the runtime anchor.

All callback timing uses deterministic synthetic weights from NumPy PCG64 `SeedSequence([20261014, 1])`, but actual fitting-sequence raw features at train row-group 37, sequence 7598721; labels and masks are not loaded. This is a local mechanism profile only. Any optimization considered afterward must preserve minGRU's represented weights, candidate/write equations, operation order, reset behavior, and complete-sequence parity. A changed numeric result would require an explicitly labeled numerical variant and separate evidence.
