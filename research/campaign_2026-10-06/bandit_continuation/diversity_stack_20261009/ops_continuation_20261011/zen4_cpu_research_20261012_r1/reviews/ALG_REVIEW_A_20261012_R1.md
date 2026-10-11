# Independent algorithmic CPU review A — 2026-10-12

**Disposition:** prioritize an exact Zen4 kernel experiment and a smaller supported fixed endpoint. The existing trio is not rescued by its smaller ZIP. Its dominant work is three independent dense recurrent paths. Exact fusion can remove intermediate traffic; a substantial reduction in arithmetic requires a different compressed model and fresh quality evidence.

Reviewer A used the user-authorized Sol 6.1/xhigh exception for this bounded independent review. This is report-only work in branch `agent/algorithmic-compute-review-a-20261012`; MAIN artifacts were read-only. No reviewer B report, current native actor proposal or private root scout note was read before sealing this report. No training, model benchmark, provider mutation, installation, new rental or child agent was used.

## Evidence and identity

Read project AGENTS instructions, specs README/objective/current-state, campaign index/current exploitation record/allocation/resume receipts, ADR0008/0012/0031, and the research-evidence-review and research-recipe-audit skills plus empirical-validation reference. Live campaign properties observed: CPU quota 500%, memory-high 18,000,000,000, maximum 20,000,000,000, no swap, memory-current 13,493,583,872 bytes. Two metadata-only Python checks ran under MAIN's absolute `scripts/limited` with CPU 50%, one thread, RAM 700,000,000 and 120-second per-command maximum; tool wall times 2.02 and 0.76 seconds. No inference timings were collected by this reviewer.

Source inspected at MAIN HEAD `e4055421a64ca5eaf97aa88050f1bb1b0c989541`; `git status --short native` was clean. Native source SHA256: rnn_fast `7e6df836313aac1f8175fd1952362f03738fcc5629318b4b01cc5a5e5057708b`; python `6f8a4107f2edcce13d6620a19970ec25f459e184d6bd3d111ed5f9d74242b322`; vnni `803b7e99a2234587eafef6861678e610c72ddf87fcc970958efdbedc7a8f3406`; half `87035e5569bb7a53295c222026337a4a08661f1d6d821c2b26074fa12b4bb9cc`; stream `e5efebc7bfb6efba8ab9ea1f694ea11c128dec5869fe7809603e89efbd3ea1e6`. Source reading is not a disassembly/rebuild proof that every compiled instruction matches these current files.

Actual ZIPs were opened and model tensors, dtypes, coefficients and hashes inspected:

| ZIP | SHA256 | Delivered members | ZIP bytes |
|---|---|---|---:|
| 10 | `837b0fb3c71b22062747d7e1e3bfd2241cb6c5cd41edac908745dd36d5469410` | H388 H384 s4/s5, 1/2 each | 17,272,496 |
| 11 | `d83723831277cf7327ab84ba781a2f2ebe883aca40cd1f7bc6a222e30be28284` | H390 H256 s3 + H388 H384 s5, 1/2 each | 12,914,019 |
| 12 | `ecbf13cd321b90400a6a48bc6f20f43df4cfbef4aed650404785384c23631b7b` | H388 H384 s3/s4/s5, 1/3 each | 13,735,320 |

All three use the same solution SHA `f025d65abb4990c05ca1677cefd2502340f9eb8cd80180cd38c14044a2c9bf54` and v3 native SHA `ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e`; precision.txt is `q8vnni`. The solution fixes `budget_s=None`, directly binds the native callback and expands every loaded tensor to contiguous FP32. ZIP12 stores the four GRU matrices and head.0 in FP16, then constructs the same dynamic Q8 path. This is storage compression, with slightly changed quantized weights, not FP16 compute or reduced recurrent shape.

All members have input width 211, two GRU layers and head width equal to hidden width. Every input projection/bias and mean/std inspected differs across the distinct members. Example input-weight hashes: H390 s3 `8c537b7601323d0f94d8fcd17767965bcaf5f4523a1f2a804b7e3312670b4e98`; H388 s3 `7a002e3b2659fe7ceb2e25b8964b878316fadc9cd556415cc6a5dd20dc6fae65`; s4 `82df8fe44e1e16b19b8bd9b9426dede0f3ca681ca4bbe74fcc3976f055305f43`; s5 `257cb29b05a70260f0b987a3aadd8f240342d38102538e29d74ed2d7332b618c`. Consequently a common learned projection/backbone is not present in these ZIPs.

Runtime evidence supplied in the assignment is a matched Py311 panel of ten complete valid sequences, two reset passes, 400,000 callbacks: old ZIP4 pair 52.74 us/row; ZIP5 single 21.77; ZIP11 74.84; ZIP10 104.48; ZIP12 158.62. Portal evidence: old pair 27m1s, old single 15m41s; ZIP12 TIMEOUT at reported 61m56s/647 MB. The last observation is right-censored execution, not a completed runtime. ZIP11 is not uploaded. The stored historical panels also vary: ZIP11 Py311 repeats 74.81/72.61 us; ZIP10 107.35/95.51; ZIP12 Py313 151.81/156.23, with a separate Py311 replay previously checked. They are historical measurements, not replacements for the newly supplied matched panel.

Primary receipts: `submissions/10-11-67/READY_TOP_CANDIDATES_R5.json`; `ops_continuation_20261011/PORTAL_TRIO_TIMEOUT_20261012_R1.json`; `submissions/10-11-exploit-now_r2/{h390_student_h388_teacher_pair_q8,h388_full_data_teacher_pair_q8}/PY311_CALLBACK_PANEL_R1.json`; `submissions/10-11-exploit-now_r3_fp16transport_q8/artifacts/h388_s3_s4_s5_equal_q8_fp16transport_CALLBACK_PANEL.json`; historical transport review `ops_continuation_20261011/fp16_transport_r3_independent_review_r1/INDEPENDENT_REVIEW_R1.json`.

## Why additional members approach the timeout

For hidden width H, two GRU layers cost `12 H²` multiply-accumulate terms per row, head.0 `2 H²`, input projection at most `211 H`, and head.2 `2 H`. Dense total is `14 H² + 213 H`, excluding activations, quantization and callback overhead. Projection skips zero input columns; recurrence/head matrices do not skip rows merely because their output is similar.

| H | GRU terms | head.0 terms | dense total | dynamic Q8 matrix bytes, including scales/offsets |
|---:|---:|---:|---:|---:|
| 256 | 786,432 | 131,072 | 972,032 | 946,176 |
| 384 | 1,769,472 | 294,912 | 2,146,176 | 2,107,392 |

Dynamic storage follows actual `DynQ8` padding: `14 H² + 112 H` for these divisible-by-64 shapes. The input I16 projection adds 108,032 or 162,048 bytes, plus bias/scales/other weights and alignment. Thus even a whole H256 member is not guaranteed to fit 1 MiB, and H384 matrices alone exceed it. AMD documents 1 MiB private L2 per Zen4 core and shared L3; a vCPU is not a promise of exclusive L3 residency. [AMD 4th Gen architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/products/epyc/4th-gen-amd-epyc-processor-architecture-whitepaper.pdf).

ZIP11/10/12 dense totals are 3,118,208 / 4,292,352 / 6,438,528. The trio/pair-H384 arithmetic ratio is exactly 1.5; supplied callback ratio is 1.518. ZIP11/10 arithmetic ratio is .726; callback ratio .716. This closely fits member work growth; it does not independently prove a memory bottleneck. The old portal pair/single ratio 1.723 differs from the local 2.423 ratio. A single laptop-to-portal multiplier, guessed hidden row count, or linear extrapolation from the censored timeout is unsupported.

Existing features and normalization are already shared in `python.rs:535` and `stream.rs`; normalization is folded into each member's own projection in `rnn_fast.rs:748`. Recurrent/temporary arenas and sparse-index capacity are allocated at construction. Q8 VNNI, eight interleaved accumulators, aligned panels, alternating member order, cross-member prefetch, fat LTO and one codegen unit already exist. They are not new remedies.

## Five prioritized mechanisms

### 1. Exact Zen4 VNNI width and tile scheduling experiment

**Locations:** `vnni.rs:331` (`gemv_vnni`), `vnni.rs:274` (existing AVX2), `vnni.rs:49` ISA dispatch, `rnn_fast.rs:1330` dynamic step. Compare existing ZMM kernel against a YMM VNNI kernel using AVX512VL+VNNI, with bounded 32/64-row tiles and 1/2 independent accumulator groups. The present AVX2 fallback widens bytes and uses `vpmaddwd`; it is not a YMM VNNI comparison.

AMD's Zen4 executes 512-bit register data through 256-bit internal paths; this motivates the experiment, not a prediction that YMM wins. [AMD architecture description](https://docs.amd.com/api/khub/documents/Fk9hEabDNdOiD4PcQu7EYw/content). Rust's `_mm256_dpbusd_epi32` requires `avx512vl` and `avx512vnni` and is stable since 1.89. [Official Rust intrinsic](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_epi32.html). Add the exact feature gate; do not confuse AVX-VNNI with AVX512VL VNNI. No numeric latency/throughput claim is established here; the current source's “5-cycle” comment is not verified Zen4 evidence.

**Proof condition:** integer dot and offset subtraction are unchanged, no int32 overflow, and FP epilogue retains `float(dot) * (row_scale * xs) + base`, without FMA or reassociation. For H≤384 the code's integer bounds are comfortably below int32. Padding and scalar tail semantics stay identical. Reordering exact integer partial sums is safe under these bounds. Complexity remains O(H²), bytes unchanged; any gain is scheduling/decoder/load efficiency.

**Falsifier:** exact archived codes and actual hidden/projection vectors, then identical full-sequence callback outputs and resets; check assembly for `vpdpbusd`, spills and expected row count. Benchmark warm and cold actual ZIP11/12 paths, not only a cache-hot toy matvec. The local Intel result cannot select the Zen4 schedule definitively. No authorized target Zen4 execution exists in this assignment; do not rent one.

### 2. Exact per-hidden-tile GRU fusion and quantization pass reuse

**Locations:** `rnn_fast.rs:1401` layer loop, `vnni.rs:249` epilogue, `rnn_fast.rs:565` gate update, `vnni.rs:417` quantization. Pack corresponding reset/update/candidate rows together. For each hidden tile compute both input and old-hidden integer products, apply their separate scales/biases, then compute r/u/c/h immediately. This can avoid writing and rereading full `gi`/`gh` arrays and reduce independent dispatches. The present path materializes all gates, then updates h, then scans h for max before packing codes.

Equations remain `r=sigmoid(ir+hr)`, `u=sigmoid(iu+hu)`, `c=tanh(ic+r*hc)`, `h'=(1-u)*c+u*h`. Preserve those operations/approximations in that order. **Keep old quantized h immutable until every hidden tile's hh product is finished.** Writing new h is safe because hh consumes old codes; writing new codes in place while subsequent tiles still read them is a causal corruption. Use separate old/new code buffers or defer code replacement. Collect max|h'| during gate update, then perform the necessary packing pass using the identical global scale. Dynamic scale prevents a one-pass update-and-quantize scheme.

Do not concatenate ih and hh into one quantized dot with one scale: their activation scales differ, and candidate gate resets only the hh candidate term. Similarly head.0 fusion must preserve `((Dh * (swh*xsh) + b1) + Dz * (swz*xsz))` with separate sums/scales, not merge integer products. Complexity and weight traffic remain O(H²); temporary gate traffic O(H) is reduced. Test hidden-state/code/scale identity after every row and layer, not only final output. Profile first; scratch savings alone may not materially rescue a trio.

### 3. Exact common preprocessing and callback micro-optimizations, explicitly bounded

**Locations:** `python.rs:535/718`, `vnni.rs:470`, `half.rs:286`, `stream.rs:88`. Build the ordered nonzero input index list once per shared width in the ensemble and feed it to members; today every input projection repeats `nz.clear/extend`. All inspected ZIP members use width211, and normalization is folded into their weights, so the zero mask of the sanitized common input is identical. Preserve original column accumulation order. This removes O(M·211) repeated scans, not the O(M·211·H) projections.

For the three bounded log-since-event counters, a lookup table generated by the identical `f64 ln_1p → f32` operation for integers0..10000 can remove repeated transcendentals; preserve CAP and reset semantics. A table is deterministic on a given supported implementation; cross-build bit identity must be measured rather than assumed from math equivalence. Input sanitization, counters and recurrent updates still run on every row.

Read need_prediction before the model head and skip only the feedforward output head when false: the head does not feed recurrent state. Measured demand is 19,901/20,000 per sequence, so this can remove head work on only .495% of rows; even removing *all* work on those rows would save at most .495%. It is not a major rescue. Returning a borrowed/reused NumPy output could save allocations but is unsafe unless scorer ownership/lifetime proves callers do not retain it. Current `PyArray1::from_slice` creates an owned result; changing it requires explicit callback-contract evidence. Do not compromise correctness for this small term.

**Falsifier:** unchanged complete-sequence output hash, two reset passes, retained-output ownership test, nonfinite-row sanitization cases. Profile callback versus native stage shares before allocating more work. No shared learned subgraph exists in these fixed members; identical family names do not establish one.

### 4. Low-rank recurrence as a quality tradeoff, with a Q8-aware storage threshold

**Locations:** existing `factorized.rs:57`, `rnn_fast.rs:1192`, constructor/import paths `python.rs:16`. For `W ∈ R^(3H×H) ≈ A_(3H×r) B_(r×H)`, cost falls from `3H²` to `4Hr` per matrix; arithmetic threshold is `r < 3H/4`. Four recurrent matrices give `16Hr` versus `12H²`. Approximation residual and recurrent feedback make this a new serving endpoint, not exact redundancy removal. Even an algebraically exact factorization changes FP rounding relative to dense execution.

The existing factor path stores FP32 factors. It beats a one-byte dense Q8 matrix's weight bytes only at `16Hr < 3H²`, i.e. **r < 3H/16** (H384: r<72; H256: r<48), ignoring scales. An H384 rank128 FP32 factorization uses 786,432 bytes per recurrent matrix, larger than the dense Q8 442,368. The existing FP32 head further weakens any blanket cache claim. A rank64 H384 screen or independently implemented quantized factors is a materially different arm; it must beat the actual Q8 endpoint in measured quality, size and callback runtime.

Fit/compress using permitted fitting inputs with clean fold ancestry, freeze ranks and sensitive matrices, then evaluate full causal sequences and the complete fixed mixture on source-qualified folds/seeds. No CPU model fitting is admitted here. Reject a tested compression for measured opportunity cost, not merely because its CI crosses zero; preserve full-size teachers.

### 5. Compress the complete useful ensemble into one student or shared-backbone family

Train an independently initialized H256/H192 student against the **complete fixed teacher mixture**, optionally with small parallel residual heads sharing one recurrent backbone. This changes `M·O(LH²)` recurrent work to one backbone plus O(number_of_heads·H) readout work. Same-width one-backbone compute is not an exact replacement for independent member states. With the current two-layer/head design, an H256 single's dense term count is 972,032 versus H384 trio6,438,528; this is arithmetic accounting, not a predicted speedup or quality result.

The present H390 H256 student is available, but its recorded teacher/parent lineage does not establish a faithful compression of this all-data H388 trio. Shared learned backbones are not in ZIP11/12. New students/residual-head variants require actual GPU admission, clean own teacher/initializer ancestry for cross-fold comparisons, seed/fold robustness, fixed-mixture official scoring and quantized endpoint checks. Do not call any proposed student ready or substitute teammate weights. This is the strongest algorithmic route if modest exact kernel gains cannot create enough runtime margin.

## Benchmark and release plan

Freeze one proposed kernel treatment and ZIP11 as a compact control; keep ZIP12 as a stress artifact, not a recommended upload. Reuse the exact panel SHA `e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3`, full 20k-row causal histories, ten sequences, two reset passes and original need masks. Preserve model coefficients, scales, activation approximation, evaluator and precision selection. Compare hashes and per-layer states before timing; approximation arms instead owe official quantized quality on complete matched endpoints.

Record initialization separately from callbacks, CPU identity/flags, Python3.11/native hashes, cgroup CPU usage/throttling, affinity, memory peaks and warm/cold repeats. Existing `vnni::prof` stage instrumentation uses unserialized RDTSC and atomics and perturbs the path; its stage totals are diagnostic, not a substitute for uninstrumented callback measurements or CPU-cycle certainty. Calibrate those observations separately. No perf/LLVM-mca installation is necessary to make the first bounded comparison useful.

Do not skip unchanged rows: even constant x usually has `h_t=f(x,h_(t-1)) != h_(t-1)`, and event-counter features change with time. Do not batch future callbacks, average nonlinear member weights, infer exact hidden row count, convert local deltas to public scores, drop members based on elapsed time, increase caps, spend prepaid credit that is gone, or include teammate code/binaries. Existing governor remains disabled. A deterministic complete endpoint can be packaged from local evidence, but ZIP11 still needs controller runtime admission; this review does not establish target completion or certify it ready.
