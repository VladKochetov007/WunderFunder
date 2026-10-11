# Independent algorithmic review B — 2026-10-12 R1

**Decision:** retain the strongest fixed smaller endpoint as the immediate release path; test two small exact optimizations before attempting a large kernel rewrite. ZIP11 is the relevant current smaller candidate, pending ROOT's runtime/release review. No inspected algorithmic change already establishes that the timed-out trio will finish on EPYC9224. Do not postpone a supported CPU-safe pair for an unmeasured compression implementation.

This first proposal was independently derived. I did not read reviewer A, ROOT scout notes, or the current native actor's proposals. Owner: `algorithmic_compute_reviewer_b`; analysis only, no source implementation, benchmark, fit, GPU/provider action, installation, or child delegation. Latest user exception permits this Sol6.1/xhigh role only. The assignment admits at most 120 seconds of CPU checks under the existing campaign slice, 50% CPU / one thread / 700 MB / 120-second scope timeout. Two metadata-only Python checks consumed approximately 0.66 CPU seconds in total. The inspected parent slice was 500% CPU, 18 GB memory-high, 20 GB maximum, no swap. Light text reads and web retrieval were also performed. MAIN was read-only; only this report and its compact JSON are written in my isolated worktree.

## Evidence and actual endpoint identity

Read project instructions, specs README/objective/current state, ADR0021 and ADR0033, recipe-audit and evidence-review skills, current allocation receipt and portal-timeout receipt. MAIN HEAD during inspection: `4fa9d3ad0fca7ea627d4519f38f1a5ab18f39dc9`. Dirty unrelated files were preserved.

| Frozen artifact | Actual model shapes | Bytes | ZIP SHA256 |
|---|---|---:|---|
| ZIP11 `11_h390_student_s3_h388_teacher_s5_all_q8vnni.zip` | independent H256/H384, two GRU layers each, head width H, input 211 | 12,914,019 | `d83723831277cf7327ab84ba781a2f2ebe883aca40cd1f7bc6a222e30be28284` |
| ZIP12 `12_h388_teacher_trio_all_fp16store_q8vnni.zip` | three independent H384 teachers, two layers, head width H | 13,735,320 | `ecbf13cd321b90400a6a48bc6f20f43df4cfbef4aed650404785384c23631b7b` |
| ZIP10 H388 s4/s5 pair | two independent H384 teachers | 17,272,496 | `837b0fb3c71b22062747d7e1e3bfd2241cb6c5cd41edac908745dd36d5469410` |
| old uploaded ZIP4 `4_z1c_z4_pair_q8vnni.zip` | two independent H256 backbones, each with head width 512 | 9,544,031 | `ff498873027318b5aa20e6ca95517bdb843b4656989a75aae148f1c141705327` |
| old uploaded ZIP5 `5_z1c_q8vnni_single.zip` | ZIP4's first H256 backbone and width-512 head | 5,419,442 | `936f88d75f2950e17e0ba093328e5abec1cf6cbcc8d390baad8b8a3e38cdf83b` |

These five ZIPs have the same `solution.py` hash `f025d65abb4990c05ca1677cefd2502340f9eb8cd80180cd38c14044a2c9bf54`, and exactly the same three native binaries: v3 `ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e`, AVX2 `e544ed55538c2c08a7b28389fb2a1cd10a91bf272b7dc4f7cbb7bc3b01ebdaf2`, v2 `0ea36be1ff45556b157f6295892f44adaa89fcfb8242329bbe530dfae07a5fed`. Each requests Q8VNNI and binds the native callback directly; `budget_s=None`. I checked their actual nested NPZ headers, shapes, dtypes and tensor hashes without executing inference.

No within-endpoint pair has identical backbone tensors. Matching architecture, source family, mean/std, or a shared teacher does not permit exact forward-pass sharing. ZIP11's teacher does match ZIP10's seed-5 teacher; this is reuse across alternative artifacts, not an opportunity inside one endpoint. ZIP12 stores selected arrays as FP16 but requests the same native Q8 kernels. Transport rounding changes some Q8 codes/quality; it does not halve native Q8 matrix traffic.

Reviewed source hashes: `python.rs` `6f8a4107f2edcce13d6620a19970ec25f459e184d6bd3d111ed5f9d74242b322`; `rnn_fast.rs` `7e6df836313aac1f8175fd1952362f03738fcc5629318b4b01cc5a5e5057708b`; `vnni.rs` `803b7e99a2234587eafef6861678e610c72ddf87fcc970958efdbedc7a8f3406`; `half.rs` `87035e5569bb7a53295c222026337a4a08661f1d6d821c2b26074fa12b4bb9cc`; `stream.rs` `e5efebc7bfb6efba8ab9ea1f694ea11c128dec5869fe7809603e89efbd3ea1e6`; `approx.rs` `9ccf56716d8416210cab7b6260df12bd30178a8cad6bb289f0b8d3111649f912`.

## Computational model and existing optimizations

Already implemented: one shared feature update and optional online normalization per row; preallocated member state/scratch; folded affine normalization; sparse zero-column input projection; aligned packed weights; cached quantized hidden state between rows; quantize-once reuse for layer input and split head; VNNI integer dot products; eight interleaved dot accumulators; vectorized epilogues and fast activations; alternating independent-member order with fixed blend order; prefetch during activation passes; fat LTO, optimization level 3, one codegen unit. These must not be presented as new gains.

For two layers and head width H, hot Q8 codes alone are `12H²` recurrent plus `2H²` head bytes. Projection is `2×211H` bytes. Q8 scales/offset sums add `112H` bytes, before biases, final head, alignment, state and interpreter traffic. H384 therefore streams 2,064,384 Q8 code bytes plus 162,048 projection bytes and 43,008 Q8 metadata bytes; H256 streams 917,504 + 108,032 + 28,672. The whole H384 member exceeds one-core L2 even before ensemble effects. Old width-512 H256 heads are larger than ZIP11's H256 head; do not use H256 alone as a complete cost description.

AMD documents Zen4 AVX512 as using 256-bit internal data paths and a 1 MB L2. A 512-bit kernel is therefore not automatically twice as fast as a 256-bit kernel. This is an architecture fact, not a measured instruction throughput or a claim that AVX2 is faster. [AMD EPYC9004 toolchain tuning guide](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58020-epyc-9004-tg-high-perf-toolchain.pdf), [AMD GDC2024 processor software optimization presentation](https://gpuopen.com/gdc-presentations/2024/GDC2024_AMD_Ryzen_Processor_Software_Optimization.pdf). Our AVX2 fallback widens bytes and uses `maddwd`; comparing it with current VNNI does not isolate vector width. A fair new YMM-VNNI arm needs AVX512VL/BW/VNNI dispatch and the same unsigned offset correction.

No stage-time or hardware-counter measurement was admitted to this review. Matrix bandwidth, instruction throughput, activation division, and CPU scheduling remain competing bottleneck explanations. Apply Amdahl's law to measured stages: if a fraction `f` of callback time is improved by factor `s`, overall speedup is `1 / ((1−f)+f/s)`. Removed operations/bytes below are bounds, not actual speedup estimates.

## Ranked changes

### 1. Precompute scaled sparse projection input once per member, and share nonzero indices

**New; bit-preserving candidate; best first implementation by effort/risk.** `vnni.rs::project_zmm` (around line 466) and `half.rs::project_as` (around line 286) rebuild `nz` and compute `x[j] * scales[j]` inside every output tile. H384 repeats that multiplication and associated loads six times; H256 repeats it four times. `python.rs::advance` already has one shared sanitized input.

Build the ascending nonzero-index list once for each required input width. For each model, compute `v_j = fl(x_j × column_scale_j)` once into preallocated scratch, then broadcast `v_j` in every tile. Member scales differ and cannot be shared. Current output remains `a_r ← fl(a_r + fl(code_rj × v_j))` in the same ascending j order, with the same decoded weights and folded bias. Repeating an identical f32 multiplication cannot change its value; storing the result in f32 retains it exactly. Preserve zero filtering on the original x, including any underflow of v, and preserve nonfinite behavior.

Bound: projection-specific scalar multiply count falls from `tiles×nnz` to `nnz`; duplicate scans fall from members×211 to 211 for these exact-width endpoints. Matrix traffic and its per-output accumulation are unchanged. Do not claim a sixfold projection or callback improvement. Check final assembly: LLVM might already hoist some arithmetic; scratch writes can erase the gain.

Parity barrier: bitwise full predictions and persistent states against the old exact ZIP on complete sequences, including resets, ±0, sparse tails and finite extreme values; test ragged generic input widths separately. No per-row Vec allocation. Keep a generic fallback. A bounded actor can implement this in isolation before one matched callback panel.

### 2. Fuse activation output with its absolute-maximum reduction

**New; bit-preserving candidate; second low-risk arm.** `rnn_fast.rs::step_dyn_impl` (1302–1422) writes projection GELU or new GRU h, then `vnni.rs::quantize_avx2` rereads the vector to compute max(abs), then rereads it to encode bytes. Return the exact max from `gelu_slice` / `gru_update_avx*`, and call a quantizer accepting that max. Preserve the code equation `q=128+round_ties_even(clamp(x×(127/m),±127))` and `xs=m/127`.

For finite values max reduction is order-independent: it chooses an existing nonnegative magnitude, without arithmetic accumulation error. The vector output must use exactly the existing polynomial/division/add/multiply sequence; no reciprocal, FMA contraction or reassociation. Nonfinite/signed-zero semantics need an explicit matching implementation, or use the old quantizer on exceptional values. Hidden byte codes must not be produced tile-by-tile before the full-vector m is known.

Bound: remove one scan of H values after projection and after each recurrent layer, i.e. `3H` f32 reads/model/row for two layers, plus loop work. No recurrent matrix bytes are removed. This also provides the max for change 3. Barrier: full-sequence bitwise Q8 predictions/state/codes/scales and reset equality; a scalar float reference alone is insufficient.

### 3. Fuse input/hidden integer dot epilogues and gate update in cell-local tiles

**New; larger exact candidate; admit only if stage measurements justify it.** Current `step_dyn_impl` materializes full `gi` and `gh`, then reads rows `j,H+j,2H+j` for each cell. Repack rows as gate triplets for a 16-cell tile. Compute the six independent dots (three input, three hidden) with unchanged individual weight scales and offset sums; convert/scale/add each bias exactly as before; evaluate the existing GRU formula from those register-local values and write h.

Integer reordering is exact under the current int32 bound: `|signed dot| ≤16129H`, only 6,193,536 for H384. The unsigned intermediate/offset correction also fits. For each gate retain the epilogue order `float(dot) × (weight_scale × activation_scale) + bias`, and retain candidate-gate `gi_n + r×gh_n`. Summing input and hidden integer products into one dot is invalid because their scales differ. Reading old hidden codes while writing new f32 h is safe only while the byte-code snapshot remains unchanged until the whole layer completes. Next layer starts after all new codes are ready. Fused max from change 2 can reduce the remaining h scan.

A 16-cell tile has six ZMM dot accumulators, or twelve with even/odd interleaving; this is more feasible than a 64-cell tile requiring many gate/activation registers. Compare one tile size first, inspect spills, preserve generic/ragged and other-cell fallbacks. Code locations: `DynQ8::from_rows`, `gemv_vnni`, `step_dyn_impl`, `gru_update_avx512`, state/scratch construction around rnn_fast.rs 990.

Bound: eliminates full gi/gh write/read traffic, about `48H` bytes/layer or `96H` bytes/model/row for two layers (36,864 bytes at H384). It does not reduce the 2,064,384 Q8 weight-code bytes. Potential value is register residency and scheduling, not an asserted order-of-magnitude bandwidth gain. Demanding full bit parity is appropriate because no quality change is intended. If it spills or loses end-to-end time, retain changes 1–2 and release the pair.

### 4. Conditional exact cache of the unchanged raw projection prefix

**New; exact only under strict keying; screen applicability before coding.** Input order is `[112 raw | 99 event]`. Cache each model's projection accumulator after bias plus raw columns, before event columns. If the sanitized raw vector is bit-identical to the previous row, reuse that prefix and process event columns in the existing order. Even unchanged raw rows still have evolving log counters and recurrent h: features, all GRU layers and output still advance.

Cache key includes sanitized raw identity, model/scales/bias identity and reset state. Each member keeps its own prefix. No later-row lookup is allowed. Persistent cache resets per sequence. The cached f32 prefix must be precisely the old ordered accumulation, not an independently reassociated matmul.

Bound: save raw-column projection work only on raw-identical adjacent rows; useful effect depends on the **unmeasured** raw-repeat rate. `need_prediction` density is not a proxy. First inspect a small existing complete-sequence panel for that rate and estimate `p_repeat × raw_projection_fraction`. Do not implement if projected benefit is negligible. General delta updates `z_t=z_(t−1)+W(x_t−x_(t−1))` are mathematically valid but not bit-preserving in f32; recurrent quantization can amplify drift. They require a separate quality-changing arm and complete-blend grade.

### 5. Selective lower-bit H384 compression with a fixed complete endpoint

**Quality-changing; future arm, not an urgent release prerequisite.** If exact changes do not buy sufficient margin, prospectively freeze a mixed Q8/lower-bit pair rather than optimize a single-model reconstruction error. Native inference must actually consume packed lower-bit codes. FP16 NPZ storage alone is not this arm. Retain projection and sensitive rows/tensors in current precision; assess true dynamic-Q8 recurrence and head behavior after lower-bit weight replacement. Dequantizing nibbles to signed bytes can reuse VNNI but adds unpack work, so smaller code bytes do not prove speed.

Full Q4 code bytes for an H384/H384-head member are `7H²=1,032,192`, before 162,048 projection bytes, scales, biases and state. **Even this does not fit the whole member into 1 MB L2.** H256 Q4 has 458,752 code bytes and a much better single-member working-set prospect; two members still need aggregate accounting. Do not promise a cache-fit cliff for a compressed H384 teacher. The existing FactorF32 route also is not a free solution: one `(3H×H)` matrix uses `16Hr` factor bytes versus `3H²` Q8 bytes, so it beats Q8 storage only for rank `r<3H/16` (r<72 at H384), ignoring metadata.

Barrier: actual producer binary fixture/import identity, package size, final ABI/offline bootstrap, deterministic full recurrent replay, matched official score of the **complete fixed blend** across currently qualified folds/seeds, and one-core callback timing. Lower quality of an individual compressed member does not reject the blend; a gain in weight error alone does not approve it. No new fit/provider spending is authorized by this report.

## Near-term release and bounded experiment plan

The parent-supplied complete matched 400k-callback timings are ZIP4 52.74 µs/row, ZIP5 21.77, ZIP11 74.84, ZIP10 104.48, ZIP12 158.62. Public ZIP4 completed 27m1s and ZIP5 15m41s; ZIP12 timeout at 61m56/647 MB is right-censored, not a completed runtime or model-quality result. These inputs were supplied by ROOT; I did not rerun benchmarks. Hidden row count is unknown. Do not treat 39.4M as known or infer a public score from any local delta.

The candidate/reference ratio changes materially with the reference: ZIP11 is 1.419× old pair callbacks but 3.438× old single callbacks. Scaling the completed portal times gives about 38m20 versus 53m55; those are **sensitivity scenarios, not runtime confidence bounds**. A two-point affine fit can absorb a fixed overhead but is underdetermined as a transferable model with only two different endpoints and CPUs. Zen4 cache/ISA behavior, portal load and callback overhead can change the relation. Actual-time evidence is required for a completion claim. ZIP12's censoring supplies no exact speedup requirement.

Proposed ROOT-admitted sequence, with one owner and no new fit:

1. Freeze ZIP11 as the exact-prediction reference and stage times for projection, gate GEMVs, activation/max, head and Python callback. Separately retain old ZIP4/5 calibration on identical data/sequence/reset boundaries.
2. Actor implements change 1 only; independent critic checks full-sequence bit parity. Run the same bounded panel once. Keep if end-to-end improvement survives matched repeat uncertainty. Actor then implements change 2 as a distinct arm and checks the interaction only if individually useful.
3. Release the strongest supported fixed smaller endpoint once ROOT's whole-ZIP ABI/size/quality/runtime gates pass. Do not wait for gate-local fusion or Q4. If exact source is unchanged and runtime review supports margin, ZIP11 remains the first current candidate; no supported claim yet that it is CPU-safe on portal.
4. Only when stage fraction warrants it, admit change 3 in its own bounded implementation/benchmark window. Gate lower-bit work on a fixed complete-blend recipe and existing resource authority, not on promised gains.

No fresh statistical holdout was created. All exposed evaluation panels remain exposed; all-data final weights have no held-out score. Kernel parity protects the chosen endpoint's prediction identity, not its public transfer. This report proposes computation experiments; ROOT retains admission and release ownership.

## Low-value or invalid shortcuts

- The reported 19,901/20,000 need-prediction density leaves at most 0.495% of rows on which a non-stateful head could be skipped. Even skipping the **entire** callback on those rows would save less than 0.5%, and recurrent/feature state cannot be skipped. Scored-label density is irrelevant to permitted prediction rows.
- Generic cross-seed common-backbone fusion is invalid for these actual artifacts. Wide stacked heads share a forward pass only after exact equality of normalization/projection/recurrent tensors and recurrent initialization is established.
- Reusing one Python output ndarray can corrupt a consumer that retains references. `PyArray1::from_slice` allocates on predicted rows; optimization needs actual scorer ownership/copy behavior and a retained-result regression. Given ~99.5% prediction density and large native models, it is not the first rescue arm without measured callback evidence.
- A replacement sigmoid/tanh approximation, reciprocal division, FMA contraction, low-rank weights, delta-cached projection, or late hidden-state refresh is quality-changing. Test complete recurrent sequences and full fixed blend; do not approve from tiny activation error.
- Wall-clock membership changes/governors, future peeking, row dropping and altered reset boundaries are outside this deterministic task.

