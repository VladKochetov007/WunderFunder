# xteam_h4 CPU policy implementation review R1

## Decision

PASS for the exact-engine and complete-sequence parity claim on the reviewed Intel VNNI host. The review found no blocking policy-selection, package-identity, or state-reset defect. This is implementation evidence only: it does not show a quality gain or a portable runtime win. A remains the measured local runtime baseline; B and C were slower in the reported single local callback sample, and D was slightly slower than the combined A control mean.

## Frozen source and artifacts

The reviewed source is the clean actor worktree commit `1dcc9e4936eee042a8542983b14c05d000d4980a`, based on native revision `7800d30473e3810e12d9ba22596f9fd6df5626ca`. Its package manifest SHA-256 is `90206bb263f914d0b766a541e02a0e19a9601b26f520baf1305ed36d9c5b451d`; the measured result is `eeab1826b8a5178aae4b46be208ef4c594c09c8ea9ecb06fa1d1a751d15a614b`. The original `xteam_h4.zip` hash is `f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27`.

The original archive identifies four ordered members, each at coefficient `0.25`, and `precision.txt` is `q8vnni`. I independently checked the original archive and the rebuilt-A/B/C/D ZIPs with Python 3.11 CRC validation. Every derivative keeps the exact entry inventory and all four weight NPZ files byte-for-byte; `members.json` is unchanged. The only modified entries are the precision file for B/C/D, all three native-library paths, and `solution.py`. The callback diff adds only startup checks that reject an environment precision override and verify the native model's reported precision.

The native library hash in the manifest matches the built file on disk (`cd8b9692d102e5173fbe3b89a15373dcccb63382892f7621c0b2a1382f0ba362`). All six native-source and three campaign-script hashes in the manifest match their frozen files. The package hashes independently match the manifest: B `5aed417e…29dae78`, C `c17c6415…5027835`, and D `e11ff6a3…c05951`. The two earlier package-builder failures are retained and correctly classified as builder defects; the final payloads passed CRC and exact-member checks.

## Policy and dispatch audit

- A remains the original shipped `q8vnni` archive. The separately rebuilt-A control uses the new library and the same package assertions, so it tests whether replacing the library changes the baseline callback.
- B maps `q8vnni256` to the strict `Vnni256` path. Construction rejects unsupported CPUs instead of falling back; the step dispatch calls the 256-bit kernel, which uses `_mm256_dpbusd_epi32`. The release-library disassembly contains EVEX-encoded `vpdpbusd ymm` instructions (three recorded sites), alongside ZMM sites. B was observed as `q8vnni256` on the VNNI-capable host.
- C maps `q8vnni_blocked` to the strict AVX-512 VNNI blocked schedule. Its kernel uses four independent accumulators per 16-row block. The callback observed the requested policy.
- D maps `q8vnni_fused` to the strict GRU-only fused path. Each layer's input and recurrent products retain separate quantized activations, scales, biases and output accumulators; the recurrent product reads the pre-update state. The fused integer arithmetic is aligned with the two separate A products. The callback observed the requested policy.

`solution.py` reads the policy from the package file and fails if an environment override differs. The harness also reads that file before import, removes the precision/native-level overrides, and checks the native `precision` getter after construction. The native constructor rejects unsupported B/C/D policies rather than silently selecting another kernel. The callback receipts report matching requested and observed policies and the exact loaded native-library hash.

## Complete-sequence and reset evidence

The recorded callback probe uses Python 3.11.16 on one pinned logical CPU of an Intel Core i9-11900H, with both complete 20,000-row validation sequences (row groups 0 and 1; sequence IDs 256678 and 751005). The input/data hashes match across policies. Every output is float32 with shape `20,000 × 2` and is bitwise equal to A. I independently compared the retained arrays for A, B, C, D, A2 and rebuilt A: both sequence arrays are exactly equal across all six runs, and their file hashes match the receipts. The two fresh-model replay checks pass for each callback. The same predictor instance handles group 0 then group 1; the native callback resets feature, normalization and recurrent state when `seq_ix` changes, and each output also matches replay from a fresh predictor.

Rebuilt A matches the original A on both full sequences, controlling for the fact that B/C/D package the newly built library while original A uses the archived library. The first and final original-A callbacks also match. This supports exact parity for these complete sequences and reset scenarios, not every possible input, CPU, or library state.

## Runtime and scope limits

The result receipt records the same two sequences in A→B→C→D→A order. The local callback means are 2.039 s (first A), 3.056 s (B), 2.422 s (C), 2.177 s (D), and 2.098 s (final A); the combined A mean is 2.068 s. These are one sequential local sample per sequence and policy. Matrix and cell probes likewise do not establish an AMD-host result. No AMD/EPYC or competition runtime, quality grade, public score, or external upload is part of this review.

All three derivative packages replace the `v2`, `avx2` and `v3` shared-library entries with the same rebuilt library. The specialized policies deliberately require AVX-512 VNNI; their lack of fallback is correct for policy identity but means deployment must first verify that the target CPU supports the selected policy. This is a portability constraint, not a parity failure on the reviewed host.

## Checks performed

I read the frozen [report](../../campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/REPORT.md), [manifest](../../campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/PACKAGE_MANIFEST_R1.json), [results](../../campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1/RESULTS_R1.json), callback receipts, build/test log, assembly excerpt, matrix/cell logs, package builder and benchmark source, and the policy dispatch/kernel changes. Read-only checks independently verified archive/library/package hashes, manifest source hashes, ZIP CRC/member inventories, original weight and `members.json` identity, exact prediction-array equality, and receipt policy/reset fields. I did not rebuild, rerun callbacks, run a fit, or perform provider actions. The actor's recorded Rust test run was 58 passed, zero failed and one ignored; it is source verification evidence, not a test rerun by this reviewer.

Related hypothesis: `H-CPU-POLICY-20261014`. Source result: `inference_policy_r1` at commit `1dcc9e4936eee042a8542983b14c05d000d4980a`.
