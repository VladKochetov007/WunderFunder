# Conditional Z5 Q8 blends (2026-10-12)

This packet contains two fixed native Q8VNNI packages prepared for root review: D, a three-member mixture, and E, a four-member mixture. Their weights were specified before packaging; no fitting, blend search, score computation, upload, or provider action occurred. These are runnable trial artifacts, not quality-selected submissions.

| Candidate | Fixed mixture | ZIP bytes | ZIP SHA-256 | Ten-sequence callback profile |
|---|---|---:|---|---:|
| D | 0.25 H390 s3 + 0.50 merged Z4 + 0.25 merged Z5 | 13,178,260 | `bf9f48dbb05668181038d28386a076774d41146efb7c4f1680fcd724a303df09` | 113.60 µs/row |
| E | 0.1875 H390 s3 + 0.1875 H390 s4 + 0.375 merged Z4 + 0.25 merged Z5 (`0.75*C + 0.25*Z5`) | 16,805,643 | `2aed2137c347344146e3481c83388d64bd5060b1c13bb66554b0bf9cc0027024` | 152.62 µs/row |

The merged members preserve their exact independently trained source pairs:

- H390 s3: `vt_all_h390_ds_homogeneous_s3_r3@ep5`
- H390 s4: `vt_all_h390_ds_homogeneous_s4_r3@ep5`
- Z4: `bd26_h230_faithful_student_seed99_muonparent_v1_core_identity_v2@ep1|z4_seed99_student_hft_alldata_v2@ep0`
- Z5: `bd26_h230_faithful_student_seed9_muonparent_v1_core_identity_v2@ep1|z5_seed9_student_hft_alldata_v2@ep0`

Create-only build receipts bind each member to its source checkpoint and provenance sidecar, exported tensor hashes, scorer source, native library hashes, ZIP member hashes, and ZIP bytes. The packages use the existing precision-aware scorer with root-level `solution.py` and `precision.txt=q8vnni`; both are below the 20 MB cap.

The Z5 source audit shows a seed-9 core followed by a frozen-trunk all-data head fine-tune. That head fit used 1,505 supplied validation groups. H390 and Z4 are also all-data endpoints. A targeted cache/artifact/current-campaign search found no matched clean-CV grade for the Z5 endpoint, so D/E have no cross-validation quality evidence and this packet makes no quality claim.

The original Z5 i16 ZIP (SHA-256 `60834d3190b76832f038abbdc505c6ac526d44b8f3a191c2e980e2f14f1ee1cc`) and the 2026-10-11 Z5 Q8VNNI ZIP (SHA-256 `75a7eb3fed92d32157e5ce8d837c95684178b184d59eda6f55e1fab0e7b95044`) contain byte-identical `weights_0.npz` payloads, SHA-256 `b302153aa55701475e6a466ccacbf5d05c5142a2294eafd89675ab7386d4df40`. The original i16 upload has the user-reported portal result 0.6664 (36m27s, 625 MB, portal `16zlko1e`). That result is **not** a Q8 score: quantization changes served predictions. The Q8 receipt proves only callback/reset behavior, not quality.

Each exact D/E ZIP passed two reset/replay passes over ten complete 20,000-row sequences (400,000 callback calls) using CPython 3.11.16, native v3, and Q8VNNI, under one CPU, a 1.5 GB memory cap, and a 120-second limit per profile. Warm-up callbacks returned `None`; required rows returned two finite predictions; prediction hashes were bitwise stable after reset. The exposed validation fixture is the unchanged row groups 0–9 of `valid.parquet` (SHA-256 `21bef3ff373b238030be700cfe857380c616a58da3975eed81675fdb99998830`). It was used for capacity only; no official metric was computed. Cgroup peaks were 216,834,048 B for D and 225,468,416 B for E, with no OOM/max events.

The local Intel i9-11900H timings are capacity measurements, not Kaggle EPYC or portal runtime predictions. For context, prior A/B/C local profiles were 44.97/49.82/83.33 µs per row and their user-reported portal runtimes were 24m31s/25m56s/36m36s. D/E have no public result. No runtime or score extrapolation is made.

`PACKET_MANIFEST_R1.json` summarizes the source, package, profile, and bounded resource receipts. `profiles/` stores the fixed input identity and exact-package callback receipts. The copied builder, callback runner, and `utils.py` are unchanged from commit `8129e61c3cdd3c4f1317049ba37711fee4447bd9`, with their hashes recorded in the manifest. Root review is still required before any integration or submission decision.
