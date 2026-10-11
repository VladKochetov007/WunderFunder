# Ready submissions — October11

Selected from our research and local evidence. Upload these ZIPs in this order:

| Rank | Submission | Exposed fold1 local score before serving quantization |
|---|---|---|
| 1 | [Student + scratch, fixed50/50](8_dm_t3_a9_e6_muonall_prop0_e4_s4_pair_fold1_q8vnni.zip) | **0.67329** |
| 2 | [Muon-all/prop0 four-epoch scratch seed4](9_muonall_prop0_e4_s4_fold1_q8vnni.zip) | **0.67195** |
| 3 | [Three-teacher alpha0.9 distilled student](7_dm_t3_a9_e6_fold1_q8vnni.zip) | **0.67103** |

All use Q8VNNI. The pair combines the two independently trained endpoints; the student and scratch singles give transfer diagnostics for its members. These are valid fold-trained exploratory submissions while the all-data finals run. The fold was repeatedly inspected and the scratch seed selected, so the scores are not fresh confirmation or expected public scores. Public LB confirms transfer.

The actual delivered ZIPs passed Python3.11 offline CPU replay, finite outputs, causal sequence reset, determinism, three ISA variants (v3/avx2/v2), bit-exact cross-ISA predictions, static ABI (GLIBC2.34), CRC, checkpoint/tensor binding and independent review. Pair v3 replay used30 sequences/600,000rows; other checks used10sequences/200,000rows. Sample callback timing was approximately33us/row for singles and65–96us/row for the pair under concurrent research load; this is a sampled check, not full-evaluation runtime certification. Quantization accuracy relative to the float evaluator is not regraded on the full fold in this release.

[READY.json](READY.json) records hashes, order and exact checks. Build receipts retain fixed members, checkpoint epochs, source hashes and the exposure note. Review limit: the critic could verify both exported checkpoints and the student initializer, but could not rehash distillation-cache payloads and every teacher checkpoint because some remain remote; their identities/hashes are recorded by producer provenance. No teammate weights are included.

Keep the remaining daily slots for later all-data final packages or informative new research results. Parallel research continues.
