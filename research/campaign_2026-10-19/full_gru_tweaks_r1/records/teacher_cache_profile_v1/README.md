# Actual teacher-cache inference profile V1

Root accepted the bounded engineering profile for S3 epoch 2. Invocation `983d65b923c54f05a25aa98d61e0bb1d` launched at 2026-10-19 21:57:10 UTC, exited successfully at 21:57:19, was captured at 21:57:53, accepted at 21:58:45, and was exactly stopped at 21:58:46. The source is commit `88becca892b3d6b97022cf06076c84a7ab165974` with 167 manifest-pinned files.

The accepted [profile receipt](capture/PROFILE.json) reports 16 sequences × 20,000 rows × 2 raw main outputs: 640,000 finite FP32 values. It records the first-16 fit-train selector, raw features 112/model input 211, H384×2, batch 16/chunk 4,000, strict FP32, TF32 and autocast off, and one torch thread. The stored normalization identity is 211-wide FP32; the root acceptance records that it matches the stored S3 native logs. The capture contains the profile and admission/launch metadata plus `unit.log`; no fitting, parameter update, checkpoint, prediction array, or quality grade was produced. This is an inference engineering profile, not a quality result.

## Evidence navigation

- [Admission and resource gates](ADMISSION.json) and [native launch](LAUNCH.json)
- [Raw native transport](STDOUT.raw.json) and [transport digest](TRANSPORT.json)
- [Captured unit, source pins, and profile payload](capture/STDOUT.raw.json)
- [Accepted root reconciliation](ACCEPTANCE.json)
- [Exact-stop capture](exact_stop/STDOUT.raw.json) and [stop transport](exact_stop/TRANSPORT.json)
- [Independent metadata review](../../preparation/TEACHER_CACHE_PROFILE_ACTUAL_REVIEW.json)

## Resource observations

| Scope | Observation |
| --- | --- |
| Admission envelope | 1 CPU; MemoryHigh 7,500,000,000 B; MemoryMax 8,000,000,000 B; swap 0; runtime 600 s |
| Inference | 3.8326194 s; process RSS peak 2,232,668,160 B |
| CUDA | RTX A4000; 4,294,967,296 B allocator cap plus 1,000,000,000 B reserve; peak allocated 901,167,616 B, peak reserved 1,140,850,688 B |
| Native service | 9.033604 s wall, 7.047671 s CPU |
| systemd `MemoryPeak` | 2,031,616 B, recorded separately; this field is not full runtime RAM |

At capture and exact stop, the parent memory event counters were unchanged and the owned-populated map was empty. These observations do not establish a memory cause. The exact-stop receipt shows the accepted invocation exited before cleanup and the unit was `not-found`/`inactive`/`dead` afterward.
