# E489 provenance and admission addendum

Registered after the independent blind critique and before E489 implementation.
This addendum makes the critic's provenance and prior-exposure gates explicit.

## Frozen model and optimizer identities

- E489 design: SHA256 `0859e6ac528b850452183afc46ab28358cac13c97ace5cb0350e413c2b9a7090`.
- E489 blind critique: SHA256 `e4a8cae780b1e49739dd9474e3c13af459ef75db4bd39eb6e224cffe0ec76e70`.
- E485 actor topology/candidate transform: `activation_duration/fit_duration.py`, SHA256 `d9a1b9d243a9c9e50082fe0d9e7d98186d63cb0c566fcae0a20c00ccd42c4b48`.
- E485 native recurrent primitive: `src/wnn/nn.py`, SHA256 `388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3`.
- E485 input/candidate scan: `recurrent_activation/fit_recurrent_candidate.py`, SHA256 `998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf`.
- E487 diagnostic optimizer implementation: `muon_orthogonalization_precision/precision_replay.py`, SHA256 `480f2ae6d3aa09a3f972a3ad9fc682e7a92e07843d6e002434b2130ec75415d2`.
- E487 production optimizer/group source: `src/wnn/optim.py`, SHA256 `620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991`.
- E487 actual saved replay result: `muon_orthogonalization_precision/gpu_probe_v2/result.json`, SHA256 `46b1edfc5b63d2da93ddc444b8c40142d4e57e16b6ceb4f80f33bc4fffd89714`.
- E487 pre-run saved-delta verifier: `muon_orthogonalization_precision/verify_probe_v4.py`, SHA256 `f717d0d30d1c6e1456b80bdd31d76896633a588ea22e5b9243652af5f92391e6`; independent review `VERIFY_PROBE_REVIEW_V4.md`, SHA256 `0685ce65335c956a94003700827798c8d3080d746bb876362074cc19545f9132`.
- E487 replay-delta export: `muon_orthogonalization_precision/gpu_probe_v2/replay_deltas.npz`, SHA256 `591a2eb248249f1925eca71c85031639b0085fbf58b148144f32bfd3b9a7c152`.
- E487 independent root arithmetic gate: `checks/e487_saved_delta_reproduction.json`, SHA256 `4fced259c38beae9a74652cf486ccd952d62c830e5927e5316be161aba6c6cac`; system verification receipt SHA256 `8820f1eb3edf954f1d2279843be85d418a8883fad39fff3f9a9689396eeae6a9`.

The candidate is fixed `alpha=0` versus `alpha=0.1` under the exact registered E485 reset-after recurrence: `u=tanh(v); n=u+alpha*sin(pi*u); h'=(1-z)*n+z*h`. Both E489 arms use the E487 FP32 Muon orthogonalizer, not the E485 production BF16 orthogonalizer. The E487 helper is reused without modifying either E487 or production optimizer source; any copied wrapper must retain the pinned formula and source attribution.

The independent E487 saved-delta arithmetic verification is a hard precondition for E489 CUDA probing and all later fitting. The pinned root proof binds `study=E487`, `status=all_checked_values_match`, the exact replay source/result/export, verified-v4 source and review hashes, actual runner attempt and system verification receipt hashes. It records nine finite delta arrays, FP32 strict-gate pass/BF16 fail, no new fits, and verified teardown/unchanged OOM counters. The E489 probe actor must recheck all proof and receipt bytes before CUDA setup. The diagnostic's strict outcomes are preserved; this does not retroactively admit E485. The root must explicitly admit the E489 probe before any CUDA work. Full-fit admission additionally requires the three-seed probe, timing projection and complete runner/source receipts specified in `E489_DESIGN.md`.

## Prior evaluation exposure check

Seeds 271836, 271837 and 271838 have no E485 evaluation-role scoring receipt. The prior E485 fit-only real-data preflight is `activation_duration/cpu_preflight_v7/fit_manifest.json`, SHA256 `bf23c9bdfb27a3f9367e8d305bcfa632bf410e6c0990fa2670e6538062bf3d8f`; it records `test_rows_loaded=false` and contains only fit-only probe arms. The E485 actual CUDA probe terminated at strict optimizer-update parity before any full trajectory or evaluation-role read, as preserved in `ACTUAL_PROBE_V3_FAILURE.md` SHA256 `1347acc2cb27f5b1a10c6a5aeda1962e7da77973a70a0865e203c7a64dde9adf`. No E485 full-fit or scoring attempt is authorized by this record.

E487's single-batch diagnostic uses seed 271836 and fit roles only. Its immutable result records `fit_only=true` and `test_rows_loaded=false`; its strict precision outcome remains `strict_precision_pass=false` in `gpu_probe_v2/result.json`. E489 is a new FP32-orthogonalizer study. E487 does not provide a predictive result and cannot retroactively validate E485.

Before E489 actual CUDA probe, controller admission must independently re-read these bound receipts, confirm the E487 saved-delta proof, confirm no intervening evaluation-use receipt for the three E489 seeds, and bind all new source/input hashes. This addendum does not itself authorize execution.
