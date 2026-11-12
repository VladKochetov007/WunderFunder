# Feature-only serving checks and native changes

- `cpu_check.py` verifies delivered ZIP/source/weight identities and compares the callback to FP32 references, checks ISA imports/reset/causality/blending, and measures one-core time. Its score computation is disabled.
- `native_opt_check_r1.py` is the preserved first native comparison attempt; its misnamed `args.label` failed before inference.
- `native_opt_check_r2.py` corrects the three parsed-argument references and compares every member/callback output in separate library subprocesses. Supply `--expected-candidate-sha` explicitly for a new library.
- `native_opt_check_r2_cli_smoke.py` exercises both child CLI dispatch paths and candidate-hash parsing without loading models.
- `bench_member_core.py` isolates recurrent cores to distinguish matrix cost from feature/encoder/cache switching; it does not represent the full tiedlead callback.
- `native_precision_panel_r1.py` compares existing Q8 modes without changing weights or blend; all tested variants retained bitwise outputs and were slower or neutral.
- `native_quick_bench_r1.py` measures candidate library callback time/effective precision; quick timings alone do not prove parity or serving admission.
- `patches/TIEDLEAD_FAST_GELU_R1.patch` reuses the existing SIMD fast-GELU implementation with scalar fallback and ragged-width parity coverage.
- `patches/FINALTL3_CACHE_ORDER_R1.patch` alternates full-mix member execution order and supplies the next-member prefetch hint, preserving each member's temporal order and canonical blending.

Rust patches target exact6493556 Git blobs. Actual combined native library SHA256
is `085dfecd1c9eecf1287606222599f25c9c89edc63d285a3ea2bc82de224032b9`.
CPU comparison passed bitwise checks but still exceeds the60-minute envelope.
