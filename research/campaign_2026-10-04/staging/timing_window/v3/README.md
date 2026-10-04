# Timing window v3 staging

This packet replaces the failed v2 timing attempt with new attempt and output identities. The only CLI change affecting the workers is that path operands are absolute paths under the repository root. The E450 and E462 callback timing source files remain pinned to their v2 hashes and loops; no timing results exist yet.

The controller waits for `wnn-e426-clock-queue-v3.service` and the exact successful E426 provenance preflight, three fixed epoch-3 exports, and paired fold-0 score. It then rechecks E462 build/parity/score and E450 parity readiness, waits at most 900 seconds for startup load1 <= 0.3 with no CUDA compute/utilization, and runs seed4 i16 first followed by public q8r/q8x. The workers retain CPU 15 affinity, load1 <= 1.0, five alternating paired trials, and 60,000 rows per trial. Each worker has a one-CPU, 2-GiB RAM, no-GPU, 1,800-second bound. There is no automatic retry or reuse of v2 outputs.

The systemd unit is staged only. Root review, sealed specs, E426 queue-v3 completion, and the quiet-window admission remain required before any execution.
