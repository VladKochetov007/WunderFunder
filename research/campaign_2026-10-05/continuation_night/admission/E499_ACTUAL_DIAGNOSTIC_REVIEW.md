# E499 actual clipping diagnostic: independent result review

Disposition: **execution and provenance GO; E489 full-fit admission remains NO-GO**.
The bounded diagnostic completed under its reviewed limits and preserved all
three registered seeds. It made no predictions, scores, or training fits.

The sealed spec SHA-256 is
`d91c0888ac4b9a7ef1c96cd4f353e7107fa0d878a54c264647702a19c0820088`.
Runner attempt `night-e499-clipping-diagnostic-v1` finished with exit code 0,
26.173896697 charged seconds, 2,576,916,480 peak host bytes, and 421,527,552
peak observed VRAM bytes. Its child cgroup OOM/high/max counters remained zero;
parent counters were already nonzero before launch and did not increase. The
model report records 108,605,952 peak allocator bytes and 419,495,936 observed
device bytes after collection. The worker used one visible NVIDIA GeForce RTX
3060 Laptop GPU, a 2-GiB allocator cap, a 3,000,000,000-byte device stop, and a
3,000,000,000-byte host cap.

The fit-only loader returned exactly the 24 registered fit-role groups, with no
evaluation groups loaded. The fit dataset hash is
`2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`; the cohort
hash is `068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2`; and
the saved E487 root-gate proof hash is
`4fced259c38beae9a74652cf486ccd952d62c830e5927e5316be161aba6c6cac`. The
diagnostic source identities match the prior independent review:

| Source | SHA-256 |
|---|---|
| `activation_transfer/e489_clipping_diagnostic/DESIGN.md` | `be338571e8031818ef85647a789fa8bdef28ec6e47fbbfcf5647e4804c608277` |
| `activation_transfer/e489_clipping_diagnostic/diagnose.py` | `4315ce41714550f2cc1f2500c767f2b3ef6797a78b3dd21d21e5804936e14249` |
| `activation_transfer/e489_clipping_diagnostic/diagnostic_math.py` | `ef7dc90425aafa20433278e03e8d1dc8156af00daf91208c1b37b0568fbeecb5` |
| `activation_transfer/e489_clipping_diagnostic/frozen_runtime.py` | `a172b96f7781ee458ae56ab67a400c04b246521996bb309884c8ec870cd7860d` |
| `activation_transfer/e489_clipping_diagnostic/seed_probe.py` | `3d63c7cb089f10397b7069a179e8779cebbe7fa9adab9970261284a76dfa1146` |
| `activation_transfer/e489_clipping_diagnostic/test_diagnostic_math.py` | `95cefbaa76fb665fa9c7a99b1622889e85f9fc5bb81d172d6dd14bf2fa08ba29` |
| frozen `fp32_activation_duration/fit_fp32_duration.py` | `af2ad34d8bac617fe5f53e28a2120c5d59a303888b54937e35e27edd1b925cdc` |
| frozen `fp32_activation_duration/gpu_probe_fp32_duration_v2.py` | `7ee1bf79dac158d7aaa6530f55173126ad5e8f8df1c4e50832d83e9b0ca19b20` |

Across all seeds, native/custom forward values, recurrent states, input
gradients, and every parameter gradient passed their registered allclose
checks. Gradient norms for seeds `271836`, `271837`, and `271838` were
`2.06236887`, `0.89045572`, and `1.55643559`. Clipping at norm 1 was active for
the first and third seeds and inactive for seed `271837`. That seed failed the
original per-seed clipping coverage gate. The native/custom one-step update
comparison also failed the original max-absolute AdamW threshold for seed
`271837` (`0.000148083` versus `0.000020000`); it passed for the other two.
The report therefore diagnoses one naturally sub-threshold seed but does not
repair or pass the frozen E489 prerequisite.

Primary result and raw-array identities:

| Artifact | SHA-256 |
|---|---|
| `actual_v1/report.json` | `7c0705d4833490e04a328780a4c52c11624d5e9a072e561f8c5ee80e587cba9d` |
| `actual_v1/status.json` | `cd504f502e9beb30936a5213f5601ccd3b80da3787b3264ee49e5f7f8d137da4` |
| `actual_v1/summary.json` | `4949e0c926b1df05e48f2e1c3fd0c65928c74edecbf43146b6d7faf474fea7e6` |
| seed `271836` JSON | `d2ba0f2865ceeb174b44260b590c5a83a1903b2ebb95c11b2225704899a0e0c7` |
| seed `271836` raw gradients | `cec790f505036846e5da1c65d15a2af19dd0805971471ef88330a43a4e972ce0` |
| seed `271837` JSON | `e45c4a23ba5f02c10128ac7cd41b3893e4edc46ef715ff3fbb2a9b882a7c22c5` |
| seed `271837` raw gradients | `46f248ac7dc29ba5abcf69598e27a5c670aec337bcd07a7ce323b4dc06e4421b` |
| seed `271838` JSON | `eb378af346b9659d8be330e472e4eab25d7c46d63976b751266a467a7060ce52` |
| seed `271838` raw gradients | `ecd1b286696c11e022c23cb68c20c94884e36766215cd4f4e167c64c2dc7f7ae` |
| runner attempt | `230c1691f5923f4f82175215cd055fbebcdd52d4d3778a74572d1e9dbd1e3bc1` |
| runner log | `05ad37d00bfecc3edb53d09eb8a06593b9f92a72773cfbbf58b7dbf07c88d3aa` |

The full report includes initial-state hashes, per-seed batch hashes, all parent
source hashes, update-delta details, and the explicit claim boundary. This
diagnostic grants no E489 pass and does not authorize its full fits.
