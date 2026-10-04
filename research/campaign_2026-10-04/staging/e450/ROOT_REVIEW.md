# Independent root review of E450 verifier

The initial source-only verifier had two integration defects: broadcasting a 20,000-row warmup vector directly to a [20,20,000,2] target array, and expecting stdout JSON from the frozen reset checker, which writes its result only to a file. The initial source was never run and is preserved in `pre_review_v1/`. The revised wrapper uses an explicit sequence/target axis and reads the exact reset result file after successful subprocess exit. The frozen callback, model, package, reset checker and grader were unchanged.

A bounded integration worker passed the full-size synthetic mask and file-only child result checks. The actual native verifier then completed all 20 full streams (400,000 rows), three fresh precision/ISA callbacks, exact targets/masks, bitwise v2/v3 results, actual parent/head trunk and merge equality, and A–B–A reset/future-suffix checks. Maximum i16/native-f32 error was 0.0000978410; official WP difference was 0.00000002859. Native f32 versus arithmetic average of separate CUDA FP32 caches differs by 0.000580743 max and 0.00000301376 WP on this callback panel; that is a recorded diagnostic, not the quantization parity gate.

The worker exited 0 with verified teardown, elapsed 164.147 seconds and peak RAM 580,894,720 bytes, with no GPU use or OOM events. Package identity is unchanged: SHA256 `c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a`.

CPU timing and public score remain unmeasured. These checks prove parity/reset/causal packaging, not manual-upload runtime readiness or predictive improvement. Raw report: `checks/e450_seed4_native_parity_v1/readiness.json`; worker: `runner/attempts/e450-seed4-native-parity-v1.json`.
