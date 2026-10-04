# E465 AdamW formula discriminator

Actual `e465-adamw-formula-referencecheck-v2` passed all7,460 used scheduler indices against Torch FP64, under a oneCPU/2GiB bounded worker:2.576seconds,388,153,344B peakRAM, noGPU/OOM, verifiedteardown. Limits1e-10 for parameters andbothmomentstates. Maxima: parameter3.0531133177191805e-15, firstmoment4.440892098500626e-16, secondmoment1.1102230246251565e-16. Exactsource expressions, functionASThashes andallsourcehashes are in `checks/e465_adamw_formula_referencecheck_v2.json`.

The earlier `e465-adamw-referencecheck-v1` FP32-versus-binary64 failure remains a failure. Matching algebra in binary64 does not establish actualJAX/Torch FP32 operation fidelity. The latter is a separate compiledscan diagnostic awaiting privateCPU execution. Neither result substitutes for full eight-device productionshape profiling or realTPU training.

Source custody: a later actor reporting-only draft edit touched the sealedv2 path after successful execution. Root restored its exact8,274bytes and verifiedSHA `ea1f18802aeb552929bb6fb566d1300c55502fed201010fab9ce6243e556ad51`; prospective changes were retained only as distinctv3source. No numerical recurrence/result changed.
