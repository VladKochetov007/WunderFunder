# E474 probe output-path incident

The first real-device worker exited0 and generated `gpu_probe_v1/probe.json`, with12 finite changed optimizer updates and baseline output/state/gradient parity. Root mistakenly registered `gpu_probe_v1/result.json` in the sealed output list. The runner failed closed with category artifact_missing and retained the job quarantine. Peak observedGPU387973120bytes/RAM1211359232bytes,8.571925834s, verified teardown and noOOM. No full fits were admitted.

The repaired v2 spec changes only the output-directory identity and expected artifact filename to `probe.json`; numerical source, tolerances, inputs, and resource bounds remain unchanged. Re-run the exact real-device probe and require a successful runner-validated artifact before full fits. The original failure is retained and counted in budget. Root owns this registration error.
