# R4 delivery erratum: outer bootstrap allowlist

This is an actor-owned delivery erratum, not an independent critic report. The R4 offline check confirmed that both target-cache paths appeared in the profile job inputs and outer kernel mount map, but it did not execute the generated outer bootstrap against the actual bundle manifest. That missed a separate boundary: the unchanged bootstrap forms its allowed-path set from `manifest.files` plus `spec_paths` and rejects any external mount path outside that set.

R4 added `cache/train_targets.npz` and `cache/valid_targets.npz` to `external_mounts` but left them out of `manifest.files`. Exact kernel v2 therefore stopped with `ValueError: Unsafe external input mapping` before launching the profile runner. The terminal error is linked and hash-bound in [R4's provider receipt](../student_shape_pair_f0_r4/PROVIDER_TERMINAL_R4.json); it is an infrastructure failure with no model evidence.

R5 adds the two producer-bound descriptors to `manifest.files` and exercises the actual outer bootstrap with the real archive and fake physical Kaggle mounts, intercepting the child before trainer/GPU execution. Future containerized profile packets should test this manifest-derived allowlist and symlink-mount path as part of their preflight.
