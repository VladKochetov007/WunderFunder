# Online relationship delivery handoff R1

## Status

This handoff text was written before panel admission. Root later committed the admitted immutable recipe, source map, invocation and execution snapshot. The current worker is identified in `PANEL_EXECUTION_R1.json`; its output paths are create-only and must not be launched again. Use the root checkout/source freeze and wait for terminal collection and independent grading before interpreting a result. This file preserves the finite command and its original safeguards for provenance, not as a new admission.

The launcher verifies the exact postboot profile receipt hashes, all 27 profiled source-file hashes, and that the admitted recipe preserves the frozen recipe fields from `PROFILE_RECIPE_R3.json`; only root's admission fields and execution metadata may differ. It refuses nonempty parent-cache paths, preexisting output directories, or less than 40 GB of physical scratch. Root's outer thermal guard holds the campaign-wide `gpu.lock`; the child script does not try to acquire it again. The focused publication branch preserves four canonical dependencies whose hashes differ from this admitted map, so the exact preflight refuses that branch.

## Finite execution

[RUN_DELIVERY_R1.sh](RUN_DELIVERY_R1.sh) runs only these commands, in order:

```bash
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -u \
  /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/online_relationships.py \
  --root /home/vlad/development/wunder/wnn_connectome_starterpack \
  prepare-cache \
  --recipe /home/vlad/development/wunder/wnn_connectome_starterpack/research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json \
  --fold all --device cpu

/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -u \
  /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/online_relationships.py \
  --root /home/vlad/development/wunder/wnn_connectome_starterpack \
  fit-panel \
  --recipe /home/vlad/development/wunder/wnn_connectome_starterpack/research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json \
  --device cpu
```

The admitted run used root's worker guard: CPU quota 190%, two numerical-library threads, 4,000,000,000-byte memory maximum, swap disabled, GPU hidden, strict FP32, physical SSD scratch, thermal pause at 80°C and resume only at or below 74°C with one-second polling, and a 7,800-second ceiling for cache preparation plus all twelve fits. The script has no queue, retries, profile mode, parameter sweep, or grader invocation. A failure stops the sequence and preserves any create-only partial artifacts for diagnosis; there is no automatic missing-head replacement. Any later execution requires a fresh admission and new output paths.

Expected outputs are two full parent-cache directories under `/home/vlad/development/wunder/research_tmp/online-relationships-20261022/parents/` and twelve run directories under `/home/vlad/development/wunder/research_tmp/online-relationships-20261022/runs/`. Each cache has fit, feature-only evaluation and parent-prediction payloads/sidecars. Each readout output must contain epoch-2 checkpoint, evaluation prediction, prediction sidecar and provenance. The launcher checks the complete inventory before returning success. The independent grader then owns the separate exact evaluation step; reserve at most 600 seconds for that step after the independent cost review. No target-quality scoring occurs in the actor launcher.

## Pinned evidence

- Frozen scientific source revision: `17470a770f2a6aa25be232c017bff45b5d56ad97` on `experiment/next-cycle-recovery-20261021`; reproducible root packet/admission/launcher commit: `33c6e33830a89187aaaf08116850864250feed56`. The earlier postboot engineering profile ran at its separately recorded source revision.
- [Postboot profile result](POSTBOOT_PROFILE_RESULT_R1.json) SHA-256: `dcbfb56e10ddef133eb9c6c19c5c20068571051416d8d96b8dc7a1c24ba3597e`.
- [Postboot execution receipt](POSTBOOT_PROFILE_EXECUTION_R1.json) SHA-256: `c635a3e372af9757fac4da534c22ba8cdf35708fb56908f77f940227498be854`.
- The result binds 27 source-file hashes and records no evaluation-label reads or target scoring. It is an engineering cost profile, not a scientific result or execution admission.
- Shell validation is limited to `bash -n`; the delivery script itself has not been run.
