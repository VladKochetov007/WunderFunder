# Online relationship delivery handoff R1

## Status

Prepared only. No full parent cache or scientific readout fit has been created. The postboot R3 engineering profile completed 16 fit and 8 evaluation sequences per fold, all 12 disposable heads, in 71.032 seconds. Its known full-panel projection is 4,755.379 seconds. Root's independent cold-I/O and exact grading cost review now passes at 8,311.440 of the 9,000-second inclusive ceiling; the bounded delivery gets at most 7,800 seconds, leaving the separate grader up to 600 seconds. The root must still freeze the separate immutable `EXECUTION_RECIPE_R1.json` with `status: admitted_for_execution` and `admitted_new_fits: 12` before the launcher can proceed.

The launcher verifies the exact postboot profile receipt hashes, all 27 profiled source-file hashes, and that the admitted recipe preserves the frozen recipe fields from `PROFILE_RECIPE_R3.json`; only root's admission fields and execution metadata may differ. It refuses nonempty parent-cache paths, preexisting output directories, or less than 40 GB of physical scratch. Root's outer thermal guard holds the campaign-wide `gpu.lock`; the child script does not try to acquire it again.

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

Run it only inside root's worker guard: CPU quota 190%, two numerical-library threads, 4,000,000,000-byte memory maximum, swap disabled, GPU hidden, strict FP32, physical SSD scratch, thermal pause at 80°C and resume only at or below 74°C with one-second polling, and a 7,800-second ceiling for cache preparation plus all twelve fits. The script has no queue, retries, profile mode, parameter sweep, or grader invocation. A failure stops the sequence and preserves any create-only partial artifacts for diagnosis; there is no automatic missing-head replacement.

Expected outputs are two full parent-cache directories under `/home/vlad/development/wunder/research_tmp/online-relationships-20261022/parents/` and twelve run directories under `/home/vlad/development/wunder/research_tmp/online-relationships-20261022/runs/`. Each cache has fit, feature-only evaluation and parent-prediction payloads/sidecars. Each readout output must contain epoch-2 checkpoint, evaluation prediction, prediction sidecar and provenance. The launcher checks the complete inventory before returning success. The independent grader then owns the separate exact evaluation step; reserve at most 600 seconds for that step after the independent cost review. No target-quality scoring occurs in the actor launcher.

## Pinned evidence

- Implementation source last changed at commit `af952697f35792966148a73998f9ebb889aaee28`; postboot profile admission ran from source commit `2ccff4a41ae35a7267398a00657cb761075de17c`.
- [Postboot profile result](POSTBOOT_PROFILE_RESULT_R1.json) SHA-256: `dcbfb56e10ddef133eb9c6c19c5c20068571051416d8d96b8dc7a1c24ba3597e`.
- [Postboot execution receipt](POSTBOOT_PROFILE_EXECUTION_R1.json) SHA-256: `c635a3e372af9757fac4da534c22ba8cdf35708fb56908f77f940227498be854`.
- The result binds 27 source-file hashes and records no evaluation-label reads or target scoring. It is an engineering cost profile, not a scientific result or execution admission.
- Shell validation is limited to `bash -n`; the delivery script itself has not been run.
