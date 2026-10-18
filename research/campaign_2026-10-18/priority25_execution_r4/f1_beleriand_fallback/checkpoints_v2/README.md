# F1 checkpoint verifier derivative V2

This isolated helper set preserves the original V1 files under `checkpoints/`. The V2 change removes one false persisted-option expectation: frozen `a266b036` `resume_contract()` deliberately excludes `save_training_state` from `contract.options`, alongside operational `name`, `resume_state`, `stop_after_epoch`, and path-only manifest arguments. The captured wide stage-1 contract confirms that omission at state SHA `8cb3dde79550355303cbede0d0953864046b7594c16af648423d4a3c9af92255`; all expected scientific recipe options checked against the captured options matched.

The operational behavior remains pinned by the exact attempt command: `validate_attempt()` requires the saved attempt command to equal the frozen plan vector. The real-producer fixture records `--save-training-state` and `--stop-after-epoch 1`; regression tests reject removal of the save flag, a changed epoch cap, a changed fold, and a teacher/distillation option. The verifier retains its other source, manifest, runtime, membership, ancestry, optimizer, tensor, history, and FP32 checks.

Use this same V2 support directory for both arms' stage-1 verifications and subsequent terminal verifications. Terminal lineage requires identical verifier and dependency hashes between both stage-1 receipts and the terminal receipt, so V1/V2 receipts must not be mixed. The collector and common helper are byte-identical to V1; the existing stage-1 collection receipt remains usable.

This is a preparation derivative only. It has not been copied to B or used to verify a real checkpoint. It makes no new training or quality claim. Local checks:

```text
uv run --no-sync python -m unittest discover -s research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints_v2 -p 'test_*.py' -v
uv run --no-sync ruff check research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints_v2
uv run --no-sync ruff format --check research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/checkpoints_v2
```
