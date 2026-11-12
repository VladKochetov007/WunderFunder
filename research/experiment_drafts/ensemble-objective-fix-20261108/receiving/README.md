# B300 fixed4E receiver R1

This external operator code reuses the accepted B300 controller resources and owned subreaper relay. It does not modify the qualified model capsule. ROOT alone performs native staging, approval, old-fit retirement and launch.

Actual qualification: native NCCL oracle passed all eight strict gradient cases (RPC143, maximum absolute gradient error 1.192e-7). Four attempt2 profiles each completed50 updates and both-rank model/optimizer/scheduler/host+CUDA RNG reload, source identity `88ad6a8637390f39906b7ada30792a822f2172298770d9f27476d10028cc0a28` (RPC161). Elapsed160.8–175.9s while sharing GPUs with old fits, peak reserved117.803–117.805GB. First attempts timed out after40–44 updates without OOM/nonfinite and remain preserved. Attempt2 used ROOT's diagnosed300s engineering bound; no third attempt. These are engineering results, not quality estimates. Native Matplotlib availability was confirmed (RPC162).

The receiver admits only the four approved cell/pair/seed identities and fresh fixed4E fits,820 updates/E,3280 total. It consumes the first native qualification clock1794091436:3h fit plus1h collection, without reset. No automatic retry/resume or new seed is permitted. Old four30E attempts remain partial, historical outcomes. The receiver only reads ROOT's retirement receipt; it never signals old processes.

Stage this directory as `$R/receiver-r1`. The launcher's `PYTHONPATH` MUST include the accepted receiver before invoking receiver.py: its top-level `from controller import ...` runs before CLI argument parsing. Actual child commands replace PYTHONPATH with the qualified new source. The imported controller and three accepted native helper hashes are checked before admission. TERM grace1.0s uses the original native lifecycle proof; callback guard/cleanup errors are latched, never raised into the legacy executor's long exception grace.

```bash
R=/data/epanov_s/tabdata/runtime/objective-fix-20261108-r1
S=$R/source-r1
Q=$R/receiver-r1
OLD=/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/receiver-b300-r1
SUPPORT=/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1/operator-r26/src
PY=/data/epanov_s/tabdata/.venv/bin/python
export PYTHONPATH="$OLD:$SUPPORT"
"$PY" "$Q/build_plan.py" \
  --profile-plan "$S/research/experiment_drafts/ensemble-objective-fix-20261108/profile-plan.json" \
  --runtime "$R" --python "$PY" \
  --source-manifest "$R/source-r1-lf-trusted-manifest.json" \
  --source-manifest-sha256 d9f9baf4ca9cf21fc3e7fbe5ec73fc7d20dcb0424ef02d5edb581e2d3ddc805e \
  --authority "$R/authority-r1.json" --profile-root "$R/profiles-r2" \
  --output "$Q/plan.json"
PLAN_SHA=$(sha256sum "$Q/plan.json" | cut -d' ' -f1)
"$PY" "$Q/receiver.py" --plan "$Q/plan.json" --plan-sha256 "$PLAN_SHA" \
  --python "$PY" --accepted-receiver "$OLD" --support-src "$SUPPORT" --detach
```

After source/software/CPU plotter readiness, all four workers wait independently for exact ROOT approval copies `$R/approvals/<cell>.json` and `$R/control/old-four-retired.json`. The latter is `{approved_by:"ROOT",status:"retired_at_safe_epoch",cells:[...]}` with exactly four records: `next_epoch` integer1..29, `global_updates=next_epoch*820`, `fixed30E_complete:false`, immutable `checkpoint:{path,sha256}`, and `stopped_trainers:[{pid,birth,pgid,...}]` each absent under the native birth check. ROOT must create this after safe archives and actual old trainer exit. The receiver cannot stop originals.

Canonical fit, prediction, official score and result bundle run in owned groups with the original fit deadline and collection reserve. Only treatment receives the fit reference cache. Both arms' primary output is fixed.9TL3+.1z before clipping; standalone z official WP/WMSE comes from the same per-epoch monitoring forwards. Each fit must end at4E/3280. External CPU plotting writes actual per-epoch PNG/SVG plus hashes; all four epochs must exist before bundling. Final comparison reports both seeds and repeatedly exposed validation, without selection or significance/public-score claims. RSS ceiling1TiB, new outputs200GB, disk floor200GB, resources checked every10s.

Local verification: six focused CPU lifecycle/receipt tests, including fake canonical executor callback guard+cleanup failure, R2 profile-path binding, both-rank reload refusal, fixed clock and read-only old retirement checks. Native model/reload qualification is separate and already measured above; fresh full fits/evaluation remain pending until ROOT launch.
