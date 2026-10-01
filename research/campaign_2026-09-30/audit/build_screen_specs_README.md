# Frozen screen design and artifact binder

`build_screen_specs.py` prepares fold0 screen inputs for the unchanged frozen `campaign_score.py`. It reads JSON metadata and hashes files; binding inspects prediction headers and provenance. It never opens target arrays or computes scores.

## Inputs and freeze order

Root first freezes the live implementation and a pipeline recipe, before any arm fits. The recipe contains `status: "frozen-before-fit"`, `source_sha256` keyed by repository-relative source paths, and `arms[arm]` with `log_directory` and ordered `stages[{stage, argv}]`. Each stage argv is an exact list. It must contain no self hash.

The binder's `--closure` may be the same recipe with `stage_expected_args` added, or a separate commitment containing:

```json
{
  "pipeline_recipe": {"path": "research/.../recipe.json", "sha256": "..."},
  "source_sha256": {"scripts/train_rnn.py": "...", "src/wnn/run_identity.py": "..."},
  "stage_expected_args": {"exact_run_name": {"name": "exact_run_name", "seed": 3}},
  "job_commands": {"exact_final_run_name": [".venv/bin/python", "scripts/campaign_screen_pipeline.py", "--recipe", "research/.../recipe.json", "--recipe-sha256", "...", "--arm", "arm_key"]}
}
```

The example arguments and sources are abbreviated: the actual commitment must supply full effective trainer arguments for every fit stage, and all required implementation sources. `job_commands` is optional. If absent, the binder locates the unique recipe arm whose stage argv contains the final run name and derives the outer command from the recipe's path and computed SHA256. The derived command is stored in the separate frozen design, preventing circular hashing. Root must launch precisely that command.

Families and exact run names are fixed in the binder: `spread` covers continuation, head warmup and final FT for both current/proxy arms; `structured` covers multi/single raw epoch3; `event` covers content/position/raw raw epoch3. Export occurs inside the same outer durable GPU job as its fit stages. The job name equals the final run name. Every stage and log path is immutable and exclusively created.

Freeze the three designs using the closed root plan and closure:

```bash
.venv/bin/python research/campaign_2026-09-30/audit/build_screen_specs.py freeze \
  --family spread --closure research/.../closure.json \
  --plan research/campaign_2026-09-30/screen_plan.json \
  --design research/campaign_2026-09-30/audit/spread_screen_design.json
```

Use separate exclusive design filenames for `structured` and `event`. Freeze refuses previously started jobs, stage provenance, or reserved pipeline log directories. The binder's own source hash, recipe, root plan, closure, full arguments, split, prediction order, cache hashes and fixed comparison weights are frozen before fitting. Do not edit the binder afterward.

## Bind after all family jobs finish

```bash
.venv/bin/python research/campaign_2026-09-30/audit/build_screen_specs.py bind \
  --design research/campaign_2026-09-30/audit/spread_screen_design.json \
  --spec research/campaign_2026-09-30/audit/spread_screen_spec.json
```

Run these commands under the campaign's CPU scope limits. Binding requires successful exact outer job argv/source closures, every exact pipeline stage argv/exit, fixed checkpoints, backward checkpoint hashes, clean split membership and propensity. Spread requires the same seed3 parent epoch3, continuation epoch0, both FT epochs0, and the full supervision-only cache commitment; both FT stages must use original proxy auxiliary labels. Historical reference seed3/seed4 controls retain their own source identity and both FT ancestor stages. All prediction sidecars must cover all1,873 VALID row groups in order.

The binder recomputes optimizer attempts from complete dataset size, batch and chunk; validates actual successful/skipped updates, completed batches, cumulative counts and EMA calls; and compares the arms' complete epoch histories, ordered membership, pre-iteration RNG/sampler identity and normalization bytes. Any differing exposure invalidates the matched comparison. These diagnostics do not materialize actual sample order or per-step skipped-update indices. Equal histories require the same code path and sampler behavior as documented by the frozen trainer.

Spread replacement pairs use `.5 unchanged seed4 FT + .5 new seed3 FT`, versus the original pair and matched proxy replacement. Raw SSM/D1 screens use fixed `.8 reference + .2 arm` blends and their declared arm controls. No weight search occurs. The root screen gates remain bound through `root_screen_plan`; the numerical scorer reports results and does not itself enforce promotion policy. Incomplete epochs/timeouts invalidate comparisons. Local screens remain inspected-panel evidence.

The generated spec can be passed to the frozen scorer only when root authorizes actual evaluation. Keep scorer source and freeze record unchanged.
