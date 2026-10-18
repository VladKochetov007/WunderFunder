$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..')).Path
$queuePath = Join-Path $repo 'research\campaign_2026-10-18\b300_queue_preparation_r1\queue.json'
$annPath = Join-Path $here 'annotations.json'
$queue = Get-Content -Raw $queuePath | ConvertFrom-Json
$annotations = Get-Content -Raw $annPath | ConvertFrom-Json

$prs = @(
  foreach ($item in $queue.pr_reviews) {
    $extra = $annotations.pr_annotations.PSObject.Properties[$item.id].Value
    if ($null -eq $extra) { throw "Missing annotation for $($item.id)" }
    [ordered]@{
      id = $item.id
      number = $item.number
      title = $item.title
      url = $item.url
      branch = $item.branch
      head_sha = $item.head_sha
      historical_review_status = $item.status
      mechanism = $item.mechanism
      draft_implementation_summary = $item.implementation
      arms = @($item.arms)
      planned_fit_count = $item.planned_fit_count
      replications_per_arm = $item.replications_per_arm
      rank25_mapping = $item.rank25_mapping
      historical_dependencies = $item.dependencies
      fidelity_and_native_binding = $extra
      source_review_record = $item
    }
  }
)

$ranked = @(
  foreach ($item in $queue.ranked25_mapping) {
    $extra = $annotations.ranked25_annotations.PSObject.Properties[$item.id].Value
    if ($null -eq $extra) { throw "Missing ranked annotation for $($item.id)" }
    [ordered]@{
      rank = $item.rank
      hypothesis_id = $item.id
      idea = $item.idea
      related_prs = @($item.prs)
      historical_match_label = $item.match
      historical_dependency = $item.dependency
      current_source_binding = $extra
      source_review_record = $item
    }
  }
)

$result = [ordered]@{
  schema = 'b300-backlog-review.v2'
  created_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
  scope = 'Local read-only source/recipe inventory for the original 16 experiment PRs plus ranked-25 hypotheses; no B300 connection, profiles, optimizer steps, fits, remote edits, publication, or spending performed by this review.'
  execution_context = [ordered]@{
    operator_report = 'Root reports B300 environment/data delivery is working and 147 source files plus canonical 61-file trainer identity are verified.'
    independently_verified_by_this_review = $false
    fit_or_profile_status = 'not inspected by this review; no result is claimed'
    no_false_ready_rule = 'A native CLI surface means implementation reuse only; every proposed family remains blocked on its listed recipe, control, lineage, actual B300 profile, cell-owner, and resource gates.'
    attribution = 'The original PR account is not evidence of proposal authorship by named collaborators; preserve each PR source SHA and do not invent authorship.'
  }
  source_bindings = [ordered]@{
    historical_review_packet = 'research/campaign_2026-10-18/b300_queue_preparation_r1/queue.json'
    canonical_trainer_revision = 'a266b036b02e8e012c6eb31072291dad618a169d'
    canonical_61_file_manifest_sha256 = 'a072293668dc8d0dd1201ea4641a1f5e9b99da718744bf33e89639abff3b8b7c'
    canonical_source_identity_sha256 = 'fce5257e02b8d4fa701f987af1b662bf4190050f9167b0e66a9ed935da375189'
    trainer_cli_sources = @('src/wnn/training/cli_model.py','src/wnn/training/cli_features.py','src/wnn/training/cli_views.py','src/wnn/training/cli_targets.py','src/wnn/research_features.py')
    ranked1_recipe = [ordered]@{
      path = 'research/campaign_2026-10-18/priority25_execution_r4/RANK01_RECIPE.json'
      source_revision = 'a266b036b02e8e012c6eb31072291dad618a169d'
      raw_current_checkout_bytes_sha256 = 'ef3c5572dccdc4dab5d2a6980de212eb48b3adc229c85c15db2ad0e5bda6d74f'
      raw_current_checkout_size_bytes = 2802
      historical_git_blob_lf_bytes_sha256 = '9292b545fd088df40c4cc54066a745025ee4d30c2bf1574bdb4df341eb16937c'
      historical_git_blob_lf_size_bytes = 2733
      qualification = 'The two SHA-256 values cover different byte representations; 9292... is the LF Git-object/source revision binding, while ef3c... is the current CRLF checkout file. Do not compare them as if they hash identical bytes.'
    }
    grader_revision = '6f9fd05e491a9700cce1adf50e58b56cdc5e65bb'
    fit_policy_reference = 'research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md'
  }
  current_priority_after_rank1 = @(
    [ordered]@{order=1; item='PR-2 / H-RR17-01'; why='Direct native GRU shapes; freeze measured-compute recipe before a controlled CP30 comparison.'},
    [ordered]@{order=2; item='PR-4'; why='Group-age versus field-age feature modes are source-native; confirm draft semantics, clean lineage, FP32 recipe and confound control.'},
    [ordered]@{order=3; item='PR-14'; why='Native innovation path has same-width zero-innovation control; historical cache-schema issue needs repair or explicit successor registration.'},
    [ordered]@{order=4; item='H-RR17-06 causal moments'; why='Small source-native moment-vs-zero-block family; this is a useful partial test, not the full raw/normalized ranked hypothesis.'},
    [ordered]@{order=5; item='H-RR17-19 ARTBP'; why='Flag exists but admit only after BPTT toy parity and actual-shape resource profile; high-cost lower-ranked follow-up.'}
  )
  minimum_family_recipes = @(
    [ordered]@{
      id='R01-F2-seed1001-labels-only-pair'; basis='Existing preregistered R4 recipe and issue #16; one paired cell only.'; arms=@('wide: --arch gru --cell gru --enc linear --hidden 512 --layers 1','deep: --arch gru --cell gru --enc linear --hidden 384 --layers 2'); common_recipe_path='research/campaign_2026-10-18/priority25_execution_r4/RANK01_RECIPE.json'; target='Original labels, FP32 wMSE/chunk denominator, six epochs, fixed epoch5, batch8/chunk2000, seeded order, matched optimizer/schedule; full settings are in referenced recipe/commands.'; admission='Recipe is source-bound, not a B300 launch claim: bind current B300 paths/owner and inspect actual-shape FP32 profiles/resources first.'; fit_count=2; status='source_recipe_pinned_profile_and_host_binding_required'
    },
    [ordered]@{
      id='PR2-native-GRU-shape-screen'; basis='Five draft shapes; new explicit compute-match recipe required.'; arms=@('H256x2','H512x1','H640x1','H512x2','H448x3'); common_delta_flags='--arch gru --cell gru --enc linear --hidden <H> --layers <L>'; target='Cold start; matched measured compute/update view and fixed-update view must be explicitly reconciled; do not copy PR draft 12-epoch WSD values as a frozen recipe.'; admission='Measure every actual shape and freeze budget, scheduler, endpoint and threshold before allocating CP30; pair all arms in fold/seed identity.'; fit_count=150; status='native_shapes_recipe_not_frozen'
    },
    [ordered]@{
      id='PR4-native-age-feature-pair'; basis='Source-native representation pair; faithful status contingent on draft/source semantic diff.'; arms=@('group clocks: --research-features group_age','per-field clocks: --research-features field_age'); target='Same base model, optimizer, input normalization, fold/seed membership and fixed endpoint; decide clean cold start versus inherited fold-clean parent before freezing.'; admission='Run causal prefix/reset and streaming-chunk parity checks; no teacher/cache/parent reuse across folds; record PR4 historic profile as engineering only.'; fit_count=60; status='feature_modes_present_recipe_lineage_profile_open'
    },
    [ordered]@{
      id='PR14-native-causal-innovation-ablation'; basis='Native-source successor candidate; historical cache implementation is a distinct lineage.'; arms=@('innovation channels active: --research-features causal_innovations --research-feature-control none','same-width moment backbone with innovation suffix zeroed: --research-features causal_innovations --research-feature-control newest'); target='80 appended channels (64 moments + 16 innovation); fitting-only moment prior; identical model/seed/fold and no cross-fold cache reuse.'; admission='Fixture-test cache producer/consumer or state this is a new native-source successor; verify current-row prediction alignment, prefix/reset and prior ancestry, then profile.'; fit_count=60; status='native_mode_present_cache_and_recipe_gates_open'
    },
    [ordered]@{
      id='H-RR17-06-minimal-causal-moments-screen'; basis='Minimum discriminating subfamily only; does not satisfy the complete ranked three-path hypothesis.'; arms=@('moments active: --research-features causal_moments --research-feature-control none','same-width zero moment block: --research-features causal_moments --research-feature-control newest'); target='Fit-only prior and fixed model budget; consider base raw-input arm as third arm only after matching feature/parameter capacity.'; admission='Prefix/reset parity, fold-local prior, both-shape profile and freeze an explicit threshold; --online-norm is a separate behavior and cannot be silently substituted.'; fit_count=60; status='native_partial_test_available_full_ranked_hypothesis_not_implemented'
    },
    [ordered]@{
      id='H-RR17-19-ARTBP-parity-first'; basis='Low-priority mechanism with native option but no parity evidence in this inventory.'; arms=@('deterministic BPTT control','ARTBP: --credit-mode artbp --credit-keep-probability <p> --credit-seed <seed>'); common_delta_flags='--chunk 2000 --update-rows 20000 --detach-rows 20000'; target='Strict FP32, SGD, no gradient clipping; preserve identical full update-window objective and fixed endpoint.'; admission='First pass exact toy BPTT gradient parity plus variance/bias tests; then measured full-shape time/VRAM. Frozen validation constraints additionally require chunk < update_rows, detach_rows 0 or update_rows, update_rows divides 20,000-row sequence, and no origin/noise/jitter options.'; fit_count=60; status='native_flag_parity_and_profile_blocked'
    }
  )
  original_prs = $prs
  ranked25 = $ranked
  issue16_rank1_anchor = $queue.issue16
  source_attribution_caution = $queue.source_gap
  historic_queue_infrastructure_review = $queue.infrastructure_review
  historical_review_adjudications = @($queue.review_adjudications)
}

$json = $result | ConvertTo-Json -Depth 100
$out = Join-Path $here 'backlog.json'
[IO.File]::WriteAllText($out, $json, [Text.UTF8Encoding]::new($false))
$check = Get-Content -Raw $out | ConvertFrom-Json
if ($check.original_prs.Count -ne 16) { throw "Expected 16 original PRs, found $($check.original_prs.Count)" }
if ($check.ranked25.Count -ne 25) { throw "Expected 25 ranked hypotheses, found $($check.ranked25.Count)" }
Write-Output "Wrote $out ($($check.original_prs.Count) PRs, $($check.ranked25.Count) hypotheses)."
