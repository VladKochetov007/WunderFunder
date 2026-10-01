# V3 execution fragment amendment

Root authorized recovery from an orchestration interruption: the proxy continuation finished, then an automatic queued-diagnostic expiry wrote the stop barrier before FT. Original proxy job, logs and checkpoint remain unchanged. Missing FT stages have not run because the subsequent runtime permission change prevented launch before any new job record was created. No candidate quality has been inspected.

`spread_operational_design_v3.json` preserves the original statistical design exactly and adds only the final proxy member's `job_name: c20260930_ft_spread_proxy_f0_resume_v1`. Its wrapper retains original data/source/argv/cache commitments and adds explicit original-preregistration and frozen-resume-commitment pointers. The generic scorer already supports this operational field; scorer source stays unchanged.

`spread_binding_amendment_v3.json` binds original and operational preregistrations, original binder/checks, the preserved v2 arithmetic amendment, v3 source/proof, frozen resume commitment and completed continuation checkpoint/provenance. V3 verifies exact projection equality and only accepts that bound operational preregistration. It requires the old successful continuation stage under its immutable failed GPU job plus the exactly prescribed missing FT_head, FT_full and export under a successful new GPU job. No completed training is repeated; source maps, stage argv and ancestry must match. Original-plus-resume elapsed must remain within720 seconds, and resume timeout stays467 seconds. Unexplained extra original stages, altered recipes, missing/failed resume, changed allowance, mismatched sources or updates reject binding.

All v2 batch arithmetic and exact completed/skipped/EMA/normalization/sampler/membership parity checks remain. The known missing-stage fragments were verified from metadata only. The final synthetic proof `screen_binder_checks_v3b.json` passes26 rejection cases and operational-projection/prefix-suffix known answers. Its resource fallback was foreground timeout30 seconds, one CPU affinity, one BLAS thread and hardRLIMIT_AS2GiB; observed peak67.40MiB. This is an address-space cap, not a2GiB cgroup/RSS guarantee. The earlier v3 development proof and all v1/v2 artifacts remain.

After root regains launch access, completes exactly the frozen resume command, and reviews this amendment:

```bash
.venv/bin/python research/campaign_2026-09-30/audit/build_screen_specs_v3.py bind \
  --design research/campaign_2026-09-30/audit/spread_operational_design_v3.json \
  --amendment research/campaign_2026-09-30/audit/spread_binding_amendment_v3.json \
  --spec research/campaign_2026-09-30/screens/spread_v1/spec_v3.json
```

Use the then-authorized CPU resource wrapper. Binding remains blocked by the absent successful resume job. No score has been produced by this work.
