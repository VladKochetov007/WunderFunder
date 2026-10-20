# NLL predecessor-gate repair — V2 local packet

V2 adds a missing lineage check to the fixed NLL child packet. Before accepting the dictionary child's terminal acceptance or rendering a future body, it now requires `launch_acceptance_sha256` to equal the original accepted dictionary launch receipt SHA-256, `33fa4fd4cb37c16374e615c2eb6d698f5f3065b252803377260fdae6dd913a95`.

V1 is preserved unchanged. Its remote predecessor validator and local receipt validator recorded the original launch-acceptance pin in their output but did not compare that pin with `launch_acceptance_sha256` inside the actual dictionary terminal acceptance. The V2 local validator rejects both a missing field and any mismatch; the V2 admission template applies the same condition before the existing terminal checks.

`NLL_LAUNCH_TEMPLATE_V2.py` is derived from the pinned V1 template by one contextual condition in the dictionary-child predecessor validator. The recipe, trainer argv, source pins, output identity, resource gates, and all other receipt checks are unchanged. The separate V2 binder's eventual `--render` path validates the full V1 terminal receipt checks plus the new launch-acceptance field and writes only V2-named output files.

This packet remains unbound. It has not read actual dictionary terminal acceptance or exact-stop receipts, rendered a submission body, or performed B/SSH/admission/training actions. Independent source review is required before any later actual render or launch.

## Focused local checks

```bash
./.venv/bin/python -B research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_nll_fit_v1/bind_nll_fit_v2.py --check
./.venv/bin/python -B -m unittest research.campaign_2026-10-19.full_gru_tweaks_r1.preparation.child_nll_fit_v1.test_bind_nll_fit_v2
```

The captured result is in [NLL_V2_LOCAL_CHECKS.json](NLL_V2_LOCAL_CHECKS.json). The four focused fixtures cover the exact accepted SHA, missing and mismatched SHA rejection through the local validator, and presence of the matching remote validator. No unchanged source, model, or profile suite was rerun.

## V2 files

- [bind_nll_fit_v2.py](bind_nll_fit_v2.py) verifies the frozen V1 inputs, derives the V2 template, and adds the exact launch-acceptance check to its local receipt validator.
- [NLL_LAUNCH_TEMPLATE_V2.py](NLL_LAUNCH_TEMPLATE_V2.py) is the fail-closed unbound remote template; it is not an admitted submission body.
- [test_bind_nll_fit_v2.py](test_bind_nll_fit_v2.py) contains only the new focused identity and missing/mismatch checks.
- [NLL_V2_LOCAL_CHECKS.json](NLL_V2_LOCAL_CHECKS.json) binds the local check results and exact V1/V2 template pins.
- [RESULT_CAPTURE_TEMPLATE.py](../child_dictionary_terminal_verifier_v1/RESULT_CAPTURE_TEMPLATE.py) belongs to the separate read-only dictionary verifier capture preparation and does not alter this NLL packet.
