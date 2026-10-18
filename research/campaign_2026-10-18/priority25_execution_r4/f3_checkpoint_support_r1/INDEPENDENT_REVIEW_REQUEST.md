# Independent review request — F3 checkpoint support R1

Please review the exact `PREPARATION_MANIFEST.json` SHA supplied with this handoff. The package is static preparation only and remains UNADMITTED.

Review whether the F3 collector/verifier correctly adapts the independently reviewed F1 checkpoint V2 support to the pinned F3 seed-1001 stage1 and terminal plan. In particular, check the 505 evaluation-group / 145-component membership identity; exact F3 plan/fold pins; native attempt contract path/hash and source-map canonical digest; 138-file checkpoint source audit; FP32/Adam/history/counter semantics; stage1 and terminal ancestry; create-only output behavior; and whether the positive producer-shaped and negative fixtures cover the relevant schemas.

The producer contract fixture is explicitly derived from captured F1 metadata, not an actual F3 receipt. The exact B fold3 manifest is not locally available, so its positive ordered-membership replay has not been run. The source-map attempt fixture is synthetic but follows the native guard record schema and canonical source-manifest digest calculation. Prediction export/result collection and grade acceptance are intentionally outside this packet and need a separate reviewed F3 helper before export.

Do not run training, remote commands, large-array reads, or prediction/grade work. Please report any actionable blocker against the manifest-bound files and whether the packet is ready for root-side fresh checks and admission.
