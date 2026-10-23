# B16 resource-adapted F1 fixed endpoint screen

**Frozen source plan; independent static review passed:** final proposal is pinned. No grade or fit admission. This separately named packet preserves the unused B8 screen unchanged. B16 keeps six epochs but uses44,760 updates and different optimizer/schedule/sample-drop behavior; report separate fold results, not an identical-arm pooled effect.

# Conditional F1 H512 endpoint screen

**Status: code prepared and statically reviewed; no grade run.** This packet defines one fixed five-endpoint comparison for the conditional seed-1004 H512 F1 fit. It does not admit a fit or promote a model. Grading requires the separate root-authored `F1_FIT_TERMINAL_ACCEPTANCE_R2.json`; the Kaggle worker receipt alone is insufficient.

The [screen plan](F1_FIXED_SCREEN_R2.json) pins the canonical fold-1 manifest, its source-only derived execution view, input registry, official target cache, oldZ4 reference, and official scorer code. The grader checks that only `sources` differs between the canonical and derived manifests, then keeps both hashes in the output. The model receipt and sidecars bind to the derived view. The oldZ4 endpoint remains bound to the unchanged canonical manifest.

The panel contains oldZ4, H512 CP5, H512 SWA345, `0.75 oldZ4 + 0.25 CP5`, and `0.75 oldZ4 + 0.25 SWA345`. All five comparisons and the 2,000-resample component bootstrap seed are fixed in advance. Mixtures use the frozen coefficients; no endpoint or coefficient search is performed. Candidate exports contain the 457 evaluation groups only. The grader aligns those groups to the oldZ4 full-validation cache and scores each sequence through the existing `campaign_score.py` official metric implementation. It derives the selected official row count from the target mask at grading time.

The grader follows the producer's standalone `wnn.kaggle-h512-swa345-weight-derivation.v1` receipt: `equal_parameter_mean_epochs_3_4_5`, exact CP3/CP4/CP5 path and SHA records, canonical and derived manifest identities, initialization and normalization identities, and output path/hash records for `model_ep5.pt`, `provenance.json`, and `model.pt`. It checks those bytes against the root acceptance and requires the derived provenance to preserve the fit arguments and normalization identity.

oldZ4 is a previously fitted, legacy FP16-inference reference; the new endpoints use FP32 inference. This is not a pure precision comparison. Fold 1 has prior evaluation exposure, and these results do not establish a fresh holdout, a public-score forecast, or automatic submission promotion. See the [conditional fit proposal](../kaggle_h512_b16_conditional_f1_r2/README.md) and the prior [F1 source/cache audit](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/FOLD_ASSET_CATALOG_R1.md) for ancestry context.

After root has accepted the fit and both fixed exports, run from the project root:

```sh
python research/campaign_2026-10-21/kaggle_h512_b16_f1_endpoint_grade_r2/grade_h512_f1_screen.py
```

The create-only output is `F1_FIXED_SCREEN_GRADE_R2.json`. The grader verifies accepted checkpoint, derivation, prediction, sidecar, manifest, target-cache and scorer hashes before and after scoring. Current work is code and metadata only; no predictions were read and no score is available.

[Root terminal acceptance CLI](accept_h512_f1_terminal.py) checks six epoch/update records and strict-FP32 evidence, validates checkpoint/provenance/export identities through the shared packet checker, scans both complete exports for finite values, and recomputes the exact canonical CP3/4/5 weight mean. It writes acceptance create-only and makes no quality or promotion claim. Run under campaign resource limits after root has collected and bound actual successful worker outputs; no execution has occurred.
