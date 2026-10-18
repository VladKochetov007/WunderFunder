# Public incumbent identity audit — 2026-10-18

This audit reads the existing quality-gap diagnosis, leaderboard snapshots, model catalog, ZIP inventory, and team release receipt. It does not query the portal, open or hash ZIP payloads, read model arrays, or run a job.

## Established

- The official 2026-10-17 snapshot records Team kotiki at **0.66832**, rank 22, with 97 submissions. The separate 2026-10-18 13:10Z baseline records **0.66832**, rank 23, with 100 submissions. These are dated leaderboard observations, not a submission identity (`[official snapshot](../../unified/reviews/audit-2026-10-17/official_leaderboard_20261017.json)`, `[2026-10-18 baseline](PUBLIC_BASELINE_20261018.json)`).
- The baseline explicitly leaves `exact_incumbent_portal_id` and `exact_incumbent_uploaded_sha256` null. The diagnosis also says the uploaded portal bytes were not hash-confirmed (`[gap diagnosis](../../unified/reviews/QUALITY_GAP_DIAGNOSIS_20261018_R1.md)`). Thus there is no exact public `.66832 → portal ID → filename → uploaded ZIP SHA` binding in the reviewed receipts.
- The model catalog says the server journal records both `tl3.zip` and `tl4.zip` at **rounded .6683**, and says the exact current-best portal-ID-to-ZIP-hash binding is unavailable (`[model catalog](../../unified/models/CURRENT_MODEL_CATALOG.md)`). The rounded score does not distinguish them or prove either is the exact `.66832` incumbent.
- The archived inventory binds local package bytes to server-copy bytes for both candidates (`matches_server: true`), but those are repository/server-copy identities, not portal-upload hashes (`[submission inventory](../../unified/reviews/audit-2026-10-17/submission_inventory.json)`).

## Strong local recipe candidates

The inventory records these exact local files and fixed member recipes:

| Candidate | Local ZIP SHA-256 | Members |
|---|---|---|
| `/Users/nb/Documents/wunder/submissions/tl3.zip` (12,538,394 B) | `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569` | 0.34 `distilled_s5@ep1` + 0.33 `fttall_tlead_s5@ep3` + 0.33 `fttall_tlead_s6@ep3` |
| `/Users/nb/Documents/wunder/submissions/tl4.zip` (16,173,182 B) | `659ad2e6c984579708ef95ec6bfc08d622e4838b518aa26628dc58153d26669b` | 0.25 each: `distilled_s5@ep1`, `distilled_s6@ep1`, `fttall_tlead_s5@ep3`, `fttall_tlead_s6@ep3` |

These are the strongest local package recipes available as counterparts to the catalog's rounded `.6683` reports. The catalog identifies the distilled family as warm-parent, two-epoch AdamW distillation and the tied-lead family as a separate scratch-parent, four-epoch Muon/AdamW/EMA recipe. Their final fits include train and valid, so supplied-valid scores are fitted diagnostics, not clean held-out estimates (`CURRENT_MODEL_CATALOG.md`, “Training lineage”).

The exact `.66832` public incumbent could be one of these packages, another submission, or a package whose portal payload differs from the repository archive. The evidence does not support choosing `tl3` or `tl4` as the exact incumbent. The 2026-10-13 team receipt covers earlier submissions and reports `pair_ftt2` at `.6682`; it does not close this identity gap (`[team public release receipt](../../campaign_2026-10-13/team_models_r1/PUBLIC_LB_UPDATE_20261013_R2.md)`).
