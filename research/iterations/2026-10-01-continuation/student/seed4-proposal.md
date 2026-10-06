# Prospective second shallow student — unallocated

**Worth considering only after the current final artifact/native quality is established.** The seed3 shallow member retained strong-four quality while reducing measured callback cost. The next question is whether spending some saved cost on a second independently optimized wide/shallow member improves the ensemble. No fit, inference, or score is authorized by this note.

Exactly four CV fits: seed4, folds0/1, each the existing h256×1 four-epoch EMA parent (`model_ep3.pt`) followed by the same two-epoch H2 (`model_ep1.pt`). Proposed names `cont_sh256_s4_f{0,1}` and `cont_sh256_s4h2_f{0,1}`. Keep same-fold teachers `h256_fF,h256s4_fF`, alpha0.5, since/propensity/aux/data/precision/optimizer recipes unchanged; H2 has no distillation/EMA/since and freezes `inp,gru`. Change only name, seed, and H2 own-parent path relative to the completed seed3 commands. No final fit, third seed, extra epoch, weight search, or automatic follow-up.

All profiles have fixed equal-quarter weights:

| Profile | Members |
|---|---|
| Proposed double shallow | shallow256-s3, shallow256-s4, ctl128-s3, ctlc500 |
| Current student4 | shallow256-s3, ctl128-s3, ctl128-s4, ctlc500 |
| Strong4 reference | ctld256, ctl128-s3, ctl128-s4, ctlc500 |
| Seed variability diagnostic | shallow256-s4, ctl128-s3, ctl128-s4, ctlc500 |

Predeclare double-shallow minus current-student4 and strong4; additionally single-seed4 minus single-seed3 to expose optimizer variability. Use the same two outer folds and whole-chain paired uncertainty. These exposed folds are not fresh confirmation. Changing seed changes initialization, normalization sample and shuffled training order; it is optimizer replication on the same data, not a new data holdout or isolated seed-only tensor comparison. No new gates proposed here.

**Duplication check:** E-226 already tested a seed4 one-layer EMA+since parent on fold0 and stopped below its paired gate. Its two-layer reference lacked since weighting; the ledger does not establish hidden width, distillation, or current H2 recipe. The referenced q48 shell script is no longer present; current base/creative run directories have no shallow/`sh256` seed4 run. Thus no exact current four-fit study was located, while historical shallow-seed4 counterevidence must remain visible. See [E-226](../../../experiments.jsonl:226) and [prior assessment](../student-capacity-assessment.md).

**Cost:** serving dense MACs/row: double1,670,912, current1,348,736, strong1,741,952. Double is +23.9% versus current and only4.1% below strong. Current [100k native probe](native-probe-01/result.json) extrapolates45.20min callbacks/47.10min trace; proportional scaling gives56.0/58.4min. Strong4's [100k smoke](../full4-int8/smoke-100k.json) gives57.89min; MAC scaling gives55.5min. These are rough forecasts on different cores/fixtures, not qualifications; current probe also had8.9% sibling activity. A double-shallow package may have insufficient60-minute margin. Existing four fits took48.75min plus33s OOF on one GPU; budget about50–60min if later allocated. Stop pursuing this composition if its actual cost loses the usable runtime margin or the fixed contrasts show no useful gain; do not rescue it by searching weights/seeds.
