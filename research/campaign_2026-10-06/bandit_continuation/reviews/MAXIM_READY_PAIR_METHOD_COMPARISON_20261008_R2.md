# Ready pair versus audited teammate method

Read-only source/provenance audit, 2026-10-08. No experiment was launched.

Among currently delivered ZIPs, `submissions/c20261008_h230_all_data_pair_i16.zip` is structurally closest to the reported 0.6674 two-student endpoint: equal GRU256x2 pair, unnormalized teacher sum mass0.8125, main distillation alpha0.7, raw family targets, family loss0.1, two additional epochs, AdamW/OneCycle/LR1e-4/B16/C2000, and structural/next auxiliaries0.3 each. It is an analogue, not a completed faithful replication.

The audited final student uses a linear input projection with GELU, a two-layer GRU, and a `Linear(512,256) -> GELU -> Linear` main readout. A separate eight-output family linear head consumes the same post-GELU hidden features. Our eight appended output rows consume those same features; initialization/RNG still differs. Neither inspected final student is a Transformer; a Transformer teacher is not established by the successful artifact.

Remaining differences: our shipped pair shares one AdamW/C2000 all-data parent, rather than separate seed-specific stronger Muon/C1000 parents; current teacher weights retain cleanfold ancestry; two strong sources0.15 each replace one0.30 source; two own raw GRU384 baselines0.03125 each replace an organizer source0.0625 with unresolved permitted-source provenance. The inspected reference teacher checkpoint ancestry is incomplete; its all-row export lane does not by itself prove teacher fitting coverage.

**Additional directly observed feature-family discrepancy:** our `raw_wmse` cache sources `c20260929_since_s3_f0` and `c20260929_since_s4_f0` both record `raw_only=false`. They are full-feature WMSE teachers, unlike the inspected `final_rawW.sh` raw-only predecessor. The final reference raw checkpoints are not fully ancestry-bound, so this is a mismatch to the available method evidence, not proof of the exact unseen final teacher recipe. It may matter for teacher diversity; its quality effect is untested. Do not call these two sources raw-feature teachers.

Fidelity fixes already delivered: include supplied valid groups in student fitting; retain source sum instead of normalizing0.8125; family targets remain raw instead of the earlier extra clipping; preserve warm parent ordinary output rows while adding family rows; fixed equal student endpoint. These are implementation/recipe fixes, not proven public improvements. Normalization and history-swap variants have not beaten0.6660. The independent S4 contextual readout's positive exposed-fold parent contrast is a separate arm, absent from these shipped ZIPs.

Deliberate exclusions: teammate code and weights; unverified organizer-weight provenance. Shared-parent ancestry and AdamW parent were earlier shortcuts/drift, not evidence-based rejection of Muon or of stronger parents. Muon/C1000 stronger-parent and 512/768 scale successors are admitted cloud jobs, not completed improvements.

References: [endpoint audit](../wide_baseline_heterogeneous_H230/H230_MAXIM_RECIPE_AUDIT_V1_20261008.md), [teacher ancestry limits](../wide_baseline_heterogeneous_H230/H230_ALLDATA_TEACHER_FAMILY_REPLICATION_AUDIT_V1_20261008.md), [shipped pair contract](../wide_baseline_heterogeneous_H230/H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V4_20261008.json).

Directly read provenance bindings:

- `artifacts/runs/bd26_h230_all_data_parent_b16_ep3_retry3_v1/provenance.json` SHA256 `3becd92c4775098cf16b9005cf6c45361bbe776eb5601a6ebe89faa8605e22dd`: hidden=256, layers=2, raw_only=False, optimizer=adamw, chunk=2000, readout_activation=gelu. Missing legacy optimizer/activation fields are not inferred as an actual setting.
- `artifacts/runs/bd26_h230_all_data_student_gru256_seed3_v1/provenance.json` SHA256 `57aa9d1fc3a28cefe365dc099aaf31f7c51c25160cf28f0475dc1e2391508cae`: hidden=256, layers=2, raw_only=False, optimizer=adamw, chunk=2000, readout_activation=gelu. Missing legacy optimizer/activation fields are not inferred as an actual setting.
- `artifacts/runs/bd26_h230_all_data_student_gru256_seed4_retry1_v1/provenance.json` SHA256 `d921614f66dbeca982564958e6b3d726058e99b1aa4026b2252269e8b93b983f`: hidden=256, layers=2, raw_only=False, optimizer=adamw, chunk=2000, readout_activation=gelu. Missing legacy optimizer/activation fields are not inferred as an actual setting.
- `artifacts/runs/c20260929_since_s3_f0/provenance.json` SHA256 `52438c2c9d1a551251aee7a18cc1cca5fce2b9aed55bc558d1adafcae29c166a`: hidden=128, layers=2, raw_only=False, optimizer=None, chunk=2000, readout_activation=None. Missing legacy optimizer/activation fields are not inferred as an actual setting.
- `artifacts/runs/c20260929_since_s4_f0/provenance.json` SHA256 `5ec060b96a202bdd13dc7ec6b03bf1b878ea70811b99d82844a99ce164d8a99c`: hidden=128, layers=2, raw_only=False, optimizer=None, chunk=2000, readout_activation=None. Missing legacy optimizer/activation fields are not inferred as an actual setting.
