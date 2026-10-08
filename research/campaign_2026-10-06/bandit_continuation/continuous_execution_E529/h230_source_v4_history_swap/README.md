# H230 all-data heterogeneous distillation source V3

This is a narrowly extended copy of the independently maintained H230 V2
source snapshot. The parent snapshot is preserved unchanged at
`../h230_source_v2`; its manifest hash and every source member are recorded in
`H230_ALLDATA_DISTILL_SOURCE_MANIFEST_V1_20261008.json`.

V3 adds an opt-in cache builder mode that emits targets for every row group in
the supplied `train_targets.npz` and `valid_targets.npz` indexes. It preserves
the fold-0 teacher split and checkpoint ancestry separately from the complete
prediction coverage. The trainer requires exact all-row fitting membership,
rejects subset/replacement sampling in this mode, validates all cache rows and
records the teacher scope and no-held-out-claim status in run provenance.

The declared recipe uses eight independently trained, fold-0-clean teacher
checkpoints and their already exported full-index predictions. The teachers
remain fold-trained; only their prediction arrays cover all supplied groups.
The all-data parent/student weights are our own. This is a distinct final-fit
variant, not an exact claim about a reference endpoint or an all-data teacher
ensemble.

The snapshot includes no data, teacher/student weights, or generated cache
arrays. Tests use tiny synthetic indexes and arrays. No all-data cache build or
student fit is authorized by this source snapshot. Root review and fresh
resource admission are required after the H230 all-data parent checkpoint is
terminal and verified.
