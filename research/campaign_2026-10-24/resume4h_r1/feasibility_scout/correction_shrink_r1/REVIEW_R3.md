# Independent R3 code and recipe review

## Decision

**Pass for root admission.** No target-grade labels or campaign prediction
payloads were opened during this review. Root retains ownership of admission,
the bounded target run, and scientific interpretation.

## Frozen identity reviewed

- [R3 recipe](CORRECTION_SHRINK_RECIPE_R3.json), SHA-256
  `b2bf2a065687e3d970db852fa55cacd0d202da43eb587527b99683a3881e6802`.
- [R3 source manifest](CORRECTION_SHRINK_SOURCE_MANIFEST_R3.json), SHA-256
  `62d96dbe1881ef2e42939fecdb8179c3dfdd6caa79122b961e31b4acf28711bb`.
- [R3 synthetic receipt](CORRECTION_SHRINK_SYNTHETIC_R3.json), SHA-256
  `55709fea6a8f16fb861ea76dd695a515079571753bf6d325c3370a0c77f47d85`.

All 62 source/configuration file pins and the synthetic evidence digest match
their current payloads. The earlier R2 recipe, source manifest, receipt, and
three implementation sources remain preserved at their original hashes in the
[R2 source snapshot](r2_source_snapshot/README.md). R2 was superseded after its
candidate-map check was found to validate only the entry count.

## Findings

R3 fixes the reviewed blockers. Before any target read, the runner checks the
exact six candidate-map keys, parent prediction metadata, the full A-lag16
duplicate identity, source-qualified group order, target metadata, admission
identity, output path, and source pins. It binds the target archive digest
before label reads and checks it again after scoring. The coefficient is
exactly decimal `0.1` and explicitly cast to float32; the score path applies
only the unchanged official metric clipping. Recipe contrasts are five
shrink-versus-parent primaries and thirteen secondary pairs, including all
five shrink-versus-raw comparisons. Raw endpoint scores must reproduce the
original grade for all bands, both targets and selected-row counts before any
new result can be written.

The analysis is limited to five unique exported predictions on exposed F0,
fold 0, seed 1001, with 468 evaluation groups and 141 source-qualified
components. It uses no new fits, inference, coefficient search, or public score.
Component bootstrap uncertainty is conditional on these fixed predictions.
The original 12-fit panel remains incomplete.

## Checks run by this reviewer

- Independently ran [the synthetic checker](check_correction_shrink_synthetic.py)
  with generated arrays only under 150 MB, one CPU core, and a 60-second wall
  limit. All checks passed, including the malformed-key rejection. The checker
  blocks campaign prediction loads and reports no campaign target or
  prediction payload reads.
- Ruff format check: all three Python files already formatted.
- Ruff check: all checks passed.
- Independently recomputed every R3 source/receipt hash and every R2 snapshot
  source hash; all matched the frozen manifests.

## Resource note

The R3 worker remains bounded to 400,000,000 bytes, one CPU core, 300 seconds,
zero swap, and physical scratch. The preceding F0 grade reached 399,998,976
bytes under the same memory cap. That previous peak is not an R3 measurement;
root should watch the new worker's cgroup memory peak and OOM/swap counters.
