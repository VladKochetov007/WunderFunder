# Independent audit: additive research tree

Status: pre-implementation audit plan and source baseline, 2026-10-06. I have
not yet inspected generated output. The independent review below targets data
loss, invented evidence, stale status, and unintended execution. The source
ledgers, original tree, bandit file, and historical bandit script are read-only
inputs.

## Baseline observed

- `research/hypotheses.jsonl`: 179 lines, all 179 parse as JSON, all 179 IDs
  unique.
- `research/experiments.jsonl`: 506 lines, all 506 parse as JSON, 491 unique
  IDs. Repeated-ID groups are E-463 (2), E-464 (2), E-489 (3), E-492 (6),
  E-496 (3), E-499 (3), and E-500 (3). These are separate source records and
  attempts; deduplicating by ID destroys evidence.
- `research/bandit/arms.json`: 18 legacy arms. Its priors predate the current
  review, and its values must not enter an eligible ranking until a current
  human review explicitly revalidates them.
- `research/experiments.jsonl` ends with E-496 `calibration_interrupted_no_report`;
  the row says zero completed calibrations and zero scientific fits. E-492 has
  both a failed v1 receipt and a later successful v2 raw-count receipt. E-499
  has a completed, explicitly non-scoring clipping diagnostic. These examples
  require attempt identity and receipt state to remain visible alongside the
  ledger state.
- `research/` contains 57,624 files on disk at inspection time. A tracked-only
  walk omits ignored, untracked, snapshot, and provider material. Filesystem
  inventory and Git inventory answer different questions.
- The current campaign evidence says the confirmed public best is 0.6660 and
  the 0.7000 target is unmet. Supplied-validation 0.700771 figures are not
  public-score forecasts. An index must preserve that claim boundary.

## Adversarial invariants

1. **Immutable evidence and provenance.** Each ledger occurrence retains its
   source path, one-based line, exact raw record, and a content hash over the
   documented byte convention. Two identical records on different lines stay
   distinct. Rebuilding the index leaves every source byte-identical, including
   `research/tree.md`, `research/bandit/arms.json`, and `scripts/bandit.py`.
2. **No ID-based collapse.** IDs are labels, not primary keys. Keep every row
   occurrence and represent repeated IDs as a group with all source locators;
   never select one row as the only record. Preserve original row order within
   each source.
3. **No invented relationships.** Resolve explicit hypothesis/experiment IDs
   and explicit parent/related-node links. Missing references are reported as
   missing; multiple plausible matches are reported as ambiguous. A nearby ID,
   matching filename, shared keyword, or later timestamp is not proof of
   association. Keep inferred links visibly separate from explicit links.
4. **Observed status is not current truth.** Imported status strings describe
   what a source record says. They must not silently become a live scientific
   verdict. A curated current node has an explicit review/provenance locator;
   auto-imported nodes use a neutral label such as `recorded/latest` and do not
   imply `supported`, `replicated`, or `contradicted`.
5. **Receipts and ledger are separate evidence.** Preserve declared ledger
   status, attempt/run identity, and authoritative receipt status as separate
   fields. Surface missing, conflicting, or mismatched receipts. Do not infer
   success from a planned/admitted row, a filename, or a stale status. Do not
   erase the failed E-492 v1 when the v2 receipt succeeds.
6. **Legacy numeric beliefs are quarantined.** The 18 old arms remain available
   as historical source records, but are excluded from ranking and numeric
   aggregation until explicitly reviewed. Never manufacture a prior from a
   score, status word, correlated effect, or absent value. Do not add correlated
   effect scores together.
7. **Filesystem completeness is explicit.** Inventory must include ignored and
   untracked research files and identify snapshots/provider outputs. Do not use
   `git ls-files` as the only walk. Report unreadable, skipped, symlinked,
   oversized, or otherwise excluded entries with a reason and count; omissions
   must not disappear silently. Cataloging binary outputs must not parse or
   execute them.
8. **Stable, non-recursive generation.** Generated indexes and their snapshots
   must be excluded from their own source walk by an explicit rule. Repeated
   builds over unchanged inputs are deterministic. Each config snapshot is
   immutable and content-addressed; a new config revision creates a new
   snapshot rather than overwriting the old one.
9. **No execution authority.** Bandit allocation can nominate only preparation
   or next-decision work. It cannot launch training, provider jobs, uploads, or
   system services; the generated configuration must keep launch authority
   false.
10. **Extensible records, bounded node kinds.** Unknown fields and future source
    metadata survive import/export. The current schema supports `arm` and
    `lineage` curated nodes; a new node kind must trigger an explicit schema and
    builder update. Missing files produce diagnostics rather than truncating a
    record or failing without an audit trail.

## Fixture attacks to require before accepting the generated tree

| Fixture | Required result |
|---|---|
| Same ID on two lines, including byte-identical JSON | One navigation node with two independent row occurrences, each carrying `(path, line, sha256, raw_record)` provenance |
| Duplicate experiment IDs with different attempt names/statuses | All attempts retained; no `dict[id] = row` overwrite |
| Malformed JSONL line between valid rows | Error with path and exact line; other rows retained only if partial-import behavior is explicit |
| Explicit reference to absent H/E ID | Missing-reference diagnostic; no fabricated placeholder treated as evidence |
| Two candidate files/records matching one weak heuristic | Ambiguity recorded; no automatic association |
| Parent cycle and self-parent | Cycle diagnostic; deterministic rendering that terminates |
| Ledger says completed but matching receipt says interrupted/zero fits | Conflict surfaced with both locators and attempt identity; no success promotion |
| Ledger has an admitted attempt but no receipt | `receipt_missing`/equivalent, never inferred complete |
| Two attempts for one E-ID, earlier failed and later successful | Both preserved; latest is chosen only by matching explicit attempt chronology/identity |
| Ignored, untracked, hidden snapshot, provider, and binary fixture files | Every in-scope path represented, correctly typed or explicitly excluded with reason |
| Symlink escaping the catalog root and a symlink loop | No traversal outside the configured catalog root or infinite walk; symlink metadata is retained |
| Existing config snapshot followed by revised config | First hash and bytes remain unchanged; second gets a distinct immutable snapshot |
| Legacy arms with positive and negative values but `reviewed=false` | All 18 visible as historical; none eligible or ranked |
| Prep-only candidate with `launch_authorized=false` | Allocation output cannot call a runner, write a launch request, or enable launch |
| Unknown future JSON fields | Raw record/config metadata is preserved as an extension, not dropped |
| New future node kind | Rejected clearly until the schema and builder add that node kind; no silent coercion to an existing type |

Use only tiny temporary fixtures for these cases. The empirical-validation
guidance in `conduct-creative-research/references/empirical-validation.md`
requires keeping evidence exposure and lineage attached to results; the
research-artifacts guidance requires append-only raw logs and content hashes.
Those principles apply to this index even though this task performs no model
evaluation.

## Verification completed

- The five independent fixture tests in
  `tests/test_research_tree_adversarial.py` passed with plugin autoload disabled
  and BLAS/OpenMP threads capped at one. The controller reports the combined
  builder and adversarial fixture suite passed 10 tests. Fixtures cover exact
  raw JSONL bytes including CRLF/trailing spaces, duplicate IDs and row order,
  missing references, ambiguous artifacts, explicit cross-links, preserved
  future fields, hidden/ignored/snapshot/provider files, a symlink and loop,
  immutable config snapshots, malformed-line locations, legacy-prior quarantine,
  and launch rejection.
- I independently inspected the first generated full tree: 723 nodes, 179
  hypothesis rows, 506 experiment rows across 491 IDs, one journal row at that
  snapshot, 18 archived legacy arms, and 57,717 catalog paths under
  `research/` and `reflect_05_day/`. Every generated hypothesis/experiment row
  had `raw_record` bytes that matched its original source line and SHA-256.
  Every original hash in `ORIGINALS.json` matched the corresponding file. The
  controller independently checked exact catalog path coverage and all
  generated Markdown links. The catalog count is snapshot-specific; later
  additions, including the preservation receipt and second journal event, are
  expected to change it.
- That first generated snapshot surfaced two false unresolved evidence links:
  one free-text summary and one document path had been treated as experiment
  IDs. The builder now stores non-ID evidence as raw evidence references and
  only constructs experiment edges for E-prefixed references. The adversarial
  tests passed again after that correction. The controller is rebuilding and
  checking the final generated files after this audit document is included in
  the catalog.
- Receipt interpretation is manually curated in `evidence_states`, with
  attempt scope and source paths rendered in the generated index. The tree does
  not infer provider state from filenames or attempt order. The latest remote
  state is unverified in this session; the status notes record the last saved
  monitor results and preserve that limitation. Imported ledger statuses stay
  labelled as recorded, and current evidence claims stay within their stated
  scope.
- No model, training, or scoring runs were started for this audit. Historical
  arm priors remain archived and ineligible for ranking; the current objective
  still records public 0.6660 and an unmet 0.7000 milestone.
