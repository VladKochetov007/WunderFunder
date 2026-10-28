# Source-separated Transformer → GRU comparison

**Prepared implementation; receiver acceptance and all six fits remain unrun.**
Target: `research-team`. Base: `7e4074314a62d0907c9608f9220cf72958708a2d`.
H-SOURCE-HOLDOUT-TRANSFORMER-KD-20261028 /
E-SOURCE-HOLDOUT-TRANSFORMER-KD-20261028-R1.

The next useful question is whether a fresh compact Transformer helps the same
GRU through distillation when **no official valid-source row enters fitting**.
This packet implements that comparison using the canonical trainer, not a copied
training loop. It does not resume any existing experiment or prepare a submission.

## Frozen panel

| Cells | Model | Optimizer / peak LR | Batch per rank | Objective |
|---|---|---|---|---|
| teacher_s1001, teacher_s1002 | Transformer768×16, 12heads, FF4, context1024 | Muon + AdamW / 1e-4 | 128 | Label WMSE |
| labels_s1001, labels_s1002 | GRU256×2 | AdamW / 1e-3 | 32 | Label WMSE |
| kd_s1001, kd_s1002 | Same GRU256×2 | Same AdamW / 1e-3 | 32 | (Label WMSE + same-seed teacher WMSE) / 2 |

Six total new fits, including two teachers; zero reused controls. Two seeds on
**one source-separated split**, not F0/F2 cross-validation. Both student arms
have the same seed, initialization recipe, ordering and update budget. Raw112,
dropout0, five epochs, W2, chunk512, accumulation8, warmup5%, cosine floor0.1,
weight decay0.01, clip1. Teacher attention is efficient and blocks are compiled;
GRU uses its canonical math path. All use explicit BF16/BF16x9 with FP32 masters,
loss/gradients/state. This is the continuing user-approved precision amendment,
not a claim of IEEE-FP32 parity. Full settings live in `configs/*.json`.

**FINAL5 is fixed before fitting.** No early stopping, per-epoch full-valid grading,
best-epoch selection or teacher search on valid. `final_only` retains the canonical
safe resumable state and final model, not five model snapshots. Batches and runtime
remain unprofiled candidates. Teacher/GRU contrasts compare recipes at different
compute, while KD/labels is the matched student comparison.

## Data and interpretation

Preparation creates `cache/source_holdout_{train,valid}_index.npz`,
`cache/source_holdout_chains.npz` and `cache/source_holdout_inputs.json` under the
provided data root. Existing historical caches are preserved. The new feature-only
graph uses mutual nearest endpoint L1 links, with the historical within-sequence
99th-percentile distance cutoff. Every physical group is included, including
train528; missing, malformed or undecodable groups cause an error. There is no
hard-coded expected group count or automatic omission.

`ROOT/split.json` uses `wnn.sequence-source-holdout.v1`: train-source fitting only,
empty valid fitting membership, all official valid groups for evaluation. Train
groups in any connected component containing valid are removed. Fit-only
normalization and fresh initialization apply to teacher and students. Teacher
targets are generated only for fitting members from that seed's fresh fixed-final
teacher; they are in-sample teacher targets, not cross-fitted targets.

The guarantee is zero shared components **under the recorded linkage rule**.
Unknown duplication/dependence remains possible. Valid features enter graph
exclusion only; valid targets never enter training/preprocessing. Source separation
does not establish calendar order, and `seq_ix` is not a verified timestamp.
The full valid set has already been exposed to team research; it is not pristine
confirmation. Source parquets are bound by size/mtime/footer identity, and graph/
indexes by full SHA256. The portable-source registry verifies full bytes when
relocating inputs. Same-size body modifications with restored metadata are outside
the local immutable-file assumption; use the portable registry for full-byte binding.

The completed [PR #43 result](https://github.com/VladKochetov007/WunderFunder/pull/43#issuecomment-6407234490)
motivates the teacher but used a different development split and fitted some
valid-source rows. Its weights are ineligible here. [PR #46](https://github.com/VladKochetov007/WunderFunder/pull/46)
uses another GRU input representation/trainer/horizon; its results are not reused
as this packet's labels-only control. No cross-protocol score subtraction.

## Receiver commands

Run these **only on the Linux B300 receiver**, in a clean checkout of the reviewed
published commit, with the team's already provisioned Python environment. No
dependency installation or job launch happens when reading this packet. The
operator must claim cells and select GPUs free of other jobs.

Copy `receiver.env.example` to a location **outside the checkout**, replace the
paths, two GPU IDs and full 40-character Git commit, then source it. Keep the data
root and a fresh run root outside the checkout and separate from each other.
Do not derive the expected commit from an arbitrary current HEAD without reviewing it.

```bash
cd /srv/WunderFunder
source /srv/experiments/source-kd.env
PACKET=research/experiment_drafts/source-holdout-transformer-kd-20261028
COMMON=(--root "$WNN_SOURCE_KD_ROOT" --data-root "$WNN_DATA_ROOT" --expected-commit "$WNN_SOURCE_KD_COMMIT")
bash "$PACKET/run.sh" prepare "${COMMON[@]}"
bash "$PACKET/run.sh" accept "${COMMON[@]}"
```

`prepare` reads source features/IDs and writes immutable caches/split/preparation
receipts. `accept` explicitly runs the bounded receiver correctness checks and
records their result. Inspect their evidence before profiles. Neither is a fit.

Choose **one cell at a time**. Start with `teacher_s1001`; repeat the same explicit
profile/admission/fit stages for `labels_s1001`, `kd_s1001`, then seed1002, honoring
the teacher-cache dependency. No script advances to the next cell automatically.

```bash
CELL=teacher_s1001
bash "$PACKET/run.sh" profile "${COMMON[@]}" --cell "$CELL"
```

Inspect `ROOT/CELL/profile/profile_receipt.json` and both rank logs: twelve real
updates, finite losses/gradients, exact config/split/source, actual devices/precision,
peak memory and startup cost. Set actual available host RAM for this job, excluding
other workers; do not enter total machine RAM if it is shared.

```bash
REVIEWER=receiver-owner
HOST_RAM_GIB=128  # Replace with this job's actual allocated RAM.
bash "$PACKET/run.sh" admit "${COMMON[@]}" --cell "$CELL" --reviewer "$REVIEWER" --host-ram-gib "$HOST_RAM_GIB" --max-vram-fraction 0.8
bash "$PACKET/run.sh" fit "${COMMON[@]}" --cell "$CELL"
```

Admission creates a separate reviewed receipt; it never edits the raw profile.
RAM/VRAM/scratch checks are engineering headroom checks, not speed or quality
results. A failing profile requires an explicit reviewed recipe revision and new
output root, never silent batch or precision fallback.

After each teacher's successful FINAL5, create its fit-only cache before that
seed's KD profile. This command accepts only teacher cells:

```bash
bash "$PACKET/run.sh" cache "${COMMON[@]}" --cell teacher_s1001
```

Evaluate every completed fixed-final cell once. `evaluate` exports full valid and
invokes the official pooled scorer. After all six complete, create the report:

```bash
bash "$PACKET/run.sh" evaluate "${COMMON[@]}" --cell "$CELL"
"$PYTHON_EXECUTABLE" "$PACKET/compare.py" "${COMMON[@]}" --output "$WNN_SOURCE_KD_ROOT/comparison.json"
```

`compare.py` rejects an incomplete panel, changed receipts, mismatched splits or
teacher ancestry. It reports every t0/t1/WP, both paired seed differences and
conditional component-bootstrap intervals. Those intervals condition on fitted
predictions and resample graph components; they do not estimate training variance
or restore fresh holdout evidence. Mean KD−labels ≥+0.001 with both seed deltas
nonnegative signals a separately allocated follow-up, not significance or a launch.
Mixed/negative results remain recorded; no automatic direction closure.

## Resume, failures and artifacts

`resume --cell CELL` reuses only an existing safe `fit/training_state.pt`, with the
exact config/source/data/runtime identity and original approved profile. It does
not overwrite a terminal run. `score --cell CELL` retries scoring a completed
evaluation export when no score exists; `seal --cell CELL` validates an existing
score and creates only its missing evaluation receipt. Incomplete exports are
preserved for diagnosis; do not delete/replace evidence automatically. Prepare,
profile, fit and export commands have explicit time limits and process-group
cleanup; a healthy pre-existing job is never a cleanup target.

Outputs: `ROOT/packet.json`, `split.json`, correctness receipt, per-cell original
and approved profiles, `fit/model.pt`, safe state, terminal status, rank logs,
teacher `cache/manifest.json`, `evaluation/manifest.json`, `score.json`,
`evaluation_receipt.json` and final `comparison.json`. Collect logs/JSON separately
from weights/arrays, with archive SHA256, actual source commit and storage paths.
Keep failures and incomplete cells in the trace. No model/data files belong in Git.

## File map and readiness

- [PR.md](PR.md): mechanism, comparison, uncertainty and all eleven research questions.
- [recipe.json](recipe.json), [configs](configs), [receiver.env.example](receiver.env.example): frozen panel and environment template.
- [run.sh](run.sh), [packet_runtime.py](packet_runtime.py): explicit receiver actions.
- [packet_contract.py](packet_contract.py), [packet_acceptance.py](packet_acceptance.py), [packet_admission.py](packet_admission.py), [packet_lineage.py](packet_lineage.py): source/config/data, receiver correctness, profile and teacher bindings. Correctness artifacts are `ROOT/acceptance/{command.json,pytest.xml,receipt.json}`.
- [compare.py](compare.py): complete paired report; no training action.
- [REVIEW.md](REVIEW.md), [VALIDATION.json](VALIDATION.json): source review, actual static checks and pending receiver evidence.
- [Source split producer](../../../scripts/prepare_sequence_source_split.py): full-source materialization and canonical loader integration.

Local activity is limited to source edits, AST/JSON/shell inspection and static
lint/format checks. No local tests, model imports, training, inference or profiles.
Actual receiver tests/profiles and all six scientific fits remain pending.
