# PR81 MIDAS metadata and evidence bridge

## Metadata-only route preview

`route-preview.json` is the unmodified JSON output of MIDAS `midas task --kind research --json` for the PR81 cards 01–03 preparation request. Its SHA-256 is `67bc1aace54cb16fb1639dfc6dd3658716b5ede89bf820f5071647af8f0b493f`. It was produced from `/Users/alfa/Documents/MIDAS_worktrees/training-evidence` at source commit `dec62b91120bfb6aaf4b0d214f4c4e2ac88bc54d`.

The preview selects the registered `research` product and reports `launchable_remotely: false` and `approval_required: true`. It is a deterministic routing preview, not a card decomposition or a research-agent run. No worker, model agent, GPU job, fit, import, or external provider call was started. The preview records the selected product's capabilities; it does not show that those capabilities were exercised.

## Exporting a fit receipt

`midas_evidence.py` exposes `export_bundle(fit: Path, score: Path | None, output: Path) -> Path` and a CLI for a future fit:

```sh
python3 research/experiment_drafts/impact-portfolio-20261111/midas_evidence.py \
  --fit /path/to/fit-output \
  --score /path/to/fit-output/score.json \
  --output /path/to/new-evidence-bundle
```

`--score` is optional. When provided, the exporter requires the fixed completed `model.pt` endpoint, its terminal hash, matching config/source/split identities, `global_wp_clip2_v1`, exactly 3,528,273 selected rows and 1,873 sequences, and finite `wp`, `t0`, and `t1` values. It rejects retained-epoch scoring and records the score as `outer_validation`; inner-epoch scores cannot be exported through this path.

The fit must contain the byte-preserved `split.json` snapshot, its `owner.json`, run metadata, training metrics, and the producer logs expected from PR75. The exporter binds the split bytes to both the fit metadata and PR81 pin. A completed fit must include the fixed model and terminal status, a complete epoch index for all-epoch retention, and monitor event/evaluation logs.

The bundle contains bounded metadata, the raw training/monitor/rank logs that exist, the split snapshot, epoch index, score JSON when supplied, and per-epoch metric producer rows. It hashes `model.pt` but never copies it; the MIDAS checkpoint entry has `path: null` and is therefore an external hash claim. Dataset and model bytes are not included. The owner receipt is hashed into the local external run ID and bundle manifest, but its owner details are not copied. `campaign.stage` is `B` as an explicit PR81 initial-screen catalog mapping; this does not identify the work as historical R4-B.

## Importing evidence later

After a real fit has produced a bundle, the documented MIDAS v1 import command is:

```sh
PR81_WORKTREE=/Users/alfa/Documents/WunderFunder_worktrees/next-impact-experiments-20261111
MIDAS_SOURCE=/Users/alfa/Documents/MIDAS_worktrees/training-evidence
BUNDLE=/path/to/new-evidence-bundle
PYTHONPATH="$MIDAS_SOURCE" python3 -m midas training-evidence import \
  --receipt "$BUNDLE/receipt.json" \
  --evidence-root "$BUNDLE" \
  --workspace "$PR81_WORKTREE" \
  --home "$PR81_WORKTREE/.midas/impact81" \
  --json
```

This command is documented for future use and was not run. MIDAS stores the immutable evidence archive in the selected PR81 workspace under `.midas/training-evidence/`; the isolated `--home` keeps its runtime database under the same PR81 worktree. The importer validates local metadata, split and producer files without launching or attaching to a run. With the checkpoint path omitted, MIDAS labels the checkpoint digest `external_claim_only`: it does not fetch weights, verify model bytes, or independently recompute the score from predictions. No score or imported observation exists yet.

## Source files

- `../midas_evidence.py`: receipt construction and CLI.
- `../midas_evidence_prepare.py`: admission of existing fit and score artifacts.
- `../midas_evidence_binding.py`: frozen cell, split, source and metric checks.
- `../midas_evidence_io.py`: bounded, symlink-rejecting JSON and file-copy helpers.
- `route-preview.json`: exact metadata-only MIDAS route output described above.
