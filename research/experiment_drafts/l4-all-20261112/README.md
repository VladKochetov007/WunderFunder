# L4 all-experiments continuation — 2026-11-12

## First runnable cohort

This packet pins the first eight L4 fits to the exact, previously qualified PR80 receiver. It contains four paired configuration-only arms, each at seeds 203301 and 203302. No source changes or new model code are required for this cohort.

| Arm | Change from the same-seed K16 control | Fits |
|---|---|---:|
| PR82 card 05, horizon 4000 | Batch 32 × 1000 rows becomes batch 8 × 4000; rows/update stays 32,000 | 2 |
| Horizon 2000 dose control | Batch 16 × 2000; rows/update stays 32,000 | 2 |
| GRU256 depth 1 | `model.layers`: 2 → 1 | 2 |
| GRU256 depth 3 | `model.layers`: 2 → 3 | 2 |

The 2000-row dose control is a declared extension to card 05, included to tell whether the gradient-horizon effect varies smoothly between the K16 control and the proposed 4000-row endpoint. It is not represented as an original PR82 card.

## Frozen source and data

The receiver is source commit `6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba`, source identity SHA-256 `38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d`. Its archived exact Git-blob/LF source is `C:/Users/admin/Documents/Codex/artifacts/l4-stem-successors-20261111/source-r1.tar.gz` (archive SHA-256 `a48b6eccbd68b8eaf2a1fc67338861301185f2ff72df6dd3c8256c3e8263af9c`; source manifest SHA-256 `e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d`). The extracted inspection copy is at `C:/Users/admin/Documents/Codex/artifacts/l4-stem-successors-20261111/source-base-r1`; do not edit that frozen copy.

Every config uses the official chronological split SHA-256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`: 10,325 earlier train sequences for fitting and fitting-only normalization, zero official-valid fit members, and all 1,873 later valid sequences / 3,528,273 rows for evaluation. Complete sequences stay intact. The root's source-qualified membership receipt remains the final admission check.

The exact same-seed K16 controls are included under [configs/controls](configs/controls). Their original control source identity is `b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188`; their frozen raw config hashes are in the manifest. Reuse the already-fitted control weights and full-valid readout. Do not fit those controls again. Historical fixed-E6 control WP is 0.683019498 (seed 203301) and 0.682629372 (seed 203302); root's exact readout receipt binds the actual model SHAs.

## Training and evaluation contract

All candidates preserve the source's K16 causal input stem, raw112 inputs, strict FP32, AdamW 3e-4, weight decay .01, gradient clip 1, five-percent warmup, cosine-v2 floor .1, six epochs, `all_after_warmup` target mask, and all-fit train-only normalization. The fixed endpoint is E6; retain every epoch and report the full official valid score at each epoch. Also keep the fixed 32-sequence train-evaluation trace, preclip norm, LR/update counts, optimizer diagnostics, nonfinite counters, throughput and memory telemetry. The main score is the canonical pooled `global_wp_clip2_v1`; report each target and each seed's paired delta before any mean summary. Validation is exposed development data and is not a pristine holdout.

Preserve the qualified launch environment from `root_fits_frozen_r1.py`: `NVIDIA_TF32_OVERRIDE=0`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, and `PYTHONDONTWRITEBYTECODE=1`, plus the receiver's existing thread and output controls. Record the effective `torch.backends.cudnn.rnn.fp32_precision` getter in the profile/run receipt. Keep the previously accepted source and controls; do not set an additional direct IEEE property as an unrecorded runtime change.

The first cohort is bound to root `CLOCK_R1` SHA-256 `0181e9889a45d0636e3b9981ddb15e17fb5cd26429068bb0ee37f78d09d71270` (first fit 1794453619.060462; fit deadline 1794468019.060462; wave deadline 1794471619.060462). Root owns the one-GB first-cohort output cap, per-device profile admission, foreign GPU 4 headroom check, and all launches. The first fit must use a receipt-approved profile for the same source/config/data/device shape.

## Validation commands

From the exact receiver environment, first load all eight candidate configs with source `scripts/train_sequence.py`'s `sequence_config.load_config`, then use root's already-qualified profile receiver for the distinct effective shapes. Example config-loader check from the extracted source root:

```bash
PYTHONPATH="$SOURCE_ROOT/src" "$PYTHON" -c 'from pathlib import Path; from wnn.training.sequence_config import load_config; paths=sorted(Path("$CONFIG_ROOT").glob("*.json")); [load_config(path) for path in paths]; print(f"loaded {len(paths)} candidate configs")'
```

For a profile, the underlying entrypoint is:

```bash
"$PYTHON" "$SOURCE_ROOT/scripts/train_sequence.py" --config "$CONFIG" --data-root "$DATA_ROOT" --split-manifest "$SPLIT_MANIFEST" --output "$PROFILE_OUTPUT" --profile-updates 30
```

For an admitted fit, use the same command without `--profile-updates` and include `--profile-receipt "$APPROVED_PROFILE_RECEIPT"`. Root's bounded wrappers supply the exact native paths, GPU UUID, environment, `CLOCK_R1` deadline, output cap, heartbeat and evidence capture. These example commands are entrypoint contracts, not authorization to bypass those wrappers.

## Decision rule

Report every treatment and target. Preserve the PR82 heuristic as a follow-up signal: mean paired E6 gain at least +0.0005 WP with both seed deltas positive. It is not an automatic stop or a significance claim. The horizon arm must also show where any gain occurs by step bucket; depth changes are interpreted as capacity comparisons, not as proof that deeper recurrence is universally better.