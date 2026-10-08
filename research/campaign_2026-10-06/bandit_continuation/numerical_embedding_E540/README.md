# E540 feature-wise numerical embedding screen

E540 is a code and recipe handoff for a small-data GPU comparison under H234. It is **not admitted**: root owns the final split/source seal, resource profile, GPU execution and independent grade. No scientific fit or public score is claimed here.

The quant-brainteaser framing is representation geometry. Each channel is an anonymous measurement; its index does not establish depth, quote side or physical meaning. The test asks whether train-only scalar quantile coordinates before temporal mixing help the recurrent model, while retaining the raw value for tails and leaving the current event/age vector unchanged. It also separates the representation claim from measurement and capacity concerns: knots carry explicit training membership, future rows cannot affect the feature map, and fixed component membership defines the comparison.

## Lineage and evidence basis

The direct parent is H234, monotone scalar calibration before causal structured input. E537's independent H-E537-C card refined this into a feature-wise basis. Its Handoff 1, source review and quant-brainteaser cards are preserved in the E537 worktree at commit `6da11221`; the relevant paper is Gorishniy et al., [On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556). This implementation is written in the project's PyTorch modules and training flow. It does not copy the teammate or paper code, and it is not an exact reproduction of the paper.

The basis is applied to the already standardized raw112 columns. For each channel it creates eight cumulative linear ramps over at most nine unique empirical quantile knots. Ramps clamp outside the observed range; the standardized raw value remains as an unbounded bypass. Constant channels receive no active ramps but keep their bypass. The existing 99 event/age features pass through untouched, producing the same map width in both arms: `112 × (8 + 1) + 99 = 1107`.

Knot fitting uses only the frozen selected fit groups, in sorted train-then-valid membership order. It samples at most 128 evenly spaced rows per complete sequence, then keeps a deterministic reservoir of at most 65,536 rows (seed 20261008). It reads feature columns only; labels and scored masks are not requested. The fitting member list and its digest, sample counts, normalization digest, knot digest and source hash are stored in checkpoint provenance. The current sentinel policy leaves finite `-5.19934` as an ordinary input value.

## Matched arms and interpretation

Both arms use a one-layer hidden128 GRU, the same input projection, head, data subset, seed, group order, optimizer, schedule, precision, loss and fixed epoch-0 checkpoint. The provisional shape is batch2/chunk256; root can revise it only before admission if the measured profile requires a new packet.

| Arm | Per raw feature | Mapped width | Allocated trainable parameters |
|---|---|---:|---|
| `linear` | eight zero ramp coordinates + raw bypass | 1107 | 274,050 |
| `ple` | up to eight train-only quantile ramps + raw bypass | 1107 | 274,050 |

The allocated counts match, but all 896 ramp input coordinates in the linear control are zero. Their connected projection weights therefore add no data-dependent expressivity. This is a matched-shape, equal-allocation treatment contrast, not an equal-effective-capacity proof. If PLE is informative, a separately registered ordinary parameter-matched input MLP is the next descendant; a future PLE gain cannot be assigned solely to spline geometry.

The screen starts with a seeded nested 10% of complete eligible components and one full epoch. The 20 GPU-minute pair target in E537 was a proposal, not a measured cost or spending authority. The generated packet records the exact campaign split, fit subset, evaluation membership and source hashes; root first profiles both arms and checks current quota/headroom. Evaluation members are previously exposed or exposure-unknown and remain fixed. Training-label clipping stays at the common default; official scoring remains `global_wp_clip2_v1` with its fixed mask, ±2 clipping and original-magnitude weights.

## Correctness and delivery

`tests/test_numeric_ple.py` covers exact finite sentinel, partially tied and constant channels, bounded training-membership hashing, feature-only reader calls, manual basis values, out-of-range raw bypass, equal parameter layouts, buffer reload, finite gradients, chunk/full parity, row streaming and reset. All eight focused synthetic tests pass. The actual-data shape helper read train row group 0 (`seq_ix=43601`) through the project's feature-only `SequenceFile` interface and records import, footer, feature-read and derived-event phases in [the phase receipt](DATA_SHAPE_SMOKE_ACTUAL_TRAIN_RG0_V1.jsonl). It verified 20,000 rows, 112 raw features, 99 causal event features and 211 combined inputs, all finite. No targets or scored masks were loaded; no knot fitting or model fit occurred. The process peaked at 1,116,422,144 bytes RSS under the authorized 2,500,000,000-byte child cap. The 10,607-row-group footer opened at 1,062,907,904 bytes RSS, which explains why the earlier 900–950 MB attempts could not complete.

Once this commit is integrated into the project checkout that contains the supplied data and cached split artifacts, root can build the provisional packet with:

```sh
python scripts/build_e540_packet.py \
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
  --output-dir research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/packet_v1
```

The builder validates the source split, writes the exact component-complete 10% fit membership, binds the code and data identity metadata, and serializes profile, fit, prediction export and causal ONNX export commands. It launches none of them. The `grade_command` remains unset by design: root supplies the independent grader over the exact evaluation components and owns the result handoff.

The implementation and recipe are in commit `976a4d07b677544c0a4a289d29b4b1176141870b` on branch `research/e540-feature-wise-ple-20261008`. Root should integrate the commit into the main project checkout before running the builder because the trainer resolves datasets and artifacts relative to its own checkout root. The isolated worktree intentionally has no dataset copy or symlink.

## Current status

The exact authorized actual-data shape read and all focused synthetic tests are complete. `DATA_SHAPE_SMOKE_ATTEMPTS_V1.json` preserves the earlier capped failures and the successful bounded smoke with its parent-headroom check and terminal status. GPU profiling, fits, grading and provider work remain pending root admission; this code handoff reports no model result or competition claim.
