# TL3 role score replay R4

Status: complete score-only replay of existing artifacts. This is development
evidence on the exposed official validation set, not an independent or public
score and not a new model result.

## Result

R4 scored the three pinned role prediction arrays on all 1,873 official
validation sequences. The `is_scored` mask after row 99 selected 3,528,273
rows. The metric is `global_wp_clip2_v1`.

| Role | Fixed checkpoint | t0 | t1 | WP |
| --- | --- | ---: | ---: | ---: |
| `tl3c_w18final_distilled_s5` | E1 | 0.6878701883065461 | 0.6949628197154484 | 0.6914165040109972 |
| `tl3c_fttall_tlead_s5` | E3 | 0.6878379834029539 | 0.6949925748101522 | 0.6914152791065531 |
| `tl3c_fttall_tlead_s6` | E3 | 0.6868911599282501 | 0.6950066790398436 | 0.6909489194840468 |
| Fixed blend, weights 0.34/0.33/0.33 | Fixed | 0.6887196914805331 | 0.6960351581533012 | 0.6923774248169172 |

The blend uses FP32 left-to-right multiplication and addition with no clipping.
Its WP exactly recovers the currently recorded baseline mixture score
`0.6923774248169172`; this is a cache grading replay, not a gain.

## Bound inputs and available weights

Native project root:

`/data/epanov_s/tabdata/runtime/architecture-execution-20261107-r1/project`

| Role | Checkpoint path | Checkpoint SHA256 | Prediction array path | Prediction SHA256 |
| --- | --- | --- | --- | --- |
| `tl3c_w18final_distilled_s5` | `artifacts/runs/tl3c_w18final_distilled_s5/model_ep1.pt` | `a1934af24bd1520b78be874c294532b1e2d62acdd8bd1dc3d6b5c220b2dad107` | `cache/pred_tl3ref_tl3c_w18final_distilled_s5_valid.npy` | `e9a664be18feec38563be730c8a1f64dbc2a9907ed0d4d4873309d674f78eeb6` |
| `tl3c_fttall_tlead_s5` | `artifacts/runs/tl3c_fttall_tlead_s5/model_ep3.pt` | `cf944dffac66d674894e4c589d839c407b3d28cb01d35a26f58ff08c7743f65f` | `cache/pred_tl3ref_tl3c_fttall_tlead_s5_valid.npy` | `3c124a7e89110472b6b8fb878e09087fddf00a2b7f4983b8115c8acbd53c7352` |
| `tl3c_fttall_tlead_s6` | `artifacts/runs/tl3c_fttall_tlead_s6/model_ep3.pt` | `24a565b03e35dc36679bf809f63f605f24f693404bd0cfda58bfc82177a33598` | `cache/pred_tl3ref_tl3c_fttall_tlead_s6_valid.npy` | `67d3c7804d571b1accf096d14b10b7c768d27cb87849654a36d04ed72b0588e6` |

The full native paths append these relative paths to the project root. Each
prediction array is 299,680,128 bytes, FP32, and ordered by the official valid
source index. Its sidecar's `row_groups` was required to equal the manifest's
validation source order exactly. Provenance fit/evaluation arrays use a separate
seed-shuffled order; R4 checked their exact member sets, lengths and uniqueness,
recorded observed and manifest order hashes, and never used those lists to
reorder prediction rows.

For all three roles, the provenance evaluation observed-order SHA256 is
`ce39df19a7d27a27da9156f0577b0de3153e5af08408adacec758ca0f1c74d8c`; its
manifest-order SHA256 and prediction-row-order SHA256 are both
`73e879bb572abbab3d4b1af9bb6ab179c54c8b2c2ee8485242bbb153fe5ef84e`. The
empty fit-valid list hashes to `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
The two observed fit-train order hashes are
`9123f52968cad144a3eeb72c05d9bedfb8a4462e8bbd680fce48928823bef1d4` (s5)
and `353ad84eb4747bd925121e0f75c889904f5cf6b2516cfb479f9e2f236173b013`
(s6); both have the same manifest-order SHA256
`3e3e8abef040d512363a557a01670696b9a1ee0cb66afd8d098d059b63f9e110`.

R4 source SHA256:

`581b30f2093dadcf6ca15eaa7d776137832d8cc1565b2bf4c1688458973eb0e0`

Native receipts: `027-live-training-and-tl3-r4.json` contains the score and
`028-fetch-tl3-result-and-first-evaluations.json` contains the full R4 result
record. The result record binds checkpoint, prediction, sidecar and provenance
hashes, split manifest SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`, and
evaluation membership SHA256
`008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7`.

The raw `provenance.json` byte hash is checked against each sidecar. The
mixture-manifest provenance pin equals that raw-file hash for all three roles;
R4 reports the different canonical JSON hash separately. After scoring, R4 rehashed the sidecars,
provenance files, prediction arrays and checkpoints. It validated the parquet
and split manifest when loading inputs, but did not rehash those two inputs at
the end. No model was loaded or run, and no cache or dataset was written.

## Execution failures are grader history, not model evidence

| Attempt | Source SHA256 | Failure or result | Interpretation |
| --- | --- | --- | --- |
| R1 | `26daeee92619cbd993f8b5c122d7e0720dca646862b3e10fee90c4621ee865f6` | Assumed nonexistent `CampaignSplit.evaluation_source` / `evaluation_mask` attributes. | API adapter error before scoring. |
| R2 | `6296d8755d3708246058b8e04b6ae439980f0fd38fc12924ebca8738bb771bc6` | Assumed fitting row groups were exactly `range(10_325)`. | The official 10,325-member fitting set is noncontiguous within 10,607 train-source groups; failure before scoring. |
| R3 | `d1f9925187e655a0bad8e2062ff7be76507f7a75a477081857d0ee66ae664d66` | Required provenance train/evaluation lists to match manifest order exactly. | Native lists were seed-shuffled but had identical member sets, lengths and no duplicates; failure before scoring. |
| R4 | `581b30f2093dadcf6ca15eaa7d776137832d8cc1565b2bf4c1688458973eb0e0` | Exact unique set/count validation; cache prediction order remains source-bound. | Score-only replay succeeded. |

Native receipt `025-diagnose-tl3-provenance-conditions.json` reports no
missing or extra fit-train members for any role. This resolves the R3 mismatch
as order only; it does not relax membership or ancestry checks. R1-R3 produced
no scores and must not be interpreted as negative model results.

## Limits

The official validation set has already been inspected and is exposed. The
reference mixture records `training_predictions: in-sample`, and the historical
ancestry of the consumed training-cache bytes is not reconstructed. R4 makes no
independent validation, public leaderboard or serving claim. It verifies and
re-scores the available role arrays and fixed blend only.
