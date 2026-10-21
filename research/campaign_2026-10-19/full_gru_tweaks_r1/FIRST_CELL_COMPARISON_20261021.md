# First-cell five-arm comparison — F0 / seed 1001

The fixed five-arm comparison is complete. AllThree has the highest observed WP in this panel: **0.6759087040264298**, versus Control **0.6748302983858399**. The paired AllThree-minus-Control difference is **+0.0010784056405884757**, with an unadjusted 95% component-bootstrap interval **[+0.00006874907122863817, +0.0022453002986553526]**. The 468 evaluation groups are fit-excluded but repeatedly exposed development evidence. ROOT accepted the original score report before exactly stopping the successful scorer; all six comparison fits and five prerequisite fits are separately closed.

## Scope and evidence

The captured result is [`RESULT.json`](records/first_cell_evaluation_v1/score_v4/status_capture_v2/RESULT.json), 33,852 bytes, SHA-256 `1646d8656a042ce49b355f22521134206d601c1dbac3697991fcd308aa1a94d5`. Its schema is `fg19.first-cell-five-arm-streaming-score.v1`, status `FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED`, and its recorded creation time is 2026-10-21 13:05:12.064112 UTC. ROOT's one bounded capture ran 13:08:50.695934–13:08:51.566585 UTC; the observation was recorded at 13:08:51.597456 UTC and joined the exact result bytes and digest. See [`OBSERVATION.json`](records/first_cell_evaluation_v1/score_v4/status_capture_v2/OBSERVATION.json), [`TRANSPORT.json`](records/first_cell_evaluation_v1/score_v4/status_capture_v2/TRANSPORT.json), and the complete [`score-v4 record`](records/first_cell_evaluation_v1/score_v4/README.md). The native score unit was `computer-vision-homework-fg19-first-cell-score-v3.service`, invocation `9bd64d6759e8462bb40985b06a9a0ad6`; systemd reported `active/exited`, main PID 0, `ExecMainCode=1`, `ExecMainStatus=0`, and `Result=success`. The launch receipt is explicitly `ACCEPTED_SINGLE_CPU_SCORE_LAUNCH_NOT_RESULT` in [`LAUNCH_ACCEPTANCE.json`](records/first_cell_evaluation_v1/score_v4/LAUNCH_ACCEPTANCE.json); it is not the score acceptance.

The export stage separately completed the fixed five prediction exports and was accepted and exact-stopped. Its root records are [`EXPORT_ACCEPTANCE.json`](records/first_cell_evaluation_v1/export_v5/EXPORT_ACCEPTANCE.json) (8,833 bytes, SHA-256 `5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d`) and [`EXACT_STOP.json`](records/first_cell_evaluation_v1/export_v5/exact_stop/EXACT_STOP.json) (1,140 bytes, SHA-256 `695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f`). The captured export manifest is [`EXPORT_MANIFEST_V1.json`](records/first_cell_evaluation_v1/export_v5/status_20261021T1107Z/EXPORT_MANIFEST_V1.json), 23,130 bytes, SHA-256 `6b93cabf0e0b4450a11823682dfeba9396076efbc7b4a2033673fc46b7f04c23`.

The scoring pipeline used source commit `68ff682977b822f38351aac0f621146f5f13b440`, source manifest SHA-256 `a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`, source-set SHA-256 `9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8`, split-manifest SHA-256 `c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7`, and ordered-membership SHA-256 `18f0c9514d25e94cbe5eb24607879324bb83d117d962670d4ddf9733645f4f0c`. The source-only scorer gate is recorded in [V7's producer-consumer review](preparation/first_cell_evaluation_score_v7/SCORE_SOURCE_REVIEW_V7.json); the one-shot score admission and its exact-body review are [V14](preparation/first_cell_evaluation_score_admission_v14/REMOTE_SUBMISSION_V14.py) and [its review](preparation/first_cell_evaluation_score_admission_v14/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V14.json).

This report is limited to one F0, seed-1001, fixed-epoch-5 five-arm panel. All five model records specify epoch 5 and 44,760 successful updates. The fit audit tracks six comparison fits separately from five prerequisite fits; the `fixed_models` list in this result identifies the five checkpointed models actually included in the scored panel. All checkpoints are 4,279,970 bytes. Their exact SHA-256 values and representation modes are:

| Arm | Representation | Run | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| Control | `base` | `fg19_child_control_f0_s1001_fit_v1` | `d3f25e52aa590029593e5af6ea06c6d4f460bc6131ffa85d4cf1cb09c36b3b3f` |
| EMA | `ema` | `fg19_child_ema_f0_s1001_fit_v1` | `6e90bc4dba8add0caf670097519f08f3ce239d0b64c37c47b3760aa22af41862` |
| Sparse | `sparse` | `fg19_child_dictionary_f0_s1001_fit_v1` | `b840d9c81d46405f2d2439b94ad6ef15ff8501bb14715ec79792d980efbf7151` |
| NLL | `base` | `fg19_child_nll_f0_s1001_fit_v1` | `b19a61363b4b49cd98d0277f2f74fd3aa3db50bb6c1fcb314b5fab571e27fd58` |
| AllThree | `ema_sparse` | `fg19_child_allthree_f0_s1001_fit_v1` | `7072077099744e68ff23de3e390b146927c23da0e1c54f5bb18eb2fb04676c07` |

## Observed metrics

The reported metric is `global_wp_clip2_v1`. The official row mask is `is_scored AND (t >= 99)`. Each arm uses the same 808,863 selected rows.

| Arm | t0 | t1 | WP |
| --- | ---: | ---: | ---: |
| Control | 0.6658497519382944 | 0.6838108448333853 | 0.6748302983858399 |
| EMA | 0.6657293529883279 | 0.6838290889699338 | 0.6747792209791308 |
| Sparse | 0.6658488141655221 | 0.6840207021282398 | 0.674934758146881 |
| NLL | 0.6664680931659178 | 0.6846693457432622 | 0.6755687194545901 |
| AllThree | 0.6670130645936857 | 0.6848043434591738 | 0.6759087040264298 |

The result reports paired candidate-minus-control differences with 95% component-bootstrap confidence intervals. The values below are copied from the captured report; they have not been recomputed or formed by summing local deltas. These intervals are unadjusted.

| Contrast | Mean difference in WP | Unadjusted 95% CI |
| --- | ---: | ---: |
| EMA − Control | -0.00005107740670973193 | [-0.00025864174751205615, 0.00008019527731853432] |
| Sparse − Control | 0.00010445976104078714 | [-0.0003913196405530839, 0.0005018296697510554] |
| NLL − Control | 0.0007384210687499948 | [-0.0002927248953398909, 0.0018695403768434257] |
| AllThree − Control | 0.0010784056405884757 | [0.00006874907122863817, 0.0022453002986553526] |

AllThree has the highest observed WP in this particular panel. Its recorded paired interval is above zero for this resampling procedure, while the EMA, Sparse, and NLL intervals include zero. This is evidence about these five fixed checkpoints on this particular development panel, not a guarantee of improvement on new data.

## Exposure and uncertainty

The 468 groups are fit-excluded but previously exposed development groups, not a pristine or public holdout. They comprise 141 components. The intervals use 2,000 centered-moment paired component-bootstrap replicates with seed 271828. They quantify resampling variability conditional on these fixed epoch-5 models; they do not estimate fit-seed uncertainty, endpoint-selection uncertainty, or public-transfer uncertainty. No arm was retuned or selected by this scoring result, and the result does not support a general forecast or public-performance claim.

## Lifecycle closure and audit

[ROOT score acceptance](records/first_cell_evaluation_v1/score_v4/SCORE_ACCEPTANCE.json) was durably recorded at 13:23:06.724383 UTC: 21,229 bytes, SHA-256 `6e9a8162321b2490d5131039e4fe9cf9aded13e58b789d5789491558861c0ed8`. It joins the original result bytes, the five fixed model identities, source/split/export evidence and scalar tables. [ROOT acceptance binding](records/first_cell_evaluation_v1/score_v4/ROOT_ACCEPTANCE_BINDING.json) records the exact local inputs. The [independent metadata audit](records/first_cell_evaluation_v1/score_v4/ACTUAL_SCORE_ACCEPTANCE_METADATA_REVIEW.json) passed those joins without reading prediction, model or data arrays and without recomputing metrics.

The [one exact successful-score stop](records/first_cell_evaluation_v1/score_v4/exact_stop/EXACT_STOP.json) was captured at 13:29:44.567799 UTC on B: 1,211 bytes, SHA-256 `7ae50e926fb66948a7a3c95ce227c4fc11c88d68e55f2483012275c662bb953e`. Only score-v3/invocation `9bd64d6759e8462bb40985b06a9a0ad6` changed from successful `active/exited` to `not-found/inactive/dead`, PID zero and empty invocation; the campaign had no populated descendants before or after. The earlier acceptance's `PENDING_EXACT_STOP` is preserved history, superseded by this separate stop receipt. Failed score-v1 and score-v2 remain preserved and excluded. ROOT transport clocks and B capture clocks are separate samples.

The comparison used one parent plus five child fits (six scientific fits total), with three shared teachers, propensity and the learned dictionary counted as five separate prerequisite fits. All six comparison fits were integrity accepted and exactly stopped before quality evaluation. Prediction export used three engineering units (two failed, one successful) and CPU scoring used three engineering units (two failed, one successful); preflight failures and local metadata repairs did not create scientific fits. No further fits, re-export, checkpoint selection, retuning or blend search followed this result.

## Matched recipe

All five children start from the same accepted parent at epoch 3, preserve its normalization and initial trunk/mean, and zero-extend 211→467 inputs and 23→25 outputs. They use H256×2 GRU, a linear one-layer mean/variance head, dropout zero, strict FP32, seed 1001, SEEDED sample order, B16/C2000, full 20k sequences, six epochs with fixed epoch-5 selection, AdamW (LR 1e-4, weight decay .01, beta2 .999), OneCycle warmup .05, clipping 1, the same auxiliary weights and teacher-mixture target. Normalization comes from the parent's first two B16 fitting batches; it is not refit on all sequences.

Control uses base inputs and WMSE. EMA adds causal signed input-history EMA with half-life 32; this is not parameter EMA or SWA. Sparse uses the same accepted learned dictionary, stable signed top-4 of 16 coefficients and 112 normalized residual features, without refitting the dictionary. NLL uses the coupled mean/log-variance objective with base inputs. AllThree combines dictionary features, input-history EMA and coupled NLL. The WMSE/chunk compatibility options and mean/variance head remain common. The current teacher-target/AdamW recipe is a declared adaptation, not a reproduction of the historical FP16 endpoint.
