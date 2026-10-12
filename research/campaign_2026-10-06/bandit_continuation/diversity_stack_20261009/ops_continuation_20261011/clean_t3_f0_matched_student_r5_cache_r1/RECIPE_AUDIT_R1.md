# Matched clean F0 strong-T3 student recipe (cache binding pending)

Status: metadata/command preparation only. No cache build, fit, or student prediction has been launched. The full-fold F1 strong student `vt_f1_dsq_dm_t3_a9_e6` is already completed and is used only as the recipe/target-semantics reference; its weights and fold membership are not reused. The existing F0 `oldz4_h230_student_f0_s3_r3` remains a comparator and is not rerun.

## Findings

The intended target is the previously selected homogeneous strong-family target: `T = float32(0.8125 * mean_float32(F0 clean-T3 seeds 3,4,5 epoch-2 predictions))`, then `0.1*y + 0.9*T` during student fitting. The student is a seed-3 GRU(256, 2 layers), six epochs, fixed epoch-5 checkpoint, FP16, batch 16, chunk 2,000, wMSE/chunk, AdamW `1e-4`/`0.01`, OneCycle, F0 propensity power 2/epsilon .03, and the existing `.3` structure/next auxiliaries.

The paired F0 initializers are the exact own control `c20261003_combo_s3_f0@ep3` (checkpoint SHA `d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`) and own treatment `kg_h382r3_f0_muonall_prop0_e4_s3@ep3` (checkpoint SHA `359d337db09f0f8e306a225b050884741f006190523b2e940d36ef87d35a25be`). Both arms use seed 3 and the same F0 split. The training pool is 10,535 FIT_TRAIN + 1,405 FIT_VALID = 11,940 groups; all 468 F0 evaluation groups are excluded. F0 is repeatedly exposed development evidence, not a fresh holdout.

Cache binding is the remaining blocker. The frozen R3 cache builder SHA `62aea91099b14cdd5ef987dcbd60675dbb650b152bdfe2a27aeda3ed24e3d12e` binds old `_r1` teacher IDs and conventional prediction filenames, while actual current teachers are `_r5`; their full-valid arrays use `_local_full_valid_r1_valid.npy` and sidecars with 1,873 rows. The final student target cache must project only the exact 1,405 FIT_VALID rows. Complete FIT_TRAIN arrays for all three current teachers still need a root-released producer path. The s3 16-group profile output is not a cache input. Use a separately reviewed, manifest-driven adapter; leave the frozen builder untouched.

The train-selector canonical-order hash is `8ebb217415c8cdc8c468bb081a7fa815e90b661fd1714f6d05da434a2fa3bbda`; training provenance records per-seed randomized fit-loader order hashes s3=`98a3e8da248f52e307af61efaeab845a76cf8e89e005ea5630c0af9a133508a9`, s4=`1c367b283bdea5b3cfb93fdea2b7e30349f99edc5d40eabe2ff056480c258c49`, s5=`6e8cb1e840301ef6c03ee3dc4a3b400c719cd4959d6f038eeddfa4b60de016c2`. All three group sets equal the same F0 membership. FIT_VALID order hashes all match at `0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75`. A cache adapter must join by row-group ID and preserve the training loader's alignment; never treat distinct ordered hashes as interchangeable.

## Recipe checks

| Dimension | Completed F1 reference | Clean F0 candidate | Assessment |
|---|---|---|---|
| Student | H256 GRU, seed 3, epochs 0–5, checkpoint 5 | Same | Required match |
| Distillation | alpha .9, homogeneous strong target | Same alpha and target transform with own F0 clean-T3 seeds 3/4/5 | Required recipe match; regenerated fold-local weights |
| Init parents | F1 own fold-1 combo parent | Same two predeclared F0 own parent arms in both paired fits | Intentional fold/parent adaptation; never load F1 weights |
| Fit rows | 10,533 train + 1,416 fit-valid | 10,535 train + 1,405 fit-valid; 468 eval excluded | Correct fold-local membership, not byte-identical to F1 |
| Propensity | fold 1, power 2, epsilon .03 | fold 0, power 2, epsilon .03 | Intentional own-fold adaptation |
| Cache producer binding | prior F1 actual metadata | current `_r5` predictions not yet bound | Pending; the existing R3 `_r1` builder cannot consume current filenames/sidecars unchanged |
| Evaluation | exposed F1 fold-1 panel | repeatedly exposed F0 468-group panel | No pristine-holdout or public claim |

## Exact source and commands reviewed

Prior student provenance SHA `327506329a6d958f6b684d04af5e6e10135c7803cd1cafa7a40fe883212a5861` and F1 cache metadata SHA `a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779`; F0 recipe SHA `fe1a3e1975fc212ad9de6d927863921efc73b4c6b1781a10e438e5903377b1cd`; parent asset SHA `30a4e7f5ebbc1a560931a5c4781b2356de289de2ac5d47d90e3bf93d9a770e4d`; stage template SHA `1f4bb362adb15fa94211e4ed585c5d0566fc58d1ea203134f598ec72762f392c`; current selector preflight SHA `701642f796a5d4fc19ccebc9c0f71d126308d74f347c784dc083fc713f6f5c7a`; current trio-grade SHA `2215ff7619767d6feb861264c2258be1d919c6a077fc4afc269b2b3ba7954470`. The student command vectors in the JSON preserve the reviewed three-stage six-epoch OneCycle schedule, point to the R3 frozen source snapshot, and change only run/cache IDs plus local resume-state paths. Relevant R3 parser defaults were checked in `scripts/train_rnn.py`: MSE, channel weights `1,1`, chunk denominator, alpha tau 32, bootstrap 0, reanchor 0, family loss 0, label weight power 1/floor 0. The R3 source delta documents update-counter telemetry only.

## Remaining limitations

No actual cache metadata/array hash is available yet; no current local fit-train prediction exists for seeds 4/5 in this recipe bundle; the local treatment parent path must be reverified at launch; the current frozen cache builder requires a new, independently reviewed input binding/projection adapter; H256/B16/C2000 profile and continuation check are still required; no fit, grade, or submission decision is authorized by this artifact.
