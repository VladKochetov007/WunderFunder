# Native GRU teacher preflight audit R1

**Decision:** `PASS_WITH_ONE_EXPLICIT_SOURCE_ADAPTATION`; GRU launch may proceed independently of the Transformer adapter after the final S_GRU source hash and the machineable pre-profile gate below pass. This is a fresh strict-FP32 adaptation of the historical plain-GRU recipe, not an exact historical replay. I performed no SSH, GPU profile, fit, or test.

## Findings

1. **Strict Muon was an optimizer precision hole in the base source.** In base `a52d07d1d254f638d83e2ff4a6f590b07fd9e903`, the legacy `--precision fp32` path disables autocast/TF32, but `src/wnn/training_dynamics.py` constructed `MuonAdamW(...)` without `strict_fp32=True`; `src/wnn/optim.py::orthogonalize` therefore converted CUDA NewtonР Р†Р вЂљРІР‚СљSchulz work to BF16. The current worktree contains the approved adaptation: both Muon constructor paths pass `strict_fp32=(args.precision == "fp32")` (current `training_dynamics.py:111-120, 150-153`). **Consequence:** passing `--precision fp32` on the base commit alone did not satisfy the contract. **Smallest correction:** bind the GRU run to the isolated patched S_GRU source and require its optimizer receipt to show `strict_fp32=true` on every group, FP32 params/grads/state, disabled autocast and disabled TF32. The adaptation must stay in the new source root; historical source/results remain unchanged.

2. **Do not use the canonical sequence-fit config as the shared GRU route.** It has strict-FP32 Muon, but its base validator only permitted event211 + Muon for GRU256x2 and its scheduler offered warmup-cosine, not OneCycle. It could not represent the declared 384x2 and 512x1 family under the same input/optimizer/schedule contract. The older `scripts/train_rnn.py` path accepts all three shapes with native 211 inputs, gate-wise Muon, and OneCycle. This is the consistent path for the six GRU cells; the Transformer remains a separate source family.

3. **The legacy CLI defaults are scientifically unsafe for this contract unless overridden.** `--precision` defaults to FP16 (`cli_performance.py`); `--loss` defaults to `mix` (`cli_targets.py`); `--chunk` defaults to 2000 (`cli_views.py`). Freeze explicit values for every fit: `--precision fp32 --loss wmse --batch 32 --chunk 1000 --update-rows 0 --detach-rows 0 --lr 0.001 --wd 0.01 --optim muon --muon-scope rnn --sched onecycle --warmup-frac 0.05 --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3`. Also set the architecture, seed, `--train-seqs 10325`, chronological manifest and `--full-valid` explicitly.

4. **Training-only normalization is valid but is a specific partial-fit recipe.** The legacy `fit_normalization` consumes the first two batches from the training loader, builds the same configured causal inputs, takes `f.reshape(-1, f.shape[-1])[::7]`, then takes every seventh row from the flattened batch-by-time feature matrix and fits mean/std. It receives no validation loader. Thus its ancestry is fitting-only, but it is not an all-10325-sequence moment fit. This matches the existing native path and is allowed by ADR0044; bind the normalization hash, fit membership, seed, loader/sampler identity, and source hash. Do not describe it as Р Р†Р вЂљРЎС™all-fit moments.Р Р†Р вЂљРЎСљ No initializer or prior checkpoint may be loaded (`--init` and `--resume-state` absent).

5. **The update clock is calculable from the native loader.** With 10,325 train sequences, batch32, and `drop_last=True`, there are 322 full batches per epoch (10,304 sequences seen that epoch; 21 tail sequences are dropped). Chunk/update rows1000 gives 20 optimizer updates per batch. The frozen four-epoch OneCycle horizon is therefore 6,440 updates/epoch and 25,760 total, with `pct_start=0.05` (nominal 1,288 warm-up updates). The fixed horizon must remain 25,760 in each full-fit scheduler receipt; do not use a shortened profile horizon for a fit or a stop-after-epoch continuation.

## Frozen GRU recipe and source comparison

| Contract item | Native implementation | Gate |
|---|---|---|
| Input representation | `build_inputs` concatenates raw112 and 99 causal event features by default; `RNNModel` uses encoder projection, GELU, GRU, and skip-connected head. | Require `n_in=211`; no `--raw-only`, heuristics, online normalization, research/privileged features, propensity input, numeric map, or tied-lead path. |
| Shapes | Plain GRU256x2 control, GRU384x2, GRU512x1. | `--arch gru --cell gru --enc linear --dropout 0`; explicitly set `--hidden/--layers` per cell. |
| Objective | `weighted_mse_loss` is selected by `--loss wmse`; chunk denominator is default. | Explicit `--loss wmse --loss-norm chunk`; no cached teachers, auxiliary heads, true-mask weights, class weights, or row thinning. |
| Optimizer | `muon_param_groups(..., "rnn")` places each GRU `weight_ih/weight_hh` matrix in one three-block Muon group; input projection, readout, biases and vectors remain AdamW. | Exactly one Muon gate group (`blocks=3`) and one AdamW-other group; both own disjoint/exhaustive trainable parameters; all strict FP32 after the approved constructor adaptation. Do not substitute `tl3-muon`, which is a different grouping. |
| Schedule | Native `build_scheduler` creates one OneCycleLR over the full computed fit; `cycle_momentum=True`, so Muon momentum follows beta1 schedule. | Require total_steps 25,760, max LR 1e-3, pct_start .05, weight decay .01, no optimizer switch, no warm restart. |
| Membership | `setup_lineage` rejects non-primary manifests, fit-valid membership, train+valid, caller-supplied random folds, partial fractions, class balancing, and disabled evaluation. `data_membership` selects from the manifest train groups and evaluates manifest evaluation groups. The internal `vfold` field is derived from the official manifest and is acceptable only when it resolves exactly to `evaluation_groups`. | Split SHA `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`; fit train set exactly 10,325; fit valid set empty; evaluation set exactly 1,873; official scored rows exactly 3,528,273. Do not order or split by `seq_ix`. |
| Endpoint | The checkpoint layer supports fixed zero-based epoch selection, and `outputs.py` retains epoch snapshots/history while evaluation runs. | Exactly four completed epochs, `selected_epoch=3`, `model_ep3.pt` and selected `model.pt` backed by provenance hashes; four complete-valid metric rows retained. Keep best-observed history separate from fixed epoch3. |
| Ancestry | No init is required for a teacher. The fit normalization is computed from its train loader. | Fresh seeded initialization; no PR60 teacher/student parent, all-corpus checkpoint, distillation cache, valid-derived preprocessing, or resumed state. Record source/config/environment/data hashes before launching. |

### Exact launch argument contract

Use the equivalent of the following per fit, with a unique output name and the indicated model/seed. The argument array is deliberately free of shell paths so the controller can bind its staged paths and hashes without rewriting recipe values. `--batch-order seeded` pairs each same-seed architecture on the same per-epoch row order, independent of model-initialization RNG consumption:

```text
scripts/train_rnn.py
--name <unique-cell-run>
--device cuda
--train-seqs 10325
--full-valid
--split-manifest <official-chronological-primary.json>
--train-source train
--arch gru --cell gru --enc linear
--hidden <256|384|512> --layers <2|2|1> --dropout 0
--seed <1001|1002>
--batch 32 --chunk 1000 --update-rows 0 --detach-rows 0
--epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3
--loss wmse --loss-norm chunk
--lr 0.001 --wd 0.01
--optim muon --muon-scope rnn --sched onecycle --warmup-frac 0.05
--precision fp32
```

Do not pass `--init`, `--resume-state`, `--no-eval-during-fit`, `--train-source train+valid`, `--valid-fold`, `--class-balance`, `--mask-weight`, `--distill*`, `--propensity`, `--prop-input`, `--raw-only`, or any profile/early-stop flag to a scientific fit. If the runnerР Р†Р вЂљРІвЂћСћs profile invocation uses a bounded `--profile-steps`, it must keep the full 4-epoch recipe and 25,760-step scheduler horizon and emit an engineering receipt; that invocation is a profile, not a teacher fit.

## Machineable pre-profile acceptance

`GRU_NATIVE_PREFLIGHT_R1.json` is the companion gate contract. A controller should reject before GPU use unless every recipe, membership, precision, optimizer-group, endpoint, provenance, and resource-floor predicate passes. In particular, the first optimizer update must execute the patched strict Muon path; inspecting only `args.precision` or the top-level optimizer class is insufficient.

Current reviewer status: source-level route is ready with the approved adaptation. No remote data receipt, final isolated S_GRU source manifest/hash, L4 profile, GPU fit, or wall-clock performance has been reviewed here; those receipts remain execution admission gates. Start/profile GRU shapes on their assigned available L4s as soon as these gates pass; Transformer readiness is independent.