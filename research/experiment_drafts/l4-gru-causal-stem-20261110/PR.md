# [Experiment] L4 GRU causal input stem: kernel 16 versus kernel 1

1. **Status, lineage and owner.** Authorized six-fit experiment design; six JSON configs are prepared. Implementation is committed at `336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2`; source capsule SHA is `b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188` and manifest SHA is `482d13529e8c65386e6f74e7e81f607eeb3b0ed0795beff44902bf703e7d342e`. Eighteen focused/existing CPU model tests, Ruff and the training structure checker passed. Independent review, native profiles and fit approval are pending ROOT. ROOT owns source pinning, the existing native RPC, profile/fit admission, process signals, and collection. The idea descends conceptually from draft PR74, *Causal input filters for Transformer quality on B300* (cached implementation commit `8a8afec964985a6ba8b13d1c9af1ab00e14c6504`), and ledger E-168, which PR74 cites as a Conv16-before-GRU trial with one exposed seed/fold and chunk-local history reset. The FIR module adapts PR74 code to the GRU; neither ancestor supplies independent quality evidence for this screen. No PR number or new H/E ledger ID is assigned by this packet.

2. **Why it should work.** A per-channel causal FIR residual may expose local row dynamics before the GRU while preserving the standardized raw input through an identity path. K16 sees the current row and preceding15 rows; K1 tests an instantaneous per-channel transform; K0 is the no-stem control. The strongest alternative is that any score difference comes from added parameters, optimizer interactions or seed noise rather than lag information. A correct K16 implementation must carry history across chunks and reset only at whole-sequence boundaries. PR74's unexecuted B300 Transformer design motivates the mechanism; E-168's chunk resets make it an imperfect ancestor. K1/K16 are not parameter-count matched.

3. **Compute advantage.** Exactly six fresh one-GPU L4 fits provide a paired two-seed comparison at six fixed epochs. The same shared-control recipe is retained across arms; no previous checkpoint is reused. Profiles are engineering gates and do not count as fits. The authorized clock is four hours from the first actual profile, with at most three hours for fitting and one hour reserved for full-valid evaluation and collection. The fit-time estimate is unknown until actual source-bound profiles. GPUs6/7 remain with existing SSL work and are outside this experiment.

4. **Frozen comparison.** The six configs clone `l4-next70-20261110/configs/shared_control_s203201.json`; only `seed` and `model.input_stem_kernel` differ. Plain uses kernel0; K1 uses1; K16 uses16. Each config retains the two-output GRU256x2, raw112 input, six epochs, 38,760 updates, batch32, chunk1000, accumulation1, all-fit normalization over10,325 official train sequences, all-after-warmup mask, standalone two-target weighted-MSE objective, FP32, one L4 rank, AdamW3e-4/weight decay0.01, clip1, warmup fraction0.05, warmup_cosine_v2/minimum-LR ratio0.1, diagnostics every20 updates, profile reload, every-epoch checkpoints and configured fixed epoch6 endpoint. KD/hint weights remain0. No other hyperparameter is changed. The new stem must be residual identity initialized; same-seed common GRU initialization must be bitwise equal across K0/K1/K16, with independent fresh paired seeds. The source loader must confirm these fields before any profile.

5. **Bounded fitting panel.** Six new scientific fits; zero reused fits, zero teacher fits and zero student fits. Each cell receives one L4 GPU and a ROOT-approved exact 30-update source/config/data/device profile before fit.

   | Seed | Plain K0 | K1 | K16 |
   |---|---|---|---|
   | 203301 | `plain_s203301` | `k1_s203301` | `k16_s203301` |
   | 203302 | `plain_s203302` | `k1_s203302` | `k16_s203302` |

   All cells are fixed6E/38,760 updates. At most six profiles and six concurrent one-GPU fits are authorized; no automatic retry, resume or added follow-up allocation. Resource caps: 2.5 GB total output, at least4 GB free disk, 128 GiB aggregate owned RSS, and two local CPU cores for local preparation. ROOT must record the native worker CPU/thread cap before launch.

6. **Validation and uncertainty.** The primary endpoint is the fixed epoch6 official `global_wp_clip2_v1` on all1,873 chronological validation sequences and3,528,273 scored rows. Primary contrast is paired K16−K1 for each seed and their descriptive mean; K16−K0 and K1−K0 are secondary. Report both target components and all six cell endpoints. Validation runs every supervised epoch as declared by the authority; epoch6 remains fixed, with no validation-based best-epoch selection or learned mixture. This is a two-seed screen, not a reliable training-variance estimate, significance test, cross-fold result or public-transfer claim. Full official validation is repeatedly exposed development evidence.

7. **Leakage and coverage.** Fit and normalization membership is official train only:10,325 whole sequences, zero valid ancestry. Evaluation uses the later official valid partition:1,873 sequences/3,528,273 scored rows. Sequence identifiers are shuffled and must not be treated as chronological order. The expected official split identity is `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`; ROOT must verify the prepared native project receipt before profiles. The input stem may read only the current and past normalized input rows, pad only at sequence start, carry the last15 rows across chunk boundaries for K16, and reset at each sequence boundary. No future values, teacher, reference cache, initializer checkpoint or valid-derived normalization is permitted.

8. **Decision-changing result.** The useful comparison is whether K16 improves over K1 on the fixed full-valid endpoint across the two paired seeds, with K0 providing a baseline check. The authority sets no numeric minimum effect threshold; this packet does not invent one. Report mixed and negative signs unchanged. Any follow-up, release or allocation requires a separate review of full endpoints, implementation gates, cost and the repeated-validation exposure. No score is promoted automatically.

9. **How to run.** ROOT must first pin the immutable L4 source capsule, loadable config contract, official data/split receipt, Python/environment and physical GPU UUIDs. Then run actual per-cell profiles through the existing canonical sequence trainer; do not use the PR74 B300 runner. The command template is:

   ```bash
   ROOTN=/data/lc-epanov_s/tabdata-stem-20261110-r1
   PYTHON=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
   SOURCE="$ROOTN/source-r1"
   PROJECT="$ROOTN/project"
   CELL=k16_s203301
   export PYTHONPATH="$SOURCE/src"
   "$PYTHON" "$SOURCE/scripts/train_sequence.py" \
     --config "$ROOTN/inputs-r1/configs/$CELL.json" \
     --data-root "$PROJECT" \
     --split-manifest "$PROJECT/chronological_primary.json" \
     --output "$ROOTN/runs/$CELL/profile" \
     --profile-updates 30
   ```

   Each source-bound actual profile must pass finite-state, exact reload, RNG, carry/reset, device, and data checks; ROOT writes a separate per-cell fit approval. Only then run the fixed six-epoch fit with the same argv, a fresh cell output directory and `--profile-receipt "$ROOTN/control/approved/$CELL.json"` instead of `--profile-updates 30`. ROOT supplies the actual pinned Python/source/project/GPU values and freezes source/config hashes before profile. Do not start if the exact trainer does not accept the new stem field or the total fitting budget no longer fits the three-hour window. ROOT owns all SSH/GPU operations; this packet created no remote worker. The separately authorized PR76 CPU callback commands and exact hash identities are documented in [README.md](README.md#separate-pr76-cpu-callback-qualification-lead); they do not train this panel.

10. **Evidence to return.** For all six cells retain raw and loader-normalized config hashes, clean source commit/capsule hash, split/data membership receipt, actual L4 profile and reload/RNG receipt, approved GPU UUID, exact argv/environment, process birth identity, update counter and per-epoch train/full-valid curves, LR, preclip norm/clip/nonfinite/update diagnostics, elapsed time, peak VRAM/RSS/disk, every epoch checkpoint hash and fixed-E6 model hash, predictions/grade receipts with exact full-valid coverage, and evidence/weights archive hashes. Source support, fit feasibility, root paths and source/environment hashes remain pending ROOT.

11. **Results and trace.** No profile or scientific fit has run for this L4 panel in this packet; no measured quality or public gain exists. The raw configuration SHA-256 values are listed in the packet README. PR74 and E-168 are cited as conceptual ancestry with their design limitations; their outcomes are not transferred to this result. PR76 CPU qualification is separately `PACKAGED_NOT_EXECUTED` in cached metadata; no ZIP bytes were found/rehashed and no CPU inference ran here. ROOT must add exact source/profile/fit receipts, new ledger links and independent critique after source freeze; PR number, merge SHA and results are not assigned.