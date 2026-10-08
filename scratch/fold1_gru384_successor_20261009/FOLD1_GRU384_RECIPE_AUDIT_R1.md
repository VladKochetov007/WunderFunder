# Fold1 raw GRU384 successor audit — 2026-10-09

## Verdict

The two planned fold1 baseline fits are ready for root admission after one short actual-device profile. Their frozen arguments are an exact counterpart to the successful fold0 seed3/seed4 runs after changing only run name, fold number, split path, and split digest. The fit-membership and source-ancestry audit passes. This is a robustness confirmation on a historically exposed fold, not fresh-holdout evidence.

The local sequence is finite: profile seed3, fit `bd26_w384raw_s3_f1`, then fit `bd26_w384raw_s4_f1`. Do not substitute fold0 weights. Exports and cache construction are a later phase, currently disk-constrained.

## Pinned inputs and recipe

- Main worktree HEAD observed: `952af5b6546f626e6c12e47b546a7675a229d160`.
- Fold1 dependency inventory: `research/campaign_2026-10-06/bandit_continuation/controller_claude_20261008/sol_continuation/FOLD1_DEPENDENCY_INVENTORY_R1.json`, SHA-256 `ff844bab24cacd6345019469ce77af544bc47b24863faf6beedec2cbd5bb08a1`.
- Split: `research/campaign_2026-09-29/artifacts/splits/fold1.json`, SHA-256 `31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae`.
- Membership: 10,533 fit-train groups, 1,416 fit-valid groups, 457 evaluation-valid groups across 149 components; 73 train groups are excluded. Fold1 is historically inspected.
- Current trainer and split source hashes: `scripts/train_rnn.py` `4de1f29b18aa7f9371010816cac04d986b0fcf881415ca562a89c22f1afbd207`; `src/wnn/splits.py` `b62797867647cc2244349700bdcf3b2e5c44fd2317b80b39a340edbbed77fec4`.
- Fold0 counterpart provenance hashes: seed3 `3301c3033587829ac5166ae6fee9eb911ee29c8a6412e15010dd3e9fc4423d6d`; seed4 `08b25696b3697f7935b2a69ec23b6b4cf4d340bf615ab9fe2228043e50b024a9`.

Each baseline is fresh initialization, raw-only GRU, 384 hidden units, 2 layers, batch 16, chunk 2,000, WMSE/chunk normalization, AdamW plus OneCycle, FP16, four epochs, fixed checkpoint epoch 3, and no evaluation during fitting. The complete argument objects are pinned in `FOLD1_DEPENDENCY_INVENTORY_R1.json`; they match the fold0 provenance after normalizing only `name`, `valid_fold`, `split_manifest`, and `split_manifest_sha256`. Both use `train_source=train+valid`, `include_heldout_valid=false`, `valid_fold=1`, and an empty initializer. The requested `train_seqs=10607` exceeds the 10,533 eligible fold1 train groups, so the trainer uses all 10,533 eligible groups. It also fits on exactly the 1,416 eligible valid groups. The 457 evaluation groups are excluded from both fitting pools.

The inspected current trainer path makes the exclusion operational: the campaign split sets the evaluation fold to the manifest's held-out groups, uses the manifest fit groups as the train/extra-valid pools, calls `CampaignSplit.assert_fit_groups`, and the frozen command sets `--no-eval-during-fit`. `full_valid=false` and `valid_seqs=600` do not expose evaluation rows to training or checkpoint selection.

## Ancestry and source findings

An independent bounded check loaded the actual fold1 split through `load_campaign_split`, checked all nine existing teacher/propensity provenance objects against its exact split digest and fit memberships, called the component-exclusion assertion for each, and verified the actual bytes of all nine selected source checkpoints against their sidecar maps. It passed. The checked runs are the two strong teachers, two raw teachers, two fullFT teachers, their two same-seed HFT initializers, and `c20260929_prop_f1`.

The clean ancestry is explicit: raw seed3/seed4 use `c20260929_prop_f1`; strong seed3/seed4 distill the two fold1 raw teachers and use the same propensity run; each fullFT teacher initializes from its same-seed fold1 HFT, and each HFT initializes from that same-seed raw fold1 teacher at epoch 3. The propensity job itself uses `heldout_fold=1`, the fold1 split digest, fixed epoch 3, and `--no-eval-during-fit`. No evaluated fit component appears in these sources.

The old fit receipts bind exact source hashes. Historical fold0 `train_rnn.py` SHA-256 was `0e622a87888769ff7467fadd1d1a230886008d1803099e8881e100939082467d`; current is `4de1f29b18aa7f9371010816cac04d986b0fcf881415ca562a89c22f1afbd207`. `src/wnn/nn.py` changed from `9293f8f4e2a2f3bd9f0157f8f83f12a6a758608dd03c7e141c290d82d4d5aeca` to `874fbf1ab5756c7dabe9e388d9d619c0a29c218d28db36a7f8747957df453181`. The reviewed diff adds optional E540 numeric-map and block-memory paths, fit-subset support, and profile evidence. The plain `numeric_map=none` model retains the same input width and parameter path; the optional branches are inactive for this recipe. `src/wnn/optim.py` and `src/wnn/run_identity.py` are byte-identical to the historical fit receipts. The only changed optimizer setup is an explicit GradScaler initial scale of 65,536, equal to the prior library default. Thus the recipe-level counterpart is supported, but it is not a byte-identical historical source build; the new profile and fit receipts must bind current code hashes.

The measured fold0 seed3/seed4 attempts each completed in about 5,163–5,179 seconds including data/setup overhead, with a 6.50 GB scoped RAM peak and 3.994 GB observed VRAM peak. These are planning references, not fold1 runtime promises. Historical per-epoch training time was about 1,364 seconds for seed3 and 1,386 seconds for seed4. Both fits must remain sequential on the single local GPU.

## Profile to admit

The exact profile command vector and acceptance contract are in `FOLD1_GRU384_PROFILE_PLAN_R1.json`. It uses seed3 and the fold1 membership, batch 16, chunk 2,000, two warmup batches and eight measured profile batches. The trainer sets the limit to ten batches, emits finite update/timing evidence, and exits before checkpoint creation. It makes no score or fit-quality claim. Request at most 200% CPU, 9,000,000,000 bytes cgroup memory maximum, zero swap, 5,000,000,000 bytes observed-VRAM ceiling, and 600 seconds. Root must admit and launch it under the exclusive GPU lane, capture cgroup peak/events and total observed VRAM, and confirm the unique profile output path is absent before starting.

The older E521 batch-16 profile used the same architecture/shape and reported a roughly 1.11-second step, but its memory ceiling was extrapolated from a batch-8 profile. It does not replace this requested fold1 actual-device profile.

## Finite delivery queue and disk gate

1. Run the admitted short profile only. Continue only on successful finite update evidence, no OOM events, cgroup peak below 9 GB, and observed total VRAM at or below 5 GB.
2. Fit seed3 fresh at `bd26_w384raw_s3_f1`, fixed epoch 3, with the frozen inventory arguments. Verify terminal status, checkpoint hash, provenance, exact split/membership, and all optimizer/update diagnostics.
3. Fit seed4 independently at `bd26_w384raw_s4_f1` with its frozen seed4 arguments and the same validation gates.
4. Then produce the missing teacher predictions and build the new fold1 cache; never relabel a fold0 cache. Reuse existing arrays only after actual producer/checkpoint/row-order checks.

The inventory shows eight absent prediction arrays for this later cache stage: train exports for strong s3/s4, fullFT s3/s4, and baseline s3/s4, plus valid exports for baseline s3/s4. The existing raw teacher arrays are 1,696,960,128 bytes per train array and 299,680,128 bytes per valid array. Using those sizes as a first-order estimate, the eight missing arrays may require about 10.8 GB before the combined cache or logs. Root measured 14.44 GB free and requires a 10 GB reserve; defer these exports/cache until a sufficient local disk or approved remote output route is available. This downstream storage constraint does not prevent the low-footprint profile and checkpoint-only fits.

No fit, export, or cache build was launched by this audit. The root admission remains the authority for the profile and both fits.
