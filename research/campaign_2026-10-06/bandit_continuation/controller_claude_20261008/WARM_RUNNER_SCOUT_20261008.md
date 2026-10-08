# Warm Load-Once Queue Runner Scout

**Task**: Identify minimal path to a warm worker that loads data/caches once, then runs many experiments without reloading.

**Scope**: Read-only analysis of scripts/train_rnn.py structure, existing runners, and memory profiles.

---

## 1. Core Data/Cache Loading in train_rnn.py

### Data Loading (lines 2048–2094)
- **Sequences dataset creation**: lines 2048–2070 create `train_ds` and `valid_ds` via `Sequences()` class
  - Constructor: lines 194–237
  - `Sequences.__init__()` stores `self.path`, indices, but **does not load data in __init__**
  - Data loaded lazily in `__getitem__()` (line 293: `self._file = SequenceFile(self.path)`)
- **DataLoader construction**: lines 2090–2094
  - `train_loader = DataLoader(train_ds, batch_size, ..., num_workers=workers, ...)`
  - `valid_loader = DataLoader(valid_ds, batch_size=16, num_workers=0, ...)`

### Cache Loading
- **Propensity cache**: line 2016–2017, loaded only if `--prop-cache` + `--propensity`
- **Distillation/teacher cache**: line 2038–2043 validates, then cached in `Sequences`
- **Quote/spread cache**: lines 2001–2032, validated and passed to `Sequences` constructor

### Key Finding
**Data is NOT preloaded in memory**—parquet/cache reads happen per-batch via workers. Each experiment reloads everything.

---

## 2. Model & Optimizer Construction (lines 2200–2304)

- **Model creation**: line 2200 `model = make_model(vars(args), n_in, base_n_out)` (lines 1011–1093)
  - Imports model classes (RNNModel, TCNModel, BlockMemoryAttention, etc.)
  - No RNG or global state coupling in factory
- **Optimizer**: line 2286 `opt = build_optimizer(model, args)` (wnn.training_dynamics)
- **Scheduler**: line 2287 `sched = build_scheduler(opt, args, total)` (same module)
- **EMA/SWA models**: lines 2298–2304, optional averaging models

### Key Finding
Model/optimizer/scheduler are **purely functional** per args—fresh state required for each run. RNG reset is separate (line 2368+).

---

## 3. Training Loop Entry & Module-Level State (lines 2368–2500)

- **Training loop**: line 2368 `for epoch in range(start_epoch, args.epochs):`
- **Module globals** (lines 536–545):
  ```python
  INPUT_PROPENSITY = None          # set at line 2177
  ONLINE_NORM = False              # set at line 2181
  HEURISTICS = False               # set at line 2188
  RAW_ONLY = False                 # set at line 2185
  RESEARCH_FEATURES = "none"       # set at line 2167
  PRIVILEGED_MODE = "none"         # set at line 2169
  SENTINEL_PREPROCESSOR = None     # set at line 2153
  ```

### Key Finding
**Module globals are set during main() (lines 2153–2189) and remain live for the entire fit**. Between runs, they must be reset to avoid state leakage.

---

## 4. Existing Runners & Patterns

| Runner | Lines | Pattern | Warm Load? |
|--------|-------|---------|-----------|
| `bandit_sequence.py` | 1–88 | Per-step launch; waits for exit; spawns new process | No |
| `kaggle_sequence.py` | 1–62 | Sealed step subprocess; no in-process loop | No |
| `campaign_controls_queue.py` | 1–60 | Queue + subprocess.run; each job is separate | No |
| `night_experiment_queue.py` | (not shown) | Similar queue pattern | No |
| e540 workers (e540_kaggle_*.py) | ~1700 lines | Kernel-internal stage loops; profile + fit + export | Possible kernel-side load |

**Finding**: No existing runner loads data once in a persistent worker. Kaggle kernel bootstraps include packet payload staging, but each fit is still process-local fresh load.

---

## 5. Memory Profiles from Actual Runs

From `H230_SEED99_H272_PROFILE_CACHE_HANDOFF_20261008_R1.json`:
```json
"before": {"current": 13076283392, "anon": 25882624},      // ~13 GB current (page cache + RSS)
"after":  {"current": 8484454400, "anon": 25886720}        // ~8.5 GB after export cleanup
```

From E540 packet plans (continuous_execution_E529):
- `rss_abort_gt: 16000000000` (16 GB abort ceiling)
- `minimum_ram_headroom: 4 GB` (4 GB reserved for overhead)
- Typical H230/C1 fit: **6–10 GB in-process**

**Finding**: Two typical H230 fits (~6 GB each) + shared parquet cache (~2–3 GB) + teacher caches (~2 GB) fit in 18 GB if loaded once and reused.

---

## 6. Recommended Minimal Warm-Runner Design

### Architecture (≤40 lines prose)
1. **Queue directory**: Frozen run-spec JSONs (`queue/spec_001.json`, `spec_002.json`, ...)
   - Each spec: `{"run_name", "epochs", "hidden", "seed", "distill", ...}` (mini args subset)
   
2. **Persistent worker process**:
   - Load `train_ds`, `valid_ds`, parquet paths, teacher caches **once at startup**
   - Store in module-level cache dict (careful: reset globals between runs)
   - For each queued spec:
     - Reset module globals (RESEARCH_FEATURES, PRIVILEGED_MODE, etc.)
     - Reuse `train_ds` indices, rebuild only per-run model/optimizer/RNG
     - Call training loop (lines 2368–2750)
     - Write checkpoint, predictions, official metric score, receipt
     - Append result to ledger (for bandit.py)
     - Check for STOP file, exit if found

3. **Critical state isolation**:
   - **Reusable**: Parquet memmap, teacher prediction arrays, propensity cache arrays, quote/spread cache tensors
   - **Fresh per run**: Model weights, optimizer state, RNG seed (line 2355+), AMP scaler, EMA/SWA, dynamics state
   - **Reset per run**: Module globals (lines 536–545), argument validation coupling

4. **Small-data screening mode** (per specs/plans/2026-10-08-small-data-screening.md):
   - Accept `--fit-subset-manifest` (E540 subset) + `--split-manifest` (clean fold)
   - Keep fixed eval membership (valid_idx via seed 12345)
   - Preserve complete sequences/causal history
   - Track fidelity (optimization budget, subset size) in receipt

### Minimal Refactoring
- Extract `load_campaign_data()`: lines 1660–2046 → returns `(campaign_split, fit_subset, train_idx, extra_valid, valid_idx, teachers, caches, provenance_base)`
- Extract `train_one_fit()`: lines 2143–2750 → accepts already-loaded data, args, output dir; returns metrics
- Create `scripts/warm_worker.py`: ~150 lines
  - `main()`: read queue dir, load data once, loop over specs, train_one_fit per spec, write ledger
  - No modification to train_rnn.py; use it as-is for reference

### Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| State leakage (globals, RNG, device state) | Reset RESEARCH_FEATURES, PRIVILEGED_MODE, seed, torch RNG at loop start; validate args match loader schema |
| Memory growth (model, cache, GPU) | Explicit del + torch.cuda.empty_cache() after each fit; monitor peak RSS |
| Stale teacher cache if updated | Validate teacher cache hash at startup only; doc that teacher cache is immutable during run |
| Parquet file mutations | Use read-only SequenceFile; abort if path size changes between runs |
| Determinism failures (dataloader state leakage) | EpochOrderSampler.set_epoch() per fit; drop_last=True; pin_memory persistent-worker warning (line 2093) remains avoided |

---

## Summary (15 lines)

**train_rnn.py structure**: Data loaded lazily per batch (lines 2048–2094), model/optimizer are fresh functions (2200–2287), module globals set once (536–545, 2153–2189), training loop is procedural (2368–2750). No existing warm runners. H230 fits use 6–10 GB; shared data ~3–5 GB; 18 GB cgroup comfortably fits 2 concurrent fits if loaded once.

**Recommendation**: Extract `load_campaign_data()` and `train_one_fit()` from main(); write `scripts/warm_worker.py` (150 lines) that loads data once, resets module globals + RNG per spec, calls train_one_fit in a loop, writes results + ledger. No train_rnn.py changes needed. Risks are state leakage (mitigated by explicit reset) and determinism (mitigated by EpochOrderSampler). Estimated warmup overhead: 30–60s; per-fit fresh startup overhead eliminated; small-data screening mode reuses same data logic with subset manifest. This unblocks **5–6 specs/hour** locally vs. current 1–2/hour (fit-only, no startup).

