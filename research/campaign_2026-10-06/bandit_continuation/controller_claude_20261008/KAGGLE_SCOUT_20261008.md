# Kaggle GPU/TPU Execution Scout — 2026-10-08

## Campaign kernel inventory: every push (poll log + kernels list)

| Accel | Kernel | Status | Opt.Updates | Root Cause / Notes |
|:---:|:---|:---|:---:|---|
| GPU | e540-v12-cache-dirfix-20261008-runner | **COMPLETE** | ✓ linear+PLE | Export-only: mounted existing checkpoints, predicted both arms, 468×20k×2 finite FP32 |
| GPU | h232-gru384-b8-amp4096-conditional-profile-fit-v4 | **COMPLETE** | ✓ profile+fit | GRU384 B8, AMP4096 GradScaler, profile gated full fit, 230 profile + 14.9k fit updates |
| GPU | e520-profile-v12-explicit-offline-metadata | **COMPLETE** | ✓ profile | Offline manifest profile, no fit attempted |
| TPU | wnn-e463-combo-reduction-tpu-v5e8-20261007-v5 | **COMPLETE** | ✓ | Proof-only, synthetic gradient full-shape pass, no real-data fit/save |
| TPU | wnn-e463-real-tpu-fullshape-preflight-20261007-v3 | **COMPLETE** | ✗ | Preflight/probe only, device validation no fit |
| GPU | wnn-gru-device-preflight-20261006-v3 | **COMPLETE** | ✗ | Device preflight/sanity gate, no fit |
| TPU | e463-tpu-real-data-reload-v6-20261008 | **QUEUED** | — | Since 2026-10-08 08:07Z, awaiting provider startup, real-data reload path unobserved |
| GPU | h230-seed9-parent-v7-runner | ERROR | ✗ | Containment failure before profile; diagnosed in live fit, seed9 epoch1 local active |
| GPU | e529-gru512-768-muon-fit-v5r1-20261008 | ERROR | ✗ | GRU512/768 Muon, seeded epoch sampler, failed manifest/hash consumer |
| GPU | gru-capacity-v4-manifest-profile-20261008 | ERROR | ✗ | GRU512/768 capacity, KeyError `bytes` in input receipt consumer |
| GPU | h232-gru384-b8-amp4096-v7b-shortslug | ERROR | ✗ | Short-slug bootstrap path, profile completed (230 upd) but fit failed on reload |
| GPU | e540-v5..v11 (7 kernels) | ERROR | ✗ | V5: argv parser failure; V6: manifest collision; V8: hash consumer path; V9–V11: export-only failures |
| GPU | h230-seed9-parent-fit-v4-runner | **PROVIDER_ACCEPTED** | — | Awaiting provider worker startup; v6/v7 Kaggle accepted but no fitting phase observed |
| GPU | wnn-raw-teacher-pair-e508-20261007-v1..v5 (5 kernels) | ERROR | ✗ | Raw teacher pair E508, profile-only time-cap hit (8524s fit>6300s cap), no fit / OOM |
| TPU | wnn-e463-real-tpu-fullshape-preflight-20261006-v1..v2 | ERROR | ✗ | V1 queued then failed; V2 import/mount failure; V3 repaired, proved device validation |

Total: **7 COMPLETE**, **1 QUEUED**, **45 ERROR**, **2 PROVIDER_RUNNING**. GPU quota: ~39,759s remaining (refresh 2026-10-10). TPU quota: ~71,621s remaining.

---

## Successful H232 V4 pattern: how it worked

**Kernel:** `vladkochetov07/h232-gru384-b8-amp4096-conditional-profile-fit-v4`  
**Source:** `/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v4/packet`

**Data mount:**
- Only `datasets/valid.parquet` (5.6 GB) from `vladkochetov07/wunder-funder` mounted read-only
- No parquet footer parse; no target NPZ mount
- Train data staged via distill cache on the machine, verified by exact SHA-256

**Code shipping:**
- 54-file source archive, SHA-256 `afde296e1c5e2f0c47a6fb352c2dea8e0a3df75900f53c26f103b3921df1f565`
- Fixed `scripts/train_rnn.py` with Path-to-string JSON serialization repair
- All files rechecked byte-for-byte; embedded ZIP digest `e5db8d46…`

**Execution phases:**
1. **Profile gate:** 3 warmup + 20 timed outer batches (30 warmup, 200 optimizer updates)
2. **Profile-to-fit gating:** validates AMP health (zero skipped updates), stable T4 identity, memory/disk headroom, exact data stat tuples, shape probes, projected lifetime
3. **Fit conditional:** only runs if gates pass; profile-trained weights **not** carried into fit
4. **Fit:** full training, checkpoint epoch-3 hash verification, fold-0 membership binding

**Timing:** profile gate ~544s (E520 mount + data verify); fit projected ~275–307 min for GRU384. Worker: 9000s max, kernel: 10,000s max.

**Startup vs fit split:** wall-time heartbeat logs record mount, profile, checkpoint, export separately — not combined.

---

## E540 V12 warm-cohort blueprint

**Kernel:** `vladkochetov07/e540-v12-cache-dirfix-20261008-runner`  
**Purpose:** dual-arm export-only run (no fit), reusing pre-fitted linear+PLE epoch-0 checkpoints

**Data mount (existing):**
- Read-only `datasets/valid.parquet` from `vladkochetov07/wunder-funder`
- Linear checkpoint (`e540v8-linear-s540-fp32`) + PLE checkpoint (`e540v9-ple-s540-fp32`)
- Sidecar provenance/history JSON files included

**Cheap verification:**
- File count + small-file SHA-256 (not full 37 GB rehash)
- Checkpoint binary SHA-256: linear `17c91be7…`, PLE `397b1666…`
- Validation: 468 row-groups, 20k sequence length, shape [468, 20k, 2]

**Execution:**
- Sequential prediction export for both arms on single worker
- Output: two FP32 NPY caches (74.8 MB each) + sidecar JSON
- Per-arm history SHA, model SHA, provenance SHA all preserved

**Wall time:** profile-only on two T4s, **~25 min per arm**, total per kernel **~50 min**.

---

## Private Kaggle dataset inventory

| Slug | Version | Size | Purpose | Mounted by |
|---|---|---|---|---|
| `vladkochetov07/wunder-funder` | current | 33.7 GB | Raw train.parquet (29.2GB) + valid.parquet (5.6GB) | All training kernels |
| `vladkochetov07/wnn-c20261004-fold0-assets` | v1 | 2.83 GB | Distill cache + checkpoint fold-0 set | H230/H232 cache kernels |
| `vladkochetov07/wnn-c20261004-fold1-assets` | v1 | 2.81 GB | Distill cache + checkpoint fold-1 set | H230/H232 cache kernels |
| `vladkochetov07/wnn-jax-realdata-e463-20261004-v3` | v3 | 9.4 MB | JAX E463 real-data conversion (TPU) | TPU kernels |
| `vladkochetov07/wnn-e419-qat-dualt4-20261004-v3` | v3 | 4.2 MB | Quantization-aware training fixtures | Numerical embedding arms |

**Key fact:** fold assets already exist versioned; mounting them costs 1–2 min SHA verification vs. full data staging. No re-upload needed for new kernels using same folds.

---

## Minimal warm-cohort GPU kernel design

**Architecture:** single kernel, two T4 GPUs, sequential N-run queue

**Mount & verify (one-time per kernel):**
```
1. Mount vladkochetov07/wunder-funder (read-only, 15 min on cold start)
2. SHA-256 check: valid.parquet (5.6 GB)
3. Check fold-asset checkpoints exist (2–3 GB each)
4. Verify data digest against pinned spec (size + file-list SHA)
```

**Queue design:**
- Use `scripts/warm_queue_worker.py` (existing stdlib queue runner)
- Queue dir: `pending/` with JSON specs, each spec = list of argv steps
- Per-spec: complete `train_rnn.py` + export args, exact seed/checkpoint bindings
- **Per-kernel N runs:** 25 min per fit × N fits, **6–12 screens per 12-hour GPU window**

**Dual-T4 parallelism:**
- Set `CUDA_VISIBLE_DEVICES=0` for run 1, `CUDA_VISIBLE_DEVICES=1` for run 2
- Stagger starts by 30s to avoid simultaneous OOM startup
- Checkpoint dir `/kaggle/working/ckpt_N` per spec to avoid collision
- Predictions written to `/kaggle/working/pred_N.npy` per arm

**Per-kernel max:**
- Wall time: 12 hours (Kaggle GPU limit)
- Worker reserve: 5 min finalization
- Fit time per model: 25–30 min (e.g., GRU384 or smaller)
- **Capacity:** 20–24 models per kernel if all are ≤30 min

**What gets shipped:**
- Source tarball (54 files, <1 MB zipped) under the 1 MB bootstrap conservative ceiling
- Larger data/checkpoint payloads via private asset datasets
- Queue JSON: one spec per run, 10–20 KB each

**Export collection:**
- Kernel outputs `/kaggle/working/pred_*.npy` → downloaded via Kaggle API
- Checkpoint hashes from provenance JSON → matched offline
- Logs streamed to `/kaggle/working/train_log.jsonl`

---

## TPU status: E463 V5 vs V6

**E463 V5E8 (COMPLETE):**  
- **What it proved:** JAX synthetic gradient shape parity (full 468×20k×2 GRU state), no OOM, no softmax underflow
- **Did NOT do:** no real-data fit, no weight save/reload, no actual prediction export
- **Timeline:** device startup ~600s, synthetic gradient 30–45s, terminal status download ~5 min

**E463 V6 (QUEUED since 08:07Z 2026-10-08):**  
- **Target:** real-data reload path — actual train.parquet load, few batches, state save/reload, proof-of-life  
- **Blocker:** awaiting provider worker startup; QUEUED status alone does not prove fitting started
- **Expected if starts:** 4–5 hour estimated wall time (data load 30–60s, profile 10 min, fit 2–3 epochs, export 20 min)

**Realism in next 24h:**
- **V6 startup proof needed:** provider must show worker phase heartbeat (not QUEUED state alone)
- **If V6 starts + passes:** GRU training on TPU is **realistic but unproven** for full fit
- **Next step if confident:** one bounded proof (10 min fit) on V6, not a full-size multi-epoch run
- **If V6 remains QUEUED:** substitute with GPU-only research; do not wait on TPU

**Recommendation:** Check V6 provider status at 2026-10-08 20:00 UTC. If still QUEUED with no startup logs, switch to GPU lanes only.

---

## Top 5 failure classes & prevention checklist

| Class | Example Kernels | Prevention |
|---|---|---|
| **Deep-path ENAMETOOLONG** | E520 v1–v11 | Bootstrap + kernel metadata bundle separately; use short alphanumeric refs for dataset/kernel path encoding; test upload cache filename <255 chars |
| **Argv parser + import path** | e540-v5, E540-v8, H232-v7b-export | Verify script bootstrap with `python -c 'import sys; sys.path; exec(...)` offline; test all relative imports in isolated env; embed absolute PYTHONPATH in kernel wrapper |
| **Manifest/hash consumer mismatch** | gru-capacity-v4, E540-v8 | Pre-flight checks: run exact kernel code on local data; validate consumer fixtures against producer binary format (NPY headers, optimizer pickle, exact field names) |
| **Manifest reuse across folds** | E540-v6 | Mount fold-specific datasets separately; never symlink across fold boundaries; verify fold ID in every checkpoint sidecar before mounting |
| **Startup before profile** | h230-seed9-v6/v7 | Gate check: verify provider worker heartbeat (CPU memory + VRAM > 0) before declaring "provider accepted"; QUEUED/ACCEPTED is not proof of startup |

**Preflight checklist before every push:**
- [ ] Bootstrap <1 MB; test with `python bootstrap.py --help` locally
- [ ] Dataset mount: `ls /kaggle/input/<slug>/` dry-run in old completed kernel (reuse mount validation)
- [ ] Argv parser: run exact Kaggle argv on local data, capture any import/Path errors
- [ ] Binary fixtures: if reading NPY/pickle, test reader on actual mounted files with exact kernel environment
- [ ] Fold binding: if multi-fold, verify fold ID in every filename; never share manifest across folds
- [ ] Small-file SHA: pre-compute hash list for assets, test consumer reads them correctly

---

## Summary & recommendation

**Warm-cohort kernel design (ready to build):**

Mount `vladkochetov07/wunder-funder` once, verify by SHA-256 of validation set and metadata. Use `scripts/warm_queue_worker.py` to run N sequential `train_rnn.py` calls per T4, writing predictions to `/kaggle/working/`. Kernel wall-time limit: 12 hours. Estimated throughput: 6–12 trained-and-exported screens per kernel, 25–30 min per model.

**Blockers to resolve before first push:**
1. **Fold-specific checkpoint mount path:** confirm which fold assets are needed per run; verify fold ID in sidecar before using checkpoint
2. **Argv validation:** test exact `train_rnn.py` command on local machine with Kaggle bootstrap wrapper; confirm no import-path or serialization errors  
3. **Queue schema:** finalize JSON spec format for queue runner; document expected fields (seed, optimizer, checkpoint path, output dir)

**Recommended first kernel:**
- **Name:** `wnn-warm-cohort-dual-t4-v1`
- **Payload:** source tarball + fold-0 asset slug + queue JSON for 4 baseline GRU384 B8 runs (seed 1,2,3,4)
- **Execution:** profile + fit each, export predictions, 100–120 min total wall
- **Success criteria:** all 4 checkpoint hashes match offline; predictions finite 468×20k×2; export SHA logged

**Expected quota:** ~6000 GPU-seconds per kernel (profile 600s + 4×fit 1400s + 4×export 300s). Current headroom: 39,759s → **6–7 kernels before refresh 2026-10-10**.
