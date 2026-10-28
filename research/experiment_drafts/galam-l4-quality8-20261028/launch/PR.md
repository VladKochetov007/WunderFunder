# Fixed eight-L4 quality panel with full chronological validation

1. **Status, lineage and owner.** Exactly eight fresh full fits are running on
eight L4s under launch `e635250b62fa489d94c987d4898ff417`. The approved panel is in
`launch/PANEL.json`; complete sources and all eight actual-source profiles passed
independent resource/identity review. Root owns delivery through results. GALAM
is reconstructed from Denis Tomin's screenshots with explicit unknown original
dimensions; no exact reproduction claim is made.
2. **Why it should work.** Test capacity, recurrent depth/type, attention history,
gradient horizon and matrix optimization against the full schematic GALAM
LSTM64 control. Plain GRU256x2/event211 is a historical whole-system benchmark,
not a pure recurrent-cell ablation. Each GALAM arm changes only its declared
treatment; dimensions needed by a wider/GRU model are part of that treatment.
3. **Compute advantage.** Eight independent L4s give one lane per fixed arm.
Strict-FP32 synthetic sweeps and eight successful full-input real profiles govern
the frozen GRU32/GALAM384 batches, one-batch prefetch and eval384 setting.
Memory occupancy is not evidence of delivered training speed or quality.
4. **Frozen comparison.** GPU0: plain GRU256x2/211 causal features/gate-wise Muon
plus AdamW/LR1e-3. GPU1: GALAM LSTM64/AdamW/LR1e-4. GPUs2–7: width128, depth2,
GRU72, history256, TBPTT256, attention/FFN-only Muon respectively. Common seed1001,
strict FP32, fresh weights, no teacher/KD/old initializer, weight decay.01,
clip1, maximum20epochs, warmup-cosine-v2 with5% warmup, train-only all-fit moments.
AdamW beta1.9/beta2.999/eps1e-8; Muon momentum.95/NS5/eps1e-7. Common GALAM batch384
is frozen and hashed; plain GRU keeps32. No LR scaling was added.
5. **Bounded fitting panel.** Exactly8 new scientific fits, one per GPU; no reused
trained control, additional seed, teacher or distillation fits. Engineering
profiles discard their weights. Stop policy: min5 completed epochs,
patience4, min_delta.0005; hard maximum20. Keep every epoch and best/last pointers.
6. **Validation and uncertainty.** Official earlier train fits; complete official
later valid reports original is_scored mask and pooled global_wp_clip2_v1 plus
t0/t1 each epoch. Train-eval32 sequences use explicitly labelled all-after-warmup
rows. Best/stop selection exposes valid; this single-seed screen cannot estimate
training variance or establish significance/private transfer.
7. **Leakage and coverage.** No random primary folds or chronology inferred from
seq_ix. Pin full sources, preserve whole sequences, exclude train components
connected to valid, audit complete source coverage including train group528.
Fit normalization and all ancestry only on retained train members. The adopted
Predictorium chronology is user policy; current calendar timestamps are unproven.
8. **Decision-changing result.** Compare full-valid best/last curves, per-target
deltas and cost to the matched GALAM control. Improvement supports follow-up;
negative or inconclusive single-seed results do not prove a direction useless.
Plain GRU compares the complete system, not an isolated architecture component.
9. **How to run.** The active isolated Torch2.8+cu128 runtime on the authorized
eight-L4 host. `launch/prepare_data.py` audits/indexes/builds the chronological
manifest and prewarmed raw112/event211 normalization on10325 retained train
sequences. Full valid keeps1873 sequences;282 train sequences connected to valid
are excluded. Eight exact-source profiles exercised diagnostics and a prefetched
successor. `scripts/launch_galam_quality8.py` admitted reviewed identity-matched
receipts and started detached owned workers with GPU/CPU affinity. The read-only
dashboard is on loopback18766; node hourly/terminal GitHub reporting survives SSH.
Do not duplicate the active panel. Foreign services are untouched.
10. **Evidence to return.** Config/source/data/runtime hashes, ownership/PIDs,
train/valid loss and WP/t0/t1 curves, LR/preclip gradient norm, sampled model and
CPU/RAM/VRAM/disk telemetry, elapsed/work counts, all epoch checkpoints and
strict-best/last pointers. Keep large payloads on approved storage; publish
bounded evidence and results with honest completion percentages.
11. **Results and trace.** GRU epoch1 reports full-valid WP0.6681328342
(t0:0.6659784031, t1:0.6702872654), all1873 sequences/3,528,273 scored rows.
Its checkpoint SHA is `94db7c641883d5de298e2dbc54d3671b894acac452f8ef03547b3b5e85798536`.
This is an intermediate single-seed result; all eight fits continue, with no final
ranking or public-score claim. Diagnostic train-eval has a different row mask and
cannot establish a train/validation gap. Exact source R9 commit50053909 and
engineering byte archives, independent reviews, process ownership and first-epoch
evidence are linked from `research/experiment_drafts/galam-l4-quality8-20261028/launch/README.md`.

Latest measured snapshot (2026-10-28T06:47:04.673788Z): GRU has completed epoch4, full-valid WP0.6760966529 (t0:0.6688052605, t1:0.6833880453), weighted MSE0.9808772327. Checkpoint SHA `008f5370056ba33c2566663a483dd79fe6b84743501ec907bc6f1915d8649b1a`. All eight fits remain running; the other seven have not completed their first full epoch at this snapshot. This is intermediate exposed-validation evidence.
