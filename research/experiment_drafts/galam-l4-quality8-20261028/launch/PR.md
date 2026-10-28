# Fixed eight-L4 quality panel with full chronological validation

1. **Status, lineage and owner.** Full execution is authorized for exactly eight
fresh fits. The fixed panel is in `launch/PANEL.json`; raw source delivery and
real-data admission are in progress. Root owns delivery through results. GALAM
is reconstructed from Denis Tomin's screenshots with explicit unknown original
dimensions; no exact reproduction claim is made.
2. **Why it should work.** Test capacity, recurrent depth/type, attention history,
gradient horizon and matrix optimization against the full schematic GALAM
LSTM64 control. Plain GRU256x2/event211 is a historical whole-system benchmark,
not a pure recurrent-cell ablation. Each GALAM arm changes only its declared
treatment; dimensions needed by a wider/GRU model are part of that treatment.
3. **Compute advantage.** Eight independent L4s give one lane per fixed arm.
Strict-FP32 synthetic batch sweeps are measured engineering evidence; full-input
prefetch, shared-reader contention and real profiles still govern admission.
Memory occupancy is not evidence of delivered training speed or quality.
4. **Frozen comparison.** GPU0: plain GRU256x2/211 causal features/gate-wise Muon
plus AdamW/LR1e-3. GPU1: GALAM LSTM64/AdamW/LR1e-4. GPUs2–7: width128, depth2,
GRU72, history256, TBPTT256, attention/FFN-only Muon respectively. Common seed1001,
strict FP32, fresh weights, no teacher/KD/old initializer, weight decay.01,
clip1, maximum20epochs, warmup-cosine-v2 with5% warmup, train-only all-fit moments.
AdamW beta1.9/beta2.999/eps1e-8; Muon momentum.95/NS5/eps1e-7. Common GALAM batch
is selected and hashed before fitting, without silent LR scaling.
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
9. **How to run.** Existing isolated Torch2.8+cu128 runtime on the authorized
eight-L4 host. `launch/prepare_data.py` audits/indexes/builds the chronological
manifest and prewarms raw112/event211 normalization. Then exact-source profiles
exercise diagnostics and consume a prefetched successor. Full fits require
reviewed identity-matched profile receipts. One persistent SSH, detached owned
workers, fixed GPU/CPU affinity and loopback dashboard; foreign services untouched.
10. **Evidence to return.** Config/source/data/runtime hashes, ownership/PIDs,
train/valid loss and WP/t0/t1 curves, LR/preclip gradient norm, sampled model and
CPU/RAM/VRAM/disk telemetry, elapsed/work counts, all epoch checkpoints and
strict-best/last pointers. Keep large payloads on approved storage; publish
bounded evidence and results with honest completion percentages.
11. **Results and trace.** Current measured evidence is only the receiving
compatibility suite and discarded strict-FP32 shape probes under
`launch/receiving/`. Scientific fits/results are not yet claimed. The canonical
packet is `research/experiment_drafts/galam-l4-quality8-20261028/launch/README.md`;
subsequent updates must retain failed engineering attempts and exact source pins.
