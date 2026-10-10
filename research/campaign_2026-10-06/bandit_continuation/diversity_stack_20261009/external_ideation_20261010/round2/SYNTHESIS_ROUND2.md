# External ideation round 2 (reframings, elephants, teacher ceiling): synthesis and tests

Providers: Grok grok-4.7 (plan mode, one prompt-file run), Freebuff (GLM 5.3 Flash via tmux), OpenCode nemotron-3-ultra-free. Shared task `workspace/TASK2.md` (measurements only). Raw outputs: `grok_round2.md`, `workspace/FREEBUFF_ROUND2.md`, `opencode_nemotron_round2.md`.

## Convergent proposals
- Event/hazard reframings (predict change timing + direction, integrate): all three. Largely already in the base recipe (`aux_next` is a next-change time + sign head, `aux_struct` a structure head, weight 0.3 each); the ideators re-invented parts of our baseline.
- Multi-rate hierarchical recurrence on the clock blocks (Freebuff, Nemotron): the earlier two-clock arch lost 0.002; not rerun in the new form.
- Privileged/hindsight teachers that transfer a hazard, regime posterior or hindsight-optimal action instead of a level (all three): untested; needs the lookahead ladder first.
- Elephants raised: label availability at serving (checked: DataPoint carries no label, so residual-to-last-label targets are impossible), class composition per fold, per-sequence coordinate redraws, run (not row) as the unit of data.

## Post-hoc tests on saved predictions (fold 1, best student, CPU, no training)
All causal inference-side families are closed:
| Test | Result |
|---|---|
| Output smoothing (EMA), sign hold, margin hold | all negative (H-326) |
| Clock gating: hold output unless a block updates (Grok R1) | every block lowers WP: fast 31% block -0.114, 70% block -0.007, 100% block -0.0000 |
| Class-conditional EMA (class decodable after 500 rows) | best coefficient per class is 1.0 (no smoothing), cross-fitted change +0.0000 |
| Change-detecting adaptive EMA (reset when output jumps by more than tau) on class 3 | -0.0001 to -0.0235 |
| Post-hoc scale by class / oracle run age / class x age | -0.0000 / -0.0001 / -0.0003 (the student is already calibrated) |
| Oracle run atoms (diagnostic, needs labels): mean prediction inside each true run | pooled -0.0078, but class 3 +0.0251 (0.4425 -> 0.4676) |
| Oracle run atoms: prediction at the first row of a run / at row 4 | -0.139 / -0.117: the new level is not known at the change; it accumulates over several rows |
Reading: inside long class-3 runs the output wanders (worth +0.025 on class 3, about +0.012 pooled, if boundaries were known), but nothing computable causally from the output or the clocks recovers boundaries; the model's own jumps are not the label's boundaries. The remaining route is a model that estimates boundaries better, a training-time change.

## Open and queued
- Information-ceiling ladder (INPUT_LEAD 2/8/32/128/512 rows) on Vast B: if near-future features determine the labels the privileged-teacher ideas stay alive, otherwise they close.
- Chunk-500 replicates and interaction with width 384.
- Grok's elephant checks not yet run: sign/magnitude swap inside class x age x target (partially done: class-level decomposition shows sign dominates also in class 3: oracle direction 0.766, oracle magnitude 0.459 vs model 0.4425), half-data curve by dropping runs vs sequences, per-sequence coordinate transfer test.
