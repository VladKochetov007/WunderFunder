# Task 3: combinations as a construction kit

Same setting as task.md and task2.md. New measured 2x2 (one seed, 4 epochs, valid score):

|              | labels only | + teacher (0.5 mix) |
|--------------|-------------|---------------------|
| no age weight| 0.6859      | 0.6881              |
| age weight   | 0.6867      | 0.6905              |

So the event-age row weight and distillation reinforce each other (interaction +0.0016) instead of substituting.
Seed noise is about 0.001. Teacher scores 0.6914. Residuals of the three available teachers correlate 0.995.

Building blocks already implemented as switches: event-age row weight (tau, floor), teacher mix (constant, or
rising with distance to the next label change), auxiliary heads (next-row inputs, book structure), orthogonalized
optimizer on recurrent matrices, hidden width, epochs, bootstrapped targets (teacher value h rows ahead), EMA
teacher, class-balanced sequence sampling, per-class loss scale.

1. Explain mechanistically why age weighting and distillation could be super-additive here (2-3 candidate
   explanations, each with a zero-training diagnostic that separates it from the others).
2. From that, propose 6 *combinations or new blocks* most likely to be super-additive with the pair, each with
   exact change and falsifier. Prefer blocks that act on a different part of the pipeline (target, row weight,
   input, optimizer, sampling, readout) than the two already combined.
3. Give a short screening design to find the best combination in at most 12 runs of 40 minutes (fractional
   factorial or successive halving), stating which interactions it can and cannot identify.
Write to ideas3.md.
