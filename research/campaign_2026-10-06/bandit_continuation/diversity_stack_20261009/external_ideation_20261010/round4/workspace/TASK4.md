# Round 4: train longer, differently. Generate ideas, we will test them.

Read `METHOD_REVIEW.md` (our current best pipeline), `NOTES_ROUND3.md` (research notes) and `learning_curve_6epochs.jsonl` (fold-1 score per saved snapshot of a 6-epoch run, 3 snapshots per epoch) in this directory. Competition: causal per-row regression on 20,000-row sequences, weighted Pearson, our 1M-parameter 2-layer GRU plateaus at about 0.666 on validation.

Facts that puzzle us:
- A teammate reports that training for MORE epochs gave better performance for them. Our default (Muon on GRU matrices, lr 1e-3, wd 0.01, one-cycle, 3-4 epochs) gets WORSE after epoch 3 (0.6671 at epoch 3, 0.662 at 4, 0.649 at 5). Everything else we tuned moves the score by under 0.002.
- We can now rent GPUs for runs of 10-20 epochs (about 3 h each) and want the most informative set of 6-10 runs.
- We are about to test: lower lr (3e-4 to 5e-4), higher weight decay (0.05-0.1), dropout 0.1-0.2, AdamW vs Muon on GRU matrices vs Muon on every 2-D weight (input projection, GRU, head), one-cycle vs warmup-stable-decay (WSD) schedules, 15-20 epochs.

Your job: from your own knowledge and judgment (we deliberately gave you little guidance; keep your own biases), propose **eight concrete training-regime experiments** for long runs (10-30 epochs) on this task. For each: the exact settings (optimizer, lr, wd, schedule shape, batch/chunk, regularisation, augmentation, weight averaging, anything else), why you expect it to work on THIS data (you may use the facts above: label depends on a long future path, sequences are 20,000 rows and highly autocorrelated so an epoch repeats the same sequences, data is order-book features with multi-rate update clocks), the failure signature in the learning curve that would tell us to stop it early, and a rough probability that it beats 0.670 on fold 1. At least two must be unconventional (not a standard lr/wd/dropout sweep). Finish with the single run you would start first.
Answer directly in Markdown. Do not edit files.
