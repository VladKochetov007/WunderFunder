# Matched reset versus bounded prefix burn-in

The standard recurrent trainer already carries state across windows. The reset arm
is a controlled intervention, not a description of that baseline. Both new modes
start each update from zero; `burnin-prefix` first replays up to 2000 immediately
preceding rows from the same sequence under current weights, without gradients or
loss, then detaches implicitly through no-grad. `burnin-reset` supplies zero state.
Both retain the exact fixed update partitions, masks, labels, and optimizer updates;
the prefix arm performs additional forward work, so this is not equal wall time.
This trainer's controlled path requires a validated source-qualified teacher cache.
It does not enable family heads, thinning or dropout. Training is not DDP; run
one process on one independently assigned B300. Exact output state/score quality
and timing have not been measured.

On the receiver, first audit the reviewed source SHA, component-split membership,
cache ancestry, GPU driver/build and free memory, and run the project's receiver
causality, reset and numerical checks. For each of two eligible cells set
`SPLIT_MANIFEST`, `DISTILL_CACHE_ROOT`, `CELL_SEED`, `RUN_NAME`, `ARM`, plus
`CUDA_VISIBLE_DEVICES` and `WNN_PROJECT_ROOT` in the environment, then run:

```bash
bash research/experiment_drafts/gold-frontier-hypotheses-20261102/burnin/run_fit.sh
```

For the same split/seed/cache, run once with `ARM=burnin-reset` and once with
`ARM=burnin-prefix`, setting unique `RUN_NAME`s. Proposed pairs F0/1001 and
F2/1002 are not reservations and are valid only when the ancestry and data match.
The receiver's standard dataset and artifact directories must match the trainer's
`TrainingEnvironment` root. Epoch 6 is fixed before evaluation; no outer score
is used for selection. This is an initial 2+2 fit screen; teacher preparation,
receiver validation and later whole-sequence prediction/scoring are additional.
A supported quality claim requires evaluating the exact trained weights on all
held-out rows with the official scorer. The script does not evaluate or submit.
