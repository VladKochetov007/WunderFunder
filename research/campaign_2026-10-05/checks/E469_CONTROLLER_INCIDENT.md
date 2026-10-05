# E469 controller incident; scientific worker preserved

The first queue controller treated the asynchronous guarded-launch receipt
(`reserved`) as a completed worker and exited with an erroneous failure claim.
The actual first-seed scientific worker continued normally and was not killed
or retrained. A first recovery command also referenced a file not yet created;
that service exited before work. Both failures remain in the systemd journal.

The reviewed recovery polls the existing attempt to terminal status, then
verifies exit0/teardown before scoring and launching the second fixed seed.
It reattaches to the same first-seed identity without a duplicate launch.
New source `run_replications_v2.py`; active service
`wnn-e469-fixed-seed-replications-v3.service`. No numerical recipe, model
weights, evaluator, resource guard or compute ceiling was changed. The
persistent checker observes the active scientific attempt and service.
