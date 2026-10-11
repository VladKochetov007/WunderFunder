# H382 export recovery R4

R4 repeats only the three prediction exports from the completed H382 R3
checkpoints. It preserves the R3 source bundle, checkpoints, split, fitting
recipe and prediction export arguments. R4 closes the final collection gap: the
worker verifies each copied NPY and JSON sidecar against that same run's fresh,
validated export receipt (run, source packet, split, checkpoint, epoch, bytes
and SHA-256) before marking the terminal inventory complete. Historical R3
prediction hashes remain diagnostic only.

The worker has zero fit commands and three export commands, each using the
exact completed R3 command `python /root/wnn/scripts/predict_run.py RUN valid
--epoch E`. A copied pair that is missing, truncated, corrupted, or bound to a
different run/source/split appends a worker error and prevents completion. The
three large NPYs are checked by bounded file hashing; the already-validated
fresh export receipts also retain the exact shape, dtype, finite scan,
checkpoint and sidecar bindings.

R3 measured export commands at 160.5, 82.2 and 160.6 seconds; worker startup
and setup were not measured separately. The 1,800-second per-export and
3,600-second worker limits are unchanged. Root owns fresh quota/session
admission and any dispatch decision. This packet is prepared only; no upload
or kernel push is performed here. The 12,441,600-byte private checkpoint
archive is rebuilt from the already rescued R3 checkpoints and is never added
to git.
