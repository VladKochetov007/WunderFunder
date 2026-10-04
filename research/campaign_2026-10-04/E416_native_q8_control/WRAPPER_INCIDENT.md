# Outer-shell error after successful v3 parity

The v3 scientific process completed all six cases successfully. Its `time -v`
record reports exit status 0; `parity_v3.json` and `parity_v3.npz` are complete,
and the inner scoped wrapper captured final memory events with all counters zero.

The outer `COMMANDS.sh` process subsequently exited 2 with:

```
COMMANDS.sh: line 42: unexpected EOF while looking for matching `"'
```

Cause: while the v3 shell was blocked waiting for its scoped command, the agent
edited the same script to add the controller-approved CPU14 selection for the
independent v2 parity launch. Bash later resumed reading the file at its old byte
offset, after the script length had changed. This outer-shell error occurred
after scientific completion and did not alter Python inputs, code, or outputs.
The outer EXIT trap preserved exit 2 and parent resource snapshots.

The v2 process started from the amended script, which passes `bash -n`, and its
command file is now held unchanged. Future concurrent stages must use separate
frozen command files. The failed outer exit remains recorded; it is not silently
rewritten to zero and the successful scientific parity is not rerun to hide it.

The original v3 command used CPU15, as recorded by its hardware metadata. The
amendment changed only control-shell CPU selection for v2 to CPU14; all actual
timing trials retain CPU15.

`COMMANDS_v3_reconstructed.sh` reconstructs the original command text from the
exact recorded patch by removing the three CPU-selection changes; this is a
reconstruction, not a pre-launch filesystem snapshot. `COMMANDS_v2_snapshot.sh`
copies the current execution text. The active v2 shell's original `COMMANDS.sh`
is held unchanged. The upcoming timing stage uses its separate frozen
`COMMANDS_timing.sh` file.

`VERIFY_PARITY.py` independently checked exactly six expected cases, fixed
shapes/gates, all prepared/source/library hashes, complete arrays and finite
requested predictions, and recomputed each saved max-absolute and pooled-WP
error against its saved reference. The bounded v3 audit exited 0 and wrote
`parity_v3_independent_audit.json`. Portable/reset claims remain evidence from
the completed scientific process; they are not inferred from array existence.
