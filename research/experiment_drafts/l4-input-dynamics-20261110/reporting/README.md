# Local progress publication

`publish_pr78.py` reads the ROOT-owned local native mirror; it opens no SSH
connection and starts no training. The detached publisher posts every30minutes,
on completed stages, and when both fixed child/control comparisons qualify.
The existing allowlisted client verifies the requested SemyonEpanov account.
Authorization is supplied only through the child process environment; it is not
written to the repository, owner receipts, logs or publisher state.

```text
python publish_pr78.py --root <next70-artifact-root> --pr78 --maxseconds 43200 --state-file <external-state-file>
```

The actual receiving admission bytes are pinned to `b544180c…`. Before showing
metrics the publisher checks source/config/data/split and fitting membership,
physical GPU, fixed epoch/update counts, complete official child validation and
the child-initializer-to-parent-model join. Parent pretext loss is training-only.
Final children are compared with same-seed PR77 shared6E controls. Model hashes
come from the producer receipts and are explicitly not described as byte-verified
by the local publisher.

Actual terminal receipts contain full provenance and are about703KB; their bounded
reader accepts2MB. The earlier PR77 64KB bound withheld valid completion evidence
and was repaired separately. The PR78 reporter inherits the corrected PR77
helpers. Publication uses persisted pending markers and reconciles ambiguous
responses before another POST. A failed final POST keeps polling within the
finite deadline instead of silently ending the completion publisher.

Focused regression checks exercise a failed final publication followed by a
successful retry of the same marker. A separate read-only implementation critic
checked admission pins, stage/count/coverage joins, exposure labels and exit
behavior after the repairs and found no remaining blocker. The actual live-mirror
dry-run identified both parents and withheld pending children/final scores.
The subsequent actual first comment was posted to PR78.

Reports describe repeatedly exposed development evidence. They do not assert a
fresh holdout, statistical significance or a public-score gain. Source timestamps
remain unchanged; collector freshness is only a local transport diagnostic.
