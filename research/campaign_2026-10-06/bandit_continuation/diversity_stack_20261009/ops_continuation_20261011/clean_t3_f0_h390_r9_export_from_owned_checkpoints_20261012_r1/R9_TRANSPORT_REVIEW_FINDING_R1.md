# R9 R1 transport review finding

R1 was not uploaded or dispatched. Its local fixture proved that the nested
checkpoint paths and bytes could be read from a simulated mount, but the actual
Kaggle dataset upload path uses directory mode that skips nested directory
trees. The six checkpoint/provenance files under `experiment/` therefore were
not established as upload payload. R1 is preserved as a failed transport
preparation record, not a failed model or export result.

R2 uses seven flat top-level dataset files (six seed-qualified checkpoint and
provenance files plus the exact manifest) and exercises the flat producer path
in its local fixture. R2 still requires independent packet review, actual
private dataset creation/READY readback, and a ROOT-authored release before any
kernel push or export worker.
