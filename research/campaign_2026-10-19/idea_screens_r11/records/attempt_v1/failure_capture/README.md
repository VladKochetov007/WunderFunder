# Captured V1 startup failure

The exact invocation exited1 in the CLI import-inventory gate: fresh `python -m` executes the entry module as `__main__`, while the guard expected its named sys.modules key. The cold named-import smoke did not exercise that entry path. The traceback precedes `execute()` and the output directory is absent: no sample payload/model/optimizer/scorer execution occurred. The failed unit is preserved, with no restart or stop.

[FAILURE_CAPTURE.json](FAILURE_CAPTURE.json) contains raw byte-verified ROOT_ADMISSION/ROOT_LAUNCH/ROOT_IMPORT_SMOKE/probe.log and systemd properties. Parent cgroups have no populated descendant; hierarchical memory events remain high2179/max561/OOM0, parent-local0. The next attempt requires a focused reviewed entrypoint fix and new disjoint identities; this is packaging evidence, not a negative sampling/model result.
