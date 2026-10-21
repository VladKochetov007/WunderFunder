# NLL terminal result-capture template

`RESULT_CAPTURE_TEMPLATE.py` is a local, inert adaptation of the reviewed dictionary result-capture reader. It is intended only for a future one-time, read-only capture of the exact NLL verifier unit `computer-vision-homework-fg19-child-nll-verify-v1.service` and private output directory `research_tmp/fg19_child_nll_verify_v1`.

The native NLL invocation is pinned to `c833c613fb92426981ab255a5faf4452`. The future support invocation and the exact support `VERIFIER_ADMISSION.json` and `VERIFIER_LAUNCH.json` sizes and hashes remain unbound because no NLL CPU verifier admission has occurred. The reader fails before any remote query while those three values remain unbound. Root may bind them only from the actual support receipts after the one authorized launch.

The capture is limited to the support and parent systemd properties, campaign cgroup population and memory event files, support-cgroup memory files when available, the exact two support receipts, the unit log, the verifier result only after the strict successful terminal tuple, and six exact private checker/helper byte pins. It also samples the fixed selected GPU UUID and compute-process rows with bounded `nvidia-smi` queries and records only PID, GPU UUID, reported memory, UID, and cgroup from `/proc`. It does not reserve GPU memory or change GPU state. If process ownership cannot be read, capture fails closed.

No model, training state, dictionary, cache array, data, source tree, or trainer payload is read or hashed. The reader does not invoke the NLL checker or any fit/evaluation entrypoint and does not stop or alter a unit. An empty own cgroup is recorded as unavailable; it never falls back to the host root cgroup.

`AST_LITERAL_ADAPTATION_V1.json` records the static AST parse and identity/pin substitutions. This was a source-only preparation: the template was not imported or executed, no tests were run, and no B/SSH operation or CPU verifier admission occurred.
