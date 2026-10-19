# COMBO R1 terminal capture

This is the terminal capture for `computer-vision-homework-idea-combo-r1-v1.service`, invocation `66d47211356441deba6a6ef9b712e844`, using preparation manifest `ec4bd56f08eaebeb13e8f359b15128f1d0a67a329532b3bf9ec664fb72480bee`.

`capture.stdout` is the exact captured JSON envelope (364,157 bytes; SHA-256 `b8e3b3d08ab9922d9139d19abe4e465862eaea5de0e17eca15192d2af3216847`). `TRANSPORT.json` records the local transport result; `capture.stderr` is empty.

The nine `REMOTE_*` payload files were extracted byte-for-byte from the envelope’s base64 payloads. Their lengths and SHA-256 values match both envelope declarations and `ROOT_CAPTURE_VALIDATION.json`; root validation status is `PASS_BEFORE_EXACT_STOP`. `REMOTE_PROBE_LOG.txt` is empty; progress is preserved in `REMOTE_PROGRESS.jsonl`.

`REMOTE_ROOT_ADMISSION.json`, `REMOTE_ROOT_LAUNCH.json`, and `REMOTE_ROOT_IMPORT_SMOKE.json` preserve the gate, invocation, and in-unit import smoke. `REMOTE_START.json`, `REMOTE_SOURCE.json`, `REMOTE_RESULT.json`, and `REMOTE_DONE.json` preserve the producer receipt chain. Output inventory contains a stat entry for the new `model_parameters.npz`; its payload is not present here.

The terminal systemd `ControlGroup` field was empty. `own_cgroup` is null/NA; no host-root cgroup fallback was used. Parent cgroup state and event counters are recorded in the envelope.
