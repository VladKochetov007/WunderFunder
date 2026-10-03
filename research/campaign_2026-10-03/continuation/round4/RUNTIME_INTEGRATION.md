The frozen actor shell proposal invokes scripts/limited directly under the parent slice. That provides a scope memory/time cap but bypasses campaign_job's exclusive GPU admission, persistent job record and periodic peak device monitoring. A static queue_finished receipt is not a substitute for live admission. Do not run that entry directly.

The controller-owned controller_gpu_payload.sh keeps the frozen Python runner and candidate hashes unchanged while using the existing runtime chain:

`campaign_job.py launch --kind gpu -> continuation/run_gpu.sh -> scripts/limited in wnn-campaign-gpu.slice -> controller_gpu_payload.sh -> frozen runner`

The payload rejects a missing campaign marker or wrong family slice and limits its child to590seconds. campaign_job owns600seconds of execution after admission, source hashes, service state, aggregate quotas and total observed device cap5.5decimalGB. The watcher preserves incidents and blocks new admissions. Parent slice remains12CPU/36GiB; GPU family8CPU/24GiB. No launch or receipt has been issued. Existing combo fold3 must finish and its terminal unit/job state must be independently verified before any queue_finished receipt.

Independent critic must still review capture semantics and CPU evidence; this wrapper is not scientific approval. The fresh trajectory/panel-normalization setup does not recover the states of completed full fits. CUDA default/online parity and skipped-state behavior remain untested.
