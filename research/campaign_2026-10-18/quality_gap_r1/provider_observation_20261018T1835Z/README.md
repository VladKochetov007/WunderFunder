# Provider observation, 18:32 UTC

The existing read-only observer completed successfully at 18:32:22 UTC. Jobs, balance and restrictions calls all succeeded: the latest 100 jobs contain 46 SUCCESS, 33 ERROR and 21 CANCELLED, with no nonterminal job. Balance remains 41,652,791 units; ALLOW_JOBS and ALLOW_SPEC_G_2_1 are true, and the project VM limit is 18. No credentials were persisted and no provider mutation was made.

The [official incident page](https://status.yandex.cloud/en/incidents/2136) still shows an ongoing incident affecting DataSphere, with the latest visible update on 16 October at 20:19 MSK. The read used a hidden Codex tab, which was closed; the user's Zen browser was not touched. This is no evidence of recovery and does not establish the exact cause of this project's earlier pre-start failures. No unchanged paid canary was retried.

`OBSERVER_RAW.json` and `EXECUTION.json` preserve the API result and exact hash. The stderr record contains a local LibreSSL warning and repeated fork/poll diagnostics; the observer still exited zero and all three provider calls returned success. Those SDK diagnostics are not a new provider failure.
