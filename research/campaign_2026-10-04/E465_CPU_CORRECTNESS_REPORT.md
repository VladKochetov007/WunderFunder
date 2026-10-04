# E465 CPU correctness evidence

The private two-device JAX CPU v3 kernel completed all four numerical tests: unequal positive shard mass, one zero-mass shard, globally zero mass, and replicated two-chunk state/gradient carry. Actual JAX/JAXlib was 0.11.1. Raw XML and worker/runtime records are under `kaggle/E465_sharding_cpu_v3/provider_output_complete/e465_sharding_cpu_v3/`. These establish CPU distributed-objective correctness on the tested cases, not a TPU training or predictive result.

The local bounded OneCycle reference v2 completed all 7,461 scheduler indices against the repository's actual Torch 2.14 production builder. Maximum LR, beta1 and beta2 error was exactly zero. The configured one-epoch schedule has 7,460 optimizer updates, warmup 0.05, peak LR 0.001, initial divisor 25, final divisor 100 and beta1 cycling 0.95–0.85–0.95. The final compared index is the permitted unused scheduler state after the last update. The worker took 7.734 seconds and peaked at 1,078,587,392 bytes RAM, with exit 0 and verified teardown.

Earlier failures are retained: v2's all-zero gradient fixture wrongly required a nonzero norm; local scheduler v1 used a literal float assertion that ignored cosine roundoff. Fresh identities corrected the fixtures while keeping their numerical acceptance tolerances and the scientific implementation unchanged.

The real-data TPU preflight remains queued. All-eight-device production memory/throughput, full one-epoch fits, saved-checkpoint inference and scores remain unverified.
