# E465 distributed objective review notes

Numerical checks remain unexecuted. The port must compare weighted global sufficient statistics against the existing unsharded objective, including unequal and zero replica masses. Averaging independently normalized replica losses is not the same objective.

Official JAX documentation describes communication in differentiated sharded programs and manual collective operations. These are implementation references, not evidence that our port is correct: [Autodiff and sharding](https://docs.jax.dev/en/latest/301/sharding-ad.html), [Manual parallelism with shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html). Check the installed JAX version and test the actual reduction and gradient path to avoid double reduction. Recurrent-state carry, detach boundaries, auxiliary terms and EMA require separate parity checks before full TPU fitting.
