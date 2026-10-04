# E465 actual FP32 JAX/Torch AdamW check

Private Kaggle CPU kernel v2 completed 7,460 optimizer updates using the actual staged JAX AdamW implementation and Torch AdamW. Maximum parameter error was 4.202127456665039e-6, first-moment error 1.1920928955078125e-7, and second-moment error 1.430511474609375e-6, all below the unchanged preregistered limits. Frozen mean/std buffers stayed unchanged and had no optimizer moments.

The bounded kernel finished in 27.89 seconds with aggregate peak RSS 855,998,464 bytes, under its 4 GiB cap. This proves this scalar FP32 schedule/update comparison on JAX 0.11.1 CPU and Torch 2.11.0 CPU. It does not prove full-model, distributed TPU, or forecasting equivalence.

The v1 kernel executed all updates but failed while aggregating NumPy errors due to tuple indexing of a one-dimensional array. Its logs remain immutable. V2 changes only that tuple to list indexing; seven root CPU regression checks passed before launch, with no tolerance change.

Raw result: `kaggle/E465_adamw_cpu_v2/provider_output_v2/e465_adamw_fp32_cpu_v2/adamw_fp32_parity.json`. Launch receipt and root admission bind source hashes, caps, and actual provider status.
