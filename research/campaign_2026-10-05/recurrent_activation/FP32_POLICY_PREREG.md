# E471 common precision policy amendment before any scientific fit

The separately guarded policy diagnostic reproduced the default CUDA parity failure and showed that disabling TF32 for both matmul and cuDNN restores alpha-zero forward/state/input-gradient/parameter-gradient equality within the original fixed tolerances. All E471 activation and optimizer arms will therefore use the same explicit full-FP32 policy, with cuDNN enabled for the reference and TF32 disabled in both affine backends.

A new wrapper sets those three backend flags then invokes unchanged probe/trainer source. Its identity and live flags are stored in the result. No equations, tolerances, data, initialization, LR or optimizer partitions are altered. The first failed probe/source/quarantine remain preserved. Before fitting, the new guarded v2 probe must repeat parity and all six real-data updates successfully, without OOM or limit violations.

Historical E470 cuDNN used its default TF32 policy. Comparisons against its trained tanh controls remain descriptive and mix precision/backend differences; only E471's internal alpha-zero versus modified-candidate contrasts share this amended policy. The planned primary four alpha-versus-zero and three optimizer contrasts stay fixed; two historical backend contrasts are diagnostics, not reproduction claims.

This is a change in the computational precision regime motivated by a pre-training failure. No test score has been seen for E471. The six-fit one-epoch recipe and exposed-cohort limits remain fixed.
