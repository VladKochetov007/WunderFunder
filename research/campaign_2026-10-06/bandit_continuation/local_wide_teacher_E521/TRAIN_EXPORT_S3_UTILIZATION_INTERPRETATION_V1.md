# Seed3 train export: bounded utilization observation

Five read-only samples span20.42s during this exact local export. Selected-device memory1136MiB, utilization68%,2%,98%,99%,98%; worker CPU counter delta corresponds to1.084 average cores. Worker io.stat was unavailable in this cgroup and is preserved as unavailable rather than inferred.

This short window is not a whole-run utilization mean, causal throughput comparison or evidence about Kaggle GPUs. It shows isolated low-utilization samples can misclassify bursty inference. Keep zero loader workers and existing memory boundary for this deliverable; obtain stage timing and a matched actual-device profile before changing loading/framework/kernel behavior. Terminal export completed555.75s, exit0. Whole worker high-water includes file cache and cannot be read as anonymous tensor-memory demand.
