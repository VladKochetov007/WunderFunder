# Direct receiving-host result and resource check ? 2026-10-28 13:07?13:10 UTC

Seven GALAM fits are healthy and progressing. Plain GRU256x2 completed its declared early stop after10epochs; no fit was restarted and no frozen config/source was changed. Exactly eight scientific fits have been launched.

| Cell | Completed epochs | Best full-valid WP | Last full-valid WP | State |
|---|---:|---:|---:|---|
| g0-plain-gru256x2 | 10 | 0.677856123 (epoch8) | 0.669716026 | early_stopped |
| g1-galam-lstm64 | 3 | 0.669442024 (epoch3) | 0.669442024 | running |
| g2-galam-lstm128 | 3 | 0.672369413 (epoch3) | 0.672369413 | running |
| g3-galam-depth2 | 3 | 0.670843114 (epoch3) | 0.670843114 | running |
| g4-galam-gru72 | 4 | 0.671357523 (epoch3) | 0.670438506 | running |
| g5-galam-history256 | 3 | 0.670725562 (epoch3) | 0.670725562 | running |
| g6-galam-tbptt256 | 3 | 0.661951271 (epoch3) | 0.661951271 | running |
| g7-galam-muon | 3 | 0.669924194 (epoch3) | 0.669924194 | running |

All best/last checkpoint bytes were SHA256-checked against their pointers. All recorded epoch evaluations cover1873 official-valid sequences and3,528,273 is_scored rows. The source/split/PANEL pins match the admitted launch. This verifies stored artifacts and reporting coverage, not an independent rescore from prediction arrays. Latest train events advance; no error event was found in each bounded last512KiB monitor tail.

LSTM128 has the highest current GALAM score. Its first3epoch WP curve is0.654305 ?0.663707 ?0.672369. The ordinary GRU is a different whole-system recipe and has already trained10epochs; this is not a matched-budget architecture ranking. GALAM-GRU72 decreased from0.671358 at epoch3 to0.670439 at epoch4; one lower evaluation does not establish persistent overfit. Primary valid is exposed for checkpoint/stop selection, and each arm uses one fitting seed. Train-eval and valid have different masks; do not infer a generalization gap from their raw values.

## Consecutive live resource sampling

24 one-second-spaced NVIDIA queries over27.47seconds, ending2026-10-28T13:09:12.975206+00:00. This is a short observed window, not whole-run FLOP efficiency. Physical device readings include any unrelated resident service.

| GPU | Work | Mean GPU busy | Used device memory | Mean power |
|---:|---|---:|---:|---:|
| 0 | g0-plain-gru256x2 | 0.0% | 0.00GiB | 16.4W |
| 1 | g1-galam-lstm64 | 20.4% | 16.76GiB | 41.7W |
| 2 | g2-galam-lstm128 | 41.1% | 19.38GiB | 57.3W |
| 3 | g3-galam-depth2 | 26.8% | 16.67GiB | 42.2W |
| 4 | g4-galam-gru72 | 34.8% | 11.43GiB | 48.5W |
| 5 | g5-galam-history256 | 81.2% | 16.04GiB | 71.2W |
| 6 | g6-galam-tbptt256 | 29.7% | 18.55GiB | 45.5W |
| 7 | g7-galam-muon | 29.7% | 16.72GiB | 44.6W |

GPU0 has no CUDA allocation after GRU completion. Active-device mean busy is37.7%; all-eight mean is33.0% within this window. This is GPU busy time, not percent of theoretical FLOPs. Per-process CPU consumption is approximately one logical core. Together with the serial per-time-step Python forward this supports CPU/kernel-launch overhead as a profiling hypothesis, not a measured time attribution. GPU5 history256 averages81.2% busy and reaches98%; direct nvidia-smi performance evidence at13:10:12UTC reports SW Power Cap active at72W, hardware/software thermal slowdown inactive, temperature82C. No power/cooling setting was changed.

The last completed GALAM fit epochs take97.7?131.4minutes plus6.8?9.2minutes for paired train-eval/full-valid evaluation. Evaluation is about5.7?6.5% of their combined duration. These fit durations include data/diagnostic overhead; separate loader/H2D/forward/backward timing and SM/FLOP counters are unavailable. No claim of optimal utilization is made.

## Artifacts

- [Results and timing](results.json).
- [Checkpoint hashes and complete epoch curves](checkpoint-and-curve-proof.json).
- [Raw live GPU samples and process CPU measurement](gpu-samples.json).

Existing node hourly/terminal PR46 publishing remains active; its process was present. SSH98495 is the current persistent session; retain it for further reads.
