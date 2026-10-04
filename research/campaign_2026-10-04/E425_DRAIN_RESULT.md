# E425 soft-routing experiment drained under user pause

The already-running four-epoch fit completed fixed epoch 3, its common-FP32 full-VALID export completed, and its planned matched control comparison is now reported. No retraining, new research iteration, head fit, or checkpoint/weight selection was performed during the drain.

| Parent predictor | Global WP, excluded 368 sequences |
|---|---:|
| Soft prefix mixture | 0.696672177 |
| Learned hard routing | 0.696446285 |
| Uniform specialists | 0.696500615 |
| Matched-budget GRU | 0.696092444 |

| Contrast | Difference | 95% paired component interval |
|---|---:|---|
| SOFT_minus_H093_LEARNED | +0.000225892 | [-0.000888640, +0.001238258] |
| SOFT_minus_H093_UNIVERSAL | +0.000171562 | [-0.000618129, +0.001088258] |
| SOFT_minus_H093_GRU | +0.000579733 | [-0.001086098, +0.003104640] |

Every interval includes zero. Soft routing provides a small positive point difference, but does not establish an improvement over hard/uniform routing or the matched GRU. The screen conditions on seed-3 fitted models and 368 historically exposed sequences /249 components, with 2,000 paired resamples. Existing teacher lineage and campaign exposure prevent an independent-holdout claim. These are parent models without the incumbent’s final head procedure; no public score or deployable ZIP is claimed.

All four experts had direct nonzero prediction gradients and parameter changes on every successful update. The full fit completed 30,240 attempts, 30,228 optimizer updates, 12 AMP skips and 30,240 EMA updates. Final-epoch route entropy was 1.364006 nats; mean probabilities were [0.250307, 0.249779, 0.263917, 0.235997]. This supports gradient coverage and absence of gross routing collapse; it does not prove useful specialization.

The fit worker took 1876.23s, peak observed VRAM 4,421,844,992 bytes, peak cgroup RAM 25,757,548,544 bytes (including charged file cache). Export took 40.03s; paired reporting took 17.77s with 431,509,504-byte peak RAM. All three exited zero with verified teardown and no new worker OOM events.

The training-time FP16 diagnostic WP was 0.696717654. The final comparison uses common FP32 exports (0.696672177); these are distinct inference conventions and are not interchangeable. The checkpoint remains fixed epoch 3, irrespective of intermediate diagnostics.

Raw score: `checks/e425_soft_report_drain_v1.json`, SHA256 `9121d02818f3b9b164cc972f06539e79ef0f8108da86c9e63d0ba9d911d27e3a`. Frozen original controls: `checks/e425_control_identity_v1.json`. Reporting design: `checks/e425_soft_screen_drain_v1.json`. Execution: `specs/e425_soft_score_drain_v2.sealed.json`, `runner/attempts/e425-soft-score-drain-v2.json`. The initial score spec was rejected before execution for a reserved environment override; v2 removes that override and leaves all scoring mathematics, controls and outputs unchanged.

E464’s eight head fits remain held and unstarted. The research goal remains paused.
