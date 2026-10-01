Fixed full4 INT8 completed its single allocated quality/runtime study. The exact ZIP retained all four members for the entire stream and has no wall-clock governor.

| Measurement | Result |
|---|---:|
| Native INT8 pooled WP, 860 pure exposed sequences | 0.7023370762430894 |
| Same four members, native FP16 pooled WP | 0.7023454478196809 |
| INT8 minus FP16 | −0.0000083715765915 |
| Fold0 / fold1 difference | −0.00000833671647 / −0.00000711321399 |
| Actual complete process, 39.4M callbacks | **57m49.44s** |
| Margin to 60 minutes | **130.56s** |
| Callback loop only | 55m46.08s |
| Runtime peak RSS | 362,228 KiB |

The actual runtime included source/input checks, extraction, imports, model loading, data reading, 39,400,000 Python callbacks, output checks and final input hashing. All 39,204,970 emitted pairs were finite. The stream contained 1,873 original sequences plus 97 repeats with fresh IDs; all 97 complete repeated prediction hashes matched their originals. All four members remained active. Average sibling activity was 1.46%, including a brief busy interval; no time was subtracted and there was no retry.

The measurement used the scorer's CPython3.11 environment, NumPy2.2.6, PyArrow19.0.1, one CPU equivalent pinned to CPU39, natural v3/VNNI dispatch, a16GiB cgroup and no swap. All scopes are inactive/dead and the owned runtime PID is absent. This qualifies the measured beleriand configuration against the official60-minute limit. AVX2/v2 fallbacks are packaged but their full4 runtime is not qualified. Public hardware and hidden-test quality remain unmeasured.

The four fixed final endpoints are `find256_s3h2`, `fin128_s3h2`, `fin128_s4h2`, `finc500b_s3h2`, each `model_ep1.pt`, in that order and with coefficient0.25. The quality comparison uses their held-fold analogues; it is not a score obtained from final-fit weights on their training data. There was no training, weight search, public submission or GitHub push.

Artifact: [quant4_fixed.zip](package/quant4_fixed.zip), 7,465,748 bytes, SHA256 `c2313be187bb152a7aa0244c2700eae90ab903b50608e23f4023bc51c35bfbdc`.

[Outcome and evidence hashes](outcome.json), [quality](quality.json), [actual launcher result](execution/launch-result.json), [worker result](execution/worker-result.json), [per-sequence trace](execution/sequences.jsonl). Full prediction arrays remain on beleriand under `full4-int8/cache/`; metadata, exported weights, ZIP and execution evidence are also local. The frozen runtime plan's unused inherited prose is documented in [its clarification](runtime-plan-clarification.json); actual source and execution establish the fixed4 policy.
