# E481 fit-only grader v2 independent review

**Disposition: fit-only gate passed; no held-out labels were loaded and no scores were computed.** The original `score_recurrent_curvature.py` and test file remain untouched. Its attempt to validate this fit manifest exposed a real defect: the v1 validator calls nonexistent `e470.file_hash`. That failure remains preserved in v1. The corrected v2 uses the available hash helpers and adds a CPU-only `--verify-fit-only` mode; the command below ran against the actual completed attempt and all 18 actual checkpoints.

Frozen grader identities:

- `score_recurrent_curvature_v2.py`: SHA256 `ce439bb122d640279f8b199e7ee3af002f4ca70e16d6a171cf47e9587be9a4f3`
- `test_score_recurrent_curvature_v2.py`: SHA256 `e89f71e9aafbaf177478fd038c969ebe45d0df8383407cf11b73747d4bcea6e4`
- Frozen v1 grader/test, unchanged: `score_recurrent_curvature.py` SHA256 `52b91f9a3eaead30c71ef824feed8370b5a1e339ad7f8005c509b115b903c4d5`; `test_score_recurrent_curvature.py` SHA256 `b233bf363ce111d96ffb81194dd54249e1d4528a8eeecdbc889fe56932b67379`.

The direct fit-only verification ran under a systemd scope capped at one CPU, 2 GiB RAM, and no swap:

```bash
python research/campaign_2026-10-05/recurrent_activation_curvature/score_recurrent_curvature_v2.py \
  --verify-fit-only --device cpu \
  --run research/campaign_2026-10-05/recurrent_activation_curvature/runs/eighteen_v1 \
  --attempt-record research/campaign_2026-10-04/runner/attempts/e481-candidate-curvature-eighteen-fit-v1.json \
  --sealed-spec research/campaign_2026-10-05/recurrent_activation_curvature/e481_eighteen_fit_v1.sealed.json
```

The command succeeded with `validated_cells=18`, `fit_role_count=24`, `test_role_count_declared_only=16`, `test_rows_loaded=false`, and `test_predictions_scored=false`. It verified the current sealed attempt, fixed spec, runtime hashes, artifact inventory, success/exit status, clean teardown, unchanged worker and parent OOM counters, dataset/cohort/source hashes, all 18 checkpoint hashes and finite tensors, model keys and identity buffers, exact 474-update recipe, fit-only normalizer, and the 24 fit-row traversal. It emitted fit mean hash `612d94fb6001c5c5c60a9c7a79276b932683fa486dab080e449e81ba018a3f6e`, fit scale hash `5f55c740c75758aa43faf7a14924de384f908051354097e25b2fc71a7d143f96`, attempt hash `a52e619382da8df25eda154fb538b2500ce28f9da3b17d0a741d756272b110e7`, and fit-manifest hash `9ddb8e20506a996187bba1235ce9c0251bb724939643d543351d04ccf781812a`.

I separately checked the actual JSON manifest against the canonical cohort permutation: all 18 arms contain the exact seed-shuffled 24 fit row groups; all 72 fixed telemetry samples identify the expected four-row minibatch, use `chunk_start=(step % 79)*256`, and have finite loss. This check read the fit manifest and cohort metadata only; it did not read test target rows. The fit-only grader tests pass (4 tests), and Ruff plus `py_compile` pass for the v2 grader/tests. The scoring branch following `--verify-fit-only` remains unrun.

Actual checkpoint identities, as verified against each checkpoint file:

| Seed | Optimizer | Shape | Checkpoint SHA256 |
|---:|---|---|---|
| 271828 | AdamW | baseline | `373f064ccfb20909ad861f6fd6adbc341ca57c0b187250ad559b6895829c77c3` |
| 271828 | AdamW | cubic | `176c535229dcc3edb7682b3f2d504164365dc84fef2a09c0c50877fbfc47c8f8` |
| 271828 | AdamW | sine | `a3250498a8e44f063211964fc1140b7acfff516e0595c1fd7ee966de94602e4f` |
| 271828 | MuonAdamW | baseline | `a0646b501e578c56d81eead33e0a6424dd5943242552dfad1b4483007a617304` |
| 271828 | MuonAdamW | cubic | `06f58b5af7abea39b5e1b3a83fb75c8a80aca3958e5002c675ab704e0698c7a9` |
| 271828 | MuonAdamW | sine | `a7effc74d7cf3fec9e577f032d5cb7cca2a8372175c6a5f44182b0d548191b10` |
| 271829 | AdamW | baseline | `bf01b618cd07de2137602c6d1171b0313fd9e9d08f0798d3bc9116e6265099d7` |
| 271829 | AdamW | cubic | `b6caf474b3b21ee03197c5ef8918373eaf8e71ddf5fdf476f04d6a466488714d` |
| 271829 | AdamW | sine | `3e686ab4ca4029021212d8d9080d2861683e80cb85fa21adfe43a1bf45581ebc` |
| 271829 | MuonAdamW | baseline | `2bd426dd6cf3980d8665a06251f0907e81780bf30a9bf1fbb0a9480499c0b968` |
| 271829 | MuonAdamW | cubic | `7822859d3b6e9e307082326e33f90e5c4e1056598222007fe34b49d464b21785` |
| 271829 | MuonAdamW | sine | `f3c93bf390a0299071f28164b0607a992d79b25f4dc2162059771ea680492e69` |
| 271830 | AdamW | baseline | `8dd8aedba06a3c425e56c9feeb91b03649562d4796cda997d4361ba607863c90` |
| 271830 | AdamW | cubic | `a509f1daa89894ae48751642abfd1cf3b6a9c95bf13475baf566546e822de466` |
| 271830 | AdamW | sine | `09d898baf6eadb09cfc6c47b392a3005fa7d8f6d4a5e05a6a038373e2340a308` |
| 271830 | MuonAdamW | baseline | `cd49711ab78091ffd151ce9af051f5f86c69aae729ea0a08f8e43e3fe1ca5aea` |
| 271830 | MuonAdamW | cubic | `ebb870b8148af83463c97b30b2905ae8b16c202f293928c666443b5ae866c402` |
| 271830 | MuonAdamW | sine | `303c99df2b115786f240cb6fb45eac0517297661208f0fe128079229d5444424` |

This receipt clears only the fit-only integrity gate. Statistical scoring and any interpretation of the exposed evaluation panel remain pending the controller’s separate authorization.
