# Experiment snapshot and current documented SOTA

Highest reported public score: **0.6641**, `candV1_ft2` (two-epoch head FT of C). Scores below are copied from the research journal and have not been independently verified against the competition service.

C recipe: equal-weight distilled GRU256 + GRU128 seed3 + GRU128 seed4 + chunk500 GRU128, head-only gcorr FT, C++ fp16 inference.

| Package | Public WP | Runtime | Bytes | SHA256 |
|---|---:|---|---:|---|
| candA_h256pair | 0.6629 | 35 min | 8115470 | `5e7f7d91c5ae61a5d64a012ed66237da0533ad0b5e4d7f84dc0965363eddea3a` |
| candB_h256_2h128 | 0.6639 | 43 min | 6417390 | `67dbdb63a38d64d8df86c76d15986789930e25052fea53853e45fb84cd6923e9` |
| candC_d256_2h128_c500 | 0.664 | 48m52s | 7382427 | `12dd3ef8bd09f9a5b72040a191d0e05602305996afc51d7299a8ed613e331247` |
| candV1_ft2 | 0.6641 | 41m42s | 7382512 | `ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d` |
| candV2_vr3 | 0.6611 | 41m42s | 7384892 | `b13dfe870ea65606867fd10266aad8b661e2ed22b1c0fd1811fa3884c8e39c75` |
| candV3_trainonly | 0.6629 | 49m26s | 7381174 | `f7943bc668dd012c3be9c944031c1f3a08ffb1950287829d345534f5a0aeb0fe` |
| cpp_h256pair | not recorded | not recorded | 8104924 | `8f8bc9268605d09b1db0f23ac79495d297402107458038cb65b291c8bff2db80` |
| cpp_smoke | not recorded | not recorded | 2781012 | `a98855954a4cba1ad31e4fc74c476d9b281548569bd6090320bda58c3642969a` |

All packaged submissions are preserved, including two engineering smoke packages. See run_index.tsv/json for 266 remote run directories (261 with files) and 860 checkpoints. The live Transformer was not interrupted.
