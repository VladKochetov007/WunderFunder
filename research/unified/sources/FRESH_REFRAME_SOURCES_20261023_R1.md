---
kind: source-receipt
date: 2026-10-23
---

# Sources for the 23 October fresh reframe

The associated [idea cards](../ideas/FRESH_REFRAME_20261023_R1.md) state how each source component might transfer and what the project assumptions are. The papers establish methods in their stated domains; none establishes a Challenge2 gain.

## Directed primary literature

- Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114), NeurIPS 2017. Establishes a shared-element map plus pooling form for permutation-invariant set functions and derives equivariant layers. It supports a conditional inductive-bias proposal only if the proposed channel-pair symmetry is useful; it does not say anonymous time-series columns are exchangeable.
- Che et al., [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://arxiv.org/abs/1606.01865), 2016 preprint (later *Scientific Reports*, 2018). GRU-D uses masks and per-variable elapsed time with trainable decay for irregular/missing observations. The reframe deliberately transfers only elapsed-time state as an analogy; Challenge2 rows are observed at every step, so informative missingness claims do not transfer.
- Campos et al., [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/abs/1708.06834), ICLR 2018. Learns to skip recurrent updates and includes update-budget pressure. This motivates a cheap input-triggered replay, but task results and quality do not transfer.
- Pham and Pagh, [Fast and Scalable Polynomial Kernels via Explicit Feature Maps](https://itu.dk/people/pagh/papers/tensorsketch.pdf), KDD 2013. TensorSketch approximates polynomial-kernel features using CountSketch and convolution, avoiding an explicit tensor expansion. Its complexity and error guarantees are source claims for its construction; the proposed 32/64-dimensional recurrent feature requires its own local profile and quality test.

## Seeded random local-source selection

- Authorized pool: non-README `.txt` and `.md` files directly in `research/unified/sources/competition-writeups/`, sorted by relative path. At selection time the candidates were `discord_tricks.txt`, `rogii_top1.txt`, `santander_transcript_english_translation.md`, and `semenov.txt`.
- File selection: compute SHA-256 over UTF-8 bytes of `20261023|<relative path>` for each candidate and select the lexicographically smallest hex digest. Digests in that order were `c67f1353979a3418bacd99f66e2654b40222db2bd99efcbfa9ea5a5ce7f780ba`, `823667e16622e9610224f08b937e917c6d0fadc4836a73daca962f312001c0c3`, `9169db7d5e9ce8adc0f400acc9a62bb24e27600bb4380fc1b16d1164070f82b0`, and `611602fabf38bfde335358b07eb7c1ecfc8d9b367abaec91fca18cc03f74fd2f`; the selected file was `semenov.txt`.
- Passage selection: the file was 1,092 lines. Partition lines into consecutive zero-based 16-line blocks (final block may be shorter), giving 69 blocks. Compute `int(first_6_hex(SHA256("20261023|research/unified/sources/competition-writeups/semenov.txt|16-line-block")), 16) mod 69`; the hash was `c54e1aa46d538e220ad87532f8d45190bf9c0dd9cc2838b4fe0cc8d4a615f4e1`, selecting block 55, source lines 881–896.
- Exact short passage from the selected block: “а выкидывать ненужные признаки потом бы с другой стороны подошёл / начал бы выкидывать сначала самые важные”. Translation: “then approach feature removal from the other side / start by removing the most important ones.”
- Source identity: `research/unified/sources/competition-writeups/semenov.txt`, SHA-256 `72ac3b3dab7b4d9d8460f5d3e4fb53e0ec88049af20e9f07eefe87e415725736`.

The transcript is an imported competition discussion, not a controlled comparison or a general feature-selection result. Its line segmentation is noisy. The card uses only the procedural possibility of reversing a greedy feature-removal order and rechecking excluded variables, as a reason to keep raw bypasses and compare the proposed pairing against a negative pairing.

## Post-commit duplicate check and mapping

The comparison was performed only after the first proposal/source commit (`34381bc38`). Per-channel time-since-change matches the already retained [H-PER-FIELD-AGE-20261022-R1](../ideas/PER_FIELD_AGE_20261022_R1.md), so this round adds GRU-D provenance and a matched test description but no new hypothesis ID. Degree-bounded polynomial sidecars are already represented by H-210 under H-182; TensorSketch is recorded only as child H-397, whose distinct question is whether hashing lowers the cost of that existing interaction representation. H-395's pooled shared-pair encoder adds pair-permutation sharing and a raw bypass; targeted searches found no prior permutation/set-encoder arm. H-396 tests input-novelty-triggered recurrent-update skipping for compute value; it is linked to the related H-231 event-time state family but changes update execution rather than adding a clock/delay representation. No tested result was interpreted as a negative for these proposals.

## Independence and exposure receipt

This was a fresh-context assignment with no access to the residual-fit implementation. However, before the proposal cards were committed, a broad `rg` search accidentally displayed fragments of older idea notes. This output contaminates the blind-proposal condition; the cards remain useful source-grounded hypotheses but must not be described as a fully independent blind round. The exact model/provider for this task was the assigned native Codex agent session; no external model or additional provider route was invoked. Web research used primary paper pages/PDFs linked above. No data, fits, tests, training, profiling, or external weights were accessed.
