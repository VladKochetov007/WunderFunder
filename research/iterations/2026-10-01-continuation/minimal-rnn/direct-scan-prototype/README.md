# Direct positive affine scan: numerical failure retained

**Do not promote this implementation to the generated GPU probe or forecasting training.** The one fixed CPU matrix completed, but only186 of194 numerical comparisons passed. The8 failures are retained in `generated-01/result.json`; formulas, test matrix and tolerances were not changed after observing them. Original `minimal_rnn.py`, its training wrapper and the separate native prototype remain untouched.

## Fixed implementation and scope

`direct_scan.py` composes FP32 affine pairs directly in blocks64:

`a = 1−sigmoid(k)`, `b = sigmoid(k)g(v)`; `(a2,b2) ∘ (a1,b1) = (a2*a1, b2+a2*b1)`.

It retains the actual incoming state, with no detach, log transform, cumulative-log subtraction, new parameters, feature change, clipping or fallback. `DirectMinimalRNN` inherits the original parameters and one-step method; only its batched forward scan differs. Positive sums remove subtractive cancellation in the affine composition, but do not eliminate coefficient rounding or order-dependent accumulation.

Seed2045 generates inputs/models only. Tests cover normal cells and full211→256×2→23 models at T2000 and20000, zero/positive incoming state, gates through±80, irregular chunks and causal prefixes. Cell and complete-model VJPs include inputs, all parameters and h0 at both zero and positive state, with actual FP32 step, old log scan and FP64 recurrence references. Ordinary tolerances were fixed at `atol=2e−5, rtol=2e−5`; extremes use `atol=2e−4` with the same relative tolerance. Source/input hashes precede comparisons in `generated-01/inputs-before-check.json`; the recipe is `plan.json`.

## Main numerical failure: slowly decaying nonzero state

The preregistered retention stress includes constant gate/candidate−17 and actual h0=20. After20,000 rows:

| Calculation | Final state |
|---|---:|
| Direct affine scan |19.9761581421|
| Actual FP32 step recurrence |19.9618530273|
| Original log-domain scan |19.9839038849|
| FP64 recurrence oracle |19.9834471032|

Direct-versus-step maximum error is **.0143051**; direct-versus-FP64 error is **.00728896**, about12.15 times its allowed mixed absolute/relative bound. The same stress already fails at T2000: errors **.00143051** versus step and **.000728279** versus FP64. These six failures concern the comparisons against step, old log and FP64 at both lengths.

Two effects must remain distinct. First, the coefficient `1−sigmoid(−17)` is `.9999999403953552` in FP32 versus `.9999999586006245` in FP64. At−20 it rounds all the way to1, so both the direct scan and literal FP32 step lose the tiny decay. Second, composing/applying many affine updates changes rounding order relative to literal per-row updates, explaining why direct and literal FP32 do not agree even though they start from the same FP32 coefficients. The old log form retains more of this slow decay across a block. Thus literal FP32 step itself is also imperfect relative to the oracle; its error is not evidence that the direct implementation meets the declared parity requirement.

This is not the product-underflow case. In the separately recorded coefficient check, positive products become zero after49 factors at gate2 and26 factors at gate4. Their exact-within-FP64 block64 carry contributions for h0=20 are only **1.53e−58** and **4.14e−111**. Loss of those particular contributions is negligible under these bounds. No blanket safety claim follows for arbitrary state magnitude, gates or gradients. FP64 sigmoid also saturates at sufficiently large logits, so it is a high-precision numerical reference rather than arbitrary-precision real arithmetic.

## Two old-reference gradient failures have a different cause

Two extreme candidate-VJP checks differ from the old scan by1.70609/1.84845, while the corresponding comparisons against actual step and FP64 pass. A separately labeled, deterministic post-result explanation (`explain_gradient.py`, `generated-01/gradient-explanation.json`) isolates this to the **exact candidate value v=0**:

- The actual `positive_g(v)` branch has observed autograd derivative1.
- The old `exp(log_g(v))` path has observed autograd derivative0 on the installed Torch2.14 build.
- For a single row with gate0, output `.25` agrees, but the candidate derivative is `.5` for direct/step/FP64 and0 for old log.

The old log expression contains `clamp_min` at this boundary; the activation is nonsmooth there. This is an existing branch/subgradient mismatch, already present for one row, not evidence that the associative direct scan computes the wrong smooth-region gradient. It was not repaired or used to excuse the independent retention failures. All complete-model VJPs in the fixed generated matrix passed. No frequency or training impact of exact-zero candidates in real models was measured.

## Cost and disposition

The fixed CPU B1/T2000/H256×2/O23 forward/backward timings (one warmup, three measured repetitions, no optimizer) were:

| Implementation | Seconds per repetition | Median |
|---|---|---:|
| Old log scan |.34303, .30384, .34516|.34303|
| Direct scan |.23957, .23437, .23853|.23853|

This descriptive CPU difference does not establish GPU B32 speed or memory savings, and does not override the numerical failure. The correct next disposition is to retain the source/result as an unsuccessful bounded reserve optimization. No alternate coefficients, precision scheme, block size or tolerance sweep was attempted; none is allocated by this result.

The main process completed in34.69s (internal31.86s), peak RSS791,200KiB, about773MiB. It ran only on CPU34, with quota200%, RAM4GiB, swap0, a300s timeout and `CUDA_VISIBLE_DEVICES=''`. The small explanatory follow-up used the same CPU/resource bounds; both owned scopes are inactive/dead. Retained source/generated inputs/results total about21.7MB and are mirrored locally and on SSH. No GPU, real data, targets, trained checkpoint, OOF, WP, forecasting fit or public upload was used.
