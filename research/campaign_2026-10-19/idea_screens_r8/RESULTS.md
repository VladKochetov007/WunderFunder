# R8: constant variance does not reproduce the R7 gain

One new constant-variance MLP arm scores **WP0.5013935202**. The stored R7 input-dependent variance endpoint remains ahead by **+0.0129033820**; the constant arm is−0.0005278490 below stored WMSE. R7 models were neither retrained nor re-inferred. This is a useful control for the positive R7 pilot, not a new higher-scoring model.

| Fixed endpoint | Target0 WP | Target1 WP | Pooled WP |
|---|---:|---:|---:|
| Stored R7 WMSE | 0.5196829254702040 | 0.4841598129320762 | 0.5019213692011400 |
| NEW R8 learned per-target constant variance | 0.5188727339471939 | 0.4839143064082209 | 0.5013935201777074 |
| Stored R7 input-dependent variance | 0.5242803195323170 | 0.5043134849177928 | 0.5142969022250550 |
| Primary: input-dependent minus constant | +0.0054075855851231 | +0.0203991785095719 | **+0.0129033820473475** |
| Secondary: constant minus WMSE | −0.0008101915230101 | −0.0002455065238553 | **−0.0005278490234326** |

All32 leave-one-component primary contrasts remain positive[+0.00943748799,+0.01430403788]; all32 secondary contrasts remain negative[−0.00079316653,−0.00041505213]. These are sensitivity checks of the same endpoints, not independent training replications or equivalence evidence for constant variance versus WMSE.

[Pre-code design](DESIGN.md) and [design review](DESIGN_REVIEW.json) froze the new control after seeing R7. It retains the224→64GELU trunk and both Linear64→2 heads, but applies the variance head to `hidden*0`. Zero variance weights stay exactly zero, the two biases learn, and variance contributes zero trunk gradient. Initial fingerprint, fitting rows/normalization, epoch permutations, loss/clamp, AdamW, gradient clipping and12epochs/756updates match R7. The reused trainer's non-control branch is explicitly named `heteroskedastic` in code; this implementation alias does not mean R7 was rerun.

The same weighted Gaussian loss is used: `sum(w*(exp(-s)*(mu-yclip)**2+s))/sum(w)`, `w=abs(yclip)`, label/clamp bounds±2/±4. Final constant logits are0.06774963/0.06474842, within the fixed clamp. They adapt global per-target scaling during fitting; this particular control did not reproduce the input-dependent arm's gain. Because it also removes variance-to-trunk gradient flow, the result **does not isolate variance expressivity from optimization coupling or demonstrate true-noise estimation**. Both are one-seed, finite12epoch recipes on the adaptively exposed panel.

The new endpoint uses the original32component/59,436row evaluation masks, after fitting only31,808stride40 rows from64separate components. No new sampler, CV/grid, hyperparameter selection, early stopping, intermediate outer score, row removal or R7 replay. The stored R3 ridge WP0.536004 remains higher under a different recipe. No full-fold, GRU, public or CP30 gain is established. All fit curves remain in the exact result; the constant arm's final fit WP0.599902/WMSE0.991317 is descriptive, not an endpoint-selection rule.

Idea22 retains a positive conditional lead beyond this tested global-scaling control. Do not automatically add a succession of variance variants or a GPU campaign. The next bounded screen may address another concrete untested mechanism among the25; any extension of this objective needs a separately frozen question and distinct-seed confirmation before broader claims.

## Exact evidence

- [Result](records/probe_v1/terminal_capture/RESULT.json): SHA`eaff53d750bbcb6e4cd9d2c174ee0bd9c61add14ef773cb74f0168787d212f99`; [source](records/probe_v1/terminal_capture/SOURCE.json): SHA`e5f623c6ba02bcc52250280cbc28a04f7a2b38c78c88e22c27b2c145d3fb0f77`.
- [Independent compact audit](RESULT_AUDIT.json) reconstructs the endpoint and component/drop-one contrasts from six sums and the frozen R7 result, without arrays/refitting/model replay. [Implementation](IMPLEMENTATION_REVIEW.json), [8tiny checks](TEST_RECEIPT.json), [42-pin manifest](PREPARATION_MANIFEST.json) and [startup review](ROOT_STARTUP_REVIEW.json) precede the sole actual invocation.
- [Execution records](records/README.md) and [resources](RESOURCE_RECONCILIATION.json): unit exited0 at02:19:31UTC, raw captured before exact stop02:22:10UTC;756/756updates,0nonfinite,0new parent/F3 memory events. No new F3 optimizer/quality measurement is implied.
- B-only archive `B/research_tmp/nf19-r8-v1/FINAL_MODEL_AND_NORMALIZER.npz`:62,640B, producer SHA`b1bdfb84a928256601759e1b6e211f4b1e34884494174cffc2c42263b76b40cc`. Capture checked stat and producer receipt only; no payload opening/rehash/copy to Mac. No predictions in the archive or JSON.
