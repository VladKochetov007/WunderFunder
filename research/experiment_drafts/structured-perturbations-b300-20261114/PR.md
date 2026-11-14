## 1. Status, lineage and owner

Implement structured training-time missingness for a causal Transformer, with
matched independent-position perturbations and fresh plain controls. Preparation
owner: Ivanhacker13/Codex. H/E-B300-STRUCTURED-PERTURBATIONS-20261114. Teacher-quality
screen; no students or targets from pretrained teachers. Ten configs and native
B300 stage commands are prepared. **Zero local tests, inference, training or
profiles; zero remote fits.** Receiver numerical acceptance remains mandatory.

## 2. Mechanism and falsifier

Correlated redundant observations may let independent masks preserve shortcuts.
Stale i1 blocks force use of the main stream; old-key blocks make attention use
distributed history. Compare each to equal-size independent omissions and clean
training. It can fail because i1 freshness or long memory contains real signal.
Different stale ages and removed attention mass limit causal interpretation.
This is a concrete new panel, not a claim that temporal masking is novel.

## 3. Compute advantage

Use the native 24×768 Transformer to test robustness at two representations before
distilling a teacher. B300 buys complete matched E4 endpoints, not a guessed LB
gain or extra parameter count. Profile the added attention mask separately; no
performance optimization is included and no training is started by publication.

## 4. Frozen comparison

PR75 source `8626000a4b965763a70011a0291a891aa6535d7e`; isolated receiver source
patch only. Width768/layers24/heads12/context1024, chunk256, gradient horizon256,
batch128/rank, accumulation4, two B300, BF16/BF16x9, Muon+AdamW, LR3e-4,
5% warmup/cosine0.1, clip1, E4/3280 updates. Label WMSE, fitting-only original
normalization, no KD. Exact configs, source manifest and checkpoint policy are
included. Eval bypasses perturbation and keeps original checkpoint shapes.

## 5. Bounded panel

`clean`, `stale_iid`, `stale_burst`, `history_iid`, `history_burst`, each on seeds
208401/208402. Input screen:6 new fits. History screen:4 new fits plus2 exact
same-source shared anchors. Ten total, no historical control replacement, zero
student fits. Every future admission needs owner, allocation ID, resource limits
and timeout; publication supplies no server access or compute authorization.

## 6. Validation and uncertainty

Full official valid, independent fixed-E4 export/scorer, official WP; two burst−clean
primary contrasts plus four diagnostic comparisons. Preserve both seed deltas and
shared component-bootstrap intervals. Two seeds on one exposed split cannot
establish significance, seed population variance or public transfer. No best epoch
selection, ensemble sweep, new leaderboard query or claim of a fresh holdout.

## 7. Leakage and coverage

10325 train-source fitting sequences, linked-group exclusions, zero valid fitted;
1873 evaluation sequences and3528273 scored rows. Original split hash is pinned.
i1 replacements come only from last observed same-sequence rows; row0 is observed.
History masks preserve causal positions/current keys/recent32 cached keys. Targets
and objective denominators are unchanged. No inference-time corruption or
label-dependent augmentation. Prepared probes check these properties on B300.

## 8. Decision-changing result

Mean paired burst−clean WP≥+.001 and both seeds positive justifies a separately
allocated confirmation. Burst−IID must be positive to support the structural
explanation. Mixed/small deltas are inconclusive; clean-quality losses are useful
negative evidence for this severity. Conditional CI crossing zero alone is not a
stop rule. Public-score changes and downstream student benefit remain unmeasured.

## 9. Run

Follow [GPU_COMMANDS.md](GPU_COMMANDS.md): isolated prepare → source/data check →
one-B300 numerical probe → each two-B30050-update profile → explicit approval →
native fit → fixed-E4 export/scorer → paired readout. No queue manager or automatic
launch. Failed stages keep logs and cannot silently overwrite successful attempts.

## 10. Evidence

Return source/config/split/environment/owner/allocation identities; native training
loss, LR, gradient norm, timing and memory; profile/reload receipts; all epoch and
terminal receipts; model/prediction/grade hashes; complete comparison. Preserve
negative and partial outcomes. Evidence and weights stay separate; no data files
in Git. MIDAS imports only validated metadata, not unmeasured estimates.

## 11. Results and trace

Implementation only. [README](README.md) indexes the code, frozen recipes,
commands, prior-PR audit and primary sources. Unified ideas index and H/E ledgers
link this packet. Existing pipelines and live jobs are untouched. GPU results,
serving eligibility and leaderboard effect remain unknown.
