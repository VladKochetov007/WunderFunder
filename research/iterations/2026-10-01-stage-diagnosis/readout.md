Six saved-parent OOF passes, zero fits, about103 seconds of inference queues. Ensemble-aware parent already underperforms ordinary parent by -0.000634790 pooled (folds -0.000573447/-0.000782120). H2 changes the gap by +0.000123500, conditional95%CI[-0.000900295,+0.001152724]. Overall failure cannot be attributed only to erasing a parent advantage during H2. The quiet64 interaction is negative but imprecise; it does not rescue the recipe.

| Fixed ensemble | Pooled WP |
|---|---:|
| Ensemble-aware parent seed3 | 0.7018652668 |
| Ensemble-aware H2 seed3 | 0.7018683782 |
| Ordinary parent seed3 | 0.7025000567 |
| Ordinary H2 seed3, incumbent | 0.7023796676 |
| Ordinary parent seed4 | 0.7022183300 |
| Ordinary H2 seed4 | 0.7022343699 |

Ordinary parent-H2: seed3 +0.000120389, seed4 -0.000016040. Mean of two observed score deltas +0.000052175, conditional95%chainCI[-0.000306482,+0.000441836]. This is not a two-seed blended forecast. No convincing material improvement or deployment change.

Seed4 within-fold deltas are both slightly positive (+0.000029676/+0.000002307), while pooled delta is negative. Tiny effects are sensitive to pooling differently calibrated fold-model predictions. This does not prove the pooled grader wrong or permit choosing a convenient metric. All forms are retained.

Both diagnostic phases use860pure sequences/250chains/1614484 scoredrows, fixed endpoints and quarterweights,2000 common paired whole-chain draws stratifiedfold-by-class. Post-readout diagnostics on repeatedly exposed folds; two already fitted seeds are not fresh confirmation of the selected search strategy. Intervals omit training, selection and transfer uncertainty. Existing H2 profiles and row identities reproduced.

Revised strategy: prioritize discriminating low-cost interventions on saved models before expensive refits. Next concrete issue is atom CE/readout consistency (inverse amplitude tilt plus clipped atoms), including possible compensation by the old joint loss; improvement is not presumed.

[Phase1 raw report](evaluation-01/report.json), [seed4 report](evaluation-02/report.json), [phase1 preservation](preservation-phase1.json), [phase2 preservation](preservation-phase2.json). All sources/logs/six prediction arrays preserved locally and on SSH. Closed overnight archive untouched. No GitHub/public submission. Current native-qualified ZIP unchanged; materially better competition solution remains an active goal.
