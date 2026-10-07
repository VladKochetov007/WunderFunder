# Directed primary source: ensemble ambiguity

Anders Krogh and Jesper Vedelsby, “Neural Network Ensembles, Cross Validation,
and Active Learning,” *Advances in Neural Information Processing Systems 7*,
1994, pp. 231–238. Primary paper: <https://papers.nips.cc/paper/1001-neural-network-ensembles-cross-validation-and-active-learning>

The paper studies real-valued regression with a convex ensemble. It defines
member disagreement around the ensemble prediction (“ambiguity”) and derives a
squared-error decomposition: ensemble error equals the weighted mean member
error minus ambiguity. It also notes that an individual member can optimally
receive zero weight when it is poor or too correlated with the rest, and that
disagreement can guide which inputs to label in an active-learning setting.
These statements are specific to the paper's setup and squared-error objective.

Transferable question: does disagreement identify examples or regimes where an
independently trained specialist contributes target-aligned correction after
conditioning on the anchor? The competition's official metric is clipped
weighted Pearson, so the paper's decomposition does not establish a gain under
`global_wp_clip2_v1`; every candidate must be graded with the frozen official
metric. At serving, the callback receives one current feature row per call. A
model may retain earlier rows in its own causal state, but the callback receives
no targets or scoring mask, so a gate cannot use test-time target residuals.

The source passage does not establish that model disagreement is a calibrated
uncertainty estimate for this task, and it does not support using the lowest
prediction correlation as a standalone selection rule.
