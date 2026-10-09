# Independent ideation assignment

Role: no-fit research ideator. Work from this brief only. Do not inspect other
weak-model scout cards or result artifacts. Commit distinct proposals before
reading other agents' ideas.

## Objective and constraints

Find ways independently trained, individually low-scoring but genuinely diverse
models could improve a strong sequence predictor and contribute to a top
competition result. Train all of our weights independently. Use allowed inputs
only. Inference must be deterministic and causal: one current feature row per
callback, with allowed causal internal history. The official endpoint is
`global_wp_clip2_v1`: pooled clipped weighted Pearson correlation, with ±2 label
and output clipping and the original target-magnitude weights. A weak standalone
score or low prediction correlation alone does not imply useful complementarity.

Same-pool fixed scalar/convex blends and a target-wise simplex already selected
the anchor on previously exposed groups. Do not re-tune those or repeat the
currently preregistered single rowwise disagreement-gate screen. That gate has
not yet been scored. Current campaign priority is independently trained
own-baseline/raw-teacher recovery on Kaggle. E-508-v5 profile-only execution
projected beyond its worker time cap and did no fit; E-463-v2 is queued with no
device result. Provider status is execution evidence, not model evidence. Local
aggregate limits are 10 CPU cores, 20 GB RAM and 5 GB VRAM; no new local model
fits after 10:00 Europe/Kyiv, and after that use Kaggle only.

## Assigned sources

Directed primary source: Krogh and Vedelsby, “Neural Network Ensembles, Cross
Validation, and Active Learning,” NIPS 7 (1994), pp. 231–238. Read for a small
transferable mechanism or diagnostic; the headline method need not apply.

Random library source: the recorded selection in `SOURCE.json`. Read the
selected page itself. Try to transfer a small component from its VC-dimension
discussion, but do not force an analogy if no defensible one exists.

## Deliverable

Return at least four genuinely distinct hypotheses, including one unconventional
interaction. For each, give its mechanism family; exact intervention and how
it differs from direct blending; source component and transfer reasoning;
inference-legal inputs; predicted observable/sign; cheapest discriminating test
and falsifier/controls; group-safe ancestry/evaluation plan; rough compute and
reusable artifact; and whether it should outrank, follow, or remain below the
active own-teacher recovery arm. Identify when the bandit tree should gain a new
branch or interaction node. Preserve unknown numerical priors. Do not propose
public-score expectations, fits, code changes or experiment launches. Do not
write files. Return proposals only so the controller can preserve them before
sharing any competing ideas.
