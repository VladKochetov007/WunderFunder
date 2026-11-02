# Target-wise PCGrad B300 handoff

This implements PR58 card B for a cold, plain GRU trained on labels only. Both
arms use identical weighted MSE. The treatment computes each target's gradient
with the common two-target weight denominator and projects only conflicting
`inp.*` and `gru.*` gradients symmetrically. The ordinary full gradient is
preserved for readout and other auxiliary losses; their contribution to trunk
gradients is preserved too. The added backward passes cost compute. Strict FP32,
no QAT, no KD, no TD, no alternate update clock, and one GPU per process are
explicit admission restrictions. DDP is rejected. GradScaler must be disabled.
Cold initialization removes parent/teacher lineage risk but does not reproduce
TL3's distillation recipe; quality transfer to TL3 remains a separate experiment.

From the source checkout, on a provisioned Linux B300 host with the repository's
training dependencies, train/valid data under `$WNN_PROJECT_ROOT/datasets/`, and
source-qualified manifests:

```bash
export WNN_PROJECT_ROOT=/absolute/path/to/data-and-output-root
export F0_MANIFEST=/absolute/path/to/verified-f0.json
export F2_MANIFEST=/absolute/path/to/verified-f2.json
export CUDA_VISIBLE_DEVICES=0
bash research/experiment_drafts/gold-frontier-hypotheses-20261102/pcgrad/run_b300.sh
```

The script runs four fits sequentially. GPU 0 is illustrative: the receiver must
select an unused GPU. Six epochs and architecture/optimizer choices are fixed
for this cold screen; no claim of optimal hyperparameters is made. Manifest
membership, input provenance, no duplicate components and source permissions
must be verified independently before admission. The fixed final epoch and
`--no-eval-during-fit` avoid selecting on an outer fold. Fit names are
`gold_pcgrad_{f0,f2}_{control,pcgrad}`; do not reuse an existing output root.
Export and evaluate full held-out sequences using the project's source-qualified
prediction and official scorer after fitting. This script does not perform that
step and cannot claim a quality result until it has been performed.

No local tests, training, inference or model imports were performed. On the
receiver first check parser/contract acceptance, finite gradients, one-GPU memory,
small-real-shape numerical reference and prefix/reset behavior before full fits.
The four cells confound seed and fold, so they cannot establish significance.
