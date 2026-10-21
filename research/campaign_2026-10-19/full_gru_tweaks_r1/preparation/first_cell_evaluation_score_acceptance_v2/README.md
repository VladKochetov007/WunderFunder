# First-cell score acceptance helper V2

This packet prepares the root-side acceptance helper for the prospective
`computer-vision-homework-fg19-first-cell-score-v2.service` run. It rebases the
helper to the frozen scorer core V6 pin and corrects one metadata assumption:
the resolved parent of a registered leaf can differ from the resolved target's
parent. The helper still requires each leaf's accepted path, resolution, stat
identity, and link flag to match the frozen binding, and each observation's
parent path to match the lexical leaf parent. Parent resolutions must be
normalized absolute non-root paths with strict integer device/inode fields;
the four phase snapshots must remain identical. No historical parent inode
baseline is asserted.

The focused fixture uses only pinned JSON metadata: the V4 export binding, the
preserved first score attempt's admission, and its read-only parent diagnostic.
It checks the physical cache parent and the aliased datasets parent, including
`cache/chain_folds.npz`, whose resolved leaf target has a different parent.
That first score attempt failed before reading labels or prediction arrays and
remains preserved. This helper packet does not query or operate any B unit,
replay the scorer, or read model, prediction, data, cache, or array payloads.

`HELPER_V2_DELTA_PROOF_FINAL.json` records the exact V1-to-V2 AST delta and the
metadata join. `HELPER_V2_DELTA_PROOF.json` is retained as the preliminary proof
from immediately before the `//` noncanonical-parent negative case was added;
the final proof supersedes it. The helper has not received its independent
source review yet, and no score result or quality claim is made here.

Run only the new focused fixture with the repository maintenance interpreter:

```sh
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B -m unittest -v test_score_result_acceptance_v2_registered_parent.py
```

The proof script is a source/metadata-only audit. It does not import or execute
the helper or scorer:

```sh
/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python -B prove_score_acceptance_v2_delta.py
```
