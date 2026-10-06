# Distillation implementation review

The independent specification checklist was committed before implementation inspection. Builder commit1fca6570 was integrated as22937c9d. Critic fix006b24be was integrated asc891104b; final integrated test state1f94462e.

A real confound was reproduced: constructing31 output rows instead of23 consumed extra global PyTorch randomness before shuffled data loading. The initial same-seed test changed30 of32 row positions. The fix constructs the legacy-width model, appends eight rows inside a forked CPU RNG context, copies all original rows, and restores the random state. The corrected pair receives the same first shuffled order.

Thirteen focused tests pass in the integrated branch under1 CPU/1.5GB. Checks include source/normalized/homogeneous target arithmetic, family layout, teacher fit membership, rejection of corrupt/evaluation groups, sampler-order matching, bitwise copying of all23 warm-parent rows, and final serving export of only the first2 outputs.

These tests establish implementation properties, not a predictive gain. The existing C0 fit uses immutable legacy source69b23318. Family candidates have not trained yet. Generic trainer/cache APIs remain extensible: controller admission must bind the actual coefficient manifest and full command to SOURCE_FAMILIES.json and the preregistered slate.
