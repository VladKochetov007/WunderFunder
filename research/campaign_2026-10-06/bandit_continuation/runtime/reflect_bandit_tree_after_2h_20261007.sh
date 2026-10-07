#!/usr/bin/env bash
set -uo pipefail

repo=/home/vlad/development/wunder/wnn_connectome_starterpack
output="$repo/research/campaign_2026-10-06/bandit_continuation/reviews/GROK_BANDIT_REFLECTION_AFTER_2H_20261007.md"
cd "$repo" || exit 111

prompt=$(cat <<'PROMPT'
Read AGENTS.md, specs/current-state.md, specs/research-objective.md, the active campaign allocation/frontier, current bandit-tree index/journal, and the newest experiment/admission/runtime/Kaggle receipts. Also read research/ideas/grok47_tail_risk_reframe_20261007/OPENCODE_MUSE_REPORT.md and any completed Grok reflection proposals. Use rg, never grep. This is a read-only research review; do not edit files, run training, submit, or change allocation.

The user specifically asks whether to continue researching ensembles that combine low-standalone-score but weakly correlated models, and what the multi-armed research tree says after about two hours. Analyze weighted prediction errors under the official metric, not raw prediction correlation. Explain the condition under which a convex blend can improve the champion despite a weaker standalone model; distinguish centered error covariance, scale/bias, group/target weighting, and validation selection. Propose the cheapest valid falsifier using existing independently trained predictions and exposure-aware groups, with a way to fit any blend weight on disjoint training/validation data rather than selecting it on the same scored groups. Consider residual correction, calibration, gated combinations and distillation only as separate testable descendants. Check whether current C1, E508/E510, and raw-vs-clipped work provide relevant evidence or merely consume resources.

Then recommend CONTINUE, HOLD, REFRAME or STOP for this arm and the next 1–3 allocation choices across: leading distillation, architecture/scale experiments beyond submission limits, and weak-model complementarity. Tie every claim to a file/receipt and classify missing or provider-censored runs correctly. Do not invent numerical priors, expected score gains, or a public forecast. The objective is top-1 and public score above 0.700; current scores do not cap the search. Output a concise decision memo with (1) actual evidence since the start of this session, (2) direction decision and rationale, (3) next preregisterable experiment and falsifier, (4) alternative arm retained, (5) resource/bandit consequences, and (6) evidence gaps.
PROMPT
)

{
    printf '# Scheduled bandit-tree reflection\n\nStarted: %s\n\n' "$(date --iso-8601=seconds)"
    /usr/bin/timeout --signal=TERM --kill-after=10s 600s \
        /home/vlad/.grok/bin/grok \
        --cwd "$repo" \
        --model grok-4.7 \
        --reasoning-effort high \
        --no-subagents \
        --disable-web-search \
        --permission-mode plan \
        --max-turns 8 \
        --single "$prompt"
    grok_status=$?
    printf '\n\nGrok CLI exit status: %s\nCompleted: %s\n' \
        "$grok_status" "$(date --iso-8601=seconds)"
} >> "$output" 2>&1

exit "$grok_status"
