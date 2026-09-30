# Cache cleanup and next quant research directions

The next research priority is **main-only head fine-tuning with an unchanged frozen GRU**. A corrected quote-event clock is the next architecture preflight; decoded queue-update dynamics is the next quant representation preflight. These are proposals, not measured improvements. The completed Transformer did not pass either of the two registered ensemble-contribution gates.

## Completed Transformer screen

The [contract](contract.json) fixed two comparisons before new scores: add the head-only or full-FT Transformer to C at weight 0.2, scaling C by 0.8. This is equal averaging of five raw member predictions before clipping. No weights, epochs, replacement arms or new seed subsets were searched.

| Fixed teacher mixture minus C | Fold 0 | Fold 1 | Pooled WP delta | Conditional chain 95% interval |
|---|---:|---:|---:|---|
| 0.8 C + 0.2 Transformer head FT | +0.000281 | −0.000049 | +0.000115 | [−0.000319, +0.000543] |
| 0.8 C + 0.2 Transformer full FT | +0.000395 | −0.000098 | +0.000126 | [−0.000372, +0.000600] |

C scored **0.702423801** on 860 pure validation sequences from folds 0/1. Both mixtures miss the registered +0.0005 spending threshold and have a negative fold-1 effect. Decision: **screen_failed for both fixed mixtures; no Transformer-student fit allocated from this screen**. This does not establish equivalence, a universal model ceiling or failure of every Transformer design.

Both evaluations used already saved predictions, two numerical threads, CPU quota 160%, an 8 GiB memory cap and a 480-second per-run limit. They completed in **15.80 seconds total**, with maximum RSS about **0.625 GiB**, and no GPU work. See the [head result](TFM-H.result.json), [full-FT result](TFM-F.result.json), [frozen input hashes and provenance](TFM-inputs.json), and [post-run identity check](TFM-postcheck.json). The new comparisons are discovery results on exposed folds; their chain intervals condition on one fitted Transformer seed and do not correct selection or estimate training variation.

## Ranked next interventions

### 1. Training: remove auxiliary objectives only during frozen-trunk head FT

`fold_ft.sh` keeps structure and next-change losses at .3/.3 even while `inp,gru` are frozen. The main outputs and auxiliary outputs still share the first nonlinear layer of the head. Those tasks can regularize the readout, but they can also compete with the final correlation objective. Existing head/full FT comparisons did not isolate this choice.

The next bounded training comparison should keep the trained parent, model shape, initialization, masks, data order, scheduler and one-epoch head schedule fixed. Introduce a separate auxiliary **loss multiplier** of zero versus one; setting existing auxiliary flags to zero changes checkpoint dimensions and is not the same experiment. Start from matched seeds 3/4 and assess replacement of the corresponding h128 slots in C, not just standalone WP. A training-side gradient audit can explain priorities, but a cosine threshold is not a universal veto on the intervention. Auxiliary tasks may remain helpful even when some gradients conflict. [Du et al.](https://arxiv.org/abs/1812.02224) motivate that diagnostic, not a claimed gain here.

This has the best immediate cost profile: parent training is reused and serving shape, state and operation count are unchanged. No new head training has been launched in this round. Exact controls, thresholds, caveats and the proposed bounded audit are in [architecture-scout.json](architecture-scout.json).

### 2. Architecture: change the event branch's clock to actual decoded quote moves

The current dual-clock and event-attention models consume `EV_I0PX`, defined as **both first encoded bid and ask coordinates changing**. This is a proxy event predicate, not an explicit test that decoded best bid or ask moved. A train-fitted transition table can supply observed quote deltas where known; unknown edges must use a fixed fallback and remain counted.

First measure decoder coverage and disagreements between the two clocks, using a train-only table with verified provenance. Use direct current transition deltas; do not use a full-sequence anchoring statistic, evaluation-label class routing or future labels. Only if the clocks differ materially should one matched dual-clock intervention change the update mask while retaining the row branch, state size and recipe. It must beat both its proxy-clock control and the fixed C replacement. The previous dual-clock and event-attention negatives make this a conditional, lower-prior direction—not an excuse for an architecture sweep. Native inference/lookup cost also needs measurement.

### 3. Quant features: identify queue updates before modeling depletion/replenishment

Two books with similar current state can behave differently when one queue is replenished and another persistently depleted. Raw cumulative volume-code changes already failed here. A distinct first question is whether **increments** of anonymized volume codes admit a stable sparse decomposition into update coordinates during unchanged-price intervals.

Fit one train-only dictionary and test reconstruction/stability on other train chains before using target scores. Passing would justify checking whether signs and levels can be identified; reconstruction alone does not identify physical add/remove events or queue size. Only after that separate identification would a fixed state-by-add/remove-rate screen compare with C **and** a capacity-matched raw-code control. Price-changing or undecodable increments remain outside the dictionary's validated domain. An unnamed-coordinate predictive feature needs a separate hypothesis rather than a post-failure reinterpretation. [Queue-reactive Hawkes models](https://arxiv.org/abs/1901.08938) motivate combining book state with flow history; they do not prove that these transformed features expose their required variables. The detailed limits, independent review correction and cheaper falsifier are in [quant-scout.json](quant-scout.json).

A lower-priority quant reserve is conditional trade-mark surprise: predict an observed trade-code mark from prior flow, then use its innovation after arrival. This differs from the already failed MO0/MO1 sums, but must beat a raw mark/frequency control; mark predictability alone does not establish price predictability. No such screen ran this round.

## Metric source resolved

The current indexed organizer [data overview](https://wundernn.io/connectome/docs/data_overview) and [FAQ](https://wundernn.io/connectome/docs/faq) explicitly specify global pooled WP. This resolves the preceding iteration's source-access uncertainty without changing the frozen evaluator. [METRIC.md](../../../METRIC.md) and the root README now describe that aggregation; the imported block specification is preserved as a historical document. See [metric-source.json](metric-source.json).

## Cleanup

Local cleanup removed the task-created publishing checkout and incomplete archive worktree only after checking the published GitHub commit and retained server archives. Available local disk increased by **3,624,886,272 bytes** at measurement time. See [local plan](cleanup-local.plan.json) and [receipt](cleanup-local.result.json).

The [server cleanup plan](cleanup-plan.json) was executed exactly: **154 derived prediction arrays and 24 aliases removed, no skips or errors**, reclaiming **54,534,419,712 logical bytes (50.789 GiB)**. The cache now contains 112 regular files and eight aliases, totaling 46,303,304,669 bytes (43.123 GiB). All 120 retained entries matched their original fingerprints. Checkpoints, experiment logs, datasets, foundational training metadata, current C/V1 controls, Transformer and prior M01/I1 inputs were preserved.

See the [execution receipt](cleanup-execution.json), [post-cleanup inventory](cleanup-after-inventory.json), and [complete immutable audit](cleanup-receipt/), including per-file SHA256 recorded before guarded deletion and regeneration mappings. All 220 source checkpoints and seven generator scripts were checked before execution. Rebuilding deleted predictions costs inference compute and is not guaranteed to recreate bit-identical bytes across hardware/software. The GitHub archive remains the historical pre-cleanup snapshot; this receipt explains which derived caches are no longer present on the live server.

No new model, competition submission or public score was produced. The next allocated research work should be the head-FT intervention; architecture and quant feature ideas first need their bounded feasibility checks.
