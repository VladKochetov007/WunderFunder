# ADR0036: QAT research with renewable local and Kaggle capacity

Status: Accepted, 2026-10-13. Basis: the user's instruction to implement the supplied QAT plan and save it as indexed Markdown.

## Decision

Apply the [QAT plan](../../research/campaign_2026-10-13/qat_renewable_r1/PLAN.md) and [contract](../../research/campaign_2026-10-13/qat_renewable_r1/CONTRACT.json). There is no campaign-wide GPU-hour ceiling; remove the proposed 4/2/2-hour allocations. Use actual local GPU capacity and all provider-enforced Kaggle GPU/TPU allowance across renewals while research remains active. This continues ADR0034's GPU authority and gives QAT/compression the current implementation priority.

Preserve healthy admitted work, one local GPU worker,5 aggregate CPU cores,17GB committed memory target,18/20GB RAM/no swap, provenance and correct evaluation. No paid rentals, top-ups, billing changes or public upload are authorized. Resource profiles and per-job checkpoint/collection windows remain mandatory safeguards rather than campaign spending ceilings.

## Scientific scope

Begin with exact native deployment baselines and matched uniform recurrent W4 versus mixed inputW4/recurrentW8 weight-only QAT; retain Q8 first head and FP32 state. Activation-aware and learned-scale training are separate controlled extensions. Quantized transport size does not imply INT4 execution or a CPU gain. Use native served endpoint measurements and clean teacher/initializer ancestry, exposed F0/F3 seeds3/4 followed by four-fold finalists. Preserve separate evaluation-sampling, fitting-seed and transfer uncertainty; keep the official metric unchanged.

## Consequences

Record quota and recover artifacts before provider exhaustion; continue useful local work and resume cloud work on renewal. Free capacity does not require uninformative repetition. Update append-only allocations/results and the unified library. Existing strategies, frozen recipes, results and historical stop receipts are preserved; this decision supersedes contrary QAT time allocations for new work only.
