---
kind: review
status: complete
created_utc: 2026-10-17T12:15:27Z
updated_utc: 2026-10-17T12:15:27Z
hypothesis_ids: []
experiment_ids: []
evidence_paths: []
---

# B300 training skill search and adaptation

## Question and method

Search for reusable B300 training agent guidance before writing project instructions. On October17 inspected skills.sh, [NVIDIA's official catalogue](https://docs.nvidia.com/skills), [NVIDIA's Nsight Systems skill](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-nsight-systems/SKILL.md), and [MIT Han Lab's ncu-report skill](https://github.com/mit-han-lab/ncu-report-skill). Search terms included B300 agent skill SKILL.md training Blackwell, NVIDIA skills Nsight, and distributed-training PyTorch skills.

## Evidence and interpretation

The official catalogue covers training products including NeMo and Megatron. The inspected Nsight and Han Lab skills concern timelines and kernel reports respectively. None establishes a validated full-precision recipe for this anonymous time-series task. The project-authored [b300-training skill](../../../.agents/skills/b300-training/SKILL.md) therefore captures our precision, causality, DDP normalization and evidence requirements with primary-source references; it is not an NVIDIA-verified product. No external bundle was installed or copied wholesale.

The [B300 operational field report](https://arxiv.org/abs/2608.05944) reports16-card32.76B fine-tuning, power-based triage and an unequal-work NCCL deadlock. Its transferable lesson is to verify collective/workload invariants and inspect operational telemetry. Its throughput and GPU-hour figures are external and do not estimate our8-card training time. Abstract inspected; no replication or claim of complete paper audit.

## Search scope

The repository arxiv-search Bash helper was used under the disk-scratch/20%CPU wrapper. Initial field-prefixed syntax was rejected; plain `B300 --max-papers 3` succeeded. The relevant result was cross-checked on arxiv.org. This is a bounded source search, not a systematic review.

## Limits and decision

No B300 hardware or software stack was accessed. Verify installed APIs, FP32 kernel support and real-shape memory/time before training. Follow current [PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [CUDA precision documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html). Profiling instructions that mention sudo or privileged containers do not grant permission to change our environment. The [workflow guide](../methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) is the source of campaign requirements.

Search receipt: the repository arxiv-search Bash helper returned an invalid query for the initial field-prefixed expression because it adds its own `all:` prefix. Retried once with plain `B300 --max-papers 3`; it returned the field report and two unrelated accelerator papers. Only the relevant field-report abstract was used.
