# Primary references and limits

- NVIDIA verified skills catalogue: https://docs.nvidia.com/skills — product-specific NeMo/Megatron/DALI guidance; inspect applicable bundle and license before adoption.
- NVIDIA Nsight Systems skill: https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-nsight-systems/SKILL.md — timeline profiling, not a B300 training recipe.
- Han Lab kernel reporting skill: https://github.com/mit-han-lab/ncu-report-skill — kernel-level diagnostics, not end-to-end quality optimization.
- PyTorch DDP: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch CUDA precision: https://docs.pytorch.org/docs/stable/notes/cuda.html
- B300 field report, arXiv2608.05944: https://arxiv.org/abs/2608.05944 — external16-card large-model operations; measurements do not predict our8-card time-series throughput.

Project review and provenance: research/unified/sources/B300_AGENT_SKILLS_REVIEW_20261017_R1.md. No external skill bundle has been installed or copied wholesale. Check current documentation and installed APIs before code changes; links do not freeze a software environment.
