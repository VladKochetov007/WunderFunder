# October 17: portable GPU experiment preparation

Latest steering: the user reports ample GPUs, asks to consider Transformers, and explicitly prohibits local tests. [Architecture wave proposal](../unified/ideas/TRANSFORMER_GPU_WAVE_20261017_R1.md) takes research priority: one scratch GRU control, two causal-window Transformer contexts, one selective-SSM alternative. New Transformer/SSM backends are not implemented or runnable; the four existing GRU recipes below remain an optional fallback. No GPU address/allocation is established and no new fit has run.

[Four-experiment queue](fast_gpu_queue_r1/README.md) prepares fixed S/W/F comparisons and one matched cold H512/H256 seed4 pair. This request authorizes source changes, input preflight and queue preparation; GPU fitting is for a later stage. No historical source-host admission barrier is changed.

[Four-arm architecture GPU packet](architecture_gpu_wave_r1/README.md) locks the scratch GRU, two causal Transformer windows and Mamba-2 comparisons, with a target-host CUDA preflight and source/data provenance gates. It is prepared but has no target-GPU validation or fit results; this Mac made source edits only.

[Implementation and evidence boundary](../unified/operations/FAST_GPU_SCREENING_PIPELINE_20261017_R1.md) and [team top/next hypotheses review](../unified/reviews/TEAM_TOP_AND_NEXT_TESTS_20261017_R1.md) are the unified navigation entries.
