# Feature-group attention row chunking

`GroupInputProjection` attends among the 16 feature groups within each input
row. Flattened sequence rows never attend to each other, so splitting only the
flattened row axis preserves the operation. The attention module keeps its
existing four heads and uses dropout `0.0`; the model configuration and
parameter/state-dict structure are unchanged.

The production call processes at most 16,000 rows at once. This bounds the
observed `B=256, T=256` input (65,536 flattened rows) to five SDPA calls. The
largest call has 64,000 batch-head entries. The component check uses a random
output projection only to make attention gradients nonzero; it does not alter
the experiment model initialization.

## CPU parity and gradients

Run on CPU, with no CUDA device visible:

```sh
env CUDA_VISIBLE_DEVICES='' /data/epanov_s/tabdata/.venv/bin/python \
  "$PACKET/runtime/group_attention_chunk_stress.py" cpu-parity \
  --output "$RB/evidence/group-attention-cpu-parity-r1.json"
```

The check compares direct attention with forced row chunking and compares the
complete group projection forward/backward on 15 rows. The local CPU run on
PyTorch 2.14.1 produced exact output and input-gradient matches; the maximum
parameter-gradient difference was `1.79e-7` for the complete projection.

## Full-shape CUDA component check

Run only after a B300 GPU is free. This is one component forward/backward, not a
Transformer profile or fit:

```sh
env CUDA_VISIBLE_DEVICES=<one-free-GPU-index> /data/epanov_s/tabdata/.venv/bin/python \
  "$PACKET/runtime/group_attention_chunk_stress.py" cuda-full \
  --output "$RB/evidence/group-attention-cuda-full-r1.json"
```

It exercises `[256, 256, 112]` FP32 inputs under BF16 autocast and requires
finite outputs, finite input and attention gradients, and a nonzero attention
gradient. No dataset, full temporal model, optimizer, or checkpoint is used.
