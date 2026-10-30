# Explicit example configurations

- [historical32.example.json](historical32.example.json): the historical32-input,
  32-output graph template. Width64, four heads, one recurrent/global/local block,
  history64 and other internal sizes are illustrative assumptions; these values
  were not recovered from Denis's model.
- [connectome112.example.json](connectome112.example.json): separately labelled
  112-input, two-output adaptation. It preserves visible operations, not historical
  tensor dimensions or empirical performance.
- [sequence-runtime.example.json](sequence-runtime.example.json): selects the
  adaptation in the canonical runtime, FP32/AdamW, batch2, train-only all-fit
  normalization, all-after-warmup training mask, fixed configured epoch count and
  all-epoch checkpoint retention. Optimizer, LR, batch, epochs and normalization
  are execution examples, not an original recipe or an admitted fit.

All graph choices include mandatory source-supported and assumed detail sets.
`fusion_mode=diagram_channel_product` implements the visible post-gate product;
the four-channel bypass and operand order remain declared source uncertainties.
Neither config may be described as an exact replication or a trained checkpoint.
