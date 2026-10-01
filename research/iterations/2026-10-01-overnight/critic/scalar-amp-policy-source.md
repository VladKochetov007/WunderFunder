# AMP policy supporting the prospective preflight correction

Independently opened the [PyTorch 2.14 AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) on 2026-10-01. The documentation describes unscaling before clipping, checking gradients for nonfinite values, and skipping an optimizer update when appropriate while updating the scaler. This supports preserving the ordinary trainer's skip/backoff semantics; it does not establish that this model recovers within three warmup attempts.

The diagnostic replay, separately audited in `scalar-diagnostic-actual-review.json`, observed finite loss with an infinite global gradient norm and a lower scale. Only a prospectively bounded replay of the unchanged computation can establish whether its later fixed steps are healthy. No change to model, loss, initial scale, fixture, optimizer or efficacy gates follows automatically.
