# Research-team experiment catalogue

Fourteen draft PRs are published with base **research-team**; GitHub reports every one as draft and its head SHA matches this catalogue. No experiment has been merged or fitted by this publication. The [unified inventory](../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) records provenance and limitations. Shared-policy main base: `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`.

- [Machine catalogue](CATALOG_20261017_R1.json) contains branch/source/base SHAs, exact PR links and publication status. It is navigation, not quality evidence.
- [Workflow](../unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) defines30 paired cells per scientific arm, strictFP32, source-clean evaluation and separate logs/weights archives.

| Draft | Idea | Source SHA |
|---|---|---|
| [#1](https://github.com/VladKochetov007/WunderFunder/pull/1) | Causal Transformer with output autoregression | `f772da4c35d9` |
| [#2](https://github.com/VladKochetov007/WunderFunder/pull/2) | Cold GRU width versus depth at matched compute | `9390f08af7a5` |
| [#3](https://github.com/VladKochetov007/WunderFunder/pull/3) | GRU versus Transformer teachers for the same GRU student | `510d971b5549` |
| [#4](https://github.com/VladKochetov007/WunderFunder/pull/4) | Per-field ages versus seven group clocks | `8be5f8bf9a3d` |
| [#5](https://github.com/VladKochetov007/WunderFunder/pull/5) | Function-preserving GRU widening | `6b96ed88c3a3` |
| [#6](https://github.com/VladKochetov007/WunderFunder/pull/6) | Frozen-parent output anchor | `400334f291c3` |
| [#7](https://github.com/VladKochetov007/WunderFunder/pull/7) | Learned causal initial-state family | `4c78577307e1` |
| [#8](https://github.com/VladKochetov007/WunderFunder/pull/8) | Grouped feature encoding for causal Transformer | `e239bab3a7d3` |
| [#9](https://github.com/VladKochetov007/WunderFunder/pull/9) | Past-only smoothing while retaining raw inputs | `2b664e31654f` |
| [#10](https://github.com/VladKochetov007/WunderFunder/pull/10) | Trailing geometry and circle residual feature | `d2716b197941` |
| [#11](https://github.com/VladKochetov007/WunderFunder/pull/11) | Permitted Chronos or TimesFM teacher transfer | `00692b0eeb31` |
| [#12](https://github.com/VladKochetov007/WunderFunder/pull/12) | Expand clean heterogeneous recurrent teacher cohort | `d2c92de6a8e6` |
| [#13](https://github.com/VladKochetov007/WunderFunder/pull/13) | Fixed checkpoint trajectory averaging | `1002c70913ae` |
| [#14](https://github.com/VladKochetov007/WunderFunder/pull/14) | Causal innovation residuals beyond moment features | `de2ade7dafdc` |

## Readiness

These are experiment handoffs. Existing GRU mechanisms reuse linked trainer/model code; new Transformer/geometry/smoothing code is a starter. Every recipe is marked launch_ready=false: receiving researchers must bind fold-clean assets, finish trainer/reporting integration and profile actual B300 shapes. The plan entry point prints a matrix and does not train. No wall time, numerical parity or LB gain is claimed. No tests were added/run; static lint/format and document review only.

TPU parity remains a local engineering branch, outside these B300 quality drafts. Preserve its originating campaign evidence without launching it here. Shared primitive files appearing in related PRs have identical contents; merge them once and retain all append-only ledger/README entries when resolving conflicts.

## Packet included by this draft

- [Per-field ages versus seven group clocks](field-age/README.md) contains this draft's English proposal, recipe, code bindings and explicit missing training/profile work. `E-DRAFT-FIELD-AGE-20261017-R1`; no new fit.
