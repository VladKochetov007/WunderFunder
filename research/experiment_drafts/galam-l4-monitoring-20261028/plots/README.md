# Dashboard browser captures

- [Desktop overview](dashboard-overview.png): actual offline dashboard rendered
  by the installed Edge browser, with two selected synthetic configs, measured
  demo points, config filtering and the explicit non-scientific banner.
- [Mobile capture](dashboard-mobile.png): the same demo at420px viewport, after
  fixing long-run/metric-label overflow. The browser check verifies no horizontal
  overflow; these are interface checks, not competition predictions.

Source scripts ran against local services only. [UI_CHECK.json](../UI_CHECK.json)
records successful live append/refresh, filtering, synthetic labels and no-data
behavior. [SPARSE_UI_CHECK.json](../SPARSE_UI_CHECK.json) verifies the six sparse
evaluation points survive a12000-event training tail. No model was fitted.

[Historical mobile-overflow capture](../UI_OVERFLOW_BEFORE_FIX.json) preserves the
failed layout observation before the d430dbf6 fix. The successful UI receipt's
`mobile_horizontal_overflow=false` refers to the corrected layout; this earlier
negative observation is retained rather than silently discarded.
