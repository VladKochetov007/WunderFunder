# Descriptive sensitivity of the six fixed mixtures

This supplementary CPU calculation resamples the579 provided feature-successor
connected components covering all1,873 valid sequences. It does not infer time
from seq_ix or change the primary split. There are600 paired resamples with
seed20261031 and stable centered moments. Point estimates independently
reproduce the fixed-mixture report within1e-10.

| Added previous model | Weight | WP delta versus TL3 | Conditional95% interval |
|---|---:|---:|---|
| GRU |5%| -0.00003045 |[-0.00010008,+0.00004995]|
| GRU |10%| -0.00011770 |[-0.00025619,+0.00004278]|
| LSTM |5%| +0.00009909 |[+0.00000557,+0.00019952]|
| LSTM |10%| +0.00012464 |[-0.00006100,+0.00032386]|
| Transformer |5%| +0.00001740 |[-0.00009891,+0.00014430]|
| Transformer |10%| -0.00005882 |[-0.00029198,+0.00001909]|

These are posthoc descriptive intervals for fixed predictions on already
development-exposed valid, after six tested mixtures. They do not include
training variance or correct the selection of the most favorable mixture.
The5%LSTM interval is not a calibrated discovery test; the10%LSTM interval
crosses zero. Neither observation proves or rules out a useful contribution.
The report's p_le_0 is a conditional resampling fraction, not a calibrated
p-value. Keep the complete panel; do not retune weights on these intervals.

[Original report](REPORT.json) and [executed CPU script](check_fixed_mixture_sensitivity.py)
preserve grouping, hashes, seed and limitations. This calculation launches no fit
and does not establish public or private transfer.
