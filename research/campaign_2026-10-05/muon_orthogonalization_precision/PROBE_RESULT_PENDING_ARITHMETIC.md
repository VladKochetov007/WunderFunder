# E487 actual one-step diagnostic — arithmetic verification pending

The actual sealed CUDA probe completed in 11.148837437 seconds, with 1,635,217,408 bytes peak worker RAM and 423,624,704 bytes observed GPU use. Exit was zero and teardown was verified; worker and parent OOM counters did not increase. This created zero full training fits and read only fit-role data.

The report records native/custom output and gradient parity. BF16 orthogonalization fails the strict maximum-update discrepancy gate for all four GRU matrices (and one also fails the 2% RMS criterion). FP32 orthogonalization passes all four matrix and three AdamW-family checks. The synthetic fixed-gradient clone controls pass. On this batch, the largest forward difference was 1.56e-7, maximum parameter-gradient difference 4.47e-8.

This isolates a contribution from orthogonalization precision to the one-step native/custom discrepancy. It does not establish a convergence or prediction benefit, and does not retroactively admit E485. The six-trajectory H128 FP32-optimizer study is separately registered as E489.

Independent saved-delta arithmetic has not yet run: its source review caught a float32-subtraction versus float64-subtraction mismatch before actual archive evaluation. The original verifier is being preserved and a versioned fix plus float32 regression fixture prepared. Keep this report provisional until that check passes. Original probe, arrays and strict gates stay unchanged.
