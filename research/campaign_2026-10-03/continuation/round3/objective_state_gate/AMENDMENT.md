Before freeze, the sequence-end fixture exposed a too-tight shared tolerance:
component-gradient reconstruction differed by 2.384e-7 and the independent
rowwise oracle by 1.192e-7 in FP32. These use different summation orders.
The failed run is preserved in checks_attempt2.log/results_attempt2.json.
Repeat identity still has measured error zero and tolerance 1e-7. Only the
arithmetic-oracle comparison now uses max(identity tolerance,
4*FP32 epsilon*(1+max absolute total gradient)). This is an explicit roundoff
allowance, not a loosened identity or positive-mismatch criterion.
The initial successful run preceded the added sequence-end/hidden-state controls;
its console results are in the session but its artifact log was overwritten by
the next development run. Final artifacts alone define the frozen interface.
