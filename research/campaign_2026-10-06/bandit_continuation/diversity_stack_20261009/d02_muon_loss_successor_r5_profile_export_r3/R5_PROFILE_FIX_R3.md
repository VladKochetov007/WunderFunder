# R5 profile export bootstrap successor R3

R3 preserves the R2 source archive and the WMSE/Huber recipe. It corrects the
runtime admission guard to the packet's R5 kernel ID. R2's exact packet remains
preserved with the review finding in `R2_ADMITTED_GATE_FAILURE_R1.json`.

The R3 offline check runs both startup branches. First it verifies source
extraction with offline mode. Then it runs the exact sealed bootstrap with the
root receipt's admitted payload and a source-only mount. The expected stop is
the missing fold-0 asset manifest, before worker launch; this tests the
admission branch without fitting or calling a provider.

The R5 private source dataset is already READY at version 1, verified from the
source receipt in this directory. Root must review the exact R3 packet before
any kernel push. The R4 output fixture still records no Huber profile file; the
R5 producer regression's reconstructed Huber fixture is not observed data.
