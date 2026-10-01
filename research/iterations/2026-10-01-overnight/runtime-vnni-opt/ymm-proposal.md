# Final bounded YMM-only dynamic path, before timing (01:34 UTC)

All ZMM variants preserveinitialquantoutputs butmiss≤76.14µs. Lastpermittedmechanism withinstage: implement theidenticaldynamicoperator using256bitVNNI instructions,256bitinputquantization and256bit registerblockedprojection, to test whether512bitexecution/frequency effects offset dotthroughput gains. This is an unverifiedhardwarehypothesis. No quantization/scales/rounding/operator/members change.

Weights use64outputpackedlayout,64bytealignedallocation. EightYMM accumulationchains hold8outputseach, same128*rowsumcorrection and int32sums. Scaleproduct/FMA identical. Nearest-eveninputquant usesAVX2division, roundinstruction andAVX512VL conversionpack; noZMM needed in servinghotpath. Projection holds64outputs/eightYMM accumulators and repeats inputloopforeachoutputblock, retaining peroutputoriginalFMAorder. Frozenfp16 path untouched.

Require integer/rescaleoracle,tails,projectionexactoracleandfull100khash. CPU18 plus sibling58loadreceipt; same50min target. IfpassesrepeatpairedagainstbestZMM, otherwiseabort stageafterfinalreceipts; do notextendbudgetorrelaxgate. NoWP.
