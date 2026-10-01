# Guarded timing correction, before repeats

The newest block128 result119.72µs had large per-sequence variation; immediately afterward CPU22's SMT sibling62 was56% busy. CPUtime/wall≈1 does not detect competition on the same physicalcore. Preserve every old run (initial89.93, packed85.49, register81.07, block128119.72) and original abort decisions; these are actual observed costs but poor controlled comparisons.

Choose CPU18 because both18/58 wereidle in the independent /proc/stat sample and critic already owns this benchmark core; wait for critic release. This choice is based on load, not best observed latency. Run the four immutable variants initial/packed/register/block128 in forward thenreverseorder A,B,C,D,D,C,B,A, separateprocesses, same100krawinputs andsame¼coefficients. Record allCPU busyfraction overeachrun. Precondition for cleanpairedcomparison: sibling58 busy<5% for eachrun. Ifnot, retainreceipt and flag it; no attempt to stop unrelatedwork. At mostoneextra cleanrepeat pervariant is permitted withinstagebudget, on samecore afterloadclears, to resolve an explicitlyflagged run.

Require unchanged rawoutputhash for everyvariant. Strong performancecondition: bothclean repetitions ofoneoperator≤76.14µs. Wholehost projections remainconditional, notpublichardwareguarantees. This timing correction does not permit WP or altering quantization/member composition.
