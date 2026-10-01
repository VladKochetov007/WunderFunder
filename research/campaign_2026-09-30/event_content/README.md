# D1 normalized event memory

Read HANDOFF.md for currentstatus and numerical_amendment_v4.md for serialized
scaledstate/basecoefficient contract. Originalv1 longstate failure and earlier
actorproof sources are preserved under revisions/. V4 actorpasses are independent
of competitivequality; independent critic v2 numerical grade and amended integration CPU proof pass.
Fresh TRAIN feature exposure also passes the frozen support gate; independent
raw replay is requested. None of these checks establishes competitive quality.

Model.forward(x,state=None) returns output plus flat6tensor state
(GRUh,S_scaled,Z_scaled,rowcount,writecount,previousraw22). EveryrowGRU androwheads
process warmup too; externalsequencechange resets withNone. Rawfirst112columns
retained; exactcurrentf32a6index110 endpoint before normalization/autocast;
codegate current/priorrawi0prices0:22, row0false. No targets/masks/IDs/decoder.
Actualendpointoriginwritesallowed, no artificialevent; firstactualwriteordinal0.

Headhalf-lives64/512rows,2/16ownevents. h96 GRU1, q/k4x8 biasfree,values4x16tanh,
clamped[-8,8]ELU+1 positivekeys, sigmoidreadgate, head256->96->nout with skips.
Same3arms clocks/qksource only; C/Rjointwrite+decaypolicy and C/Pperiod128Walsh.
Clamps/tanh can saturate gradients. Rowheadspositionrowordinal,eventheadwriteordinal.
128x96Walshmatrix Gram128I/rank96 syntheticexposure, includes legitimateintercept.
Actual TRAIN position exposure passes on 16 fresh allowed-fit components; see
exposure/REPORT.md for exact coverage and modulo alias limitations.

Parallelblock<=64 uses bounded nonnegativepowers ofserializedbasealpha and local
64square causalweights, carryingonlyfinalS/Z. Literal reference.py uses independent
row/head arithmetic and constructedWalshbasis. FP32S/Z underouterautocast, FP64
moduleallows oraclegradients; inputrawpricesf32/countersint64. Stateboundsfromzero
areZ_scaled<=9,Sabs<=9; retrieval abs<=1. Effective weights nonnegative: D>=c*eps
normalizesconvexwrittenvalues, D<c*eps addszeroasconvexhullvertex. Arbitraryinvalid
incomingS/Z lacks this bound; carriedstate tests cover numericparity separately.

results_v4.json/run_v4.log/exit/start/end and frozen_sources_v4.json provide actor
rawproof andexacthashes. Originaldurablecommand command_v4.txt. To independently
replay withoutoverwriting immutableevidence, copy model.py/reference.py/proof.py/
preregistration.json to a newemptydirectory; run that proof.py with GPUs hidden,
WNN_THREADS=1,WNN_SLICE=wnn-campaign.slice,WNN_CPU_QUOTA=100%,WNN_MEM_MAX=2G,
WNN_RUNTIME_MAX=300 via nestedlimited/durablejob, recordingnewlogs/exits/hashes.
Historicalwrapper commands targetoriginalevidence; do not rerun overexistingfiles.

integration.patch unapplied; generictrainer forward_chunk/tupledetachcompatible.
Root must merge this separatefactorypatch with SSM/spread patches and independently
checkstrictcheckpointmetadata loading. Oldstates lacknewscaledrepresentation;
legacycheckpoints mustnotloadsilently. Native implementation notprovided.

gpu_smoke.py is runnable butUNLAUNCHED/rootonly. ReadGPUprofileprereg. It imports
GPUdependencies only on explicit execution, not in this actor's syntaxchecks.
No GPU, competitionlabels/scores orlivefiles modified. exposure/ is a completed feature-only falsifier with preserved critic/builder
authorization before TRAIN access. No targets/masks/VALID/external columns read.
