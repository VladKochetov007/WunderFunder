# Initial real-model deployment checks

[Frozen manifest](MANIFEST_R1.json) binds the clean-T3 H256 CP5 F0 seed3 checkpoint, native library, evaluator sources, split membership and four actual weight archives. Large payloads/predictions remain in `artifacts/qat_renewable_r1/deployment_r1`; this manifest records access paths and hashes.

[Profile summary](PROFILE_SUMMARY_R1.json) records two fixed fitting components, each with complete20,000-row callback trajectories and exact reset/replay. Native Q8 and its packed transport have identical prediction-file hashes. Weight archives: FP32 3,628,746 bytes, Q8 1,086,824, uniform W4 678,883 and mixed inputW4/recurrentW8 882,966. Quantized policies all stream1,071,552 resident weight bytes because Q4 unpacks once into byte kernels. This is archive compression evidence, not INT4 execution or runtime savings.

[Recurrent-state diagnostic](STATE_DIAGNOSTIC_R1.json) samples20 points per complete fitting trajectory, with shape2 sequences ×20 times ×2 layers ×256 hidden units. Original nativeQ8 and packedQ8 states are exact; each compressed policy's error relative to nativeFP32 is recorded by time/layer. This is a fitting-only numerical diagnostic, not held-out quality.

The profile timings are local CPU observations with other bounded work present; two fitting components do not certify EPYC or competition runtime. Full-fold comparisons, QAT/control fits, independent result review and cross-fold/seed validation remain required. Related H-QAT-W4-20261013 / E-QAT-DEPLOYMENT-20261013-R1. [Campaign](../README.md).
