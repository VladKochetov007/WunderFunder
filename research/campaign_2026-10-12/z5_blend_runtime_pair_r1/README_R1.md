# Paired C/E callback-cost review (R1)

This receipt measures the exact integrated C and E Q8VNNI ZIPs on the same 10 complete validation sequences, with two reset-separated passes each (400,000 callbacks). It computes no metric and supports no quality claim. C-before and C-after per-sequence prediction hashes match exactly.

The paired callback-cost ratio is 1.3911 for E versus C. Applied to C's controller-reported 2,196-second elapsed runtime, that yields a 50.92-minute capacity heuristic. It sits below the 55-minute caution line by about 4.08 minutes, but is not an EPYC/Kaggle timing guarantee: callbacks were timed after sequence decode on an Intel i9-11900H, with dynamic clocks and no thermal telemetry. Do not infer quality from these measurements.

The JSON receipt records package/build hashes, source runner pins, workload identity, load and cgroup observations, and all three raw profile hashes. The raw profiles are preserved under `profiles/`.
