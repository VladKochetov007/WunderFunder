# Private D1 v5 scan proposal — before checks

The v4 synthetic GPU content arm passed ten optimizer/EMA steps without AMP skips
and used about 2.73 GB, but warm forward+teacher 0.048s and backward 0.059s imply
about 1640 seconds of core work for four full epochs. The existing 1080s/arm cap
remains conditional. No real fitting or silently increased cap.

Batch at most32 compressed tiles, each at most64 rows. All learned tensors,
normalized positive q/k maps, bounded values, half-lives, endpoint/code gates,
period128 positions, counters, scaled S/Z contents and serialized coefficients are
unchanged. Native literal row grammar remains the independently graded v4 grammar.

For tile j, compute local prefix writecounts c_j,t, total C_j and compressed drives
D^S_j=sum_t alpha^(C_j-c_j,t)*scale*gate_t*k_t*v_t;
D^Z similarly without values. Let P_j=sum_{l<j} C_l. The state BEFORE tile i is
alpha^P_i*state_in + sum_{j<i} alpha^(P_i-P_j-C_j)*D_j.
This strictly lower triangular <=32x32 kernel combines compressed states only.
Exponent differences are nonnegative; future terms are masked before use. Local
row readout then adds the carried state times alpha^c_j,t to ordinary causal tile
attention. Final state combines all completed drives with nonnegative powers.
Padded rows have gate0 on EVERY head, no value insertion and no returned output.

Memory arithmetic uses FP32 with outerautocast disabled, FP64 for the oracle.
No inverse powers, global sequence-square attention, detached memory, denominator
or key/value bound changes. Only exact-real affine composition and rounding order
change. Power underflow remains ordinary representable decay, not a clipped gate.
State, forward and gradient tolerances stay the previous prereg values.

Actor proof retains original independent literal recurrence and all three arms,
including full20kquiet/eventful. Add originalv4 scan forward/state/input/initialstate/
parameter-gradient identity checks with two-tile superblocks to exercise carry
across superblocks cheaply. Default32 full20k checks exercise grouping/padding.
One CPU, hard2GiB, sharedslice, durable300s; CUDAhidden. Independent grader follows.
CPU speed is engineering evidence only; root exclusively runs later GPU profile.
