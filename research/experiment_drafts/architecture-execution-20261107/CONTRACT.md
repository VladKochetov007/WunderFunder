# Authorized B300/L4 architecture launch,7 November2026

The user now explicitly authorizes configuring monitoring and launching on
epanov_s@10.14.14.26 and lc-epanov_s@10.61.10.209. This supersedes this branch's
earlier planning-only status, not the chronological validation policy.

ROOT owns the only new SSH sessions:79138 B300 and79688 L4. The old83229 handle
is gone; no reconnect loop. Both new connections succeeded without password
entry. All8B300 GPUs are idle; L4 has no training, but foreign GPU4 services
remain. Never signal foreign processes or alter their environments. No rentals,
billing changes, old fitting resumption or unrelated filesystem cleanup.

First scientific allocation: exactly12 fresh fits, with no automatic expansion.
L4: common fresh GRU256x2 label-only control, GRU/local-attention, compact Mamba-2,
causal EMA/TabM-style regressor; each seeds101001/101002,12epochs/fixed final.
All L4 arms use the same declared input view/masks/supervised objective, AdamW
recipe and schedule; no inherited teacher, initializer or KD is substituted.
Parameter/compute costs differ by architecture and are measured, not equated.
An unavailable adapter is an execution issue, not permission to substitute a
different architecture or relabel its result.

B300: PR67's label versus fixed90%cleanTL3+10%Transformer objective, two seeds,
four30-epoch teachers. Reuse qualified train-only TL3 reference assets. No new
teacher dependencies/student fits are silently counted as zero-cost. Runtime
packing/precision changes require explicit new config identities and matched
arms, baseline replay and receiving profiles; do not edit old frozen sources.
The Mamba/patch large-teacher follow-ups are not part of this first12-fit wave.

Fit only the frozen qualified official train10325 membership;0valid fitting;
evaluate full official valid1873/3,528,273 scored rows. Full sequences20k/112raw;
2k is TBPTT, not sequence length; event211 is112raw+99derived. Train mask is all
after99; valid scoring uses need_prediction AND is_scored while every row advances
state. No random-fold primary validation, inferred seq_ix chronology, valid-fitted
statistics, parents or blend weights. Valid is already development-exposed.

Frozen comparison/model configs and source/ancestry identities must precede fits.
Receiver-owned profiles include actual shape, dtype, device, finite gradients,
peak memory and successful updates. Root admits successful reviewed profiles.
Builders do no SSH/GPU work; independent criticism precedes scientific launch.

Both nodes get durable bounded workers, stage/heartbeat/per-device/rank monitoring,
full-valid evaluations, raw journals and standalone graph collection. Persist
checkpoint/config/source/membership identities and honest failures. Candidate
quality is the fixed final endpoint and TL3 blend, not falling loss or GPU busy.

Root will choose receiving storage/CPU/RAM and lifetime from the actual inventory
and measured profile. Initial engineering preparation is bounded; use a finite
24h receiving allocation from first actual profile, with1h collection reserve,
unless actual measured full-fit cost needs an explicitly documented reassessment.
No arbitrary partial fit is relabelled a completed12/30-epoch comparison. Deadline
stops only receipt-owned jobs and preserves partial evidence. Numerical gain and
wall time remain unknown until measured. L4/data2 currently has only5.85GB free;
use a fresh scoped output placement with adequate real free capacity, not guessed
free space or deletion of historical raw artifacts.
