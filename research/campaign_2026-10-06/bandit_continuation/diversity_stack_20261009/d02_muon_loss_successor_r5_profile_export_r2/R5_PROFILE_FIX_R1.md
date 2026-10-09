# D02 R5 first-batch profile fixture repair

R4 reached the actual two-T4 gate. The WMSE gate completed 10/10 finite FP32
updates and wrote `profile_inputs.npz`; the consecutive Huber gate completed its
updates, reload and other pre-fit exports, then the worker failed because that
file was absent. The terminal receipt is
`../d02_muon_loss_successor_r4_admitted_r2/GPU_D02_R4_TERMINAL_OPERATOR_RECEIPT_R1.json`
(SHA-256 `546c267622e6968d318e6ff5d27bb1e4011e3766c3431d9f34420e9b3bafeba8`).
No full fits started.

The R4 producer captured the first batch only when `args.loss == "wmse"`, while
the worker required a first-batch file in each loss-specific gate directory.
R5 factors the first-epoch/first-batch condition into a shared producer helper,
so both WMSE and Huber persist their own actual pre-update inputs. The worker
also compares both gates' initialization, normalization, sample order, profile
arrays and short 10-attempt OneCycle traces before fit admission. The later
completed-fit comparison retains its 220-attempt requirement.

The offline replay uses the two collected R4 gate outputs, verifies their
receipt-bound hashes and matching pre-fit controls, then writes a **reconstructed
consumer-test Huber fixture** from the observed WMSE first-batch arrays through
the actual trainer export function. It is not represented as an observed Huber
input file: R4 did not produce one. This checks the corrected producer contract
and the paired artifact validator without rerunning either gate or a fit.

The scientific recipe and conditional-fit policy are unchanged. This candidate
is sealed for root review only; it has no execution or provider authorization.
