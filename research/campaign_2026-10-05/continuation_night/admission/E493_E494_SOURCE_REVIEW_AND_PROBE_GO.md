# Independent E-493/E-494 source review and control probe GO

Disposition: **GO for the shared seed-3/fold-0 control's initial full-shape
resource probe only**, after the root controller seals the draft with current
input identities and repeats the contract admission checks. This review does
not admit a scientific fit, a candidate resource probe, a score, or a submission.
The same fresh control remains the preregistered control for E-493 and E-494.

## Source review

The E-493 change is confined to the plain GRU readout activation. The fixed
mixture is parameter-free, uses the documented 0.5 GELU / 0.5 LeakyReLU(0.01)
formula, preserves initialization RNG and state-dict keys, and reconstructs from
checkpoint arguments. The default GELU output, gradients and optimizer update
match the frozen historical implementation byte-for-byte in the authored
fixture. Head-only inheritance keeps the parent's activation and rejects an
explicit conflict.

The E-494 target change applies only where `next_change > t + horizon` and the
row at `t + horizon` remains in the sequence. It mixes the original clipped
label into the final bootstrap blend at the preregistered 0.1 fraction; loss
weights still come from the original clipped labels. Invalid coefficients,
missing distillation, a zero horizon and adaptive alpha are rejected before
data loading. The ordinary trainer and pooled macro-loss helper both consume
the same target function; the independent pooled-loss fixture compares loss,
state and gradients with a separately written whole-window oracle for both
0.0 and 0.1 reanchor values.

Exporter review is deliberately restrictive. ONNX, single native export,
merged-head export and the package entry point reject a non-GELU checkpoint.
The package entry point validates members before deleting an existing package.
E-493 `fixed_mix` therefore remains research-only pending an independently
reviewed native implementation and deployment parity; these guards do not
claim native support.

I reproduced the treatment suite under one CPU, two GiB RAM and no visible GPU:

```bash
env WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G CUDA_VISIBLE_DEVICES= \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:scripts scripts/limited \
  .venv/bin/python -m pytest tests/test_training_controls.py \
  tests/test_continuation_treatments.py -q
```

Result: **28 passed in 2.11 seconds**. The recorded source-authored JUnit
receipt is `activation_transfer/cpu_tests_v2.xml` (SHA256
`ca79566270687040465be2a5c44a3aafa9fa7f2ec5578dd66d6f5b73a28c1a58`). Both
E-493 and E-494 ledger packets' eight source hashes match the current files.
The packets themselves are hash-addressed here:

| Artifact | SHA256 |
|---|---|
| E-493 ledger packet | `21785c4cf607d324a0d7f485be0a61c3dd8a99fba237eb43bbc95dc4489d9a72` |
| E-494 ledger packet | `79b336446c6c5b0a2fe39723da4a3b8423369064f845faff487eb0b5a70bff6e` |
| `src/wnn/nn.py` | `9293f8f4e2a2f3bd9f0157f8f83f12a6a758608dd03c7e141c290d82d4d5aeca` |
| `src/wnn/training_controls.py` | `67c759fdc0f053d97fe86c19e32790a8184ee671e501c7c76444dbe85137ef39` |
| `scripts/train_rnn.py` | `8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf` |
| `scripts/export_onnx.py` | `8faa1c835d6cad0eec20de03112c62b8d750ec6e380c80cc1d3b9962d3cf84b5` |
| `scripts/package_submission.py` | `cc609367b4aba93a7ba6d022fe0ad44625b1e63b898232e762c8e53532897bb2` |

## Control probe reviewed for root sealing

The shared control's runner draft SHA256 is
`2f68b4c7c7c36552a061e3f11336d28e2d791cfafcd34e78f9ebb014d2895ffb`; its
three-batch plan SHA256 is
`ea53c59dfda63371650f6ae39b81e0e2965cea4c3908dd805381695e247b85aa`.
The plan is the full registered parent recipe at seed 3 / fold 0 with profile
arguments added: B32, complete sequences of 20,000 rows, chunk 1,000, cached
since teachers at alpha 0.5 with stopped-bootstrap horizon 8, cached propensity
power 2 / floor 0.03, `wmse`, structural and next-event auxiliaries at 0.3 each,
EMA consistency weight 1.0 / decay 0.999 / EMA evaluation, and Muon on the RNN
parameters. It uses the fixed epoch-3 scheduler horizon and the component-excluded
fold-0 split. This is the actual control objective and optimizer path used by
the registered fit, not an inference-only allocator exercise.

`--profile-warmup 1 --profile-steps 2` limits this attempt to **three full
loader batches**: one warm-up plus two measured. Each batch still traverses the
full sequence in the registered 1,000-row chunks and executes the normal forward,
loss, backward, optimizer, scheduler and EMA paths. The profile exit occurs
before evaluation or checkpoint writing. The paired candidate probe must keep
these settings and all control inputs fixed, changing only its registered
treatment. The current GO covers the shared control probe; it does not authorize
the candidate probe or any of the 24 parent/head fits.

The draft reserves two CPUs and 18,000,000,000 bytes host RAM, 4,200,000,000
bytes allocator RAM, and stops on 4,800,000,000 observed accelerator bytes, with
a 900-second timeout. The device stop is below the contract's 5-GB ceiling. Root
reported approximately 36 GB current `MemAvailable`; the controller must still
recheck the eight-decimal-GB reserve, aggregate parent limits, current device
usage, source and input hashes, quota, and the local UTC start cutoff at sealing
and immediately before launch. The probe is a charged resource attempt. Its
result can inform feasibility and full-fit time estimates only; it cannot grant
full-fit admission, establish a scientific effect, or validate deployed output.

No data fit, GPU probe, provider operation, ledger write or submission was
performed by this independent reviewer. Remote packet admission remains pending
the separate cgroup-v1 guard repair and immutable packet review.
