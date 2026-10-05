# Blind specification critique: E478 two-layer recurrent candidate

This review was written from `E478_DESIGN.md` before inspecting any E478 implementation.

The stated cell equations match PyTorch's GRU reset-after-hidden-affine form if the gate chunks are kept in the registered `r,z,n` order:

```text
r = sigmoid(x_r + b_ir + W_hr h + b_hr)
z = sigmoid(x_z + b_iz + W_hz h + b_hz)
u = tanh(x_n + b_in + r * (W_hn h + b_hn))
n = u + alpha * sin(pi * u)
h_next = (1 - z) * n + z * h
```

In particular, the hidden candidate bias remains inside the reset multiplication. Replacing this with `W_hn (r*h) + b_hn`, moving `b_hn` outside the reset product, changing the interpolation order, or using `r,n,z` gate order would break equivalence even at alpha zero. Use both bias vectors from each layer. The custom scan must apply the same alpha to the candidate activation in both layers; layer 0's per-time outputs become layer 1's inputs, and both final hidden tensors are returned in `[2,B,64]` order.

Retain the ordinary registered two-layer `nn.GRU(64,64,batch_first=True,dropout=0)` only as the source of the exact named parameters and initialization. The custom forward must reuse the inherited input projection/normalization, initial hidden state, `[recurrent, embedding]` concatenation and head; it should replace only the GRU recurrence. Preserve every key and shape, including both layers' separate input and hidden biases. The alpha-zero custom model must match ordinary `RNNModel` outputs and final states on CPU and actual B4×256 CUDA data under the same explicit IEEE FP32 policy. Compare input gradients and every parameter gradient; compare one clipped MuonAdamW step from identical states and optimizer groups. Tolerances are preregistered and must stay fixed after any failure.

The parameter and optimizer partition gate should independently prove an exact, disjoint cover: Muon receives only `gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.weight_ih_l1`, and `gru.weight_hh_l1`, with three gate blocks per matrix. AdamW receives every other trainable tensor, including input/head matrices and all biases. Count parameters at 61,058 and compare complete initial `state_dict` keys, shapes, values, and buffers across alpha arms. Check group object identities to catch omissions or duplicate assignment.

The candidate's post-tanh transform is monotone on `[-1,1]`: its derivative is at least `1 - 0.1*pi > 0`, it fixes both endpoints, and therefore maps the interval into itself. Combined with sigmoid interpolation and a valid initial state, both layers remain bounded. This is an activation-range statement only; it does not imply long-horizon gradient stability or score gain.

The fixed training loop should share each seed's NumPy-shuffled fit-row order across alpha values, reject non-fit nodes, reset the two-layer hidden state for each B4 batch, and carry/detach both layer states through 79 contiguous chunks. The training loss stays E470 weighted MSE with `need_prediction`; neither `is_scored` nor any test component enters fitting. The six fixed fits retain identical 474 updates, Muon recipe, clipping and seed-specific initialization. Telemetry at updates 0/46/236/473 remains descriptive and fit-only.

Adversarial tests should cover gate-order and bias-placement mutations, equality of each layer's first and final step at alpha zero, stacked-state shape/order, all-finite hidden trajectories, bounds at alpha `0.1`, full-versus-chunked state carry, zero-state reset, future-input prefix invariance, exact parameter-group coverage, update changes, and exact shuffled-order receipts. The actual CUDA probe must use the registered artifact filename it writes, validate the production B4×256 dimensions, compare output/state/input-and-parameter gradients under common math policy, verify one matched clipped optimizer step, and include finite/changed updates for all six fixed seed/alpha arms. A passing probe permits only controller review of the six full fits.

Interpret any later result as a transfer test from the smaller recurrent screen to the incumbent two-layer H64 model on the previously exposed 16-component panel. Preserve both alpha arms for all three seeds; no pooled seed-generalization interval or public-forecast claim follows from this screen.
