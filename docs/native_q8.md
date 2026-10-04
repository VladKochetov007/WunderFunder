# Experimental native q8 inference

`q8r` stores recurrent matrices as signed bytes; `q8x` also stores the first
head matrix as separate hidden/projection byte blocks. Both retain float32 input
projection, final output layer, biases and recurrent state. Recurrent matvecs use
transient int16 activation codes, so they add a distinct approximation beyond
weight-only PTQ/QAT. Each mode has a `_portable` sibling with the same fixed
reduction order and rounding.

The dtype contract is NumPy `int8` codes in [-127,127] and NumPy `float32` row
scales. Keys match checkpoint `quantization.encoded`: for example
`gru.weight_ih_l0.0.codes`, `gru.weight_ih_l0.0.scales`, and
`head.0.weight.0/1.codes/scales`. Preserved tensors are imported through
`PackedQ8::from_rows`; the final runtime representation retains their exact
bytes, including scales that cannot be recovered by quantizing reconstructed
float weights again.

```python
from wnn.quantization_export import read_q8_export, write_q8_export

# For a QAT checkpoint, convert each encoded CPU tensor to NumPy without a
# dtype cast, and pass that mapping as encoded. No quantizer refit is performed.
write_q8_export(path, inference_weights, preset, encoded=preserved_codes)
weights, codes, preset = read_q8_export(path)
precision = {"recurrent": "q8r", "recurrent_head": "q8x"}[preset]
predictor = wnn_native.EnsemblePredictor(
    [weights], [1.0], precision=precision, q8_codes=[codes], budget_s=None
)
assert predictor.quantization_codes()[0].keys() == codes.keys()
```

Export the two target outputs in `head.2.weight/bias` before using the ensemble
interface, as in the existing checkpoint exporter. The encoded matrices do not
include that final output layer. `weight_bytes` reports all streamed matrix and
bias arenas; `q8_weight_bytes` reports the actual padded byte panels and scales.
The NPZ export omits duplicate float matrices for the quantized fields.

E416's fixed gates compare against **the represented-weight float reference**:
max absolute prediction difference <=3e-4; absolute pooled WP difference <=5e-6;
bitwise reset, portable/native and v2/v3 predictions. Weight degradation against
the original model is a separate measurement, reported by E411. Neither the
E411 float-compute pilot nor a matrix-only microbenchmark establishes recurrent
byte-kernel performance. E416 times the actual merged D256 and trained one-head
W384 payloads against q16x, preserving five alternating-order trials and current
workload snapshots. These host-built libraries require a scorer-image build and
package verification before deployment.
