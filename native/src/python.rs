use std::collections::HashMap;
use std::time::{Duration, Instant};

use numpy::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

use crate::rnn::{GruLayer, RnnModel, RnnWeights};
use crate::rnn_fast::{Activations, FastRnn, Precision};
use crate::state_init::{StateInitializer, StateInitializerConfig};

type Q8Codes = HashMap<String, Py<PyAny>>;
type FactorTensors = HashMap<String, HashMap<String, Py<PyAny>>>;

fn model_with_factors(
    py: Python<'_>,
    weights: RnnWeights,
    act: Activations,
    precision: Precision,
    factors: Option<FactorTensors>,
) -> PyResult<FastRnn> {
    let result = if let Some(factors) = factors {
        let mut parsed = Vec::with_capacity(factors.len());
        for (name, mut pair) in factors {
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "factor matrix needs exactly left and right",
                ));
            }
            let left = pair
                .remove("left")
                .ok_or_else(|| PyValueError::new_err("missing left factor"))?;
            let right = pair
                .remove("right")
                .ok_or_else(|| PyValueError::new_err("missing right factor"))?;
            let left = left
                .bind(py)
                .extract::<numpy::PyReadonlyArray2<'_, f32>>()?;
            let right = right
                .bind(py)
                .extract::<numpy::PyReadonlyArray2<'_, f32>>()?;
            let matrix = crate::factorized::FloatFactors::new(
                left.as_slice()?.to_vec(),
                right.as_slice()?.to_vec(),
                left.shape()[0],
                right.shape()[1],
                left.shape()[1],
            )
            .map_err(PyValueError::new_err)?;
            parsed.push((name, matrix));
        }
        FastRnn::with_factors(weights, act, precision, parsed)
    } else {
        FastRnn::with_precision(weights, act, precision)
    };
    result.map_err(|error| PyValueError::new_err(error.0))
}

fn export_factors(py: Python<'_>, model: &FastRnn) -> PyResult<FactorTensors> {
    let mut result = HashMap::new();
    for (name, matrix) in model.factor_matrices() {
        let mut pair = HashMap::new();
        pair.insert(
            "left".into(),
            PyArray1::from_vec(py, matrix.left().to_vec())
                .reshape([matrix.rows(), matrix.rank()])?
                .into_any()
                .unbind(),
        );
        pair.insert(
            "right".into(),
            PyArray1::from_vec(py, matrix.right().to_vec())
                .reshape([matrix.rank(), matrix.cols()])?
                .into_any()
                .unbind(),
        );
        result.insert(name.clone(), pair);
    }
    Ok(result)
}

fn storage_bytes(model: &FastRnn) -> HashMap<String, usize> {
    HashMap::from([
        // `factor_weights` is a labeled subset of total_streamed_weights, not an additive bucket.
        ("total_streamed_weights".into(), model.weight_bytes()),
        (
            "factor_weights".into(),
            model
                .factor_matrices()
                .iter()
                .map(|(_, matrix)| matrix.storage_bytes())
                .sum(),
        ),
        ("recurrent_state".into(), model.recurrent_state_bytes()),
        ("scratch".into(), model.scratch_bytes()),
        (
            "shared_backbone_step_payload".into(),
            model.shared_backbone_step_payload_bytes(),
        ),
        (
            "independent_head_step_payload".into(),
            model.independent_head_step_payload_bytes(),
        ),
        (
            "import_only_dense_recurrent".into(),
            model.import_only_dense_bytes(),
        ),
    ])
}

fn import_q8(py: Python<'_>, model: &mut FastRnn, mut codes: Q8Codes) -> PyResult<()> {
    if model.is_dynamic_int8() {
        let expected: Vec<_> = model.dyn8_matrices().collect();
        if expected.is_empty() || codes.len() != 2 * expected.len() {
            return Err(PyValueError::new_err(
                "q8 codes/scales do not match dynamic precision preset",
            ));
        }
        let mut matrices = Vec::with_capacity(expected.len());
        for (name, expected_matrix) in expected {
            let raw = codes
                .remove(&format!("{name}.codes"))
                .ok_or_else(|| PyValueError::new_err(format!("missing q8 codes: {name}")))?;
            let scale = codes
                .remove(&format!("{name}.scales"))
                .ok_or_else(|| PyValueError::new_err(format!("missing q8 scales: {name}")))?;
            let raw = raw.bind(py).extract::<numpy::PyReadonlyArray2<'_, i8>>()?;
            let scale = scale.bind(py).extract::<PyReadonlyArray1<'_, f32>>()?;
            let shape = raw.shape();
            if shape != [expected_matrix.rows(), expected_matrix.cols()]
                || scale.shape() != [expected_matrix.rows()]
            {
                return Err(PyValueError::new_err(format!(
                    "q8 tensor identity or shape mismatch: {name}"
                )));
            }
            let matrix = crate::vnni::DynQ8::from_rows(
                raw.as_slice()?,
                scale.as_slice()?,
                shape[0],
                shape[1],
            )
            .map_err(PyValueError::new_err)?;
            matrices.push((name.to_owned(), matrix));
        }
        model
            .load_dyn8_matrices(matrices)
            .map_err(|e| PyValueError::new_err(e.0))?;
        if !codes.is_empty() {
            return Err(PyValueError::new_err("unexpected q8 code/scale keys"));
        }
        return Ok(());
    }

    if codes.len() != 2 * model.q8_matrices().len() || codes.is_empty() {
        return Err(PyValueError::new_err(
            "q8 codes/scales do not match precision preset",
        ));
    }
    let mut matrices = Vec::new();
    for (name, _) in model.q8_matrices() {
        let raw = codes
            .remove(&format!("{name}.codes"))
            .ok_or_else(|| PyValueError::new_err(format!("missing q8 codes: {name}")))?;
        let scale = codes
            .remove(&format!("{name}.scales"))
            .ok_or_else(|| PyValueError::new_err(format!("missing q8 scales: {name}")))?;
        let raw = raw.bind(py).extract::<numpy::PyReadonlyArray2<'_, i8>>()?;
        let scale = scale.bind(py).extract::<PyReadonlyArray1<'_, f32>>()?;
        let shape = raw.shape();
        let matrix = crate::quant8::PackedQ8::from_rows(
            raw.as_slice()?,
            scale.as_slice()?,
            shape[0],
            shape[1],
        )
        .map_err(PyValueError::new_err)?;
        matrices.push((name.clone(), matrix));
    }
    model
        .load_q8_matrices(matrices)
        .map_err(|e| PyValueError::new_err(e.0))
}

fn export_q8(py: Python<'_>, model: &FastRnn) -> PyResult<Q8Codes> {
    let mut result = HashMap::new();
    if model.is_dynamic_int8() {
        for (name, matrix) in model.dyn8_matrices() {
            let (codes, scales) = matrix.to_rows();
            result.insert(
                format!("{name}.codes"),
                PyArray1::from_vec(py, codes)
                    .reshape([matrix.rows(), matrix.cols()])?
                    .into_any()
                    .unbind(),
            );
            result.insert(
                format!("{name}.scales"),
                PyArray1::from_vec(py, scales).into_any().unbind(),
            );
        }
        return Ok(result);
    }
    for (name, matrix) in model.q8_matrices() {
        let (codes, scales) = matrix.to_rows();
        result.insert(
            format!("{name}.codes"),
            PyArray1::from_vec(py, codes)
                .reshape([matrix.rows(), matrix.cols()])?
                .into_any()
                .unbind(),
        );
        result.insert(
            format!("{name}.scales"),
            PyArray1::from_vec(py, scales).into_any().unbind(),
        );
    }
    Ok(result)
}

fn export_hidden_state<'py>(
    py: Python<'py>,
    model: &FastRnn,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (layers, hidden) = model.hidden_state_shape();
    let state = PyArray1::from_vec(py, model.hidden_state().to_vec()).reshape([layers, hidden])?;
    // The returned array is a detached snapshot and is also marked read-only for callers.
    state.call_method1(intern!(py, "setflags"), (false,))?;
    Ok(state)
}

fn parse_precision(name: &str) -> PyResult<Precision> {
    match name {
        "f32" => Ok(Precision::F32),
        "f16" => Ok(Precision::F16),
        "i16" => Ok(Precision::I16),
        "q16x" => Ok(Precision::Q16x),
        "q16x_portable" => Ok(Precision::Q16xPortable),
        "q8r" => Ok(Precision::Q8r),
        "q8r_portable" => Ok(Precision::Q8rPortable),
        "q8x" => Ok(Precision::Q8x),
        "q8x_portable" => Ok(Precision::Q8xPortable),
        "q8vnni" => Ok(Precision::Q8Vnni),
        "q8vnni_avx2" => Ok(Precision::Q8VnniAvx2),
        "q8vnni_portable" => Ok(Precision::Q8VnniPortable),
        "q8vnni256" => Ok(Precision::Q8Vnni256),
        "q8vnni_blocked" => Ok(Precision::Q8VnniBlocked),
        "q8vnni_fused" => Ok(Precision::Q8VnniFused),
        "factor_f32" => Ok(Precision::FactorF32),
        "factor_f32_portable" => Ok(Precision::FactorF32Portable),
        other => Err(PyValueError::new_err(format!(
            "precision {other:?}; expected f32, f16, i16, q16x, q8r, q8x or a portable sibling"
        ))),
    }
}

fn precision_name(p: Precision) -> &'static str {
    match p {
        Precision::F32 => "f32",
        Precision::F16 => "f16",
        Precision::I16 => "i16",
        Precision::Q16x => "q16x",
        Precision::Q16xPortable => "q16x_portable",
        Precision::Q8r => "q8r",
        Precision::Q8rPortable => "q8r_portable",
        Precision::Q8x => "q8x",
        Precision::Q8xPortable => "q8x_portable",
        Precision::Q8Vnni => "q8vnni",
        Precision::Q8VnniAvx2 => "q8vnni_avx2",
        Precision::Q8VnniPortable => "q8vnni_portable",
        Precision::Q8Vnni256 => "q8vnni256",
        Precision::Q8VnniBlocked => "q8vnni_blocked",
        Precision::Q8VnniFused => "q8vnni_fused",
        Precision::FactorF32 => "factor_f32",
        Precision::FactorF32Portable => "factor_f32_portable",
    }
}

/// Copies a raw 112-feature row into `dst`. float32 contiguous arrays (what the scorer passes)
/// are read in place; anything else goes through numpy.ascontiguousarray(.., float32).
fn read_row(py: Python<'_>, row: &Bound<'_, PyAny>, dst: &mut [f32]) -> PyResult<()> {
    let copy = |src: &[f32], dst: &mut [f32]| {
        if src.len() != dst.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} features, got {}",
                dst.len(),
                src.len()
            )));
        }
        dst.copy_from_slice(src);
        Ok(())
    };
    if let Ok(array) = row.cast::<PyArray1<f32>>() {
        // SAFETY: the slice lives only for this copy, under the GIL, and nothing in this
        // extension holds a mutable view of a caller's array.
        if let Ok(src) = unsafe { array.as_slice() } {
            return copy(src, dst);
        }
    }
    let converted = py.import(intern!(py, "numpy"))?.call_method1(
        intern!(py, "ascontiguousarray"),
        (row, intern!(py, "float32")),
    )?;
    let array = converted.cast::<PyArray1<f32>>()?;
    copy(array.readonly().as_slice()?, dst)
}

fn take(w: &mut HashMap<String, Vec<f32>>, key: &str) -> PyResult<Vec<f32>> {
    w.remove(key)
        .ok_or_else(|| PyValueError::new_err(format!("missing weight {key}")))
}

fn weights_from_dict(
    weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
) -> PyResult<RnnWeights> {
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    for (k, v) in weights {
        let arr = v.as_array();
        w.insert(k, arr.iter().copied().collect());
    }
    let mut layers = Vec::new();
    while w.contains_key(&format!("gru.weight_ih_l{}", layers.len())) {
        let k = layers.len();
        layers.push(GruLayer {
            w_ih: take(&mut w, &format!("gru.weight_ih_l{k}"))?,
            w_hh: take(&mut w, &format!("gru.weight_hh_l{k}"))?,
            b_ih: take(&mut w, &format!("gru.bias_ih_l{k}"))?,
            b_hh: take(&mut w, &format!("gru.bias_hh_l{k}"))?,
        });
    }
    Ok(RnnWeights {
        mean: take(&mut w, "mean")?,
        std: take(&mut w, "std")?,
        w_in: take(&mut w, "inp.weight")?,
        b_in: take(&mut w, "inp.bias")?,
        layers,
        w1: take(&mut w, "head.0.weight")?,
        b1: take(&mut w, "head.0.bias")?,
        w2: take(&mut w, "head.2.weight")?,
        b2: take(&mut w, "head.2.bias")?,
    })
}

fn take_initializer_value<'py>(
    py: Python<'py>,
    config: &mut HashMap<String, Py<PyAny>>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    config
        .remove(key)
        .map(|value| value.into_bound(py))
        .ok_or_else(|| PyValueError::new_err(format!("missing initializer field {key}")))
}

fn take_initializer_array(
    py: Python<'_>,
    config: &mut HashMap<String, Py<PyAny>>,
    key: &str,
    shape: &[usize],
) -> PyResult<Vec<f32>> {
    let value = take_initializer_value(py, config, key)?;
    let array: PyReadonlyArrayDyn<'_, f32> = value.extract()?;
    if array.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "initializer field {key} shape mismatch: expected {shape:?}, got {:?}",
            array.shape()
        )));
    }
    Ok(array.as_slice()?.to_vec())
}

fn take_initializer_bound(
    py: Python<'_>,
    config: &mut HashMap<String, Py<PyAny>>,
) -> PyResult<f32> {
    take_initializer_value(py, config, "residual_bound")?.extract()
}

fn reject_initializer_extras(config: &HashMap<String, Py<PyAny>>) -> PyResult<()> {
    if config.is_empty() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "unexpected initializer fields: {:?}",
            config.keys().collect::<Vec<_>>()
        )))
    }
}

fn parse_state_initializer(
    py: Python<'_>,
    mut config: HashMap<String, Py<PyAny>>,
    model: &FastRnn,
) -> PyResult<StateInitializer> {
    let mode = take_initializer_value(py, &mut config, "mode")?.extract::<String>()?;
    let (layers, hidden) = model.hidden_state_shape();
    let n_in = model.n_in();
    let initializer = match mode.as_str() {
        "zero" => {
            reject_initializer_extras(&config)?;
            StateInitializerConfig::Zero
        }
        "constant" => {
            let bound = take_initializer_bound(py, &mut config)?;
            let values = take_initializer_array(py, &mut config, "constant", &[layers, hidden])?;
            reject_initializer_extras(&config)?;
            StateInitializerConfig::Constant { bound, values }
        }
        "first_row" | "prefix" => {
            let bound = take_initializer_bound(py, &mut config)?;
            let input_width = if mode == "prefix" { 3 * n_in } else { n_in };
            let input_weight = take_initializer_array(
                py,
                &mut config,
                "predictor_input_weight",
                &[64, input_width],
            )?;
            let input_bias =
                take_initializer_array(py, &mut config, "predictor_input_bias", &[64])?;
            let output_weight = take_initializer_array(
                py,
                &mut config,
                "predictor_output_weight",
                &[layers * hidden, 64],
            )?;
            let output_bias = take_initializer_array(
                py,
                &mut config,
                "predictor_output_bias",
                &[layers * hidden],
            )?;
            reject_initializer_extras(&config)?;
            if mode == "prefix" {
                StateInitializerConfig::Prefix {
                    bound,
                    input_weight,
                    input_bias,
                    output_weight,
                    output_bias,
                }
            } else {
                StateInitializerConfig::FirstRow {
                    bound,
                    input_weight,
                    input_bias,
                    output_weight,
                    output_bias,
                }
            }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown initializer mode {mode:?}"
            )));
        }
    };
    if mode != "zero" && model.cell() != crate::rnn_fast::Cell::Gru {
        return Err(PyValueError::new_err(
            "state initialization requires a GRU backbone",
        ));
    }
    StateInitializer::new(initializer, n_in, layers, hidden).map_err(PyValueError::new_err)
}

fn same_float_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn same_shared_backbone(left: &RnnWeights, right: &RnnWeights) -> bool {
    left.layers.len() == right.layers.len()
        && same_float_bits(&left.mean, &right.mean)
        && same_float_bits(&left.std, &right.std)
        && same_float_bits(&left.w_in, &right.w_in)
        && same_float_bits(&left.b_in, &right.b_in)
        && left.layers.iter().zip(&right.layers).all(|(left, right)| {
            same_float_bits(&left.w_ih, &right.w_ih)
                && same_float_bits(&left.w_hh, &right.w_hh)
                && same_float_bits(&left.b_ih, &right.b_ih)
                && same_float_bits(&left.b_hh, &right.b_hh)
        })
}

fn validate_shared_backbone_pair(
    weights: &[RnnWeights],
    coefficients: &[f32],
    precision: Precision,
    has_q8_codes: bool,
    has_factors: bool,
    has_governor: bool,
) -> PyResult<()> {
    if weights.len() != 2 {
        return Err(PyValueError::new_err(
            "shared_backbone requires exactly two models",
        ));
    }
    if coefficients.len() != 2
        || coefficients
            .iter()
            .any(|coefficient| coefficient.to_bits() != 0.5f32.to_bits())
    {
        return Err(PyValueError::new_err(
            "shared_backbone requires fixed 0.5/0.5 output averaging",
        ));
    }
    if weights.iter().any(|member| {
        member.b2.len() != 2 || member.w2.len() != 2 * member.b1.len()
    }) {
        return Err(PyValueError::new_err(
            "shared_backbone requires two-output heads",
        ));
    }
    if !matches!(
        precision,
        Precision::Q8Vnni | Precision::Q8VnniAvx2 | Precision::Q8VnniPortable
    ) || has_q8_codes
        || has_factors
        || has_governor
    {
        return Err(PyValueError::new_err(
            "shared_backbone requires dynamic Q8VNNI without imported codes, factors or governor",
        ));
    }
    if !same_shared_backbone(&weights[0], &weights[1]) {
        return Err(PyValueError::new_err(
            "shared_backbone rejected non-identical input, normalization or recurrent tensors",
        ));
    }
    Ok(())
}

/// Stateful twin of wnn.nn.RNNModel. Construct from `model.state_dict()` arrays.
#[pyclass(name = "GruStepper")]
struct PyGruStepper {
    inner: RnnModel,
}

#[pymethods]
impl PyGruStepper {
    #[new]
    fn new(weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>) -> PyResult<Self> {
        let weights = weights_from_dict(weights)?;
        let inner = RnnModel::new(weights).map_err(|e| PyValueError::new_err(e.0))?;
        Ok(Self { inner })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x = x
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if x.len() != self.inner.n_in() {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.inner.n_in(),
                x.len()
            )));
        }
        Ok(PyArray1::from_slice(py, self.inner.step(x)))
    }
}

#[pymodule]
fn wnn_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGruStepper>()?;
    m.add_class::<PyPredictor>()?;
    m.add_class::<PyEnsemblePredictor>()?;
    m.add_class::<PyModelStepper>()?;
    Ok(())
}

/// One `FastRnn` on precomputed model inputs (no stream features), for parity checks against
/// the training model. GRU or LSTM is inferred from the weight shapes.
#[pyclass(name = "ModelStepper")]
struct PyModelStepper {
    model: FastRnn,
}

#[pymethods]
impl PyModelStepper {
    #[new]
    #[pyo3(signature = (weights, fast_activations=true, precision="f32", q8_codes=None, factors=None))]
    fn new(
        py: Python<'_>,
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        fast_activations: bool,
        precision: &str,
        q8_codes: Option<Q8Codes>,
        factors: Option<FactorTensors>,
    ) -> PyResult<Self> {
        let act = if fast_activations {
            Activations::Fast
        } else {
            Activations::Exact
        };
        let mut model = model_with_factors(
            py,
            weights_from_dict(weights)?,
            act,
            parse_precision(precision)?,
            factors,
        )?;
        if let Some(codes) = q8_codes {
            import_q8(py, &mut model, codes)?;
        }
        Ok(Self { model })
    }

    fn quantization_codes(&self, py: Python<'_>) -> PyResult<Q8Codes> {
        export_q8(py, &self.model)
    }

    fn factorization_tensors(&self, py: Python<'_>) -> PyResult<FactorTensors> {
        export_factors(py, &self.model)
    }

    #[getter]
    fn storage_bytes(&self) -> HashMap<String, usize> {
        storage_bytes(&self.model)
    }

    #[getter]
    fn hidden_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        export_hidden_state(py, &self.model)
    }

    #[getter]
    fn weight_bytes(&self) -> usize {
        self.model.weight_bytes()
    }

    #[getter]
    fn q8_weight_bytes(&self) -> usize {
        self.model
            .q8_matrices()
            .iter()
            .map(|(_, matrix)| matrix.storage_bytes())
            .sum()
    }

    fn reset(&mut self) {
        self.model.reset();
    }

    /// Steps every row of `xs` (T x n_in) from the current state; returns T x n_out.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        xs: numpy::PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let n_in = self.model.n_in();
        let xs = xs.as_array();
        if xs.ncols() != n_in {
            return Err(PyValueError::new_err(format!(
                "expected {n_in} inputs, got {}",
                xs.ncols()
            )));
        }
        let n_out = self.model.n_out();
        let mut out = vec![0.0f32; xs.nrows() * n_out];
        let mut row = vec![0.0f32; n_in];
        for (x, y) in xs.rows().into_iter().zip(out.chunks_exact_mut(n_out)) {
            row.iter_mut().zip(x.iter()).for_each(|(r, v)| *r = *v);
            self.model.step(&row, y);
        }
        numpy::PyArray1::from_vec(py, out).reshape([xs.nrows(), n_out])
    }

    #[getter]
    fn precision(&self) -> &'static str {
        precision_name(self.model.precision())
    }

    #[getter]
    fn cell(&self) -> &'static str {
        match self.model.cell() {
            crate::rnn_fast::Cell::Gru => "gru",
            crate::rnn_fast::Cell::Lstm => "lstm",
        }
    }
}

/// Several GRUs with one causal feature update and one Python call per row. Members read
/// [raw 112 | event 99] (211 inputs) or, when trained with online norm, additionally the 112
/// causal per-sequence z-scores of the raw row (323 inputs); z-scores are computed only if some
/// member needs them.
///
/// `precision="f16"` / `"i16"` stores each member's input projection, GRU and head.0 matrices in
/// 16 bits (fp32 accumulation; see `rnn_fast::Precision`); CPUs without AVX2/FMA/F16C silently
/// keep fp32, which `precision` reports. `"q16x"` adds integer GRU/head.0 matvecs and runs on
/// every CPU (portable kernels without AVX2, same bits). `predict(data_point)` is the whole scorer callback in native code (reset
/// on a new `seq_ix`, None during warm-up). With `budget_s`, a governor runs at each sequence
/// boundary: when the measured cost projects past the budget for `expected_rows`, it keeps the
/// longest prefix of members that fits and rescales their coefficients to the original sum.
#[pyclass(name = "EnsemblePredictor")]
struct PyEnsemblePredictor {
    features: crate::stream::StreamFeatures,
    norm: Option<crate::stream::OnlineNorm>,
    widths: Vec<usize>,
    models: Vec<FastRnn>,
    initializers: Vec<StateInitializer>,
    /// Each member's first weights, prefetched by the member stepped before it.
    firsts: Vec<crate::half::Ahead>,
    input: Vec<f32>,
    base_coefficients: Vec<f32>,
    coefficients: Vec<f32>,
    active: usize,
    scratch: Vec<[f32; 2]>,
    /// Step members in reverse order on odd rows (dynamic-int8 ensembles; see `advance`).
    alternate: bool,
    odd_row: bool,
    shared_backbone: bool,
    out: [f32; 2],
    seq: Option<i64>,
    governor: Option<Governor>,
}

/// Wall-clock budget bookkeeping for `EnsemblePredictor.predict`.
struct Governor {
    budget: f64,
    expected_rows: f64,
    start: Option<Instant>,
    rows: u64,
    sequences: u64,
    window_start: Instant,
    window_rows: u64,
    /// Time inside each member's step during the window.
    member_time: Vec<Duration>,
}

impl Governor {
    /// Number of leading members that fit the remaining budget, or None to keep `active`.
    fn plan(&mut self, now: Instant, active: usize) -> Option<usize> {
        self.sequences += 1;
        let Some(start) = self.start else {
            self.start = Some(now);
            return None;
        };
        if self.sequences == 2 {
            // The first sequence is cold; measure from the second on.
            self.open_window(now);
            return None;
        }
        let rows = self.rows - self.window_rows;
        let remaining = self.expected_rows - self.rows as f64;
        if active == 1 || rows < 20_000 || remaining <= 0.0 {
            return None;
        }
        let allowed = (self.budget - (now - start).as_secs_f64()) / remaining;
        let per_row = (now - self.window_start).as_secs_f64() / rows as f64;
        if per_row <= allowed {
            return None;
        }
        let member: Vec<f64> = self.member_time[..active]
            .iter()
            .map(|t| t.as_secs_f64() / rows as f64)
            .collect();
        let overhead = per_row - member.iter().sum::<f64>();
        let mut keep = 1;
        let mut cost = overhead + member[0];
        while keep < active - 1 && cost + member[keep] <= allowed {
            cost += member[keep];
            keep += 1;
        }
        eprintln!(
            "[wnn] governor: rows={} elapsed={:.0}s per_row={:.1}us allowed={:.1}us members {}->{}",
            self.rows,
            (now - start).as_secs_f64(),
            per_row * 1e6,
            allowed * 1e6,
            active,
            keep
        );
        self.open_window(now);
        Some(keep)
    }

    fn open_window(&mut self, now: Instant) {
        self.window_start = now;
        self.window_rows = self.rows;
        self.member_time
            .iter_mut()
            .for_each(|t| *t = Duration::ZERO);
    }
}

impl PyEnsemblePredictor {
    fn reset_state(&mut self) {
        self.features.reset();
        if let Some(norm) = &mut self.norm {
            norm.reset();
        }
        for model in &mut self.models {
            model.reset();
        }
        for initializer in &mut self.initializers {
            initializer.reset();
        }
    }

    fn set_active(&mut self, active: usize) {
        let total: f32 = self.base_coefficients.iter().sum();
        let kept: f32 = self.base_coefficients[..active].iter().sum();
        self.active = active;
        self.coefficients = self.base_coefficients[..active]
            .iter()
            .map(|c| c * (total / kept))
            .collect();
    }

    fn initialize_member(&mut self, index: usize) -> PyResult<()> {
        if self.initializers.is_empty() {
            return Ok(());
        }
        let width = self.widths[index];
        self.initializers[index]
            .before_step(&mut self.models[index], &self.input[..width])
            .map_err(PyValueError::new_err)
    }

    /// Features for the row already in `input[..N_RAW]`, then the blend of the active members.
    fn advance(&mut self) -> PyResult<()> {
        let (raw, rest) = self.input.split_at_mut(crate::stream::N_RAW);
        self.features.sanitize(raw);
        let (event, zscores) = rest.split_at_mut(crate::stream::N_EVENT);
        self.features.update(raw, event);
        if let Some(norm) = &mut self.norm {
            norm.update(raw, zscores);
        }
        self.out = [0.0; 2];
        let active = self.active;
        // Dynamic-int8 members together exceed L2: step them in reverse on every other row so the
        // member stepped last (still cached) is first on the next row. Members are independent
        // and the blend below always sums in index order, so the output bits do not change.
        let reverse = self.alternate && self.odd_row;
        self.odd_row = !self.odd_row;
        let order = |k: usize| if reverse { active - 1 - k } else { k };
        if self.shared_backbone {
            debug_assert!(self.governor.is_none());
            let (first, second) = self.models.split_at_mut(1);
            let (first_scratch, second_scratch) = self.scratch.split_at_mut(1);
            first[0].step_with_shared_backbone(
                &mut second[0],
                &self.input[..self.widths[0]],
                &mut first_scratch[0],
                &mut second_scratch[0],
            );
        } else if self.governor.is_none() {
            for k in 0..active {
                let i = order(k);
                self.initialize_member(i)?;
                // Prefetch the next member's first weights (for the last member, the next
                // row's first) during this member's final activation pass.
                let after = self.firsts[order((k + 1) % active)];
                let width = self.widths[i];
                self.models[i].step_then(&self.input[..width], &mut self.scratch[i], after);
            }
        } else {
            for k in 0..active {
                let i = order(k);
                self.initialize_member(i)?;
                let t = Instant::now();
                let width = self.widths[i];
                self.models[i].step(&self.input[..width], &mut self.scratch[i]);
                if let Some(governor) = &mut self.governor {
                    governor.member_time[i] += t.elapsed();
                }
            }
            if let Some(governor) = &mut self.governor {
                governor.rows += 1;
            }
        }
        for (scratch, coefficient) in self.scratch.iter().zip(&self.coefficients).take(active) {
            for (total, value) in self.out.iter_mut().zip(scratch) {
                *total += *coefficient * *value;
            }
        }
        for value in &mut self.out {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(())
    }
}

#[pymethods]
impl PyEnsemblePredictor {
    #[new]
    #[pyo3(signature = (
        weights,
        coefficients,
        fast_activations=true,
        precision="f32",
        budget_s=None,
        expected_rows=39_400_000.0,
        q8_codes=None,
        factors=None,
        shared_backbone=false,
        state_initializers=None
    ))]
    fn new(
        py: Python<'_>,
        weights: Vec<HashMap<String, PyReadonlyArrayDyn<'_, f32>>>,
        coefficients: Vec<f32>,
        fast_activations: bool,
        precision: &str,
        budget_s: Option<f64>,
        expected_rows: f64,
        q8_codes: Option<Vec<Q8Codes>>,
        factors: Option<Vec<FactorTensors>>,
        shared_backbone: bool,
        state_initializers: Option<Vec<HashMap<String, Py<PyAny>>>>,
    ) -> PyResult<Self> {
        if weights.is_empty() || weights.len() != coefficients.len() {
            return Err(PyValueError::new_err(
                "provide one coefficient for each nonempty model",
            ));
        }
        if coefficients.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("coefficients must be finite"));
        }
        if q8_codes
            .as_ref()
            .is_some_and(|codes| codes.len() != weights.len())
        {
            return Err(PyValueError::new_err(
                "provide q8 tensors for every ensemble member",
            ));
        }
        let has_q8_codes = q8_codes.is_some();
        let mut encoded = q8_codes.map(|codes| codes.into_iter());
        if factors
            .as_ref()
            .is_some_and(|items| items.len() != weights.len())
        {
            return Err(PyValueError::new_err(
                "provide factors for every ensemble member",
            ));
        }
        let has_factors = factors.is_some();
        let mut factors = factors.map(|items| items.into_iter());
        let activation = if fast_activations {
            Activations::Fast
        } else {
            Activations::Exact
        };
        let precision = parse_precision(precision)?;
        let parsed_members = weights
            .into_iter()
            .map(weights_from_dict)
            .collect::<PyResult<Vec<_>>>()?;
        if shared_backbone {
            validate_shared_backbone_pair(
                &parsed_members,
                &coefficients,
                precision,
                has_q8_codes,
                has_factors,
                budget_s.is_some(),
            )?;
        }
        use crate::stream::{N_EVENT, N_RAW};
        let base = N_RAW + N_EVENT;
        let mut models = Vec::with_capacity(parsed_members.len());
        let mut widths = Vec::with_capacity(parsed_members.len());
        for parsed in parsed_members {
            let width = parsed.mean.len();
            if width != base && width != base + N_RAW {
                return Err(PyValueError::new_err(format!(
                    "model input width {width}; expected {base} or {} (online norm)",
                    base + N_RAW
                )));
            }
            widths.push(width);
            let mut model = model_with_factors(
                py,
                parsed,
                activation,
                precision,
                factors.as_mut().and_then(Iterator::next),
            )?;
            if let Some(codes) = encoded.as_mut().and_then(Iterator::next) {
                import_q8(py, &mut model, codes)?;
            }
            if model.n_out() != 2 {
                return Err(PyValueError::new_err("ensemble members need two outputs"));
            }
            models.push(model);
        }
        let needs_norm = widths.iter().any(|&w| w > base);
        let initializers = match state_initializers {
            Some(configs) => {
                if configs.len() != models.len() {
                    return Err(PyValueError::new_err(
                        "provide one state initializer config for each ensemble member",
                    ));
                }
                let parsed = configs
                    .into_iter()
                    .zip(&models)
                    .map(|(config, model)| parse_state_initializer(py, config, model))
                    .collect::<PyResult<Vec<_>>>()?;
                if shared_backbone && parsed.iter().any(|initializer| !initializer.is_zero()) {
                    return Err(PyValueError::new_err(
                        "nonzero state initialization is not supported with shared_backbone",
                    ));
                }
                parsed
            }
            None => Vec::new(),
        };
        let governor = budget_s.map(|budget| Governor {
            budget,
            expected_rows,
            start: None,
            rows: 0,
            sequences: 0,
            window_start: Instant::now(),
            window_rows: 0,
            member_time: vec![Duration::ZERO; models.len()],
        });
        Ok(Self {
            features: Default::default(),
            norm: needs_norm.then(Default::default),
            scratch: vec![[0.0; 2]; models.len()],
            initializers,
            alternate: !shared_backbone
                && models.iter().all(FastRnn::is_dynamic_int8)
                && models.len() > 1,
            odd_row: false,
            shared_backbone,
            input: vec![0.0; if needs_norm { base + N_RAW } else { base }],
            widths,
            active: models.len(),
            firsts: models.iter().map(FastRnn::first_weights).collect(),
            models,
            base_coefficients: coefficients.clone(),
            coefficients,
            out: [0.0; 2],
            seq: None,
            governor,
        })
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        row: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        read_row(py, row, &mut self.input[..crate::stream::N_RAW])?;
        self.advance()?;
        Ok(PyArray1::from_slice(py, &self.out))
    }

    /// The scorer callback: `data_point` needs `seq_ix`, `state` and `need_prediction`.
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        data_point: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let seq: i64 = data_point.getattr(intern!(py, "seq_ix"))?.extract()?;
        if self.seq != Some(seq) {
            let now = Instant::now();
            let active = self.active;
            if let Some(keep) = self.governor.as_mut().and_then(|g| g.plan(now, active)) {
                self.set_active(keep);
            }
            self.seq = Some(seq);
            self.reset_state();
        }
        let state = data_point.getattr(intern!(py, "state"))?;
        read_row(py, &state, &mut self.input[..crate::stream::N_RAW])?;
        self.advance()?;
        if data_point
            .getattr(intern!(py, "need_prediction"))?
            .is_truthy()?
        {
            Ok(Some(PyArray1::from_slice(py, &self.out)))
        } else {
            Ok(None)
        }
    }

    /// The precision every member runs ("mixed" if they differ, e.g. after a fallback).
    #[getter]
    fn precision(&self) -> &'static str {
        let first = self.models[0].precision();
        if self.models.iter().all(|m| m.precision() == first) {
            precision_name(first)
        } else {
            "mixed"
        }
    }

    #[getter]
    fn active_members(&self) -> usize {
        self.active
    }

    #[getter]
    fn coefficients(&self) -> Vec<f32> {
        self.coefficients.clone()
    }

    /// Weight bytes each member streams per row.
    #[getter]
    fn weight_bytes(&self) -> Vec<usize> {
        self.models.iter().map(FastRnn::weight_bytes).collect()
    }

    #[getter]
    fn q8_weight_bytes(&self) -> Vec<usize> {
        self.models
            .iter()
            .map(|model| {
                model
                    .q8_matrices()
                    .iter()
                    .map(|(_, matrix)| matrix.storage_bytes())
                    .sum()
            })
            .collect()
    }

    fn quantization_codes(&self, py: Python<'_>) -> PyResult<Vec<Q8Codes>> {
        self.models
            .iter()
            .map(|model| export_q8(py, model))
            .collect()
    }

    fn factorization_tensors(&self, py: Python<'_>) -> PyResult<Vec<FactorTensors>> {
        self.models
            .iter()
            .map(|model| export_factors(py, model))
            .collect()
    }

    #[getter]
    fn storage_bytes(&self) -> Vec<HashMap<String, usize>> {
        self.models.iter().map(storage_bytes).collect()
    }

    #[getter]
    fn hidden_states<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray2<f32>>>> {
        self.models
            .iter()
            .map(|model| export_hidden_state(py, model))
            .collect()
    }
}

/// Streaming predictor: event features + FastRnn, one row per call, no per-row allocation
/// besides the returned 2-element array.
#[pyclass(name = "Predictor")]
struct PyPredictor {
    features: crate::stream::StreamFeatures,
    model: FastRnn,
    input: Vec<f32>,
    out: [f32; 2],
}

#[pymethods]
impl PyPredictor {
    #[new]
    #[pyo3(signature = (weights, fast_activations=true, precision="f32", q8_codes=None, factors=None))]
    fn new(
        py: Python<'_>,
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        fast_activations: bool,
        precision: &str,
        q8_codes: Option<Q8Codes>,
        factors: Option<FactorTensors>,
    ) -> PyResult<Self> {
        let rnn_weights = weights_from_dict(weights)?;
        let n_in = rnn_weights.mean.len();
        if n_in != crate::stream::N_RAW + crate::stream::N_EVENT {
            return Err(PyValueError::new_err(format!(
                "model expects {n_in} inputs, predictor provides 211"
            )));
        }
        let act = if fast_activations {
            Activations::Fast
        } else {
            Activations::Exact
        };
        let mut model =
            model_with_factors(py, rnn_weights, act, parse_precision(precision)?, factors)?;
        if let Some(codes) = q8_codes {
            import_q8(py, &mut model, codes)?;
        }
        if model.n_out() < 2 {
            return Err(PyValueError::new_err("model must have at least 2 outputs"));
        }
        Ok(Self {
            features: Default::default(),
            model,
            input: vec![0.0; n_in],
            out: [0.0; 2],
        })
    }

    fn reset(&mut self) {
        self.features.reset();
        self.model.reset();
    }

    #[getter]
    fn hidden_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        export_hidden_state(py, &self.model)
    }

    fn quantization_codes(&self, py: Python<'_>) -> PyResult<Q8Codes> {
        export_q8(py, &self.model)
    }

    fn factorization_tensors(&self, py: Python<'_>) -> PyResult<FactorTensors> {
        export_factors(py, &self.model)
    }

    #[getter]
    fn storage_bytes(&self) -> HashMap<String, usize> {
        storage_bytes(&self.model)
    }

    #[getter]
    fn weight_bytes(&self) -> usize {
        self.model.weight_bytes()
    }

    #[getter]
    fn precision(&self) -> &'static str {
        precision_name(self.model.precision())
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        row: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let row = row
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if row.len() != crate::stream::N_RAW {
            return Err(PyValueError::new_err(format!(
                "expected 112 features, got {}",
                row.len()
            )));
        }
        let (raw, event) = self.input.split_at_mut(crate::stream::N_RAW);
        raw.copy_from_slice(row);
        self.features.sanitize(raw);
        self.features.update(raw, event);
        let n_out = self.model.n_out();
        if n_out == 2 {
            self.model.step(&self.input, &mut self.out);
        } else {
            let mut full = vec![0.0f32; n_out];
            self.model.step(&self.input, &mut full);
            self.out.copy_from_slice(&full[..2]);
        }
        Ok(PyArray1::from_slice(py, &self.out))
    }

    /// Like `step`, but returns every head output (targets, then auxiliary heads such as the
    /// market-class logits), for callers that route on them.
    fn step_all<'py>(
        &mut self,
        py: Python<'py>,
        row: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let row = row
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if row.len() != crate::stream::N_RAW {
            return Err(PyValueError::new_err(format!(
                "expected 112 features, got {}",
                row.len()
            )));
        }
        let (raw, event) = self.input.split_at_mut(crate::stream::N_RAW);
        raw.copy_from_slice(row);
        self.features.sanitize(raw);
        self.features.update(raw, event);
        let mut full = vec![0.0f32; self.model.n_out()];
        self.model.step(&self.input, &mut full);
        Ok(PyArray1::from_slice(py, &full))
    }
}

#[cfg(test)]
mod shared_backbone_tests {
    use super::*;

    fn fixture_weights(seed: usize) -> RnnWeights {
        let values = |count: usize, offset: usize| {
            (0..count)
                .map(|index| ((index + offset + seed) % 31) as f32 / 31.0)
                .collect::<Vec<_>>()
        };
        let hidden = 4;
        let input = 3;
        RnnWeights {
            mean: values(input, 0),
            std: values(input, 1),
            w_in: values(hidden * input, 2),
            b_in: values(hidden, 3),
            layers: (0..2)
                .map(|layer| GruLayer {
                    w_ih: values(3 * hidden * hidden, 4 + layer),
                    w_hh: values(3 * hidden * hidden, 6 + layer),
                    b_ih: values(3 * hidden, 8 + layer),
                    b_hh: values(3 * hidden, 10 + layer),
                })
                .collect(),
            w1: values(hidden * 2 * hidden, 12),
            b1: values(hidden, 13),
            w2: values(2 * hidden, 14),
            b2: values(2, 15),
        }
    }

    #[test]
    fn shared_backbone_preflight_allows_distinct_heads_and_rejects_changed_trunk() {
        let first = fixture_weights(1);
        let mut second = fixture_weights(1);
        second.w1[0] = f32::from_bits(second.w1[0].to_bits() ^ 1);
        second.w2[0] = f32::from_bits(second.w2[0].to_bits() ^ 1);
        let half = [0.5, 0.5];
        let precision = Precision::Q8VnniPortable;

        assert!(
            validate_shared_backbone_pair(
                &[first, second],
                &half,
                precision,
                false,
                false,
                false,
            )
            .is_ok(),
            "different heads remain eligible for exact sharing"
        );

        let first = fixture_weights(1);
        let mut incompatible = fixture_weights(1);
        incompatible.layers[0].w_hh[0] = f32::from_bits(
            incompatible.layers[0].w_hh[0].to_bits() ^ 1,
        );
        assert!(
            validate_shared_backbone_pair(
                &[first, incompatible],
                &half,
                precision,
                false,
                false,
                false,
            )
            .is_err(),
            "a changed recurrent tensor must reject sharing"
        );

        let mut incompatible_output = fixture_weights(1);
        incompatible_output.w2 = (0..12).map(|index| index as f32).collect();
        incompatible_output.b2 = vec![0.0; 3];
        assert!(
            validate_shared_backbone_pair(
                &[fixture_weights(1), incompatible_output],
                &half,
                precision,
                false,
                false,
                false,
            )
            .is_err(),
            "a head without exactly two outputs must reject sharing"
        );
    }
}

#[cfg(test)]
mod dyn_q8_python_tests {
    use super::*;

    type Expected = Vec<(String, usize, usize, Vec<i8>, Vec<f32>)>;

    fn fixture_weights(seed: usize) -> RnnWeights {
        let values = |count: usize, offset: usize| {
            (0..count)
                .map(|index| ((index + offset + seed) % 29) as f32 / 29.0 - 0.5)
                .collect::<Vec<_>>()
        };
        let hidden = 4;
        let input = 3;
        RnnWeights {
            mean: values(input, 0),
            std: vec![1.0; input],
            w_in: values(hidden * input, 1),
            b_in: values(hidden, 2),
            layers: (0..2)
                .map(|layer| GruLayer {
                    w_ih: values(3 * hidden * hidden, 3 + layer),
                    w_hh: values(3 * hidden * hidden, 5 + layer),
                    b_ih: values(3 * hidden, 7 + layer),
                    b_hh: values(3 * hidden, 9 + layer),
                })
                .collect(),
            w1: values(2 * hidden * hidden, 11),
            b1: values(hidden, 12),
            w2: values(2 * hidden, 13),
            b2: values(2, 14),
        }
    }

    fn q4_payload(py: Python<'_>, model: &FastRnn) -> PyResult<(Q8Codes, Expected)> {
        let mut payload = HashMap::new();
        let mut expected = Vec::new();
        for (matrix_index, (name, matrix)) in model.dyn8_matrices().enumerate() {
            let (rows, cols) = (matrix.rows(), matrix.cols());
            let mut codes = vec![0i8; rows * cols];
            let mut scales = vec![0.0f32; rows];
            for row in 0..rows {
                if row % 5 == 0 {
                    continue;
                }
                scales[row] = (row + matrix_index + 1) as f32 * 0.00390625;
                for col in 0..cols {
                    codes[row * cols + col] = ((row * 7 + col * 3 + matrix_index) % 15) as i8 - 7;
                }
            }
            let code_key = format!("{name}.codes");
            let scale_key = format!("{name}.scales");
            payload.insert(
                code_key,
                PyArray1::from_vec(py, codes.clone())
                    .reshape([rows, cols])?
                    .into_any()
                    .unbind(),
            );
            payload.insert(
                scale_key,
                PyArray1::from_vec(py, scales.clone()).into_any().unbind(),
            );
            expected.push((name.to_owned(), rows, cols, codes, scales));
        }
        Ok((payload, expected))
    }

    fn q8_model() -> PyResult<FastRnn> {
        FastRnn::with_precision(
            fixture_weights(17),
            Activations::Fast,
            Precision::Q8VnniPortable,
        )
        .map_err(|error| PyValueError::new_err(error.0))
    }

    fn with_python<T>(
        f: impl for<'py> FnOnce(Python<'py>) -> PyResult<T>,
    ) -> PyResult<T> {
        Python::initialize();
        Python::attach(f)
    }

    #[test]
    fn dynamic_q8_import_export_is_exact_and_hidden_getters_are_copied() {
        with_python(|py| -> PyResult<()> {
            let mut model = q8_model()?;
            let (payload, expected) = q4_payload(py, &model)?;
            import_q8(py, &mut model, payload)?;
            let exported = export_q8(py, &model)?;
            assert_eq!(exported.len(), expected.len() * 2);
            for (name, rows, cols, codes, scales) in expected {
                let code_key = format!("{name}.codes");
                let scale_key = format!("{name}.scales");
                let actual_codes = exported[&code_key]
                    .bind(py)
                    .extract::<numpy::PyReadonlyArray2<'_, i8>>()?;
                assert_eq!(actual_codes.shape(), [rows, cols]);
                assert_eq!(actual_codes.as_slice()?, codes);
                let actual_scales = exported[&scale_key]
                    .bind(py)
                    .extract::<PyReadonlyArray1<'_, f32>>()?;
                assert_eq!(
                    actual_scales
                        .as_slice()?
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    scales.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                );
            }

            let mut output = [0.0f32; 2];
            model.step(&[0.2, -0.4, 1.0], &mut output);
            let before = model.hidden_state().to_vec();
            let stepper = PyModelStepper { model };
            let snapshot = stepper.hidden_state(py)?;
            assert_eq!(snapshot.shape(), [2, 4]);
            assert!(!snapshot
                .getattr("flags")?
                .getattr("writeable")?
                .extract::<bool>()?);
            snapshot.call_method1(intern!(py, "setflags"), (true,))?;
            snapshot.call_method1(intern!(py, "fill"), (f32::NAN,))?;
            assert_eq!(stepper.model.hidden_state(), before);

            let mut predictor = PyPredictor {
                features: Default::default(),
                model: q8_model()?,
                input: vec![0.0; 3],
                out: [0.0; 2],
            };
            predictor.model.step(&[0.2, -0.4, 1.0], &mut output);
            let predictor_snapshot = predictor.hidden_state(py)?;
            assert_eq!(predictor_snapshot.shape(), [2, 4]);
            assert!(!predictor_snapshot
                .getattr("flags")?
                .getattr("writeable")?
                .extract::<bool>()?);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn dynamic_q8_import_rejects_missing_excess_bad_codes_and_wrong_shapes() {
        with_python(|py| -> PyResult<()> {
            let mut model = q8_model()?;
            let (payload, _expected) = q4_payload(py, &model)?;

            let mut missing = payload;
            missing.remove("head.0.weight.1.codes");
            missing.insert(
                "unexpected.codes".into(),
                PyArray1::from_vec(py, vec![1i8]).into_any().unbind(),
            );
            assert!(import_q8(py, &mut model, missing).is_err());

            let (mut excess, _expected) = q4_payload(py, &model)?;
            excess.insert(
                "unexpected.codes".into(),
                PyArray1::from_vec(py, vec![1i8]).into_any().unbind(),
            );
            assert!(import_q8(py, &mut model, excess).is_err());

            let (mut bad_codes, expected) = q4_payload(py, &model)?;
            let (name, rows, cols, _, _) = &expected[0];
            bad_codes.insert(
                format!("{name}.codes"),
                PyArray1::from_vec(py, vec![-128i8; *rows * *cols])
                    .reshape([*rows, *cols])?
                    .into_any()
                    .unbind(),
            );
            assert!(import_q8(py, &mut model, bad_codes).is_err());

            let (mut bad_shape, expected) = q4_payload(py, &model)?;
            let (name, rows, cols, codes, _) = &expected[0];
            assert_eq!(*cols % 2, 0);
            bad_shape.insert(
                format!("{name}.codes"),
                PyArray1::from_vec(py, codes.clone())
                    .reshape([*rows * 2, *cols / 2])?
                    .into_any()
                    .unbind(),
            );
            bad_shape.insert(
                format!("{name}.scales"),
                PyArray1::from_vec(py, vec![0.5f32; *rows * 2])
                    .into_any()
                    .unbind(),
            );
            assert!(import_q8(py, &mut model, bad_shape).is_err());
            Ok(())
        })
        .unwrap();
    }
}
