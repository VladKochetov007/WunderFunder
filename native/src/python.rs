use std::collections::HashMap;
use std::time::{Duration, Instant};

use numpy::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

use crate::rnn::{GruLayer, RnnModel, RnnWeights};
use crate::rnn_fast::{Activations, FastRnn, Precision};

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
            "import_only_dense_recurrent".into(),
            model.import_only_dense_bytes(),
        ),
    ])
}

fn import_q8(py: Python<'_>, model: &mut FastRnn, mut codes: Q8Codes) -> PyResult<()> {
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

    /// Features for the row already in `input[..N_RAW]`, then the blend of the active members.
    fn advance(&mut self) {
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
        match &mut self.governor {
            None => {
                for k in 0..active {
                    let i = order(k);
                    // Prefetch the next member's first weights (for the last member, the next
                    // row's first) during this member's final activation pass.
                    let after = self.firsts[order((k + 1) % active)];
                    let width = self.widths[i];
                    self.models[i].step_then(&self.input[..width], &mut self.scratch[i], after);
                }
            }
            Some(governor) => {
                for k in 0..active {
                    let i = order(k);
                    let t = Instant::now();
                    let width = self.widths[i];
                    self.models[i].step(&self.input[..width], &mut self.scratch[i]);
                    governor.member_time[i] += t.elapsed();
                }
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
        factors=None
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
        let mut encoded = q8_codes.map(|codes| codes.into_iter());
        if factors
            .as_ref()
            .is_some_and(|items| items.len() != weights.len())
        {
            return Err(PyValueError::new_err(
                "provide factors for every ensemble member",
            ));
        }
        let mut factors = factors.map(|items| items.into_iter());
        let activation = if fast_activations {
            Activations::Fast
        } else {
            Activations::Exact
        };
        let precision = parse_precision(precision)?;
        use crate::stream::{N_EVENT, N_RAW};
        let base = N_RAW + N_EVENT;
        let mut models = Vec::with_capacity(weights.len());
        let mut widths = Vec::with_capacity(weights.len());
        for member in weights {
            let parsed = weights_from_dict(member)?;
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
            alternate: models.iter().all(FastRnn::is_dynamic_int8) && models.len() > 1,
            odd_row: false,
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
        self.advance();
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
        self.advance();
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
