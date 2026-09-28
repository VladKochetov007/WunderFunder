use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rnn::{GruLayer, RnnModel, RnnWeights};

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
    Ok(())
}

/// Several GRUs with one causal feature update and one Python call per row. Members read
/// [raw 112 | event 99] (211 inputs) or, when trained with online norm, additionally the 112
/// causal per-sequence z-scores of the raw row (323 inputs); z-scores are computed only if some
/// member needs them.
#[pyclass(name = "EnsemblePredictor")]
struct PyEnsemblePredictor {
    features: crate::stream::StreamFeatures,
    norm: Option<crate::stream::OnlineNorm>,
    widths: Vec<usize>,
    models: Vec<crate::rnn_fast::FastRnn>,
    input: Vec<f32>,
    coefficients: Vec<f32>,
    scratch: Vec<[f32; 2]>,
    out: [f32; 2],
}

#[pymethods]
impl PyEnsemblePredictor {
    #[new]
    #[pyo3(signature = (weights, coefficients, fast_activations=true))]
    fn new(
        weights: Vec<HashMap<String, PyReadonlyArrayDyn<'_, f32>>>,
        coefficients: Vec<f32>,
        fast_activations: bool,
    ) -> PyResult<Self> {
        if weights.is_empty() || weights.len() != coefficients.len() {
            return Err(PyValueError::new_err(
                "provide one coefficient for each nonempty model",
            ));
        }
        if coefficients.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("coefficients must be finite"));
        }
        let activation = if fast_activations {
            crate::rnn_fast::Activations::Fast
        } else {
            crate::rnn_fast::Activations::Exact
        };
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
            let model = crate::rnn_fast::FastRnn::new(parsed, activation)
                .map_err(|e| PyValueError::new_err(e.0))?;
            if model.n_out() != 2 {
                return Err(PyValueError::new_err("ensemble members need two outputs"));
            }
            models.push(model);
        }
        let needs_norm = widths.iter().any(|&w| w > base);
        Ok(Self {
            features: Default::default(),
            norm: needs_norm.then(Default::default),
            scratch: vec![[0.0; 2]; models.len()],
            input: vec![0.0; if needs_norm { base + N_RAW } else { base }],
            widths,
            models,
            coefficients,
            out: [0.0; 2],
        })
    }

    fn reset(&mut self) {
        self.features.reset();
        if let Some(norm) = &mut self.norm {
            norm.reset();
        }
        for model in &mut self.models {
            model.reset();
        }
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
        let (raw, rest) = self.input.split_at_mut(crate::stream::N_RAW);
        raw.copy_from_slice(row);
        self.features.sanitize(raw);
        let (event, zscores) = rest.split_at_mut(crate::stream::N_EVENT);
        self.features.update(raw, event);
        if let Some(norm) = &mut self.norm {
            norm.update(raw, zscores);
        }
        self.out = [0.0; 2];
        for (((model, scratch), coefficient), &width) in self
            .models
            .iter_mut()
            .zip(self.scratch.iter_mut())
            .zip(&self.coefficients)
            .zip(&self.widths)
        {
            model.step(&self.input[..width], scratch);
            for (total, value) in self.out.iter_mut().zip(scratch) {
                *total += *coefficient * *value;
            }
        }
        for value in &mut self.out {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(PyArray1::from_slice(py, &self.out))
    }
}

/// Streaming predictor: event features + FastRnn, one row per call, no per-row allocation
/// besides the returned 2-element array.
#[pyclass(name = "Predictor")]
struct PyPredictor {
    features: crate::stream::StreamFeatures,
    model: crate::rnn_fast::FastRnn,
    input: Vec<f32>,
    out: [f32; 2],
}

#[pymethods]
impl PyPredictor {
    #[new]
    #[pyo3(signature = (weights, fast_activations=true))]
    fn new(
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        fast_activations: bool,
    ) -> PyResult<Self> {
        let rnn_weights = weights_from_dict(weights)?;
        let n_in = rnn_weights.mean.len();
        if n_in != crate::stream::N_RAW + crate::stream::N_EVENT {
            return Err(PyValueError::new_err(format!(
                "model expects {n_in} inputs, predictor provides 211"
            )));
        }
        let act = if fast_activations {
            crate::rnn_fast::Activations::Fast
        } else {
            crate::rnn_fast::Activations::Exact
        };
        let model = crate::rnn_fast::FastRnn::new(rnn_weights, act)
            .map_err(|e| PyValueError::new_err(e.0))?;
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
