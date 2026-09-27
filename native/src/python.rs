use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rnn::{ConvEncoder, GruLayer, RnnModel, RnnWeights};

/// Encoders this build understands (`wnn.nn.RNNModel(enc=...)`); packaging checks it.
const ENCODERS: [&str; 2] = ["linear", "conv"];

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
    let b_in = take(&mut w, "inp.bias")?;
    // enc="conv" is selected by the presence of its weights; k comes from conv_dw.weight (hidden, 1, k).
    let conv = match w.remove("conv_dw.weight") {
        None => None,
        Some(w_dw) => {
            let hidden = b_in.len();
            if hidden == 0 || w_dw.len() % hidden != 0 {
                return Err(PyValueError::new_err(format!(
                    "conv_dw.weight: {} values is not a multiple of hidden {hidden}",
                    w_dw.len()
                )));
            }
            Some(ConvEncoder {
                k: w_dw.len() / hidden,
                w_dw,
                b_dw: take(&mut w, "conv_dw.bias")?,
                w_pw: take(&mut w, "conv_pw.weight")?,
                b_pw: take(&mut w, "conv_pw.bias")?,
            })
        }
    };
    let out = RnnWeights {
        mean: take(&mut w, "mean")?,
        std: take(&mut w, "std")?,
        w_in: take(&mut w, "inp.weight")?,
        b_in,
        layers,
        w1: take(&mut w, "head.0.weight")?,
        b1: take(&mut w, "head.0.bias")?,
        w2: take(&mut w, "head.2.weight")?,
        b2: take(&mut w, "head.2.bias")?,
        conv,
    };
    // A weight this engine does not consume means a model variant it would silently mis-run.
    if !w.is_empty() {
        let mut extra: Vec<_> = w.into_keys().collect();
        extra.sort();
        return Err(PyValueError::new_err(format!(
            "unsupported weights (model variant not implemented natively): {}",
            extra.join(", ")
        )));
    }
    Ok(out)
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
    m.add("ENCODERS", ENCODERS)?;
    Ok(())
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

    /// "conv" or "linear": the encoder selected from the weights.
    #[getter]
    fn encoder(&self) -> &'static str {
        if self.model.has_conv() {
            "conv"
        } else {
            "linear"
        }
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
}
