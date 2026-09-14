use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rnn::{GruLayer, RnnModel, RnnWeights};

fn take(w: &mut HashMap<String, Vec<f32>>, key: &str) -> PyResult<Vec<f32>> {
    w.remove(key)
        .ok_or_else(|| PyValueError::new_err(format!("missing weight {key}")))
}

/// Stateful twin of wnn.nn.RNNModel. Construct from `model.state_dict()` arrays.
#[pyclass(name = "GruStepper", unsendable)]
struct PyGruStepper {
    inner: RnnModel,
}

#[pymethods]
impl PyGruStepper {
    #[new]
    fn new(weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>) -> PyResult<Self> {
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
        let weights = RnnWeights {
            mean: take(&mut w, "mean")?,
            std: take(&mut w, "std")?,
            w_in: take(&mut w, "inp.weight")?,
            b_in: take(&mut w, "inp.bias")?,
            layers,
            w1: take(&mut w, "head.0.weight")?,
            b1: take(&mut w, "head.0.bias")?,
            w2: take(&mut w, "head.2.weight")?,
            b2: take(&mut w, "head.2.bias")?,
        };
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
    Ok(())
}
