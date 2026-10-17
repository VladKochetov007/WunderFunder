//! Stateful Python binding for the causal moment-geometry submission callback.

use numpy::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::moment_geometry::{DICTIONARY_WIDTH, FeatureState, MOMENT_WIDTH};
use crate::rnn_fast::{Activations, FastRnn, Precision};
use crate::stream::{N_EVENT, N_RAW, StreamFeatures};

#[derive(Clone, Copy)]
enum Arm {
    Control,
    Moments,
    Scaled,
    Dictionary,
    Combined,
}

impl Arm {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "control" => Ok(Self::Control),
            "moments" => Ok(Self::Moments),
            "scaled" => Ok(Self::Scaled),
            "dictionary" => Ok(Self::Dictionary),
            "combined" => Ok(Self::Combined),
            other => Err(PyValueError::new_err(format!(
                "unknown geometry arm {other:?}; expected control, moments, scaled, dictionary, or combined"
            ))),
        }
    }

    fn geometry_active(self, channel: usize) -> bool {
        match self {
            Self::Control | Self::Dictionary => false,
            Self::Moments => channel < 16,
            Self::Scaled | Self::Combined => true,
        }
    }

    fn dictionary_active(self, channel: usize) -> bool {
        matches!(self, Self::Dictionary | Self::Combined) && channel < DICTIONARY_WIDTH
    }

    fn uses_dictionary(self) -> bool {
        matches!(self, Self::Dictionary | Self::Combined)
    }
}

fn four_f64(array: PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<[f64; 4]> {
    let values = array.as_array();
    if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain four finite values"
        )));
    }
    Ok(std::array::from_fn(|index| values[index]))
}

fn dictionary_array(
    array: PyReadonlyArray2<'_, f32>,
) -> PyResult<[f32; DICTIONARY_WIDTH / 10 * 3]> {
    let values = array.as_array();
    if values.shape() != [4, 3] || values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "dictionary_levels must be a finite (4, 3) array",
        ));
    }
    Ok(std::array::from_fn(|index| values[[index / 3, index % 3]]))
}

/// Stateful scorer callback that builds raw, event, and reset-local geometry inputs in Rust.
#[pyclass(name = "GeometryPredictor")]
struct PyGeometryPredictor {
    stream: StreamFeatures,
    features: FeatureState,
    model: FastRnn,
    input: Vec<f32>,
    input_mean: Vec<f32>,
    output: [f32; 2],
    arm: Arm,
    dictionary_enabled: bool,
    seq: Option<i64>,
}

impl PyGeometryPredictor {
    fn reset_state(&mut self) {
        self.stream.reset();
        self.features.reset();
        self.model.reset();
        self.output = [0.0; 2];
    }

    fn advance(&mut self) -> PyResult<()> {
        let (raw, event) = self.input.split_at_mut(N_RAW);
        self.stream.sanitize(raw);
        self.stream.update(raw, event);

        let mut geometry = [0.0f32; MOMENT_WIDTH];
        self.features.moments(raw, &mut geometry);
        self.input[N_RAW + N_EVENT..N_RAW + N_EVENT + MOMENT_WIDTH].copy_from_slice(&geometry);

        if self.input.len() == N_RAW + N_EVENT + MOMENT_WIDTH + DICTIONARY_WIDTH {
            let mut dictionary = [0.0f32; DICTIONARY_WIDTH];
            if self.dictionary_enabled {
                self.features.discrete_features(raw, &mut dictionary);
            }
            let start = N_RAW + N_EVENT + MOMENT_WIDTH;
            self.input[start..].copy_from_slice(&dictionary);
        }

        for index in N_RAW + N_EVENT..self.input.len() {
            let channel = index - N_RAW - N_EVENT;
            let active = if channel < MOMENT_WIDTH {
                self.arm.geometry_active(channel)
            } else {
                self.arm.dictionary_active(channel - MOMENT_WIDTH)
            };
            if !active {
                // FastRnn folds `(x - mean) / std` into its first projection; filling the
                // unstandardized coordinate with mean therefore recreates a post-scale zero.
                self.input[index] = self.input_mean[index];
            }
        }

        self.model.step(&self.input, &mut self.output);
        Ok(())
    }
}

#[pymethods]
impl PyGeometryPredictor {
    #[new]
    #[pyo3(signature = (weights, beta, intercept, atoms, dictionary_levels=None, dictionary_enabled=false, arm="scaled", fast_activations=true))]
    fn new(
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        beta: PyReadonlyArray1<'_, f64>,
        intercept: PyReadonlyArray1<'_, f64>,
        atoms: PyReadonlyArray1<'_, f64>,
        dictionary_levels: Option<PyReadonlyArray2<'_, f32>>,
        dictionary_enabled: bool,
        arm: &str,
        fast_activations: bool,
    ) -> PyResult<Self> {
        let arm = Arm::parse(arm)?;
        let input_mean = weights
            .get("mean")
            .ok_or_else(|| PyValueError::new_err("missing weight mean"))?
            .as_array()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let rnn_weights = crate::python::weights_from_dict(weights)?;
        let width = rnn_weights.mean.len();
        let max_width = N_RAW + N_EVENT + MOMENT_WIDTH + DICTIONARY_WIDTH;
        if width != max_width && width != max_width - DICTIONARY_WIDTH {
            return Err(PyValueError::new_err(format!(
                "geometry predictor needs231 or271 input channels, got {width}"
            )));
        }
        if input_mean.len() != width {
            return Err(PyValueError::new_err(
                "input mean width does not match weights",
            ));
        }
        let has_dictionary_width = width == max_width;
        if arm.uses_dictionary() && (!has_dictionary_width || !dictionary_enabled) {
            return Err(PyValueError::new_err(
                "dictionary and combined arms require271 inputs and dictionary_enabled=true",
            ));
        }
        if dictionary_enabled && !has_dictionary_width {
            return Err(PyValueError::new_err(
                "dictionary features require271 input channels",
            ));
        }
        let vocabulary = match (dictionary_enabled, dictionary_levels) {
            (true, Some(levels)) => Some(dictionary_array(levels)?),
            (true, None) => {
                return Err(PyValueError::new_err(
                    "dictionary_enabled=true requires dictionary_levels",
                ));
            }
            (false, _) => None,
        };
        let model = FastRnn::with_precision(
            rnn_weights,
            if fast_activations {
                Activations::Fast
            } else {
                Activations::Exact
            },
            Precision::F32,
        )
        .map_err(|error| PyValueError::new_err(error.0))?;
        if model.n_out() != 2 {
            return Err(PyValueError::new_err(format!(
                "geometry predictor requires two outputs, got {}",
                model.n_out()
            )));
        }

        Ok(Self {
            stream: StreamFeatures::default(),
            features: FeatureState::new(
                four_f64(beta, "beta")?,
                four_f64(intercept, "intercept")?,
                four_f64(atoms, "atoms")?,
                vocabulary,
            ),
            model,
            input: vec![0.0; width],
            input_mean,
            output: [0.0; 2],
            arm,
            dictionary_enabled,
            seq: None,
        })
    }

    /// Resets all raw/event/geometry/dictionary and recurrent state for a new sequence.
    fn reset(&mut self) {
        self.seq = None;
        self.reset_state();
    }

    /// Advances one raw112-row and returns the two predictions.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        row: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        crate::python::read_row(py, row, &mut self.input[..N_RAW])?;
        self.advance()?;
        Ok(PyArray1::from_slice(py, &self.output))
    }

    /// Diagnostic step returning predictions and post-mask unstandardized model inputs.
    fn step_with_features<'py>(
        &mut self,
        py: Python<'py>,
        row: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        crate::python::read_row(py, row, &mut self.input[..N_RAW])?;
        self.advance()?;
        let prediction = PyArray1::from_slice(py, &self.output);
        let features = PyArray1::from_slice(py, &self.input);
        Ok((prediction, features))
    }

    /// Full scorer callback; every row advances state, including rows with no requested score.
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        data_point: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let seq = data_point
            .getattr(intern!(py, "seq_ix"))?
            .extract::<i64>()?;
        if self.seq != Some(seq) {
            self.seq = Some(seq);
            self.reset_state();
        }
        let state = data_point.getattr(intern!(py, "state"))?;
        crate::python::read_row(py, &state, &mut self.input[..N_RAW])?;
        self.advance()?;
        if data_point
            .getattr(intern!(py, "need_prediction"))?
            .is_truthy()?
        {
            Ok(Some(PyArray1::from_slice(py, &self.output)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn input_width(&self) -> usize {
        self.input.len()
    }

    #[getter]
    fn precision(&self) -> &'static str {
        "f32"
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGeometryPredictor>()?;
    Ok(())
}
