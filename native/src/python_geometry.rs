//! Native streaming event and geometry features with an optional recurrent scorer.
//!
//! This child module keeps the PyO3 bindings and per-row feature math out of `python.rs`.

use super::*;
use crate::stream::{StreamFeatures, N_EVENT, N_RAW};

const N_GEOMETRY: usize = 60;
const N_FEATURES: usize = N_EVENT + N_GEOMETRY;
const MODEL_INPUT: usize = N_RAW + N_FEATURES;
const SELECTED: [usize; 4] = [0, 3, 52, 55];
const DWELL_CAP: usize = 10_000;

struct GeometryFeatures {
    beta: [f64; 4],
    intercept: [f64; 4],
    atoms: [f64; 4],
    vocabulary: [[f64; 3]; 4],
    position: usize,
    count: usize,
    previous: Option<[f64; 4]>,
    total: [f64; 4],
    total_squared: [f64; 4],
    atom_count: [f64; 4],
    occupancy: [[f64; 4]; 4],
    event_count: [usize; 4],
    dominant_count: [usize; 4],
    last_change: [usize; 4],
    category: [[bool; 4]; 4],
}

impl GeometryFeatures {
    fn new(
        beta: [f64; 4],
        intercept: [f64; 4],
        atoms: [f64; 4],
        vocabulary: [[f64; 3]; 4],
    ) -> Self {
        Self {
            beta,
            intercept,
            atoms,
            vocabulary,
            position: 0,
            count: 0,
            previous: None,
            total: [0.0; 4],
            total_squared: [0.0; 4],
            atom_count: [0.0; 4],
            occupancy: [[0.0; 4]; 4],
            event_count: [0; 4],
            dominant_count: [0; 4],
            last_change: [0; 4],
            category: [[false; 4]; 4],
        }
    }

    fn reset(&mut self) {
        self.position = 0;
        self.count = 0;
        self.previous = None;
        self.total = [0.0; 4];
        self.total_squared = [0.0; 4];
        self.atom_count = [0.0; 4];
        self.occupancy = [[0.0; 4]; 4];
        self.event_count = [0; 4];
        self.dominant_count = [0; 4];
        self.last_change = [0; 4];
        self.category = [[false; 4]; 4];
    }

    fn update(&mut self, row: &[f32], output: &mut [f32]) {
        debug_assert_eq!(output.len(), N_GEOMETRY);
        let values = SELECTED.map(|index| row[index] as f64);
        let changed = self
            .previous
            .map(|previous| std::array::from_fn(|field| values[field] != previous[field]))
            .unwrap_or([true; 4]);
        for field in 0..4 {
            let value = values[field];
            self.total[field] += value;
            self.total_squared[field] += value * value;
        }
        self.count += 1;
        let count = self.count as f64;
        for field in 0..4 {
            let mean = self.total[field] / count;
            let second = self.total_squared[field] / count;
            let std = (second - mean * mean).max(0.0).sqrt();
            self.atom_count[field] += if values[field] == self.atoms[field] {
                1.0
            } else {
                0.0
            };
            let base = field * 4;
            output[base] = mean as f32;
            output[4 + base] = std as f32;
            output[8 + base] = (second - self.beta[field] * mean - self.intercept[field]) as f32;
            output[12 + base] = (self.atom_count[field] / count) as f32;
            output[16 + base] = ((values[field] - mean) / std.max(0.05)).clamp(-8.0, 8.0) as f32;

            let mut any_known = false;
            for level in 0..3 {
                let is_level = values[field] == self.vocabulary[field][level];
                self.category[field][level] = is_level;
                any_known |= is_level;
            }
            self.category[field][3] = !any_known;
            for level in 0..4 {
                self.occupancy[field][level] += if self.category[field][level] {
                    1.0
                } else {
                    0.0
                };
                output[20 + base + level] = if self.category[field][level] {
                    1.0
                } else {
                    0.0
                };
                output[36 + base + level] = (self.occupancy[field][level] / count) as f32;
            }
            if changed[field] {
                self.event_count[field] += 1;
                if self.category[field][0] {
                    self.dominant_count[field] += 1;
                }
                self.last_change[field] = self.position;
            }
            output[52 + field] =
                self.dominant_count[field] as f32 / self.event_count[field].max(1) as f32;
            let dwell = self
                .position
                .saturating_sub(self.last_change[field])
                .min(DWELL_CAP);
            output[56 + field] = (dwell as f64).ln_1p() as f32;
        }
        self.previous = Some(values);
        self.position += 1;
    }
}

struct FeatureState {
    events: StreamFeatures,
    geometry: GeometryFeatures,
    output: [f32; N_FEATURES],
}

impl FeatureState {
    fn new(geometry: GeometryFeatures) -> Self {
        Self {
            events: StreamFeatures::default(),
            geometry,
            output: [0.0; N_FEATURES],
        }
    }

    fn reset(&mut self) {
        self.events.reset();
        self.geometry.reset();
    }

    fn update(&mut self, row: &[f32]) -> &[f32] {
        let (events, geometry) = self.output.split_at_mut(N_EVENT);
        self.events.update(row, events);
        self.geometry.update(row, geometry);
        &self.output
    }
}

fn parse_prior(
    beta: PyReadonlyArray1<'_, f32>,
    intercept: PyReadonlyArray1<'_, f32>,
    atoms: PyReadonlyArray1<'_, f32>,
    vocabulary: numpy::PyReadonlyArray2<'_, f32>,
) -> PyResult<GeometryFeatures> {
    let beta = beta.as_slice()?;
    let intercept = intercept.as_slice()?;
    let atoms = atoms.as_slice()?;
    let vocabulary_view = vocabulary.as_array();
    if beta.len() != 4
        || intercept.len() != 4
        || atoms.len() != 4
        || vocabulary_view.shape() != [4, 3]
    {
        return Err(PyValueError::new_err(
            "geometry prior needs beta/intercept/atoms of length 4 and vocabulary shape (4,3)",
        ));
    }
    let vocabulary_values: Vec<f64> = vocabulary_view.iter().map(|value| *value as f64).collect();
    let finite = beta
        .iter()
        .chain(intercept)
        .chain(atoms)
        .all(|value| value.is_finite())
        && vocabulary_values.iter().all(|value| value.is_finite());
    if !finite {
        return Err(PyValueError::new_err(
            "geometry prior values must all be finite",
        ));
    }
    Ok(GeometryFeatures::new(
        std::array::from_fn(|index| beta[index] as f64),
        std::array::from_fn(|index| intercept[index] as f64),
        std::array::from_fn(|index| atoms[index] as f64),
        std::array::from_fn(|field| {
            std::array::from_fn(|level| vocabulary_values[field * 3 + level])
        }),
    ))
}

fn copy_raw_row(row: &[f32]) -> PyResult<[f32; N_RAW]> {
    if row.len() != N_RAW || row.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "expected one finite raw row with shape (112,)",
        ));
    }
    row.try_into()
        .map_err(|_| PyValueError::new_err("raw row must have width 112"))
}

/// Native event and geometry feature stream, for serving mixed-width ensembles.
#[pyclass(name = "GeometryFeatureStepper")]
pub(super) struct PyGeometryFeatureStepper {
    state: FeatureState,
}

#[pymethods]
impl PyGeometryFeatureStepper {
    #[new]
    fn new(
        beta: PyReadonlyArray1<'_, f32>,
        intercept: PyReadonlyArray1<'_, f32>,
        atoms: PyReadonlyArray1<'_, f32>,
        vocabulary: numpy::PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Self> {
        Ok(Self {
            state: FeatureState::new(parse_prior(beta, intercept, atoms, vocabulary)?),
        })
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        row: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let row = copy_raw_row(row.as_slice()?)?;
        let features = self.state.update(&row);
        Ok(PyArray1::from_slice(py, features))
    }
}

/// Full native per-row geometry GRU callback core.
#[pyclass(name = "GeometryPredictor")]
pub(super) struct PyGeometryPredictor {
    state: FeatureState,
    model: FastRnn,
    input: Vec<f32>,
    output: Vec<f32>,
}

#[pymethods]
impl PyGeometryPredictor {
    #[new]
    #[pyo3(signature = (weights, beta, intercept, atoms, vocabulary, precision="f32"))]
    fn new(
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        beta: PyReadonlyArray1<'_, f32>,
        intercept: PyReadonlyArray1<'_, f32>,
        atoms: PyReadonlyArray1<'_, f32>,
        vocabulary: numpy::PyReadonlyArray2<'_, f32>,
        precision: &str,
    ) -> PyResult<Self> {
        let state = FeatureState::new(parse_prior(beta, intercept, atoms, vocabulary)?);
        let weights = weights_from_dict(weights)?;
        let model =
            FastRnn::with_precision(weights, Activations::Fast, parse_precision(precision)?)
                .map_err(|error| PyValueError::new_err(error.0))?;
        if model.n_in() != MODEL_INPUT {
            return Err(PyValueError::new_err(format!(
                "geometry scorer expects {MODEL_INPUT} model inputs, got {}",
                model.n_in()
            )));
        }
        if model.n_out() < 2 {
            return Err(PyValueError::new_err(
                "geometry model needs at least two outputs",
            ));
        }
        let mut input = vec![0.0f32; MODEL_INPUT];
        let output = vec![0.0f32; model.n_out()];
        input[N_RAW..].fill(0.0);
        Ok(Self {
            state,
            model,
            input,
            output,
        })
    }

    fn reset(&mut self) {
        self.state.reset();
        self.model.reset();
    }

    #[pyo3(signature = (row, need_prediction=true))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        row: PyReadonlyArray1<'py, f32>,
        need_prediction: bool,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let row = copy_raw_row(row.as_slice()?)?;
        self.input[..N_RAW].copy_from_slice(&row);
        self.input[N_RAW..].copy_from_slice(self.state.update(&row));
        self.model.step(&self.input, &mut self.output);
        if need_prediction {
            Ok(Some(PyArray1::from_slice(py, &self.output[..2])))
        } else {
            Ok(None)
        }
    }
}
