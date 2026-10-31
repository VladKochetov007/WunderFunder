use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;

use crate::min_gru::{MinGruCellWeights, MinGruModel, MinGruPrecision, MinGruWeights};
use crate::rnn::{GruLayer, RnnModel, RnnWeights};
use crate::rnn_fast::{Activations, FastRnn, Precision};
use crate::state_init::{StateInitializer, StateInitializerConfig};
use crate::tiedlead::{TiedLeadEncoder, TiedLeadWeights};

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
        (
            "dynamic_q8_weights".into(),
            model
                .dyn8_matrices()
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

fn take_optional(w: &mut HashMap<String, Vec<f32>>, key: &str) -> Vec<f32> {
    w.remove(key).unwrap_or_default()
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
        w_mid: take_optional(&mut w, "head.extra.weight"),
        b_mid: take_optional(&mut w, "head.extra.bias"),
        w2: take(&mut w, "head.2.weight")?,
        b2: take(&mut w, "head.2.bias")?,
    })
}

const FINAL_TL3_WIDTHS: [usize; 4] = [211, 263, 263, 112];
const FINAL_TL3_ENCODER_WIDTHS: [usize; 4] = [256, 256, 256, 192];
const FINAL_TL3_HIDDEN_WIDTHS: [usize; 4] = [256, 256, 256, 192];
const FINAL_TL3_LAYER_COUNTS: [usize; 4] = [2, 2, 2, 1];
const FINAL_TIED_KEYS: [&str; 8] = [
    "tied.0.weight",
    "tied.0.bias",
    "tied.2.weight",
    "tied.2.bias",
    "tied_out.weight",
    "tied_out.bias",
    "lead_out.weight",
    "lead_out.bias",
];

fn final_tensor_shape(
    weights: &HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    name: &str,
) -> PyResult<Vec<usize>> {
    weights
        .get(name)
        .map(|array| array.shape().to_vec())
        .ok_or_else(|| PyValueError::new_err(format!("missing final predictor tensor {name}")))
}

fn require_final_shape(
    weights: &HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    name: &str,
    expected: &[usize],
) -> PyResult<()> {
    let got = final_tensor_shape(weights, name)?;
    if got != expected {
        return Err(PyValueError::new_err(format!(
            "final predictor tensor {name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn take_final_array(
    weights: &mut HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    name: &str,
) -> PyResult<Vec<f32>> {
    let array = weights
        .remove(name)
        .ok_or_else(|| PyValueError::new_err(format!("missing final predictor tensor {name}")))?;
    Ok(array.as_array().iter().copied().collect())
}

/// Validate the model-specific checkpoint schema before the general recurrent loader runs.
fn parse_final_member(
    mut weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    role: usize,
) -> PyResult<(RnnWeights, Option<TiedLeadWeights>)> {
    if role >= FINAL_TL3_WIDTHS.len() {
        return Err(PyValueError::new_err("invalid final predictor member role"));
    }
    let input_width = FINAL_TL3_WIDTHS[role];
    require_final_shape(&weights, "mean", &[input_width])?;
    require_final_shape(&weights, "std", &[input_width])?;
    if weights.get("std").is_some_and(|array| {
        array
            .as_array()
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    }) {
        return Err(PyValueError::new_err(
            "final predictor std must be finite and positive",
        ));
    }
    if weights
        .values()
        .any(|array| array.as_array().iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err(
            "final predictor tensors must be finite",
        ));
    }

    let input_projection = final_tensor_shape(&weights, "inp.weight")?;
    if input_projection.len() != 2 || input_projection[1] != input_width || input_projection[0] == 0
    {
        return Err(PyValueError::new_err(format!(
            "inp.weight must have shape [encoder_width, {input_width}], got {input_projection:?}"
        )));
    }
    let encoder_width = input_projection[0];
    if encoder_width != FINAL_TL3_ENCODER_WIDTHS[role] {
        return Err(PyValueError::new_err(format!(
            "final member {role} encoder width must be {}, got {encoder_width}",
            FINAL_TL3_ENCODER_WIDTHS[role]
        )));
    }
    require_final_shape(&weights, "inp.bias", &[encoder_width])?;

    let mut allowed = HashSet::from([
        "mean".to_owned(),
        "std".to_owned(),
        "inp.weight".to_owned(),
        "inp.bias".to_owned(),
        "head.0.weight".to_owned(),
        "head.0.bias".to_owned(),
        "head.2.weight".to_owned(),
        "head.2.bias".to_owned(),
    ]);
    let recurrent_input = final_tensor_shape(&weights, "gru.weight_ih_l0")?;
    let recurrent_hidden = final_tensor_shape(&weights, "gru.weight_hh_l0")?;
    if recurrent_input.len() != 2 || recurrent_hidden.len() != 2 || recurrent_hidden[1] == 0 {
        return Err(PyValueError::new_err(
            "recurrent weight tensors must be matrices",
        ));
    }
    let hidden_width = recurrent_hidden[1];
    if hidden_width != FINAL_TL3_HIDDEN_WIDTHS[role] {
        return Err(PyValueError::new_err(format!(
            "final member {role} recurrent width must be {}, got {hidden_width}",
            FINAL_TL3_HIDDEN_WIDTHS[role]
        )));
    }
    let gate_rows = recurrent_hidden[0];
    if gate_rows % hidden_width != 0 || ![3, 4].contains(&(gate_rows / hidden_width)) {
        return Err(PyValueError::new_err(
            "recurrent weights must describe a GRU or LSTM",
        ));
    }
    let gates = gate_rows / hidden_width;
    if role == 3 && gates != 4 {
        return Err(PyValueError::new_err(
            "final Source4 member tensors must use LSTM gates",
        ));
    }
    if recurrent_input != [gates * hidden_width, encoder_width] {
        return Err(PyValueError::new_err(format!(
            "gru.weight_ih_l0 must have shape [{}, {encoder_width}], got {recurrent_input:?}",
            gates * hidden_width
        )));
    }
    let mut layer = 0usize;
    loop {
        let input_weight = format!("gru.weight_ih_l{layer}");
        if !weights.contains_key(&input_weight) {
            break;
        }
        let recurrent_input_width = if layer == 0 {
            encoder_width
        } else {
            hidden_width
        };
        for (name, shape) in [
            (
                input_weight.clone(),
                vec![gates * hidden_width, recurrent_input_width],
            ),
            (
                format!("gru.weight_hh_l{layer}"),
                vec![gates * hidden_width, hidden_width],
            ),
            (format!("gru.bias_ih_l{layer}"), vec![gates * hidden_width]),
            (format!("gru.bias_hh_l{layer}"), vec![gates * hidden_width]),
        ] {
            require_final_shape(&weights, &name, &shape)?;
            allowed.insert(name);
        }
        layer += 1;
    }
    if layer == 0 {
        return Err(PyValueError::new_err(
            "final predictor needs at least one recurrent layer",
        ));
    }

    let head_shape = final_tensor_shape(&weights, "head.0.weight")?;
    if head_shape.len() != 2 || head_shape[0] == 0 || head_shape[1] != hidden_width + encoder_width
    {
        return Err(PyValueError::new_err(format!(
            "head.0.weight must have shape [readout_width, {}], got {head_shape:?}",
            hidden_width + encoder_width
        )));
    }
    let readout_width = head_shape[0];
    if readout_width != FINAL_TL3_HIDDEN_WIDTHS[role] {
        return Err(PyValueError::new_err(format!(
            "final member {role} readout width must be {}, got {readout_width}",
            FINAL_TL3_HIDDEN_WIDTHS[role]
        )));
    }
    require_final_shape(&weights, "head.0.bias", &[readout_width])?;
    require_final_shape(&weights, "head.2.weight", &[2, readout_width])?;
    require_final_shape(&weights, "head.2.bias", &[2])?;
    let tied = role == 1 || role == 2;
    if tied {
        let first = FINAL_TIED_KEYS[0];
        let first_shape = final_tensor_shape(&weights, first)?;
        if first_shape != [128, 104] {
            return Err(PyValueError::new_err(format!(
                "tied.0.weight must have shape [128, 104], got {first_shape:?}"
            )));
        }
        for (name, shape) in [
            ("tied.0.bias", vec![128]),
            ("tied.2.weight", vec![64, 128]),
            ("tied.2.bias", vec![64]),
            ("tied_out.weight", vec![encoder_width, 128]),
            ("tied_out.bias", vec![encoder_width]),
            ("lead_out.weight", vec![encoder_width, 64]),
            ("lead_out.bias", vec![encoder_width]),
        ] {
            require_final_shape(&weights, name, &shape)?;
            allowed.insert(name.to_owned());
        }
        allowed.insert("tied.0.weight".into());
    }
    let mut unknown: Vec<String> = weights
        .keys()
        .filter(|name| !allowed.contains(*name))
        .cloned()
        .collect();
    unknown.sort();
    if !unknown.is_empty() {
        return Err(PyValueError::new_err(format!(
            "unsupported final predictor tensors: {}",
            unknown.join(", ")
        )));
    }

    let tied_weights = if tied {
        let width = encoder_width;
        let tied_w1 = take_final_array(&mut weights, "tied.0.weight")?;
        let tied_b1 = take_final_array(&mut weights, "tied.0.bias")?;
        let tied_w2 = take_final_array(&mut weights, "tied.2.weight")?;
        let tied_b2 = take_final_array(&mut weights, "tied.2.bias")?;
        let tied_out_w = take_final_array(&mut weights, "tied_out.weight")?;
        let tied_out_b = take_final_array(&mut weights, "tied_out.bias")?;
        let lead_out_w = take_final_array(&mut weights, "lead_out.weight")?;
        let lead_out_b = take_final_array(&mut weights, "lead_out.bias")?;
        Some(
            TiedLeadWeights::new(
                width, tied_w1, tied_b1, tied_w2, tied_b2, tied_out_w, tied_out_b, lead_out_w,
                lead_out_b,
            )
            .map_err(PyValueError::new_err)?,
        )
    } else {
        None
    };
    Ok((weights_from_dict(weights)?, tied_weights))
}


fn min_gru_array_shape(
    weights: &HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    key: &str,
) -> PyResult<Vec<usize>> {
    weights
        .get(key)
        .map(|array| array.shape().to_vec())
        .ok_or_else(|| PyValueError::new_err(format!("missing minGRU tensor {key}")))
}

fn take_min_gru_array(
    weights: &mut HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    key: &str,
    expected: &[usize],
) -> PyResult<Vec<f32>> {
    let array = weights
        .remove(key)
        .ok_or_else(|| PyValueError::new_err(format!("missing minGRU tensor {key}")))?;
    if array.shape() != expected {
        return Err(PyValueError::new_err(format!(
            "minGRU tensor {key} must have shape {expected:?}, got {:?}",
            array.shape()
        )));
    }
    Ok(array
        .as_slice()
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .to_vec())
}

fn min_gru_weights_from_dict(
    mut weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
) -> PyResult<MinGruWeights> {
    let n_in = 211;
    let mean = take_min_gru_array(&mut weights, "mean", &[n_in])?;
    let std = take_min_gru_array(&mut weights, "std", &[n_in])?;
    let inp_shape = min_gru_array_shape(&weights, "inp.weight")?;
    if inp_shape.len() != 2 || inp_shape[1] != n_in || inp_shape[0] == 0 {
        return Err(PyValueError::new_err(
            "minGRU inp.weight must have shape [encoder_width, 211]",
        ));
    }
    let encoder = inp_shape[0];
    let inp_weight = take_min_gru_array(&mut weights, "inp.weight", &inp_shape)?;
    let inp_bias = take_min_gru_array(&mut weights, "inp.bias", &[encoder])?;

    let mut cells = Vec::new();
    loop {
        let index = cells.len();
        let weight_key = format!("cells.{index}.input_projection.weight");
        if !weights.contains_key(&weight_key) {
            break;
        }
        let weight_shape = min_gru_array_shape(&weights, &weight_key)?;
        if weight_shape.len() != 2 || weight_shape[0] == 0 || weight_shape[0] % 2 != 0 {
            return Err(PyValueError::new_err(format!(
                "minGRU tensor {weight_key} must have shape [2 * recurrent_width, input_width]"
            )));
        }
        let recurrent = weight_shape[0] / 2;
        let input_size = if index == 0 { encoder } else { recurrent };
        if weight_shape[1] != input_size {
            return Err(PyValueError::new_err(format!(
                "minGRU tensor {weight_key} must have {input_size} input columns"
            )));
        }
        let bias_key = format!("cells.{index}.input_projection.bias");
        let projection = take_min_gru_array(&mut weights, &weight_key, &weight_shape)?;
        let bias = take_min_gru_array(&mut weights, &bias_key, &[2 * recurrent])?;
        cells.push(MinGruCellWeights {
            input_size,
            projection,
            bias,
        });
    }
    if cells.is_empty() {
        return Err(PyValueError::new_err("minGRU needs at least one cell"));
    }
    let recurrent = cells[0].bias.len() / 2;
    let head0_shape = min_gru_array_shape(&weights, "head.0.weight")?;
    if head0_shape.len() != 2 || head0_shape[0] == 0 {
        return Err(PyValueError::new_err(
            "minGRU head.0.weight must have shape [readout_width, recurrent_width + encoder_width]",
        ));
    }
    let readout = head0_shape[0];
    if head0_shape[1] != recurrent + encoder {
        return Err(PyValueError::new_err(format!(
            "minGRU head.0.weight must have {} input columns",
            recurrent + encoder
        )));
    }
    let head0_weight = take_min_gru_array(&mut weights, "head.0.weight", &head0_shape)?;
    let head0_bias = take_min_gru_array(&mut weights, "head.0.bias", &[readout])?;
    let head2_weight = take_min_gru_array(&mut weights, "head.2.weight", &[2, readout])?;
    let head2_bias = take_min_gru_array(&mut weights, "head.2.bias", &[2])?;
    if !weights.is_empty() {
        let mut extra: Vec<_> = weights.into_keys().collect();
        extra.sort();
        return Err(PyValueError::new_err(format!(
            "unexpected minGRU tensors: {}",
            extra.join(", ")
        )));
    }
    Ok(MinGruWeights {
        mean,
        std,
        inp_weight,
        inp_bias,
        cells,
        head0_weight,
        head0_bias,
        head2_weight,
        head2_bias,
    })
}

fn min_gru_model_from_dict(
    weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
    fast_activations: bool,
    precision: &str,
) -> PyResult<MinGruModel> {
    let activations = if fast_activations {
        Activations::Fast
    } else {
        Activations::Exact
    };
    let precision = MinGruPrecision::parse(precision).map_err(PyValueError::new_err)?;
    MinGruModel::new(min_gru_weights_from_dict(weights)?, activations, precision)
        .map_err(PyValueError::new_err)
}

fn min_gru_storage_bytes(model: &MinGruModel) -> HashMap<String, usize> {
    let storage = model.storage_bytes();
    HashMap::from([
        ("matrix_weights".into(), storage.matrix_weights),
        ("biases".into(), storage.biases),
        ("normalization".into(), storage.normalization),
        ("recurrent_state".into(), storage.recurrent_state),
        ("scratch".into(), storage.scratch),
    ])
}

fn min_gru_weight_bytes(model: &MinGruModel) -> usize {
    let storage = model.storage_bytes();
    storage.matrix_weights + storage.biases + storage.normalization
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
    if weights.iter().any(|member| {
        let recurrent = member.layers.first().map_or(0, |layer| layer.b_hh.len() / 3);
        member.b_in.len() != recurrent
            || member.b1.len() != recurrent
            || !member.w_mid.is_empty()
            || !member.b_mid.is_empty()
    }) {
        return Err(PyValueError::new_err(
            "shared_backbone currently requires equal-width one-hidden heads",
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

/// Direct model-step API for already engineered 211-column inputs.
#[pyclass(name = "MinGruStepper")]
struct PyMinGruStepper {
    model: MinGruModel,
}

#[pymethods]
impl PyMinGruStepper {
    #[new]
    #[pyo3(signature = (weights, fast_activations=true, precision="f32"))]
    fn new(
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        fast_activations: bool,
        precision: &str,
    ) -> PyResult<Self> {
        let model = min_gru_model_from_dict(weights, fast_activations, precision)?;
        Ok(Self { model })
    }

    fn reset(&mut self) {
        self.model.reset();
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input = input
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if input.len() != 211 {
            return Err(PyValueError::new_err(format!(
                "expected 211 engineered features, got {}",
                input.len()
            )));
        }
        let mut output = [0.0f32; 2];
        self.model.step(input, &mut output);
        Ok(PyArray1::from_slice(py, &output))
    }

    /// Run a chunk of already engineered rows while preserving state across calls.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        inputs: numpy::PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        if inputs.shape()[1] != 211 {
            return Err(PyValueError::new_err(format!(
                "expected 211 engineered features, got {}",
                inputs.shape()[1]
            )));
        }
        let nrows = inputs.shape()[0];
        let inputs = inputs
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut outputs = vec![0.0f32; nrows * 2];
        for (input, output) in inputs.chunks_exact(211).zip(outputs.chunks_exact_mut(2)) {
            self.model.step(input, output);
        }
        PyArray1::from_vec(py, outputs).reshape([nrows, 2])
    }

    #[getter]
    fn hidden_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        PyArray1::from_vec(py, self.model.hidden_state().to_vec())
            .reshape([self.model.layers(), self.model.recurrent_width()])
    }

    #[getter]
    fn storage_bytes(&self) -> HashMap<String, usize> {
        min_gru_storage_bytes(&self.model)
    }

    #[getter]
    fn weight_bytes(&self) -> usize {
        min_gru_weight_bytes(&self.model)
    }

    #[getter]
    fn precision(&self) -> &'static str {
        self.model.precision().name()
    }

    #[getter]
    fn fast_activations(&self) -> bool {
        self.model.fast_activations()
    }

    #[getter]
    fn avx2_fastpath_enabled(&self) -> bool {
        self.model.avx2_fastpath_enabled()
    }
}

/// Competition callback prototype: accepts one raw 112-feature row and builds its causal
/// 99-column event features before stepping the minGRU model on 211 columns.
#[pyclass(name = "MinGruPredictor")]
struct PyMinGruPredictor {
    features: crate::stream::StreamFeatures,
    model: MinGruModel,
    input: Vec<f32>,
}

fn min_gru_raw_step(
    features: &mut crate::stream::StreamFeatures,
    model: &mut MinGruModel,
    input: &mut [f32],
    row: &[f32],
    output: &mut [f32],
) -> PyResult<()> {
    if row.len() != crate::stream::N_RAW {
        return Err(PyValueError::new_err(format!(
            "expected 112 raw features, got {}",
            row.len()
        )));
    }
    let (raw, event) = input.split_at_mut(crate::stream::N_RAW);
    raw.copy_from_slice(row);
    features.sanitize(raw);
    features.update(raw, event);
    model.step(input, output);
    Ok(())
}

#[pymethods]
impl PyMinGruPredictor {
    #[new]
    #[pyo3(signature = (weights, fast_activations=true, precision="f32"))]
    fn new(
        weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>,
        fast_activations: bool,
        precision: &str,
    ) -> PyResult<Self> {
        let model = min_gru_model_from_dict(weights, fast_activations, precision)?;
        Ok(Self {
            features: Default::default(),
            model,
            input: vec![0.0; crate::stream::N_RAW + crate::stream::N_EVENT],
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
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut output = [0.0f32; 2];
        min_gru_raw_step(
            &mut self.features,
            &mut self.model,
            &mut self.input,
            row,
            &mut output,
        )?;
        Ok(PyArray1::from_slice(py, &output))
    }

    /// Batched test/measurement helper; it executes the same row callback internally.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        rows: numpy::PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        if rows.shape()[1] != crate::stream::N_RAW {
            return Err(PyValueError::new_err(format!(
                "expected 112 raw features, got {}",
                rows.shape()[1]
            )));
        }
        let nrows = rows.shape()[0];
        let rows = rows
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut outputs = vec![0.0f32; nrows * 2];
        for (row, output) in rows
            .chunks_exact(crate::stream::N_RAW)
            .zip(outputs.chunks_exact_mut(2))
        {
            min_gru_raw_step(
                &mut self.features,
                &mut self.model,
                &mut self.input,
                row,
                output,
            )?;
        }
        PyArray1::from_vec(py, outputs).reshape([nrows, 2])
    }

    #[getter]
    fn hidden_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        PyArray1::from_vec(py, self.model.hidden_state().to_vec())
            .reshape([self.model.layers(), self.model.recurrent_width()])
    }

    #[getter]
    fn storage_bytes(&self) -> HashMap<String, usize> {
        let mut storage = min_gru_storage_bytes(&self.model);
        storage.insert(
            "feature_builder_state".into(),
            std::mem::size_of::<crate::stream::StreamFeatures>() + self.input.len() * 4,
        );
        storage
    }

    #[getter]
    fn weight_bytes(&self) -> usize {
        min_gru_weight_bytes(&self.model)
    }

    #[getter]
    fn precision(&self) -> &'static str {
        self.model.precision().name()
    }

    #[getter]
    fn fast_activations(&self) -> bool {
        self.model.fast_activations()
    }

    #[getter]
    fn avx2_fastpath_enabled(&self) -> bool {
        self.model.avx2_fastpath_enabled()
    }
}

#[pymodule]
fn wnn_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGruStepper>()?;
    m.add_class::<PyMinGruStepper>()?;
    m.add_class::<PyMinGruPredictor>()?;
    m.add_class::<PyPredictor>()?;
    m.add_class::<PyEnsemblePredictor>()?;
    m.add_class::<PyFinalTl3Predictor>()?;
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

/// Four-member chronological TL3 + raw112 LSTM predictor with exact staged blending.
#[pyclass(name = "FinalTl3Predictor")]
struct PyFinalTl3Predictor {
    features: crate::stream::StreamFeatures,
    models: Vec<FastRnn>,
    tied: [TiedLeadEncoder; 2],
    tied_inputs: [Vec<f32>; 2],
    tied_restored_inputs: [Vec<f32>; 2],
    tied_residuals: [Vec<f32>; 2],
    w18_input: [f32; 211],
    raw: [f32; 112],
    event: [f32; 99],
    member_outputs: [[f32; 2]; 4],
    sequence: Option<i64>,
    next_step: usize,
    fast_activations: bool,
}

fn final_tl3_blend(members: &[[f32; 2]; 4]) -> [f32; 2] {
    std::array::from_fn(|target| {
        let mut baseline = 0.0f64;
        baseline += f64::from(members[0][target]) * 0.34;
        baseline += f64::from(members[1][target]) * 0.33;
        baseline += f64::from(members[2][target]) * 0.33;
        let baseline = baseline as f32;

        let mut final_value = 0.0f64;
        final_value += f64::from(baseline) * 0.9;
        final_value += f64::from(members[3][target]) * 0.1;
        final_value as f32
    })
}

fn final_members_array<'py>(
    py: Python<'py>,
    members: &[[f32; 2]; 4],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let flat = members.iter().flat_map(|row| row.iter().copied()).collect();
    Ok(PyArray1::from_vec(py, flat).reshape([4, 2])?)
}

impl PyFinalTl3Predictor {
    fn reset_sequence_state(&mut self) {
        self.features.reset();
        for model in &mut self.models {
            model.reset();
        }
        for tied in &mut self.tied {
            tied.reset();
        }
        self.member_outputs = [[0.0; 2]; 4];
        self.next_step = 0;
    }

    fn advance_row(&mut self) -> PyResult<()> {
        self.features.sanitize(&mut self.raw);
        self.features.update(&self.raw, &mut self.event);
        self.w18_input[..112].copy_from_slice(&self.raw);
        self.w18_input[112..].copy_from_slice(&self.event);

        self.models[0].step(&self.w18_input, &mut self.member_outputs[0]);
        self.tied[0]
            .assemble_input(&self.raw, &self.event, &mut self.tied_inputs[0])
            .map_err(PyValueError::new_err)?;
        self.models[1]
            .restore_normalized_input(&self.tied_inputs[0], &mut self.tied_restored_inputs[0]);
        self.tied[0]
            .encoder_residual(&self.tied_restored_inputs[0], &mut self.tied_residuals[0])
            .map_err(PyValueError::new_err)?;
        self.models[1]
            .step_with_encoder_residual(
                &self.tied_inputs[0],
                &self.tied_residuals[0],
                &mut self.member_outputs[1],
            )
            .map_err(|error| PyValueError::new_err(error.0))?;
        self.tied[1]
            .assemble_input(&self.raw, &self.event, &mut self.tied_inputs[1])
            .map_err(PyValueError::new_err)?;
        self.models[2]
            .restore_normalized_input(&self.tied_inputs[1], &mut self.tied_restored_inputs[1]);
        self.tied[1]
            .encoder_residual(&self.tied_restored_inputs[1], &mut self.tied_residuals[1])
            .map_err(PyValueError::new_err)?;
        self.models[2]
            .step_with_encoder_residual(
                &self.tied_inputs[1],
                &self.tied_residuals[1],
                &mut self.member_outputs[2],
            )
            .map_err(|error| PyValueError::new_err(error.0))?;
        self.models[3].step(&self.raw, &mut self.member_outputs[3]);

        if self
            .member_outputs
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(PyValueError::new_err(
                "final predictor produced a non-finite member score",
            ));
        }
        self.next_step += 1;
        Ok(())
    }

    fn process_data_point(
        &mut self,
        py: Python<'_>,
        data_point: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let sequence: i64 = data_point.getattr(intern!(py, "seq_ix"))?.extract()?;
        let step: usize = data_point.getattr(intern!(py, "step_in_seq"))?.extract()?;
        if self.sequence != Some(sequence) {
            if step != 0 {
                return Err(PyValueError::new_err(
                    "final predictor requires each new sequence to begin at step 0",
                ));
            }
            self.reset_sequence_state();
            self.sequence = Some(sequence);
        } else if step != self.next_step {
            return Err(PyValueError::new_err(format!(
                "final predictor expected step {}, received {step}",
                self.next_step
            )));
        }
        let state = data_point.getattr(intern!(py, "state"))?;
        read_row(py, &state, &mut self.raw)?;
        self.advance_row()?;
        data_point
            .getattr(intern!(py, "need_prediction"))?
            .is_truthy()
    }
}

#[pymethods]
impl PyFinalTl3Predictor {
    #[new]
    #[pyo3(signature = (weights, precision="f32", q8_codes=None, fast_activations=false))]
    fn new(
        py: Python<'_>,
        weights: Vec<HashMap<String, PyReadonlyArrayDyn<'_, f32>>>,
        precision: &str,
        q8_codes: Option<Vec<Q8Codes>>,
        fast_activations: bool,
    ) -> PyResult<Self> {
        if weights.len() != 4 {
            return Err(PyValueError::new_err(
                "final predictor requires ordered weights [W18, tiedlead_s5, tiedlead_s6, LSTM]",
            ));
        }
        let precision = parse_precision(precision)?;
        let activations = if fast_activations {
            Activations::Fast
        } else {
            Activations::Exact
        };
        let dynamic_q8 = matches!(
            precision,
            Precision::Q8Vnni
                | Precision::Q8VnniAvx2
                | Precision::Q8VnniPortable
                | Precision::Q8Vnni256
                | Precision::Q8VnniBlocked
                | Precision::Q8VnniFused
        );
        if precision != Precision::F32 && !dynamic_q8 {
            return Err(PyValueError::new_err(
                "final predictor supports f32 or dynamic q8vnni precision",
            ));
        }
        if q8_codes.as_ref().is_some_and(|codes| codes.len() != 4) {
            return Err(PyValueError::new_err(
                "final predictor q8_codes must contain one map per ordered member",
            ));
        }
        if q8_codes.is_some() && precision == Precision::F32 {
            return Err(PyValueError::new_err(
                "final predictor q8_codes require dynamic q8vnni precision",
            ));
        }
        let mut q8_codes = q8_codes.map(Vec::into_iter);
        let mut models = Vec::with_capacity(4);
        let mut tied = Vec::with_capacity(2);
        for (role, member) in weights.into_iter().enumerate() {
            let (rnn, tied_weights) = parse_final_member(member, role)?;
            let mut model = FastRnn::with_precision(rnn, activations, precision)
                .map_err(|error| PyValueError::new_err(error.0))?;
            if let Some(codes) = q8_codes.as_mut() {
                import_q8(
                    py,
                    &mut model,
                    codes
                        .next()
                        .ok_or_else(|| PyValueError::new_err("missing final member q8 codes"))?,
                )?;
            }
            if model.n_in() != FINAL_TL3_WIDTHS[role] || model.n_out() != 2 {
                return Err(PyValueError::new_err(format!(
                    "final member {role} has incompatible input or output width"
                )));
            }
            let expected_cell = if role == 3 {
                crate::rnn_fast::Cell::Lstm
            } else {
                crate::rnn_fast::Cell::Gru
            };
            let (layers, hidden_width) = model.hidden_state_shape();
            if model.cell() != expected_cell
                || model.encoder_width() != FINAL_TL3_ENCODER_WIDTHS[role]
                || model.hidden() != FINAL_TL3_HIDDEN_WIDTHS[role]
                || layers != FINAL_TL3_LAYER_COUNTS[role]
            {
                return Err(PyValueError::new_err(format!(
                    "final member {role} cell/width/layer layout is unsupported: cell={:?}, encoder={}, hidden={hidden_width}, layers={layers}",
                    model.cell(),
                    model.encoder_width()
                )));
            }
            if let Some(tied_weights) = tied_weights {
                if model.encoder_width() != final_tl3_tied_encoder_width(&tied_weights) {
                    return Err(PyValueError::new_err(
                        "tiedlead residual width differs from the model encoder width",
                    ));
                }
                tied.push(TiedLeadEncoder::with_activations(
                    tied_weights,
                    activations,
                ));
            }
            models.push(model);
        }
        let [tied_s5, tied_s6]: [TiedLeadEncoder; 2] = tied
            .try_into()
            .map_err(|_| PyValueError::new_err("both final tiedlead branches are required"))?;
        let tied_inputs = std::array::from_fn(|_| vec![0.0; crate::tiedlead::INPUT_WIDTH]);
        let tied_restored_inputs = std::array::from_fn(|_| vec![0.0; crate::tiedlead::INPUT_WIDTH]);
        let tied_residuals = [
            vec![0.0; models[1].encoder_width()],
            vec![0.0; models[2].encoder_width()],
        ];
        let _ = py;
        Ok(Self {
            features: Default::default(),
            models,
            tied: [tied_s5, tied_s6],
            tied_inputs,
            tied_restored_inputs,
            tied_residuals,
            w18_input: [0.0; 211],
            raw: [0.0; 112],
            event: [0.0; 99],
            member_outputs: [[0.0; 2]; 4],
            sequence: None,
            next_step: 0,
            fast_activations,
        })
    }

    /// Reset all recurrent and causal feature state.
    fn reset(&mut self) {
        self.reset_sequence_state();
        self.sequence = None;
    }

    /// Scorer callback; advances every member, returning None only for warm-up rows.
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        data_point: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        if !self.process_data_point(py, data_point)? {
            return Ok(None);
        }
        let result = final_tl3_blend(&self.member_outputs);
        Ok(Some(PyArray1::from_slice(py, &result)))
    }

    /// Advance the same callback path and return all four raw member outputs for parity checks.
    fn step_members<'py>(
        &mut self,
        py: Python<'py>,
        data_point: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.process_data_point(py, data_point)?;
        final_members_array(py, &self.member_outputs)
    }

    /// Apply the exact nested TL3-then-LSTM FP64 blend and FP32 casts.
    #[staticmethod]
    fn blend_members<'py>(
        py: Python<'py>,
        members: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let values = members.as_array();
        if values.shape() != [4, 2] {
            return Err(PyValueError::new_err(
                "member scores must have shape (4, 2)",
            ));
        }
        let rows: [[f32; 2]; 4] =
            std::array::from_fn(|index| [values[[index, 0]], values[[index, 1]]]);
        if rows.iter().flatten().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("member scores must be finite"));
        }
        Ok(PyArray1::from_slice(py, &final_tl3_blend(&rows)))
    }

    #[getter]
    fn precision(&self) -> &'static str {
        let first = self.models[0].precision();
        if self.models.iter().all(|model| model.precision() == first) {
            precision_name(first)
        } else {
            "mixed"
        }
    }

    #[getter]
    fn fast_activations(&self) -> bool {
        self.fast_activations
    }

    fn quantization_codes(&self, py: Python<'_>) -> PyResult<Vec<Q8Codes>> {
        if !self.models.iter().all(FastRnn::is_dynamic_int8) {
            return Err(PyValueError::new_err(
                "final predictor quantization_codes requires dynamic q8vnni precision",
            ));
        }
        self.models
            .iter()
            .map(|model| export_q8(py, model))
            .collect()
    }

    #[getter]
    fn member_widths(&self) -> Vec<usize> {
        self.models.iter().map(FastRnn::n_in).collect()
    }

    #[getter]
    fn member_weight_bytes(&self) -> Vec<usize> {
        self.models.iter().map(FastRnn::weight_bytes).collect()
    }

    #[getter]
    fn member_dynamic_q8_weight_bytes(&self) -> Vec<usize> {
        self.models
            .iter()
            .map(|model| {
                model
                    .dyn8_matrices()
                    .map(|(_, matrix)| matrix.storage_bytes())
                    .sum()
            })
            .collect()
    }
}

fn final_tl3_tied_encoder_width(weights: &TiedLeadWeights) -> usize {
    weights.encoder_width()
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
            w_mid: Vec::new(),
            b_mid: Vec::new(),
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
            w_mid: Vec::new(),
            b_mid: Vec::new(),
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
