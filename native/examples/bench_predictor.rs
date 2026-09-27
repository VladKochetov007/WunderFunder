//! Engine-only cost of the streaming Predictor pipeline (features + FastRnn), without and with
//! online-norm inputs, on real rows (little-endian f32, 112 per row):
//!   cargo run --release --example bench_predictor -- rows.f32 [hidden] [passes]
use std::hint::black_box;
use std::time::Instant;
use wnn_native::rnn::{GruLayer, RnnWeights};
use wnn_native::rnn_fast::{Activations, FastRnn};
use wnn_native::stream::{N_EVENT, N_RAW, OnlineNorm, StreamFeatures};

fn weights(n_in: usize, h: usize) -> RnnWeights {
    let mut s = 1u64;
    let mut v = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / 16777216.0 - 0.5) * 0.2
            })
            .collect()
    };
    RnnWeights {
        mean: v(n_in),
        std: vec![1.0; n_in],
        w_in: v(h * n_in),
        b_in: v(h),
        layers: (0..2)
            .map(|_| GruLayer {
                w_ih: v(3 * h * h),
                w_hh: v(3 * h * h),
                b_ih: v(3 * h),
                b_hh: v(3 * h),
            })
            .collect(),
        w1: v(2 * h * h),
        b1: v(h),
        w2: v(2 * h),
        b2: v(2),
    }
}

struct Pipeline {
    features: StreamFeatures,
    norm: Option<OnlineNorm>,
    model: FastRnn,
    input: Vec<f32>,
}

impl Pipeline {
    fn new(online: bool, h: usize) -> Self {
        let n_in = N_RAW + N_EVENT + if online { N_RAW } else { 0 };
        Self {
            features: StreamFeatures::default(),
            norm: online.then(OnlineNorm::default),
            model: FastRnn::new(weights(n_in, h), Activations::Fast).unwrap(),
            input: vec![0.0; n_in],
        }
    }

    /// Mirrors PyPredictor::step.
    fn run(&mut self, rows: &[f32]) -> f64 {
        self.features.reset();
        if let Some(n) = &mut self.norm {
            n.reset();
        }
        self.model.reset();
        let mut out = [0.0f32; 2];
        let mut acc = 0.0f32;
        let t = Instant::now();
        for row in rows.chunks_exact(N_RAW) {
            let (raw, rest) = self.input.split_at_mut(N_RAW);
            raw.copy_from_slice(black_box(row));
            self.features.sanitize(raw);
            let (event, z) = rest.split_at_mut(N_EVENT);
            self.features.update(raw, event);
            if let Some(n) = &mut self.norm {
                n.update(raw, z);
            }
            self.model.step(&self.input, &mut out);
            acc += out[0];
        }
        black_box(acc);
        t.elapsed().as_secs_f64() / (rows.len() / N_RAW) as f64 * 1e6
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&a[0]).expect("rows file");
    let rows: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let h = a.get(1).map_or(128, |s| s.parse().unwrap());
    let passes = a.get(2).map_or(15, |s| s.parse().unwrap());
    let mut cases = [
        ("linear 211", Pipeline::new(false, h)),
        ("online 323", Pipeline::new(true, h)),
    ];
    let mut t = vec![Vec::new(); cases.len()];
    for _ in 0..passes {
        for (k, (_, p)) in cases.iter_mut().enumerate() {
            t[k].push(p.run(&rows));
        }
    }
    let mut norm = OnlineNorm::default();
    let mut z = [0.0f32; N_RAW];
    let n_rows = rows.len() / N_RAW;
    let mut tn = Vec::new();
    for _ in 0..passes {
        norm.reset();
        let s = Instant::now();
        for row in rows.chunks_exact(N_RAW) {
            norm.update(black_box(row), &mut z);
            black_box(&z);
        }
        tn.push(s.elapsed().as_secs_f64() / n_rows as f64 * 1e6);
    }
    let names = cases.iter().map(|c| c.0).chain(["OnlineNorm only"]);
    for (name, mut v) in names.zip(t.into_iter().chain([tn])) {
        v.sort_by(f64::total_cmp);
        println!(
            "{name:16} min {:.3} median {:.3} us/row ({passes} passes x {n_rows} rows, hidden {h})",
            v[0],
            v[v.len() / 2]
        );
    }
}
