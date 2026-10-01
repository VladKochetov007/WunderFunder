//! Per-row cost of the native streaming path (features + GRU members), no Python.
//!
//!   cargo run --release --example bench_engine -- [--rows FILE] [--reps N] SPEC [SPEC ...]
//!   cargo run --release --example bench_engine -- --kernel
//!
//! SPEC is `[lstm]hidden[xlayers]:f32|f16|i16` (GRU, 2 layers by default), e.g. `128:f16 lstm128:i16`. FILE holds
//! little-endian f32 raw rows (112 per row), e.g. one valid sequence; without it a synthetic
//! 20k-row sequence is used. Reports features alone, each member alone and the whole ensemble
//! stepped member by member per row, as `EnsemblePredictor` does.
use std::hint::black_box;
use std::time::Instant;

use wnn_native::rnn::{GruLayer, RnnWeights};
use wnn_native::rnn_fast::{Activations, FastRnn, Precision};
use wnn_native::stream::{N_EVENT, N_RAW, StreamFeatures};

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / 16777216.0
    }

    /// Uniform with the standard deviation of PyTorch's default init for `fan_in`.
    fn init(&mut self, len: usize, fan_in: usize) -> Vec<f32> {
        let bound = 1.0 / (fan_in as f32).sqrt();
        (0..len)
            .map(|_| (2.0 * self.uniform() - 1.0) * bound)
            .collect()
    }
}

fn weights(n_in: usize, h: usize, layers: usize, gates: usize, seed: u64) -> RnnWeights {
    let mut r = Rng(seed);
    RnnWeights {
        mean: vec![0.0; n_in],
        std: vec![1.0; n_in],
        w_in: r.init(h * n_in, n_in),
        b_in: r.init(h, n_in),
        layers: (0..layers)
            .map(|_| GruLayer {
                w_ih: r.init(gates * h * h, h),
                w_hh: r.init(gates * h * h, h),
                b_ih: r.init(gates * h, h),
                b_hh: r.init(gates * h, h),
            })
            .collect(),
        w1: r.init(2 * h * h, 2 * h),
        b1: r.init(h, 2 * h),
        w2: r.init(2 * h, h),
        b2: r.init(2, h),
    }
}

/// Synthetic raw rows: every column a random walk that moves on ~30% of rows.
fn synthetic(rows: usize) -> Vec<[f32; N_RAW]> {
    let mut r = Rng(3);
    let mut cur = [0.0f32; N_RAW];
    (0..rows)
        .map(|_| {
            for v in cur.iter_mut() {
                if r.uniform() < 0.3 {
                    *v += 2.0 * r.uniform() - 1.0;
                }
            }
            cur
        })
        .collect()
}

fn read_rows(path: &str) -> Vec<[f32; N_RAW]> {
    let bytes = std::fs::read(path).expect("rows file");
    bytes
        .chunks_exact(4 * N_RAW)
        .map(|row| {
            std::array::from_fn(|c| f32::from_le_bytes(row[4 * c..4 * c + 4].try_into().unwrap()))
        })
        .collect()
}

/// Precomputed [raw | event] inputs so member timings exclude features.
fn inputs(raw: &[[f32; N_RAW]]) -> Vec<f32> {
    let mut f = StreamFeatures::default();
    let mut out = Vec::with_capacity(raw.len() * (N_RAW + N_EVENT));
    let mut x = [0.0f32; N_RAW + N_EVENT];
    for r in raw {
        let (a, b) = x.split_at_mut(N_RAW);
        a.copy_from_slice(r);
        f.update(a, b);
        out.extend_from_slice(&x);
    }
    out
}

fn per_row(reps: usize, rows: usize, mut body: impl FnMut()) -> f64 {
    body(); // warm caches and branch predictors
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        body();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best / rows as f64 * 1e6
}

fn parse(spec: &str) -> (usize, usize, usize, Precision) {
    let (shape, prec) = spec.split_once(':').unwrap_or((spec, "f32"));
    let (gates, shape) = match shape.strip_prefix("lstm") {
        Some(rest) => (4, rest),
        None => (3, shape),
    };
    let (h, l) = shape.split_once('x').unwrap_or((shape, "2"));
    let prec = match prec {
        "f16" => Precision::F16,
        "i16" => Precision::I16,
        "f32" => Precision::F32,
        other => panic!("precision {other}"),
    };
    (h.parse().unwrap(), l.parse().unwrap(), gates, prec)
}

fn kernel_bench() {
    use wnn_native::half::{HalfArena, gemv_pf};
    // Eight 3n x n matrices for n = 256: 1.6 MB of halves, beyond a 1.25 MB L2, cycled like
    // the layers of an ensemble.
    let (n, mats) = (256usize, 8usize);
    let mut r = Rng(9);
    let w: Vec<Vec<f32>> = (0..mats).map(|_| r.init(3 * n * n, n)).collect();
    let mut arena = HalfArena::new();
    let ranges: Vec<_> = w.iter().map(|m| arena.push_panel(m, 3 * n, n)).collect();
    let b = vec![0.0f32; 3 * n];
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut out = vec![0.0f32; 3 * n];
    let reps = 400;
    macro_rules! run {
        ($pf:expr) => {{
            let us = per_row(5, reps, || {
                for _ in 0..reps {
                    for rg in &ranges {
                        unsafe {
                            gemv_pf::<$pf>(&arena.as_ref()[rg.clone()], &b, black_box(&x), &mut out)
                        };
                    }
                    black_box(&mut out);
                }
            });
            println!(
                "f16 panel PF={:>4} halves: {:.2} us per {} matvecs ({:.1} GB/s)",
                $pf,
                us,
                mats,
                (mats * 3 * n * n * 2) as f64 / us / 1e3
            );
        }};
    }
    run!(0);
    run!(256);
    run!(512);
    run!(1024);
    run!(2048);
}

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    let mut raw = None;
    let mut reps = 5;
    let mut specs = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kernel" => return kernel_bench(),
            "--rows" => raw = Some(read_rows(&args.next().unwrap())),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            _ => specs.push(a),
        }
    }
    let raw = raw.unwrap_or_else(|| synthetic(20_000));
    let rows = raw.len();
    let n_in = N_RAW + N_EVENT;
    let xs = inputs(&raw);
    let zero_share = xs.iter().filter(|v| **v == 0.0).count() as f64 / xs.len() as f64;
    println!(
        "rows {rows}, input zeros {:.1}%, f16 kernel available {}",
        zero_share * 100.0,
        wnn_native::half::available()
    );

    let mut features = StreamFeatures::default();
    let mut x = [0.0f32; N_RAW + N_EVENT];
    let us = per_row(reps, rows, || {
        features.reset();
        for r in &raw {
            let (a, b) = x.split_at_mut(N_RAW);
            a.copy_from_slice(r);
            features.sanitize(a);
            features.update(a, b);
            black_box(&mut x);
        }
    });
    println!("features only: {us:.2} us/row");

    let mut models: Vec<FastRnn> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (h, l, gates, p) = parse(s);
            let m = FastRnn::with_precision(
                weights(n_in, h, l, gates, 11 + i as u64),
                Activations::Fast,
                p,
            )
            .unwrap();
            assert_eq!(m.precision(), p, "{s}: requested precision unavailable");
            m
        })
        .collect();
    let mut out = [0.0f32; 2];
    for (m, s) in models.iter_mut().zip(&specs) {
        let us = per_row(reps, rows, || {
            m.reset();
            for x in xs.chunks_exact(n_in) {
                m.step(black_box(x), &mut out);
            }
            black_box(&out);
        });
        println!(
            "{s:>10} alone: {us:.2} us/row ({} KiB weights)",
            m.weight_bytes() / 1024
        );
    }
    if models.len() > 1 {
        let us = per_row(reps, rows, || {
            for m in models.iter_mut() {
                m.reset();
            }
            for x in xs.chunks_exact(n_in) {
                for m in models.iter_mut() {
                    m.step(black_box(x), &mut out);
                }
            }
            black_box(&out);
        });
        let kib: usize = models.iter().map(|m| m.weight_bytes()).sum::<usize>() / 1024;
        println!(
            "ensemble {}: {us:.2} us/row ({kib} KiB weights)",
            specs.join("+")
        );
    }
}
