//! Per-row cost of the native streaming path with real checkpoint weights, per precision and
//! member count, plus the output error of each precision against float32 on the same rows.
//!
//!   cargo run --release --example bench_int8 -- --rows ROWS.f32 --reps 7 \
//!       --members A.wbin B.wbin C.wbin D.wbin [--wide W384.wbin] [--precisions i16,q8x,q8vnni]
//!
//! `.wbin` files come from `scripts/export_native_weights.py`; ROWS.f32 holds little-endian
//! float32 raw rows (112 per row) of whole 20,000-row sequences. For every precision and every
//! member count 1..=len(members) the loop is the per-row work of an ensemble callback without
//! Python: sanitize + stream features once, then each member `step_then`, resetting all state at
//! every sequence boundary. Reports the median over `--reps` alternating-order trials. The
//! controller supplies CPU affinity (taskset) and resource limits.
use std::hint::black_box;
use std::time::Instant;

use wnn_native::rnn::{GruLayer, RnnWeights};
use wnn_native::rnn_fast::{Activations, FastRnn, Precision};
use wnn_native::stream::{N_EVENT, N_RAW, StreamFeatures};

const SEQ: usize = 20_000;
const WARMUP: usize = 99;

fn load(path: &str) -> RnnWeights {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&bytes[..8], b"WNNWTS01", "{path}: not a .wbin file");
    let dim = |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (n_in, hidden, layers, head, n_out) = (dim(0), dim(1), dim(2), dim(3), dim(4));
    let mut at = 28;
    let mut take = |len: usize| -> Vec<f32> {
        let out = bytes[at..at + 4 * len]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        at += 4 * len;
        out
    };
    let mean = take(n_in);
    let std = take(n_in);
    let w_in = take(hidden * n_in);
    let b_in = take(hidden);
    let layers = (0..layers)
        .map(|_| GruLayer {
            w_ih: take(3 * hidden * hidden),
            w_hh: take(3 * hidden * hidden),
            b_ih: take(3 * hidden),
            b_hh: take(3 * hidden),
        })
        .collect();
    let w = RnnWeights {
        mean,
        std,
        w_in,
        b_in,
        layers,
        w1: take(head * 2 * hidden),
        b1: take(head),
        w2: take(n_out * head),
        b2: take(n_out),
    };
    assert_eq!(at, bytes.len(), "{path}: trailing bytes");
    w
}

fn precision(name: &str) -> Precision {
    match name {
        "f32" => Precision::F32,
        "i16" => Precision::I16,
        "q16x" => Precision::Q16x,
        "q8r" => Precision::Q8r,
        "q8x" => Precision::Q8x,
        "q8vnni" => Precision::Q8Vnni,
        "q8vnni_avx2" => Precision::Q8VnniAvx2,
        "q8vnni_portable" => Precision::Q8VnniPortable,
        other => panic!("precision {other}"),
    }
}

fn read_rows(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("rows file");
    let rows: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(rows.len() % (N_RAW * SEQ), 0, "rows must be whole sequences");
    rows
}

struct Ensemble {
    models: Vec<FastRnn>,
    features: StreamFeatures,
}

impl Ensemble {
    fn new(paths: &[String], p: Precision) -> Self {
        let models = paths
            .iter()
            .map(|path| {
                let m = FastRnn::with_precision(load(path), Activations::Fast, p).unwrap();
                assert_eq!(m.precision(), p, "requested precision {p:?} unavailable");
                m
            })
            .collect();
        Self { models, features: StreamFeatures::default() }
    }

    fn reset(&mut self) {
        self.features.reset();
        self.models.iter_mut().for_each(FastRnn::reset);
    }

    /// Run every row; `sink(row, member, out)` sees each member's two outputs.
    fn run(&mut self, raw: &[f32], mut sink: impl FnMut(usize, usize, &[f32; 2])) {
        let mut x = [0.0f32; N_RAW + N_EVENT];
        let mut out = [0.0f32; 2];
        let firsts: Vec<_> = self.models.iter().map(FastRnn::first_weights).collect();
        for (r, row) in raw.chunks_exact(N_RAW).enumerate() {
            if r % SEQ == 0 {
                self.reset();
            }
            let (a, b) = x.split_at_mut(N_RAW);
            a.copy_from_slice(row);
            self.features.sanitize(a);
            self.features.update(a, b);
            for i in 0..self.models.len() {
                let after = firsts[(i + 1) % firsts.len()];
                self.models[i].step_then(black_box(&x), &mut out, after);
                sink(r, i, &out);
            }
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Matvec-only throughput of the dynamic int8 kernels: 768 x 256 matrices cycled so the working
/// set is `count` x 192 KiB (L2-resident for small counts, streamed from L3 for large ones).
fn kernel_bench() {
    use wnn_native::vnni::{DynQ8, Isa, padded_cols, quantize, supported};
    let (rows, cols) = (768usize, 256usize);
    let mut s = 7u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / 16777216.0 - 0.5) * 0.2
    };
    let x: Vec<f32> = (0..cols).map(|_| rnd() * 5.0).collect();
    let mut q = vec![0u8; padded_cols(cols)];
    let xs = quantize(&x, &mut q);
    let bias = vec![0.0f32; rows];
    let mut out = vec![0.0f32; rows];
    for count in [2usize, 4, 8, 16] {
        let mats: Vec<DynQ8> = (0..count)
            .map(|_| {
                let w: Vec<f32> = (0..rows * cols).map(|_| rnd()).collect();
                DynQ8::pack(&w, rows, cols).unwrap()
            })
            .collect();
        for isa in [Isa::Vnni512, Isa::Avx2] {
            if !supported(isa) {
                continue;
            }
            let reps = 2000 / count;
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let t = Instant::now();
                for _ in 0..reps {
                    for m in &mats {
                        m.gemv(isa, black_box(&q), xs, Some(&bias), &mut out);
                    }
                    black_box(&mut out);
                }
                best = best.min(t.elapsed().as_secs_f64() / (reps * count) as f64 * 1e6);
            }
            let bytes = (rows * cols) as f64;
            println!(
                "{isa:?} {count:2} mats ({:.2} MB): {best:.2} us/matvec, {:.1} GB/s",
                count as f64 * bytes / 1e6,
                bytes / best / 1e3
            );
        }
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--kernel") {
        return kernel_bench();
    }
    let mut args = std::env::args().skip(1).peekable();
    let (mut rows_path, mut reps) = (String::new(), 7usize);
    let (mut members, mut wide) = (Vec::new(), None);
    let mut names = "i16,q8x,q8vnni_avx2,q8vnni".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rows" => rows_path = args.next().unwrap(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--wide" => wide = args.next(),
            "--precisions" => names = args.next().unwrap(),
            "--members" => {
                while args.peek().is_some_and(|p| !p.starts_with("--")) {
                    members.push(args.next().unwrap());
                }
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let raw = read_rows(&rows_path);
    let n_rows = raw.len() / N_RAW;
    let precisions: Vec<(&str, Precision)> = names.split(',').map(|n| (n, precision(n))).collect();
    println!(
        "rows {n_rows}, reps {reps}, vnni512 {}, avx2 {}",
        wnn_native::vnni::supported(wnn_native::vnni::Isa::Vnni512),
        wnn_native::vnni::supported(wnn_native::vnni::Isa::Avx2)
    );

    // Cases: (label, member files). Median us/row over alternating-order trials.
    let mut cases: Vec<(String, Vec<String>)> = (1..=members.len())
        .map(|k| (format!("w256 x{k}"), members[..k].to_vec()))
        .collect();
    if let Some(w) = wide {
        cases.push(("w384 x1".into(), vec![w]));
    }
    println!("{:<10}{}", "case", precisions.iter().map(|(n, _)| format!("{n:>16}")).collect::<String>());
    for (label, paths) in &cases {
        let mut samples = vec![Vec::new(); precisions.len()];
        let mut ensembles: Vec<Ensemble> = precisions.iter().map(|(_, p)| Ensemble::new(paths, *p)).collect();
        for rep in 0..=reps {
            for j in 0..precisions.len() {
                // Rotate the order every trial; trial 0 only warms caches and branch predictors.
                let i = (j + rep) % precisions.len();
                let t = Instant::now();
                let mut sum = 0.0f32;
                ensembles[i].run(&raw, |_, _, o| sum += o[0]);
                black_box(sum);
                if rep > 0 {
                    samples[i].push(t.elapsed().as_secs_f64() / n_rows as f64 * 1e6);
                }
            }
        }
        let cells: String = samples.into_iter().map(|s| format!("{:>16.2}", median(s))).collect();
        println!("{label:<10}{cells}");
    }

    // Output error against float32 on the real rows (scored region: after the warm-up rows).
    println!("\nmax |output - f32| over scored rows, member 0 of each case");
    for (label, paths) in cases.iter().filter(|(l, _)| l == "w256 x1" || l == "w384 x1") {
        let mut reference = Vec::with_capacity(n_rows);
        Ensemble::new(&paths[..1], Precision::F32).run(&raw, |_, _, o| reference.push(*o));
        for (name, p) in &precisions {
            let mut worst = [0.0f32; 2];
            let (mut sq, mut count) = (0.0f64, 0usize);
            let mut r = 0;
            Ensemble::new(&paths[..1], *p).run(&raw, |row, _, o| {
                if row % SEQ >= WARMUP {
                    for k in 0..2 {
                        let d = (o[k] - reference[r][k]).abs();
                        worst[k] = worst[k].max(d);
                        sq += f64::from(d) * f64::from(d);
                        count += 1;
                    }
                }
                r += 1;
            });
            println!(
                "{label:<10}{name:<16} max {:.5} / {:.5}  rms {:.6}",
                worst[0],
                worst[1],
                (sq / count as f64).sqrt()
            );
        }
    }
}
