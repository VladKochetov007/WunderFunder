//! cargo run --release --example bench_fast -- <hidden> <layers>
use std::hint::black_box;
use std::time::Instant;
use wnn_native::rnn::{GruLayer, RnnModel, RnnWeights};
use wnn_native::rnn_fast::{Activations, FastRnn};

fn weights(n_in: usize, h: usize, l: usize) -> RnnWeights {
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
        layers: (0..l)
            .map(|_| GruLayer {
                w_ih: v(3 * h * h),
                w_hh: v(3 * h * h),
                b_ih: v(3 * h),
                b_hh: v(3 * h),
            })
            .collect(),
        w1: v(2 * h * h),
        b1: v(h),
        w_mid: Vec::new(),
        b_mid: Vec::new(),
        w2: v(2 * h),
        b2: v(2),
    }
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let (h, l, n_in) = (
        a.first().copied().unwrap_or(256),
        a.get(1).copied().unwrap_or(2),
        211,
    );
    // 112 dense features, 99 event features with ~85% zeros.
    let rows: Vec<Vec<f32>> = (0..256)
        .map(|r| {
            (0..n_in)
                .map(|c| {
                    if c >= 112 && (r * 7 + c * 13) % 20 < 17 {
                        0.0
                    } else {
                        ((r * 31 + c * 17) % 200) as f32 / 50.0 - 2.0
                    }
                })
                .collect()
        })
        .collect();
    let n = 100_000;
    let mut reference = RnnModel::new(weights(n_in, h, l)).unwrap();
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        acc += reference.step(black_box(&rows[i % rows.len()]))[0];
    }
    let base = t.elapsed().as_secs_f64() / n as f64 * 1e6;
    println!("hidden {h} layers {l}: reference {base:.2} us/row (checksum {acc:.3})");
    for act in [Activations::Exact, Activations::Fast] {
        let mut m = FastRnn::new(weights(n_in, h, l), act).unwrap();
        let mut out = [0.0f32; 2];
        let t = Instant::now();
        let mut acc = 0.0;
        for i in 0..n {
            m.step(black_box(&rows[i % rows.len()]), &mut out);
            acc += out[0];
        }
        let us = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        println!(
            "  FastRnn {act:?}: {us:.2} us/row  speedup {:.2}x (checksum {acc:.3})",
            base / us
        );
    }
}
