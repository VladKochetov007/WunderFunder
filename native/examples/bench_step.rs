//! cargo run --release --example bench_step -- <hidden> <layers> <n_in>
use std::time::Instant;
use wnn_native::rnn::{GruLayer, RnnModel, RnnWeights};

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let (h, l, n_in) = (
        a.first().copied().unwrap_or(256),
        a.get(1).copied().unwrap_or(2),
        a.get(2).copied().unwrap_or(211),
    );
    let mut s = 1u64;
    let mut v = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 40) as f32 / 16777216.0 - 0.5) * 0.2
            })
            .collect()
    };
    let w = RnnWeights {
        mean: vec![0.0; n_in],
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
        w2: v(2 * h),
        b2: v(2),
        conv: None,
    };
    let mut m = RnnModel::new(w).unwrap();
    let xs = v(n_in * 64);
    let rows = 200_000;
    let t = Instant::now();
    let mut acc = 0.0f32;
    for i in 0..rows {
        acc += m.step(&xs[(i % 64) * n_in..(i % 64 + 1) * n_in])[0];
    }
    println!(
        "hidden {h} layers {l}: {:.2} us/row (checksum {acc})",
        t.elapsed().as_secs_f64() / rows as f64 * 1e6
    );
}
