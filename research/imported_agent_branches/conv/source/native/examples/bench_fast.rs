//! cargo run --release --example bench_fast -- <hidden> <layers> [conv_k]
//! conv_k > 0 adds the enc="conv" encoder (causal depthwise conv of width conv_k + pointwise).
use std::hint::black_box;
use std::time::Instant;
use wnn_native::rnn::{ConvEncoder, GruLayer, RnnModel, RnnWeights};
use wnn_native::rnn_fast::{Activations, FastRnn};

fn weights(n_in: usize, h: usize, l: usize, k: usize) -> RnnWeights {
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
        w2: v(2 * h),
        b2: v(2),
        conv: (k > 0).then(|| ConvEncoder {
            k,
            w_dw: v(h * k),
            b_dw: v(h),
            w_pw: v(h * h),
            b_pw: v(h),
        }),
    }
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let (h, l, k, n_in) = (
        a.first().copied().unwrap_or(256),
        a.get(1).copied().unwrap_or(2),
        a.get(2).copied().unwrap_or(0),
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
    let mut reference = RnnModel::new(weights(n_in, h, l, k)).unwrap();
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        acc += reference.step(black_box(&rows[i % rows.len()]))[0];
    }
    let base = t.elapsed().as_secs_f64() / n as f64 * 1e6;
    println!("hidden {h} layers {l} conv_k {k}: reference {base:.2} us/row (checksum {acc:.3})");
    for act in [Activations::Exact, Activations::Fast] {
        let mut m = FastRnn::new(weights(n_in, h, l, k), act).unwrap();
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
    if k > 0 {
        // Paired blocks: identical GRU/head weights with and without the conv encoder step the
        // same rows back to back in alternating order, so clock and co-tenant drift cancel.
        let mut lin = FastRnn::new(weights(n_in, h, l, 0), Activations::Fast).unwrap();
        let mut conv = FastRnn::new(weights(n_in, h, l, k), Activations::Fast).unwrap();
        let (block, blocks) = (250, 800);
        let mut out = [0.0f32; 2];
        let mut ratio = Vec::with_capacity(blocks);
        let (mut t_lin, mut t_conv) = (0.0, 0.0);
        for b in 0..blocks {
            let mut t = [0.0f64; 2];
            for j in 0..2 {
                let which = (j + b) % 2;
                let m = if which == 0 { &mut lin } else { &mut conv };
                let s = Instant::now();
                for i in 0..block {
                    m.step(black_box(&rows[(b * block + i) % rows.len()]), &mut out);
                }
                t[which] = s.elapsed().as_secs_f64();
            }
            t_lin += t[0];
            t_conv += t[1];
            ratio.push(t[1] / t[0]);
        }
        ratio.sort_by(f64::total_cmp);
        let q = |p: f64| ratio[((blocks - 1) as f64 * p) as usize] - 1.0;
        let per = |t: f64| t / (blocks * block) as f64 * 1e6;
        println!(
            "  paired Fast: linear {:.2} conv {:.2} us/row; conv/linear median {:+.1}% (IQR {:+.1}% .. {:+.1}%)",
            per(t_lin),
            per(t_conv),
            100.0 * q(0.5),
            100.0 * q(0.25),
            100.0 * q(0.75)
        );
    }
}
