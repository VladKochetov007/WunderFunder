use std::hint::black_box;
use std::time::Instant;
use wnn_native::kernels::{gelu, gemv, sigmoid};

fn main() {
    let h = 256;
    let w: Vec<f32> = (0..3 * h * h)
        .map(|i| ((i * 7919 % 1000) as f32 - 500.0) * 1e-4)
        .collect();
    let b = vec![0.1f32; 3 * h];
    let x: Vec<f32> = (0..h).map(|i| (i as f32).sin()).collect();
    let mut out = vec![0.0f32; 3 * h];
    let n = 20_000;
    let t = Instant::now();
    for _ in 0..n {
        gemv(black_box(&w), &b, black_box(&x), &mut out);
    }
    let gemv_us = t.elapsed().as_secs_f64() / n as f64 * 1e6;
    let flops = (3 * h * h) as f64;
    println!(
        "gemv 768x256: {gemv_us:.2} us  -> {:.2} GFLOPS",
        flops / gemv_us / 1e3
    );
    let t = Instant::now();
    let mut s = 0.0f32;
    for _ in 0..n {
        for v in out.iter() {
            s += sigmoid(black_box(*v)) + (*v).tanh() + gelu(*v);
        }
    }
    println!(
        "768x(sigmoid+tanh+gelu) scalar: {:.2} us, checksum {s}",
        t.elapsed().as_secs_f64() / n as f64 * 1e6
    );
}
