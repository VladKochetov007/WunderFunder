use std::hint::black_box;
use std::time::Instant;
use wnn_native::kernels::gemv;

#[inline(never)]
fn gemv_colmajor(wc: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    out.copy_from_slice(bias);
    for (col, &xj) in wc.chunks_exact(rows).zip(x) {
        for (o, w) in out.iter_mut().zip(col) {
            *o += w * xj;
        }
    }
}

#[inline(never)]
fn gemv_colmajor_skip0(wc: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    out.copy_from_slice(bias);
    for (col, &xj) in wc.chunks_exact(rows).zip(x) {
        if xj == 0.0 {
            continue;
        }
        for (o, w) in out.iter_mut().zip(col) {
            *o += w * xj;
        }
    }
}

fn main() {
    for (rows, cols) in [(768usize, 256usize), (256, 211), (256, 512)] {
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) * 1e-4)
            .collect();
        let mut wc = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                wc[c * rows + r] = w[r * cols + c];
            }
        }
        let b = vec![0.1f32; rows];
        let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin()).collect();
        let (mut o1, mut o2) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        let n = 20_000;
        let t = Instant::now();
        for _ in 0..n {
            gemv(black_box(&w), &b, black_box(&x), &mut o1);
        }
        let a = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        let t = Instant::now();
        for _ in 0..n {
            gemv_colmajor(black_box(&wc), &b, black_box(&x), &mut o2);
        }
        let s = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        let err = o1
            .iter()
            .zip(&o2)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        let f = (rows * cols) as f64;
        let t = Instant::now();
        let xs: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 3 == 0 { *v } else { 0.0 })
            .collect();
        for _ in 0..n {
            gemv_colmajor_skip0(black_box(&wc), &b, black_box(&xs), &mut o2);
        }
        let z = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        println!(
            "{rows}x{cols}: rowdot {a:.2}us ({:.1} GFLOPS) colsaxpy {s:.2}us ({:.1} GFLOPS) maxdiff {err:.1e} | sparse-input(1/3 nonzero) {z:.2}us",
            f / a / 1e3,
            f / s / 1e3
        );
    }
}
