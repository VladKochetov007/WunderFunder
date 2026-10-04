//! Isolated weight-bandwidth probe, not a streaming-model or public-runtime estimate.
//! Usage: bench_quant8 ROWS COLS MATRICES ITERATIONS REPEATS
//! Example shapes: 768 256 4 256 5; 1152 384 4 256 5; 384 768 1 256 5.
//! Alternates execution order and reports every paired trial, including unavailable
//! hardware counters as null. The controller supplies CPU affinity and resource caps.

use std::hint::black_box;
use std::io::Read;
use std::ops::Range;
use std::time::Instant;

use wnn_native::half::{self, Encoding, HalfArena};
use wnn_native::quant8::PackedQ8;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn cycle_counter() -> Option<std::fs::File> {
    use std::os::fd::FromRawFd;
    unsafe extern "C" {
        fn syscall(n: i64, ...) -> i64;
    }
    let mut attr = [0u64; 8];
    attr[0] = 64 << 32;
    attr[5] = 0x60;
    // SAFETY: perf_event_open on Linux x86_64 reads this valid v0 attribute struct.
    let fd = unsafe { syscall(298, attr.as_ptr(), 0i32, -1i32, -1i32, 0u64) };
    // SAFETY: a successful syscall returns a new owned file descriptor.
    (fd >= 0).then(|| unsafe { std::fs::File::from_raw_fd(fd as i32) })
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn cycle_counter() -> Option<std::fs::File> {
    None
}

fn read_cycles(counter: &mut Option<std::fs::File>) -> Option<u64> {
    let mut bytes = [0; 8];
    counter.as_mut()?.read_exact(&mut bytes).ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn measure(iterations: usize, matrices: usize, mut body: impl FnMut()) -> (f64, Option<f64>) {
    body();
    let mut counter = cycle_counter();
    let before = read_cycles(&mut counter);
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let after = read_cycles(&mut counter);
    let calls = (iterations * matrices) as f64;
    let cycles = before
        .zip(after)
        .map(|(a, b)| (b - a) as f64 / calls / 1000.0);
    (elapsed * 1e6 / calls, cycles)
}

fn q16_run(
    arena: &HalfArena,
    ranges: &[Range<usize>],
    scales: &[Vec<f32>],
    cols: usize,
    q: &[i16],
    bias: &[f32],
    out: &mut [f32],
) {
    for (range, scale) in ranges.iter().zip(scales) {
        #[cfg(target_arch = "x86_64")]
        if half::available() {
            // SAFETY: checked runtime features, packed panel and padded input lengths.
            unsafe {
                half::gemv_q16x(
                    &arena.as_ref()[range.clone()],
                    scale,
                    1.0 / 16383.0,
                    q,
                    cols,
                    Some(bias),
                    out,
                )
            };
            black_box(&*out);
            continue;
        }
        half::gemv_q16x_portable(
            &arena.as_ref()[range.clone()],
            scale,
            1.0 / 16383.0,
            q,
            cols,
            Some(bias),
            out,
        );
        black_box(&*out);
    }
}

fn main() -> Result<(), String> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let [rows, cols, count, iterations, repeats] = args.as_slice() else {
        return Err("expected ROWS COLS MATRICES ITERATIONS REPEATS".into());
    };
    let (rows, cols, count, iterations, repeats) = (*rows, *cols, *count, *iterations, *repeats);
    if rows == 0
        || rows > 8192
        || cols == 0
        || cols > 8192
        || count == 0
        || count > 64
        || iterations == 0
        || iterations > 10_000
        || repeats == 0
        || repeats > 20
        || rows * cols * count > 32 * 1024 * 1024
    {
        return Err("shape/work exceeds bounded probe limits".into());
    }
    let mut random = 271828u64;
    let mut arena = HalfArena::new();
    let mut ranges = Vec::new();
    let mut scales = Vec::new();
    let mut byte_matrices = Vec::new();
    for _ in 0..count {
        let weights: Vec<f32> = (0..rows * cols)
            .map(|_| {
                random = random
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((random >> 40) as f32 / 16777216.0 - 0.5) / (cols as f32).sqrt()
            })
            .collect();
        let mut scale = half::row_scales(&weights, rows, cols);
        scale.resize(rows.div_ceil(4) * 4, 0.0);
        ranges.push(arena.push_panel_as(Encoding::I16, &weights, &scale, rows, cols));
        scales.push(scale);
        byte_matrices.push(PackedQ8::pack(&weights, rows, cols)?);
    }
    let q: Vec<i16> = (0..cols.div_ceil(16) * 16)
        .map(|i| {
            if i < cols {
                ((i * 97) % 32767) as i16 - 16383
            } else {
                0
            }
        })
        .collect();
    let bias = vec![0.0; rows];
    let mut out16 = vec![0.0; rows];
    let mut out8 = vec![0.0; rows];
    let bytes16 = arena.as_ref().len() * 2 + scales.iter().map(|x| x.len() * 4).sum::<usize>();
    let bytes8 = byte_matrices
        .iter()
        .map(PackedQ8::storage_bytes)
        .sum::<usize>();
    for trial in 0..repeats {
        let mut run16 = || {
            measure(iterations, count, || {
                q16_run(&arena, &ranges, &scales, cols, &q, &bias, &mut out16)
            })
        };
        let mut run8 = || {
            measure(iterations, count, || {
                for matrix in &byte_matrices {
                    matrix.gemv(&q, 1.0 / 16383.0, Some(&bias), &mut out8);
                    black_box(&out8);
                }
            })
        };
        let (a, b) = if trial % 2 == 0 {
            (run16(), run8())
        } else {
            let b = run8();
            (run16(), b)
        };
        let cycle_json = |v: Option<f64>| v.map_or("null".to_owned(), |v| v.to_string());
        println!(
            "{{\"probe\":\"matrix_only\",\"rows\":{rows},\"cols\":{cols},\"matrices\":{count},\"iterations\":{iterations},\"trial\":{trial},\"q16_bytes\":{bytes16},\"q8_bytes\":{bytes8},\"q16_us_per_matvec\":{},\"q8_us_per_matvec\":{},\"q16_kcycles_per_matvec\":{},\"q8_kcycles_per_matvec\":{}}}",
            a.0,
            b.0,
            cycle_json(a.1),
            cycle_json(b.1)
        );
    }
    Ok(())
}
