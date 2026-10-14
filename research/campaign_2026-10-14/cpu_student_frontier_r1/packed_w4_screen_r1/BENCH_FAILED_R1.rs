//! Exact represented-W4 execution screen, not an optimized T-MAC implementation.
//! Compare expanded VNNI, direct nibble decode, and a scalar bit-plane LUT.
use std::hint::black_box;
use std::time::Instant;

use wnn_native::vnni::{supported, DynQ8, Isa};

fn nibble(code: u8) -> i32 {
    ((code as i8) << 4 >> 4) as i32
}

fn decode_dot(weights: &[u8], x: &[u8], out: &mut [f32]) {
    let cols = x.len();
    for (row, result) in out.iter_mut().enumerate() {
        let mut sum = 0i32;
        for col in (0..cols).step_by(2) {
            let codes = weights[row * cols / 2 + col / 2];
            sum += nibble(codes & 15) * (i32::from(x[col]) - 128);
            sum += nibble(codes >> 4) * (i32::from(x[col + 1]) - 128);
        }
        *result = sum as f32;
    }
}

fn build_tables(x: &[u8], tables: &mut [[i16; 16]]) {
    for (group, table) in tables.iter_mut().enumerate() {
        table[0] = 0;
        for pattern in 1usize..16 {
            let bit = pattern.trailing_zeros() as usize;
            table[pattern] = table[pattern & (pattern - 1)] + i16::from(x[group * 4 + bit]) - 128;
        }
    }
}

fn lookup_dot(planes: &[u8], tables: &[[i16; 16]], out: &mut [f32]) {
    let groups = tables.len();
    for (row, result) in out.iter_mut().enumerate() {
        let mut sum = 0i32;
        for (group, table) in tables.iter().enumerate() {
            let a = planes[2 * (row * groups + group)];
            let b = planes[2 * (row * groups + group) + 1];
            sum += i32::from(table[usize::from(a & 15)])
                + 2 * i32::from(table[usize::from(a >> 4)])
                + 4 * i32::from(table[usize::from(b & 15)])
                - 8 * i32::from(table[usize::from(b >> 4)]);
        }
        *result = sum as f32;
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    assert!(
        supported(Isa::Vnni512),
        "expanded control requires VNNI; no fallback"
    );
    println!("rows,cols,expanded_matrix_bytes,packed_nibble_bytes,packed_plane_bytes,lut_scratch_bytes,expanded_us,decode_us,lut_total_us,lut_build_us,exact");
    for cols in [192usize, 256, 384, 448, 512, 640] {
        let rows = 3 * cols;
        // Declared symmetric W4 range matches the current [-7, 7] exporter.
        let codes: Vec<i8> = (0..rows * cols)
            .map(|index| ((index * 73 + index / cols * 19 + 5) % 15) as i8 - 7)
            .collect();
        let x: Vec<u8> = (0..cols)
            .map(|index| ((index * 137 + 13) % 255 + 1) as u8)
            .collect();
        let scales = vec![1.0f32; rows];
        let expanded = DynQ8::from_rows(&codes, &scales, rows, cols).unwrap();
        let packed: Vec<u8> = codes
            .chunks_exact(2)
            .map(|pair| (pair[0] as u8 & 15) | ((pair[1] as u8 & 15) << 4))
            .collect();
        let mut planes = vec![0u8; rows * cols / 2];
        for row in 0..rows {
            for group in 0..cols / 4 {
                for plane in 0..4 {
                    let mut pattern = 0u8;
                    for bit in 0..4 {
                        pattern |=
                            ((codes[row * cols + 4 * group + bit] as u8 >> plane) & 1) << bit;
                    }
                    planes[row * cols / 2 + 2 * group + plane / 2] |= pattern << (4 * (plane % 2));
                }
            }
        }
        let reference: Vec<f32> = codes
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&x)
                    .map(|(w, q)| i32::from(*w) * (i32::from(*q) - 128))
                    .sum::<i32>() as f32
            })
            .collect();
        let mut out = vec![0.0f32; rows];
        let mut tables = vec![[0i16; 16]; cols / 4];
        let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for repeat in 0..9 {
            let order = if repeat % 2 == 0 {
                [0, 1, 2, 3]
            } else {
                [3, 2, 1, 0]
            };
            for method in order {
                let start = Instant::now();
                for _ in 0..40 {
                    match method {
                        0 => expanded.gemv(Isa::Vnni512, black_box(&x), 1.0, None, &mut out),
                        1 => decode_dot(black_box(&packed), black_box(&x), &mut out),
                        2 => {
                            build_tables(black_box(&x), &mut tables);
                            lookup_dot(black_box(&planes), &tables, &mut out);
                        }
                        3 => build_tables(black_box(&x), &mut tables),
                        _ => unreachable!(),
                    }
                    black_box(&out);
                    black_box(&tables);
                }
                samples[method].push(start.elapsed().as_secs_f64() * 1e6 / 40.0);
                if method != 3 {
                    assert_eq!(
                        out, reference,
                        "method {method} changed represented weights at {rows}x{cols}"
                    );
                }
            }
        }
        let medians: Vec<_> = samples.iter_mut().map(|values| median(values)).collect();
        println!(
            "{rows},{cols},{},{},{},{},{:.6},{:.6},{:.6},{:.6},true",
            expanded.storage_bytes(),
            packed.len(),
            planes.len(),
            tables.len() * 32,
            medians[0],
            medians[1],
            medians[2],
            medians[3]
        );
    }
}
