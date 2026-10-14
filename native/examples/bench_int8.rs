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
use std::path::Path;
use std::time::Instant;

use wnn_native::rnn::{GruLayer, RnnWeights};
use wnn_native::rnn_fast::{Activations, FastRnn, Precision};
use wnn_native::stream::{N_EVENT, N_RAW, StreamFeatures};

const SEQ: usize = 20_000;
const WARMUP: usize = 99;

fn load(path: &str) -> RnnWeights {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let dim = |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (n_in, encoder, recurrent, layer_count, head, head_layers, n_out, mut at) =
        if &bytes[..8] == b"WNNWTS01" {
            let (n_in, hidden, layer_count, head, n_out) =
                (dim(0), dim(1), dim(2), dim(3), dim(4));
            (n_in, hidden, hidden, layer_count, head, 1, n_out, 28)
        } else if &bytes[..8] == b"WNNWTS02" {
            (
                dim(0), dim(1), dim(2), dim(3), dim(4), dim(5), dim(6), 36,
            )
        } else {
            panic!("{path}: unsupported .wbin version");
        };
    assert!(layer_count > 0 && matches!(head_layers, 1 | 2));
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
    let w_in = take(encoder * n_in);
    let b_in = take(encoder);
    let mut layers = Vec::with_capacity(layer_count);
    for index in 0..layer_count {
        let input_width = if index == 0 { encoder } else { recurrent };
        layers.push(GruLayer {
            w_ih: take(3 * recurrent * input_width),
            w_hh: take(3 * recurrent * recurrent),
            b_ih: take(3 * recurrent),
            b_hh: take(3 * recurrent),
        });
    }
    let w1 = take(head * (recurrent + encoder));
    let b1 = take(head);
    let (w_mid, b_mid) = if head_layers == 2 {
        (take(head * head), take(head))
    } else {
        (Vec::new(), Vec::new())
    };
    let w2 = take(n_out * head);
    let b2 = take(n_out);
    let w = RnnWeights {
        mean,
        std,
        w_in,
        b_in,
        layers,
        w1,
        b1,
        w_mid,
        b_mid,
        w2,
        b2,
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
        "q8vnni256" => Precision::Q8Vnni256,
        "q8vnni_blocked" => Precision::Q8VnniBlocked,
        "q8vnni_fused" => Precision::Q8VnniFused,
        other => panic!("precision {other}"),
    }
}

fn matrix_isa(name: &str) -> wnn_native::vnni::Isa {
    use wnn_native::vnni::Isa;
    match name {
        "q8vnni" => Isa::Vnni512,
        "q8vnni256" => Isa::Vnni256,
        "q8vnni_blocked" => Isa::Vnni512Blocked,
        other => panic!("{other} has no independent matrix kernel"),
    }
}

/// Compare A/B/C matrix kernels on the exact xteam_h4 Q8 matrices and realistic widths.
fn matrix_bench(paths: &[String], names: &str, reps: usize) {
    use wnn_native::vnni::{Isa, padded_cols, quantize, supported};

    let policies: Vec<_> = names
        .split(',')
        .map(|name| (name, matrix_isa(name)))
        .collect();
    assert!(!paths.is_empty(), "--matrix requires --members");
    assert!(reps > 0, "--reps must be positive");
    for (name, isa) in &policies {
        assert!(supported(*isa), "selected policy {name} is unsupported; no fallback");
    }
    for path in paths {
        let model = FastRnn::with_precision(load(path), Activations::Fast, Precision::Q8Vnni)
            .unwrap();
        assert_eq!(model.precision(), Precision::Q8Vnni, "A matrix source precision");
        for (matrix_name, matrix) in model.dyn8_matrices() {
            let x: Vec<f32> = (0..matrix.cols())
                .map(|index| ((index * 73 + 19) % 257) as f32 / 128.0 - 1.0)
                .collect();
            let mut q = vec![128u8; padded_cols(matrix.cols())];
            let xs = quantize(&x, &mut q);
            let bias = vec![0.0f32; matrix.rows()];
            let mut expected = vec![0.0f32; matrix.rows()];
            matrix.gemv(Isa::Portable, &q, xs, Some(&bias), &mut expected);
            let mut samples = vec![Vec::new(); policies.len()];
            for cycle in 0..reps {
                let order = [0usize, 1, 2, 2, 1, 0];
                for position in 0..order.len() {
                    let policy_index = if cycle % 2 == 0 {
                        order[position]
                    } else {
                        order[order.len() - 1 - position]
                    };
                    if policy_index >= policies.len() {
                        continue;
                    }
                    let (policy_name, isa) = policies[policy_index];
                    let mut out = vec![0.0f32; matrix.rows()];
                    let start = Instant::now();
                    for _ in 0..40 {
                        matrix.gemv(isa, black_box(&q), xs, Some(&bias), &mut out);
                        black_box(&mut out);
                    }
                    samples[policy_index].push(start.elapsed().as_secs_f64() / 40.0 * 1e6);
                    assert_eq!(
                        out.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        "{policy_name} changed matrix outputs for {matrix_name}"
                    );
                }
            }
            for ((policy_name, isa), times) in policies.iter().zip(samples) {
                let mut times = times;
                times.sort_by(f64::total_cmp);
                let median_us = times[times.len() / 2];
                println!(
                    "{} {} {}x{} {:?} median={median_us:.3}us exact=true",
                    Path::new(path).file_name().unwrap().to_string_lossy(),
                    matrix_name,
                    matrix.rows(),
                    matrix.cols(),
                    isa,
                );
                black_box(policy_name);
            }
        }
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
    /// Step the members in reverse order on odd rows (env WNN_ALTERNATE=1).
    alternate: bool,
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
        Self {
            models,
            features: StreamFeatures::default(),
            alternate: std::env::var_os("WNN_ALTERNATE").is_some_and(|v| v == "1"),
        }
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
            let count = self.models.len();
            for k in 0..count {
                // Odd rows step the members in reverse, so the member stepped last on one row
                // (whose weights are still in L2) is the first on the next. Members are
                // independent and callers combine outputs by index, so results do not change.
                let i = if self.alternate && r % 2 == 1 { count - 1 - k } else { k };
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
            let mut cyc = counter(0, 0);
            let mut best_cyc = f64::INFINITY;
            for _ in 0..7 {
                let c0 = read_counter(&mut cyc).unwrap_or(0);
                let t = Instant::now();
                for _ in 0..reps {
                    for m in &mats {
                        m.gemv(isa, black_box(&q), xs, Some(&bias), &mut out);
                    }
                    black_box(&mut out);
                }
                best = best.min(t.elapsed().as_secs_f64() / (reps * count) as f64 * 1e6);
                let c1 = read_counter(&mut cyc).unwrap_or(0);
                best_cyc = best_cyc.min((c1 - c0) as f64 / (reps * count) as f64);
            }
            let bytes = (rows * cols) as f64;
            println!(
                "{isa:?} {count:2} mats ({:.2} MB): {best:.2} us/matvec, {:.1} GB/s, {:.0} cycles ({:.2}/line, {:.2} GHz)",
                count as f64 * bytes / 1e6,
                bytes / best / 1e3,
                best_cyc,
                best_cyc / (bytes / 64.0),
                best_cyc / best / 1e3
            );
        }
    }
}

unsafe extern "C" {
    fn syscall(n: i64, ...) -> i64;
}

/// One user-space hardware counter of this thread (perf_event_open); None when refused.
fn counter(kind: u32, config: u64) -> Option<std::fs::File> {
    use std::os::fd::FromRawFd;
    let mut attr = [0u64; 16];
    attr[0] = u64::from(kind) | (112u64 << 32); // type, size
    attr[1] = config;
    attr[5] = 0x60; // exclude_kernel | exclude_hv
    // SAFETY: perf_event_open reads this valid attribute struct.
    let fd = unsafe { syscall(298, attr.as_ptr(), 0i32, -1i32, -1i32, 0u64) };
    // SAFETY: a successful syscall returns a new owned descriptor.
    (fd >= 0).then(|| unsafe { std::fs::File::from_raw_fd(fd as i32) })
}

fn read_counter(f: &mut Option<std::fs::File>) -> Option<u64> {
    use std::io::Read;
    let mut b = [0u8; 8];
    f.as_mut()?.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// Hardware counters and per-stage cycle shares of a K-member ensemble over the rows.
fn profile(raw: &[f32], paths: &[String], p: Precision) {
    const EVENTS: [(&str, u32, u64); 7] = [
        ("cycles", 0, 0),
        ("instructions", 0, 1),
        ("llc_refs", 0, 2),
        ("llc_misses", 0, 3),
        ("l2_demand_data_rd_miss", 4, 0x2124),
        ("l2_all_miss", 4, 0x3f24),
        ("l1d_replacement", 4, 0x0151),
    ];
    let mut counters: Vec<_> = EVENTS.iter().map(|(_, k, c)| counter(*k, *c)).collect();
    let mut ens = Ensemble::new(paths, p);
    let n_rows = (raw.len() / N_RAW) as f64;
    ens.run(raw, |_, _, _| {}); // warm
    #[cfg(feature = "profile")]
    wnn_native::vnni::prof::CYCLES
        .iter()
        .for_each(|c| c.store(0, std::sync::atomic::Ordering::Relaxed));
    let before: Vec<_> = counters.iter_mut().map(read_counter).collect();
    let t = Instant::now();
    ens.run(raw, |_, _, _| {});
    let secs = t.elapsed().as_secs_f64();
    let after: Vec<_> = counters.iter_mut().map(read_counter).collect();
    println!("{} member(s), {:.2} us/row", paths.len(), secs / n_rows * 1e6);
    for ((name, _, _), (b, a)) in EVENTS.iter().zip(before.iter().zip(&after)) {
        match (b, a) {
            (Some(b), Some(a)) => println!("  {name:<24}{:>10.1} per row", (a - b) as f64 / n_rows),
            _ => println!("  {name:<24}unavailable"),
        }
    }
    if let (Some(Some(c0)), Some(Some(c1)), Some(Some(i0)), Some(Some(i1))) =
        (before.first(), after.first(), before.get(1), after.get(1))
    {
        println!("  IPC {:.2}", (i1 - i0) as f64 / (c1 - c0) as f64);
    }
    // Features alone.
    let mut f = StreamFeatures::default();
    let mut x = [0.0f32; N_RAW + N_EVENT];
    let t = Instant::now();
    for (r, row) in raw.chunks_exact(N_RAW).enumerate() {
        if r % SEQ == 0 {
            f.reset();
        }
        let (a, b) = x.split_at_mut(N_RAW);
        a.copy_from_slice(row);
        f.sanitize(a);
        f.update(a, b);
        black_box(&mut x);
    }
    println!("  features only           {:>10.2} us/row", t.elapsed().as_secs_f64() / n_rows * 1e6);
    #[cfg(feature = "profile")]
    {
        let cyc: Vec<u64> = wnn_native::vnni::prof::CYCLES
            .iter()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .collect();
        let total: u64 = cyc.iter().sum();
        println!("  stage (TSC share of the model step, both members summed)");
        for (i, c) in cyc.iter().enumerate().filter(|(_, c)| **c > 0) {
            println!("    slot {i:2}  {:>6.1}%  {:>9.0} tsc/row", *c as f64 / total as f64 * 100.0, *c as f64 / n_rows);
        }
    }
}

/// Interleaved comparison of fixed and alternating member order for K members (env WNN_SWEEP=K).
fn sweep(raw: &[f32], paths: &[String], p: Precision, reps: usize) {
    let cfgs: Vec<bool> = vec![false, true];
    let n_rows = (raw.len() / N_RAW) as f64;
    let mut ens = Ensemble::new(paths, p);
    let mut samples = vec![Vec::new(); cfgs.len()];
    for rep in 0..=reps {
        for j in 0..cfgs.len() {
            let i = (j + rep) % cfgs.len();
            ens.alternate = cfgs[i];
            let t = Instant::now();
            ens.run(raw, |_, _, _| {});
            if rep > 0 {
                samples[i].push(t.elapsed().as_secs_f64() / n_rows * 1e6);
            }
        }
    }
    for (c, s) in cfgs.iter().zip(samples) {
        let min = s.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("alternate={} median {:.2}  min {:.2}", *c as u8, median(s), min);
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--kernel") {
        return kernel_bench();
    }
    let mut args = std::env::args().skip(1).peekable();
    let mut profile_members = 0usize;
    let mut matrix_mode = false;
    let (mut rows_path, mut reps) = (String::new(), 7usize);
    let (mut members, mut wide) = (Vec::new(), None);
    let mut names = "i16,q8x,q8vnni_avx2,q8vnni".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rows" => rows_path = args.next().unwrap(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--wide" => wide = args.next(),
            "--profile" => profile_members = args.next().unwrap().parse().unwrap(),
            "--matrix" => matrix_mode = true,
            "--precisions" => names = args.next().unwrap(),
            "--members" => {
                while args.peek().is_some_and(|p| !p.starts_with("--")) {
                    members.push(args.next().unwrap());
                }
            }
            other => panic!("unknown argument {other}"),
        }
    }
    if matrix_mode {
        return matrix_bench(&members, &names, reps);
    }
    let raw = read_rows(&rows_path);
    let n_rows = raw.len() / N_RAW;
    let precisions: Vec<(&str, Precision)> = names.split(',').map(|n| (n, precision(n))).collect();
    if let Some(k) = std::env::var("WNN_SWEEP").ok().and_then(|v| v.parse::<usize>().ok()) {
        return sweep(&raw, &members[..k], precisions[0].1, reps);
    }
    if profile_members > 0 {
        return profile(&raw, &members[..profile_members], precisions[0].1);
    }
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
