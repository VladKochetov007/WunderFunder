//! Per-row cost of the native streaming path (features + GRU members), no Python.
//!
//!   cargo run --release --example bench_engine -- [--rows FILE] [--reps N] SPEC [SPEC ...]
//!   cargo run --release --example bench_engine -- --kernel
//!
//! SPEC is `[lstm]hidden[xlayers][wheads]:f32|f16|i16|q16x` (GRU, 2 layers, one head by default),
//! e.g. `128:f16 lstm128:i16 256w2:q16x`. FILE holds
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

fn weights(n_in: usize, h: usize, layers: usize, gates: usize, heads: usize, seed: u64) -> RnnWeights {
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
        w1: r.init(heads * 2 * h * h, 2 * h),
        b1: r.init(heads * h, 2 * h),
        w2: r.init(heads * 2 * h, heads * h),
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

unsafe extern "C" {
    fn syscall(n: i64, ...) -> i64;
}

/// User-space core cycles of this thread (perf_event_open), or None when refused.
fn cycle_counter() -> Option<std::fs::File> {
    use std::os::fd::FromRawFd;
    // perf_event_attr v0: PERF_TYPE_HARDWARE, CPU_CYCLES, exclude_kernel | exclude_hv.
    let mut attr = [0u64; 8];
    attr[0] = 64 << 32;
    attr[5] = 0x60;
    let fd = unsafe { syscall(298, attr.as_ptr(), 0i32, -1i32, -1i32, 0u64) };
    (fd >= 0).then(|| unsafe { std::fs::File::from_raw_fd(fd as i32) })
}

fn read_cycles(f: &mut Option<std::fs::File>) -> u64 {
    use std::io::Read;
    let mut b = [0u8; 8];
    f.as_mut().map_or(0, |f| {
        f.read_exact(&mut b).unwrap();
        u64::from_le_bytes(b)
    })
}

static mut BEST_KCYC: f64 = 0.0;

fn per_row(reps: usize, rows: usize, mut body: impl FnMut()) -> f64 {
    let mut counter = cycle_counter();
    body(); // warm caches and branch predictors
    let mut best = f64::INFINITY;
    let mut best_cyc = u64::MAX;
    for _ in 0..reps {
        let c0 = read_cycles(&mut counter);
        let t = Instant::now();
        body();
        best = best.min(t.elapsed().as_secs_f64());
        best_cyc = best_cyc.min(read_cycles(&mut counter) - c0);
    }
    unsafe { BEST_KCYC = best_cyc as f64 / rows as f64 / 1e3 };
    best / rows as f64 * 1e6
}

fn parse(spec: &str) -> (usize, usize, usize, usize, Precision) {
    let (shape, prec) = spec.split_once(':').unwrap_or((spec, "f32"));
    let (gates, shape) = match shape.strip_prefix("lstm") {
        Some(rest) => (4, rest),
        None => (3, shape),
    };
    let (shape, heads) = shape.split_once('w').unwrap_or((shape, "1"));
    let (h, l) = shape.split_once('x').unwrap_or((shape, "2"));
    let prec = match prec {
        "f16" => Precision::F16,
        "i16" => Precision::I16,
        "f32" => Precision::F32,
        "q16x" => Precision::Q16x,
        "q16x_portable" => Precision::Q16xPortable,
        other => panic!("precision {other}"),
    };
    (h.parse().unwrap(), l.parse().unwrap(), gates, heads.parse().unwrap(), prec)
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

    // Integer kernel: L2-resident (2 matrices, 0.4 MB) vs L3-streamed (8 matrices, 1.6 MB).
    use wnn_native::half::{Encoding, gemv_q16x, quantize, row_scales};
    let mut q = vec![0i16; n];
    let xs = quantize(&x, Some(1.0), &mut q);
    for count in [2usize, 8, 16] {
        let mut arena = HalfArena::new();
        let mut packed = Vec::new();
        for m in w.iter().cycle().take(count) {
            let sc = row_scales(m, 3 * n, n);
            packed.push((arena.push_panel_as(Encoding::I16, m, &sc, 3 * n, n), sc));
        }
        {
            let pf = wnn_native::half::Q_PREFETCH;
            let us = per_row(5, reps, || {
                for _ in 0..reps {
                    for (rg, sc) in &packed {
                        unsafe {
                            gemv_q16x(&arena.as_ref()[rg.clone()], sc, xs, black_box(&q), n, Some(&b), &mut out)
                        };
                    }
                    black_box(&mut out);
                }
            });
            let bytes = (count * 3 * n * n * 2) as f64;
            println!(
                "q16x {count:2} mats ({:.2} MB) PF {pf:4}: {us:.2} us, {:.1} kcyc, {:.1} B/cyc, {:.1} MAC/cyc",
                bytes / 1e6,
                unsafe { BEST_KCYC } * reps as f64,
                bytes / (unsafe { BEST_KCYC } * 1e3),
                bytes / 2.0 / (unsafe { BEST_KCYC } * 1e3)
            );
        }
    }
}

/// L2 residency: 11 matrices 768 x 256 (4.3 MB, two GRU256 members' worth) cycled per "row",
/// the first `k` read with T0 prefetches (meant to stay in L2), the rest with NTA prefetches.
fn l2_bench() {
    use wnn_native::half::{Encoding, HalfArena, gemv_q16x_hint, quantize, row_scales};
    let (n, mats) = (256usize, 11usize);
    let mut r = Rng(9);
    let mut arena = HalfArena::new();
    let mut packed = Vec::new();
    for _ in 0..mats {
        let m = r.init(3 * n * n, n);
        let sc = row_scales(&m, 3 * n, n);
        packed.push((arena.push_panel_as(Encoding::I16, &m, &sc, 3 * n, n), sc));
    }
    let b = vec![0.0f32; 3 * n];
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut q = vec![0i16; n];
    let xs = quantize(&x, Some(1.0), &mut q);
    let mut out = vec![0.0f32; 3 * n];
    let reps = 300;
    let bytes = (mats * 3 * n * n * 2) as f64;
    macro_rules! run {
        ($k:expr, $nta:expr, $pf:expr) => {{
            per_row(5, reps, || {
                for _ in 0..reps {
                    for (i, (rg, sc)) in packed.iter().enumerate() {
                        let p = &arena.as_ref()[rg.clone()];
                        unsafe {
                            if i < $k {
                                gemv_q16x_hint::<false, 4096>(p, sc, xs, black_box(&q), n, Some(&b), &mut out)
                            } else {
                                gemv_q16x_hint::<$nta, $pf>(p, sc, xs, black_box(&q), n, Some(&b), &mut out)
                            }
                        };
                    }
                    black_box(&mut out);
                }
            });
            let kc = unsafe { BEST_KCYC } * reps as f64;
            println!(
                "resident {:2} ({:.2} MB) rest nta={} pf={:5}: {:.1} kcyc/row {:.1} B/cyc",
                $k,
                ($k * 3 * n * n * 2) as f64 / 1e6,
                $nta,
                $pf,
                kc,
                bytes / (kc * 1e3)
            );
        }};
    }
    for k in [0usize, 1, 2, 3] {
        run!(k, false, 4096);
        run!(k, true, 1024);
        run!(k, true, 2048);
        run!(k, true, 4096);
        run!(k, true, 8192);
    }
}

unsafe extern "C" {
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}

/// Page coloring on 2 MB pages: L2 set index bits [15:14] are physical bits we control inside a
/// huge page. Streamed matrices live in the first `stream_kib` of every 64 KiB chunk, resident
/// ones in the rest, so streaming only thrashes its own L2 sets.
fn color_bench() {
    use wnn_native::half::{Encoding, HalfArena, gemv_q16x_hint, quantize, row_scales};
    let (n, mats) = (256usize, 11usize);
    let total = 48usize << 20;
    let layout = std::alloc::Layout::from_size_align(total, 2 << 20).unwrap();
    let base = unsafe { std::alloc::alloc(layout) };
    let rc = unsafe { madvise(base, total, 14) };
    unsafe { std::ptr::write_bytes(base, 0, total) };
    let smaps = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    let huge = smaps.lines().find(|l| l.starts_with("AnonHugePages")).unwrap_or("?").to_string();
    println!("madvise rc {rc}; {huge}");
    let mut r = Rng(9);
    let mut arena = HalfArena::new();
    let mut packed = Vec::new();
    for _ in 0..mats {
        let m = r.init(3 * n * n, n);
        let sc = row_scales(&m, 3 * n, n);
        packed.push((arena.push_panel_as(Encoding::I16, &m, &sc, 3 * n, n), sc));
    }
    let b = vec![0.0f32; 3 * n];
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut q = vec![0i16; n];
    let xs = quantize(&x, Some(1.0), &mut q);
    let mut out = vec![0.0f32; 3 * n];
    let reps = 300;
    let panel_bytes = 4 * n * 2;
    for stream_kib in [64usize, 32, 16] {
        for k in [0usize, 1, 2, 3] {
            // Place: resident matrices fill [stream_kib, 64) KiB of chunks from the start; streamed
            // matrices fill [0, stream_kib) of chunks (separately numbered). 64 = contiguous.
            let piece_s = stream_kib << 10;
            let piece_r = (64 - stream_kib) << 10;
            let (mut cs, mut cr) = (0usize, 0usize); // bytes placed in each color
            let mut pieces: Vec<Vec<(*const u16, usize)>> = Vec::new(); // per matrix: (ptr, rows)
            for (i, (rg, _)) in packed.iter().enumerate() {
                let src = &arena.as_ref()[rg.clone()];
                let bytes = src.len() * 2;
                let resident = i < k && stream_kib < 64;
                let mut list = Vec::new();
                let mut done = 0;
                while done < bytes {
                    let (addr, room) = if resident {
                        let chunk = cr / piece_r;
                        let off = cr % piece_r;
                        ((chunk << 16) + (stream_kib << 10) + off, piece_r - off)
                    } else {
                        let chunk = cs / piece_s;
                        let off = cs % piece_s;
                        // Streamed data starts after 24 MiB so the colors never overlap.
                        ((24 << 20) + (chunk << 16) + off, piece_s - off)
                    };
                    let take = room.min(bytes - done) / panel_bytes * panel_bytes;
                    assert!(take > 0);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (src.as_ptr() as *const u8).add(done),
                            base.add(addr),
                            take,
                        )
                    };
                    list.push((unsafe { base.add(addr) } as *const u16, take / panel_bytes * 4));
                    done += take;
                    if resident { cr += take } else { cs += take }
                }
                pieces.push(list);
            }
            per_row(5, reps, || {
                for _ in 0..reps {
                    for (i, list) in pieces.iter().enumerate() {
                        let sc = &packed[i].1;
                        let mut r0 = 0;
                        for &(p, rows) in list {
                            let panel = unsafe { std::slice::from_raw_parts(p, rows * n) };
                            unsafe {
                                gemv_q16x_hint::<false, 0>(
                                    panel, &sc[r0..], xs, black_box(&q), n, Some(&b[r0..]), &mut out[r0..r0 + rows],
                                )
                            };
                            r0 += rows;
                        }
                    }
                    black_box(&mut out);
                }
            });
            println!(
                "stream {stream_kib:2} KiB/64 resident mats {k}: {:.1} kcyc/row ({} pieces)",
                unsafe { BEST_KCYC },
                pieces.iter().map(|l| l.len()).sum::<usize>()
            );
            if stream_kib == 64 {
                break;
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    let mut raw = None;
    let mut reps = 5;
    let mut specs = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--kernel" => return kernel_bench(),
            "--l2" => return l2_bench(),
            "--color" => return color_bench(),
            #[cfg(feature = "phases")]
            "--parts" => return unsafe { wnn_native::rnn_fast::bench_parts() },
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
            let (h, l, gates, heads, p) = parse(s);
            let m = FastRnn::with_precision(
                weights(n_in, h, l, gates, heads, 11 + i as u64),
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
        #[cfg(feature = "phases")]
        unsafe {
            let ph = wnn_native::rnn_fast::PHASES;
            let tot: u64 = ph.iter().sum();
            let names = ["project", "gelu+quant z", "w_ih", "w_hh", "update+quant", "head.0", "gelu+head.2"];
            for (k, nm) in names.iter().enumerate() {
                println!("   {nm:14} {:5.1}%  {:.1} kcyc/row", ph[k] as f64 * 100.0 / tot as f64, ph[k] as f64 / tot as f64 * BEST_KCYC);
            }
            wnn_native::rnn_fast::PHASES = [0; 8];
        }
        println!(
            "{s:>10} alone: {us:.2} us/row {:.1} kcyc/row ({} KiB weights)",
            unsafe { BEST_KCYC },
            m.weight_bytes() / 1024
        );
    }
    if models.len() > 1 {
        let us = per_row(reps, rows, || {
            for m in models.iter_mut() {
                m.reset();
            }
            let firsts: Vec<_> = models.iter().map(|m| m.first_weights()).collect();
            for x in xs.chunks_exact(n_in) {
                for (i, m) in models.iter_mut().enumerate() {
                    m.step_then(black_box(x), &mut out, firsts[(i + 1) % firsts.len()]);
                }
            }
            black_box(&out);
        });
        let kib: usize = models.iter().map(|m| m.weight_bytes()).sum::<usize>() / 1024;
        println!(
            "ensemble {}: {us:.2} us/row {:.1} kcyc/row ({kib} KiB weights)",
            specs.join("+"),
            unsafe { BEST_KCYC }
        );
    }
}
