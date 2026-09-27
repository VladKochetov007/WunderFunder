//! Streaming event features, the Rust twin of `wnn.features.StreamingEventFeatures`.
//!
//! Layout (99 values): d[i0 prices 0..22], d[i0 volumes 22..44], d[i1 prices 52..74],
//! 8 event flags, 3 log-counters since (i0 price, i0 trade, i1 price) events,
//! 22 held i0 price deltas from the last i0 price event.
//! `OnlineNorm` adds the optional 112 causal per-sequence z-scores of the raw row.

pub const N_RAW: usize = 112;
pub const N_EVENT: usize = 99;
const CAP: f64 = 10_000.0;

#[derive(Debug, Clone)]
pub struct StreamFeatures {
    prev: [f32; N_RAW],
    first: bool,
    step: i64,
    last: [i64; 3],
    held: [f32; 22],
}

impl Default for StreamFeatures {
    fn default() -> Self {
        Self {
            prev: [0.0; N_RAW],
            first: true,
            step: 0,
            last: [0; 3],
            held: [0.0; 22],
        }
    }
}

#[inline]
fn any_changed(prev: &[f32], cur: &[f32]) -> bool {
    prev.iter().zip(cur).any(|(a, b)| a != b)
}

impl StreamFeatures {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Replaces non-finite values with the column's previous value (0 before the first row),
    /// so a corrupt input reads as "unchanged" and cannot poison recurrent state.
    pub fn sanitize(&self, row: &mut [f32]) {
        for (j, v) in row.iter_mut().enumerate() {
            if !v.is_finite() {
                *v = if self.first { 0.0 } else { self.prev[j] };
            }
        }
    }

    /// Writes the 99 event features for `row` into `out` and advances the state.
    pub fn update(&mut self, row: &[f32], out: &mut [f32]) {
        assert_eq!(row.len(), N_RAW, "row width");
        assert_eq!(out.len(), N_EVENT, "feature width");
        if self.first {
            self.prev.copy_from_slice(row);
            self.first = false;
        }
        let p = &self.prev;
        for j in 0..44 {
            out[j] = row[j] - p[j];
        }
        for j in 0..22 {
            out[44 + j] = row[52 + j] - p[52 + j];
        }
        let px = row[0] != p[0] && row[11] != p[11];
        let trade_vol = any_changed(&p[48..52], &row[48..52]);
        let i1_px = row[52] != p[52] || row[63] != p[63];
        let flags = [
            px,
            any_changed(&p[22..33], &row[22..33]),
            any_changed(&p[33..44], &row[33..44]),
            any_changed(&p[44..48], &row[44..48]),
            trade_vol,
            i1_px,
            any_changed(&p[74..96], &row[74..96]),
            any_changed(&p[100..104], &row[100..104]),
        ];
        for (o, f) in out[66..74].iter_mut().zip(flags) {
            *o = if f { 1.0 } else { 0.0 };
        }
        for (k, event) in [px, trade_vol, i1_px].into_iter().enumerate() {
            if event {
                self.last[k] = self.step;
            }
            let since = ((self.step - self.last[k]) as f64).min(CAP);
            out[74 + k] = since.ln_1p() as f32;
        }
        if px {
            for j in 0..22 {
                self.held[j] = row[j] - p[j];
            }
        }
        out[77..99].copy_from_slice(&self.held);
        self.prev.copy_from_slice(row);
        self.step += 1;
    }
}

/// Variance floor of `wnn.features.online_zscore_torch`.
pub const ONLINE_EPS: f32 = 1e-4;

/// Causal per-sequence z-scores of the raw features, the Rust twin of
/// `wnn.features.online_zscore_torch` as training evaluates it on CUDA. With n rows seen (the
/// current one included): mean = sum/n, var = max(sumsq/n - mean^2, 0),
/// z = (x - mean) / sqrt(var + 1e-4).
///
/// The sums are float32 and accumulated in row order on purpose: CUDA `cumsum` along the time
/// axis of a (B, T, F) tensor is a sequential float32 loop per column, so the features the model
/// was trained on carry that rounding (up to ~2 z units late in a 20k-row sequence). f64 sums
/// would be closer to the true statistics but not to what the model saw (native/ONLINE_NORM_PORT.md).
/// Every later op is one float32 op in torch's order, with no reciprocal or FMA shortcuts.
#[derive(Debug, Clone)]
pub struct OnlineNorm {
    n: u32,
    sum: [f32; N_RAW],
    sumsq: [f32; N_RAW],
}

impl Default for OnlineNorm {
    fn default() -> Self {
        Self {
            n: 0,
            sum: [0.0; N_RAW],
            sumsq: [0.0; N_RAW],
        }
    }
}

impl OnlineNorm {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Adds `row` to the running sums and writes its 112 z-scores into `out`.
    pub fn update(&mut self, row: &[f32], out: &mut [f32]) {
        assert_eq!(row.len(), N_RAW, "row width");
        assert_eq!(out.len(), N_RAW, "z-score width");
        self.n += 1;
        let n = self.n as f32; // exact below 2^24 rows, like torch.arange in float32
        for ((o, &x), (s, sq)) in out
            .iter_mut()
            .zip(row)
            .zip(self.sum.iter_mut().zip(self.sumsq.iter_mut()))
        {
            *s += x;
            *sq += x * x;
            let mean = *s / n;
            let var = (*sq / n - mean * mean).max(0.0);
            *o = (x - mean) / (var + ONLINE_EPS).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(v: f32) -> [f32; N_RAW] {
        [v; N_RAW]
    }

    fn rows(n: usize) -> Vec<[f32; N_RAW]> {
        (0..n)
            .map(|r| std::array::from_fn(|c| ((r * 37 + c * 11) % 97) as f32 / 20.0 - 2.4))
            .collect()
    }

    #[test]
    fn online_norm_is_the_float32_prefix_computation_bit_for_bit() {
        let rows = rows(3000);
        let mut norm = OnlineNorm::default();
        let mut out = [0.0f32; N_RAW];
        for (t, r) in rows.iter().enumerate() {
            norm.update(r, &mut out);
            for c in [0, 5, 111] {
                let prefix = rows[..=t].iter().map(|v| v[c]);
                let s = prefix.clone().fold(0.0f32, |a, x| a + x);
                let sq = prefix.fold(0.0f32, |a, x| a + x * x);
                let n = (t + 1) as f32;
                let mean = s / n;
                let var = (sq / n - mean * mean).max(0.0);
                let want = (r[c] - mean) / (var + 1e-4f32).sqrt();
                assert_eq!(out[c].to_bits(), want.to_bits(), "t {t} c {c}");
            }
        }
    }

    #[test]
    fn online_norm_tracks_two_pass_statistics_and_resets() {
        let rows = rows(300);
        let mut norm = OnlineNorm::default();
        let mut out = [0.0f32; N_RAW];
        for (t, r) in rows.iter().enumerate() {
            norm.update(r, &mut out);
            for c in [0, 5, 111] {
                let xs: Vec<f64> = rows[..=t].iter().map(|v| v[c] as f64).collect();
                let mean = xs.iter().sum::<f64>() / xs.len() as f64;
                let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
                let want = (r[c] as f64 - mean) / (var + 1e-4).sqrt();
                assert!((out[c] as f64 - want).abs() < 1e-3, "t {t} c {c}");
            }
        }
        norm.reset();
        norm.update(&rows[5], &mut out);
        assert!(
            out.iter().all(|v| *v == 0.0),
            "first row of a sequence is its own mean"
        );
    }

    #[test]
    fn first_row_is_all_zero_except_nothing() {
        let mut s = StreamFeatures::default();
        let mut out = [9.0; N_EVENT];
        s.update(&row(1.5), &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn price_event_sets_flag_counter_and_held_delta() {
        let mut s = StreamFeatures::default();
        let mut out = [0.0; N_EVENT];
        let a = row(0.0);
        s.update(&a, &mut out);
        let mut b = a;
        b[0] = 1.0;
        b[11] = -2.0; // joint bid/ask price change
        s.update(&b, &mut out);
        assert_eq!(out[66], 1.0);
        assert_eq!(out[74], 0.0); // event at this step
        assert_eq!(out[77], 1.0);
        assert_eq!(out[88], -2.0);
        s.update(&b, &mut out); // quiet row: held deltas persist, counter grows
        assert_eq!(out[66], 0.0);
        assert_eq!(out[77], 1.0);
        assert!((out[74] - (1.0f64.ln_1p() as f32)).abs() < 1e-7);
    }

    #[test]
    fn sanitize_holds_previous_value_for_nonfinite_inputs() {
        let mut s = StreamFeatures::default();
        let mut out = [0.0; N_EVENT];
        let mut first = row(f32::NAN);
        s.sanitize(&mut first);
        assert!(first.iter().all(|v| *v == 0.0));
        s.update(&row(2.0), &mut out);
        let mut bad = row(3.0);
        bad[7] = f32::INFINITY;
        bad[50] = f32::NAN;
        s.sanitize(&mut bad);
        assert_eq!(bad[7], 2.0);
        assert_eq!(bad[50], 2.0);
        assert_eq!(bad[8], 3.0);
    }

    #[test]
    fn one_sided_price_change_is_not_an_event() {
        let mut s = StreamFeatures::default();
        let mut out = [0.0; N_EVENT];
        s.update(&row(0.0), &mut out);
        let mut b = row(0.0);
        b[0] = 1.0;
        s.update(&b, &mut out);
        assert_eq!(out[66], 0.0);
        assert_eq!(out[0], 1.0);
    }
}
