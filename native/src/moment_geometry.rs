//! Reset-local causal moment and discrete occupancy features for four raw fields.

pub const SELECTED: [usize; 4] = [0, 3, 52, 55];
pub const MOMENT_WIDTH: usize = 20;
pub const DICTIONARY_WIDTH: usize = 40;

#[derive(Clone)]
pub struct FeatureState {
    beta: [f64; 4],
    intercept: [f64; 4],
    atoms: [f64; 4],
    vocabulary: Option<[f32; 12]>,
    count: u64,
    sums: [f64; 4],
    squares: [f64; 4],
    atom_counts: [u64; 4],
    previous: [f32; 4],
    category_counts: [u64; 16],
    dominant_counts: [u64; 4],
    event_counts: [u64; 4],
    last_change: [u64; 4],
}

impl FeatureState {
    pub fn new(
        beta: [f64; 4],
        intercept: [f64; 4],
        atoms: [f64; 4],
        vocabulary: Option<[f32; 12]>,
    ) -> Self {
        Self {
            beta,
            intercept,
            atoms,
            vocabulary,
            count: 0,
            sums: [0.0; 4],
            squares: [0.0; 4],
            atom_counts: [0; 4],
            previous: [0.0; 4],
            category_counts: [0; 16],
            dominant_counts: [0; 4],
            event_counts: [0; 4],
            last_change: [0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.sums = [0.0; 4];
        self.squares = [0.0; 4];
        self.atom_counts = [0; 4];
        self.previous = [0.0; 4];
        self.category_counts = [0; 16];
        self.dominant_counts = [0; 4];
        self.event_counts = [0; 4];
        self.last_change = [0; 4];
    }

    /// Appends the current raw row to its causal prefix and writes20 moment channels.
    pub fn moments(&mut self, raw: &[f32], out: &mut [f32; MOMENT_WIDTH]) {
        self.count += 1;
        let count = self.count as f64;
        let mut means = [0.0; 4];
        let mut stds = [0.0; 4];
        for k in 0..4 {
            let value = raw[SELECTED[k]] as f64;
            self.sums[k] += value;
            self.squares[k] += value * value;
            means[k] = self.sums[k] / count;
            let second = self.squares[k] / count;
            stds[k] = (second - means[k] * means[k]).max(0.0).sqrt();
            if value == self.atoms[k] {
                self.atom_counts[k] += 1;
            }
            out[k] = means[k] as f32;
            out[4 + k] = stds[k] as f32;
            out[8 + k] = (second - self.beta[k] * means[k] - self.intercept[k]) as f32;
            out[12 + k] = (self.atom_counts[k] as f64 / count) as f32;
            let normalized = ((value - means[k]) / stds[k].max(0.05)).clamp(-8.0, 8.0);
            out[16 + k] = normalized as f32;
        }
    }

    /// Appends a raw row and writes the40-channel current/category/occupancy/event suffix.
    pub fn discrete_features(&mut self, raw: &[f32], out: &mut [f32; DICTIONARY_WIDTH]) {
        out.fill(0.0);
        let Some(vocabulary) = self.vocabulary else {
            return;
        };
        let values = std::array::from_fn(|k| raw[SELECTED[k]]);
        self.discrete(&values, vocabulary, out);
    }

    fn discrete(
        &mut self,
        values: &[f32; 4],
        vocabulary: [f32; 12],
        out: &mut [f32; DICTIONARY_WIDTH],
    ) {
        let count = self.count;
        for field in 0..4 {
            let value = values[field];
            let mut category = 3usize;
            for level in 0..3 {
                if value == vocabulary[field * 3 + level] {
                    category = level;
                    break;
                }
            }
            let channel = field * 4 + category;
            self.category_counts[channel] += 1;
            out[channel] = 1.0;
            out[16 + channel] = (self.category_counts[channel] as f64 / count as f64) as f32;

            let changed = count == 1 || value != self.previous[field];
            if changed {
                self.event_counts[field] += 1;
                self.last_change[field] = count - 1;
                if category == 0 {
                    self.dominant_counts[field] += 1;
                }
            }
            let events = self.event_counts[field] as f64;
            out[32 + field] = (self.dominant_counts[field] as f64 / events) as f32;
            let position = count - 1;
            let age = position.saturating_sub(self.last_change[field]).min(10_000);
            out[36 + field] = (age as f64).ln_1p() as f32;
            self.previous[field] = value;
        }
    }
}
