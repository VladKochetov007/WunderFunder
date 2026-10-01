//! Native kernels for research and submission inference.

pub mod approx;
pub mod kernels;
pub mod merged;
pub mod rnn;
pub mod rnn_fast;
pub mod stream;

#[cfg(feature = "python")]
mod python;
