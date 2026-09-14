//! Native kernels for research and submission inference.

pub mod approx;
pub mod kernels;
pub mod rnn;
pub mod rnn_fast;

#[cfg(feature = "python")]
mod python;
