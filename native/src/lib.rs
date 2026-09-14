//! Native kernels for research and submission inference.

pub mod kernels;
pub mod rnn;

#[cfg(feature = "python")]
mod python;
