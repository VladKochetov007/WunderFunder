//! Native kernels for research and submission inference.

pub mod approx;
pub mod factorized;
pub mod half;
pub mod kernels;
pub mod quant8;
pub mod rnn;
pub mod rnn_fast;
pub mod stream;

#[cfg(feature = "python")]
mod python;
