//! Native kernels for research and submission inference.

pub mod approx;
pub mod factorized;
pub mod half;
pub mod kernels;
pub mod min_gru;
pub mod quant8;
pub mod rnn;
pub mod rnn_fast;
pub mod state_init;
pub mod stream;
pub mod vnni;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
mod python_geometry;

pub mod moment_geometry;
