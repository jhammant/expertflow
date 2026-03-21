//! Compute dispatch: router, Metal GPU, ANE.

pub mod router;

#[cfg(feature = "metal4")]
pub mod metal4;
