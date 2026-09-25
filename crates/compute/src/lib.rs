pub mod cvar;
pub mod data;
pub mod error;
pub mod experiments;
pub mod format;
pub mod model;
pub mod regime;
pub mod scenarios;
pub mod trace;

pub use error::{ComputeError, Result};
pub use format::format_inr;
