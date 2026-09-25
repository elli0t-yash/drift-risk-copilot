//! Shared context threaded into `dispatch::run_experiment` for experiment
//! types that need more than their own input to run. Currently only
//! `RiskDrift` uses it (read access to the snapshot store, to resolve its
//! baseline); every other experiment type receives it but ignores it.

use std::sync::Arc;

use store::SnapshotStore;

pub struct ExperimentContext {
    pub store: Arc<SnapshotStore>,
    /// The current request's portfolio, hashed the same way
    /// `store::RiskSnapshot::portfolio_hash` is (see
    /// `crate::portfolio::portfolio_hash`) -- computed once by the caller
    /// (`server::backend`) and reused here rather than re-hashed per
    /// experiment.
    pub portfolio_hash: String,
}
