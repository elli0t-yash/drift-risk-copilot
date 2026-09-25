//! In-memory, fixed-capacity store of recent `/ask` results, keyed by a
//! server-generated UUID, so `GET /report/{id}` can render a PDF for a
//! result without the client re-sending the whole trace. Deliberately not
//! persistent (no database, no disk) -- a demo-scale ring buffer, evicting
//! the oldest entry once full.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent::pipeline::PipelineResult;
use uuid::Uuid;

const DEFAULT_CAPACITY: usize = 20;

pub struct ResultStore {
    inner: Arc<Mutex<VecDeque<(Uuid, PipelineResult)>>>,
    capacity: usize,
}

impl ResultStore {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        ResultStore {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    /// Inserts `result` under a freshly generated UUID v4, evicting the
    /// oldest entry first if already at capacity. Returns the new UUID.
    pub fn insert(&self, result: PipelineResult) -> Uuid {
        let id = Uuid::new_v4();
        let mut queue = self.inner.lock().unwrap();
        if queue.len() >= self.capacity {
            queue.pop_front();
        }
        queue.push_back((id, result));
        id
    }

    /// A clone of the stored result for `id`, if still present (not yet
    /// evicted). Cloning (rather than returning a reference) is required
    /// here: the entry stays in the shared queue for any other caller,
    /// this just hands the caller its own copy.
    pub fn get(&self, id: &Uuid) -> Option<PipelineResult> {
        let queue = self.inner.lock().unwrap();
        queue.iter().find(|(entry_id, _)| entry_id == id).map(|(_, r)| r.clone())
    }
}

impl Default for ResultStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::grounding::GroundedNarration;
    use compute::data::DataQuality;
    use compute::experiments::{Experiment, Holding, Portfolio, RiskDecompositionInput};
    use compute::model::Frequency;
    use compute::trace::{DataWindow, EvidenceTrace, ModelParams};

    fn sample_result(tag: &str) -> PipelineResult {
        PipelineResult {
            experiment: Experiment::RiskDecomposition(RiskDecompositionInput {
                portfolio: Portfolio {
                    holdings: vec![Holding {
                        ticker: tag.to_string(),
                        weight: 1.0,
                    }],
                    total_value_inr: 1.0,
                },
                frequency: Frequency::Daily,
                window: None,
                regime_covariance: false,
            }),
            trace: EvidenceTrace {
                experiment: "RiskDecomposition".to_string(),
                inputs: serde_json::json!({}),
                data_window: DataWindow {
                    frequency: Frequency::Daily,
                    window_periods: 252,
                    start: chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                    end: chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                },
                data_quality: DataQuality {
                    date_range_start: chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
                    date_range_end: chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                    trading_days: 1,
                    per_series: vec![],
                },
                model_params: ModelParams {
                    frequency: Frequency::Daily,
                    window_periods: 252,
                    factor_names: vec![],
                    shrinkage_intensity: 0.0,
                    annualization_factor: 252.0,
                    regime_state: None,
                    regime_fallback_warnings: vec![],
                    cap_source: None,
                },
                outputs: serde_json::json!({}),
                invariants: vec![],
                engine_version: "0.1.0".to_string(),
            },
            narration: GroundedNarration {
                narration: tag.to_string(),
                grounding_warnings: vec![],
            },
            assistant_turn: agent::ConversationTurn::assistant(tag),
            suggestion: "follow up?".to_string(),
        }
    }

    #[test]
    fn oldest_is_evicted_once_over_capacity_and_the_newest_20_remain() {
        let store = ResultStore::with_capacity(20);
        let ids: Vec<Uuid> = (0..21).map(|i| store.insert(sample_result(&i.to_string()))).collect();

        assert!(store.get(&ids[0]).is_none(), "oldest (index 0) should have been evicted");
        for id in &ids[1..] {
            assert!(store.get(id).is_some(), "entry {id} should still be retrievable");
        }
        assert_eq!(ids.len(), 21);
    }
}
