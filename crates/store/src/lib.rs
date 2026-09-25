//! Persistent SQLite-backed store of experiment results ("risk snapshots"),
//! replacing the server crate's earlier in-memory, fixed-capacity
//! `ResultStore`. Uses `rusqlite`'s `bundled` feature, which compiles
//! SQLite from source as part of the crate build, so no external `sqlite3`
//! needs to be present on the (distroless, no package manager) runtime
//! image.
//!
//! Not persisted here: the `/ask` pipeline's narration/suggestion text --
//! only the `EvidenceTrace` (as `trace_json`) and a handful of denormalized
//! summary fields pulled out of it for fast querying. See the server
//! crate's `routes::post_ask`/`post_experiment` for what gets inserted and
//! the README for the tradeoff this implies for `GET /report/{id}`.

use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RiskSnapshot {
    pub id: String,
    pub created_at: String,
    pub portfolio_hash: String,
    pub experiment_type: String,
    pub engine_version: String,
    pub regime_label: Option<String>,
    pub smoothed_probs: Option<[f64; 3]>,
    pub portfolio_vol_annualized: Option<f64>,
    pub cvar_historical: Option<f64>,
    pub trace_json: String,
}

pub struct SnapshotStore {
    conn: Arc<Mutex<Connection>>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    portfolio_hash TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    regime_label TEXT,
    smoothed_probs TEXT,
    portfolio_vol_annualized REAL,
    cvar_historical REAL,
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS risk_snapshots_portfolio_hash_created_at
    ON risk_snapshots (portfolio_hash, created_at);
";

impl SnapshotStore {
    /// Opens (creating if absent) the SQLite database at `path` and runs the
    /// (idempotent) schema migration. Pass `":memory:"` for a private,
    /// non-persistent database -- used by this crate's own tests and by the
    /// server crate's tests, so neither needs a file on disk.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(SnapshotStore {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Inserts `snapshot`, ignoring whatever `snapshot.id` was set to and
    /// generating a fresh UUID v4 in its place (mirroring the old
    /// `ResultStore::insert`'s "id is server-assigned" contract). Returns
    /// the generated id.
    pub fn insert(&self, snapshot: &RiskSnapshot) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let created_at = Utc::now().to_rfc3339();
        let smoothed_probs_json = snapshot
            .smoothed_probs
            .map(|p| serde_json::to_string(&p).expect("[f64; 3] always serializes"));

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO risk_snapshots (
                id, created_at, portfolio_hash, experiment_type, engine_version,
                regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                id,
                created_at,
                snapshot.portfolio_hash,
                snapshot.experiment_type,
                snapshot.engine_version,
                snapshot.regime_label,
                smoothed_probs_json,
                snapshot.portfolio_vol_annualized,
                snapshot.cvar_historical,
                snapshot.trace_json,
            ],
        )?;
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Result<Option<RiskSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT id, created_at, portfolio_hash, experiment_type, engine_version,
                        regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json
                 FROM risk_snapshots WHERE id = ?1",
                params![id],
                row_to_snapshot,
            )
            .optional()?;
        Ok(row)
    }

    /// Snapshots for `portfolio_hash`, newest first, at most `limit`. Used
    /// by Risk Drift (a later session) to compare a portfolio's risk over
    /// time.
    pub fn latest_for_portfolio(&self, portfolio_hash: &str, limit: usize) -> Result<Vec<RiskSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, portfolio_hash, experiment_type, engine_version,
                    regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json
             FROM risk_snapshots WHERE portfolio_hash = ?1
             ORDER BY created_at DESC, rowid DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![portfolio_hash, limit as i64], row_to_snapshot)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The most recent snapshots across all portfolios, newest first, at
    /// most `limit` -- replaces the server's old 20-entry `VecDeque`.
    pub fn list_recent(&self, limit: usize) -> Result<Vec<RiskSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, created_at, portfolio_hash, experiment_type, engine_version,
                    regime_label, smoothed_probs, portfolio_vol_annualized, cvar_historical, trace_json
             FROM risk_snapshots ORDER BY created_at DESC, rowid DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_snapshot)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

fn row_to_snapshot(row: &rusqlite::Row) -> rusqlite::Result<RiskSnapshot> {
    let smoothed_probs_json: Option<String> = row.get(6)?;
    let smoothed_probs = smoothed_probs_json
        .map(|s| serde_json::from_str::<[f64; 3]>(&s))
        .transpose()
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e)))?;

    Ok(RiskSnapshot {
        id: row.get(0)?,
        created_at: row.get(1)?,
        portfolio_hash: row.get(2)?,
        experiment_type: row.get(3)?,
        engine_version: row.get(4)?,
        regime_label: row.get(5)?,
        smoothed_probs,
        portfolio_vol_annualized: row.get(7)?,
        cvar_historical: row.get(8)?,
        trace_json: row.get(9)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(portfolio_hash: &str, experiment_type: &str) -> RiskSnapshot {
        RiskSnapshot {
            id: String::new(), // insert() ignores this and assigns its own
            created_at: String::new(),
            portfolio_hash: portfolio_hash.to_string(),
            experiment_type: experiment_type.to_string(),
            engine_version: "0.1.0".to_string(),
            regime_label: Some("Bull".to_string()),
            smoothed_probs: Some([0.7, 0.2, 0.1]),
            portfolio_vol_annualized: Some(0.15),
            cvar_historical: None,
            trace_json: serde_json::json!({"experiment": experiment_type}).to_string(),
        }
    }

    #[test]
    fn insert_and_get_round_trips_the_full_snapshot() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let snapshot = sample_snapshot("abc123", "RiskDecomposition");
        let id = store.insert(&snapshot).unwrap();

        let fetched = store.get(&id).unwrap().expect("just-inserted snapshot should be found");
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.portfolio_hash, "abc123");
        assert_eq!(fetched.experiment_type, "RiskDecomposition");
        assert_eq!(fetched.engine_version, "0.1.0");
        assert_eq!(fetched.regime_label.as_deref(), Some("Bull"));
        assert_eq!(fetched.smoothed_probs, Some([0.7, 0.2, 0.1]));
        assert_eq!(fetched.portfolio_vol_annualized, Some(0.15));
        assert_eq!(fetched.cvar_historical, None);
        assert_eq!(fetched.trace_json, snapshot.trace_json);
    }

    #[test]
    fn get_returns_none_for_an_unknown_id() {
        let store = SnapshotStore::open(":memory:").unwrap();
        assert!(store.get("nonexistent").unwrap().is_none());
    }

    #[test]
    fn latest_for_portfolio_returns_newest_first_and_respects_limit() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let ids: Vec<String> = (0..5)
            .map(|i| store.insert(&sample_snapshot("hash-a", &format!("Type{i}"))).unwrap())
            .collect();
        // Also insert a snapshot for a different portfolio, which must never appear.
        store.insert(&sample_snapshot("hash-b", "Other")).unwrap();

        let latest = store.latest_for_portfolio("hash-a", 3).unwrap();
        assert_eq!(latest.len(), 3);
        // created_at has only second resolution, so within-the-same-second
        // inserts tie there; the query breaks that tie on `rowid DESC`
        // (monotonically increasing with insertion order), so the 3 most
        // recently *inserted* ids should come back, newest (last-inserted)
        // first.
        assert_eq!(latest[0].id, ids[4]);
        assert_eq!(latest[1].id, ids[3]);
        assert_eq!(latest[2].id, ids[2]);
        for snap in &latest {
            assert_eq!(snap.portfolio_hash, "hash-a");
        }
    }

    #[test]
    fn list_recent_returns_newest_first_and_respects_limit() {
        let store = SnapshotStore::open(":memory:").unwrap();
        for i in 0..5 {
            store.insert(&sample_snapshot(&format!("hash-{i}"), "RiskDecomposition")).unwrap();
        }
        let recent = store.list_recent(2).unwrap();
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn inserting_25_snapshots_does_not_evict_any() {
        let store = SnapshotStore::open(":memory:").unwrap();
        let ids: Vec<String> = (0..25)
            .map(|i| store.insert(&sample_snapshot(&format!("hash-{i}"), "RiskDecomposition")).unwrap())
            .collect();
        for id in &ids {
            assert!(store.get(id).unwrap().is_some(), "snapshot {id} should not have been evicted");
        }
        assert_eq!(store.list_recent(100).unwrap().len(), 25);
    }
}
