//! Factor model: per-stock OLS on the five fixed factors, Ledoit-Wolf
//! shrinkage of the factor covariance matrix, and the resulting stock
//! covariance Sigma = B F B^T + D.

use nalgebra::{DMatrix, DVector};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::{MarketData, FACTOR_NAMES};
use crate::error::{ComputeError, Result};

/// Return frequency the factor model is fit at. Non-overlapping: `Weekly`
/// returns are computed between successive week-end closes, not a rolling
/// 5-day window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub enum Frequency {
    #[default]
    Daily,
    Weekly,
}

impl Frequency {
    /// Periods per year at this frequency. This is the single place the
    /// annualization factor is defined: everything upstream (OLS,
    /// Ledoit-Wolf shrinkage) is computed on period returns at whatever
    /// frequency was chosen, and `annualize_matrix`/`annualize_scalar` are
    /// the only functions that scale a period (co)variance into annual
    /// terms, using this value.
    pub fn annualization_factor(&self) -> f64 {
        match self {
            Frequency::Daily => 252.0,
            Frequency::Weekly => 52.0,
        }
    }

    /// Default trailing window, in periods at this frequency: 252 trading
    /// days, or 156 weeks (~3 years, chosen to keep a comparable number of
    /// independent observations to the daily 252-day/~1yr default given
    /// weekly data's lower observation density).
    pub fn default_window(&self) -> usize {
        match self {
            Frequency::Daily => 252,
            Frequency::Weekly => 156,
        }
    }
}

pub fn annualize_scalar(period_variance: f64, frequency: Frequency) -> f64 {
    period_variance * frequency.annualization_factor()
}

pub fn annualize_matrix(period_covariance: &DMatrix<f64>, frequency: Frequency) -> DMatrix<f64> {
    period_covariance * frequency.annualization_factor()
}

/// serde `default =` helper: default trailing window for experiment inputs
/// (daily convention; callers on `Frequency::Weekly` should override).
pub fn default_window() -> usize {
    Frequency::Daily.default_window()
}

/// Per-stock OLS fit against the five fixed factors.
#[derive(Debug, Clone)]
pub struct StockFit {
    pub ticker: String,
    pub intercept: f64,
    /// Betas in `FACTOR_NAMES` order.
    pub betas: Vec<f64>,
    /// Daily residual variance (unannualized).
    pub residual_variance_daily: f64,
    pub r_squared: f64,
    pub n_obs: usize,
}

/// The fitted factor model for a set of stocks over a trailing window.
pub struct FactorModel {
    pub frequency: Frequency,
    pub window: usize,
    pub tickers: Vec<String>,
    pub fits: Vec<StockFit>,
    /// Per-period (daily or weekly, per `frequency`) factor covariance
    /// after Ledoit-Wolf shrinkage, in `FACTOR_NAMES` order (rows/cols).
    pub factor_covariance_daily: DMatrix<f64>,
    pub shrinkage_intensity: f64,
}

impl FactorModel {
    /// Beta matrix B (n_stocks x n_factors), `FACTOR_NAMES` column order.
    pub fn beta_matrix(&self) -> DMatrix<f64> {
        let n = self.fits.len();
        let k = FACTOR_NAMES.len();
        DMatrix::from_fn(n, k, |i, j| self.fits[i].betas[j])
    }

    /// Diagonal matrix D of per-period residual variances (n_stocks x n_stocks).
    pub fn residual_matrix_daily(&self) -> DMatrix<f64> {
        let n = self.fits.len();
        DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                self.fits[i].residual_variance_daily
            } else {
                0.0
            }
        })
    }

    /// Annualized stock covariance Sigma = B F B^T + D.
    pub fn stock_covariance(&self) -> DMatrix<f64> {
        let b = self.beta_matrix();
        let f = &self.factor_covariance_daily;
        let d = self.residual_matrix_daily();
        let period_sigma = &b * f * b.transpose() + d;
        annualize_matrix(&period_sigma, self.frequency)
    }

    /// Annualized factor covariance F.
    pub fn factor_covariance(&self) -> DMatrix<f64> {
        annualize_matrix(&self.factor_covariance_daily, self.frequency)
    }
}

/// Ordinary least squares with intercept: y ~ 1 + X, solved via the
/// Moore-Penrose pseudo-inverse (SVD) for numerical robustness on
/// possibly collinear factor windows.
fn ols_fit(y: &DVector<f64>, factors: &DMatrix<f64>) -> Result<(f64, Vec<f64>, f64, f64)> {
    let n = y.len();
    let k = factors.ncols();
    if n <= k + 1 {
        return Err(ComputeError::Model(format!(
            "not enough observations ({n}) for {k} factors plus intercept"
        )));
    }

    let mut design = DMatrix::from_element(n, k + 1, 1.0);
    for i in 0..n {
        for j in 0..k {
            design[(i, j + 1)] = factors[(i, j)];
        }
    }

    let svd = design.clone().svd(true, true);
    let coeffs = svd
        .solve(y, 1e-12)
        .map_err(|e| ComputeError::Model(format!("OLS solve failed: {e}")))?;

    let intercept = coeffs[0];
    let betas: Vec<f64> = (0..k).map(|j| coeffs[j + 1]).collect();

    let fitted = &design * &coeffs;
    let residuals = y - &fitted;
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let y_mean = y.mean();
    let tss: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };
    // Residual variance uses n - k - 1 degrees of freedom (intercept + k betas).
    let dof = (n - k - 1).max(1) as f64;
    let residual_variance = rss / dof;

    Ok((intercept, betas, residual_variance, r_squared))
}

/// Ledoit-Wolf shrinkage of the sample covariance of `data` (T x N, rows =
/// observations) toward a scaled-identity target `mu * I`, following
/// Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".
///
/// Returns `(shrunk_covariance, shrinkage_intensity)`.
pub fn ledoit_wolf_shrink_identity(data: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
    let t = data.nrows() as f64;
    let n = data.ncols();

    // Demean each column (factor) over the window.
    let means: Vec<f64> = (0..n)
        .map(|j| data.column(j).iter().sum::<f64>() / t)
        .collect();
    let mut centered = data.clone();
    for j in 0..n {
        for i in 0..data.nrows() {
            centered[(i, j)] -= means[j];
        }
    }

    // Sample covariance S = (1/T) X'X.
    let sample_cov: DMatrix<f64> = (&centered.transpose() * &centered) / t;

    // Target: mu * I, mu = trace(S) / N.
    let mu = sample_cov.trace() / n as f64;
    let target = DMatrix::identity(n, n) * mu;

    // pi_hat: average over t of ||x_t x_t' - S||_F^2.
    let mut pi_sum = 0.0;
    for i in 0..data.nrows() {
        let row = centered.row(i).transpose();
        let outer = &row * row.transpose();
        let diff = &outer - &sample_cov;
        pi_sum += diff.iter().map(|v| v * v).sum::<f64>();
    }
    let pi_hat = pi_sum / t;

    // gamma_hat = ||S - target||_F^2 (rho_hat = 0 for an identity target:
    // the target has no estimation error and zero off-diagonal covariance
    // with the sample covariance's off-diagonal entries).
    let diff_target = &sample_cov - &target;
    let gamma_hat = diff_target.iter().map(|v| v * v).sum::<f64>();

    let shrinkage = if gamma_hat > 0.0 {
        (pi_hat / (t * gamma_hat)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let shrunk = &target * shrinkage + &sample_cov * (1.0 - shrinkage);
    (shrunk, shrinkage)
}

/// Fits the factor model for `tickers` over the trailing `window` periods
/// (trading days or weeks, per `frequency`) of `data` (the most recent
/// `window` return observations, which must already be at `frequency`).
pub fn fit_factor_model(
    data: &MarketData,
    tickers: &[String],
    window: usize,
    frequency: Frequency,
) -> Result<FactorModel> {
    let factor_matrices: Vec<&Vec<f64>> = FACTOR_NAMES
        .iter()
        .map(|name| {
            data.factor_returns
                .get(*name)
                .ok_or_else(|| ComputeError::Model(format!("missing factor series {name}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let total_obs = factor_matrices[0].len();
    if total_obs < window {
        return Err(ComputeError::Model(format!(
            "only {total_obs} return observations available, need {window} for the trailing window"
        )));
    }
    let start = total_obs - window;

    let k = FACTOR_NAMES.len();
    let factors_window = DMatrix::from_fn(window, k, |i, j| factor_matrices[j][start + i]);

    let (factor_covariance_daily, shrinkage_intensity) =
        ledoit_wolf_shrink_identity(&factors_window);

    let mut fits = Vec::with_capacity(tickers.len());
    for ticker in tickers {
        let series = data
            .stock_returns
            .get(ticker)
            .ok_or_else(|| ComputeError::Model(format!("missing return series for {ticker}")))?;
        if series.len() < window {
            return Err(ComputeError::Model(format!(
                "ticker {ticker} has only {} observations, need {window}",
                series.len()
            )));
        }
        let series_start = series.len() - window;
        let y = DVector::from_fn(window, |i, _| series[series_start + i]);

        let (intercept, betas, residual_variance_daily, r_squared) =
            ols_fit(&y, &factors_window)?;

        fits.push(StockFit {
            ticker: ticker.clone(),
            intercept,
            betas,
            residual_variance_daily,
            r_squared,
            n_obs: window,
        });
    }

    Ok(FactorModel {
        frequency,
        window,
        tickers: tickers.to_vec(),
        fits,
        factor_covariance_daily,
        shrinkage_intensity,
    })
}
