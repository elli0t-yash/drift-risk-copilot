//! GARCH(1,1) volatility forecasting, fit from scratch via gradient-ascent
//! MLE on the portfolio's own daily log returns (not any single holding's).
//!
//! sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}, with
//! eps_t = r_t - mu (demeaned portfolio return). Forecasting uses the
//! closed-form mean-reversion identity toward the long-run variance
//! sigma2_inf = omega / (1 - alpha - beta).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{ComputeError, Result};

/// Minimum portfolio return observations to attempt a fit -- below this,
/// a GARCH(1,1) MLE (3 free parameters) has no meaningful signal to work
/// with. Matches `model::MIN_TICKER_OBSERVATIONS`, which already floors
/// every per-ticker return series this crate fits anything on, so a
/// portfolio return series (built from those same series, see
/// `experiments::portfolio_daily_log_returns`) is never shorter than this
/// in practice.
pub const MIN_OBSERVATIONS: usize = 30;
const DEFAULT_MAX_ITER: u32 = 500;
const CONVERGENCE_TOL: f64 = 1e-7;
/// Forecast horizons reported, in trading days.
const HORIZONS: [u32; 4] = [5, 10, 20, 60];
/// A forecast is only flagged "increasing"/"decreasing" (vs. "stable") if
/// it differs from the current annualized vol by more than this many
/// percentage points (spec value, mirrored from the narration rule).
const DIRECTION_THRESHOLD_PP: f64 = 0.5;
const TRADING_DAYS_PER_YEAR: f64 = 252.0;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GarchHorizonForecast {
    pub horizon_days: u32,
    pub vol_annualized: f64,
    /// "increasing" | "decreasing" | "stable", relative to
    /// `GarchForecast::current_vol_annualized` (see `DIRECTION_THRESHOLD_PP`).
    pub vol_direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GarchForecast {
    pub omega: f64,
    /// ARCH term.
    pub alpha: f64,
    /// GARCH term.
    pub beta: f64,
    /// `alpha + beta` -- how slowly volatility shocks decay; must be < 1
    /// for the process to be stationary (a long-run variance to exist at
    /// all).
    pub persistence: f64,
    /// `sqrt(252 * sigma2_inf)`, `sigma2_inf = omega / (1 - alpha - beta)`.
    pub long_run_vol_annualized: f64,
    /// `sqrt(252 * sigma2_T)`, the last fitted period's conditional
    /// variance.
    pub current_vol_annualized: f64,
    pub forecasts: Vec<GarchHorizonForecast>,
    /// `false` if the MLE never reached `CONVERGENCE_TOL` within the
    /// iteration budget -- `omega`/`alpha`/`beta` are then the documented
    /// fallback (`omega = var(r) * (1 - 0.05 - 0.90), alpha = 0.05, beta =
    /// 0.90`), not a fitted result.
    pub converged: bool,
    /// The Gaussian log-likelihood at the final (fitted or fallback)
    /// parameters.
    pub log_likelihood: f64,
}

/// `sigma2_t` and each parameter's derivative of it, built up recursively
/// alongside the main GARCH recursion itself: `d_sigma2_t/d_theta` equals
/// `d_omega/d_theta` plus `beta` times `d_sigma2_{t-1}/d_theta`, plus
/// (for the alpha/beta terms specifically) `eps_{t-1}^2` or `sigma2_{t-1}`
/// respectively -- the standard GARCH(1,1) score derivation. `sigma2[0]`
/// is seeded at the sample variance and is treated as a constant, not a
/// function of the parameters, so its own derivative row is zero.
struct GarchPath {
    sigma2: Vec<f64>,
    log_likelihood: f64,
    d_omega: f64,
    d_alpha: f64,
    d_beta: f64,
}

fn garch_path(eps: &[f64], sigma2_seed: f64, omega: f64, alpha: f64, beta: f64) -> GarchPath {
    let n = eps.len();
    let mut sigma2 = vec![0.0; n];
    let mut d_omega = vec![0.0; n];
    let mut d_alpha = vec![0.0; n];
    let mut d_beta = vec![0.0; n];

    sigma2[0] = sigma2_seed;
    let two_pi_ln = (2.0 * std::f64::consts::PI).ln();
    let mut log_likelihood = -0.5 * (two_pi_ln + sigma2[0].ln() + eps[0] * eps[0] / sigma2[0]);

    let mut grad_omega = 0.0;
    let mut grad_alpha = 0.0;
    let mut grad_beta = 0.0;

    for t in 1..n {
        sigma2[t] = omega + alpha * eps[t - 1] * eps[t - 1] + beta * sigma2[t - 1];
        d_omega[t] = 1.0 + beta * d_omega[t - 1];
        d_alpha[t] = eps[t - 1] * eps[t - 1] + beta * d_alpha[t - 1];
        d_beta[t] = sigma2[t - 1] + beta * d_beta[t - 1];

        let s2 = sigma2[t].max(1e-12);
        log_likelihood += -0.5 * (two_pi_ln + s2.ln() + eps[t] * eps[t] / s2);

        // d(log_likelihood_t)/d(sigma2_t) = 0.5 * (eps_t^2/sigma2_t^2 - 1/sigma2_t),
        // chained through d(sigma2_t)/d(theta) via the recursion above.
        let common = 0.5 * (eps[t] * eps[t] / (s2 * s2) - 1.0 / s2);
        grad_omega += common * d_omega[t];
        grad_alpha += common * d_alpha[t];
        grad_beta += common * d_beta[t];
    }

    GarchPath {
        sigma2,
        log_likelihood,
        d_omega: grad_omega,
        d_alpha: grad_alpha,
        d_beta: grad_beta,
    }
}

/// Projects `(omega, alpha, beta)` onto the constraint set (`omega > 0`,
/// `alpha > 0`, `beta > 0`, `alpha + beta < 1`) after a gradient step --
/// per spec. The stationarity bound is enforced at `0.999`, not `1.0`, so
/// `sigma2_inf = omega / (1 - alpha - beta)` never divides by (near) zero.
fn project(omega: f64, alpha: f64, beta: f64) -> (f64, f64, f64) {
    let omega = omega.max(1e-10);
    let alpha = alpha.max(1e-8);
    let beta = beta.max(1e-8);
    if alpha + beta >= 0.999 {
        let scale = 0.999 / (alpha + beta);
        (omega, alpha * scale, beta * scale)
    } else {
        (omega, alpha, beta)
    }
}

/// `fit_garch` with an explicit iteration cap, so a test can force
/// non-convergence (`max_iter = 1`) without waiting out the real budget --
/// `fit_garch` itself always calls this with `DEFAULT_MAX_ITER` (500, per
/// spec).
///
/// Gradient ascent with backtracking line search: the spec fixes the
/// initial parameters, the constraint set, the iteration budget, and the
/// convergence tolerance, but not a step size -- a single fixed learning
/// rate can't work well for `omega` (~1e-6 scale) and `alpha`/`beta`
/// (~0.05-0.9 scale) at once. Backtracking (halve the step until the
/// log-likelihood improves, or give up after a bounded number of halvings)
/// adapts the step size per iteration instead of needing a hand-tuned
/// per-parameter schedule; this is a judgment call the spec left open; see
/// the report.
pub fn fit_garch_with_max_iter(portfolio_returns: &[f64], max_iter: u32) -> Result<GarchForecast> {
    let n = portfolio_returns.len();
    if n < MIN_OBSERVATIONS {
        return Err(ComputeError::Model(format!(
            "GARCH(1,1) needs at least {MIN_OBSERVATIONS} portfolio return observations, got {n}"
        )));
    }

    let mu = portfolio_returns.iter().sum::<f64>() / n as f64;
    let eps: Vec<f64> = portfolio_returns.iter().map(|r| r - mu).collect();
    let sample_var = (eps.iter().map(|e| e * e).sum::<f64>() / n as f64).max(1e-12);

    let mut omega = 1e-6;
    let mut alpha = 0.05;
    let mut beta = 0.90;
    let mut ll = garch_path(&eps, sample_var, omega, alpha, beta).log_likelihood;
    let mut converged = false;

    for _ in 0..max_iter {
        let path = garch_path(&eps, sample_var, omega, alpha, beta);
        let grad_norm = (path.d_omega.powi(2) + path.d_alpha.powi(2) + path.d_beta.powi(2))
            .sqrt()
            .max(1e-300);

        let mut step = 1e-2 / grad_norm;
        let mut accepted = None;
        for _ in 0..40 {
            let (cand_omega, cand_alpha, cand_beta) = project(
                omega + step * path.d_omega,
                alpha + step * path.d_alpha,
                beta + step * path.d_beta,
            );
            let cand_ll = garch_path(&eps, sample_var, cand_omega, cand_alpha, cand_beta).log_likelihood;
            if cand_ll > ll {
                accepted = Some((cand_omega, cand_alpha, cand_beta, cand_ll));
                break;
            }
            step *= 0.5;
        }

        let Some((new_omega, new_alpha, new_beta, new_ll)) = accepted else {
            // No improving step found even after 40 halvings: the
            // gradient is effectively zero here -- a stationary point,
            // i.e. converged, not a failure to converge.
            converged = true;
            break;
        };

        let improved = new_ll - ll;
        omega = new_omega;
        alpha = new_alpha;
        beta = new_beta;
        ll = new_ll;
        if improved.abs() < CONVERGENCE_TOL {
            converged = true;
            break;
        }
    }

    if !converged {
        omega = sample_var * (1.0 - 0.05 - 0.90);
        alpha = 0.05;
        beta = 0.90;
        ll = garch_path(&eps, sample_var, omega, alpha, beta).log_likelihood;
    }

    let sigma2_t = *garch_path(&eps, sample_var, omega, alpha, beta).sigma2.last().unwrap();
    let persistence = alpha + beta;
    let long_run_variance = omega / (1.0 - persistence).max(1e-8);

    let current_vol_annualized = (TRADING_DAYS_PER_YEAR * sigma2_t).sqrt();
    let long_run_vol_annualized = (TRADING_DAYS_PER_YEAR * long_run_variance).sqrt();

    let forecasts = HORIZONS
        .iter()
        .map(|&h| {
            let sigma2_h = forecast_sigma2(sigma2_t, long_run_variance, persistence, h);
            let vol_annualized = (TRADING_DAYS_PER_YEAR * sigma2_h.max(0.0)).sqrt();
            let diff_pp = (vol_annualized - current_vol_annualized) * 100.0;
            let vol_direction = if diff_pp > DIRECTION_THRESHOLD_PP {
                "increasing"
            } else if diff_pp < -DIRECTION_THRESHOLD_PP {
                "decreasing"
            } else {
                "stable"
            }
            .to_string();
            GarchHorizonForecast { horizon_days: h, vol_annualized, vol_direction }
        })
        .collect();

    Ok(GarchForecast {
        omega,
        alpha,
        beta,
        persistence,
        long_run_vol_annualized,
        current_vol_annualized,
        forecasts,
        converged,
        log_likelihood: ll,
    })
}

/// `E[sigma2_{T+h}] = sigma2_inf + persistence^h * (sigma2_T - sigma2_inf)`
/// -- exposed standalone (not just inlined into `fit_garch_with_max_iter`)
/// so its `h=0` identity (`sigma2_T` exactly) is directly testable.
pub fn forecast_sigma2(sigma2_t: f64, long_run_variance: f64, persistence: f64, horizon_days: u32) -> f64 {
    long_run_variance + persistence.powi(horizon_days as i32) * (sigma2_t - long_run_variance)
}

/// Fits GARCH(1,1) on `portfolio_returns` (the portfolio's own daily log
/// returns, not any single holding's) via gradient-ascent MLE, with the
/// documented method-of-moments fallback if it fails to converge within
/// 500 iterations.
pub fn fit_garch(portfolio_returns: &[f64]) -> Result<GarchForecast> {
    fit_garch_with_max_iter(portfolio_returns, DEFAULT_MAX_ITER)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_signed(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            unit * 2.0 - 1.0
        }
    }

    /// Generates returns from a *known* ARCH(1) process (beta=0, so the
    /// conditional variance has no memory beyond the last shock) via
    /// Box-Muller-driven standard normal innovations scaled by the
    /// process's own sigma_t -- a deterministic, dependency-free synthetic
    /// generator matching this crate's established no-`rand` convention.
    fn synthetic_arch_process(true_omega: f64, true_alpha: f64, true_beta: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng(seed.max(1));
        let mut standard_normal = || -> f64 {
            // Box-Muller, using the xorshift PRNG's uniform-in-[-1,1]
            // output remapped to (0,1) for both draws.
            let u1 = (rng.next_signed() + 1.0) / 2.0;
            let u2 = (rng.next_signed() + 1.0) / 2.0;
            let u1 = u1.max(1e-12);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut sigma2 = true_omega / (1.0 - true_alpha - true_beta);
        let mut eps_prev2 = sigma2;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            sigma2 = true_omega + true_alpha * eps_prev2 + true_beta * sigma2;
            let eps = standard_normal() * sigma2.sqrt();
            eps_prev2 = eps * eps;
            out.push(eps);
        }
        out
    }

    #[test]
    fn fitted_params_satisfy_constraints() {
        let returns = synthetic_arch_process(1e-6, 0.08, 0.85, 800, 11);
        let fit = fit_garch(&returns).unwrap();
        assert!(fit.omega > 0.0);
        assert!(fit.alpha > 0.0);
        assert!(fit.beta > 0.0);
        assert!(fit.alpha + fit.beta < 0.9999, "persistence {} not < 0.9999", fit.persistence);
    }

    #[test]
    fn long_run_vol_is_positive() {
        let returns = synthetic_arch_process(1e-6, 0.08, 0.85, 800, 12);
        let fit = fit_garch(&returns).unwrap();
        assert!(fit.long_run_vol_annualized > 0.0);
    }

    #[test]
    fn forecast_at_horizon_zero_matches_current_vol() {
        let sigma2_t = 0.0002;
        let long_run_variance = 0.00015;
        let persistence = 0.95;
        let sigma2_0 = forecast_sigma2(sigma2_t, long_run_variance, persistence, 0);
        assert!((sigma2_0 - sigma2_t).abs() < 1e-6, "h=0 forecast {sigma2_0} != sigma2_T {sigma2_t}");
    }

    #[test]
    fn persistence_is_in_zero_one_range() {
        let returns = synthetic_arch_process(1e-6, 0.08, 0.85, 800, 13);
        let fit = fit_garch(&returns).unwrap();
        assert!((0.0..1.0).contains(&fit.persistence), "persistence {} not in [0,1)", fit.persistence);
    }

    /// On a synthetic process with known true alpha/beta, the fit should
    /// recover both within 0.05 (spec tolerance).
    #[test]
    fn recovers_known_alpha_beta_on_synthetic_arch_process() {
        let true_alpha = 0.08;
        let true_beta = 0.85;
        let returns = synthetic_arch_process(1e-6, true_alpha, true_beta, 1500, 21);
        let fit = fit_garch(&returns).unwrap();
        assert!(
            (fit.alpha - true_alpha).abs() < 0.05,
            "fitted alpha {} too far from true {true_alpha}",
            fit.alpha
        );
        assert!(
            (fit.beta - true_beta).abs() < 0.05,
            "fitted beta {} too far from true {true_beta}",
            fit.beta
        );
    }

    /// Forcing `max_iter=1` (one gradient step, nowhere near convergence
    /// for a generic starting point) must fall back to the documented
    /// method-of-moments parameters and report `converged: false`.
    #[test]
    fn insufficient_iterations_falls_back_to_method_of_moments() {
        let returns = synthetic_arch_process(1e-6, 0.08, 0.85, 800, 14);
        let fit = fit_garch_with_max_iter(&returns, 1).unwrap();
        assert!(!fit.converged);

        let mu = returns.iter().sum::<f64>() / returns.len() as f64;
        let sample_var =
            returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / returns.len() as f64;
        let expected_omega = sample_var * (1.0 - 0.05 - 0.90);
        assert!((fit.omega - expected_omega).abs() < 1e-12);
        assert_eq!(fit.alpha, 0.05);
        assert_eq!(fit.beta, 0.90);
    }

    #[test]
    fn errors_on_too_few_observations() {
        let returns = vec![0.001; 10];
        assert!(fit_garch(&returns).is_err());
    }
}
