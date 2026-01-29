"""
CUPED: Controlled-Experiment Using Pre-Experiment Data
=======================================================

Variance reduction technique that uses pre-experiment covariates to reduce
standard errors and increase statistical power without increasing sample size.

Reference:
----------
Deng et al. (2013): "Improving the Sensitivity of Online Controlled Experiments
by Utilizing Pre-Experiment Data", WSDM '13.

Example Usage:
--------------
>>> from ab_testing.variance_reduction import cuped
>>> import numpy as np
>>>
>>> # Outcome and pre-experiment covariate
>>> y = np.random.normal(100, 20, 1000)
>>> x = y * 0.7 + np.random.normal(0, 15, 1000)  # Correlated
>>> treatment = np.random.binomial(1, 0.5, 1000)
>>>
>>> # Apply CUPED adjustment
>>> y_adjusted = cuped.cuped_adjustment(y, x)
>>> print(f"Variance reduction: {(1 - y_adjusted.var() / y.var()) * 100:.1f}%")
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def cuped_adjustment(
    y: np.ndarray,
    x: np.ndarray,
    theta: Optional[float] = None,
) -> np.ndarray:
    """
    Apply CUPED variance reduction adjustment.

    Formula: Y_adj = Y - θ(X - X̄)
    where θ = Cov(Y,X) / Var(X)

    Parameters
    ----------
    y : np.ndarray
        Outcome metric (post-experiment)
    x : np.ndarray
        Pre-experiment covariate
    theta : float, optional
        Adjustment coefficient. If None, computed from data.

    Returns
    -------
    np.ndarray
        Adjusted outcome metric

    Notes
    -----
    - Variance reduction ≈ ρ² (correlation squared)
    - If ρ = 0.5, expect ~25% variance reduction
    - θ is estimated on control group in production to avoid bias

    Example
    -------
    >>> y = np.random.normal(100, 20, 1000)
    >>> x = y * 0.7 + np.random.normal(0, 15, 1000)
    >>> y_adj = cuped_adjustment(y, x)
    >>> print(f"Original variance: {y.var():.2f}")
    >>> print(f"Adjusted variance: {y_adj.var():.2f}")
    """
    if len(y) != len(x):
        raise ValueError("y and x must have same length")
    if len(y) < 2:
        raise ValueError("Need at least 2 observations")

    # Compute theta if not provided
    if theta is None:
        theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)

    # CUPED adjustment
    y_adjusted = y - theta * (x - x.mean())

    return y_adjusted


def cuped_ab_test(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,
    x_treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Run A/B test with CUPED variance reduction.

    Uses control group to estimate theta, then applies adjustment to both groups.
    This ensures the adjustment is unbiased.

    Parameters
    ----------
    y_control : np.ndarray
        Outcome metric for control group
    y_treatment : np.ndarray
        Outcome metric for treatment group
    x_control : np.ndarray
        Pre-experiment covariate for control group
    x_treatment : np.ndarray
        Pre-experiment covariate for treatment group
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with keys:
        - correlation: Correlation between outcome and covariate
        - theta: Adjustment coefficient
        - var_reduction: Variance reduction percentage
        - effect_raw: Raw treatment effect
        - effect_adjusted: CUPED-adjusted treatment effect
        - se_raw: Raw standard error
        - se_adjusted: CUPED-adjusted standard error
        - se_reduction: SE reduction percentage
        - ci_raw: Raw 95% confidence interval
        - ci_adjusted: CUPED-adjusted 95% confidence interval
        - p_value_raw: Raw p-value
        - p_value_adjusted: CUPED-adjusted p-value
        - sample_size_reduction: Equivalent sample size reduction percentage

    Example
    -------
    >>> np.random.seed(42)
    >>> y_c = np.random.normal(100, 20, 500)
    >>> y_t = np.random.normal(115, 20, 500)
    >>> x_c = y_c * 0.7 + np.random.normal(0, 15, 500)
    >>> x_t = y_t * 0.7 + np.random.normal(0, 15, 500)
    >>> result = cuped_ab_test(y_c, y_t, x_c, x_t)
    >>> print(f"SE reduction: {result['se_reduction']:.1f}%")
    """
    if len(y_control) != len(x_control):
        raise ValueError("y_control and x_control must have same length")
    if len(y_treatment) != len(x_treatment):
        raise ValueError("y_treatment and x_treatment must have same length")

    # Combine data to estimate correlation
    y_all = np.concatenate([y_control, y_treatment])
    x_all = np.concatenate([x_control, x_treatment])
    correlation = np.corrcoef(y_all, x_all)[0, 1]

    # Estimate theta on control group (unbiased)
    theta = np.cov(y_control, x_control, ddof=1)[0, 1] / x_control.var(ddof=1)

    # Apply CUPED adjustment
    y_control_adj = cuped_adjustment(y_control, x_control, theta=theta)
    y_treatment_adj = cuped_adjustment(y_treatment, x_treatment, theta=theta)

    # Raw analysis
    effect_raw = y_treatment.mean() - y_control.mean()
    se_raw = np.sqrt(y_control.var(ddof=1)/len(y_control) +
                     y_treatment.var(ddof=1)/len(y_treatment))
    t_stat_raw = effect_raw / se_raw
    p_value_raw = 2 * (1 - stats.t.cdf(abs(t_stat_raw),
                       df=len(y_control) + len(y_treatment) - 2))
    ci_raw = (effect_raw - 1.96*se_raw, effect_raw + 1.96*se_raw)

    # CUPED analysis
    effect_adjusted = y_treatment_adj.mean() - y_control_adj.mean()
    se_adjusted = np.sqrt(y_control_adj.var(ddof=1)/len(y_control_adj) +
                          y_treatment_adj.var(ddof=1)/len(y_treatment_adj))
    t_stat_adj = effect_adjusted / se_adjusted
    p_value_adjusted = 2 * (1 - stats.t.cdf(abs(t_stat_adj),
                            df=len(y_control) + len(y_treatment) - 2))
    ci_adjusted = (effect_adjusted - 1.96*se_adjusted,
                   effect_adjusted + 1.96*se_adjusted)

    # Variance reduction
    var_raw = y_all.var(ddof=1)
    var_adjusted = np.concatenate([y_control_adj, y_treatment_adj]).var(ddof=1)
    var_reduction = 1 - var_adjusted / var_raw

    # SE reduction
    se_reduction = 1 - se_adjusted / se_raw

    # Sample size reduction (power equivalence)
    # With same power, need (SE_adj/SE_raw)² × original sample size
    sample_size_reduction = 1 - (se_adjusted / se_raw)**2

    # Determine significance
    significant = p_value_adjusted < alpha

    return {
        'correlation': float(correlation),
        'theta': float(theta),
        'var_reduction': float(var_reduction),
        'mean_control': float(y_control.mean()),
        'mean_treatment': float(y_treatment.mean()),
        'difference': float(effect_adjusted),
        'effect_raw': float(effect_raw),
        'effect_adjusted': float(effect_adjusted),
        'se_raw': float(se_raw),
        'se_adjusted': float(se_adjusted),
        'se_adj_diff': float(se_adjusted),
        'se_reduction': float(se_reduction),
        'ci_raw': (float(ci_raw[0]), float(ci_raw[1])),
        'ci_adjusted': (float(ci_adjusted[0]), float(ci_adjusted[1])),
        'p_value': float(p_value_adjusted),
        'p_value_raw': float(p_value_raw),
        'p_value_adjusted': float(p_value_adjusted),
        'significant': bool(significant),
        'sample_size_reduction': float(sample_size_reduction),
    }


def multi_covariate_cuped(
    y: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    CUPED with multiple covariates using OLS residualization.

    Regresses Y on X and returns residuals + mean(Y).
    This is equivalent to CUPED with multiple θ coefficients.

    Parameters
    ----------
    y : np.ndarray
        Outcome metric (shape: n,)
    X : np.ndarray
        Pre-experiment covariates (shape: n × k)

    Returns
    -------
    np.ndarray
        Adjusted outcome metric

    Notes
    -----
    - More variance reduction than single-covariate CUPED
    - Watch for overfitting with many covariates
    - Consider regularization (Ridge/Lasso) if k is large

    Example
    -------
    >>> y = np.random.normal(100, 20, 1000)
    >>> X = np.random.normal(0, 1, (1000, 3))
    >>> y_adj = multi_covariate_cuped(y, X)
    """
    if len(y) != len(X):
        raise ValueError("y and X must have same number of rows")
    if X.ndim == 1:
        raise ValueError("X must be 2-dimensional (use X.reshape(-1, 1) for single feature)")

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # OLS: β = (X'X)⁻¹X'y
    beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

    # Residuals + mean(y) to preserve mean
    y_pred = X_with_intercept @ beta
    y_adjusted = y - (y_pred - y.mean())

    return y_adjusted


def estimate_required_sample_size_with_cuped(
    baseline_std: float,
    correlation: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Tuple[int, int, float]:
    """
    Estimate sample size reduction from CUPED.

    Parameters
    ----------
    baseline_std : float
        Standard deviation of outcome metric
    correlation : float
        Expected correlation between outcome and covariate
    mde : float
        Minimum detectable effect (absolute)
    alpha : float
        Significance level
    power : float
        Statistical power

    Returns
    -------
    tuple
        (n_raw, n_cuped, reduction_pct)
        - n_raw: Sample size needed without CUPED
        - n_cuped: Sample size needed with CUPED
        - reduction_pct: Percentage reduction

    Example
    -------
    >>> n_raw, n_cuped, reduction = estimate_required_sample_size_with_cuped(
    ...     baseline_std=80,
    ...     correlation=0.7,
    ...     mde=15
    ... )
    >>> print(f"Without CUPED: {n_raw:,} per group")
    >>> print(f"With CUPED: {n_cuped:,} per group ({reduction:.1f}% reduction)")
    """
    from ab_testing.core import power

    # Sample size without CUPED
    n_raw = power.required_samples_continuous(
        baseline_mean=0,  # Doesn't matter for sample size
        baseline_std=baseline_std,
        mde=mde,
        alpha=alpha,
        power=power
    )

    # Variance reduction from CUPED
    var_reduction = correlation**2

    # Adjusted std after CUPED
    adjusted_std = baseline_std * np.sqrt(1 - var_reduction)

    # Sample size with CUPED
    n_cuped = power.required_samples_continuous(
        baseline_mean=0,
        baseline_std=adjusted_std,
        mde=mde,
        alpha=alpha,
        power=power
    )

    reduction_pct = (1 - n_cuped / n_raw) * 100

    return n_raw, n_cuped, reduction_pct


def variance_reduction(
    y: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Calculate variance reduction achieved by CUPED.

    Parameters
    ----------
    y : np.ndarray
        Outcome metric
    x : np.ndarray
        Pre-experiment covariate

    Returns
    -------
    float
        Variance reduction percentage (0-1)

    Example
    -------
    >>> y = np.random.normal(100, 20, 1000)
    >>> x = y * 0.7 + np.random.normal(0, 15, 1000)
    >>> vr = variance_reduction(y, x)
    >>> print(f"Variance reduction: {vr*100:.1f}%")
    """
    y_adj = cuped_adjustment(y, x)
    var_red = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    return float(var_red)


def power_gain_cuped(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,
    x_treatment: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate statistical power gain from using CUPED.

    Parameters
    ----------
    y_control, y_treatment : np.ndarray
        Outcome metrics
    x_control, x_treatment : np.ndarray
        Pre-experiment covariates

    Returns
    -------
    dict
        Dictionary with keys:
        - var_reduction: Variance reduction percentage
        - se_reduction: Standard error reduction percentage
        - power_multiplier: Power gain factor (1/(1-se_reduction))
        - equivalent_n: Equivalent sample size

    Example
    -------
    >>> y_c = np.random.normal(100, 20, 500)
    >>> y_t = np.random.normal(115, 20, 500)
    >>> x_c = y_c * 0.7 + np.random.normal(0, 15, 500)
    >>> x_t = y_t * 0.7 + np.random.normal(0, 15, 500)
    >>> pg = power_gain_cuped(y_c, y_t, x_c, x_t)
    >>> print(f"Power multiplier: {pg['power_multiplier']:.2f}x")
    """
    result = cuped_ab_test(y_control, y_treatment, x_control, x_treatment)

    # Power multiplier
    se_red = result['se_reduction']
    if se_red > 0 and se_red < 1:
        power_multiplier = 1 / (1 - se_red)
    else:
        power_multiplier = 1.0

    # Equivalent sample size
    n_actual = len(y_control) + len(y_treatment)
    equivalent_n = n_actual * power_multiplier

    return {
        'var_reduction': result['var_reduction'],
        'se_reduction': result['se_reduction'],
        'power_multiplier': float(power_multiplier),
        'equivalent_n': float(equivalent_n),
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("CUPED Variance Reduction Demo")
    print("=" * 80)

    # Simulate data with correlation
    np.random.seed(42)
    n_control = 500
    n_treatment = 500

    # Create correlated pre-experiment data
    x_control = np.random.normal(100, 20, n_control)
    x_treatment = np.random.normal(100, 20, n_treatment)

    # Outcome correlated with pre-experiment data
    y_control = x_control * 0.7 + np.random.normal(0, 15, n_control)
    y_treatment = x_treatment * 0.7 + np.random.normal(15, 15, n_treatment)  # +15 treatment effect

    # Run CUPED analysis
    result = cuped_ab_test(y_control, y_treatment, x_control, x_treatment)

    print("\n📊 CUPED ANALYSIS RESULTS")
    print("-" * 80)
    print(f"Correlation (Y ↔ X): {result['correlation']:.4f}")
    print(f"θ (adjustment coefficient): {result['theta']:.4f}")
    print(f"Variance reduction: {result['var_reduction']*100:.1f}%")
    print(f"SE reduction: {result['se_reduction']*100:.1f}%")

    print("\n┌─────────────┬──────────────────┬──────────────────┐")
    print("│             │ Raw              │ CUPED            │")
    print("├─────────────┼──────────────────┼──────────────────┤")
    print(f"│ Effect      │ {result['effect_raw']:>16.2f} │ {result['effect_adjusted']:>16.2f} │")
    print(f"│ SE          │ {result['se_raw']:>16.2f} │ {result['se_adjusted']:>16.2f} │")
    print(f"│ CI Width    │ {result['ci_raw'][1]-result['ci_raw'][0]:>16.2f} │ {result['ci_adjusted'][1]-result['ci_adjusted'][0]:>16.2f} │")
    print(f"│ P-value     │ {result['p_value_raw']:>16.6f} │ {result['p_value_adjusted']:>16.6f} │")
    print("└─────────────┴──────────────────┴──────────────────┘")

    print(f"\n💡 PRACTICAL IMPACT:")
    print(f"   • Could run with {result['sample_size_reduction']*100:.0f}% fewer users for same power")
    print(f"   • Or achieve {(1/(1-result['se_reduction'])-1)*100:.0f}% more power with same sample size")

    # Sample size estimation
    print("\n📊 SAMPLE SIZE ESTIMATION")
    print("-" * 80)
    n_raw, n_cuped, reduction = estimate_required_sample_size_with_cuped(
        baseline_std=80,
        correlation=0.7,
        mde=15
    )
    print(f"Without CUPED: {n_raw:,} users per group")
    print(f"With CUPED: {n_cuped:,} users per group")
    print(f"Reduction: {reduction:.1f}%")
