"""
Ratio Metrics and Delta Method
===============================

Proper statistical inference for ratio metrics like:
- CTR = Clicks / Impressions
- ARPU = Revenue / Users
- Conversion Rate = Conversions / Sessions
- Pages per Session = Page Views / Sessions

Problem: Ratio of means ≠ Mean of ratios
Solution: Delta method for variance estimation

Reference:
----------
- Deng et al. (2018): "Applying the Delta Method in Metric Analytics"
- Kohavi et al. (2020): "Trustworthy Online Controlled Experiments" Chapter 16

Example Usage:
--------------
>>> from ab_testing.advanced import ratio_metrics
>>> import numpy as np
>>>
>>> # Revenue per user
>>> revenue_control = np.random.gamma(2, 50, 1000)
>>> revenue_treatment = np.random.gamma(2, 60, 1000)
>>>
>>> result = ratio_metrics.ratio_metric_test(
...     numerator_control=revenue_control,
...     denominator_control=np.ones(1000),  # Per user
...     numerator_treatment=revenue_treatment,
...     denominator_treatment=np.ones(1000)
... )
>>> print(f"ARPU lift: ${result['ratio_diff']:.2f}")
>>> print(f"P-value: {result['p_value']:.4f}")
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats


def delta_method_variance(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float:
    """
    Calculate variance of ratio using delta method.

    Formula for Var(Y/X):
    Var(Y/X) ≈ (1/μ_X²)[Var(Y) - 2(μ_Y/μ_X)Cov(Y,X) + (μ_Y²/μ_X²)Var(X)]

    Parameters
    ----------
    numerator : np.ndarray
        Numerator values (e.g., revenue)
    denominator : np.ndarray
        Denominator values (e.g., users)

    Returns
    -------
    float
        Variance of the ratio

    Notes
    -----
    - Valid for large samples (CLT)
    - Assumes numerator and denominator are from same units
    - More accurate than naive approach

    Example
    -------
    >>> revenue = np.random.gamma(2, 50, 1000)
    >>> users = np.ones(1000)
    >>> var_ratio = delta_method_variance(revenue, users)
    >>> se_ratio = np.sqrt(var_ratio / len(revenue))
    """
    n = len(numerator)

    mu_num = numerator.mean()
    mu_denom = denominator.mean()
    var_num = numerator.var(ddof=1)
    var_denom = denominator.var(ddof=1)
    cov_num_denom = np.cov(numerator, denominator, ddof=1)[0, 1]

    # Delta method variance
    var_ratio = (
        (1 / mu_denom**2) * var_num
        - (2 * mu_num / mu_denom**3) * cov_num_denom
        + (mu_num**2 / mu_denom**4) * var_denom
    )

    return var_ratio / n


def ratio_metric_test(
    numerator_control: np.ndarray,
    denominator_control: np.ndarray,
    numerator_treatment: np.ndarray,
    denominator_treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Two-sample test for ratio metrics using delta method.

    Tests H0: ratio_treatment = ratio_control

    Parameters
    ----------
    numerator_control : np.ndarray
        Numerator for control (e.g., revenue)
    denominator_control : np.ndarray
        Denominator for control (e.g., users)
    numerator_treatment : np.ndarray
        Numerator for treatment
    denominator_treatment : np.ndarray
        Denominator for treatment
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with:
        - ratio_control: Mean ratio in control
        - ratio_treatment: Mean ratio in treatment
        - ratio_diff: Difference in ratios
        - relative_lift: Relative lift percentage
        - se_control: SE of control ratio
        - se_treatment: SE of treatment ratio
        - se_diff: SE of difference
        - z_statistic: Test statistic
        - p_value: Two-sided p-value
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - significant: Whether result is significant

    Example
    -------
    >>> # CTR = Clicks / Impressions
    >>> clicks_c = np.random.binomial(100, 0.05, 1000)
    >>> impressions_c = np.full(1000, 100)
    >>> clicks_t = np.random.binomial(100, 0.06, 1000)
    >>> impressions_t = np.full(1000, 100)
    >>> result = ratio_metric_test(clicks_c, impressions_c, clicks_t, impressions_t)
    >>> print(f"CTR lift: {result['relative_lift']*100:.1f}%")
    """
    if len(numerator_control) != len(denominator_control):
        raise ValueError("Control numerator and denominator must have same length")
    if len(numerator_treatment) != len(denominator_treatment):
        raise ValueError("Treatment numerator and denominator must have same length")
    if len(numerator_control) < 2 or len(numerator_treatment) < 2:
        raise ValueError("Need at least 2 observations per group")

    # Calculate ratios
    ratio_control = numerator_control.sum() / denominator_control.sum()
    ratio_treatment = numerator_treatment.sum() / denominator_treatment.sum()

    # Delta method variances
    var_control = delta_method_variance(numerator_control, denominator_control)
    var_treatment = delta_method_variance(numerator_treatment, denominator_treatment)

    # Standard errors
    se_control = np.sqrt(var_control)
    se_treatment = np.sqrt(var_treatment)
    se_diff = np.sqrt(var_control + var_treatment)

    # Test statistic
    ratio_diff = ratio_treatment - ratio_control
    z_stat = ratio_diff / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval
    z_critical = stats.norm.ppf(1 - alpha/2)
    ci_lower = ratio_diff - z_critical * se_diff
    ci_upper = ratio_diff + z_critical * se_diff

    # Relative lift
    relative_lift = (ratio_treatment / ratio_control - 1) if ratio_control > 0 else np.nan

    return {
        'ratio_control': ratio_control,
        'ratio_treatment': ratio_treatment,
        'ratio_diff': ratio_diff,
        'relative_lift': relative_lift,
        'se_control': se_control,
        'se_treatment': se_treatment,
        'se_diff': se_diff,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha,
    }


def ctr_test(
    clicks_control: np.ndarray,
    impressions_control: np.ndarray,
    clicks_treatment: np.ndarray,
    impressions_treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Test for Click-Through Rate (CTR) difference.

    Convenience wrapper for ratio_metric_test specialized for CTR.

    Parameters
    ----------
    clicks_control : np.ndarray
        Number of clicks per unit (user/session) in control
    impressions_control : np.ndarray
        Number of impressions per unit in control
    clicks_treatment : np.ndarray
        Clicks in treatment
    impressions_treatment : np.ndarray
        Impressions in treatment
    alpha : float
        Significance level

    Returns
    -------
    dict
        Same as ratio_metric_test but with 'ctr_*' keys

    Example
    -------
    >>> clicks_c = np.random.binomial(100, 0.05, 1000)
    >>> impr_c = np.full(1000, 100)
    >>> clicks_t = np.random.binomial(100, 0.06, 1000)
    >>> impr_t = np.full(1000, 100)
    >>> result = ctr_test(clicks_c, impr_c, clicks_t, impr_t)
    >>> print(f"Control CTR: {result['ctr_control']:.2%}")
    >>> print(f"Treatment CTR: {result['ctr_treatment']:.2%}")
    """
    result = ratio_metric_test(
        clicks_control, impressions_control,
        clicks_treatment, impressions_treatment,
        alpha=alpha
    )

    # Rename keys for clarity
    return {
        'ctr_control': result['ratio_control'],
        'ctr_treatment': result['ratio_treatment'],
        'ctr_diff': result['ratio_diff'],
        'relative_lift': result['relative_lift'],
        'se_diff': result['se_diff'],
        'z_statistic': result['z_statistic'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'significant': result['significant'],
    }


def arpu_test(
    revenue_control: np.ndarray,
    revenue_treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Test for Average Revenue Per User (ARPU) difference.

    Assumes each observation is one user's revenue.

    Parameters
    ----------
    revenue_control : np.ndarray
        Revenue per user in control
    revenue_treatment : np.ndarray
        Revenue per user in treatment
    alpha : float
        Significance level

    Returns
    -------
    dict
        Same as ratio_metric_test but with 'arpu_*' keys

    Example
    -------
    >>> rev_c = np.random.gamma(2, 50, 1000)  # Skewed revenue
    >>> rev_t = np.random.gamma(2, 60, 1000)
    >>> result = arpu_test(rev_c, rev_t)
    >>> print(f"ARPU lift: ${result['arpu_diff']:.2f} ({result['relative_lift']*100:.1f}%)")
    """
    # ARPU = Total Revenue / Users
    # Each observation is already per-user, so denominator = 1
    result = ratio_metric_test(
        revenue_control,
        np.ones(len(revenue_control)),
        revenue_treatment,
        np.ones(len(revenue_treatment)),
        alpha=alpha
    )

    return {
        'arpu_control': result['ratio_control'],
        'arpu_treatment': result['ratio_treatment'],
        'arpu_diff': result['ratio_diff'],
        'relative_lift': result['relative_lift'],
        'se_diff': result['se_diff'],
        'z_statistic': result['z_statistic'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'significant': result['significant'],
    }


def compare_delta_vs_naive(
    numerator_control: np.ndarray,
    denominator_control: np.ndarray,
    numerator_treatment: np.ndarray,
    denominator_treatment: np.ndarray,
) -> Dict[str, float]:
    """
    Compare delta method vs naive ratio approach.

    Shows why delta method is necessary for proper inference.

    Parameters
    ----------
    numerator_control, denominator_control : np.ndarray
        Control group data
    numerator_treatment, denominator_treatment : np.ndarray
        Treatment group data

    Returns
    -------
    dict
        Comparison of delta method vs naive approach

    Example
    -------
    >>> rev_c = np.random.gamma(2, 50, 1000)
    >>> rev_t = np.random.gamma(2, 60, 1000)
    >>> comparison = compare_delta_vs_naive(
    ...     rev_c, np.ones(1000), rev_t, np.ones(1000)
    ... )
    >>> print(f"Delta SE: {comparison['se_delta']:.4f}")
    >>> print(f"Naive SE: {comparison['se_naive']:.4f}")
    """
    # Delta method (correct)
    delta_result = ratio_metric_test(
        numerator_control, denominator_control,
        numerator_treatment, denominator_treatment
    )

    # Naive approach (incorrect): Compute ratios first, then treat as regular metric
    ratios_control = numerator_control / denominator_control
    ratios_treatment = numerator_treatment / denominator_treatment

    # Filter out infinite/nan values for naive approach
    ratios_control = ratios_control[np.isfinite(ratios_control)]
    ratios_treatment = ratios_treatment[np.isfinite(ratios_treatment)]

    naive_se = np.sqrt(
        ratios_control.var(ddof=1) / len(ratios_control) +
        ratios_treatment.var(ddof=1) / len(ratios_treatment)
    )

    naive_diff = ratios_treatment.mean() - ratios_control.mean()
    naive_z = naive_diff / naive_se
    naive_p = 2 * (1 - stats.norm.cdf(abs(naive_z)))

    return {
        'se_delta': delta_result['se_diff'],
        'se_naive': naive_se,
        'se_ratio': delta_result['se_diff'] / naive_se,
        'p_value_delta': delta_result['p_value'],
        'p_value_naive': naive_p,
        'ratio_diff': delta_result['ratio_diff'],
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Ratio Metrics with Delta Method Demo")
    print("=" * 80)

    np.random.seed(42)

    # Example 1: CTR (Click-Through Rate)
    print("\n📊 EXAMPLE 1: CLICK-THROUGH RATE (CTR)")
    print("-" * 80)

    n_users = 1000
    clicks_control = np.random.binomial(100, 0.05, n_users)
    impressions_control = np.full(n_users, 100)
    clicks_treatment = np.random.binomial(100, 0.06, n_users)
    impressions_treatment = np.full(n_users, 100)

    ctr_result = ctr_test(clicks_control, impressions_control,
                          clicks_treatment, impressions_treatment)

    print(f"Control CTR: {ctr_result['ctr_control']:.4f} ({ctr_result['ctr_control']*100:.2f}%)")
    print(f"Treatment CTR: {ctr_result['ctr_treatment']:.4f} ({ctr_result['ctr_treatment']*100:.2f}%)")
    print(f"Absolute lift: {ctr_result['ctr_diff']*100:.3f} percentage points")
    print(f"Relative lift: {ctr_result['relative_lift']*100:.1f}%")
    print(f"Z-statistic: {ctr_result['z_statistic']:.4f}")
    print(f"P-value: {ctr_result['p_value']:.4f}")
    print(f"95% CI: ({ctr_result['ci_lower']*100:.3f}pp, {ctr_result['ci_upper']*100:.3f}pp)")
    print(f"Significant: {'✅ Yes' if ctr_result['significant'] else '❌ No'}")

    # Example 2: ARPU (Average Revenue Per User)
    print("\n📊 EXAMPLE 2: AVERAGE REVENUE PER USER (ARPU)")
    print("-" * 80)

    revenue_control = np.random.gamma(2, 50, 1000)  # Skewed revenue distribution
    revenue_treatment = np.random.gamma(2, 60, 1000)

    arpu_result = arpu_test(revenue_control, revenue_treatment)

    print(f"Control ARPU: ${arpu_result['arpu_control']:.2f}")
    print(f"Treatment ARPU: ${arpu_result['arpu_treatment']:.2f}")
    print(f"Absolute lift: ${arpu_result['arpu_diff']:.2f}")
    print(f"Relative lift: {arpu_result['relative_lift']*100:.1f}%")
    print(f"Z-statistic: {arpu_result['z_statistic']:.4f}")
    print(f"P-value: {arpu_result['p_value']:.4f}")
    print(f"95% CI: (${arpu_result['ci_lower']:.2f}, ${arpu_result['ci_upper']:.2f})")
    print(f"Significant: {'✅ Yes' if arpu_result['significant'] else '❌ No'}")

    # Example 3: Delta vs Naive comparison
    print("\n📊 WHY DELTA METHOD MATTERS")
    print("-" * 80)

    comparison = compare_delta_vs_naive(
        revenue_control, np.ones(len(revenue_control)),
        revenue_treatment, np.ones(len(revenue_treatment))
    )

    print(f"Delta method SE: {comparison['se_delta']:.4f}")
    print(f"Naive approach SE: {comparison['se_naive']:.4f}")
    print(f"SE ratio (delta/naive): {comparison['se_ratio']:.4f}")
    print()
    print(f"Delta method p-value: {comparison['p_value_delta']:.6f}")
    print(f"Naive approach p-value: {comparison['p_value_naive']:.6f}")
    print()
    print("💡 Delta method provides correct variance estimation for ratio metrics.")
    print("   Using naive approach can lead to incorrect inference!")
