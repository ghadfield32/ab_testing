"""
Frequentist Statistical Tests for A/B Testing
==============================================

Functions for z-tests (proportions), t-tests (means), non-parametric tests
(Mann-Whitney U), and bootstrap confidence intervals.

Example Usage:
--------------
>>> from ab_testing.core import frequentist
>>>
>>> # Z-test for conversion rates
>>> result = frequentist.z_test_proportions(
...     x_control=50, n_control=500,
...     x_treatment=60, n_treatment=500
... )
>>> print(f"P-value: {result['p_value']:.4f}")
>>>
>>> # Welch's t-test for revenue
>>> import numpy as np
>>> control_rev = np.random.normal(100, 20, 500)
>>> treatment_rev = np.random.normal(115, 22, 500)
>>> result = frequentist.welch_ttest(control_rev, treatment_rev)
>>> print(f"Effect: ${result['difference']:.2f}, p={result['p_value']:.4f}")
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def z_test_proportions(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Dict[str, float]:
    """
    Two-sample Z-test for proportions (e.g., conversion rates).

    Uses pooled standard error for the test statistic and non-pooled
    standard error for the confidence interval.

    Parameters
    ----------
    x_control : int
        Number of successes in control group
    n_control : int
        Total sample size in control group
    x_treatment : int
        Number of successes in treatment group
    n_treatment : int
        Total sample size in treatment group
    alpha : float, default=0.05
        Significance level
    two_sided : bool, default=True
        Whether to use two-sided test

    Returns
    -------
    dict
        Dictionary with keys:
        - p_control: Control proportion
        - p_treatment: Treatment proportion
        - absolute_lift: Treatment - Control (in pp)
        - relative_lift: (Treatment - Control) / Control
        - z_statistic: Z-test statistic
        - p_value: P-value
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - significant: Whether result is significant at alpha

    Notes
    -----
    - Uses pooled SE for hypothesis test: SE = √[p̄(1-p̄)(1/n₁ + 1/n₂)]
    - Uses non-pooled SE for CI: SE = √[p₁(1-p₁)/n₁ + p₂(1-p₂)/n₂]
    - This is standard practice in hypothesis testing

    Example
    -------
    >>> # 5% vs 5.5% conversion rate
    >>> result = z_test_proportions(50, 1000, 55, 1000)
    >>> print(f"Absolute lift: {result['absolute_lift']*100:.2f}pp")
    >>> print(f"Relative lift: {result['relative_lift']*100:.1f}%")
    >>> print(f"Significant: {result['significant']}")
    """
    if x_control < 0 or x_treatment < 0:
        raise ValueError("x_control and x_treatment must be non-negative")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be positive")
    if x_control > n_control or x_treatment > n_treatment:
        raise ValueError("Number of successes cannot exceed sample size")

    # Calculate proportions
    p_control = x_control / n_control
    p_treatment = x_treatment / n_treatment

    # Pooled proportion (for test statistic)
    p_pooled = (x_control + x_treatment) / (n_control + n_treatment)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    # Z-statistic
    z_stat = (p_treatment - p_control) / se_pooled

    # P-value
    if two_sided:
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1 - stats.norm.cdf(z_stat)

    # Confidence interval (using non-pooled SE)
    se_diff = np.sqrt(p_control*(1-p_control)/n_control +
                      p_treatment*(1-p_treatment)/n_treatment)
    z_critical = stats.norm.ppf(1 - alpha/2) if two_sided else stats.norm.ppf(1 - alpha)
    ci_lower = (p_treatment - p_control) - z_critical * se_diff
    ci_upper = (p_treatment - p_control) + z_critical * se_diff

    return {
        'p_control': p_control,
        'p_treatment': p_treatment,
        'absolute_lift': p_treatment - p_control,
        'relative_lift': (p_treatment - p_control) / p_control if p_control > 0 else np.nan,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha,
    }


def welch_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Dict[str, float]:
    """
    Welch's t-test for comparing means of continuous metrics.

    Does NOT assume equal variances (unlike Student's t-test).
    This is more robust and should be the default choice.

    Parameters
    ----------
    control : np.ndarray
        Observations from control group
    treatment : np.ndarray
        Observations from treatment group
    alpha : float, default=0.05
        Significance level
    two_sided : bool, default=True
        Whether to use two-sided test

    Returns
    -------
    dict
        Dictionary with keys:
        - mean_control: Control group mean
        - mean_treatment: Treatment group mean
        - difference: Treatment - Control
        - relative_lift: (Treatment - Control) / Control
        - se_diff: Standard error of the difference
        - t_statistic: T-test statistic
        - p_value: P-value
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - cohens_d: Effect size (Cohen's d)
        - significant: Whether result is significant at alpha

    Notes
    -----
    - Uses Welch-Satterthwaite degrees of freedom
    - Assumes independent samples
    - Does NOT assume equal variances or normality (robust to violations)

    Example
    -------
    >>> import numpy as np
    >>> control_rev = np.random.normal(100, 20, 500)
    >>> treatment_rev = np.random.normal(115, 22, 500)
    >>> result = welch_ttest(control_rev, treatment_rev)
    >>> print(f"Effect: ${result['difference']:.2f}")
    >>> print(f"Cohen's d: {result['cohens_d']:.4f}")
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Each group must have at least 2 observations")

    # Calculate means and variances
    mean_c = control.mean()
    mean_t = treatment.mean()
    var_c = control.var(ddof=1)
    var_t = treatment.var(ddof=1)
    n_c = len(control)
    n_t = len(treatment)

    # Welch's t-test (using scipy for correct df calculation)
    alternative = 'two-sided' if two_sided else 'greater'
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False, alternative=alternative)

    # Standard error of difference
    se = np.sqrt(var_c/n_c + var_t/n_t)

    # Confidence interval
    # Calculate Welch-Satterthwaite degrees of freedom
    df = (var_c/n_c + var_t/n_t)**2 / (
        (var_c/n_c)**2 / (n_c - 1) + (var_t/n_t)**2 / (n_t - 1)
    )
    t_critical = stats.t.ppf(1 - alpha/2, df) if two_sided else stats.t.ppf(1 - alpha, df)
    difference = mean_t - mean_c
    ci_lower = difference - t_critical * se
    ci_upper = difference + t_critical * se

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
    cohens_d = difference / pooled_std

    return {
        'mean_control': float(mean_c),
        'mean_treatment': float(mean_t),
        'difference': float(difference),
        'relative_lift': float(difference / mean_c if mean_c != 0 else np.nan),
        'se_diff': float(se),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < alpha),
    }


def mann_whitney_u(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Compares distributions, not just means. Robust to outliers and
    does not assume normality.

    Parameters
    ----------
    control : np.ndarray
        Observations from control group
    treatment : np.ndarray
        Observations from treatment group
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with keys:
        - median_control: Control group median
        - median_treatment: Treatment group median
        - u_statistic: Mann-Whitney U statistic
        - p_value: P-value (two-sided)
        - rank_biserial: Rank-biserial correlation (effect size)
        - significant: Whether result is significant at alpha

    Notes
    -----
    - Rank-biserial correlation: r = 1 - (2U)/(n₁×n₂)
      Interpretation: |r| > 0.5 (large), |r| > 0.3 (medium), |r| > 0.1 (small)
    - Tests whether distributions differ (not just medians)
    - Equivalent to Wilcoxon rank-sum test

    Example
    -------
    >>> # Heavily skewed revenue data
    >>> control = np.random.exponential(100, 500)
    >>> treatment = np.random.exponential(115, 500)
    >>> result = mann_whitney_u(control, treatment)
    >>> print(f"Median difference: ${result['median_treatment'] - result['median_control']:.2f}")
    >>> print(f"Effect size (r): {result['rank_biserial']:.4f}")
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Each group must have at least 2 observations")

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(control, treatment, alternative='two-sided')

    # Rank-biserial correlation (effect size)
    n_c = len(control)
    n_t = len(treatment)
    rank_biserial = 1 - (2 * u_stat) / (n_c * n_t)

    return {
        'median_control': np.median(control),
        'median_treatment': np.median(treatment),
        'u_statistic': u_stat,
        'p_value': p_value,
        'rank_biserial': rank_biserial,
        'significant': p_value < alpha,
    }


def bootstrap_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    statistic: str = 'mean',
    n_iterations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for difference between groups.

    Non-parametric method that works for any statistic (mean, median, etc.)
    and does not assume normality.

    Parameters
    ----------
    control : np.ndarray
        Observations from control group
    treatment : np.ndarray
        Observations from treatment group
    statistic : str, default='mean'
        Statistic to compute: 'mean' or 'median'
    n_iterations : int, default=10000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level (for CI)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - point_estimate: Observed difference (treatment - control)
        - bootstrap_se: Bootstrap standard error
        - ci_lower: Lower bound of CI (percentile method)
        - ci_upper: Upper bound of CI (percentile method)
        - significant: Whether CI excludes zero

    Notes
    -----
    - Uses percentile method for CI (not BCa)
    - Good for skewed distributions (like revenue)
    - Can handle any statistic, not just mean
    - Computationally intensive for large datasets

    Example
    -------
    >>> # Revenue with outliers
    >>> control = np.random.lognormal(4, 1, 500)
    >>> treatment = np.random.lognormal(4.2, 1, 500)
    >>> result = bootstrap_ci(control, treatment, statistic='median')
    >>> print(f"Median difference: {result['point_estimate']:.2f}")
    >>> print(f"95% CI: ({result['ci_lower']:.2f}, {result['ci_upper']:.2f})")
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Each group must have at least 2 observations")
    if statistic not in ['mean', 'median']:
        raise ValueError("statistic must be 'mean' or 'median'")
    if n_iterations < 100:
        raise ValueError("n_iterations must be at least 100")

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Choose statistic function
    stat_func = np.mean if statistic == 'mean' else np.median

    # Point estimate
    point_estimate = stat_func(treatment) - stat_func(control)

    # Bootstrap
    boot_diffs = []
    for _ in range(n_iterations):
        boot_c = np.random.choice(control, len(control), replace=True)
        boot_t = np.random.choice(treatment, len(treatment), replace=True)
        boot_diffs.append(stat_func(boot_t) - stat_func(boot_c))
    boot_diffs = np.array(boot_diffs)

    # Percentile method CI
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return {
        'point_estimate': point_estimate,
        'bootstrap_se': boot_diffs.std(),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': not (ci_lower <= 0 <= ci_upper),
    }


def interpret_effect_size(
    effect_size: float,
    metric_type: str = 'd',
) -> str:
    """
    Interpret Cohen's d or rank-biserial correlation.

    Parameters
    ----------
    effect_size : float
        Effect size (Cohen's d or rank-biserial r)
    metric_type : str, default='d'
        Either 'd' (Cohen's d) or 'r' (rank-biserial)

    Returns
    -------
    str
        Interpretation: "Negligible", "Small", "Medium", or "Large"

    Notes
    -----
    Cohen's d thresholds: 0.2 (small), 0.5 (medium), 0.8 (large)
    Rank-biserial thresholds: 0.1 (small), 0.3 (medium), 0.5 (large)

    Example
    -------
    >>> interpretation = interpret_effect_size(0.65, metric_type='d')
    >>> print(interpretation)
    Medium
    """
    abs_effect = abs(effect_size)

    if metric_type == 'd':
        # Cohen's d thresholds
        if abs_effect > 0.8:
            return "Large"
        elif abs_effect > 0.5:
            return "Medium"
        elif abs_effect > 0.2:
            return "Small"
        else:
            return "Negligible"
    elif metric_type == 'r':
        # Rank-biserial thresholds
        if abs_effect > 0.5:
            return "Large"
        elif abs_effect > 0.3:
            return "Medium"
        elif abs_effect > 0.1:
            return "Small"
        else:
            return "Negligible"
    else:
        raise ValueError("metric_type must be 'd' or 'r'")


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Frequentist Tests Demo")
    print("=" * 80)

    # Binary metric example
    print("\n📊 Z-TEST (Conversion Rate)")
    print("-" * 80)
    result = z_test_proportions(x_control=50, n_control=500, x_treatment=60, n_treatment=500)
    print(f"Control: {result['p_control']:.2%}")
    print(f"Treatment: {result['p_treatment']:.2%}")
    print(f"Absolute lift: {result['absolute_lift']*100:.2f}pp")
    print(f"Relative lift: {result['relative_lift']*100:.1f}%")
    print(f"Z-statistic: {result['z_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"95% CI: ({result['ci_lower']*100:.2f}pp, {result['ci_upper']*100:.2f}pp)")
    print(f"Significant: {'✅' if result['significant'] else '❌'}")

    # Continuous metric example
    print("\n📊 WELCH'S T-TEST (Revenue)")
    print("-" * 80)
    np.random.seed(42)
    control_rev = np.random.normal(100, 20, 500)
    treatment_rev = np.random.normal(115, 22, 500)
    result = welch_ttest(control_rev, treatment_rev)
    print(f"Control: ${result['mean_control']:.2f}")
    print(f"Treatment: ${result['mean_treatment']:.2f}")
    print(f"Difference: ${result['difference']:.2f} ({result['relative_lift']*100:.1f}%)")
    print(f"T-statistic: {result['t_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"95% CI: (${result['ci_lower']:.2f}, ${result['ci_upper']:.2f})")
    print(f"Cohen's d: {result['cohens_d']:.4f} ({interpret_effect_size(result['cohens_d'], 'd')})")
    print(f"Significant: {'✅' if result['significant'] else '❌'}")

    # Non-parametric example
    print("\n📊 MANN-WHITNEY U (Skewed Revenue)")
    print("-" * 80)
    control_skewed = np.random.exponential(100, 500)
    treatment_skewed = np.random.exponential(115, 500)
    result = mann_whitney_u(control_skewed, treatment_skewed)
    print(f"Control median: ${result['median_control']:.2f}")
    print(f"Treatment median: ${result['median_treatment']:.2f}")
    print(f"U-statistic: {result['u_statistic']:.0f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Rank-biserial r: {result['rank_biserial']:.4f} ({interpret_effect_size(result['rank_biserial'], 'r')})")
    print(f"Significant: {'✅' if result['significant'] else '❌'}")

    # Bootstrap example
    print("\n📊 BOOTSTRAP CI (Median Difference)")
    print("-" * 80)
    result = bootstrap_ci(control_skewed, treatment_skewed, statistic='median', n_iterations=10000, random_state=42)
    print(f"Point estimate: ${result['point_estimate']:.2f}")
    print(f"Bootstrap SE: ${result['bootstrap_se']:.2f}")
    print(f"95% CI: (${result['ci_lower']:.2f}, ${result['ci_upper']:.2f})")
    print(f"Significant: {'✅' if result['significant'] else '❌'}")
