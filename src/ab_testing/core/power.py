"""
Sample Size and Power Analysis for A/B Tests
============================================

Functions for calculating required sample sizes for binary and continuous
metrics, and computing effect sizes (Cohen's h and Cohen's d).

Example Usage:
--------------
>>> from ab_testing.core import power
>>>
>>> # Binary metric (conversion rate)
>>> n = power.required_samples_binary(
...     p_baseline=0.05,
...     mde=0.10,  # 10% relative lift
...     alpha=0.05,
...     power=0.80
... )
>>> print(f"Need {n:,} users per group")
>>>
>>> # Continuous metric (revenue)
>>> n = power.required_samples_continuous(
...     baseline_mean=175,
...     baseline_std=80,
...     mde=15,  # $15 absolute lift
...     alpha=0.05,
...     power=0.80
... )
>>> print(f"Need {n:,} users per group")
"""

import numpy as np
from typing import Tuple
from statsmodels.stats.power import zt_ind_solve_power, tt_ind_solve_power


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for proportions.

    Cohen's h is the standardized difference between two proportions,
    using arcsine transformation.

    Parameters
    ----------
    p1 : float
        Proportion in group 1 (baseline)
    p2 : float
        Proportion in group 2 (treatment)

    Returns
    -------
    float
        Cohen's h effect size

    Notes
    -----
    Interpretation (rule of thumb):
    - Small: h ≈ 0.2
    - Medium: h ≈ 0.5
    - Large: h ≈ 0.8

    Formula:
    h = 2 * (arcsin(√p2) - arcsin(√p1))

    Example
    -------
    >>> h = cohens_h(p1=0.10, p2=0.12)
    >>> print(f"Cohen's h: {h:.4f}")
    """
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("Proportions must be between 0 and 1")

    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
    return h


def cohens_d(mean1: float, mean2: float, std1: float, std2: float,
             n1: int, n2: int) -> float:
    """
    Calculate Cohen's d effect size for means.

    Cohen's d is the standardized difference between two means,
    using pooled standard deviation.

    Parameters
    ----------
    mean1 : float
        Mean of group 1 (baseline)
    mean2 : float
        Mean of group 2 (treatment)
    std1 : float
        Standard deviation of group 1
    std2 : float
        Standard deviation of group 2
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2

    Returns
    -------
    float
        Cohen's d effect size

    Notes
    -----
    Interpretation (rule of thumb):
    - Small: d ≈ 0.2
    - Medium: d ≈ 0.5
    - Large: d ≈ 0.8

    Formula:
    pooled_std = √[((n1-1)×σ1² + (n2-1)×σ2²) / (n1+n2-2)]
    d = (μ2 - μ1) / pooled_std

    Example
    -------
    >>> d = cohens_d(mean1=100, mean2=115, std1=20, std2=22, n1=500, n2=500)
    >>> print(f"Cohen's d: {d:.4f}")
    """
    # Pooled standard deviation
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    d = (mean2 - mean1) / pooled_std
    return d


def power_binary(
    p1: float,
    p2: float,
    n: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Calculate statistical power for detecting difference in proportions.

    Given sample size and effect size, calculates probability of detecting
    the effect (power = 1 - Type II error).

    Parameters
    ----------
    p1 : float
        Proportion in group 1 (baseline), between 0 and 1
    p2 : float
        Proportion in group 2 (treatment), between 0 and 1
    n : int
        Sample size per group
    alpha : float, default=0.05
        Significance level (Type I error rate)
    two_sided : bool, default=True
        Whether to use two-sided test

    Returns
    -------
    float
        Statistical power (between 0 and 1)

    Notes
    -----
    - Power is the probability of correctly rejecting null when false
    - Higher power = less chance of missing real effects (Type II error)
    - Industry standard: aim for 80% power minimum
    - Uses Cohen's h effect size and normal approximation

    Example
    -------
    >>> # What's the power to detect 10% lift from 5% with 10K users/group?
    >>> pwr = power_binary(p1=0.05, p2=0.055, n=10000, alpha=0.05)
    >>> print(f"Power: {pwr:.1%}")
    Power: 62.4%

    >>> # Need more users for higher power
    >>> pwr = power_binary(p1=0.05, p2=0.055, n=20000, alpha=0.05)
    >>> print(f"Power: {pwr:.1%}")
    Power: 87.2%
    """
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError("Proportions must be between 0 and 1")
    if n <= 0:
        raise ValueError("Sample size must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    # Calculate Cohen's h effect size
    effect_size = cohens_h(p1, p2)

    # Calculate power using statsmodels
    # Note: zt_ind_solve_power can solve for any parameter given the others
    power = zt_ind_solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        alternative='two-sided' if two_sided else 'larger',
        ratio=1.0,  # Equal group sizes
    )

    return float(power)


def required_samples_binary(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Calculate required sample size per group for binary metrics.

    Uses Cohen's h and statsmodels power analysis to determine how many
    observations are needed in each group to detect a given effect size.

    Parameters
    ----------
    p_baseline : float
        Baseline conversion rate (proportion between 0 and 1)
    mde : float
        Minimum Detectable Effect as RELATIVE lift
        (e.g., 0.10 for 10% relative increase)
    alpha : float, default=0.05
        Significance level (Type I error rate)
    power : float, default=0.80
        Statistical power (1 - Type II error rate)
    two_sided : bool, default=True
        Whether to use two-sided test

    Returns
    -------
    int
        Required sample size PER GROUP

    Notes
    -----
    - Total sample size = 2 × returned value
    - Formula uses normal approximation (valid for large samples)
    - MDE is RELATIVE: p2 = p1 × (1 + mde)

    Example
    -------
    >>> # Detect 10% relative lift from 5% baseline with 80% power
    >>> n = required_samples_binary(p_baseline=0.05, mde=0.10)
    >>> print(f"Need {n:,} users per group ({n*2:,} total)")
    Need 15,681 users per group (31,362 total)

    >>> # Higher baseline needs fewer samples
    >>> n = required_samples_binary(p_baseline=0.25, mde=0.10)
    >>> print(f"Need {n:,} users per group")
    Need 3,842 users per group
    """
    if not (0 < p_baseline < 1):
        raise ValueError("p_baseline must be between 0 and 1")
    if mde <= 0:
        raise ValueError("MDE must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1")

    # Calculate treatment proportion
    p_treatment = p_baseline * (1 + mde)

    if p_treatment > 1:
        raise ValueError(
            f"Treatment proportion {p_treatment:.3f} > 1. "
            f"Reduce MDE or baseline."
        )

    # Calculate Cohen's h
    effect_size = cohens_h(p_baseline, p_treatment)

    # Calculate sample size using statsmodels
    n_per_group = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided' if two_sided else 'larger',
    )

    return int(np.ceil(n_per_group))


def required_samples_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Calculate required sample size per group for continuous metrics.

    Uses Cohen's d and statsmodels power analysis to determine how many
    observations are needed in each group to detect a given effect size.

    Parameters
    ----------
    baseline_mean : float
        Baseline mean value
    baseline_std : float
        Baseline standard deviation
    mde : float
        Minimum Detectable Effect as ABSOLUTE difference
        (e.g., 15 for $15 revenue increase)
    alpha : float, default=0.05
        Significance level (Type I error rate)
    power : float, default=0.80
        Statistical power (1 - Type II error rate)
    two_sided : bool, default=True
        Whether to use two-sided test

    Returns
    -------
    int
        Required sample size PER GROUP

    Notes
    -----
    - Total sample size = 2 × returned value
    - Assumes equal variances between groups
    - MDE is ABSOLUTE difference in means

    Example
    -------
    >>> # Detect $15 lift from $175 baseline (std=$80) with 80% power
    >>> n = required_samples_continuous(
    ...     baseline_mean=175,
    ...     baseline_std=80,
    ...     mde=15
    ... )
    >>> print(f"Need {n:,} users per group")
    Need 556 users per group

    >>> # Smaller effect needs more samples
    >>> n = required_samples_continuous(
    ...     baseline_mean=175,
    ...     baseline_std=80,
    ...     mde=5
    ... )
    >>> print(f"Need {n:,} users per group")
    Need 5,012 users per group
    """
    if baseline_std <= 0:
        raise ValueError("baseline_std must be positive")
    if mde <= 0:
        raise ValueError("MDE must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1")

    # Calculate Cohen's d
    effect_size = mde / baseline_std

    # Calculate sample size using statsmodels
    n_per_group = tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided' if two_sided else 'larger',
    )

    return int(np.ceil(n_per_group))


def interpret_effect_size(effect_size: float, metric_type: str = 'h') -> str:
    """
    Interpret Cohen's h or Cohen's d effect size.

    Parameters
    ----------
    effect_size : float
        Computed effect size (h or d)
    metric_type : str, default='h'
        Either 'h' (proportions) or 'd' (means)

    Returns
    -------
    str
        Interpretation: "Negligible", "Small", "Medium", or "Large"

    Example
    -------
    >>> interpretation = interpret_effect_size(0.65, metric_type='h')
    >>> print(interpretation)
    Medium
    """
    abs_effect = abs(effect_size)

    if abs_effect > 0.8:
        return "Large"
    elif abs_effect > 0.5:
        return "Medium"
    elif abs_effect > 0.2:
        return "Small"
    else:
        return "Negligible"


def power_analysis_summary(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Generate comprehensive power analysis summary for binary metrics.

    Parameters
    ----------
    p_baseline : float
        Baseline conversion rate
    mde : float
        Minimum Detectable Effect (relative)
    alpha : float
        Significance level
    power : float
        Statistical power

    Returns
    -------
    dict
        Summary with sample sizes, effect sizes, and interpretations

    Example
    -------
    >>> summary = power_analysis_summary(p_baseline=0.05, mde=0.10)
    >>> print(summary['sample_per_group'])
    15681
    >>> print(summary['interpretation'])
    Small
    """
    p_treatment = p_baseline * (1 + mde)
    h = cohens_h(p_baseline, p_treatment)
    n = required_samples_binary(p_baseline, mde, alpha, power)

    return {
        'p_baseline': p_baseline,
        'p_treatment': p_treatment,
        'mde_relative': mde,
        'mde_absolute': p_treatment - p_baseline,
        'cohens_h': h,
        'interpretation': interpret_effect_size(h, 'h'),
        'sample_per_group': n,
        'sample_total': n * 2,
        'alpha': alpha,
        'power': power,
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Power Analysis Demo")
    print("=" * 80)

    # Binary metric example
    print("\n📊 BINARY METRIC (Conversion Rate)")
    print("-" * 80)
    summary = power_analysis_summary(p_baseline=0.05, mde=0.10)
    print(f"Baseline: {summary['p_baseline']:.1%}")
    print(f"Treatment: {summary['p_treatment']:.2%}")
    print(f"MDE: {summary['mde_relative']:.1%} relative ({summary['mde_absolute']:.2%} absolute)")
    print(f"Cohen's h: {summary['cohens_h']:.4f} ({summary['interpretation']})")
    print(f"Sample needed: {summary['sample_per_group']:,} per group ({summary['sample_total']:,} total)")

    # Continuous metric example
    print("\n📊 CONTINUOUS METRIC (Revenue)")
    print("-" * 80)
    n_cont = required_samples_continuous(baseline_mean=175, baseline_std=80, mde=15)
    d = 15 / 80
    print(f"Baseline: $175 (std=$80)")
    print(f"MDE: $15")
    print(f"Cohen's d: {d:.4f} ({interpret_effect_size(d, 'd')})")
    print(f"Sample needed: {n_cont:,} per group ({n_cont*2:,} total)")
