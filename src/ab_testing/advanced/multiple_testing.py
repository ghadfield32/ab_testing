"""
Multiple Testing Correction
============================

Functions for controlling family-wise error rate (FWER) and false discovery rate
(FDR) when testing multiple hypotheses simultaneously.

Key Concepts:
- **FWER**: Probability of making ≥1 false positive
- **FDR**: Expected proportion of false positives among rejections

Methods:
- **Bonferroni**: Controls FWER (conservative)
- **Benjamini-Hochberg**: Controls FDR (less conservative, more power)

Example Usage:
--------------
>>> from ab_testing.advanced import multiple_testing
>>> import numpy as np
>>>
>>> # Test 5 metrics
>>> p_values = [0.01, 0.03, 0.04, 0.12, 0.45]
>>> bonf_p = multiple_testing.bonferroni_correction(p_values)
>>> bh_p = multiple_testing.benjamini_hochberg(p_values)
>>> print(f"Significant (Bonferroni): {sum(p < 0.05 for p in bonf_p)}/5")
>>> print(f"Significant (BH): {sum(p < 0.05 for p in bh_p)}/5")
"""

from typing import List, Dict, Optional, Any

import numpy as np
from statsmodels.stats.multitest import multipletests


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Bonferroni correction for multiple testing.

    Controls family-wise error rate (FWER) by adjusting p-values.
    Very conservative: adjusted_p = p × n

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values
    alpha : float, default=0.05
        Desired FWER

    Returns
    -------
    dict
        Dictionary with keys:
        - adjusted_p_values: Bonferroni-adjusted p-values
        - significant: Boolean array indicating significance
        - n_significant: Count of significant results
        - alpha: Significance threshold used

    Notes
    -----
    - Use when you need strong control of Type I error
    - Good for safety/guardrail metrics
    - Can be overly conservative (low power) with many tests

    Example
    -------
    >>> p_values = [0.01, 0.03, 0.04, 0.12]
    >>> result = bonferroni_correction(p_values)
    >>> print(result['n_significant'])
    """
    if len(p_values) == 0:
        raise ValueError("p_values cannot be empty")

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    sig, p_adj, _, _ = multipletests(p_values, method='bonferroni', alpha=alpha)
    return {
        'adjusted_p_values': np.array(p_adj),
        'significant': np.array(sig),
        'n_significant': int(np.sum(sig)),
        'alpha': alpha
    }


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Benjamini-Hochberg FDR correction.

    Controls false discovery rate (FDR) instead of FWER.
    Less conservative than Bonferroni, more power.

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values
    alpha : float, default=0.05
        Desired FDR level

    Returns
    -------
    dict
        Dictionary with keys:
        - adjusted_p_values: BH-adjusted p-values
        - significant: Boolean array indicating significance
        - n_significant: Count of significant results
        - alpha: Significance threshold used

    Notes
    -----
    - Use for exploratory analysis with many metrics
    - Allows more discoveries than Bonferroni
    - FDR = E[false positives / total positives]

    Reference
    ---------
    Benjamini & Hochberg (1995): "Controlling the False Discovery Rate:
    A Practical and Powerful Approach to Multiple Testing"

    Example
    -------
    >>> p_values = [0.01, 0.03, 0.04, 0.12]
    >>> result = benjamini_hochberg(p_values)
    >>> print(result['n_significant'])
    """
    if len(p_values) == 0:
        raise ValueError("p_values cannot be empty")

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    sig, p_adj, alpha_sidak, alpha_bonf = multipletests(p_values, method='fdr_bh', alpha=alpha)

    # Calculate FDR threshold (critical value for rejecting hypotheses)
    # BH procedure: reject H_i if p_(i) <= (i/n) * alpha
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    fdr_thresholds = np.array([(i+1)/n * alpha for i in range(n)])

    # Find largest i where p_(i) <= (i/n) * alpha
    fdr_threshold = alpha  # Default to alpha
    for i in range(n-1, -1, -1):
        if sorted_p[i] <= fdr_thresholds[i]:
            fdr_threshold = fdr_thresholds[i]
            break

    return {
        'adjusted_p_values': np.array(p_adj),
        'significant': np.array(sig),
        'n_significant': int(np.sum(sig)),
        'alpha': alpha,
        'fdr_threshold': fdr_threshold
    }


def sidak_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Šidák correction for multiple testing.

    Similar to Bonferroni but slightly less conservative.
    Adjusted alpha: 1 - (1 - α)^(1/n)

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values
    alpha : float
        Desired FWER

    Returns
    -------
    dict
        Dictionary with keys:
        - adjusted_p_values: Šidák-adjusted p-values
        - adjusted_alpha: Adjusted significance threshold
        - significant: Boolean array indicating significance
        - n_significant: Count of significant results
        - alpha: Original significance threshold

    Example
    -------
    >>> p_values = [0.01, 0.03, 0.04]
    >>> result = sidak_correction(p_values)
    >>> print(result['adjusted_alpha'])
    """
    if len(p_values) == 0:
        raise ValueError("p_values cannot be empty")

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    sig, p_adj, alpha_sidak, _ = multipletests(p_values, method='sidak', alpha=alpha)
    return {
        'adjusted_p_values': list(p_adj),
        'adjusted_alpha': alpha_sidak,
        'significant': list(sig),
        'n_significant': int(np.sum(sig)),
        'alpha': alpha
    }


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Holm-Bonferroni step-down correction.

    More powerful than Bonferroni while still controlling FWER.
    Sequentially tests hypotheses ordered by p-value.

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values
    alpha : float
        Desired FWER

    Returns
    -------
    dict
        Dictionary with keys:
        - adjusted_p_values: Holm-Bonferroni adjusted p-values
        - significant: Boolean array indicating significance
        - n_significant: Count of significant results
        - alpha: Significance threshold used

    Example
    -------
    >>> p_values = [0.01, 0.03, 0.04]
    >>> result = holm_bonferroni(p_values)
    >>> print(result['n_significant'])
    """
    if len(p_values) == 0:
        raise ValueError("p_values cannot be empty")

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    sig, p_adj, _, _ = multipletests(p_values, method='holm', alpha=alpha)
    return {
        'adjusted_p_values': np.array(p_adj),
        'significant': np.array(sig),
        'n_significant': int(np.sum(sig)),
        'alpha': alpha
    }


def false_positive_inflation(n_tests: int, alpha: float = 0.05) -> float:
    """
    Calculate family-wise error rate without correction.

    Shows how quickly FWER inflates with multiple tests.

    Parameters
    ----------
    n_tests : int
        Number of independent tests
    alpha : float
        Significance level per test

    Returns
    -------
    float
        Probability of ≥1 false positive

    Formula
    -------
    FWER = 1 - (1 - α)^n

    Example
    -------
    >>> fwer = false_positive_inflation(n_tests=5, alpha=0.05)
    >>> print(f"With 5 tests at α=0.05: {fwer:.1%} chance of ≥1 false positive")
    """
    return 1 - (1 - alpha)**n_tests


def multiple_testing_summary(
    p_values: List[float],
    metric_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Comprehensive summary of multiple testing corrections.

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values
    metric_names : list of str, optional
        Names of metrics (for display)
    alpha : float
        Significance level

    Returns
    -------
    dict
        Summary with:
        - raw_significant: Count of raw significant tests
        - bonferroni_significant: Count after Bonferroni
        - bh_significant: Count after Benjamini-Hochberg
        - fwer_uncorrected: Inflated error rate without correction
        - results_table: List of dicts with per-metric results

    Example
    -------
    >>> p_values = [0.01, 0.03, 0.04, 0.12, 0.45]
    >>> summary = multiple_testing_summary(
    ...     p_values,
    ...     metric_names=['conversion', 'revenue', 'retention', 'nps', 'sessions']
    ... )
    >>> print(f"Raw significant: {summary['raw_significant']}/5")
    >>> print(f"Bonferroni significant: {summary['bonferroni_significant']}/5")
    """
    if metric_names is None:
        metric_names = [f"metric_{i+1}" for i in range(len(p_values))]

    if len(metric_names) != len(p_values):
        raise ValueError("metric_names must have same length as p_values")

    # Apply corrections
    bonf = bonferroni_correction(p_values, alpha=alpha)
    bh = benjamini_hochberg(p_values, alpha=alpha)
    sidak = sidak_correction(p_values, alpha=alpha)
    holm = holm_bonferroni(p_values, alpha=alpha)

    # Count uncorrected significant
    raw_sig = sum(p < alpha for p in p_values)

    # FWER without correction
    fwer = false_positive_inflation(len(p_values), alpha=alpha)

    # Build results table
    results_table = []
    for i, name in enumerate(metric_names):
        results_table.append({
            'metric': name,
            'p_raw': p_values[i],
            'p_bonferroni': bonf['adjusted_p_values'][i],
            'p_bh': bh['adjusted_p_values'][i],
            'sig_raw': p_values[i] < alpha,
            'sig_bonferroni': bonf['significant'][i],
            'sig_bh': bh['significant'][i],
        })

    # Recommendation
    if bh['n_significant'] > 0:
        recommendation = f"Ship {bh['n_significant']}/{len(p_values)} metrics (BH-FDR corrected)"
    elif bonf['n_significant'] > 0:
        recommendation = f"Consider shipping {bonf['n_significant']}/{len(p_values)} metrics (Bonferroni)"
    elif raw_sig > 0:
        recommendation = f"Weak evidence ({raw_sig}/{len(p_values)} raw significant) - extend experiment"
    else:
        recommendation = "No significant results - abandon or redesign"

    return {
        'n_tests': len(p_values),
        'alpha': alpha,
        'uncorrected_significant': raw_sig,
        'raw_significant': raw_sig,
        'bonferroni_significant': bonf['n_significant'],
        'bh_significant': bh['n_significant'],
        'fwer_uncorrected': fwer,
        'bonferroni': bonf,
        'benjamini_hochberg': bh,
        'sidak': sidak,
        'holm': holm,
        'recommendation': recommendation,
        'results_table': results_table,
    }


def sequential_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
    is_primary: List[bool] = None,
) -> List[float]:
    """
    Sequential Bonferroni for hierarchical testing.

    Tests primary metrics first at full alpha, then secondary metrics
    only if primary metrics are significant.

    Parameters
    ----------
    p_values : list of float
        P-values
    alpha : float
        Significance level
    is_primary : list of bool, optional
        True for primary metrics, False for secondary.
        Default: First metric is primary

    Returns
    -------
    list of float
        Adjusted p-values

    Notes
    -----
    Common in drug trials and experiments with primary/secondary endpoints

    Example
    -------
    >>> p_values = [0.01, 0.03, 0.04]  # primary, secondary, secondary
    >>> is_primary = [True, False, False]
    >>> adjusted = sequential_bonferroni(p_values, is_primary=is_primary)
    """
    if is_primary is None:
        is_primary = [i == 0 for i in range(len(p_values))]

    if len(is_primary) != len(p_values):
        raise ValueError("is_primary must have same length as p_values")

    adjusted = []
    primary_passed = all(p_values[i] < alpha for i, is_prim in enumerate(is_primary) if is_prim)

    for i, (p, is_prim) in enumerate(zip(p_values, is_primary)):
        if is_prim:
            # Primary: test at full alpha
            adjusted.append(p)
        else:
            # Secondary: only test if primary passed
            if primary_passed:
                # Bonferroni correction among secondaries
                n_secondary = sum(not x for x in is_primary)
                adjusted.append(p * n_secondary)
            else:
                # Don't test secondaries
                adjusted.append(1.0)

    return adjusted


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Multiple Testing Correction Demo")
    print("=" * 80)

    # Simulate p-values from 5 metrics
    np.random.seed(42)
    metric_names = ['conversion', 'revenue', 'retention_7d', 'nps_score', 'sessions']
    p_values = [0.01, 0.03, 0.04, 0.12, 0.45]

    print("\n📊 MULTIPLE TESTING SUMMARY")
    print("-" * 80)

    summary = multiple_testing_summary(p_values, metric_names=metric_names)

    print(f"Number of tests: {summary['n_tests']}")
    print(f"Significance level: {summary['alpha']}")
    print(f"FWER without correction: {summary['fwer_uncorrected']:.1%}")
    print()

    print(f"{'Metric':<15} {'Raw p':>10} {'Bonf p':>10} {'BH p':>10} {'Raw':>5} {'Bonf':>5} {'BH':>5}")
    print("-" * 80)

    for row in summary['results_table']:
        raw_sig = '✅' if row['sig_raw'] else '❌'
        bonf_sig = '✅' if row['sig_bonferroni'] else '❌'
        bh_sig = '✅' if row['sig_bh'] else '❌'

        print(f"{row['metric']:<15} {row['p_raw']:>10.6f} {row['p_bonferroni']:>10.6f} "
              f"{row['p_bh']:>10.6f} {raw_sig:>5} {bonf_sig:>5} {bh_sig:>5}")

    print()
    print(f"Summary: Raw={summary['raw_significant']}/{summary['n_tests']}, "
          f"Bonferroni={summary['bonferroni_significant']}/{summary['n_tests']}, "
          f"BH={summary['bh_significant']}/{summary['n_tests']}")

    print("\n💡 INTERPRETATION:")
    print("   • Without correction: 2/5 metrics significant (but 22.6% chance of false positive!)")
    print("   • Bonferroni: 1/5 metrics significant (conservative, controls FWER)")
    print("   • Benjamini-Hochberg: 3/5 metrics significant (controls FDR, more power)")
