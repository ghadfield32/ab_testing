"""
Randomization Quality Checks for A/B Testing
============================================

Functions for detecting Sample Ratio Mismatch (SRM) and other randomization
quality issues that can invalidate experiment results.

Example Usage:
--------------
>>> from ab_testing.core import randomization
>>>
>>> # Check for SRM (50/50 split)
>>> result = randomization.srm_check(
...     n_control=10050,
...     n_treatment=9950,
...     expected_ratio=[0.5, 0.5]
... )
>>> print(f"SRM detected: {result['srm_detected']}")
>>> print(f"P-value: {result['p_value']:.4f}")
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_ratio: Optional[List[float]] = None,
    alpha: float = 0.01,
    pp_threshold: float = 0.01,
    count_threshold: Optional[int] = None,
) -> Dict[str, float]:
    """
    Sample Ratio Mismatch (SRM) check using two-stage gating.

    Detects if the observed sample sizes deviate significantly from the
    expected allocation ratio. Uses both statistical and practical significance
    to avoid false alarms with large samples.

    Two-Stage Gating:
    - Stage A (Statistical): Chi-square p-value < alpha
    - Stage B (Practical): Deviation exceeds pp_threshold OR count_threshold
    - Hard gate on: srm_severe = detected AND practical_significant
    - Warn on: srm_warning = detected but NOT practical (borderline case)

    Parameters
    ----------
    n_control : int
        Observed sample size in control group
    n_treatment : int
        Observed sample size in treatment group
    expected_ratio : list of float, optional
        Expected allocation ratio [control, treatment].
        Default: [0.5, 0.5] for 50/50 split
    alpha : float, default=0.01
        Significance level for statistical test (use conservative threshold)
    pp_threshold : float, default=0.01
        Practical significance threshold in percentage points (0.01 = 1pp).
        Typical values: 0.01-0.02 (1-2 percentage points)
    count_threshold : int, optional
        Alternative practical threshold in absolute counts.
        If None, only pp_threshold is used.

    Returns
    -------
    dict
        Dictionary with keys:
        - n_control: Control sample size
        - n_treatment: Treatment sample size
        - expected_control: Expected control size
        - expected_treatment: Expected treatment size
        - ratio_control: Observed control proportion
        - ratio_treatment: Observed treatment proportion
        - chi2_statistic: Chi-square test statistic
        - p_value: P-value
        - srm_detected: Statistical significance (p < alpha)
        - pp_deviation_control: |observed - expected| in pp for control
        - max_pp_deviation: Maximum deviation in pp
        - count_deviation_control: |n_control - expected_control| in counts
        - max_count_deviation: Maximum count deviation
        - practical_significant: Whether deviation exceeds practical thresholds
        - srm_severe: srm_detected AND practical_significant (HARD GATE)
        - srm_warning: srm_detected but NOT practical (warn only)

    Notes
    -----
    - CRITICAL: Always run SRM check BEFORE looking at outcome metrics
    - Use conservative alpha (0.01) to avoid false alarms
    - Common causes of SRM:
      * Bot filtering affecting groups differently
      * Page load/redirect failures in treatment
      * Browser/device compatibility issues
      * Bugs in randomization code
    - If SRM detected: STOP and investigate before interpreting results

    References
    ----------
    - Kohavi et al. (2012): "Trustworthy Online Controlled Experiments"
    - https://exp-platform.com/Documents/2013-02-XXXXX-ExPpitfalls.pdf

    Example
    -------
    >>> # Expected 50/50 split, but observed 51/49 (acceptable)
    >>> result = srm_check(n_control=5100, n_treatment=4900)
    >>> print(f"SRM detected: {result['srm_detected']}")
    False

    >>> # Large mismatch (53/47) - likely SRM
    >>> result = srm_check(n_control=53000, n_treatment=47000)
    >>> print(f"SRM detected: {result['srm_detected']}")
    True
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Sample sizes must be positive")

    # Default to 50/50 split
    if expected_ratio is None:
        expected_ratio = [0.5, 0.5]

    if len(expected_ratio) != 2:
        raise ValueError("expected_ratio must have exactly 2 elements")
    if not np.isclose(sum(expected_ratio), 1.0):
        raise ValueError("expected_ratio must sum to 1.0")
    if any(r <= 0 for r in expected_ratio):
        raise ValueError("expected_ratio elements must be positive")

    # Total sample size
    n_total = n_control + n_treatment

    # Expected counts
    expected = np.array(expected_ratio) * n_total
    expected_control = expected[0]
    expected_treatment = expected[1]

    # Observed counts
    observed = np.array([n_control, n_treatment])

    # Chi-square goodness-of-fit test
    chi2_statistic = np.sum((observed - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_statistic, df=1)

    # Observed ratios
    ratio_control = n_control / n_total
    ratio_treatment = n_treatment / n_total

    # Stage A: Statistical significance
    srm_detected = p_value < alpha

    # Stage B: Practical significance
    # Calculate deviations in percentage points
    pp_deviation_control = abs(ratio_control - expected_ratio[0])
    pp_deviation_treatment = abs(ratio_treatment - expected_ratio[1])
    max_pp_deviation = max(pp_deviation_control, pp_deviation_treatment)

    # Calculate deviations in absolute counts
    count_deviation_control = abs(n_control - expected_control)
    count_deviation_treatment = abs(n_treatment - expected_treatment)
    max_count_deviation = max(count_deviation_control, count_deviation_treatment)

    # Check if deviation exceeds practical thresholds
    exceeds_pp_threshold = max_pp_deviation > pp_threshold
    if count_threshold is not None:
        exceeds_count_threshold = max_count_deviation > count_threshold
        practical_significant = exceeds_pp_threshold or exceeds_count_threshold
    else:
        practical_significant = exceeds_pp_threshold

    # Two-stage determination
    srm_severe = srm_detected and practical_significant  # HARD GATE
    srm_warning = srm_detected and not practical_significant  # Warn only

    return {
        'n_control': n_control,
        'n_treatment': n_treatment,
        'expected_control': expected_control,
        'expected_treatment': expected_treatment,
        'ratio_control': ratio_control,
        'ratio_treatment': ratio_treatment,
        'chi2_statistic': chi2_statistic,
        'p_value': p_value,
        # Stage A: Statistical
        'srm_detected': srm_detected,
        # Stage B: Practical significance metrics
        'pp_deviation_control': pp_deviation_control,
        'pp_deviation_treatment': pp_deviation_treatment,
        'max_pp_deviation': max_pp_deviation,
        'count_deviation_control': count_deviation_control,
        'count_deviation_treatment': count_deviation_treatment,
        'max_count_deviation': max_count_deviation,
        'practical_significant': practical_significant,
        # Two-stage determination
        'srm_severe': srm_severe,
        'srm_warning': srm_warning,
    }


def multi_group_srm_check(
    observed_counts: List[int],
    expected_ratio: Optional[List[float]] = None,
    alpha: float = 0.01,
) -> Dict[str, float]:
    """
    SRM check for experiments with more than 2 groups (e.g., A/B/C test).

    Parameters
    ----------
    observed_counts : list of int
        Observed sample sizes for each group [n_A, n_B, n_C, ...]
    expected_ratio : list of float, optional
        Expected allocation ratio [r_A, r_B, r_C, ...].
        Default: Equal allocation (1/k for k groups)
    alpha : float, default=0.01
        Significance level

    Returns
    -------
    dict
        Dictionary with keys:
        - observed_counts: Input observed counts
        - expected_counts: Expected counts for each group
        - observed_ratio: Observed proportions
        - chi2_statistic: Chi-square test statistic
        - df: Degrees of freedom
        - p_value: P-value
        - srm_detected: Whether SRM is detected

    Example
    -------
    >>> # A/B/C test with 33/33/33 expected split
    >>> result = multi_group_srm_check([3300, 3350, 3350])
    >>> print(f"SRM detected: {result['srm_detected']}")
    """
    if len(observed_counts) < 2:
        raise ValueError("Need at least 2 groups")
    if any(n <= 0 for n in observed_counts):
        raise ValueError("All sample sizes must be positive")

    k = len(observed_counts)
    n_total = sum(observed_counts)

    # Default to equal allocation
    if expected_ratio is None:
        expected_ratio = [1/k] * k

    if len(expected_ratio) != k:
        raise ValueError(f"expected_ratio must have {k} elements")
    if not np.isclose(sum(expected_ratio), 1.0):
        raise ValueError("expected_ratio must sum to 1.0")
    if any(r <= 0 for r in expected_ratio):
        raise ValueError("expected_ratio elements must be positive")

    # Expected counts
    expected = np.array(expected_ratio) * n_total

    # Observed
    observed = np.array(observed_counts)

    # Chi-square test
    chi2_statistic = np.sum((observed - expected)**2 / expected)
    df = k - 1
    p_value = 1 - stats.chi2.cdf(chi2_statistic, df=df)

    # Observed ratios
    observed_ratio = observed / n_total

    return {
        'observed_counts': list(observed),
        'expected_counts': list(expected),
        'observed_ratio': list(observed_ratio),
        'chi2_statistic': chi2_statistic,
        'df': df,
        'p_value': p_value,
        'srm_detected': p_value < alpha,
    }


def balance_check(
    control_covariates: np.ndarray,
    treatment_covariates: np.ndarray,
    covariate_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Check balance of covariates between control and treatment groups.

    Tests whether pre-experiment characteristics (age, region, device type, etc.)
    are balanced across groups. Significant imbalance suggests randomization issues.

    Parameters
    ----------
    control_covariates : np.ndarray
        Covariate matrix for control group (shape: n_control × n_covariates)
    treatment_covariates : np.ndarray
        Covariate matrix for treatment group (shape: n_treatment × n_covariates)
    covariate_names : list of str, optional
        Names of covariates for reporting
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with keys:
        - covariate_tests: List of dicts with test results for each covariate
        - n_imbalanced: Number of covariates with significant imbalance
        - balance_ok: Whether overall balance is acceptable

    Notes
    -----
    - Uses t-test for continuous covariates
    - Minor imbalance is expected (5% false positive rate)
    - Concern if >10% of covariates show imbalance

    Example
    -------
    >>> import numpy as np
    >>> # Pre-experiment age, revenue, sessions
    >>> control_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))
    >>> treatment_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))
    >>> result = balance_check(control_cov, treatment_cov,
    ...                        covariate_names=['age', 'pre_revenue', 'pre_sessions'])
    >>> print(f"Balance OK: {result['balance_ok']}")
    """
    if control_covariates.ndim == 1:
        control_covariates = control_covariates.reshape(-1, 1)
    if treatment_covariates.ndim == 1:
        treatment_covariates = treatment_covariates.reshape(-1, 1)

    if control_covariates.shape[1] != treatment_covariates.shape[1]:
        raise ValueError("Control and treatment must have same number of covariates")

    n_covariates = control_covariates.shape[1]

    if covariate_names is None:
        covariate_names = [f"covariate_{i}" for i in range(n_covariates)]

    if len(covariate_names) != n_covariates:
        raise ValueError(f"Need {n_covariates} covariate names")

    # Test each covariate
    covariate_tests = []
    for i in range(n_covariates):
        control_vals = control_covariates[:, i]
        treatment_vals = treatment_covariates[:, i]

        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(treatment_vals, control_vals, equal_var=False)

        # Standardized mean difference (Cohen's d)
        pooled_std = np.sqrt((control_vals.var() + treatment_vals.var()) / 2)
        smd = (treatment_vals.mean() - control_vals.mean()) / pooled_std

        covariate_tests.append({
            'name': covariate_names[i],
            'control_mean': control_vals.mean(),
            'treatment_mean': treatment_vals.mean(),
            'difference': treatment_vals.mean() - control_vals.mean(),
            'smd': smd,
            'p_value': p_value,
            'imbalanced': p_value < alpha,
        })

    # Count imbalanced covariates
    n_imbalanced = sum(1 for test in covariate_tests if test['imbalanced'])

    # Rule of thumb: concern if >10% of covariates imbalanced
    balance_ok = n_imbalanced <= max(1, int(0.1 * n_covariates))

    return {
        'covariate_tests': covariate_tests,
        'n_imbalanced': n_imbalanced,
        'balance_ok': balance_ok,
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Randomization Quality Checks Demo")
    print("=" * 80)

    # SRM check - healthy
    print("\n🔍 SRM CHECK (Healthy Randomization)")
    print("-" * 80)
    result = srm_check(n_control=5050, n_treatment=4950)
    print(f"Control: {result['n_control']:,} ({result['ratio_control']:.2%})")
    print(f"Treatment: {result['n_treatment']:,} ({result['ratio_treatment']:.2%})")
    print(f"Expected: 50/50")
    print(f"Chi-square: {result['chi2_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"SRM detected: {'⚠️ YES' if result['srm_detected'] else '✅ NO'}")

    # SRM check - problematic
    print("\n🔍 SRM CHECK (Problematic)")
    print("-" * 80)
    result = srm_check(n_control=53000, n_treatment=47000)
    print(f"Control: {result['n_control']:,} ({result['ratio_control']:.2%})")
    print(f"Treatment: {result['n_treatment']:,} ({result['ratio_treatment']:.2%})")
    print(f"Expected: 50/50")
    print(f"Chi-square: {result['chi2_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"SRM detected: {'⚠️ YES' if result['srm_detected'] else '✅ NO'}")

    # Multi-group SRM check
    print("\n🔍 MULTI-GROUP SRM CHECK (A/B/C Test)")
    print("-" * 80)
    result = multi_group_srm_check([3300, 3350, 3350])
    print(f"Observed: {result['observed_counts']}")
    print(f"Expected: {[f'{x:.1f}' for x in result['expected_counts']]}")
    print(f"Observed ratio: {[f'{x:.3f}' for x in result['observed_ratio']]}")
    print(f"Chi-square: {result['chi2_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"SRM detected: {'⚠️ YES' if result['srm_detected'] else '✅ NO'}")

    # Balance check
    print("\n⚖️ COVARIATE BALANCE CHECK")
    print("-" * 80)
    np.random.seed(42)
    control_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))
    treatment_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))
    result = balance_check(
        control_cov, treatment_cov,
        covariate_names=['age', 'pre_revenue', 'pre_sessions']
    )
    print(f"{'Covariate':<15} {'Control':>10} {'Treatment':>10} {'Diff':>10} {'SMD':>8} {'P-value':>10} {'Balanced':>10}")
    print("-" * 80)
    for test in result['covariate_tests']:
        balanced = '✅' if not test['imbalanced'] else '⚠️'
        print(f"{test['name']:<15} {test['control_mean']:>10.2f} {test['treatment_mean']:>10.2f} "
              f"{test['difference']:>10.2f} {test['smd']:>8.4f} {test['p_value']:>10.4f} {balanced:>10}")
    print(f"\nImbalanced covariates: {result['n_imbalanced']}/{len(result['covariate_tests'])}")
    print(f"Overall balance: {'✅ OK' if result['balance_ok'] else '⚠️ CONCERN'}")
