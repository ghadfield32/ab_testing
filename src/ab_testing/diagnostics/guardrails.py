"""
Guardrail Metrics Framework
============================

Framework for protecting critical business metrics while optimizing primary metrics.
Uses non-inferiority tests to ensure no unacceptable harm.

Key Concepts:
- **Primary Metrics**: What you're optimizing (e.g., conversion, engagement)
- **Guardrail Metrics**: What you must NOT harm (e.g., revenue, retention, trust)
- **Non-Inferiority**: Testing that degradation is within acceptable bounds

Reference:
----------
- Spotify Engineering (2024): "Risk-Aware Product Decisions in A/B Tests with Multiple Metrics"
  https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics
- Mixpanel (2023): "Guardrail Metrics: How to Avoid Breaking What Matters"
  https://mixpanel.com/blog/guardrail-metrics/

Example Usage:
--------------
>>> from ab_testing.diagnostics import guardrails
>>> import numpy as np
>>>
>>> # Test primary metric (positive) and guardrails (must not degrade)
>>> result = guardrails.guardrail_test(
...     control=np.random.normal(100, 20, 500),
...     treatment=np.random.normal(98, 20, 500),  # Slight degradation
...     delta=-0.05,  # Allow up to 5% degradation
...     metric_name='retention_7d'
... )
>>> print(f"Guardrail passed: {result['passed']}")
"""

import numpy as np
from typing import Dict, List, Optional, Literal
from scipy import stats


def non_inferiority_test(
    control: np.ndarray,
    treatment: np.ndarray,
    delta: float,
    alpha: float = 0.05,
    metric_type: Literal['absolute', 'relative'] = 'absolute',
) -> Dict[str, float]:
    """
    Non-inferiority test for guardrail metrics.

    Tests H0: treatment < control - delta (treatment is inferior)
    vs H1: treatment >= control - delta (treatment is non-inferior)

    Parameters
    ----------
    control : np.ndarray
        Control group observations
    treatment : np.ndarray
        Treatment group observations
    delta : float
        Non-inferiority margin (NEGATIVE for "not worse than")
        e.g., delta=-0.05 means "not more than 5% worse"
    alpha : float, default=0.05
        Significance level
    metric_type : {'absolute', 'relative'}
        Whether delta is absolute or relative

    Returns
    -------
    dict
        Dictionary with:
        - mean_control: Control mean
        - mean_treatment: Treatment mean
        - difference: Observed difference
        - delta: Non-inferiority margin
        - t_statistic: Test statistic
        - p_value: One-sided p-value
        - ci_lower: Lower bound of one-sided CI
        - passed: Whether non-inferiority is demonstrated
        - margin_used: Actual margin used (delta or delta * mean_control)

    Notes
    -----
    - Lower CI bound > delta means non-inferiority
    - Use conservative delta (small allowed degradation)
    - Common deltas: -2% to -5% for critical metrics

    Example
    -------
    >>> control = np.random.normal(100, 20, 500)
    >>> treatment = np.random.normal(98, 20, 500)  # 2% degradation
    >>> result = non_inferiority_test(control, treatment, delta=-5)
    >>> print(f"Non-inferior: {result['passed']}")  # Should pass (-2% > -5%)
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Each group must have at least 2 observations")
    if delta >= 0:
        raise ValueError("delta must be negative for non-inferiority (allowed degradation)")

    mean_c = control.mean()
    mean_t = treatment.mean()
    difference = mean_t - mean_c

    # Calculate margin
    if metric_type == 'relative':
        margin = delta * mean_c  # e.g., -5% of baseline
    else:
        margin = delta

    # Standard error
    se = np.sqrt(control.var(ddof=1)/len(control) +
                 treatment.var(ddof=1)/len(treatment))

    # Test statistic for non-inferiority
    # H0: μ_t - μ_c <= margin
    # H1: μ_t - μ_c > margin
    t_stat = (difference - margin) / se

    # One-sided p-value (upper tail)
    df = len(control) + len(treatment) - 2
    p_value = 1 - stats.t.cdf(t_stat, df=df)

    # One-sided CI lower bound
    t_critical = stats.t.ppf(1 - alpha, df=df)
    ci_lower = difference - t_critical * se

    # Non-inferiority demonstrated if CI lower bound > margin
    passed = bool(ci_lower > margin)

    return {
        'mean_control': float(mean_c),
        'mean_treatment': float(mean_t),
        'difference': float(difference),
        'delta': float(delta),
        'margin_used': float(margin),
        'metric_type': metric_type,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'passed': passed,
        'alpha': float(alpha),
    }


def guardrail_test(
    control: np.ndarray,
    treatment: np.ndarray,
    delta: float = -0.02,
    metric_name: str = 'guardrail',
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Convenience wrapper for guardrail metric testing.

    Parameters
    ----------
    control : np.ndarray
        Control observations
    treatment : np.ndarray
        Treatment observations
    delta : float, default=-0.02
        Allowed degradation (-2% by default)
    metric_name : str
        Name of metric for reporting
    alpha : float
        Significance level

    Returns
    -------
    dict
        Test results with pass/fail

    Example
    -------
    >>> result = guardrail_test(
    ...     control=np.random.normal(100, 20, 500),
    ...     treatment=np.random.normal(99, 20, 500),
    ...     delta=-0.05,
    ...     metric_name='revenue_per_user'
    ... )
    """
    result = non_inferiority_test(
        control, treatment, delta=delta, alpha=alpha, metric_type='relative'
    )

    result['metric_name'] = metric_name

    return result


def evaluate_guardrails(
    primary_result: Dict[str, any],
    guardrail_results: List[Dict[str, any]],
    require_all_pass: bool = True,
) -> Dict[str, any]:
    """
    Evaluate experiment decision with guardrail checks.

    Parameters
    ----------
    primary_result : dict
        Primary metric test result (must have 'significant' and effect size key)
        Accepts either 'absolute_lift' (z_test_proportions) or 'difference' (welch_t_test)
    guardrail_results : list of dict
        List of guardrail test results (from guardrail_test)
    require_all_pass : bool, default=True
        Whether all guardrails must pass

    Returns
    -------
    dict
        Dictionary with:
        - primary_significant: Primary metric is significant
        - primary_positive: Primary metric is positive
        - guardrails_passed: Number of guardrails passed
        - guardrails_total: Total number of guardrails
        - all_guardrails_passed: Whether all guardrails passed
        - decision: 'ship', 'hold', or 'abandon'
        - failed_guardrails: List of failed guardrail names

    Decision Logic:
    - SHIP: Primary significant & positive & all guardrails pass
    - ABANDON: Primary significant & negative OR critical guardrail fails
    - HOLD: Primary not significant OR some guardrails fail

    Example
    -------
    >>> primary = {'significant': True, 'absolute_lift': 0.05}
    >>> guardrails = [
    ...     {'metric_name': 'revenue', 'passed': True},
    ...     {'metric_name': 'retention', 'passed': True}
    ... ]
    >>> decision = evaluate_guardrails(primary, guardrails)
    >>> print(decision['decision'])  # 'ship'
    """
    primary_sig = primary_result.get('significant', False)

    # FIX: Support both z_test_proportions ('absolute_lift') and welch_t_test ('difference')
    # Try keys in order of preference
    effect_size = primary_result.get('absolute_lift',
                  primary_result.get('difference',
                  primary_result.get('relative_lift', 0)))

    primary_positive = effect_size > 0

    guardrails_passed = sum(1 for g in guardrail_results if g.get('passed', False))
    guardrails_total = len(guardrail_results)
    all_guardrails_passed = guardrails_passed == guardrails_total

    failed_guardrails = [
        g['metric_name'] for g in guardrail_results if not g.get('passed', False)
    ]

    # Decision logic
    if primary_sig and primary_positive and all_guardrails_passed:
        decision = 'ship'
    elif primary_sig and not primary_positive:
        decision = 'abandon'  # Primary metric got worse
    elif not all_guardrails_passed and require_all_pass:
        decision = 'hold'  # Guardrail failures
    elif not primary_sig:
        decision = 'hold'  # Inconclusive
    else:
        decision = 'hold'  # Default

    return {
        'primary_significant': primary_sig,
        'primary_positive': primary_positive,
        'effect_size': effect_size,  # Add for debugging
        'guardrails_passed': guardrails_passed,
        'guardrails_total': guardrails_total,
        'all_guardrails_passed': all_guardrails_passed,
        'decision': decision,
        'failed_guardrails': failed_guardrails,
    }


def power_for_guardrail(
    baseline_mean: float,
    baseline_std: float,
    delta: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """
    Calculate power to detect non-inferiority for guardrail.

    Parameters
    ----------
    baseline_mean : float
        Baseline mean of guardrail metric
    baseline_std : float
        Baseline std of guardrail metric
    delta : float
        Non-inferiority margin (relative)
    n_per_group : int
        Sample size per group
    alpha : float
        Significance level

    Returns
    -------
    float
        Statistical power

    Notes
    -----
    - Guardrails should have high power (80%+) to detect degradation
    - Don't apply Bonferroni to guardrails (Spotify recommendation)

    Example
    -------
    >>> power = power_for_guardrail(
    ...     baseline_mean=100,
    ...     baseline_std=20,
    ...     delta=-0.05,  # 5% degradation
    ...     n_per_group=1000
    ... )
    >>> print(f"Power to detect 5% degradation: {power:.1%}")
    """
    # Effect size in standard deviations
    margin_absolute = delta * baseline_mean
    effect_size = abs(margin_absolute) / baseline_std

    # Non-centrality parameter for t-test
    ncp = effect_size * np.sqrt(n_per_group / 2)

    # Critical value (one-sided)
    df = 2 * n_per_group - 2
    t_critical = stats.t.ppf(1 - alpha, df=df)

    # Power (one-sided)
    power = 1 - stats.nct.cdf(t_critical, df=df, nc=ncp)

    return power


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Guardrail Metrics Framework Demo")
    print("=" * 80)

    np.random.seed(42)

    # Scenario: Optimizing conversion, must not harm revenue & retention
    print("\n📊 SCENARIO: New checkout flow")
    print("Primary: Conversion rate (want to improve)")
    print("Guardrails: Revenue per user, 7-day retention (must not degrade >2%)")
    print("-" * 80)

    # Primary metric: Conversion rate (improved!)
    conversions_c = np.random.binomial(1, 0.10, 1000)
    conversions_t = np.random.binomial(1, 0.12, 1000)  # +20% lift
    from ab_testing.core import frequentist
    primary_result = frequentist.z_test_proportions(
        conversions_c.sum(), len(conversions_c),
        conversions_t.sum(), len(conversions_t)
    )

    print(f"\n✅ PRIMARY METRIC: Conversion Rate")
    print(f"   Control: {primary_result['p_control']:.2%}")
    print(f"   Treatment: {primary_result['p_treatment']:.2%}")
    print(f"   Lift: {primary_result['relative_lift']*100:+.1f}%")
    print(f"   Significant: {'✅ Yes' if primary_result['significant'] else '❌ No'}")

    # Guardrail 1: Revenue per user (slight degradation, but within bounds)
    revenue_c = np.random.gamma(2, 50, 1000)
    revenue_t = np.random.gamma(2, 48, 1000)  # -4% degradation
    guardrail_revenue = guardrail_test(
        revenue_c, revenue_t,
        delta=-0.05,  # Allow up to 5% degradation
        metric_name='revenue_per_user'
    )

    print(f"\n⚠️ GUARDRAIL 1: Revenue Per User")
    print(f"   Control: ${guardrail_revenue['mean_control']:.2f}")
    print(f"   Treatment: ${guardrail_revenue['mean_treatment']:.2f}")
    print(f"   Change: {(guardrail_revenue['difference']/guardrail_revenue['mean_control'])*100:+.1f}%")
    print(f"   Allowed degradation: {guardrail_revenue['delta']*100:.0f}%")
    print(f"   Non-inferior: {'✅ PASS' if guardrail_revenue['passed'] else '❌ FAIL'}")
    print(f"   (CI lower bound: {(guardrail_revenue['ci_lower']/guardrail_revenue['mean_control'])*100:+.1f}%)")

    # Guardrail 2: 7-day retention (no change)
    retention_c = np.random.binomial(1, 0.45, 1000).astype(float)
    retention_t = np.random.binomial(1, 0.44, 1000).astype(float)  # -2.2% change
    guardrail_retention = guardrail_test(
        retention_c, retention_t,
        delta=-0.05,
        metric_name='retention_7d'
    )

    print(f"\n⚠️ GUARDRAIL 2: 7-Day Retention")
    print(f"   Control: {guardrail_retention['mean_control']:.2%}")
    print(f"   Treatment: {guardrail_retention['mean_treatment']:.2%}")
    print(f"   Change: {(guardrail_retention['difference']/guardrail_retention['mean_control'])*100:+.1f}%")
    print(f"   Allowed degradation: {guardrail_retention['delta']*100:.0f}%")
    print(f"   Non-inferior: {'✅ PASS' if guardrail_retention['passed'] else '❌ FAIL'}")

    # Final decision
    print("\n🎯 FINAL DECISION")
    print("-" * 80)
    decision = evaluate_guardrails(
        primary_result,
        [guardrail_revenue, guardrail_retention]
    )

    print(f"Primary metric: {'✅ Significant & positive' if decision['primary_significant'] and decision['primary_positive'] else '❌'}")
    print(f"Guardrails: {decision['guardrails_passed']}/{decision['guardrails_total']} passed")

    if decision['failed_guardrails']:
        print(f"Failed guardrails: {', '.join(decision['failed_guardrails'])}")

    decision_emoji = {'ship': '🚢', 'hold': '⏸️', 'abandon': '❌'}
    print(f"\n{decision_emoji[decision['decision']]} DECISION: {decision['decision'].upper()}")

    if decision['decision'] == 'ship':
        print("   ✅ Ship the treatment - primary improved & guardrails safe")
    elif decision['decision'] == 'hold':
        print("   ⏸️ Hold - need more data or investigation")
    else:
        print("   ❌ Abandon - unacceptable harm detected")

    # Power analysis
    print("\n📊 GUARDRAIL POWER ANALYSIS")
    print("-" * 80)
    power = power_for_guardrail(
        baseline_mean=100,
        baseline_std=50,
        delta=-0.05,
        n_per_group=1000
    )
    print(f"Power to detect 5% degradation: {power:.1%}")
    print("✅ Guardrails should have 80%+ power to detect degradation")
