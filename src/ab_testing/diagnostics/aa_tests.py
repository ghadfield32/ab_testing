"""
A/A Test Validation
===================

Validate experimentation infrastructure by running A/A tests where both groups are identical.

Key Concepts:
- **A/A Test**: Experiment where control and treatment are identical (no real change)
- **False Positive Rate**: Should match alpha (e.g., ~5% for α=0.05)
- **Instrumentation Validation**: Verify metrics, randomization, and analysis are correct
- **Power Check**: Ensure system can detect real effects when they exist

Reference:
----------
- Kohavi et al. (2020): "Trustworthy Online Controlled Experiments" Chapter 3
- Statsig (2021): "How We Ensure Statistical Rigor"
  https://www.statsig.com/blog/p-values
- Meta Engineering: "Building Confidence in A/B Testing Infrastructure"

Example Usage:
--------------
>>> from ab_testing.diagnostics import aa_tests
>>> import numpy as np
>>>
>>> # Simulate A/A test (both groups identical)
>>> np.random.seed(42)
>>> control = np.random.binomial(1, 0.10, 10000)
>>> treatment = np.random.binomial(1, 0.10, 10000)  # Same distribution!
>>>
>>> # Run A/A validation
>>> result = aa_tests.run_aa_test(control, treatment)
>>> print(f"P-value: {result['p_value']:.4f}")
>>> print(f"False positive: {result['false_positive']}")
>>>
>>> # Run multiple A/A tests to validate infrastructure
>>> validation = aa_tests.validate_infrastructure(
...     n_tests=100,
...     sample_size=1000,
...     p_baseline=0.10
... )
>>> print(f"False positive rate: {validation['false_positive_rate']:.2%}")
>>> print(f"Expected: {validation['expected_fp_rate']:.2%}")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats


def run_aa_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    test_type: str = 'auto',
) -> Dict[str, Any]:
    """
    Run a single A/A test to check for false positives.

    In a valid A/A test, both groups are identical, so:
    - True effect = 0
    - P-value should be uniformly distributed on [0, 1]
    - ~5% of tests should have p < 0.05 (if α=0.05)

    Parameters
    ----------
    control : np.ndarray
        Control group data
    treatment : np.ndarray
        Treatment group data (should be from same distribution!)
    alpha : float, default=0.05
        Significance level
    test_type : {'auto', 'proportion', 'continuous'}, default='auto'
        Type of test to run

    Returns
    -------
    dict
        Dictionary with:
        - mean_control: Mean of control group
        - mean_treatment: Mean of treatment group
        - difference: Observed difference
        - p_value: P-value from statistical test
        - significant: Whether p < alpha (FALSE POSITIVE if true)
        - false_positive: Same as significant
        - test_used: Which test was applied

    Example
    -------
    >>> control = np.random.binomial(1, 0.10, 1000)
    >>> treatment = np.random.binomial(1, 0.10, 1000)
    >>> result = run_aa_test(control, treatment)
    >>> if result['false_positive']:
    ...     print("⚠️ False positive detected in A/A test")
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Need at least 2 observations per group")

    mean_c = control.mean()
    mean_t = treatment.mean()
    diff = mean_t - mean_c

    # Determine test type
    if test_type == 'auto':
        # Binary if all values are 0 or 1
        if np.all((control == 0) | (control == 1)) and np.all((treatment == 0) | (treatment == 1)):
            test_type = 'proportion'
        else:
            test_type = 'continuous'

    # Run appropriate test
    if test_type == 'proportion':
        # Z-test for proportions
        x_c = control.sum()
        n_c = len(control)
        x_t = treatment.sum()
        n_t = len(treatment)

        p_pooled = (x_c + x_t) / (n_c + n_t)
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_c + 1/n_t))

        if se_pooled == 0:
            z_stat = 0
            p_value = 1.0
        else:
            z_stat = (mean_t - mean_c) / se_pooled
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        test_used = 'z_test_proportions'

    else:
        # T-test for continuous metrics
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        test_used = 'welch_ttest'

    significant = p_value < alpha

    return {
        'mean_control': mean_c,
        'mean_treatment': mean_t,
        'difference': diff,
        'p_value': p_value,
        'significant': significant,
        'false_positive': significant,  # In A/A test, significant = false positive
        'test_used': test_used,
        'alpha': alpha,
    }


def validate_infrastructure(
    n_tests: int = 100,
    sample_size: int = 1000,
    p_baseline: float = 0.10,
    alpha: float = 0.05,
    metric_type: str = 'binary',
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run multiple A/A tests to validate experimentation infrastructure.

    Simulates many A/A tests and checks if false positive rate matches expected alpha.

    Parameters
    ----------
    n_tests : int, default=100
        Number of A/A tests to run
    sample_size : int, default=1000
        Sample size per group in each test
    p_baseline : float, default=0.10
        Baseline rate for binary metrics
    alpha : float, default=0.05
        Significance level
    metric_type : {'binary', 'continuous'}, default='binary'
        Type of metric to simulate
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with:
        - n_tests: Number of A/A tests run
        - false_positive_count: Number of false positives
        - false_positive_rate: Observed false positive rate
        - expected_fp_rate: Expected rate (alpha)
        - fp_rate_ci: 95% CI for false positive rate
        - passed: Whether FP rate is within expected range
        - p_values: Array of all p-values
        - p_value_uniform_test: KS test for p-value uniformity

    Notes
    -----
    - Expected false positive rate = alpha (e.g., 5%)
    - Binomial 95% CI used for false positive rate
    - P-values should be uniformly distributed on [0, 1]

    Example
    -------
    >>> result = validate_infrastructure(n_tests=100, sample_size=1000)
    >>> print(f"False positive rate: {result['false_positive_rate']:.2%}")
    >>> if not result['passed']:
    ...     print("⚠️ Infrastructure validation failed!")
    """
    if n_tests < 10:
        raise ValueError("Need at least 10 tests for reliable validation")
    if sample_size < 10:
        raise ValueError("Sample size must be at least 10")

    if random_state is not None:
        np.random.seed(random_state)

    false_positives = []
    p_values = []

    for _ in range(n_tests):
        # Generate A/A data (both groups identical)
        if metric_type == 'binary':
            control = np.random.binomial(1, p_baseline, sample_size)
            treatment = np.random.binomial(1, p_baseline, sample_size)
        else:
            # Continuous metric
            control = np.random.normal(100, 20, sample_size)
            treatment = np.random.normal(100, 20, sample_size)

        # Run A/A test
        result = run_aa_test(control, treatment, alpha=alpha)
        false_positives.append(result['false_positive'])
        p_values.append(result['p_value'])

    false_positives = np.array(false_positives)
    p_values = np.array(p_values)

    # Calculate false positive rate
    fp_count = false_positives.sum()
    fp_rate = fp_count / n_tests

    # Binomial 95% CI for false positive rate
    # Using Wilson score interval
    z = stats.norm.ppf(1 - 0.05/2)
    denominator = 1 + z**2 / n_tests
    center = (fp_rate + z**2 / (2 * n_tests)) / denominator
    margin = z * np.sqrt(fp_rate * (1 - fp_rate) / n_tests + z**2 / (4 * n_tests**2)) / denominator
    fp_ci_lower = max(0, center - margin)
    fp_ci_upper = min(1, center + margin)

    # Check if CI contains expected alpha
    passed = (fp_ci_lower <= alpha <= fp_ci_upper)

    # Test if p-values are uniform on [0, 1]
    # Use Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(p_values, 'uniform')

    return {
        'n_tests': n_tests,
        'false_positive_count': int(fp_count),
        'false_positive_rate': fp_rate,
        'expected_fp_rate': alpha,
        'fp_rate_ci': (fp_ci_lower, fp_ci_upper),
        'passed': passed,
        'p_values': p_values,
        'p_value_ks_statistic': ks_stat,
        'p_value_ks_p_value': ks_p,
        'p_values_uniform': ks_p > 0.05,  # P-values should be uniform
    }


def power_check(
    n_tests: int = 100,
    sample_size: int = 1000,
    true_effect: float = 0.10,
    p_baseline: float = 0.10,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify that infrastructure CAN detect real effects (power check).

    Simulates A/B tests with a known true effect and checks detection rate.

    Parameters
    ----------
    n_tests : int, default=100
        Number of tests to run
    sample_size : int, default=1000
        Sample size per group
    true_effect : float, default=0.10
        True relative effect (e.g., 0.10 = 10% lift)
    p_baseline : float, default=0.10
        Baseline conversion rate
    alpha : float, default=0.05
        Significance level
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with:
        - n_tests: Number of tests run
        - true_positive_count: Number of correct detections
        - observed_power: Empirical power (detection rate)
        - theoretical_power: Expected power from formula
        - power_ci: 95% CI for observed power
        - passed: Whether observed power is reasonable

    Example
    -------
    >>> result = power_check(n_tests=100, sample_size=5000, true_effect=0.10)
    >>> print(f"Observed power: {result['observed_power']:.1%}")
    >>> if result['observed_power'] < 0.70:
    ...     print("⚠️ Low power - may miss real effects")
    """
    if random_state is not None:
        np.random.seed(random_state)

    detections = []
    effects = []

    p_treatment = p_baseline * (1 + true_effect)

    for _ in range(n_tests):
        # Generate A/B data with real effect
        control = np.random.binomial(1, p_baseline, sample_size)
        treatment = np.random.binomial(1, p_treatment, sample_size)

        # Run test
        result = run_aa_test(control, treatment, alpha=alpha, test_type='proportion')
        detections.append(result['significant'])
        effects.append(result['difference'])

    detections = np.array(detections)

    # Observed power
    tp_count = detections.sum()
    observed_power = tp_count / n_tests

    # 95% CI for power (binomial proportion)
    z = stats.norm.ppf(1 - 0.05/2)
    denominator = 1 + z**2 / n_tests
    center = (observed_power + z**2 / (2 * n_tests)) / denominator
    margin = z * np.sqrt(observed_power * (1 - observed_power) / n_tests + z**2 / (4 * n_tests**2)) / denominator
    power_ci_lower = max(0, center - margin)
    power_ci_upper = min(1, center + margin)

    # Theoretical power (using statsmodels would be more accurate, but approximate here)
    from ab_testing.core import power as power_module
    theoretical_power = power_module.power_binary(
        p1=p_baseline,
        p2=p_treatment,
        n=sample_size,
        alpha=alpha,
    )

    # Check if observed power is within 10% of theoretical
    passed = abs(observed_power - theoretical_power) < 0.10

    return {
        'n_tests': n_tests,
        'true_positive_count': int(tp_count),
        'observed_power': observed_power,
        'theoretical_power': theoretical_power,
        'power_ci': (power_ci_lower, power_ci_upper),
        'passed': passed,
        'mean_detected_effect': np.array(effects)[detections].mean() if tp_count > 0 else 0,
    }


def diagnose_issues(
    validation_result: Dict[str, Any],
    power_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Diagnose potential issues with experimentation infrastructure.

    Parameters
    ----------
    validation_result : dict
        Result from validate_infrastructure()
    power_result : dict, optional
        Result from power_check()

    Returns
    -------
    dict
        Dictionary with:
        - issues_detected: List of potential issues
        - severity: 'critical', 'warning', or 'ok'
        - recommendations: List of recommended actions

    Example
    -------
    >>> validation = validate_infrastructure(n_tests=100)
    >>> diagnosis = diagnose_issues(validation)
    >>> for issue in diagnosis['issues_detected']:
    ...     print(f"⚠️ {issue}")
    """
    issues = []
    recommendations = []

    # Check false positive rate
    fp_rate = validation_result['false_positive_rate']
    expected = validation_result['expected_fp_rate']
    passed = validation_result['passed']

    if not passed:
        if fp_rate > expected * 1.5:
            issues.append(
                f"False positive rate ({fp_rate:.2%}) is significantly higher than expected ({expected:.2%})"
            )
            recommendations.append("Check for:")
            recommendations.append("  - Randomization issues (SRM)")
            recommendations.append("  - Peeking (checking results multiple times)")
            recommendations.append("  - Multiple testing without correction")
            recommendations.append("  - Data quality issues")

        elif fp_rate < expected * 0.5:
            issues.append(
                f"False positive rate ({fp_rate:.2%}) is significantly lower than expected ({expected:.2%})"
            )
            recommendations.append("Check for:")
            recommendations.append("  - Overly conservative variance estimates")
            recommendations.append("  - Sample size calculation errors")

    # Check p-value uniformity
    if not validation_result['p_values_uniform']:
        issues.append("P-values are not uniformly distributed (KS test failed)")
        recommendations.append("Check statistical test implementation")

    # Check power (if provided)
    if power_result is not None:
        obs_power = power_result['observed_power']
        theo_power = power_result['theoretical_power']

        if obs_power < theo_power * 0.80:
            issues.append(
                f"Observed power ({obs_power:.1%}) is much lower than theoretical ({theo_power:.1%})"
            )
            recommendations.append("Check for:")
            recommendations.append("  - Metric implementation errors")
            recommendations.append("  - Higher-than-expected variance")
            recommendations.append("  - Data pipeline issues")

    # Determine severity
    if len(issues) == 0:
        severity = 'ok'
    elif fp_rate > expected * 2 or (power_result and power_result['observed_power'] < 0.50):
        severity = 'critical'
    else:
        severity = 'warning'

    return {
        'issues_detected': issues,
        'severity': severity,
        'recommendations': recommendations,
        'summary': (
            '✅ Infrastructure validation passed' if severity == 'ok'
            else f'⚠️ Issues detected ({severity})'
        ),
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("A/A Test Validation Demo")
    print("=" * 80)

    np.random.seed(42)

    # Run single A/A test
    print("\n📊 SINGLE A/A TEST")
    print("-" * 80)

    control = np.random.binomial(1, 0.10, 1000)
    treatment = np.random.binomial(1, 0.10, 1000)

    result = run_aa_test(control, treatment)
    print(f"Control rate: {result['mean_control']:.2%}")
    print(f"Treatment rate: {result['mean_treatment']:.2%}")
    print(f"Difference: {result['difference']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant at α=0.05: {'✅ YES' if result['significant'] else '❌ NO'}")
    if result['false_positive']:
        print("⚠️ FALSE POSITIVE detected in A/A test")

    # Validate infrastructure with 100 A/A tests
    print("\n🔬 INFRASTRUCTURE VALIDATION (100 A/A TESTS)")
    print("-" * 80)

    validation = validate_infrastructure(
        n_tests=100,
        sample_size=1000,
        p_baseline=0.10,
        alpha=0.05,
        random_state=42,
    )

    print(f"Tests run: {validation['n_tests']}")
    print(f"False positives: {validation['false_positive_count']}")
    print(f"False positive rate: {validation['false_positive_rate']:.2%}")
    print(f"Expected rate: {validation['expected_fp_rate']:.2%}")
    print(f"95% CI: [{validation['fp_rate_ci'][0]:.2%}, {validation['fp_rate_ci'][1]:.2%}]")
    print(f"P-values uniform: {'✅ YES' if validation['p_values_uniform'] else '❌ NO'}")
    print(f"Validation: {'✅ PASSED' if validation['passed'] else '❌ FAILED'}")

    # Power check
    print("\n⚡ POWER CHECK (100 A/B TESTS WITH TRUE EFFECT)")
    print("-" * 80)

    power_result = power_check(
        n_tests=100,
        sample_size=5000,  # Larger sample for good power
        true_effect=0.10,  # 10% relative lift
        p_baseline=0.10,
        random_state=42,
    )

    print(f"Tests run: {power_result['n_tests']}")
    print(f"True positives: {power_result['true_positive_count']}")
    print(f"Observed power: {power_result['observed_power']:.1%}")
    print(f"Theoretical power: {power_result['theoretical_power']:.1%}")
    print(f"95% CI: [{power_result['power_ci'][0]:.1%}, {power_result['power_ci'][1]:.1%}]")
    print(f"Mean detected effect: {power_result['mean_detected_effect']:.4f}")
    print(f"Power check: {'✅ PASSED' if power_result['passed'] else '❌ FAILED'}")

    # Diagnose issues
    print("\n🔍 DIAGNOSIS")
    print("-" * 80)

    diagnosis = diagnose_issues(validation, power_result)
    print(f"Severity: {diagnosis['severity'].upper()}")
    print(f"Summary: {diagnosis['summary']}")

    if diagnosis['issues_detected']:
        print("\nIssues detected:")
        for issue in diagnosis['issues_detected']:
            print(f"  ⚠️ {issue}")

        if diagnosis['recommendations']:
            print("\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  {rec}")
    else:
        print("\n✅ No issues detected - infrastructure is working correctly")

    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS:")
    print("   - Run A/A tests BEFORE launching your experimentation platform")
    print("   - False positive rate should match alpha (±10%)")
    print("   - P-values should be uniformly distributed on [0, 1]")
    print("   - Power checks verify you CAN detect real effects")
    print("   - Regular A/A tests catch infrastructure degradation")
    print("=" * 80)
