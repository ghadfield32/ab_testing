"""
Bayesian A/B Testing
====================

Functions for Bayesian inference in A/B tests, including Beta-Binomial models
for proportions and Normal models for continuous metrics.

Example Usage:
--------------
>>> from ab_testing.core import bayesian
>>>
>>> # Binary metric (conversion rate)
>>> result = bayesian.beta_binomial_ab_test(
...     x_control=50, n_control=500,
...     x_treatment=60, n_treatment=500
... )
>>> print(f"P(Treatment > Control) = {result['prob_treatment_better']:.2%}")
>>> print(f"Expected lift: {result['expected_lift']*100:.2f}%")
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def beta_binomial_ab_test(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Bayesian A/B test for binary metrics using Beta-Binomial conjugate model.

    Uses Beta prior and Binomial likelihood to get Beta posterior.
    Computes probability that treatment is better, expected loss, and
    credible intervals via Monte Carlo sampling.

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
    prior_alpha : float, default=1.0
        Alpha parameter of Beta prior (1.0 = uniform prior)
    prior_beta : float, default=1.0
        Beta parameter of Beta prior (1.0 = uniform prior)
    n_samples : int, default=100000
        Number of Monte Carlo samples
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - posterior_control: (alpha, beta) of control posterior
        - posterior_treatment: (alpha, beta) of treatment posterior
        - prob_treatment_better: P(Treatment > Control)
        - expected_lift: Expected relative lift
        - credible_interval: (lower, upper) 95% CI for lift
        - expected_loss_control: Loss if we choose control
        - expected_loss_treatment: Loss if we choose treatment
        - recommendation: 'control' or 'treatment'

    Notes
    -----
    - Posterior: Beta(prior_alpha + x, prior_beta + n - x)
    - Default prior Beta(1,1) is uniform (non-informative)
    - Use prior Beta(0.5, 0.5) for Jeffreys prior
    - Expected loss = E[max(0, θ_other - θ_chosen)]
    - No p-values: gives direct probability statements!

    References
    ----------
    - Kruschke (2014): "Doing Bayesian Data Analysis"
    - VWO Bayesian Calculator: https://vwo.com/ab-split-test-significance-calculator/

    Example
    -------
    >>> result = beta_binomial_ab_test(50, 500, 60, 500)
    >>> print(f"P(Treatment > Control) = {result['prob_treatment_better']:.2%}")
    >>> if result['prob_treatment_better'] > 0.95:
    ...     print("Strong evidence for treatment!")
    """
    if x_control < 0 or x_treatment < 0:
        raise ValueError("Number of successes must be non-negative")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Sample sizes must be positive")
    if x_control > n_control or x_treatment > n_treatment:
        raise ValueError("Number of successes cannot exceed sample size")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive")

    # Posterior parameters (conjugate update)
    post_control = (prior_alpha + x_control, prior_beta + (n_control - x_control))
    post_treatment = (prior_alpha + x_treatment, prior_beta + (n_treatment - x_treatment))

    # Monte Carlo sampling
    if random_state is not None:
        np.random.seed(random_state)

    samples_control = np.random.beta(*post_control, n_samples)
    samples_treatment = np.random.beta(*post_treatment, n_samples)

    # P(Treatment > Control)
    prob_treatment_better = (samples_treatment > samples_control).mean()

    # Relative lift: (Treatment - Control) / Control
    lift_samples = (samples_treatment / samples_control - 1)
    expected_lift = lift_samples.mean()
    credible_interval = (np.percentile(lift_samples, 2.5), np.percentile(lift_samples, 97.5))

    # Expected loss (for decision making)
    # Loss if we choose control = E[max(0, Treatment - Control)]
    expected_loss_control = np.maximum(0, samples_treatment - samples_control).mean()
    # Loss if we choose treatment = E[max(0, Control - Treatment)]
    expected_loss_treatment = np.maximum(0, samples_control - samples_treatment).mean()

    # Recommendation based on expected loss
    recommendation = 'treatment' if expected_loss_treatment < expected_loss_control else 'control'

    return {
        'posterior_control': post_control,
        'posterior_treatment': post_treatment,
        'prob_treatment_better': prob_treatment_better,
        'expected_lift': expected_lift,
        'credible_interval': credible_interval,
        'expected_loss_control': expected_loss_control,
        'expected_loss_treatment': expected_loss_treatment,
        'recommendation': recommendation,
    }


def normal_ab_test(
    control: np.ndarray,
    treatment: np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 1000.0,
    n_samples: int = 100000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Bayesian A/B test for continuous metrics using Normal-Normal conjugate model.

    Uses Normal prior and Normal likelihood to get Normal posterior.
    Appropriate for metrics like revenue, session duration, etc.

    Parameters
    ----------
    control : np.ndarray
        Observations from control group
    treatment : np.ndarray
        Observations from treatment group
    prior_mean : float, default=0.0
        Mean of Normal prior for difference
    prior_std : float, default=1000.0
        Std of Normal prior for difference (large = non-informative)
    n_samples : int, default=100000
        Number of Monte Carlo samples
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - posterior_mean: Posterior mean of difference
        - posterior_std: Posterior std of difference
        - prob_treatment_better: P(Treatment > Control)
        - expected_difference: Expected absolute difference
        - credible_interval: (lower, upper) 95% CI for difference
        - expected_loss_control: Loss if we choose control
        - expected_loss_treatment: Loss if we choose treatment
        - recommendation: 'control' or 'treatment'

    Notes
    -----
    - Uses known variance approximation (assumes large samples)
    - For small samples, use Student's t posterior instead
    - Default prior is very weak (std=1000)

    Example
    -------
    >>> import numpy as np
    >>> control_rev = np.random.normal(100, 20, 500)
    >>> treatment_rev = np.random.normal(115, 22, 500)
    >>> result = normal_ab_test(control_rev, treatment_rev)
    >>> print(f"P(Treatment > Control) = {result['prob_treatment_better']:.2%}")
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Each group must have at least 2 observations")
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")

    # Data statistics
    mean_c = control.mean()
    mean_t = treatment.mean()
    var_c = control.var(ddof=1)
    var_t = treatment.var(ddof=1)
    n_c = len(control)
    n_t = len(treatment)

    # Posterior for difference (Normal-Normal conjugacy)
    # Likelihood variance for difference
    likelihood_var = var_c/n_c + var_t/n_t
    likelihood_std = np.sqrt(likelihood_var)

    # Posterior parameters (precision-weighted average)
    prior_precision = 1 / (prior_std**2)
    likelihood_precision = 1 / likelihood_var
    posterior_precision = prior_precision + likelihood_precision
    posterior_var = 1 / posterior_precision

    # Observed difference
    observed_diff = mean_t - mean_c

    # Posterior mean (weighted average of prior and likelihood)
    posterior_mean = (prior_precision * prior_mean + likelihood_precision * observed_diff) / posterior_precision
    posterior_std = np.sqrt(posterior_var)

    # Monte Carlo sampling from posterior
    if random_state is not None:
        np.random.seed(random_state)

    diff_samples = np.random.normal(posterior_mean, posterior_std, n_samples)

    # P(Treatment > Control) = P(difference > 0)
    prob_treatment_better = (diff_samples > 0).mean()

    # Credible interval
    credible_interval = (np.percentile(diff_samples, 2.5), np.percentile(diff_samples, 97.5))

    # Expected loss
    expected_loss_control = np.maximum(0, diff_samples).mean()  # Loss if choose control
    expected_loss_treatment = np.maximum(0, -diff_samples).mean()  # Loss if choose treatment

    # Recommendation
    recommendation = 'treatment' if expected_loss_treatment < expected_loss_control else 'control'

    return {
        'posterior_mean': posterior_mean,
        'posterior_std': posterior_std,
        'prob_treatment_better': prob_treatment_better,
        'expected_difference': posterior_mean,
        'credible_interval': credible_interval,
        'expected_loss_control': expected_loss_control,
        'expected_loss_treatment': expected_loss_treatment,
        'recommendation': recommendation,
    }


def probability_to_beat_threshold(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    threshold: float,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100000,
    random_state: Optional[int] = None,
) -> float:
    """
    Calculate probability that treatment beats control by at least threshold.

    Useful for decision-making: "What's the probability that treatment
    lifts conversion by at least 5%?"

    Parameters
    ----------
    x_control, n_control : int
        Control group successes and sample size
    x_treatment, n_treatment : int
        Treatment group successes and sample size
    threshold : float
        Minimum relative lift threshold (e.g., 0.05 for 5%)
    prior_alpha, prior_beta : float
        Beta prior parameters
    n_samples : int
        Number of Monte Carlo samples
    random_state : int, optional
        Random seed

    Returns
    -------
    float
        P(Treatment / Control - 1 > threshold)

    Example
    -------
    >>> # What's P(treatment lifts conversion by ≥5%)?
    >>> prob = probability_to_beat_threshold(50, 500, 60, 500, threshold=0.05)
    >>> print(f"P(lift ≥ 5%) = {prob:.2%}")
    """
    # Posterior parameters
    post_control = (prior_alpha + x_control, prior_beta + (n_control - x_control))
    post_treatment = (prior_alpha + x_treatment, prior_beta + (n_treatment - x_treatment))

    # Monte Carlo sampling
    if random_state is not None:
        np.random.seed(random_state)

    samples_control = np.random.beta(*post_control, n_samples)
    samples_treatment = np.random.beta(*post_treatment, n_samples)

    # Relative lift
    lift = samples_treatment / samples_control - 1

    # P(lift > threshold)
    prob = (lift > threshold).mean()

    return prob


def stopping_rule_bayesian(
    prob_treatment_better: float,
    threshold_high: float = 0.95,
    threshold_low: float = 0.05,
) -> Tuple[bool, str]:
    """
    Simple Bayesian stopping rule for continuous monitoring.

    Parameters
    ----------
    prob_treatment_better : float
        P(Treatment > Control)
    threshold_high : float, default=0.95
        Stop if prob > this (treatment wins)
    threshold_low : float, default=0.05
        Stop if prob < this (control wins)

    Returns
    -------
    tuple
        (should_stop: bool, decision: str)
        decision is 'treatment', 'control', or 'continue'

    Example
    -------
    >>> should_stop, decision = stopping_rule_bayesian(prob_treatment_better=0.97)
    >>> print(decision)
    treatment
    """
    if prob_treatment_better > threshold_high:
        return (True, 'treatment')
    elif prob_treatment_better < threshold_low:
        return (True, 'control')
    else:
        return (False, 'continue')


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Bayesian A/B Testing Demo")
    print("=" * 80)

    # Binary metric example
    print("\n🎲 BETA-BINOMIAL (Conversion Rate)")
    print("-" * 80)
    result = beta_binomial_ab_test(
        x_control=50, n_control=500,
        x_treatment=60, n_treatment=500,
        random_state=42
    )
    print(f"Posterior Control: Beta({result['posterior_control'][0]}, {result['posterior_control'][1]})")
    print(f"Posterior Treatment: Beta({result['posterior_treatment'][0]}, {result['posterior_treatment'][1]})")
    print(f"\nP(Treatment > Control) = {result['prob_treatment_better']:.4f} ({result['prob_treatment_better']:.2%})")
    print(f"Expected lift: {result['expected_lift']*100:.2f}%")
    print(f"95% Credible Interval: ({result['credible_interval'][0]*100:.2f}%, {result['credible_interval'][1]*100:.2f}%)")
    print(f"\nExpected loss if choose Control: {result['expected_loss_control']:.6f}")
    print(f"Expected loss if choose Treatment: {result['expected_loss_treatment']:.6f}")
    print(f"Recommendation: {result['recommendation'].upper()}")

    # Probability to beat threshold
    print("\n📊 PROBABILITY TO BEAT THRESHOLD")
    print("-" * 80)
    prob_5pct = probability_to_beat_threshold(50, 500, 60, 500, threshold=0.05, random_state=42)
    prob_10pct = probability_to_beat_threshold(50, 500, 60, 500, threshold=0.10, random_state=42)
    print(f"P(lift ≥ 5%) = {prob_5pct:.2%}")
    print(f"P(lift ≥ 10%) = {prob_10pct:.2%}")

    # Continuous metric example
    print("\n🎲 NORMAL MODEL (Revenue)")
    print("-" * 80)
    np.random.seed(42)
    control_rev = np.random.normal(100, 20, 500)
    treatment_rev = np.random.normal(115, 22, 500)
    result = normal_ab_test(control_rev, treatment_rev, random_state=42)
    print(f"Posterior mean difference: ${result['posterior_mean']:.2f}")
    print(f"Posterior std: ${result['posterior_std']:.2f}")
    print(f"\nP(Treatment > Control) = {result['prob_treatment_better']:.4f} ({result['prob_treatment_better']:.2%})")
    print(f"95% Credible Interval: (${result['credible_interval'][0]:.2f}, ${result['credible_interval'][1]:.2f})")
    print(f"\nExpected loss if choose Control: ${result['expected_loss_control']:.2f}")
    print(f"Expected loss if choose Treatment: ${result['expected_loss_treatment']:.2f}")
    print(f"Recommendation: {result['recommendation'].upper()}")

    # Stopping rule
    print("\n🛑 STOPPING RULE")
    print("-" * 80)
    should_stop, decision = stopping_rule_bayesian(0.97)
    print(f"P(Treatment > Control) = 0.97")
    print(f"Should stop: {should_stop}")
    print(f"Decision: {decision.upper()}")
