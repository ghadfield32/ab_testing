"""
Novelty Effect Detection
========================

Detect when treatment effects decay over time due to novelty wearing off.

Key Concepts:
- **Novelty Effect**: Temporary spike in metrics due to user curiosity about new feature
- **Primacy Effect**: Similar phenomenon where existing users resist change initially
- **Sustained Effect**: Long-term treatment effect after novelty wears off
- **Holdout**: Continue experiment post-launch to detect novelty decay

Reference:
----------
- Statsig (2023): "Novelty Effects in A/B Testing"
  https://www.statsig.com/blog/novelty-effects
- Kohavi et al. (2020): "Trustworthy Online Controlled Experiments" Chapter 20
- Hohnhold et al. (2015): "Focusing on the Long-term: It's Good for Users and Business"
  Google research on novelty effects

Example Usage:
--------------
>>> from ab_testing.diagnostics import novelty
>>> import numpy as np
>>> import pandas as pd
>>>
>>> # Simulate experiment data with novelty effect
>>> dates = pd.date_range('2024-01-01', periods=30, freq='D')
>>> # Effect starts at 10% but decays to 3% over 30 days
>>> time_idx = np.arange(30)
>>> effect = 0.10 * np.exp(-0.1 * time_idx) + 0.03
>>>
>>> # Generate metrics for each day
>>> metrics_control = np.random.normal(0.50, 0.02, 30)
>>> metrics_treatment = metrics_control + effect + np.random.normal(0, 0.01, 30)
>>>
>>> # Detect novelty
>>> result = novelty.detect_novelty_effect(
...     metrics_control=metrics_control,
...     metrics_treatment=metrics_treatment,
...     time_index=dates
... )
>>> print(f"Novelty detected: {result['novelty_detected']}")
>>> print(f"Early effect: {result['early_effect']:.2%}")
>>> print(f"Late effect: {result['late_effect']:.2%}")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import curve_fit


def detect_novelty_effect(
    metrics_control: np.ndarray,
    metrics_treatment: np.ndarray,
    time_index: Optional[np.ndarray] = None,
    early_period: float = 0.25,
    late_period: float = 0.25,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect novelty effect by comparing early vs late experiment periods.

    Parameters
    ----------
    metrics_control : np.ndarray
        Time-series of control group metric (e.g., daily conversion rates)
    metrics_treatment : np.ndarray
        Time-series of treatment group metric
    time_index : np.ndarray, optional
        Time index (dates or days). If None, uses sequential integers
    early_period : float, default=0.25
        Fraction of data to use for "early" period (first 25%)
    late_period : float, default=0.25
        Fraction of data to use for "late" period (last 25%)
    alpha : float, default=0.05
        Significance level for novelty test

    Returns
    -------
    dict
        Dictionary with:
        - early_effect: Mean effect in early period
        - late_effect: Mean effect in late period
        - effect_decay: Percentage decay from early to late
        - early_ci: 95% CI for early effect
        - late_ci: 95% CI for late effect
        - novelty_test_statistic: t-statistic for early vs late
        - novelty_p_value: p-value for novelty test
        - novelty_detected: Boolean indicating if novelty detected
        - recommendation: Text recommendation

    Notes
    -----
    - Novelty detected if early effect significantly > late effect
    - Common pattern: early effect positive, late effect near zero
    - If novelty detected, recommend post-launch holdout

    Example
    -------
    >>> result = detect_novelty_effect(
    ...     metrics_control=np.random.normal(0.5, 0.01, 30),
    ...     metrics_treatment=np.random.normal(0.55, 0.01, 30)
    ... )
    >>> if result['novelty_detected']:
    ...     print("⚠️ Novelty effect detected - use caution before shipping")
    """
    if len(metrics_control) != len(metrics_treatment):
        raise ValueError("metrics_control and metrics_treatment must have same length")
    if len(metrics_control) < 10:
        raise ValueError("Need at least 10 time points for novelty detection")

    n = len(metrics_control)

    # Calculate treatment effects over time
    effects = metrics_treatment - metrics_control

    # Define early and late periods
    early_n = int(n * early_period)
    late_start = int(n * (1 - late_period))

    if early_n < 2 or (n - late_start) < 2:
        raise ValueError("early_period and late_period too small - need at least 2 points per period")

    effects_early = effects[:early_n]
    effects_late = effects[late_start:]

    # Calculate effects
    early_effect = effects_early.mean()
    late_effect = effects_late.mean()
    effect_decay = (early_effect - late_effect) / early_effect if early_effect != 0 else 0

    # CIs
    early_se = effects_early.std(ddof=1) / np.sqrt(len(effects_early))
    late_se = effects_late.std(ddof=1) / np.sqrt(len(effects_late))
    z_critical = stats.norm.ppf(1 - alpha / 2)

    early_ci = (
        early_effect - z_critical * early_se,
        early_effect + z_critical * early_se,
    )
    late_ci = (
        late_effect - z_critical * late_se,
        late_effect + z_critical * late_se,
    )

    # Test if early effect > late effect (one-sided t-test)
    t_stat, p_value = stats.ttest_ind(effects_early, effects_late, alternative='greater')

    # Novelty detected if:
    # 1. Early effect significantly > late effect (p < alpha)
    # 2. Decay is substantial (>30%)
    novelty_detected = (p_value < alpha) and (effect_decay > 0.30)

    # Recommendation
    if novelty_detected:
        recommendation = (
            "⚠️ NOVELTY EFFECT DETECTED: Early effect is significantly higher than late effect. "
            "The treatment impact may decay over time. Recommend:\n"
            "  1. Run post-launch holdout (4-8 weeks) to measure sustained effect\n"
            "  2. Monitor metrics weekly to detect continued decay\n"
            "  3. Consider shipping only if late effect is still positive and significant"
        )
    elif early_effect > 0 and late_effect > 0:
        recommendation = (
            "✅ NO NOVELTY EFFECT: Effect is sustained across early and late periods. "
            "Safe to ship based on overall results."
        )
    else:
        recommendation = (
            "⚠️ INCONCLUSIVE: Effects are mixed or negative. Need more data or investigation."
        )

    return {
        'early_effect': early_effect,
        'late_effect': late_effect,
        'effect_decay': effect_decay,
        'early_ci': early_ci,
        'late_ci': late_ci,
        'novelty_test_statistic': t_stat,
        'novelty_p_value': p_value,
        'novelty_detected': novelty_detected,
        'recommendation': recommendation,
        'early_period_days': early_n,
        'late_period_days': len(effects_late),
    }


def fit_decay_curve(
    time: np.ndarray,
    effects: np.ndarray,
    model: str = 'exponential',
) -> Dict[str, Any]:
    """
    Fit decay curve to treatment effects over time.

    Parameters
    ----------
    time : np.ndarray
        Time index (e.g., days since experiment start)
    effects : np.ndarray
        Treatment effects at each time point
    model : {'exponential', 'linear', 'log'}, default='exponential'
        Decay model to fit:
        - 'exponential': effect(t) = a * exp(-b * t) + c
        - 'linear': effect(t) = a - b * t
        - 'log': effect(t) = a - b * log(t + 1)

    Returns
    -------
    dict
        Dictionary with:
        - model: Model type
        - parameters: Fitted parameters
        - initial_effect: Effect at t=0
        - asymptotic_effect: Long-term effect (t→∞)
        - half_life: Time to reach 50% of decay (exponential only)
        - r_squared: Goodness of fit
        - predictions: Fitted values

    Example
    -------
    >>> time = np.arange(30)
    >>> effects = 0.10 * np.exp(-0.1 * time) + 0.03
    >>> result = fit_decay_curve(time, effects, model='exponential')
    >>> print(f"Half-life: {result['half_life']:.1f} days")
    """
    if len(time) != len(effects):
        raise ValueError("time and effects must have same length")
    if len(time) < 5:
        raise ValueError("Need at least 5 data points for curve fitting")

    if model == 'exponential':
        # effect(t) = a * exp(-b * t) + c
        def func(t, a, b, c):
            return a * np.exp(-b * t) + c

        # Initial guess
        p0 = [effects[0] - effects[-1], 0.1, effects[-1]]

        try:
            params, _ = curve_fit(func, time, effects, p0=p0, maxfev=10000)
            a, b, c = params

            initial_effect = a + c
            asymptotic_effect = c
            half_life = np.log(2) / b if b > 0 else np.inf
            predictions = func(time, a, b, c)

        except Exception as e:
            # Fallback to simple exponential
            params = [effects[0], 0.0, effects.mean()]
            initial_effect = effects[0]
            asymptotic_effect = effects.mean()
            half_life = np.inf
            predictions = np.full_like(effects, effects.mean())

    elif model == 'linear':
        # effect(t) = a - b * t
        slope, intercept = np.polyfit(time, effects, 1)
        params = [intercept, -slope]

        initial_effect = intercept
        asymptotic_effect = intercept + slope * time[-1]
        half_life = None  # Not applicable for linear
        predictions = intercept + slope * time

    elif model == 'log':
        # effect(t) = a - b * log(t + 1)
        def func(t, a, b):
            return a - b * np.log(t + 1)

        try:
            params, _ = curve_fit(func, time, effects, maxfev=10000)
            a, b = params

            initial_effect = a
            asymptotic_effect = a - b * np.log(time[-1] + 1)
            half_life = None  # Complex to compute
            predictions = func(time, a, b)

        except Exception:
            params = [effects[0], 0.0]
            initial_effect = effects[0]
            asymptotic_effect = effects.mean()
            half_life = None
            predictions = np.full_like(effects, effects.mean())

    else:
        raise ValueError(f"Unknown model: {model}")

    # R-squared
    ss_res = ((effects - predictions) ** 2).sum()
    ss_tot = ((effects - effects.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'model': model,
        'parameters': params,
        'initial_effect': initial_effect,
        'asymptotic_effect': asymptotic_effect,
        'half_life': half_life,
        'r_squared': r_squared,
        'predictions': predictions,
    }


def recommend_holdout_duration(
    effect_trajectory: np.ndarray,
    time_index: np.ndarray,
    min_weeks: int = 4,
    target_stability: float = 0.90,
) -> Dict[str, Any]:
    """
    Recommend post-launch holdout duration based on effect trajectory.

    Parameters
    ----------
    effect_trajectory : np.ndarray
        Time-series of treatment effects
    time_index : np.ndarray
        Time index (days)
    min_weeks : int, default=4
        Minimum holdout duration in weeks
    target_stability : float, default=0.90
        Target stability (ratio of late effect to peak effect)

    Returns
    -------
    dict
        Dictionary with:
        - recommended_days: Recommended holdout duration in days
        - recommended_weeks: Recommended holdout duration in weeks
        - current_stability: Current stability ratio
        - rationale: Explanation

    Example
    -------
    >>> effects = np.array([0.10, 0.08, 0.06, 0.05, 0.05])
    >>> time = np.arange(5)
    >>> result = recommend_holdout_duration(effects, time)
    >>> print(f"Recommend {result['recommended_weeks']} weeks post-launch holdout")
    """
    if len(effect_trajectory) < 5:
        return {
            'recommended_days': min_weeks * 7,
            'recommended_weeks': min_weeks,
            'current_stability': np.nan,
            'rationale': f"Insufficient data - use minimum {min_weeks} weeks",
        }

    # Detect peak effect
    peak_effect = effect_trajectory.max()
    current_effect = effect_trajectory[-5:].mean()  # Last 5 periods average

    # Stability ratio
    stability = current_effect / peak_effect if peak_effect > 0 else 1.0

    # Recommendations based on stability
    if stability >= target_stability:
        # Effect is stable
        rec_weeks = min_weeks
        rationale = (
            f"Effect appears stable ({stability:.1%} of peak). "
            f"Recommend minimum {min_weeks} weeks to confirm sustained impact."
        )
    elif stability >= 0.70:
        # Moderate decay
        rec_weeks = max(min_weeks, 6)
        rationale = (
            f"Moderate novelty decay detected ({stability:.1%} of peak). "
            f"Recommend {rec_weeks} weeks to ensure effect stabilizes."
        )
    else:
        # Significant decay
        rec_weeks = 8
        rationale = (
            f"Significant novelty decay ({stability:.1%} of peak). "
            f"Recommend {rec_weeks} weeks minimum. Consider not shipping if effect continues to decline."
        )

    return {
        'recommended_days': rec_weeks * 7,
        'recommended_weeks': rec_weeks,
        'current_stability': stability,
        'peak_effect': peak_effect,
        'current_effect': current_effect,
        'rationale': rationale,
    }


def cohort_analysis(
    data: pd.DataFrame,
    date_col: str,
    cohort_col: str,
    metric_col: str,
    treatment_col: str,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze novelty effect by user cohort (new vs existing users).

    Parameters
    ----------
    data : pd.DataFrame
        Experiment data
    date_col : str
        Column with dates
    cohort_col : str
        Column identifying cohort (e.g., 'is_new_user')
    metric_col : str
        Metric column
    treatment_col : str
        Treatment indicator column

    Returns
    -------
    dict
        Dictionary with:
        - new_users: DataFrame with effects for new users over time
        - existing_users: DataFrame with effects for existing users over time

    Notes
    -----
    - New users often show novelty effects
    - Existing users may show primacy effects (resistance to change)

    Example
    -------
    >>> result = cohort_analysis(
    ...     data=df,
    ...     date_col='date',
    ...     cohort_col='is_new_user',
    ...     metric_col='conversion',
    ...     treatment_col='treatment'
    ... )
    >>> print(result['new_users'].head())
    """
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in data")
    if cohort_col not in data.columns:
        raise ValueError(f"Column '{cohort_col}' not found in data")
    if metric_col not in data.columns:
        raise ValueError(f"Column '{metric_col}' not found in data")
    if treatment_col not in data.columns:
        raise ValueError(f"Column '{treatment_col}' not found in data")

    results = {}

    for cohort_value in data[cohort_col].unique():
        cohort_data = data[data[cohort_col] == cohort_value]

        # Aggregate by date and treatment
        agg = cohort_data.groupby([date_col, treatment_col])[metric_col].mean().unstack()

        # Calculate treatment effect
        if 0 in agg.columns and 1 in agg.columns:
            agg['effect'] = agg[1] - agg[0]
            agg['effect_pct'] = (agg[1] / agg[0] - 1) * 100

        cohort_name = f"cohort_{cohort_value}"
        results[cohort_name] = agg

    return results


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Novelty Effect Detection Demo")
    print("=" * 80)

    np.random.seed(42)

    # Simulate experiment with novelty effect
    print("\n📊 SIMULATING EXPERIMENT WITH NOVELTY EFFECT")
    print("-" * 80)

    n_days = 30
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    time_idx = np.arange(n_days)

    # True effect decays from 10% to 3% over 30 days (exponential decay)
    true_effect = 0.10 * np.exp(-0.1 * time_idx) + 0.03

    # Generate daily metrics
    baseline = 0.50
    metrics_control = np.random.normal(baseline, 0.01, n_days)
    metrics_treatment = metrics_control + true_effect + np.random.normal(0, 0.005, n_days)

    print(f"Generated {n_days} days of data")
    print(f"Initial effect: {true_effect[0]:.2%}")
    print(f"Final effect: {true_effect[-1]:.2%}")
    print(f"Decay: {(1 - true_effect[-1]/true_effect[0])*100:.1f}%")

    # Detect novelty effect
    print("\n🔍 DETECTING NOVELTY EFFECT")
    print("-" * 80)

    novelty_result = detect_novelty_effect(
        metrics_control=metrics_control,
        metrics_treatment=metrics_treatment,
        time_index=dates,
        early_period=0.25,
        late_period=0.25,
    )

    print(f"Early effect (days 1-{novelty_result['early_period_days']}): {novelty_result['early_effect']:.4f}")
    print(f"  95% CI: [{novelty_result['early_ci'][0]:.4f}, {novelty_result['early_ci'][1]:.4f}]")
    print(f"Late effect (last {novelty_result['late_period_days']} days): {novelty_result['late_effect']:.4f}")
    print(f"  95% CI: [{novelty_result['late_ci'][0]:.4f}, {novelty_result['late_ci'][1]:.4f}]")
    print(f"Effect decay: {novelty_result['effect_decay']*100:.1f}%")
    print(f"t-statistic: {novelty_result['novelty_test_statistic']:.4f}")
    print(f"p-value: {novelty_result['novelty_p_value']:.6f}")
    print(f"\nNovelty detected: {'✅ YES' if novelty_result['novelty_detected'] else '❌ NO'}")
    print(f"\n{novelty_result['recommendation']}")

    # Fit decay curve
    print("\n📉 FITTING DECAY CURVE")
    print("-" * 80)

    effects = metrics_treatment - metrics_control
    decay_fit = fit_decay_curve(time_idx, effects, model='exponential')

    print(f"Model: {decay_fit['model']}")
    print(f"Parameters: a={decay_fit['parameters'][0]:.4f}, b={decay_fit['parameters'][1]:.4f}, c={decay_fit['parameters'][2]:.4f}")
    print(f"Initial effect (t=0): {decay_fit['initial_effect']:.4f}")
    print(f"Asymptotic effect (t→∞): {decay_fit['asymptotic_effect']:.4f}")
    print(f"Half-life: {decay_fit['half_life']:.1f} days")
    print(f"R²: {decay_fit['r_squared']:.4f}")

    # Recommend holdout duration
    print("\n⏱️ POST-LAUNCH HOLDOUT RECOMMENDATION")
    print("-" * 80)

    holdout_rec = recommend_holdout_duration(
        effect_trajectory=effects,
        time_index=time_idx,
        min_weeks=4,
        target_stability=0.90,
    )

    print(f"Peak effect: {holdout_rec['peak_effect']:.4f}")
    print(f"Current effect: {holdout_rec['current_effect']:.4f}")
    print(f"Stability: {holdout_rec['current_stability']:.1%}")
    print(f"\nRecommended holdout: {holdout_rec['recommended_weeks']} weeks ({holdout_rec['recommended_days']} days)")
    print(f"Rationale: {holdout_rec['rationale']}")

    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS:")
    print("   - Novelty effects are common in product experiments")
    print("   - Always compare early vs late periods to detect decay")
    print("   - Use post-launch holdouts (4-8 weeks) to measure sustained impact")
    print("   - Don't ship based on week 1 results alone!")
    print("=" * 80)
