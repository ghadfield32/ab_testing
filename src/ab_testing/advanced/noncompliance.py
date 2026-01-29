"""
Noncompliance Analysis and Instrumental Variables
==================================================

Handling experiments where not all users assigned to treatment actually receive it
(e.g., email campaigns, encouragement designs, optional feature rollouts).

Key Concepts:
- **ITT (Intent-to-Treat)**: Effect of assignment (unbiased, conservative)
- **Per-Protocol**: Effect on compliers only (biased if compliance is selective)
- **CACE/LATE**: Complier Average Causal Effect / Local Average Treatment Effect
- **Instrumental Variables**: Using assignment as instrument for treatment

Reference:
----------
- Spotify Engineering (2023): "Encouragement Designs and Instrumental Variables"
  https://engineering.atspotify.com/2023/08/encouragement-designs-and-instrumental-variables-for-a-b-testing
- Imbens & Angrist (1994): "Identification and Estimation of Local Average Treatment Effects"

Example Usage:
--------------
>>> from ab_testing.advanced import noncompliance
>>> import numpy as np
>>>
>>> # Simulate noncompliance
>>> assigned = np.random.binomial(1, 0.5, 1000)
>>> treated = assigned.copy()
>>> treated[assigned == 1] = np.random.binomial(1, 0.7, (assigned == 1).sum())  # 70% compliance
>>> outcome = treated * 0.1 + np.random.normal(0, 1, 1000)
>>>
>>> result = noncompliance.itt_analysis(assigned, outcome)
>>> cace = noncompliance.compute_cace(result['itt_effect'], 0.7, 0.0)
>>> print(f"ITT: {result['itt_effect']:.4f}, CACE: {cace:.4f}")
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def itt_analysis(
    assigned_treatment: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Intent-to-Treat (ITT) analysis.

    Analyzes effect based on treatment ASSIGNMENT, regardless of actual receipt.
    This is the gold standard - preserves randomization and is unbiased.

    Parameters
    ----------
    assigned_treatment : np.ndarray
        Binary array (0/1) indicating treatment assignment
    outcome : np.ndarray
        Outcome metric
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with:
        - itt_effect: Intent-to-treat effect
        - se: Standard error
        - t_statistic: Test statistic
        - p_value: P-value
        - ci_lower: Lower CI bound
        - ci_upper: Upper CI bound
        - n_control: Sample size control
        - n_treatment: Sample size treatment

    Notes
    -----
    - ALWAYS report ITT as primary analysis
    - Unbiased estimate of effect of being OFFERED treatment
    - Conservative if compliance < 100%

    Example
    -------
    >>> assigned = np.random.binomial(1, 0.5, 1000)
    >>> outcome = assigned * 0.05 + np.random.normal(0, 1, 1000)
    >>> result = itt_analysis(assigned, outcome)
    >>> print(f"ITT effect: {result['itt_effect']:.4f}")
    """
    if len(assigned_treatment) != len(outcome):
        raise ValueError("assigned_treatment and outcome must have same length")
    if not np.all(np.isin(assigned_treatment, [0, 1])):
        raise ValueError("assigned_treatment must be binary (0/1)")

    control_mask = assigned_treatment == 0
    treatment_mask = assigned_treatment == 1

    y_control = outcome[control_mask]
    y_treatment = outcome[treatment_mask]

    itt_effect = y_treatment.mean() - y_control.mean()
    se = np.sqrt(y_control.var(ddof=1)/len(y_control) +
                 y_treatment.var(ddof=1)/len(y_treatment))

    t_stat = itt_effect / se
    df = len(outcome) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    t_critical = stats.t.ppf(1 - alpha/2, df=df)
    ci_lower = itt_effect - t_critical * se
    ci_upper = itt_effect + t_critical * se

    return {
        'itt_effect': itt_effect,
        'se': se,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_control': len(y_control),
        'n_treatment': len(y_treatment),
    }


def per_protocol_analysis(
    actually_treated: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Per-Protocol (PP) analysis.

    Analyzes effect based on actual treatment RECEIPT, not assignment.

    ⚠️ WARNING: BIASED if compliance is selective (correlated with outcomes).
    Only valid if compliance is random.

    Parameters
    ----------
    actually_treated : np.ndarray
        Binary array (0/1) indicating actual treatment receipt
    outcome : np.ndarray
        Outcome metric
    alpha : float
        Significance level

    Returns
    -------
    dict
        Dictionary with PP effect estimates

    Notes
    -----
    - BIASED if compliance depends on user characteristics
    - Report as secondary, not primary
    - Use CACE instead for causal effect on compliers

    Example
    -------
    >>> treated = np.random.binomial(1, 0.5, 1000)
    >>> outcome = treated * 0.1 + np.random.normal(0, 1, 1000)
    >>> result = per_protocol_analysis(treated, outcome)
    >>> print(f"Per-protocol effect: {result['pp_effect']:.4f}")
    """
    if len(actually_treated) != len(outcome):
        raise ValueError("actually_treated and outcome must have same length")
    if not np.all(np.isin(actually_treated, [0, 1])):
        raise ValueError("actually_treated must be binary (0/1)")

    control_mask = actually_treated == 0
    treatment_mask = actually_treated == 1

    y_control = outcome[control_mask]
    y_treatment = outcome[treatment_mask]

    pp_effect = y_treatment.mean() - y_control.mean()
    se = np.sqrt(y_control.var(ddof=1)/len(y_control) +
                 y_treatment.var(ddof=1)/len(y_treatment))

    t_stat = pp_effect / se
    df = len(outcome) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    t_critical = stats.t.ppf(1 - alpha/2, df=df)
    ci_lower = pp_effect - t_critical * se
    ci_upper = pp_effect + t_critical * se

    return {
        'pp_effect': pp_effect,
        'se': se,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_control': len(y_control),
        'n_treatment': len(y_treatment),
    }


def compute_cace(
    itt_effect: float,
    compliance_rate_treatment: float,
    compliance_rate_control: float = 0.0,
) -> float:
    """
    Compute Complier Average Causal Effect (CACE) / Local Average Treatment Effect (LATE).

    CACE is the causal effect for users who would comply if assigned to treatment.

    Formula:
    CACE = ITT / (compliance_rate_treatment - compliance_rate_control)

    Parameters
    ----------
    itt_effect : float
        Intent-to-treat effect
    compliance_rate_treatment : float
        Proportion who comply when assigned to treatment
    compliance_rate_control : float, default=0.0
        Proportion who get treatment when assigned to control (crossover)

    Returns
    -------
    float
        CACE estimate

    Notes
    -----
    - CACE identifies effect on compliers only
    - Requires 3 assumptions:
      1. Monotonicity: No defiers (users hurt by being offered treatment)
      2. Exclusion restriction: Assignment only affects outcome through treatment
      3. Relevance: Assignment affects treatment receipt
    - CACE > ITT when compliance < 100%

    Reference
    ---------
    Imbens & Angrist (1994): "Identification and Estimation of LATE"

    Example
    -------
    >>> # ITT = 5%, compliance = 70% -> CACE = 7.1%
    >>> cace = compute_cace(itt_effect=0.05, compliance_rate_treatment=0.70)
    >>> print(f"CACE: {cace:.4f}")
    """
    compliance_diff = compliance_rate_treatment - compliance_rate_control

    if compliance_diff <= 0:
        raise ValueError("compliance_rate_treatment must be > compliance_rate_control")
    if compliance_rate_treatment < 0 or compliance_rate_treatment > 1:
        raise ValueError("compliance_rate_treatment must be in [0, 1]")
    if compliance_rate_control < 0 or compliance_rate_control > 1:
        raise ValueError("compliance_rate_control must be in [0, 1]")

    cace = itt_effect / compliance_diff

    return cace


def iv_estimation(
    assigned_treatment: np.ndarray,  # Instrument Z
    actually_treated: np.ndarray,    # Treatment D
    outcome: np.ndarray,              # Outcome Y
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Instrumental Variables (IV) estimation for CACE.

    Uses assignment (Z) as instrument for actual treatment (D) to estimate
    causal effect on compliers.

    Two-Stage Least Squares (2SLS):
    1. First stage: D = α + βZ + ε
    2. Second stage: Y = γ + δD̂ + η

    Parameters
    ----------
    assigned_treatment : np.ndarray
        Binary assignment (instrument Z)
    actually_treated : np.ndarray
        Binary actual treatment (D)
    outcome : np.ndarray
        Outcome metric (Y)
    alpha : float
        Significance level

    Returns
    -------
    dict
        Dictionary with:
        - cace: Complier average causal effect
        - se: Standard error of CACE
        - t_statistic: Test statistic
        - p_value: P-value
        - ci_lower: Lower CI bound
        - ci_upper: Upper CI bound
        - first_stage_f: First-stage F-statistic (instrument strength)
        - compliance_rate: Compliance rate in treatment group

    Notes
    -----
    - First-stage F > 10 indicates strong instrument
    - Requires same 3 assumptions as CACE

    Example
    -------
    >>> assigned = np.random.binomial(1, 0.5, 1000)
    >>> treated = assigned.copy()
    >>> treated[assigned == 1] = np.random.binomial(1, 0.7, (assigned == 1).sum())
    >>> outcome = treated * 0.1 + np.random.normal(0, 1, 1000)
    >>> result = iv_estimation(assigned, treated, outcome)
    >>> print(f"CACE: {result['cace']:.4f}, First-stage F: {result['first_stage_f']:.2f}")
    """
    if len(assigned_treatment) != len(actually_treated) or len(assigned_treatment) != len(outcome):
        raise ValueError("All arrays must have same length")
    if not np.all(np.isin(assigned_treatment, [0, 1])):
        raise ValueError("assigned_treatment must be binary")
    if not np.all(np.isin(actually_treated, [0, 1])):
        raise ValueError("actually_treated must be binary")

    # First stage: D ~ Z
    # E[D|Z=1] - E[D|Z=0] = compliance rate difference
    compliance_treatment = actually_treated[assigned_treatment == 1].mean()
    compliance_control = actually_treated[assigned_treatment == 0].mean()
    first_stage_coef = compliance_treatment - compliance_control

    # First-stage F-statistic (simplified)
    # Full 2SLS would use regression, this is the simplified version
    n = len(outcome)
    first_stage_se = np.sqrt(
        compliance_treatment * (1 - compliance_treatment) / (assigned_treatment == 1).sum() +
        compliance_control * (1 - compliance_control) / (assigned_treatment == 0).sum()
    )
    first_stage_f = (first_stage_coef / first_stage_se) ** 2

    # Reduced form: Y ~ Z (ITT)
    reduced_form = outcome[assigned_treatment == 1].mean() - outcome[assigned_treatment == 0].mean()

    # IV estimate (Wald estimator)
    if abs(first_stage_coef) < 1e-10:
        raise ValueError("First stage is zero - instrument is irrelevant")

    cace = reduced_form / first_stage_coef

    # Standard error (simplified - full 2SLS would be more accurate)
    # Using delta method approximation
    var_reduced_form = (
        outcome[assigned_treatment == 1].var(ddof=1) / (assigned_treatment == 1).sum() +
        outcome[assigned_treatment == 0].var(ddof=1) / (assigned_treatment == 0).sum()
    )
    se = np.sqrt(var_reduced_form) / abs(first_stage_coef)

    t_stat = cace / se
    df = n - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    t_critical = stats.t.ppf(1 - alpha/2, df=df)
    ci_lower = cace - t_critical * se
    ci_upper = cace + t_critical * se

    return {
        'cace': cace,
        'se': se,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'first_stage_f': first_stage_f,
        'compliance_rate': compliance_treatment,
        'compliance_diff': first_stage_coef,
    }


def noncompliance_summary(
    assigned_treatment: np.ndarray,
    actually_treated: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Comprehensive noncompliance analysis.

    Computes ITT, Per-Protocol, and CACE/LATE estimates.

    Parameters
    ----------
    assigned_treatment : np.ndarray
        Binary assignment
    actually_treated : np.ndarray
        Binary actual treatment
    outcome : np.ndarray
        Outcome metric
    alpha : float
        Significance level

    Returns
    -------
    dict
        Comprehensive summary with all estimates

    Example
    -------
    >>> assigned = np.random.binomial(1, 0.5, 1000)
    >>> treated = assigned.copy()
    >>> treated[assigned == 1] = np.random.binomial(1, 0.7, (assigned == 1).sum())
    >>> outcome = treated * 0.1 + np.random.normal(0, 1, 1000)
    >>> summary = noncompliance_summary(assigned, treated, outcome)
    >>> print(f"ITT: {summary['itt']['itt_effect']:.4f}")
    >>> print(f"CACE: {summary['iv']['cace']:.4f}")
    """
    # ITT analysis
    itt_result = itt_analysis(assigned_treatment, outcome, alpha=alpha)

    # Per-protocol analysis
    pp_result = per_protocol_analysis(actually_treated, outcome, alpha=alpha)

    # IV/CACE analysis
    iv_result = iv_estimation(assigned_treatment, actually_treated, outcome, alpha=alpha)

    # Compliance statistics
    compliance_treatment = actually_treated[assigned_treatment == 1].mean()
    compliance_control = actually_treated[assigned_treatment == 0].mean()

    return {
        'itt': itt_result,
        'per_protocol': pp_result,
        'iv': iv_result,
        'compliance': {
            'treatment_group': compliance_treatment,
            'control_group': compliance_control,
            'difference': compliance_treatment - compliance_control,
        },
        'interpretation': {
            'primary': 'ITT (unbiased)',
            'complier_effect': 'CACE/IV (effect on compliers)',
            'warning': 'Per-protocol is biased if compliance is selective',
        }
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Noncompliance Analysis with Instrumental Variables Demo")
    print("=" * 80)

    np.random.seed(42)
    n = 1000

    # Simulate encouragement design (e.g., email campaign)
    # Assignment (Z): 50/50 randomization
    assigned = np.random.binomial(1, 0.5, n)

    # Actual treatment (D): Not everyone assigned opens the email
    # Compliance in treatment group: 70%
    # Crossover in control group: 5% (organically see it)
    actually_treated = np.zeros(n)
    actually_treated[assigned == 1] = np.random.binomial(1, 0.70, (assigned == 1).sum())
    actually_treated[assigned == 0] = np.random.binomial(1, 0.05, (assigned == 0).sum())

    # Outcome: True effect is 0.10 for those who actually get treated
    outcome = 50 + actually_treated * 10 + np.random.normal(0, 20, n)

    # Run comprehensive analysis
    summary = noncompliance_summary(assigned, actually_treated, outcome)

    print("\n📊 COMPLIANCE STATISTICS")
    print("-" * 80)
    print(f"Treatment group compliance: {summary['compliance']['treatment_group']:.1%}")
    print(f"Control group crossover: {summary['compliance']['control_group']:.1%}")
    print(f"Compliance difference: {summary['compliance']['difference']:.1%}")

    print("\n📊 INTENT-TO-TREAT (ITT) - PRIMARY ANALYSIS")
    print("-" * 80)
    itt = summary['itt']
    print(f"ITT effect: {itt['itt_effect']:.4f}")
    print(f"SE: {itt['se']:.4f}")
    print(f"P-value: {itt['p_value']:.4f}")
    print(f"95% CI: ({itt['ci_lower']:.4f}, {itt['ci_upper']:.4f})")
    print("✅ Unbiased estimate of effect of ASSIGNMENT")

    print("\n📊 PER-PROTOCOL ANALYSIS - BIASED!")
    print("-" * 80)
    pp = summary['per_protocol']
    print(f"Per-protocol effect: {pp['pp_effect']:.4f}")
    print(f"SE: {pp['se']:.4f}")
    print(f"P-value: {pp['p_value']:.4f}")
    print("⚠️ BIASED if compliance is selective - don't trust this!")

    print("\n📊 CACE/LATE (Instrumental Variables) - COMPLIER EFFECT")
    print("-" * 80)
    iv = summary['iv']
    print(f"CACE: {iv['cace']:.4f}")
    print(f"SE: {iv['se']:.4f}")
    print(f"P-value: {iv['p_value']:.4f}")
    print(f"95% CI: ({iv['ci_lower']:.4f}, {iv['ci_upper']:.4f})")
    print(f"First-stage F: {iv['first_stage_f']:.2f} {'✅ Strong' if iv['first_stage_f'] > 10 else '⚠️ Weak'}")
    print("✅ Causal effect for COMPLIERS (users who would comply if assigned)")

    print("\n💡 INTERPRETATION")
    print("-" * 80)
    print(f"ITT = {itt['itt_effect']:.2f}: Effect of being OFFERED treatment")
    print(f"CACE = {iv['cace']:.2f}: Effect for users who ACTUALLY GET treated")
    print(f"CACE > ITT because compliance < 100%")
    print()
    print("RECOMMENDATION: Report ITT as primary, CACE as secondary")
