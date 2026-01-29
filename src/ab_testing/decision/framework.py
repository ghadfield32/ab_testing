"""
Decision Framework for A/B Test Results
========================================

Structured decision-making for experiment outcomes with clear ship/hold/abandon logic.

Decision Matrix:
- **SHIP**: Primary significant & positive & guardrails pass
- **HOLD**: Inconclusive results or minor guardrail issues
- **ABANDON**: Primary negative OR critical guardrail failure

Example Usage:
--------------
>>> from ab_testing.decision import framework
>>>
>>> decision = framework.make_decision(
...     primary_significant=True,
...     primary_positive=True,
...     guardrails_passed=True,
...     effect_size=0.05,
...     business_significant=True
... )
>>> print(decision['decision'])  # 'ship'
>>> print(decision['rationale'])
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric test results."""
    name: str
    significant: bool
    effect: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class GuardrailResult:
    """Container for guardrail test results."""
    name: str
    passed: bool
    degradation_pct: float
    threshold_pct: float


def make_decision(
    primary_significant: bool,
    primary_positive: bool,
    guardrails_passed: bool,
    effect_size: float,
    business_significant: bool = True,
    min_effect_threshold: float = 0.01,
) -> Dict[str, any]:
    """
    Make structured experiment decision.

    Parameters
    ----------
    primary_significant : bool
        Whether primary metric is statistically significant
    primary_positive : bool
        Whether primary metric effect is positive
    guardrails_passed : bool
        Whether all guardrail metrics passed
    effect_size : float
        Observed effect size
    business_significant : bool, default=True
        Whether effect is large enough to matter
    min_effect_threshold : float, default=0.01
        Minimum effect to consider business-significant

    Returns
    -------
    dict
        Dictionary with:
        - decision: 'ship', 'hold', or 'abandon'
        - rationale: Explanation of decision
        - confidence: 'high', 'medium', or 'low'
        - next_steps: Recommended actions

    Example
    -------
    >>> decision = make_decision(
    ...     primary_significant=True,
    ...     primary_positive=True,
    ...     guardrails_passed=True,
    ...     effect_size=0.05,
    ...     business_significant=True
    ... )
    >>> print(decision['decision'])
    """
    # Decision logic
    if primary_significant and primary_positive and guardrails_passed:
        if abs(effect_size) >= min_effect_threshold and business_significant:
            decision = 'ship'
            confidence = 'high'
            rationale = (
                "Primary metric shows statistically and business-significant positive effect. "
                "All guardrails passed. Safe to ship."
            )
            next_steps = [
                "Ship to 100% of users",
                "Monitor metrics for 1-2 weeks post-launch",
                "Document learnings"
            ]
        else:
            decision = 'hold'
            confidence = 'medium'
            rationale = (
                "Primary metric is statistically significant but effect size is too small "
                "to justify implementation costs."
            )
            next_steps = [
                "Iterate on treatment to increase effect size",
                "Re-evaluate business case",
                "Consider alternative approaches"
            ]

    elif primary_significant and not primary_positive:
        decision = 'abandon'
        confidence = 'high'
        rationale = "Primary metric shows significant NEGATIVE effect. Do not ship."
        next_steps = [
            "Analyze why treatment underperformed",
            "Document insights",
            "Design improved variant"
        ]

    elif not primary_significant and primary_positive:
        decision = 'hold'
        confidence = 'low'
        rationale = (
            "Primary metric trends positive but is not statistically significant. "
            "Insufficient evidence to ship."
        )
        next_steps = [
            "Extend experiment to gather more data",
            "Check power analysis - was sample size sufficient?",
            "Consider whether effect exists but is smaller than expected"
        ]

    elif not guardrails_passed:
        decision = 'hold'
        confidence = 'medium'
        rationale = (
            "One or more guardrail metrics failed non-inferiority test. "
            "Primary gains may come at unacceptable cost."
        )
        next_steps = [
            "Investigate which guardrails failed and why",
            "Determine if degradation is acceptable given primary gains",
            "Consider redesigning treatment to avoid guardrail harm"
        ]

    else:
        decision = 'abandon'
        confidence = 'medium'
        rationale = "No significant effect detected on primary metric."
        next_steps = [
            "Analyze why treatment had no effect",
            "Check implementation - did users actually see treatment?",
            "Consider alternative hypotheses"
        ]

    return {
        'decision': decision,
        'rationale': rationale,
        'confidence': confidence,
        'next_steps': next_steps,
    }


def comprehensive_decision(
    primary_metrics: List[MetricResult],
    guardrail_metrics: List[GuardrailResult],
    business_context: Optional[Dict[str, any]] = None,
) -> Dict[str, any]:
    """
    Comprehensive decision incorporating multiple metrics and business context.

    Parameters
    ----------
    primary_metrics : list of MetricResult
        Primary metrics (usually 1, max 2-3)
    guardrail_metrics : list of GuardrailResult
        Guardrail metrics
    business_context : dict, optional
        Additional business considerations (ROI, implementation cost, etc.)

    Returns
    -------
    dict
        Comprehensive decision with detailed breakdown

    Example
    -------
    >>> from ab_testing.decision.framework import MetricResult, GuardrailResult
    >>>
    >>> primary = [MetricResult(
    ...     name='conversion',
    ...     significant=True,
    ...     effect=0.02,
    ...     p_value=0.001,
    ...     ci_lower=0.01,
    ...     ci_upper=0.03
    ... )]
    >>> guardrails = [GuardrailResult(
    ...     name='revenue',
    ...     passed=True,
    ...     degradation_pct=-1.5,
    ...     threshold_pct=-5.0
    ... )]
    >>> decision = comprehensive_decision(primary, guardrails)
    """
    # Check primary metrics
    primary_sig = all(m.significant for m in primary_metrics)
    primary_positive = all(m.effect > 0 for m in primary_metrics)

    # Check guardrails
    guardrails_passed = all(g.passed for g in guardrail_metrics)
    failed_guardrails = [g.name for g in guardrail_metrics if not g.passed]

    # Average effect size across primaries
    avg_effect = sum(m.effect for m in primary_metrics) / len(primary_metrics) if primary_metrics else 0

    # Business significance check
    if business_context:
        min_roi = business_context.get('min_roi', 1.0)
        estimated_roi = business_context.get('estimated_roi', 0)
        business_significant = estimated_roi >= min_roi
    else:
        business_significant = True

    # Make decision
    base_decision = make_decision(
        primary_significant=primary_sig,
        primary_positive=primary_positive,
        guardrails_passed=guardrails_passed,
        effect_size=avg_effect,
        business_significant=business_significant
    )

    # Add detailed breakdown
    return {
        **base_decision,
        'primary_metrics': [
            {
                'name': m.name,
                'effect': m.effect,
                'significant': m.significant,
                'p_value': m.p_value,
                'ci': (m.ci_lower, m.ci_upper)
            }
            for m in primary_metrics
        ],
        'guardrails': [
            {
                'name': g.name,
                'passed': g.passed,
                'degradation_pct': g.degradation_pct,
                'threshold_pct': g.threshold_pct
            }
            for g in guardrail_metrics
        ],
        'failed_guardrails': failed_guardrails,
        'business_context': business_context or {},
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Decision Framework Demo")
    print("=" * 80)

    # Scenario 1: Clear winner
    print("\n📊 SCENARIO 1: Clear Winner")
    print("-" * 80)
    decision1 = make_decision(
        primary_significant=True,
        primary_positive=True,
        guardrails_passed=True,
        effect_size=0.05,
        business_significant=True
    )
    print(f"Decision: {decision1['decision'].upper()}")
    print(f"Confidence: {decision1['confidence'].upper()}")
    print(f"Rationale: {decision1['rationale']}")
    print("Next steps:")
    for step in decision1['next_steps']:
        print(f"  • {step}")

    # Scenario 2: Guardrail failure
    print("\n📊 SCENARIO 2: Guardrail Failure")
    print("-" * 80)
    decision2 = make_decision(
        primary_significant=True,
        primary_positive=True,
        guardrails_passed=False,
        effect_size=0.03,
        business_significant=True
    )
    print(f"Decision: {decision2['decision'].upper()}")
    print(f"Confidence: {decision2['confidence'].upper()}")
    print(f"Rationale: {decision2['rationale']}")

    # Scenario 3: Comprehensive decision
    print("\n📊 SCENARIO 3: Comprehensive Decision with Multiple Metrics")
    print("-" * 80)

    primary = [
        MetricResult(
            name='conversion_rate',
            significant=True,
            effect=0.02,
            p_value=0.001,
            ci_lower=0.01,
            ci_upper=0.03
        ),
        MetricResult(
            name='engagement',
            significant=True,
            effect=0.05,
            p_value=0.003,
            ci_lower=0.02,
            ci_upper=0.08
        )
    ]

    guardrails = [
        GuardrailResult(
            name='revenue_per_user',
            passed=True,
            degradation_pct=-1.5,
            threshold_pct=-5.0
        ),
        GuardrailResult(
            name='retention_7d',
            passed=True,
            degradation_pct=-0.5,
            threshold_pct=-5.0
        )
    ]

    business_context = {
        'min_roi': 1.5,
        'estimated_roi': 2.3,
        'implementation_cost': 50000,
        'annual_value': 115000
    }

    decision3 = comprehensive_decision(primary, guardrails, business_context)

    print(f"Decision: {decision3['decision'].upper()}")
    print(f"Confidence: {decision3['confidence'].upper()}")
    print(f"\nPrimary Metrics:")
    for m in decision3['primary_metrics']:
        print(f"  • {m['name']}: {m['effect']:+.2%} (p={m['p_value']:.4f})")

    print(f"\nGuardrails:")
    for g in decision3['guardrails']:
        status = '✅ PASS' if g['passed'] else '❌ FAIL'
        print(f"  • {g['name']}: {g['degradation_pct']:+.1f}% vs {g['threshold_pct']:.1f}% threshold {status}")

    if decision3['business_context']:
        print(f"\nBusiness Context:")
        print(f"  • Estimated ROI: {decision3['business_context']['estimated_roi']:.1f}x")
        print(f"  • Annual Value: ${decision3['business_context']['annual_value']:,}")

    print(f"\nRationale: {decision3['rationale']}")
