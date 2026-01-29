"""
Business Impact Translation
============================

Translate statistical results into business metrics for executive communication.

Converts experiment results into:
- Annual revenue impact
- ROI calculations
- Customer lifetime value (LTV) impact
- User acquisition equivalent

Example Usage:
--------------
>>> from ab_testing.decision import business_impact
>>>
>>> impact = business_impact.calculate_annual_impact(
...     effect=0.02,  # 2pp lift in conversion
...     annual_users=10_000_000,
...     value_per_conversion=150
... )
>>> print(f"Annual value: ${impact['annual_value']:,.0f}")
"""

from typing import Dict, Optional
import numpy as np


def calculate_annual_impact(
    effect: float,
    annual_users: int,
    value_per_conversion: float,
    baseline_rate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate annualized business impact from experiment effect.

    Parameters
    ----------
    effect : float
        Absolute effect (e.g., 0.02 for 2 percentage point lift)
    annual_users : int
        Expected annual users exposed to treatment
    value_per_conversion : float
        Value per conversion (LTV, revenue, etc.)
    baseline_rate : float, optional
        Baseline conversion rate (for relative lift calculation)

    Returns
    -------
    dict
        Dictionary with:
        - additional_conversions: Extra conversions per year
        - annual_value: Annual revenue/value impact
        - monthly_value: Monthly value
        - relative_lift: Relative lift percentage (if baseline provided)

    Example
    -------
    >>> impact = calculate_annual_impact(
    ...     effect=0.02,  # 2pp lift
    ...     annual_users=10_000_000,
    ...     value_per_conversion=150
    ... )
    >>> print(f"Annual impact: ${impact['annual_value']:,.0f}")
    """
    additional_conversions = effect * annual_users
    annual_value = additional_conversions * value_per_conversion
    monthly_value = annual_value / 12

    result = {
        'additional_conversions': additional_conversions,
        'annual_value': annual_value,
        'monthly_value': monthly_value,
    }

    if baseline_rate is not None:
        relative_lift = effect / baseline_rate if baseline_rate > 0 else np.nan
        result['relative_lift'] = relative_lift
        result['baseline_rate'] = baseline_rate
        result['new_rate'] = baseline_rate + effect

    return result


def calculate_roi(
    annual_value: float,
    implementation_cost: float,
    annual_maintenance_cost: float = 0,
    time_horizon_years: int = 1,
) -> Dict[str, float]:
    """
    Calculate ROI for experiment treatment.

    Parameters
    ----------
    annual_value : float
        Annual value from treatment
    implementation_cost : float
        One-time implementation cost
    annual_maintenance_cost : float, default=0
        Annual ongoing maintenance cost
    time_horizon_years : int, default=1
        Time horizon for ROI calculation

    Returns
    -------
    dict
        Dictionary with:
        - total_value: Total value over time horizon
        - total_cost: Total costs
        - net_value: Net value (total_value - total_cost)
        - roi: Return on investment (net_value / total_cost)
        - payback_months: Months to break even

    Example
    -------
    >>> roi = calculate_roi(
    ...     annual_value=500000,
    ...     implementation_cost=100000,
    ...     annual_maintenance_cost=20000
    ... )
    >>> print(f"ROI: {roi['roi']:.1f}x")
    >>> print(f"Payback: {roi['payback_months']:.1f} months")
    """
    total_value = annual_value * time_horizon_years
    total_cost = implementation_cost + (annual_maintenance_cost * time_horizon_years)
    net_value = total_value - total_cost
    roi = net_value / total_cost if total_cost > 0 else np.inf

    # Payback months
    if annual_value > annual_maintenance_cost:
        monthly_net = (annual_value - annual_maintenance_cost) / 12
        payback_months = implementation_cost / monthly_net if monthly_net > 0 else np.inf
    else:
        payback_months = np.inf

    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'net_value': net_value,
        'roi': roi,
        'payback_months': payback_months,
        'time_horizon_years': time_horizon_years,
    }


def ltv_impact(
    effect: float,
    annual_users: int,
    ltv: float,
    churn_rate: float = 0.20,
) -> Dict[str, float]:
    """
    Calculate lifetime value impact from improved conversion/retention.

    Parameters
    ----------
    effect : float
        Absolute effect on conversion or retention
    annual_users : int
        Annual users
    ltv : float
        Average customer lifetime value
    churn_rate : float, default=0.20
        Annual churn rate (for retention improvements)

    Returns
    -------
    dict
        Dictionary with LTV impact calculations

    Example
    -------
    >>> impact = ltv_impact(
    ...     effect=0.05,  # 5pp retention improvement
    ...     annual_users=1_000_000,
    ...     ltv=500,
    ...     churn_rate=0.20
    ... )
    """
    additional_customers = effect * annual_users
    ltv_impact_value = additional_customers * ltv

    # If this is a retention improvement, calculate extended LTV
    if churn_rate > 0:
        avg_lifetime_months = 1 / (churn_rate / 12)
        extended_lifetime_months = avg_lifetime_months * (1 + effect)
        lifetime_extension = extended_lifetime_months - avg_lifetime_months
    else:
        lifetime_extension = 0

    return {
        'additional_customers': additional_customers,
        'ltv_impact': ltv_impact_value,
        'avg_ltv': ltv,
        'lifetime_extension_months': lifetime_extension,
    }


def user_acquisition_equivalent(
    annual_value: float,
    cost_per_acquisition: float,
) -> Dict[str, float]:
    """
    Translate value into user acquisition equivalent.

    "This improvement is worth the same as acquiring X new users."

    Parameters
    ----------
    annual_value : float
        Annual value from experiment
    cost_per_acquisition : float
        Average cost to acquire one user (CAC)

    Returns
    -------
    dict
        User acquisition equivalent

    Example
    -------
    >>> equiv = user_acquisition_equivalent(
    ...     annual_value=500000,
    ...     cost_per_acquisition=50
    ... )
    >>> print(f"Equivalent to acquiring {equiv['equivalent_users']:,.0f} users")
    """
    equivalent_users = annual_value / cost_per_acquisition if cost_per_acquisition > 0 else np.inf

    return {
        'annual_value': annual_value,
        'cost_per_acquisition': cost_per_acquisition,
        'equivalent_users': equivalent_users,
        'equivalent_budget': annual_value,
    }


def executive_summary(
    effect: float,
    annual_users: int,
    value_per_conversion: float,
    implementation_cost: float,
    baseline_rate: Optional[float] = None,
    experiment_name: str = "Treatment",
) -> str:
    """
    Generate executive summary for experiment results.

    Parameters
    ----------
    effect : float
        Absolute effect
    annual_users : int
        Annual users
    value_per_conversion : float
        Value per conversion
    implementation_cost : float
        Implementation cost
    baseline_rate : float, optional
        Baseline rate
    experiment_name : str
        Name of experiment

    Returns
    -------
    str
        Executive summary text

    Example
    -------
    >>> summary = executive_summary(
    ...     effect=0.02,
    ...     annual_users=10_000_000,
    ...     value_per_conversion=150,
    ...     implementation_cost=100_000,
    ...     baseline_rate=0.10,
    ...     experiment_name="New Checkout Flow"
    ... )
    >>> print(summary)
    """
    # Calculate impact
    impact = calculate_annual_impact(
        effect, annual_users, value_per_conversion, baseline_rate
    )

    # Calculate ROI
    roi_result = calculate_roi(
        annual_value=impact['annual_value'],
        implementation_cost=implementation_cost
    )

    # Build summary
    summary = f"**Executive Summary: {experiment_name}**\n\n"

    if baseline_rate:
        summary += (
            f"The {experiment_name} increased conversion from {baseline_rate:.1%} to "
            f"{impact['new_rate']:.1%} (a {impact['relative_lift']*100:+.1f}% relative lift).\n\n"
        )
    else:
        summary += (
            f"The {experiment_name} shows an effect of {effect*100:+.2f} percentage points.\n\n"
        )

    summary += (
        f"**Business Impact:**\n"
        f"- Additional conversions: {impact['additional_conversions']:,.0f} per year\n"
        f"- Annual value: ${impact['annual_value']:,.0f}\n"
        f"- Monthly value: ${impact['monthly_value']:,.0f}\n\n"
    )

    summary += (
        f"**ROI:**\n"
        f"- Implementation cost: ${implementation_cost:,.0f}\n"
        f"- ROI: {roi_result['roi']:.1f}x\n"
        f"- Payback period: {roi_result['payback_months']:.1f} months\n\n"
    )

    summary += (
        f"**Recommendation:** "
        f"{'Ship this treatment - strong positive ROI.' if roi_result['roi'] > 2 else 'Evaluate implementation costs vs benefits.'}"
    )

    return summary


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Business Impact Translation Demo")
    print("=" * 80)

    # Scenario: New onboarding flow
    print("\n📊 SCENARIO: New Onboarding Flow")
    print("-" * 80)
    print("Effect: +2 percentage points in conversion")
    print("Baseline: 10% conversion rate")
    print("Annual users: 10,000,000")
    print("Value per conversion: $150 (average LTV)")
    print()

    # Calculate impact
    impact = calculate_annual_impact(
        effect=0.02,
        annual_users=10_000_000,
        value_per_conversion=150,
        baseline_rate=0.10
    )

    print("📈 ANNUAL IMPACT")
    print("-" * 80)
    print(f"Baseline conversion: {impact['baseline_rate']:.1%}")
    print(f"New conversion: {impact['new_rate']:.1%}")
    print(f"Relative lift: {impact['relative_lift']*100:+.1f}%")
    print()
    print(f"Additional conversions: {impact['additional_conversions']:,.0f} per year")
    print(f"Annual value: ${impact['annual_value']:,.0f}")
    print(f"Monthly value: ${impact['monthly_value']:,.0f}")

    # Calculate ROI
    roi = calculate_roi(
        annual_value=impact['annual_value'],
        implementation_cost=100_000,
        annual_maintenance_cost=20_000,
        time_horizon_years=3
    )

    print("\n💰 ROI ANALYSIS")
    print("-" * 80)
    print(f"Implementation cost: ${100_000:,.0f}")
    print(f"Annual maintenance: ${20_000:,.0f}")
    print(f"Time horizon: {roi['time_horizon_years']} years")
    print()
    print(f"Total value: ${roi['total_value']:,.0f}")
    print(f"Total cost: ${roi['total_cost']:,.0f}")
    print(f"Net value: ${roi['net_value']:,.0f}")
    print()
    print(f"ROI: {roi['roi']:.1f}x")
    print(f"Payback period: {roi['payback_months']:.1f} months")

    # User acquisition equivalent
    equiv = user_acquisition_equivalent(
        annual_value=impact['annual_value'],
        cost_per_acquisition=50
    )

    print("\n👥 USER ACQUISITION EQUIVALENT")
    print("-" * 80)
    print(f"This improvement is worth ${equiv['annual_value']:,.0f},")
    print(f"equivalent to acquiring {equiv['equivalent_users']:,.0f} new users")
    print(f"at ${equiv['cost_per_acquisition']:.2f} per user.")

    # Executive summary
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    summary = executive_summary(
        effect=0.02,
        annual_users=10_000_000,
        value_per_conversion=150,
        implementation_cost=100_000,
        baseline_rate=0.10,
        experiment_name="New Onboarding Flow"
    )
    print(summary)
