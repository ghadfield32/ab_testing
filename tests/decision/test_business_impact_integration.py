"""
Integration tests for business impact calculations.

Tests the entire chain: metric → lift → CI → business impact → ROI
to ensure sign preservation and correct magnitude calculations.
"""

import pytest
import numpy as np
from ab_testing.core import frequentist
from ab_testing.decision import business_impact


class TestBusinessImpactSignPreservation:
    """Test that signs are preserved through the entire calculation chain."""

    def test_positive_lift_positive_impact(self):
        """When lift > 0 and CI fully positive, business impact must be positive."""
        # Arrange: Clear positive effect (6% vs 5% conversion)
        x_control, n_control = 50, 1000
        x_treatment, n_treatment = 60, 1000

        # Act: Run statistical test
        result = frequentist.z_test_proportions(
            x_control=x_control,
            n_control=n_control,
            x_treatment=x_treatment,
            n_treatment=n_treatment,
            alpha=0.05,
            two_sided=True
        )

        # Assert: Positive lift
        assert result['absolute_lift'] > 0, "Lift should be positive"
        assert result['ci_lower'] < result['ci_upper'], "CI should be properly ordered"

        # Act: Calculate business impact
        annual_users = 1_000_000
        value_per_conversion = 100

        # Using point estimate
        impact_point = business_impact.calculate_annual_impact(
            effect=result['absolute_lift'],
            annual_users=annual_users,
            value_per_conversion=value_per_conversion
        )

        # Using CI bounds (simulating pipeline logic)
        worst_case_conversions = annual_users * result['ci_lower']
        best_case_conversions = annual_users * result['ci_upper']
        worst_case_annual = worst_case_conversions * value_per_conversion
        best_case_annual = best_case_conversions * value_per_conversion

        # Assert: All positive when lift is positive
        assert impact_point['annual_value'] > 0, "Point estimate must be positive"
        assert impact_point['additional_conversions'] > 0, "Additional conversions must be positive"

        # CI might cross zero, but if it doesn't, both bounds should be positive
        if result['ci_lower'] > 0:
            assert worst_case_annual > 0, "Worst case must be positive when CI lower bound > 0"
            assert best_case_annual > 0, "Best case must be positive"
            assert best_case_annual > worst_case_annual, "Best > worst"

    def test_negative_lift_negative_impact(self):
        """When lift < 0 and CI fully negative, business impact must be negative."""
        # Arrange: Clear negative effect (4% vs 5% conversion)
        x_control, n_control = 50, 1000
        x_treatment, n_treatment = 40, 1000

        # Act: Run statistical test
        result = frequentist.z_test_proportions(
            x_control=x_control,
            n_control=n_control,
            x_treatment=x_treatment,
            n_treatment=n_treatment,
            alpha=0.05,
            two_sided=True
        )

        # Assert: Negative lift
        assert result['absolute_lift'] < 0, "Lift should be negative"

        # Act: Calculate business impact
        annual_users = 1_000_000
        value_per_conversion = 100

        impact_point = business_impact.calculate_annual_impact(
            effect=result['absolute_lift'],
            annual_users=annual_users,
            value_per_conversion=value_per_conversion
        )

        # Using CI bounds
        worst_case_conversions = annual_users * result['ci_lower']
        best_case_conversions = annual_users * result['ci_upper']
        worst_case_annual = worst_case_conversions * value_per_conversion
        best_case_annual = best_case_conversions * value_per_conversion

        # Assert: All negative when lift is negative
        assert impact_point['annual_value'] < 0, "Point estimate must be negative"
        assert impact_point['additional_conversions'] < 0, "Additional conversions must be negative"

        # If CI fully negative
        if result['ci_upper'] < 0:
            assert worst_case_annual < 0, "Worst case must be negative"
            assert best_case_annual < 0, "Best case must be negative when CI upper bound < 0"
            assert best_case_annual > worst_case_annual, "Best (less negative) > worst (more negative)"

    def test_ci_crosses_zero_uncertain_impact(self):
        """When CI crosses zero, impact range should clearly show uncertainty."""
        # Arrange: Small effect with wide CI (likely to cross zero)
        x_control, n_control = 48, 1000
        x_treatment, n_treatment = 52, 1000

        # Act: Run statistical test
        result = frequentist.z_test_proportions(
            x_control=x_control,
            n_control=n_control,
            x_treatment=x_treatment,
            n_treatment=n_treatment,
            alpha=0.05,
            two_sided=True
        )

        # Assert: CI likely crosses zero
        if result['ci_lower'] < 0 < result['ci_upper']:
            # Act: Calculate business impact
            annual_users = 1_000_000
            value_per_conversion = 100

            worst_case_conversions = annual_users * result['ci_lower']
            best_case_conversions = annual_users * result['ci_upper']
            worst_case_annual = worst_case_conversions * value_per_conversion
            best_case_annual = best_case_conversions * value_per_conversion

            # Assert: Range crosses zero - shows uncertainty
            assert worst_case_annual < 0, "Worst case should be negative"
            assert best_case_annual > 0, "Best case should be positive"
            assert best_case_annual > worst_case_annual, "Best > worst"

            # Point estimate sign matches lift sign
            point_impact = business_impact.calculate_annual_impact(
                effect=result['absolute_lift'],
                annual_users=annual_users,
                value_per_conversion=value_per_conversion
            )
            assert np.sign(point_impact['annual_value']) == np.sign(result['absolute_lift']), \
                "Point estimate sign must match lift sign"


class TestBusinessImpactMagnitude:
    """Test that magnitudes are calculated correctly."""

    def test_known_values_small_effect(self):
        """Test with known inputs and expected outputs."""
        # Arrange: 1 percentage point lift
        effect = 0.01  # 1pp lift
        annual_users = 1_000_000
        value_per_conversion = 100
        baseline_rate = 0.05

        # Act
        impact = business_impact.calculate_annual_impact(
            effect=effect,
            annual_users=annual_users,
            value_per_conversion=value_per_conversion,
            baseline_rate=baseline_rate
        )

        # Assert: Manual calculation
        expected_conversions = 1_000_000 * 0.01  # 10,000 conversions
        expected_annual_value = 10_000 * 100  # $1,000,000
        expected_monthly_value = expected_annual_value / 12
        expected_relative_lift = 0.01 / 0.05  # 20% relative lift

        assert abs(impact['additional_conversions'] - expected_conversions) < 1, \
            f"Expected {expected_conversions}, got {impact['additional_conversions']}"
        assert abs(impact['annual_value'] - expected_annual_value) < 1, \
            f"Expected {expected_annual_value}, got {impact['annual_value']}"
        assert abs(impact['monthly_value'] - expected_monthly_value) < 1
        assert abs(impact['relative_lift'] - expected_relative_lift) < 0.001
        assert abs(impact['new_rate'] - 0.06) < 0.001, "New rate should be 5% + 1pp = 6%"

    def test_known_values_large_effect(self):
        """Test with larger effect to ensure linear scaling."""
        # Arrange: 5 percentage point lift
        effect = 0.05
        annual_users = 500_000
        value_per_conversion = 200

        # Act
        impact = business_impact.calculate_annual_impact(
            effect=effect,
            annual_users=annual_users,
            value_per_conversion=value_per_conversion
        )

        # Assert
        expected_conversions = 500_000 * 0.05  # 25,000 conversions
        expected_annual_value = 25_000 * 200  # $5,000,000

        assert abs(impact['additional_conversions'] - expected_conversions) < 1
        assert abs(impact['annual_value'] - expected_annual_value) < 1

    def test_negative_effect_magnitude(self):
        """Test that negative effects have correct magnitude (not just sign)."""
        # Arrange: -2pp lift (harmful treatment)
        effect = -0.02
        annual_users = 1_000_000
        value_per_conversion = 100

        # Act
        impact = business_impact.calculate_annual_impact(
            effect=effect,
            annual_users=annual_users,
            value_per_conversion=value_per_conversion
        )

        # Assert: Magnitude correct (negative)
        expected_conversions = 1_000_000 * (-0.02)  # -20,000 conversions
        expected_annual_value = -20_000 * 100  # -$2,000,000

        assert abs(impact['additional_conversions'] - expected_conversions) < 1
        assert abs(impact['annual_value'] - expected_annual_value) < 1
        assert impact['annual_value'] < 0, "Must be negative"
        assert impact['additional_conversions'] < 0, "Must be negative"


class TestROICalculation:
    """Test ROI calculations preserve signs and magnitudes."""

    def test_positive_value_positive_roi(self):
        """Positive annual value should give positive ROI (assuming reasonable costs)."""
        # Arrange
        annual_value = 1_000_000
        implementation_cost = 100_000

        # Act
        roi = business_impact.calculate_roi(
            annual_value=annual_value,
            implementation_cost=implementation_cost
        )

        # Assert
        expected_net_value = 1_000_000 - 100_000  # $900,000
        expected_roi = 900_000 / 100_000  # 9x

        assert roi['net_value'] > 0, "Net value must be positive"
        assert roi['roi'] > 0, "ROI must be positive"
        assert abs(roi['roi'] - expected_roi) < 0.1
        assert 0 < roi['payback_months'] < 12, "Should break even in < 1 year"

    def test_negative_value_negative_roi(self):
        """Negative annual value should give negative ROI."""
        # Arrange: Harmful treatment
        annual_value = -500_000  # Losing money
        implementation_cost = 100_000

        # Act
        roi = business_impact.calculate_roi(
            annual_value=annual_value,
            implementation_cost=implementation_cost
        )

        # Assert
        expected_net_value = -500_000 - 100_000  # -$600,000
        expected_roi = -600_000 / 100_000  # -6x (losing money)

        assert roi['net_value'] < 0, "Net value must be negative"
        assert roi['roi'] < 0, "ROI must be negative"
        assert abs(roi['roi'] - expected_roi) < 0.1
        assert roi['payback_months'] == np.inf, "Never pays back"


class TestEndToEndBusinessImpactChain:
    """Test the complete chain from statistical test to business decision."""

    def test_complete_chain_positive_lift(self):
        """Test: Statistical test → Business impact → ROI → Decision."""
        # Arrange: Run experiment
        x_control, n_control = 50, 1000
        x_treatment, n_treatment = 60, 1000

        # Act: Statistical test
        stat_result = frequentist.z_test_proportions(
            x_control=x_control,
            n_control=n_control,
            x_treatment=x_treatment,
            n_treatment=n_treatment,
            alpha=0.05,
            two_sided=True
        )

        # Business impact (point estimate)
        impact = business_impact.calculate_annual_impact(
            effect=stat_result['absolute_lift'],
            annual_users=1_000_000,
            value_per_conversion=100,
            baseline_rate=stat_result['p_control']
        )

        # ROI
        roi = business_impact.calculate_roi(
            annual_value=impact['annual_value'],
            implementation_cost=100_000
        )

        # Assert: Chain consistency
        assert stat_result['absolute_lift'] > 0, "Lift positive"
        assert impact['annual_value'] > 0, "Impact positive"
        assert roi['roi'] > 0, "ROI positive"

        # Sign preservation throughout chain
        assert np.sign(stat_result['absolute_lift']) == np.sign(impact['annual_value']), \
            "Impact sign must match lift sign"
        assert np.sign(impact['annual_value']) == np.sign(roi['net_value']), \
            "Net value sign must match impact sign"

    def test_complete_chain_negative_lift(self):
        """Test chain with harmful treatment."""
        # Arrange: Harmful treatment
        x_control, n_control = 50, 1000
        x_treatment, n_treatment = 40, 1000

        # Act: Statistical test
        stat_result = frequentist.z_test_proportions(
            x_control=x_control,
            n_control=n_control,
            x_treatment=x_treatment,
            n_treatment=n_treatment,
            alpha=0.05,
            two_sided=True
        )

        # Business impact
        impact = business_impact.calculate_annual_impact(
            effect=stat_result['absolute_lift'],
            annual_users=1_000_000,
            value_per_conversion=100,
            baseline_rate=stat_result['p_control']
        )

        # ROI
        roi = business_impact.calculate_roi(
            annual_value=impact['annual_value'],
            implementation_cost=100_000
        )

        # Assert: Negative throughout chain
        assert stat_result['absolute_lift'] < 0, "Lift negative"
        assert impact['annual_value'] < 0, "Impact negative (losing money)"
        assert roi['roi'] < 0, "ROI negative"
        assert roi['net_value'] < 0, "Net value negative"

        # Decision: Should ABANDON (negative impact)
        should_ship = roi['roi'] > 1  # Positive ROI threshold
        assert should_ship == False, "Should NOT ship harmful treatment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
