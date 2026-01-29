"""
Tests for guardrail metrics module.
"""

import pytest
import numpy as np
from ab_testing.diagnostics import guardrails


class TestNonInferiorityTest:
    """Tests for non_inferiority_test function."""

    def test_non_inferiority_pass(self):
        """Test non-inferiority when degradation is within margin."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 1000)
        treatment = np.random.normal(98, 20, 1000)  # 2% degradation

        result = guardrails.non_inferiority_test(
            control, treatment, delta=-0.05, metric_type='relative'  # Allow 5% degradation
        )

        assert 'passed' in result
        assert 'ci_lower' in result
        assert 'margin_used' in result

        # 2% degradation within 5% margin - should pass
        assert result['passed'] is True

    def test_non_inferiority_fail(self):
        """Test non-inferiority when degradation exceeds margin."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 1000)
        treatment = np.random.normal(92, 20, 1000)  # 8% degradation

        result = guardrails.non_inferiority_test(
            control, treatment, delta=-0.05, metric_type='relative'
        )

        # 8% degradation exceeds 5% margin - should fail
        assert result['passed'] is False

    def test_non_inferiority_absolute_margin(self):
        """Test non-inferiority with absolute margin."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 1000)
        treatment = np.random.normal(98, 20, 1000)

        result = guardrails.non_inferiority_test(
            control, treatment, delta=-3, metric_type='absolute'
        )

        # Mean difference ~2, margin is 3 - should pass
        assert result['passed'] is True
        assert result['margin_used'] == -3

    def test_non_inferiority_raises_positive_delta(self):
        """Test that positive delta raises error."""
        control = np.random.normal(100, 20, 100)
        treatment = np.random.normal(100, 20, 100)

        with pytest.raises(ValueError, match="delta must be negative"):
            guardrails.non_inferiority_test(control, treatment, delta=0.05)

    def test_non_inferiority_small_sample(self):
        """Test that small samples raise error."""
        control = np.array([1])
        treatment = np.array([1])

        with pytest.raises(ValueError, match="at least 2 observations"):
            guardrails.non_inferiority_test(control, treatment, delta=-0.05)


class TestGuardrailTest:
    """Tests for guardrail_test wrapper function."""

    def test_guardrail_test_basic(self):
        """Test basic guardrail test."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 1000)
        treatment = np.random.normal(99, 20, 1000)

        result = guardrails.guardrail_test(
            control, treatment, delta=-0.02, metric_name='revenue_per_user'
        )

        assert 'metric_name' in result
        assert result['metric_name'] == 'revenue_per_user'
        assert 'passed' in result

    def test_guardrail_test_default_delta(self):
        """Test guardrail test uses -2% default."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 1000)
        treatment = np.random.normal(99, 20, 1000)

        result = guardrails.guardrail_test(control, treatment, metric_name='retention')

        # Should use default -2%
        assert result['delta'] == -0.02


class TestEvaluateGuardrails:
    """Tests for evaluate_guardrails function."""

    def test_evaluate_ship_decision(self):
        """Test SHIP decision when all conditions met."""
        primary = {'significant': True, 'difference': 0.05}
        guardrails_results = [
            {'metric_name': 'revenue', 'passed': True},
            {'metric_name': 'retention', 'passed': True}
        ]

        decision = guardrails.evaluate_guardrails(primary, guardrails_results)

        assert decision['decision'] == 'ship'
        assert decision['primary_significant'] is True
        assert decision['primary_positive'] is True
        assert decision['all_guardrails_passed'] is True

    def test_evaluate_abandon_negative_primary(self):
        """Test ABANDON decision when primary is negative."""
        primary = {'significant': True, 'difference': -0.05}
        guardrails_results = [
            {'metric_name': 'revenue', 'passed': True}
        ]

        decision = guardrails.evaluate_guardrails(primary, guardrails_results)

        assert decision['decision'] == 'abandon'
        assert decision['primary_positive'] is False

    def test_evaluate_hold_guardrail_failure(self):
        """Test HOLD decision when guardrails fail."""
        primary = {'significant': True, 'difference': 0.05}
        guardrails_results = [
            {'metric_name': 'revenue', 'passed': True},
            {'metric_name': 'retention', 'passed': False}
        ]

        decision = guardrails.evaluate_guardrails(primary, guardrails_results)

        assert decision['decision'] == 'hold'
        assert decision['all_guardrails_passed'] is False
        assert 'retention' in decision['failed_guardrails']

    def test_evaluate_hold_not_significant(self):
        """Test HOLD decision when primary not significant."""
        primary = {'significant': False, 'difference': 0.03}
        guardrails_results = [
            {'metric_name': 'revenue', 'passed': True}
        ]

        decision = guardrails.evaluate_guardrails(primary, guardrails_results)

        assert decision['decision'] == 'hold'
        assert decision['primary_significant'] is False

    def test_evaluate_no_guardrails(self):
        """Test evaluation with no guardrails."""
        primary = {'significant': True, 'difference': 0.05}
        guardrails_results = []

        decision = guardrails.evaluate_guardrails(primary, guardrails_results)

        # Should ship if primary is good
        assert decision['decision'] == 'ship'
        assert decision['guardrails_total'] == 0


class TestPowerForGuardrail:
    """Tests for power_for_guardrail function."""

    def test_power_calculation_basic(self):
        """Test basic power calculation for guardrail."""
        power = guardrails.power_for_guardrail(
            baseline_mean=100,
            baseline_std=20,
            delta=-0.05,
            n_per_group=1000
        )

        # Power should be between 0 and 1
        assert 0 < power < 1
        assert power > 0.5  # Should have decent power with n=1000

    def test_power_increases_with_sample_size(self):
        """Test that power increases with sample size."""
        power_small = guardrails.power_for_guardrail(
            baseline_mean=100,
            baseline_std=20,
            delta=-0.05,
            n_per_group=500
        )

        power_large = guardrails.power_for_guardrail(
            baseline_mean=100,
            baseline_std=20,
            delta=-0.05,
            n_per_group=2000
        )

        assert power_large > power_small

    def test_power_decreases_with_smaller_delta(self):
        """Test that power decreases for smaller (harder to detect) effects."""
        # Larger margin (easier to meet)
        power_large_margin = guardrails.power_for_guardrail(
            baseline_mean=100,
            baseline_std=20,
            delta=-0.10,  # 10% degradation
            n_per_group=1000
        )

        # Smaller margin (harder to meet)
        power_small_margin = guardrails.power_for_guardrail(
            baseline_mean=100,
            baseline_std=20,
            delta=-0.02,  # 2% degradation
            n_per_group=1000
        )

        # Larger margin should have higher power
        assert power_large_margin > power_small_margin


class TestIntegration:
    """Integration tests for guardrail workflow."""

    def test_full_workflow(self):
        """Test complete guardrail workflow."""
        np.random.seed(42)

        # Primary metric: conversion (improved)
        conv_control = np.random.binomial(1, 0.10, 1000)
        conv_treatment = np.random.binomial(1, 0.12, 1000)

        from ab_testing.core import frequentist
        primary = frequentist.z_test_proportions(
            conv_control.sum(), len(conv_control),
            conv_treatment.sum(), len(conv_treatment)
        )

        # Guardrail 1: Revenue (slight degradation but within bounds)
        rev_control = np.random.gamma(2, 50, 1000)
        rev_treatment = np.random.gamma(2, 48, 1000)
        guardrail_rev = guardrails.guardrail_test(
            rev_control, rev_treatment,
            delta=-0.05, metric_name='revenue_per_user'
        )

        # Guardrail 2: Retention (no degradation)
        ret_control = np.random.binomial(1, 0.45, 1000)
        ret_treatment = np.random.binomial(1, 0.44, 1000)
        guardrail_ret = guardrails.guardrail_test(
            ret_control.astype(float), ret_treatment.astype(float),
            delta=-0.05, metric_name='retention_7d'
        )

        # Make decision
        decision = guardrails.evaluate_guardrails(
            primary, [guardrail_rev, guardrail_ret]
        )

        # Verify decision structure
        assert 'decision' in decision
        assert decision['decision'] in ['ship', 'hold', 'abandon']
        assert 'guardrails_passed' in decision
        assert 'guardrails_total' in decision
        assert decision['guardrails_total'] == 2
