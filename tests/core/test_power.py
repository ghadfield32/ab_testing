"""Unit tests for power analysis module."""

import pytest
import numpy as np
from ab_testing.core import power


class TestCohensH:
    """Tests for Cohen's h effect size calculation."""

    def test_cohens_h_positive_effect(self):
        """Test Cohen's h for positive effect."""
        h = power.cohens_h(p1=0.10, p2=0.12)
        assert h > 0
        assert 0.05 < h < 0.10  # Should be small effect

    def test_cohens_h_negative_effect(self):
        """Test Cohen's h for negative effect."""
        h = power.cohens_h(p1=0.12, p2=0.10)
        assert h < 0

    def test_cohens_h_no_effect(self):
        """Test Cohen's h when proportions are equal."""
        h = power.cohens_h(p1=0.10, p2=0.10)
        assert abs(h) < 1e-10

    def test_cohens_h_invalid_proportions(self):
        """Test error handling for invalid proportions."""
        with pytest.raises(ValueError, match="Proportions must be between 0 and 1"):
            power.cohens_h(p1=-0.1, p2=0.5)
        with pytest.raises(ValueError, match="Proportions must be between 0 and 1"):
            power.cohens_h(p1=0.5, p2=1.5)


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_positive_effect(self):
        """Test Cohen's d for positive effect."""
        d = power.cohens_d(mean1=100, mean2=115, std1=20, std2=22, n1=500, n2=500)
        assert d > 0
        assert 0.5 < d < 1.0  # Should be medium to large effect

    def test_cohens_d_negative_effect(self):
        """Test Cohen's d for negative effect."""
        d = power.cohens_d(mean1=115, mean2=100, std1=20, std2=22, n1=500, n2=500)
        assert d < 0

    def test_cohens_d_no_effect(self):
        """Test Cohen's d when means are equal."""
        d = power.cohens_d(mean1=100, mean2=100, std1=20, std2=20, n1=500, n2=500)
        assert abs(d) < 1e-10


class TestRequiredSamplesBinary:
    """Tests for binary metric sample size calculation."""

    def test_required_samples_baseline_5pct(self):
        """Test sample size for 5% baseline with 10% relative lift."""
        n = power.required_samples_binary(p_baseline=0.05, mde=0.10)
        # Should be around 31,000 per group (per-group, not total/2)
        assert 30000 < n < 32000

    def test_required_samples_higher_baseline(self):
        """Test that higher baseline requires fewer samples."""
        n_low = power.required_samples_binary(p_baseline=0.05, mde=0.10)
        n_high = power.required_samples_binary(p_baseline=0.25, mde=0.10)
        assert n_high < n_low

    def test_required_samples_larger_mde(self):
        """Test that larger MDE requires fewer samples."""
        n_small = power.required_samples_binary(p_baseline=0.05, mde=0.05)
        n_large = power.required_samples_binary(p_baseline=0.05, mde=0.20)
        assert n_large < n_small

    def test_required_samples_higher_power(self):
        """Test that higher power requires more samples."""
        n_low_power = power.required_samples_binary(p_baseline=0.05, mde=0.10, power=0.70)
        n_high_power = power.required_samples_binary(p_baseline=0.05, mde=0.10, power=0.90)
        assert n_high_power > n_low_power

    def test_required_samples_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="p_baseline must be between 0 and 1"):
            power.required_samples_binary(p_baseline=0, mde=0.10)
        with pytest.raises(ValueError, match="MDE must be positive"):
            power.required_samples_binary(p_baseline=0.05, mde=-0.10)
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            power.required_samples_binary(p_baseline=0.05, mde=0.10, alpha=1.5)

    def test_required_samples_treatment_exceeds_one(self):
        """Test error when treatment proportion would exceed 1."""
        with pytest.raises(ValueError, match="Treatment proportion .* > 1"):
            power.required_samples_binary(p_baseline=0.95, mde=0.10)


class TestRequiredSamplesContinuous:
    """Tests for continuous metric sample size calculation."""

    def test_required_samples_revenue(self):
        """Test sample size for revenue metric."""
        n = power.required_samples_continuous(
            baseline_mean=175,
            baseline_std=80,
            mde=15
        )
        # Should be around 500-600 per group
        assert 400 < n < 700

    def test_required_samples_smaller_effect(self):
        """Test that smaller effect requires more samples."""
        n_large = power.required_samples_continuous(
            baseline_mean=175, baseline_std=80, mde=15
        )
        n_small = power.required_samples_continuous(
            baseline_mean=175, baseline_std=80, mde=5
        )
        assert n_small > n_large

    def test_required_samples_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="baseline_std must be positive"):
            power.required_samples_continuous(baseline_mean=100, baseline_std=0, mde=10)
        with pytest.raises(ValueError, match="MDE must be positive"):
            power.required_samples_continuous(baseline_mean=100, baseline_std=20, mde=-5)


class TestInterpretEffectSize:
    """Tests for effect size interpretation."""

    def test_interpret_negligible(self):
        """Test interpretation of negligible effect."""
        interp = power.interpret_effect_size(0.15, metric_type='h')
        assert interp == "Negligible"

    def test_interpret_small(self):
        """Test interpretation of small effect."""
        interp = power.interpret_effect_size(0.35, metric_type='h')
        assert interp == "Small"

    def test_interpret_medium(self):
        """Test interpretation of medium effect."""
        interp = power.interpret_effect_size(0.65, metric_type='h')
        assert interp == "Medium"

    def test_interpret_large(self):
        """Test interpretation of large effect."""
        interp = power.interpret_effect_size(0.95, metric_type='h')
        assert interp == "Large"


class TestPowerAnalysisSummary:
    """Tests for comprehensive power analysis summary."""

    def test_summary_output_keys(self):
        """Test that summary contains all expected keys."""
        summary = power.power_analysis_summary(p_baseline=0.05, mde=0.10)

        expected_keys = [
            'p_baseline', 'p_treatment', 'mde_relative', 'mde_absolute',
            'cohens_h', 'interpretation', 'sample_per_group', 'sample_total',
            'alpha', 'power'
        ]

        for key in expected_keys:
            assert key in summary

    def test_summary_values(self):
        """Test that summary values are correct."""
        summary = power.power_analysis_summary(p_baseline=0.05, mde=0.10, alpha=0.05, power=0.80)

        assert summary['p_baseline'] == 0.05
        assert abs(summary['p_treatment'] - 0.055) < 1e-10
        assert summary['mde_relative'] == 0.10
        assert abs(summary['mde_absolute'] - 0.005) < 1e-10
        assert summary['alpha'] == 0.05
        assert summary['power'] == 0.80
        assert summary['sample_total'] == summary['sample_per_group'] * 2
