"""Unit tests for frequentist tests module."""

import pytest
import numpy as np
from ab_testing.core import frequentist


class TestZTestProportions:
    """Tests for z-test for proportions."""

    def test_z_test_no_effect(self):
        """Test z-test when proportions are equal."""
        result = frequentist.z_test_proportions(
            x_control=50, n_control=500,
            x_treatment=50, n_treatment=500
        )

        assert abs(result['absolute_lift']) < 1e-10
        assert abs(result['z_statistic']) < 0.1
        assert result['p_value'] > 0.90
        assert not result['significant']

    def test_z_test_positive_effect(self):
        """Test z-test with positive treatment effect."""
        result = frequentist.z_test_proportions(
            x_control=50, n_control=500,
            x_treatment=71, n_treatment=500
        )

        assert result['absolute_lift'] > 0
        assert result['relative_lift'] > 0
        assert result['z_statistic'] > 0
        assert result['p_value'] < 0.05
        assert result['significant']

    def test_z_test_confidence_interval(self):
        """Test that CI contains true difference."""
        result = frequentist.z_test_proportions(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500
        )

        # CI should contain the observed difference
        assert result['ci_lower'] < result['absolute_lift'] < result['ci_upper']

    def test_z_test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="must be non-negative"):
            frequentist.z_test_proportions(-1, 500, 50, 500)

        with pytest.raises(ValueError, match="must be positive"):
            frequentist.z_test_proportions(50, 0, 50, 500)

        with pytest.raises(ValueError, match="cannot exceed sample size"):
            frequentist.z_test_proportions(600, 500, 50, 500)


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_welch_no_effect(self):
        """Test t-test when means are equal."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(100, 20, 500)

        result = frequentist.welch_ttest(control, treatment)

        assert abs(result['difference']) < 5  # Should be close to 0
        assert result['p_value'] > 0.05
        assert not result['significant']

    def test_welch_positive_effect(self):
        """Test t-test with positive treatment effect."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = frequentist.welch_ttest(control, treatment)

        assert result['difference'] > 10
        assert result['relative_lift'] > 0.10
        assert result['p_value'] < 0.001
        assert result['significant']

    def test_welch_confidence_interval(self):
        """Test that CI is calculated correctly."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = frequentist.welch_ttest(control, treatment)

        # CI should contain the observed difference
        assert result['ci_lower'] < result['difference'] < result['ci_upper']
        # CI should not contain 0 for significant result
        assert not (result['ci_lower'] <= 0 <= result['ci_upper'])

    def test_welch_cohens_d(self):
        """Test Cohen's d calculation."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = frequentist.welch_ttest(control, treatment)

        # Cohen's d should be around 0.75 (large effect)
        assert 0.5 < result['cohens_d'] < 1.0

    def test_welch_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            frequentist.welch_ttest(np.array([1]), np.array([1, 2]))


class TestMannWhitneyU:
    """Tests for Mann-Whitney U test."""

    def test_mann_whitney_no_effect(self):
        """Test Mann-Whitney when distributions are equal."""
        np.random.seed(42)
        control = np.random.exponential(100, 500)
        treatment = np.random.exponential(100, 500)

        result = frequentist.mann_whitney_u(control, treatment)

        assert result['p_value'] > 0.05
        assert not result['significant']

    def test_mann_whitney_positive_effect(self):
        """Test Mann-Whitney with positive treatment effect."""
        np.random.seed(42)
        control = np.random.exponential(100, 500)
        treatment = np.random.exponential(130, 500)

        result = frequentist.mann_whitney_u(control, treatment)

        assert result['median_treatment'] > result['median_control']
        assert result['p_value'] < 0.05
        assert result['significant']

    def test_mann_whitney_rank_biserial(self):
        """Test rank-biserial correlation calculation."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = frequentist.mann_whitney_u(control, treatment)

        # Rank-biserial should be positive for positive effect
        assert result['rank_biserial'] > 0

    def test_mann_whitney_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            frequentist.mann_whitney_u(np.array([1]), np.array([1, 2]))


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_mean(self):
        """Test bootstrap for mean difference."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = frequentist.bootstrap_ci(
            control, treatment,
            statistic='mean',
            n_iterations=1000,
            random_state=42
        )

        # Point estimate should be close to true difference
        assert 10 < result['point_estimate'] < 20
        # CI should not contain 0
        assert not (result['ci_lower'] <= 0 <= result['ci_upper'])
        assert result['significant']

    def test_bootstrap_median(self):
        """Test bootstrap for median difference."""
        np.random.seed(42)
        control = np.random.exponential(100, 500)
        treatment = np.random.exponential(130, 500)

        result = frequentist.bootstrap_ci(
            control, treatment,
            statistic='median',
            n_iterations=1000,
            random_state=42
        )

        # Point estimate should be positive
        assert result['point_estimate'] > 0
        # Bootstrap SE should be positive
        assert result['bootstrap_se'] > 0

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with random_state."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 100)
        treatment = np.random.normal(115, 20, 100)

        result1 = frequentist.bootstrap_ci(control, treatment, random_state=42, n_iterations=1000)
        result2 = frequentist.bootstrap_ci(control, treatment, random_state=42, n_iterations=1000)

        assert abs(result1['point_estimate'] - result2['point_estimate']) < 1e-10
        assert abs(result1['bootstrap_se'] - result2['bootstrap_se']) < 1e-10

    def test_bootstrap_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        control = np.random.normal(100, 20, 100)
        treatment = np.random.normal(115, 20, 100)

        with pytest.raises(ValueError, match="at least 2 observations"):
            frequentist.bootstrap_ci(np.array([1]), treatment)

        with pytest.raises(ValueError, match="must be 'mean' or 'median'"):
            frequentist.bootstrap_ci(control, treatment, statistic='variance')

        with pytest.raises(ValueError, match="at least 100"):
            frequentist.bootstrap_ci(control, treatment, n_iterations=50)


class TestInterpretEffectSize:
    """Tests for effect size interpretation."""

    def test_interpret_cohens_d(self):
        """Test interpretation of Cohen's d."""
        assert frequentist.interpret_effect_size(0.15, 'd') == "Negligible"
        assert frequentist.interpret_effect_size(0.35, 'd') == "Small"
        assert frequentist.interpret_effect_size(0.65, 'd') == "Medium"
        assert frequentist.interpret_effect_size(0.95, 'd') == "Large"

    def test_interpret_rank_biserial(self):
        """Test interpretation of rank-biserial correlation."""
        assert frequentist.interpret_effect_size(0.05, 'r') == "Negligible"
        assert frequentist.interpret_effect_size(0.20, 'r') == "Small"
        assert frequentist.interpret_effect_size(0.40, 'r') == "Medium"
        assert frequentist.interpret_effect_size(0.60, 'r') == "Large"

    def test_interpret_invalid_type(self):
        """Test error for invalid metric type."""
        with pytest.raises(ValueError, match="must be 'd' or 'r'"):
            frequentist.interpret_effect_size(0.5, 'invalid')
