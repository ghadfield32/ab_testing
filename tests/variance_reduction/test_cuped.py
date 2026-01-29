"""
Tests for CUPED variance reduction module.
"""

import pytest
import numpy as np
from ab_testing.variance_reduction import cuped


class TestCupedAdjustment:
    """Tests for cuped_adjustment function."""

    def test_cuped_reduces_variance(self):
        """Test that CUPED reduces variance when Y and X are correlated."""
        np.random.seed(42)
        n = 1000

        # Generate correlated X and Y
        x = np.random.normal(100, 20, n)
        y = 2 * x + np.random.normal(0, 10, n)

        # Apply CUPED
        y_adjusted = cuped.cuped_adjustment(y, x)

        # Variance should be reduced
        var_original = y.var(ddof=1)
        var_adjusted = y_adjusted.var(ddof=1)

        assert var_adjusted < var_original
        assert var_adjusted > 0  # Should not be zero

    def test_cuped_preserves_mean(self):
        """Test that CUPED preserves the mean of Y."""
        np.random.seed(42)
        x = np.random.normal(100, 20, 500)
        y = np.random.normal(50, 10, 500)

        y_adjusted = cuped.cuped_adjustment(y, x)

        # Mean should be preserved (within numerical precision)
        assert abs(y_adjusted.mean() - y.mean()) < 1e-10

    def test_cuped_with_custom_theta(self):
        """Test CUPED with custom theta parameter."""
        np.random.seed(42)
        x = np.random.normal(100, 20, 500)
        y = np.random.normal(50, 10, 500)

        theta_custom = 0.5
        y_adjusted = cuped.cuped_adjustment(y, x, theta=theta_custom)

        # Should use custom theta
        expected = y - theta_custom * (x - x.mean())
        np.testing.assert_array_almost_equal(y_adjusted, expected)

    def test_cuped_uncorrelated_no_reduction(self):
        """Test that CUPED doesn't reduce variance when X and Y are uncorrelated."""
        np.random.seed(42)
        x = np.random.normal(100, 20, 500)
        y = np.random.normal(50, 10, 500)  # Uncorrelated with x

        y_adjusted = cuped.cuped_adjustment(y, x)

        # Variance reduction should be minimal
        var_original = y.var(ddof=1)
        var_adjusted = y_adjusted.var(ddof=1)

        # Should be close (within 20%)
        assert abs(var_adjusted - var_original) / var_original < 0.20

    def test_cuped_raises_on_length_mismatch(self):
        """Test that CUPED raises ValueError when y and x have different lengths."""
        y = np.array([1, 2, 3])
        x = np.array([1, 2])

        with pytest.raises(ValueError, match="y and x must have same length"):
            cuped.cuped_adjustment(y, x)

    def test_cuped_raises_on_insufficient_data(self):
        """Test that CUPED raises ValueError with < 2 observations."""
        y = np.array([1])
        x = np.array([1])

        with pytest.raises(ValueError, match="Need at least 2 observations"):
            cuped.cuped_adjustment(y, x)


class TestCupedABTest:
    """Tests for cuped_ab_test function."""

    def test_cuped_ab_test_basic(self):
        """Test basic CUPED A/B test."""
        np.random.seed(42)
        n = 500

        # Pre-experiment covariate
        x_control = np.random.normal(100, 20, n)
        x_treatment = np.random.normal(100, 20, n)

        # Post-experiment outcome (correlated with pre)
        y_control = 0.5 * x_control + np.random.normal(0, 10, n)
        y_treatment = 0.5 * x_treatment + 5 + np.random.normal(0, 10, n)

        result = cuped.cuped_ab_test(y_control, y_treatment, x_control, x_treatment)

        # Should have all required keys
        assert 'mean_control' in result
        assert 'mean_treatment' in result
        assert 'difference' in result
        assert 'var_reduction' in result
        assert 'se_reduction' in result
        assert 'sample_size_reduction' in result
        assert 'p_value' in result
        assert 'significant' in result

        # Variance reduction should be positive (correlated data)
        assert result['var_reduction'] > 0
        assert result['se_reduction'] > 0

    def test_cuped_improves_power(self):
        """Test that CUPED improves statistical power."""
        np.random.seed(42)
        n = 200  # Small sample

        # Highly correlated pre-experiment data
        x_control = np.random.normal(100, 20, n)
        x_treatment = np.random.normal(100, 20, n)

        # Small effect but high correlation
        y_control = 2 * x_control + np.random.normal(0, 5, n)
        y_treatment = 2 * x_treatment + 3 + np.random.normal(0, 5, n)

        # CUPED result
        cuped_result = cuped.cuped_ab_test(y_control, y_treatment, x_control, x_treatment)

        # Naive result (without CUPED) - simulate by using y directly
        from ab_testing.core import frequentist
        naive_result = frequentist.welch_ttest(y_control, y_treatment)

        # CUPED should have smaller p-value (better power)
        assert cuped_result['p_value'] < naive_result['p_value']
        assert cuped_result['se_adj_diff'] < naive_result['se_diff']

    def test_cuped_ab_test_raises_on_length_mismatch(self):
        """Test that cuped_ab_test raises on mismatched lengths."""
        y_control = np.array([1, 2, 3])
        y_treatment = np.array([1, 2])
        x_control = np.array([1, 2, 3])
        x_treatment = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            cuped.cuped_ab_test(y_control, y_treatment, x_control, x_treatment)

    def test_cuped_equivalent_sample_size(self):
        """Test that equivalent sample size calculation is reasonable."""
        np.random.seed(42)
        n = 500

        x_control = np.random.normal(100, 20, n)
        x_treatment = np.random.normal(100, 20, n)
        y_control = 0.8 * x_control + np.random.normal(0, 10, n)
        y_treatment = 0.8 * x_treatment + np.random.normal(0, 10, n)

        result = cuped.cuped_ab_test(y_control, y_treatment, x_control, x_treatment)

        # Sample size reduction is a ratio (e.g., 0.70 = 70% reduction)
        # Calculate equivalent absolute sample size: n / (1 - ratio)
        sample_size_ratio = result['sample_size_reduction']
        assert 0 < sample_size_ratio < 1  # Verify it's a valid ratio
        equiv_n_absolute = n / max(1 - sample_size_ratio, 0.01)
        assert equiv_n_absolute > n  # Should be equivalent to larger sample
        assert equiv_n_absolute < n * 5  # Allow up to 80% variance reduction (n/0.2 = 5n)


class TestMultiCovariateCuped:
    """Tests for multi_covariate_cuped function."""

    def test_multi_covariate_basic(self):
        """Test CUPED with multiple covariates."""
        np.random.seed(42)
        n = 500

        # Multiple pre-experiment covariates
        X = np.random.randn(n, 3)
        # Y correlated with all covariates
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + np.random.normal(0, 5, n)

        y_adjusted = cuped.multi_covariate_cuped(y, X)

        # Variance should be reduced
        assert y_adjusted.var(ddof=1) < y.var(ddof=1)

        # Mean preserved
        assert abs(y_adjusted.mean() - y.mean()) < 1e-10

    def test_multi_covariate_single_feature(self):
        """Test multi-covariate CUPED with single feature (should match single-covariate)."""
        np.random.seed(42)
        n = 500
        x = np.random.normal(100, 20, n)
        y = 2 * x + np.random.normal(0, 10, n)

        # Single covariate as 2D array
        X = x.reshape(-1, 1)
        y_adjusted_multi = cuped.multi_covariate_cuped(y, X)

        # Single covariate function
        y_adjusted_single = cuped.cuped_adjustment(y, x)

        # Should be very close (may differ slightly due to OLS vs. direct formula)
        np.testing.assert_array_almost_equal(y_adjusted_multi, y_adjusted_single, decimal=5)

    def test_multi_covariate_raises_on_1d_X(self):
        """Test that multi_covariate_cuped raises on 1D X."""
        y = np.array([1, 2, 3])
        X = np.array([1, 2, 3])  # 1D

        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            cuped.multi_covariate_cuped(y, X)

    def test_multi_covariate_raises_on_length_mismatch(self):
        """Test that multi_covariate_cuped raises on length mismatch."""
        y = np.array([1, 2, 3])
        X = np.array([[1, 2], [3, 4]])  # Only 2 rows

        with pytest.raises(ValueError, match="y and X must have same number of rows"):
            cuped.multi_covariate_cuped(y, X)


class TestVarianceReduction:
    """Tests for variance_reduction function."""

    def test_variance_reduction_calculation(self):
        """Test variance reduction percentage calculation."""
        np.random.seed(42)
        n = 1000

        x = np.random.normal(100, 20, n)
        y = 2 * x + np.random.normal(0, 10, n)

        var_reduction = cuped.variance_reduction(y, x)

        # Should be positive for correlated data
        assert var_reduction > 0
        # Should be less than 100%
        assert var_reduction < 1.0

        # Calculate manually
        y_adj = cuped.cuped_adjustment(y, x)
        expected = 1 - y_adj.var(ddof=1) / y.var(ddof=1)

        assert abs(var_reduction - expected) < 1e-10

    def test_variance_reduction_zero_for_uncorrelated(self):
        """Test that variance reduction is near zero for uncorrelated data."""
        np.random.seed(42)
        x = np.random.normal(100, 20, 500)
        y = np.random.normal(50, 10, 500)

        var_reduction = cuped.variance_reduction(y, x)

        # Should be close to zero (within ±10%)
        assert abs(var_reduction) < 0.10


class TestPowerGainCuped:
    """Tests for power_gain_cuped function."""

    def test_power_gain_basic(self):
        """Test basic power gain calculation."""
        np.random.seed(42)
        n = 500

        x_control = np.random.normal(100, 20, n)
        x_treatment = np.random.normal(100, 20, n)
        y_control = 0.8 * x_control + np.random.normal(0, 10, n)
        y_treatment = 0.8 * x_treatment + 5 + np.random.normal(0, 10, n)

        power_gain = cuped.power_gain_cuped(
            y_control, y_treatment, x_control, x_treatment
        )

        # Should have all keys
        assert 'var_reduction' in power_gain
        assert 'power_multiplier' in power_gain
        assert 'equivalent_n' in power_gain

        # Power multiplier should be > 1 for correlated data
        assert power_gain['power_multiplier'] > 1.0

    def test_power_gain_no_correlation(self):
        """Test power gain with uncorrelated data."""
        np.random.seed(42)
        n = 500

        x_control = np.random.normal(100, 20, n)
        x_treatment = np.random.normal(100, 20, n)
        y_control = np.random.normal(50, 10, n)  # Uncorrelated
        y_treatment = np.random.normal(55, 10, n)

        power_gain = cuped.power_gain_cuped(
            y_control, y_treatment, x_control, x_treatment
        )

        # Power multiplier should be close to 1
        assert 0.8 < power_gain['power_multiplier'] < 1.2
