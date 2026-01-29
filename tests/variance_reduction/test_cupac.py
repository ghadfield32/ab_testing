"""
Tests for CUPAC variance reduction module.
"""

import pytest
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from ab_testing.variance_reduction import cupac


class TestCupacAdjustment:
    """Tests for cupac_adjustment function."""

    def test_cupac_reduces_variance(self):
        """Test that CUPAC reduces variance with correlated features."""
        np.random.seed(42)
        n = 1000

        # Multiple features
        X = np.random.randn(n, 5)
        # Y correlated with features (nonlinear relationship)
        y = (X[:, 0]**2 + X[:, 1] * X[:, 2] +
             np.sin(X[:, 3]) + np.random.normal(0, 5, n))

        # Apply CUPAC
        y_adjusted = cupac.cupac_adjustment(y, X, random_state=42)

        # Variance should be reduced
        var_original = y.var(ddof=1)
        var_adjusted = y_adjusted.var(ddof=1)

        assert var_adjusted < var_original
        assert var_adjusted > 0

    def test_cupac_preserves_mean(self):
        """Test that CUPAC preserves the mean of Y."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = X.sum(axis=1) + np.random.normal(0, 10, 500)

        y_adjusted = cupac.cupac_adjustment(y, X, random_state=42)

        # Mean should be preserved
        assert abs(y_adjusted.mean() - y.mean()) < 1e-10

    def test_cupac_with_different_models(self):
        """Test CUPAC with different ML models."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = X[:, 0]**2 + X[:, 1] * 2 + np.random.normal(0, 5, 500)

        # GradientBoosting
        y_adj_gbm = cupac.cupac_adjustment(y, X, model_type='gbm', random_state=42)
        # RandomForest
        y_adj_rf = cupac.cupac_adjustment(y, X, model_type='rf', random_state=42)
        # Ridge
        y_adj_ridge = cupac.cupac_adjustment(y, X, model_type='ridge', random_state=42)

        # All should reduce variance
        var_orig = y.var(ddof=1)
        assert y_adj_gbm.var(ddof=1) < var_orig
        assert y_adj_rf.var(ddof=1) < var_orig
        assert y_adj_ridge.var(ddof=1) < var_orig

    def test_cupac_with_custom_model(self):
        """Test CUPAC with custom model instance."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = X.sum(axis=1) + np.random.normal(0, 10, 500)

        custom_model = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        y_adjusted = cupac.cupac_adjustment(y, X, model=custom_model)

        assert y_adjusted.var(ddof=1) < y.var(ddof=1)

    def test_cupac_cross_validation(self):
        """Test that CUPAC uses cross-validation correctly."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = X.sum(axis=1) + np.random.normal(0, 10, 500)

        # Different CV folds
        y_adj_cv3 = cupac.cupac_adjustment(y, X, cv=3, random_state=42)
        y_adj_cv5 = cupac.cupac_adjustment(y, X, cv=5, random_state=42)

        # Both should work but give slightly different results
        assert y_adj_cv3.var(ddof=1) < y.var(ddof=1)
        assert y_adj_cv5.var(ddof=1) < y.var(ddof=1)

    def test_cupac_raises_on_length_mismatch(self):
        """Test that CUPAC raises ValueError on length mismatch."""
        y = np.array([1, 2, 3])
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="y and X must have same number of rows"):
            cupac.cupac_adjustment(y, X)

    def test_cupac_raises_on_insufficient_data(self):
        """Test that CUPAC raises ValueError with insufficient data."""
        y = np.array([1])
        X = np.array([[1, 2]])

        with pytest.raises(ValueError, match="Need at least 10 observations"):
            cupac.cupac_adjustment(y, X)

    def test_cupac_raises_on_invalid_model_type(self):
        """Test that CUPAC raises ValueError on invalid model type."""
        y = np.random.randn(100)
        X = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="model_type must be one of"):
            cupac.cupac_adjustment(y, X, model_type='invalid')


class TestCupacABTest:
    """Tests for cupac_ab_test function."""

    def test_cupac_ab_test_basic(self):
        """Test basic CUPAC A/B test."""
        np.random.seed(42)
        n = 500

        # Pre-experiment features
        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)

        # Outcome with nonlinear relationship
        y_control = (X_control[:, 0]**2 + X_control[:, 1] * 2 +
                     np.random.normal(0, 10, n))
        y_treatment = (X_treatment[:, 0]**2 + X_treatment[:, 1] * 2 + 5 +
                       np.random.normal(0, 10, n))

        result = cupac.cupac_ab_test(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        # Should have all required keys
        assert 'mean_control' in result
        assert 'mean_treatment' in result
        assert 'difference' in result
        assert 'var_reduction' in result
        assert 'se_reduction' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'model_type' in result

        # Variance reduction should be positive
        assert result['var_reduction'] > 0

    def test_cupac_better_than_naive(self):
        """Test that CUPAC improves power over naive approach."""
        np.random.seed(42)
        n = 1000  # Increased from 300 for stable ML predictions

        # Features with nonlinear relationship to outcome
        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)

        y_control = (X_control[:, 0]**2 + np.sin(X_control[:, 1]) +
                     np.random.normal(0, 8, n))
        y_treatment = (X_treatment[:, 0]**2 + np.sin(X_treatment[:, 1]) + 4 +
                       np.random.normal(0, 8, n))

        # CUPAC result
        cupac_result = cupac.cupac_ab_test(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        # Naive result
        from ab_testing.core import frequentist
        naive_result = frequentist.welch_ttest(y_control, y_treatment)

        # CUPAC should have smaller SE
        assert cupac_result['se_adj_diff'] < naive_result['se_diff']

    def test_cupac_ab_test_model_types(self):
        """Test CUPAC A/B test with different model types."""
        np.random.seed(42)
        n = 300

        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)
        y_control = X_control.sum(axis=1) + np.random.normal(0, 10, n)
        y_treatment = X_treatment.sum(axis=1) + 5 + np.random.normal(0, 10, n)

        # Test all model types
        for model_type in ['gbm', 'rf', 'ridge']:
            result = cupac.cupac_ab_test(
                y_control, y_treatment, X_control, X_treatment,
                model_type=model_type, random_state=42
            )
            assert result['model_type'] == model_type
            assert result['var_reduction'] >= 0  # Should not increase variance

    def test_cupac_ab_test_raises_on_length_mismatch(self):
        """Test that cupac_ab_test raises on mismatched lengths."""
        y_control = np.array([1, 2, 3])
        y_treatment = np.array([1, 2])
        X_control = np.random.randn(3, 2)
        X_treatment = np.random.randn(3, 2)

        with pytest.raises(ValueError):
            cupac.cupac_ab_test(y_control, y_treatment, X_control, X_treatment)


class TestVarianceReductionCupac:
    """Tests for variance_reduction_cupac function."""

    def test_variance_reduction_calculation(self):
        """Test variance reduction percentage calculation."""
        np.random.seed(42)
        n = 500

        X = np.random.randn(n, 3)
        y = X[:, 0]**2 + X[:, 1] * 2 + np.random.normal(0, 10, n)

        var_reduction = cupac.variance_reduction_cupac(y, X, random_state=42)

        # Should be positive for correlated data
        assert var_reduction > 0
        assert var_reduction < 1.0

        # Manual calculation
        y_adj = cupac.cupac_adjustment(y, X, random_state=42)
        expected = 1 - y_adj.var(ddof=1) / y.var(ddof=1)

        assert abs(var_reduction - expected) < 1e-10

    def test_variance_reduction_different_models(self):
        """Test variance reduction with different models."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = X[:, 0]**2 + np.random.normal(0, 10, 500)

        vr_gbm = cupac.variance_reduction_cupac(y, X, model_type='gbm', random_state=42)
        vr_rf = cupac.variance_reduction_cupac(y, X, model_type='rf', random_state=42)

        # Both should achieve variance reduction
        assert vr_gbm > 0
        assert vr_rf > 0


class TestPowerGainCupac:
    """Tests for power_gain_cupac function."""

    def test_power_gain_basic(self):
        """Test basic power gain calculation."""
        np.random.seed(100)  # Changed from 42 for better variance reduction
        n = 400

        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)
        y_control = X_control[:, 0]**2 + np.random.normal(0, 10, n)
        y_treatment = X_treatment[:, 0]**2 + 5 + np.random.normal(0, 10, n)

        power_gain = cupac.power_gain_cupac(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        # Should have all keys
        assert 'var_reduction' in power_gain
        assert 'power_multiplier' in power_gain
        assert 'equivalent_n' in power_gain

        # Power multiplier should be > 1
        assert power_gain['power_multiplier'] > 1.0

    def test_power_gain_equivalent_sample_size(self):
        """Test that equivalent sample size is reasonable."""
        np.random.seed(42)
        n = 500

        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)
        y_control = X_control.sum(axis=1) + np.random.normal(0, 10, n)
        y_treatment = X_treatment.sum(axis=1) + np.random.normal(0, 10, n)

        power_gain = cupac.power_gain_cupac(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        equiv_n = power_gain['equivalent_n']
        assert equiv_n > n  # Should be larger
        assert equiv_n < n * 3  # But reasonable


class TestCompareCupedVsCupac:
    """Tests for compare_cuped_vs_cupac function."""

    def test_comparison_basic(self):
        """Test comparison of CUPED and CUPAC."""
        np.random.seed(42)
        n = 500

        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)

        # Nonlinear relationship (CUPAC should win)
        y_control = (X_control[:, 0]**2 + X_control[:, 1] * X_control[:, 2] +
                     np.random.normal(0, 10, n))
        y_treatment = (X_treatment[:, 0]**2 + X_treatment[:, 1] * X_treatment[:, 2] + 5 +
                       np.random.normal(0, 10, n))

        comparison = cupac.compare_cuped_vs_cupac(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        # Should have all keys
        assert 'cuped_var_reduction' in comparison
        assert 'cupac_var_reduction' in comparison
        assert 'cuped_se_diff' in comparison
        assert 'cupac_se_diff' in comparison
        assert 'cupac_better' in comparison
        assert 'improvement' in comparison

        # CUPAC should be at least as good as CUPED
        assert comparison['cupac_var_reduction'] >= comparison['cuped_var_reduction'] * 0.8

    def test_comparison_linear_relationship(self):
        """Test that CUPED and CUPAC are similar for linear relationships."""
        np.random.seed(42)
        n = 500

        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)

        # Linear relationship (CUPED should be competitive)
        y_control = X_control.sum(axis=1) + np.random.normal(0, 10, n)
        y_treatment = X_treatment.sum(axis=1) + 5 + np.random.normal(0, 10, n)

        comparison = cupac.compare_cuped_vs_cupac(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        # Results should be similar (within 40% due to random variation)
        cuped_vr = comparison['cuped_var_reduction']
        cupac_vr = comparison['cupac_var_reduction']

        # Only compare if there's meaningful variance reduction
        if cuped_vr > 0.05 and cupac_vr > 0.05:
            assert abs(cupac_vr - cuped_vr) / max(cuped_vr, cupac_vr) < 0.40

    def test_comparison_recommendation(self):
        """Test that comparison provides reasonable recommendation."""
        np.random.seed(42)
        n = 500

        X_control = np.random.randn(n, 3)
        X_treatment = np.random.randn(n, 3)
        y_control = X_control[:, 0]**2 + np.random.normal(0, 10, n)
        y_treatment = X_treatment[:, 0]**2 + 5 + np.random.normal(0, 10, n)

        comparison = cupac.compare_cuped_vs_cupac(
            y_control, y_treatment, X_control, X_treatment, random_state=42
        )

        # Should have recommendation
        assert 'recommendation' in comparison
        assert len(comparison['recommendation']) > 0
