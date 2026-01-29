"""
Tests for novelty effect detection module.
"""

import pytest
import numpy as np
import pandas as pd
from ab_testing.diagnostics import novelty


class TestDetectNoveltyEffect:
    """Tests for detect_novelty_effect function."""

    def test_detect_novelty_with_decay(self):
        """Test detection when true novelty effect exists."""
        np.random.seed(42)
        n_days = 30
        time_idx = np.arange(n_days)

        # Simulate decaying effect
        true_effect = 0.10 * np.exp(-0.1 * time_idx) + 0.03

        control = np.random.normal(0.50, 0.01, n_days)
        treatment = control + true_effect + np.random.normal(0, 0.005, n_days)

        result = novelty.detect_novelty_effect(control, treatment)

        assert 'novelty_detected' in result
        assert 'early_effect' in result
        assert 'late_effect' in result
        assert 'effect_decay' in result

        # Should detect novelty (early > late)
        assert result['early_effect'] > result['late_effect']

    def test_no_novelty_stable_effect(self):
        """Test when effect is stable (no novelty)."""
        np.random.seed(42)
        n_days = 30

        # Stable effect
        stable_effect = 0.05

        control = np.random.normal(0.50, 0.01, n_days)
        treatment = control + stable_effect + np.random.normal(0, 0.005, n_days)

        result = novelty.detect_novelty_effect(control, treatment)

        # Should not detect novelty
        # Early and late effects should be similar
        assert abs(result['early_effect'] - result['late_effect']) < 0.02

    def test_detect_novelty_custom_periods(self):
        """Test novelty detection with custom early/late periods."""
        np.random.seed(42)
        n_days = 40

        control = np.random.normal(0.50, 0.01, n_days)
        treatment = control + 0.05 + np.random.normal(0, 0.005, n_days)

        result = novelty.detect_novelty_effect(
            control, treatment, early_period=0.3, late_period=0.3
        )

        # Should use 30% for early and late
        expected_early_days = int(40 * 0.3)
        assert result['early_period_days'] == expected_early_days

    def test_detect_novelty_raises_insufficient_data(self):
        """Test that insufficient data raises error."""
        control = np.array([0.5, 0.5, 0.5])
        treatment = np.array([0.55, 0.55, 0.55])

        with pytest.raises(ValueError, match="at least 10 time points"):
            novelty.detect_novelty_effect(control, treatment)

    def test_detect_novelty_raises_length_mismatch(self):
        """Test that length mismatch raises error."""
        control = np.random.normal(0.5, 0.01, 30)
        treatment = np.random.normal(0.5, 0.01, 25)

        with pytest.raises(ValueError, match="must have same length"):
            novelty.detect_novelty_effect(control, treatment)


class TestFitDecayCurve:
    """Tests for fit_decay_curve function."""

    def test_fit_exponential_decay(self):
        """Test fitting exponential decay curve."""
        np.random.seed(42)
        time = np.arange(30)
        true_a, true_b, true_c = 0.07, 0.1, 0.03
        effects = true_a * np.exp(-true_b * time) + true_c + np.random.normal(0, 0.005, 30)

        result = novelty.fit_decay_curve(time, effects, model='exponential')

        assert 'model' in result
        assert result['model'] == 'exponential'
        assert 'parameters' in result
        assert 'initial_effect' in result
        assert 'asymptotic_effect' in result
        assert 'half_life' in result
        assert 'r_squared' in result

        # R² should be high for good fit
        assert result['r_squared'] > 0.70

    def test_fit_linear_decay(self):
        """Test fitting linear decay curve."""
        np.random.seed(42)
        time = np.arange(30)
        effects = 0.10 - 0.002 * time + np.random.normal(0, 0.005, 30)

        result = novelty.fit_decay_curve(time, effects, model='linear')

        assert result['model'] == 'linear'
        assert 'initial_effect' in result
        assert result['r_squared'] > 0.50

    def test_fit_raises_insufficient_data(self):
        """Test that insufficient data raises error."""
        time = np.array([0, 1, 2])
        effects = np.array([0.1, 0.08, 0.06])

        with pytest.raises(ValueError, match="at least 5 data points"):
            novelty.fit_decay_curve(time, effects)


class TestRecommendHoldoutDuration:
    """Tests for recommend_holdout_duration function."""

    def test_recommend_stable_effect(self):
        """Test recommendation when effect is stable."""
        # Stable effect
        effects = np.array([0.05] * 30)
        time = np.arange(30)

        result = novelty.recommend_holdout_duration(effects, time)

        assert 'recommended_weeks' in result
        assert 'current_stability' in result
        assert 'rationale' in result

        # Stable effect should recommend minimum weeks
        assert result['recommended_weeks'] == 4

    def test_recommend_decaying_effect(self):
        """Test recommendation when effect is decaying."""
        # Strong decay
        time = np.arange(30)
        effects = 0.10 * np.exp(-0.15 * time) + 0.01

        result = novelty.recommend_holdout_duration(effects, time)

        # Decaying effect should recommend longer holdout
        assert result['recommended_weeks'] >= 6

    def test_recommend_insufficient_data(self):
        """Test recommendation with insufficient data."""
        effects = np.array([0.05, 0.04, 0.03])
        time = np.arange(3)

        result = novelty.recommend_holdout_duration(effects, time)

        # Should use minimum with rationale
        assert 'Insufficient data' in result['rationale']


class TestCohortAnalysis:
    """Tests for cohort_analysis function."""

    def test_cohort_analysis_basic(self):
        """Test basic cohort analysis."""
        np.random.seed(42)

        # Create sample data
        n = 1000
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = []

        for date in dates:
            # New users
            for _ in range(20):
                data.append({
                    'date': date,
                    'is_new_user': True,
                    'conversion': np.random.binomial(1, 0.12),
                    'treatment': np.random.binomial(1, 0.5)
                })

            # Existing users
            for _ in range(30):
                data.append({
                    'date': date,
                    'is_new_user': False,
                    'conversion': np.random.binomial(1, 0.10),
                    'treatment': np.random.binomial(1, 0.5)
                })

        df = pd.DataFrame(data)

        result = novelty.cohort_analysis(
            df,
            date_col='date',
            cohort_col='is_new_user',
            metric_col='conversion',
            treatment_col='treatment'
        )

        # Should have results for both cohorts
        assert len(result) == 2
        assert 'cohort_True' in result or 'cohort_False' in result

    def test_cohort_analysis_raises_missing_column(self):
        """Test that missing column raises error."""
        df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=10)})

        with pytest.raises(ValueError, match="not found in data"):
            novelty.cohort_analysis(
                df,
                date_col='date',
                cohort_col='missing',
                metric_col='metric',
                treatment_col='treatment'
            )


class TestIntegration:
    """Integration tests for novelty detection workflow."""

    def test_full_workflow_with_novelty(self):
        """Test complete novelty detection workflow."""
        np.random.seed(42)
        n_days = 30
        time = np.arange(n_days)

        # Simulate novelty effect
        true_effect = 0.10 * np.exp(-0.1 * time) + 0.03

        control = np.random.normal(0.50, 0.01, n_days)
        treatment = control + true_effect + np.random.normal(0, 0.005, n_days)

        # Detect novelty
        detection = novelty.detect_novelty_effect(control, treatment)

        # Fit decay curve
        effects = treatment - control
        decay_fit = novelty.fit_decay_curve(time, effects, model='exponential')

        # Recommend holdout
        holdout_rec = novelty.recommend_holdout_duration(effects, time)

        # Verify complete workflow
        assert detection['novelty_detected'] is not None
        assert decay_fit['r_squared'] > 0
        assert holdout_rec['recommended_weeks'] >= 4

        # If novelty detected, should recommend longer holdout
        if detection['novelty_detected']:
            assert holdout_rec['recommended_weeks'] >= 4
