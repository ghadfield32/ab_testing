"""Unit tests for Bayesian A/B testing module."""

import pytest
import numpy as np
from ab_testing.core import bayesian


class TestBetaBinomialABTest:
    """Tests for Beta-Binomial Bayesian A/B test."""

    def test_beta_binomial_no_effect(self):
        """Test Beta-Binomial when conversion rates are equal."""
        result = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=50, n_treatment=500,
            random_state=42
        )

        # P(Treatment > Control) should be around 50%
        assert 0.45 < result['prob_treatment_better'] < 0.55
        # Expected lift should be near 0
        assert abs(result['expected_lift']) < 0.05
        # Expected losses should be similar
        assert abs(result['expected_loss_control'] - result['expected_loss_treatment']) < 0.002

    def test_beta_binomial_positive_effect(self):
        """Test Beta-Binomial with positive treatment effect."""
        result = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=70, n_treatment=500,
            random_state=42
        )

        # P(Treatment > Control) should be high
        assert result['prob_treatment_better'] > 0.95
        # Expected lift should be positive
        assert result['expected_lift'] > 0.30
        # Should recommend treatment
        assert result['recommendation'] == 'treatment'
        # Expected loss for choosing treatment should be low
        assert result['expected_loss_treatment'] < result['expected_loss_control']

    def test_beta_binomial_credible_interval(self):
        """Test credible interval calculation."""
        result = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            random_state=42
        )

        ci_lower, ci_upper = result['credible_interval']
        # CI should be reasonable range around expected lift
        assert ci_lower < result['expected_lift'] < ci_upper
        # For positive effect, CI should likely not contain 0
        if result['prob_treatment_better'] > 0.95:
            assert ci_lower > 0

    def test_beta_binomial_posterior_parameters(self):
        """Test posterior parameter calculation."""
        result = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            prior_alpha=1.0,
            prior_beta=1.0,
            random_state=42
        )

        # Posterior should be Beta(prior_alpha + x, prior_beta + n - x)
        assert result['posterior_control'][0] == 1 + 50
        assert result['posterior_control'][1] == 1 + (500 - 50)
        assert result['posterior_treatment'][0] == 1 + 60
        assert result['posterior_treatment'][1] == 1 + (500 - 60)

    def test_beta_binomial_custom_prior(self):
        """Test with custom prior (Jeffreys prior)."""
        result = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            prior_alpha=0.5,
            prior_beta=0.5,
            random_state=42
        )

        # Posterior should use custom prior
        assert result['posterior_control'][0] == 0.5 + 50
        assert result['posterior_control'][1] == 0.5 + (500 - 50)

    def test_beta_binomial_reproducibility(self):
        """Test reproducibility with random_state."""
        result1 = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            random_state=42
        )
        result2 = bayesian.beta_binomial_ab_test(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            random_state=42
        )

        assert result1['prob_treatment_better'] == result2['prob_treatment_better']
        assert result1['expected_lift'] == result2['expected_lift']

    def test_beta_binomial_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="must be non-negative"):
            bayesian.beta_binomial_ab_test(-1, 500, 50, 500)

        with pytest.raises(ValueError, match="must be positive"):
            bayesian.beta_binomial_ab_test(50, 0, 50, 500)

        with pytest.raises(ValueError, match="cannot exceed sample size"):
            bayesian.beta_binomial_ab_test(600, 500, 50, 500)

        with pytest.raises(ValueError, match="Prior parameters must be positive"):
            bayesian.beta_binomial_ab_test(50, 500, 50, 500, prior_alpha=0)


class TestNormalABTest:
    """Tests for Normal Bayesian A/B test."""

    def test_normal_no_effect(self):
        """Test Normal model when means are equal."""
        np.random.seed(123)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(100, 20, 500)

        result = bayesian.normal_ab_test(control, treatment, random_state=42)

        # P(Treatment > Control) should be around 50%
        assert 0.40 < result['prob_treatment_better'] < 0.60
        # Expected difference should be near 0
        assert abs(result['expected_difference']) < 5

    def test_normal_positive_effect(self):
        """Test Normal model with positive treatment effect."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = bayesian.normal_ab_test(control, treatment, random_state=42)

        # P(Treatment > Control) should be very high
        assert result['prob_treatment_better'] > 0.99
        # Expected difference should be positive
        assert result['expected_difference'] > 10
        # Should recommend treatment
        assert result['recommendation'] == 'treatment'

    def test_normal_credible_interval(self):
        """Test credible interval calculation."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = bayesian.normal_ab_test(control, treatment, random_state=42)

        ci_lower, ci_upper = result['credible_interval']
        # CI should contain expected difference
        assert ci_lower < result['expected_difference'] < ci_upper
        # For strong effect, CI should not contain 0
        assert ci_lower > 0

    def test_normal_weak_prior(self):
        """Test with weak prior (default)."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(115, 20, 500)

        result = bayesian.normal_ab_test(
            control, treatment,
            prior_mean=0.0,
            prior_std=1000.0,
            random_state=42
        )

        # With weak prior and strong data, posterior should be driven by data
        assert 10 < result['posterior_mean'] < 20

    def test_normal_reproducibility(self):
        """Test reproducibility with random_state."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 100)
        treatment = np.random.normal(115, 20, 100)

        result1 = bayesian.normal_ab_test(control, treatment, random_state=42)
        result2 = bayesian.normal_ab_test(control, treatment, random_state=42)

        assert result1['prob_treatment_better'] == result2['prob_treatment_better']
        assert result1['expected_difference'] == result2['expected_difference']

    def test_normal_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            bayesian.normal_ab_test(np.array([1]), np.array([1, 2]))

        with pytest.raises(ValueError, match="must be positive"):
            bayesian.normal_ab_test(
                np.random.normal(100, 20, 100),
                np.random.normal(115, 20, 100),
                prior_std=0
            )


class TestProbabilityToBeatThreshold:
    """Tests for probability to beat threshold."""

    def test_threshold_low(self):
        """Test probability to beat low threshold."""
        prob = bayesian.probability_to_beat_threshold(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            threshold=0.05,  # 5% lift
            random_state=42
        )

        # With 10% -> 12% conversion, P(lift > 5%) should be high
        assert prob > 0.50

    def test_threshold_high(self):
        """Test probability to beat high threshold."""
        prob = bayesian.probability_to_beat_threshold(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            threshold=0.50,  # 50% lift (unrealistic)
            random_state=42
        )

        # P(lift > 50%) should be very low (widen tolerance for Monte Carlo variation)
        assert prob < 0.11

    def test_threshold_reproducibility(self):
        """Test reproducibility with random_state."""
        prob1 = bayesian.probability_to_beat_threshold(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            threshold=0.10,
            random_state=42
        )
        prob2 = bayesian.probability_to_beat_threshold(
            x_control=50, n_control=500,
            x_treatment=60, n_treatment=500,
            threshold=0.10,
            random_state=42
        )

        assert prob1 == prob2


class TestStoppingRuleBayesian:
    """Tests for Bayesian stopping rule."""

    def test_stopping_rule_treatment_wins(self):
        """Test stopping when treatment clearly wins."""
        should_stop, decision = bayesian.stopping_rule_bayesian(
            prob_treatment_better=0.97
        )

        assert should_stop
        assert decision == 'treatment'

    def test_stopping_rule_control_wins(self):
        """Test stopping when control clearly wins."""
        should_stop, decision = bayesian.stopping_rule_bayesian(
            prob_treatment_better=0.03
        )

        assert should_stop
        assert decision == 'control'

    def test_stopping_rule_continue(self):
        """Test continuing when evidence is insufficient."""
        should_stop, decision = bayesian.stopping_rule_bayesian(
            prob_treatment_better=0.75
        )

        assert not should_stop
        assert decision == 'continue'

    def test_stopping_rule_custom_thresholds(self):
        """Test stopping rule with custom thresholds."""
        should_stop, decision = bayesian.stopping_rule_bayesian(
            prob_treatment_better=0.92,
            threshold_high=0.90,
            threshold_low=0.10
        )

        assert should_stop
        assert decision == 'treatment'
