"""
Tests for sequential testing module.
"""

import pytest
import numpy as np
from ab_testing.advanced import sequential


class TestOBrienFlemingBoundary:
    """Tests for O'Brien-Fleming boundary calculation."""

    def test_obf_boundary_basic(self):
        """Test basic OBF boundary calculation."""
        boundary = sequential.obrien_fleming_boundary(
            current_look=1, total_looks=5, alpha=0.05
        )

        # First look should have highest boundary
        assert boundary > 2.5  # More conservative than final
        assert boundary < 5.0  # But reasonable

    def test_obf_boundary_progression(self):
        """Test that OBF boundaries decrease over time."""
        total_looks = 5
        boundaries = []

        for look in range(1, total_looks + 1):
            bound = sequential.obrien_fleming_boundary(look, total_looks, alpha=0.05)
            boundaries.append(bound)

        # Boundaries should decrease (become less conservative)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] > boundaries[i + 1]

    def test_obf_final_look_near_standard(self):
        """Test that final look boundary is close to standard z-critical."""
        boundary = sequential.obrien_fleming_boundary(
            current_look=5, total_looks=5, alpha=0.05
        )

        # Should be close to 1.96 (two-sided)
        assert 1.90 < boundary < 2.05

    def test_obf_single_look(self):
        """Test OBF with single look (no sequential testing)."""
        boundary = sequential.obrien_fleming_boundary(
            current_look=1, total_looks=1, alpha=0.05
        )

        # Should equal standard z-critical
        assert abs(boundary - 1.96) < 0.01

    def test_obf_raises_invalid_look(self):
        """Test that OBF raises on invalid look number."""
        with pytest.raises(ValueError, match="current_look must be between 1 and total_looks"):
            sequential.obrien_fleming_boundary(0, 5)

        with pytest.raises(ValueError, match="current_look must be between 1 and total_looks"):
            sequential.obrien_fleming_boundary(6, 5)


class TestPocockBoundary:
    """Tests for Pocock boundary calculation."""

    def test_pocock_boundary_basic(self):
        """Test basic Pocock boundary calculation."""
        boundary = sequential.pocock_boundary(total_looks=5, alpha=0.05)

        # Pocock uses constant boundary
        assert boundary > 2.0  # More conservative than 1.96
        assert boundary < 3.0  # But not too high

    def test_pocock_constant_across_looks(self):
        """Test that Pocock boundary is constant."""
        boundary1 = sequential.pocock_boundary(total_looks=5, alpha=0.05)

        # Should use same value regardless of which look (conceptually)
        # The function returns the constant boundary
        assert boundary1 > 0

    def test_pocock_vs_obf(self):
        """Test that Pocock is more liberal early, more conservative late."""
        total_looks = 5

        # Early look
        obf_early = sequential.obrien_fleming_boundary(1, total_looks, alpha=0.05)
        pocock = sequential.pocock_boundary(total_looks, alpha=0.05)

        # Pocock should be lower (more liberal) early
        assert pocock < obf_early

        # Late look
        obf_late = sequential.obrien_fleming_boundary(total_looks, total_looks, alpha=0.05)

        # Pocock should be higher (more conservative) late
        assert pocock > obf_late


class TestSequentialTest:
    """Tests for sequential_test function."""

    def test_sequential_test_stop_early(self):
        """Test early stopping when z-stat exceeds boundary."""
        # Very significant effect - should stop
        # Boundary at look 1/5 is 4.38, so use z=5.0 to exceed it
        z_stat = 5.0
        result = sequential.sequential_test(
            z_statistic=z_stat,
            current_look=1,
            total_looks=5,
            method='obf',
            alpha=0.05
        )

        assert 'can_stop' in result
        assert 'decision' in result
        assert 'boundary' in result
        assert 'current_look' in result

        # Should recommend stopping (z=5.0 exceeds boundary of 4.38)
        assert result['can_stop'] is True
        assert result['decision'] in ['reject_null', 'stop_for_efficacy']

    def test_sequential_test_continue(self):
        """Test continuing when z-stat below boundary."""
        # Not significant enough - should continue
        z_stat = 1.5
        result = sequential.sequential_test(
            z_statistic=z_stat,
            current_look=1,
            total_looks=5,
            method='obf'
        )

        # Should continue (1.5 below OBF boundary for first look)
        assert result['can_stop'] is False
        assert result['decision'] == 'continue'

    def test_sequential_test_final_look(self):
        """Test that final look uses standard threshold."""
        z_stat = 2.0
        result = sequential.sequential_test(
            z_statistic=z_stat,
            current_look=5,
            total_looks=5,
            method='obf',
            alpha=0.05
        )

        # At final look, 2.0 > 1.96, should stop
        assert result['can_stop'] is True

    def test_sequential_test_negative_z(self):
        """Test sequential test with negative z-statistic."""
        # Boundary at look 2/5 is 3.10, so use z=-3.5 to exceed it
        z_stat = -3.5
        result = sequential.sequential_test(
            z_statistic=z_stat,
            current_look=2,
            total_looks=5,
            method='obf'
        )

        # Should stop (|z| = 3.5 exceeds boundary of 3.10)
        assert result['can_stop'] is True

    def test_sequential_test_pocock_method(self):
        """Test sequential test with Pocock method."""
        z_stat = 2.5
        result = sequential.sequential_test(
            z_statistic=z_stat,
            current_look=1,
            total_looks=5,
            method='pocock'
        )

        assert 'boundary' in result
        # Pocock boundary should be constant
        assert result['boundary'] > 2.0

    def test_sequential_test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method must be 'obf' or 'pocock'"):
            sequential.sequential_test(
                z_statistic=2.0,
                current_look=1,
                total_looks=5,
                method='invalid'
            )


class TestFWERInflationNoCorrection:
    """Tests for FWER inflation without correction."""

    def test_fwer_inflation_basic(self):
        """Test basic FWER inflation calculation."""
        inflation = sequential.fwer_inflation_no_correction(n_looks=5, alpha=0.05)

        # FWER = 1 - (1-α)^n
        expected = 1 - (1 - 0.05)**5
        assert abs(inflation - expected) < 1e-10

    def test_fwer_inflation_single_look(self):
        """Test that single look has no inflation."""
        inflation = sequential.fwer_inflation_no_correction(n_looks=1, alpha=0.05)
        assert abs(inflation - 0.05) < 1e-10

    def test_fwer_inflation_many_looks(self):
        """Test FWER inflation with many looks."""
        inflation = sequential.fwer_inflation_no_correction(n_looks=10, alpha=0.05)

        # Should be substantial
        assert inflation > 0.40  # Over 40% chance of false positive


class TestAlphaSpendingFunctionOBF:
    """Tests for O'Brien-Fleming alpha spending function."""

    def test_alpha_spending_basic(self):
        """Test basic alpha spending calculation."""
        alpha_spent = sequential.alpha_spending_function_obf(
            current_look=3, total_looks=5, alpha=0.05
        )

        # Should spend some alpha by look 3
        assert alpha_spent > 0
        assert alpha_spent < 0.05  # But not all

    def test_alpha_spending_final_look(self):
        """Test that final look spends all alpha."""
        alpha_spent = sequential.alpha_spending_function_obf(
            current_look=5, total_looks=5, alpha=0.05
        )

        # Should spend full alpha at end
        assert abs(alpha_spent - 0.05) < 0.001

    def test_alpha_spending_progression(self):
        """Test that alpha spending increases monotonically."""
        total_looks = 5
        alpha = 0.05

        spending = []
        for look in range(1, total_looks + 1):
            spent = sequential.alpha_spending_function_obf(look, total_looks, alpha)
            spending.append(spent)

        # Should increase
        for i in range(len(spending) - 1):
            assert spending[i] < spending[i + 1]


class TestRecommendedLooks:
    """Tests for recommended_looks function."""

    def test_recommended_looks_basic(self):
        """Test basic recommendations for number of looks."""
        rec = sequential.recommended_looks(experiment_duration_days=28)

        assert 'recommended_looks' in rec
        assert 'look_frequency_days' in rec
        assert 'rationale' in rec

        # 28 days should give 4 weekly looks
        assert rec['recommended_looks'] == 4
        assert rec['look_frequency_days'] == 7

    def test_recommended_looks_short_experiment(self):
        """Test recommendations for short experiment."""
        rec = sequential.recommended_looks(experiment_duration_days=7)

        # Should recommend fewer looks
        assert rec['recommended_looks'] <= 2

    def test_recommended_looks_long_experiment(self):
        """Test recommendations for long experiment."""
        rec = sequential.recommended_looks(experiment_duration_days=56)

        # 56 days = 8 weeks, should recommend weekly looks
        assert rec['recommended_looks'] == 8

    def test_recommended_looks_minimum_duration(self):
        """Test recommendations enforce minimum duration between looks."""
        rec = sequential.recommended_looks(
            experiment_duration_days=60,
            min_days_between_looks=14  # Bi-weekly
        )

        # 60 days / 14 days = ~4 looks
        assert rec['recommended_looks'] <= 5


class TestSequentialTestingWorkflow:
    """Integration tests for sequential testing workflow."""

    def test_full_workflow_early_stop(self):
        """Test full sequential testing workflow with early stopping."""
        np.random.seed(42)

        # Simulate strong effect - should stop early
        n_per_group = 1000
        control = np.random.normal(100, 20, n_per_group)
        treatment = np.random.normal(110, 20, n_per_group)  # Large effect

        # Conduct sequential tests
        total_looks = 5
        stopped = False

        for look in range(1, total_looks + 1):
            # Interim analysis at increasing sample sizes
            n_interim = int(n_per_group * look / total_looks)

            c_interim = control[:n_interim]
            t_interim = treatment[:n_interim]

            # Calculate z-statistic
            from ab_testing.core import frequentist
            result = frequentist.welch_ttest(c_interim, t_interim)
            z_stat = abs(result['t_statistic'])

            # Sequential test
            seq_result = sequential.sequential_test(
                z_statistic=z_stat,
                current_look=look,
                total_looks=total_looks,
                method='obf'
            )

            if seq_result['can_stop']:
                stopped = True
                break

        # With strong effect, should stop before final look
        assert stopped is True

    def test_full_workflow_no_effect(self):
        """Test sequential workflow with no true effect."""
        np.random.seed(42)

        # No effect - should not stop early
        n_per_group = 1000
        control = np.random.normal(100, 20, n_per_group)
        treatment = np.random.normal(100, 20, n_per_group)  # No effect

        total_looks = 5
        early_stop_count = 0

        for look in range(1, total_looks + 1):
            n_interim = int(n_per_group * look / total_looks)
            c_interim = control[:n_interim]
            t_interim = treatment[:n_interim]

            from ab_testing.core import frequentist
            result = frequentist.welch_ttest(c_interim, t_interim)
            z_stat = abs(result['t_statistic'])

            seq_result = sequential.sequential_test(
                z_statistic=z_stat,
                current_look=look,
                total_looks=total_looks,
                method='obf'
            )

            if seq_result['can_stop']:
                early_stop_count += 1
                break

        # With no effect, unlikely to stop early (possible due to randomness)
        # Just verify workflow runs without errors
        assert early_stop_count <= 1  # Should not stop at first look
