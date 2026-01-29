"""Unit tests for randomization quality checks."""

import pytest
import numpy as np
from ab_testing.core import randomization


class TestSRMCheck:
    """Tests for Sample Ratio Mismatch check."""

    def test_srm_perfect_balance(self):
        """Test SRM check with perfect 50/50 split."""
        result = randomization.srm_check(n_control=5000, n_treatment=5000)

        assert result['n_control'] == 5000
        assert result['n_treatment'] == 5000
        assert result['ratio_control'] == 0.5
        assert result['ratio_treatment'] == 0.5
        assert result['p_value'] == 1.0  # Perfect match
        assert not result['srm_detected']

    def test_srm_small_imbalance_ok(self):
        """Test SRM check with small acceptable imbalance."""
        result = randomization.srm_check(n_control=5050, n_treatment=4950)

        assert result['p_value'] > 0.01  # Should pass SRM check
        assert not result['srm_detected']

    def test_srm_large_imbalance_detected(self):
        """Test SRM check detects large imbalance."""
        result = randomization.srm_check(n_control=53000, n_treatment=47000)

        assert result['p_value'] < 0.01  # Should fail SRM check
        assert result['srm_detected']

    def test_srm_custom_ratio(self):
        """Test SRM check with custom allocation ratio (70/30)."""
        result = randomization.srm_check(
            n_control=7000,
            n_treatment=3000,
            expected_ratio=[0.7, 0.3]
        )

        assert result['expected_control'] == 7000
        assert result['expected_treatment'] == 3000
        assert not result['srm_detected']

    def test_srm_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="Sample sizes must be positive"):
            randomization.srm_check(n_control=0, n_treatment=1000)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            randomization.srm_check(n_control=5000, n_treatment=5000,
                                   expected_ratio=[0.6, 0.6])

        with pytest.raises(ValueError, match="must be positive"):
            randomization.srm_check(n_control=5000, n_treatment=5000,
                                   expected_ratio=[-0.5, 1.5])


class TestMultiGroupSRMCheck:
    """Tests for multi-group SRM check (A/B/C tests)."""

    def test_multi_group_equal_allocation(self):
        """Test multi-group SRM with equal allocation."""
        result = randomization.multi_group_srm_check([3333, 3333, 3334])

        assert len(result['observed_counts']) == 3
        assert len(result['expected_counts']) == 3
        assert result['df'] == 2
        assert not result['srm_detected']

    def test_multi_group_custom_ratio(self):
        """Test multi-group SRM with custom allocation ratio."""
        result = randomization.multi_group_srm_check(
            observed_counts=[5000, 3000, 2000],
            expected_ratio=[0.5, 0.3, 0.2]
        )

        assert abs(result['expected_counts'][0] - 5000) < 1
        assert abs(result['expected_counts'][1] - 3000) < 1
        assert abs(result['expected_counts'][2] - 2000) < 1
        assert not result['srm_detected']

    def test_multi_group_imbalance(self):
        """Test multi-group SRM detects imbalance."""
        result = randomization.multi_group_srm_check([5500, 3000, 1500])

        # Should detect imbalance (expected 33/33/33, got 55/30/15)
        assert result['srm_detected']

    def test_multi_group_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="Need at least 2 groups"):
            randomization.multi_group_srm_check([1000])

        with pytest.raises(ValueError, match="must be positive"):
            randomization.multi_group_srm_check([1000, 0, 500])


class TestBalanceCheck:
    """Tests for covariate balance check."""

    def test_balance_check_balanced(self):
        """Test balance check with balanced covariates."""
        np.random.seed(42)
        # Same distribution for both groups
        control_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))
        treatment_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))

        result = randomization.balance_check(
            control_cov, treatment_cov,
            covariate_names=['age', 'pre_revenue', 'pre_sessions']
        )

        # Should find 0 or very few imbalanced covariates (by chance)
        assert result['n_imbalanced'] <= 1  # At most 1 false positive
        assert result['balance_ok']

    def test_balance_check_imbalanced(self):
        """Test balance check detects imbalance."""
        np.random.seed(42)
        # Different distributions
        control_cov = np.random.normal([35, 100, 10], [10, 50, 5], (500, 3))
        treatment_cov = np.random.normal([40, 120, 12], [10, 50, 5], (500, 3))

        result = randomization.balance_check(
            control_cov, treatment_cov,
            covariate_names=['age', 'pre_revenue', 'pre_sessions']
        )

        # Should detect imbalance in all 3 covariates
        assert result['n_imbalanced'] >= 2
        # May or may not fail overall (depends on 10% threshold)

    def test_balance_check_single_covariate(self):
        """Test balance check with single covariate."""
        np.random.seed(42)
        control_cov = np.random.normal(35, 10, 500)
        treatment_cov = np.random.normal(35, 10, 500)

        result = randomization.balance_check(
            control_cov, treatment_cov,
            covariate_names=['age']
        )

        assert len(result['covariate_tests']) == 1
        assert result['covariate_tests'][0]['name'] == 'age'
        assert result['balance_ok']

    def test_balance_check_default_names(self):
        """Test balance check with default covariate names."""
        np.random.seed(42)
        control_cov = np.random.normal([35, 100], [10, 50], (500, 2))
        treatment_cov = np.random.normal([35, 100], [10, 50], (500, 2))

        result = randomization.balance_check(control_cov, treatment_cov)

        assert result['covariate_tests'][0]['name'] == 'covariate_0'
        assert result['covariate_tests'][1]['name'] == 'covariate_1'

    def test_balance_check_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        control_cov = np.random.normal(35, 10, (500, 3))
        treatment_cov = np.random.normal(35, 10, (500, 2))

        with pytest.raises(ValueError, match="same number of covariates"):
            randomization.balance_check(control_cov, treatment_cov)

        with pytest.raises(ValueError, match="Need 3 covariate names"):
            randomization.balance_check(
                np.random.normal(35, 10, (500, 3)),
                np.random.normal(35, 10, (500, 3)),
                covariate_names=['age', 'revenue']  # Only 2 names for 3 covariates
            )


class TestTwoStageSRMGating:
    """
    Tests for two-stage SRM gating: statistical + practical significance.

    Motivation: Large samples make tiny deviations statistically significant
    but not practically meaningful. Two-stage gating distinguishes:
    - Stage A (statistical): p < alpha
    - Stage B (practical): deviation > threshold

    Severity levels:
    - srm_severe: BOTH statistical AND practical (hard gate)
    - srm_warning: statistical ONLY (proceed with caution)
    - no SRM: neither statistical nor practical (all clear)
    """

    def test_no_srm_clean_case(self):
        """Test when neither statistical nor practical SRM detected."""
        # 50/50 split with tiny natural variation
        result = randomization.srm_check(
            n_control=50100,
            n_treatment=49900,
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.01  # 1 percentage point
        )

        # Should pass both stages
        assert not result['srm_detected'], "No statistical SRM"
        assert not result['practical_significant'], "No practical SRM"
        assert not result['srm_severe'], "Not severe"
        assert not result['srm_warning'], "No warning needed"

    def test_srm_warning_borderline_case(self):
        """Test borderline case: statistical but not practical (large sample)."""
        # Cookie Cats scenario: 90K sample, 49.56% vs 50.44%
        # This creates p < 0.01 but deviation is only 0.44pp < 1pp threshold
        result = randomization.srm_check(
            n_control=44700,  # 49.56%
            n_treatment=45489,  # 50.44%
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.01  # 1pp threshold
        )

        # Should detect statistical but not practical
        assert result['srm_detected'], "Statistical SRM detected (p < 0.01)"
        assert not result['practical_significant'], "Deviation < 1pp threshold"
        assert not result['srm_severe'], "NOT severe (only statistical)"
        assert result['srm_warning'], "Should be WARNING (borderline case)"

        # Verify deviation magnitude
        assert abs(result['max_pp_deviation']) < 0.01, "Deviation should be < 1pp"

    def test_srm_severe_both_conditions(self):
        """Test severe SRM: both statistical AND practical."""
        # Large sample with large deviation: 53% vs 47% (3pp deviation)
        result = randomization.srm_check(
            n_control=53000,
            n_treatment=47000,
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.01  # 1pp threshold
        )

        # Should fail both stages
        assert result['srm_detected'], "Statistical SRM detected"
        assert result['practical_significant'], "Deviation > 1pp threshold"
        assert result['srm_severe'], "SEVERE (both conditions met)"
        assert not result['srm_warning'], "Not just a warning - this is severe"

        # Verify deviation magnitude
        assert abs(result['max_pp_deviation']) > 0.01, "Deviation should be > 1pp"

    def test_small_sample_statistical_not_practical(self):
        """Test small sample with large deviation that's not statistically significant."""
        # Small sample: 52% vs 48% (2pp deviation) but n=200 total
        result = randomization.srm_check(
            n_control=104,  # 52%
            n_treatment=96,  # 48%
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.01  # 1pp threshold
        )

        # Deviation is large (2pp) but sample too small for statistical significance
        assert not result['srm_detected'], "p > 0.01 (small sample)"
        assert result['practical_significant'], "Deviation > 1pp"
        assert not result['srm_severe'], "Not severe (no statistical significance)"
        assert not result['srm_warning'], "No warning (no statistical SRM)"

    def test_count_threshold_with_small_allocation(self):
        """Test count-based threshold for imbalanced allocations."""
        # 15/85 split (Criteo scenario) with 100 expected in control
        # Actual: 85 control (15 fewer than expected)
        result = randomization.srm_check(
            n_control=85,  # Expected: 100
            n_treatment=915,  # Expected: 900
            expected_ratio=[0.1, 0.9],
            alpha=0.01,
            pp_threshold=0.02,  # 2pp threshold
            count_threshold=20  # 20 users absolute threshold
        )

        # Count deviation: |85 - 100| = 15 < 20 threshold
        # PP deviation: |8.5% - 10%| = 1.5pp < 2pp threshold
        assert result['count_deviation_control'] == 15
        assert not result['practical_significant'], "Below both thresholds"

    def test_imbalanced_allocation_two_stage(self):
        """Test two-stage gating with intentionally imbalanced allocation."""
        # Criteo 15/85 allocation with large sample (1M total)
        # Tiny deviation: 15.02% vs 84.98% (0.02pp off)
        result = randomization.srm_check(
            n_control=150200,  # 15.02%
            n_treatment=849800,  # 84.98%
            expected_ratio=[0.15, 0.85],
            alpha=0.01,
            pp_threshold=0.01  # 1pp threshold
        )

        # Should pass both stages (deviation tiny)
        assert not result['srm_detected'], "No statistical SRM"
        assert not result['practical_significant'], "Deviation < 1pp"
        assert not result['srm_severe'], "Not severe"
        assert not result['srm_warning'], "No warning"

    def test_pp_threshold_sensitivity(self):
        """Test sensitivity to pp_threshold parameter."""
        # Same data, different thresholds
        # Use Cookie Cats actual numbers: 49.56% vs 50.44%
        # Deviation from expected 50%: 0.44pp per group (NOT 0.88pp total difference)
        # Large sample (90K) makes this statistically significant
        n_control = 44700  # 49.56% (0.44pp below 50%)
        n_treatment = 45489  # 50.44% (0.44pp above 50%)

        # Strict threshold (0.3pp) - should be practical
        result_strict = randomization.srm_check(
            n_control=n_control,
            n_treatment=n_treatment,
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.003  # 0.3pp threshold (strict)
        )

        # Lenient threshold (1pp) - should NOT be practical
        result_lenient = randomization.srm_check(
            n_control=n_control,
            n_treatment=n_treatment,
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.01  # 1pp threshold (standard)
        )

        # Both detect statistical SRM (large sample, 0.44pp deviation per group)
        assert result_strict['srm_detected'], "Statistical SRM with 90K sample"
        assert result_lenient['srm_detected'], "Statistical SRM with 90K sample"

        # But practical significance differs
        assert result_strict['practical_significant'], "0.44pp > 0.3pp threshold"
        assert not result_lenient['practical_significant'], "0.44pp < 1pp threshold"

        # Therefore severity differs
        assert result_strict['srm_severe'], "Strict: both conditions met"
        assert result_lenient['srm_warning'], "Lenient: warning only"

    def test_return_fields_complete(self):
        """Test that all expected two-stage fields are returned."""
        result = randomization.srm_check(
            n_control=45000,
            n_treatment=55000,
            expected_ratio=[0.5, 0.5],
            alpha=0.01,
            pp_threshold=0.01,
            count_threshold=500
        )

        # Original fields
        assert 'n_control' in result
        assert 'n_treatment' in result
        assert 'chi2_statistic' in result
        assert 'p_value' in result
        assert 'srm_detected' in result

        # New two-stage fields
        assert 'pp_deviation_control' in result
        assert 'pp_deviation_treatment' in result
        assert 'max_pp_deviation' in result
        assert 'count_deviation_control' in result
        assert 'count_deviation_treatment' in result
        assert 'max_count_deviation' in result
        assert 'practical_significant' in result
        assert 'srm_severe' in result
        assert 'srm_warning' in result
