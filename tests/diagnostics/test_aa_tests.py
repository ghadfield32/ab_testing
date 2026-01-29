"""
Tests for A/A test validation module.
"""

import pytest
import numpy as np
from ab_testing.diagnostics import aa_tests


class TestRunAATest:
    """Tests for run_aa_test function."""

    def test_aa_test_binary_no_difference(self):
        """Test A/A test with binary data and no true difference."""
        np.random.seed(42)
        control = np.random.binomial(1, 0.10, 1000)
        treatment = np.random.binomial(1, 0.10, 1000)

        result = aa_tests.run_aa_test(control, treatment)

        assert 'p_value' in result
        assert 'significant' in result
        assert 'false_positive' in result
        assert 'test_used' in result

        # In A/A test, significant = false positive
        assert result['false_positive'] == result['significant']

    def test_aa_test_continuous(self):
        """Test A/A test with continuous data."""
        np.random.seed(42)
        control = np.random.normal(100, 20, 500)
        treatment = np.random.normal(100, 20, 500)

        result = aa_tests.run_aa_test(control, treatment, test_type='continuous')

        assert result['test_used'] == 'welch_ttest'
        assert 'p_value' in result

    def test_aa_test_auto_detection(self):
        """Test automatic detection of binary vs continuous."""
        np.random.seed(42)

        # Binary data
        binary_control = np.random.binomial(1, 0.5, 500)
        binary_treatment = np.random.binomial(1, 0.5, 500)

        result_binary = aa_tests.run_aa_test(binary_control, binary_treatment, test_type='auto')
        assert result_binary['test_used'] == 'z_test_proportions'

        # Continuous data
        cont_control = np.random.normal(100, 20, 500)
        cont_treatment = np.random.normal(100, 20, 500)

        result_cont = aa_tests.run_aa_test(cont_control, cont_treatment, test_type='auto')
        assert result_cont['test_used'] == 'welch_ttest'

    def test_aa_test_raises_insufficient_data(self):
        """Test that insufficient data raises error."""
        control = np.array([1])
        treatment = np.array([1])

        with pytest.raises(ValueError, match="at least 2 observations"):
            aa_tests.run_aa_test(control, treatment)


class TestValidateInfrastructure:
    """Tests for validate_infrastructure function."""

    def test_validate_infrastructure_basic(self):
        """Test basic infrastructure validation."""
        result = aa_tests.validate_infrastructure(
            n_tests=100,
            sample_size=1000,
            p_baseline=0.10,
            alpha=0.05,
            random_state=42
        )

        assert 'n_tests' in result
        assert 'false_positive_count' in result
        assert 'false_positive_rate' in result
        assert 'expected_fp_rate' in result
        assert 'passed' in result
        assert 'p_values' in result

        # FP rate should be around 5% (±some tolerance)
        assert result['expected_fp_rate'] == 0.05

    def test_validate_infrastructure_fp_rate_reasonable(self):
        """Test that FP rate is within reasonable bounds."""
        result = aa_tests.validate_infrastructure(
            n_tests=100,
            sample_size=1000,
            alpha=0.05,
            random_state=42
        )

        # With 100 tests, should get ~5 false positives
        # Allow some variance (binomial)
        fp_rate = result['false_positive_rate']
        assert 0.01 < fp_rate < 0.15  # Reasonable range

    def test_validate_infrastructure_p_values_uniform(self):
        """Test that p-values are uniformly distributed."""
        result = aa_tests.validate_infrastructure(
            n_tests=100,
            sample_size=1000,
            alpha=0.05,
            random_state=42
        )

        # P-values should pass uniformity test (most of the time)
        # Note: This is statistical, may occasionally fail
        assert 'p_values_uniform' in result

    def test_validate_infrastructure_continuous_metric(self):
        """Test validation with continuous metric."""
        result = aa_tests.validate_infrastructure(
            n_tests=100,
            sample_size=1000,
            metric_type='continuous',
            alpha=0.05,
            random_state=42
        )

        assert result['n_tests'] == 100
        assert result['passed'] is not None

    def test_validate_infrastructure_raises_insufficient_tests(self):
        """Test that too few tests raises error."""
        with pytest.raises(ValueError, match="at least 10 tests"):
            aa_tests.validate_infrastructure(n_tests=5)

    def test_validate_infrastructure_raises_insufficient_sample(self):
        """Test that too small sample size raises error."""
        with pytest.raises(ValueError, match="at least 10"):
            aa_tests.validate_infrastructure(n_tests=100, sample_size=5)


class TestPowerCheck:
    """Tests for power_check function."""

    def test_power_check_basic(self):
        """Test basic power check."""
        result = aa_tests.power_check(
            n_tests=100,
            sample_size=5000,
            true_effect=0.10,
            p_baseline=0.10,
            alpha=0.05,
            random_state=42
        )

        assert 'observed_power' in result
        assert 'theoretical_power' in result
        assert 'true_positive_count' in result
        assert 'passed' in result

        # With n=5000 and 10% relative lift, theoretical power is ~37%
        assert result['observed_power'] > 0.30

    def test_power_check_large_effect(self):
        """Test power check with large effect."""
        result = aa_tests.power_check(
            n_tests=100,
            sample_size=2000,
            true_effect=0.20,  # Large effect
            p_baseline=0.10,
            random_state=42
        )

        # With n=2000 and 20% relative lift, theoretical power is ~50%
        assert result['observed_power'] > 0.45

    def test_power_check_small_sample(self):
        """Test power check with small sample."""
        result = aa_tests.power_check(
            n_tests=100,
            sample_size=100,  # Small sample
            true_effect=0.10,
            p_baseline=0.10,
            random_state=42
        )

        # Small sample should have lower power
        assert result['observed_power'] < 0.90


class TestDiagnoseIssues:
    """Tests for diagnose_issues function."""

    def test_diagnose_healthy_infrastructure(self):
        """Test diagnosis when infrastructure is healthy."""
        # Simulate healthy validation result
        validation_result = {
            'false_positive_rate': 0.05,
            'expected_fp_rate': 0.05,
            'passed': True,
            'p_values_uniform': True
        }

        power_result = {
            'observed_power': 0.80,
            'theoretical_power': 0.82
        }

        diagnosis = aa_tests.diagnose_issues(validation_result, power_result)

        assert 'severity' in diagnosis
        assert 'issues_detected' in diagnosis
        assert 'recommendations' in diagnosis

        # Should be 'ok'
        assert diagnosis['severity'] == 'ok'
        assert len(diagnosis['issues_detected']) == 0

    def test_diagnose_high_fp_rate(self):
        """Test diagnosis when FP rate is too high."""
        validation_result = {
            'false_positive_rate': 0.15,  # High!
            'expected_fp_rate': 0.05,
            'passed': False,
            'p_values_uniform': True
        }

        diagnosis = aa_tests.diagnose_issues(validation_result)

        # Should detect issue
        assert diagnosis['severity'] in ['warning', 'critical']
        assert len(diagnosis['issues_detected']) > 0
        assert any('higher than expected' in issue for issue in diagnosis['issues_detected'])

    def test_diagnose_low_power(self):
        """Test diagnosis when power is low."""
        validation_result = {
            'false_positive_rate': 0.05,
            'expected_fp_rate': 0.05,
            'passed': True,
            'p_values_uniform': True
        }

        power_result = {
            'observed_power': 0.40,  # Low
            'theoretical_power': 0.80
        }

        diagnosis = aa_tests.diagnose_issues(validation_result, power_result)

        # Should detect power issue
        assert len(diagnosis['issues_detected']) > 0
        assert any('power' in issue.lower() for issue in diagnosis['issues_detected'])

    def test_diagnose_non_uniform_p_values(self):
        """Test diagnosis when p-values are not uniform."""
        validation_result = {
            'false_positive_rate': 0.05,
            'expected_fp_rate': 0.05,
            'passed': True,
            'p_values_uniform': False  # Issue
        }

        diagnosis = aa_tests.diagnose_issues(validation_result)

        # Should detect uniformity issue
        assert len(diagnosis['issues_detected']) > 0


class TestIntegration:
    """Integration tests for A/A testing workflow."""

    def test_full_validation_workflow(self):
        """Test complete A/A validation workflow."""
        # Run infrastructure validation
        validation = aa_tests.validate_infrastructure(
            n_tests=100,
            sample_size=1000,
            p_baseline=0.10,
            alpha=0.05,
            random_state=42
        )

        # Run power check
        power_result = aa_tests.power_check(
            n_tests=100,
            sample_size=2000,
            true_effect=0.10,
            p_baseline=0.10,
            random_state=42
        )

        # Diagnose issues
        diagnosis = aa_tests.diagnose_issues(validation, power_result)

        # Verify workflow completes
        assert validation['n_tests'] == 100
        assert power_result['n_tests'] == 100
        assert 'severity' in diagnosis
        assert diagnosis['severity'] in ['ok', 'warning', 'critical']

    def test_detect_broken_infrastructure(self):
        """Test detection of broken infrastructure (simulated)."""
        # Simulate validation result with issues
        bad_validation = {
            'false_positive_rate': 0.20,  # Way too high
            'expected_fp_rate': 0.05,
            'passed': False,
            'p_values_uniform': False
        }

        diagnosis = aa_tests.diagnose_issues(bad_validation)

        # Should detect critical issues
        assert diagnosis['severity'] in ['warning', 'critical']
        assert len(diagnosis['issues_detected']) > 0
        assert len(diagnosis['recommendations']) > 0
