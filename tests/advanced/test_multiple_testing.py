"""
Tests for multiple testing correction module.
"""

import pytest
import numpy as np
from ab_testing.advanced import multiple_testing


class TestBonferroniCorrection:
    """Tests for Bonferroni correction."""

    def test_bonferroni_basic(self):
        """Test basic Bonferroni correction."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        assert 'adjusted_p_values' in result
        assert 'significant' in result
        assert 'n_significant' in result

        # Adjusted p-values should be p * n
        expected_adj = p_values * len(p_values)
        np.testing.assert_array_almost_equal(result['adjusted_p_values'], np.minimum(expected_adj, 1.0))

    def test_bonferroni_all_significant(self):
        """Test Bonferroni with all very significant p-values."""
        p_values = np.array([0.001, 0.002, 0.003])
        result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # All should still be significant
        assert result['n_significant'] == 3
        assert np.all(result['significant'])

    def test_bonferroni_none_significant(self):
        """Test Bonferroni with no significant p-values."""
        p_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # None should be significant
        assert result['n_significant'] == 0
        assert not np.any(result['significant'])

    def test_bonferroni_capped_at_one(self):
        """Test that adjusted p-values are capped at 1.0."""
        p_values = np.array([0.5, 0.6, 0.7])
        result = multiple_testing.bonferroni_correction(p_values)

        # All adjusted p-values should be <= 1.0
        assert np.all(result['adjusted_p_values'] <= 1.0)


class TestBenjaminiHochberg:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_bh_basic(self):
        """Test basic BH correction."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05])
        result = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)

        assert 'adjusted_p_values' in result
        assert 'significant' in result
        assert 'n_significant' in result
        assert 'fdr_threshold' in result

        # More permissive than Bonferroni
        bonf_result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)
        assert result['n_significant'] >= bonf_result['n_significant']

    def test_bh_sorted_input(self):
        """Test BH with already sorted p-values."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.04])
        result = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)

        # Should handle sorted input correctly
        assert result['n_significant'] >= 1

    def test_bh_unsorted_input(self):
        """Test BH with unsorted p-values."""
        p_values = np.array([0.04, 0.001, 0.03, 0.01, 0.02])
        result = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)

        # Should sort internally and produce correct results
        assert result['n_significant'] >= 1

    def test_bh_all_significant(self):
        """Test BH with all very significant p-values."""
        p_values = np.array([0.0001, 0.0002, 0.0003])
        result = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)

        # All should be significant
        assert result['n_significant'] == 3

    def test_bh_none_significant(self):
        """Test BH with no significant p-values."""
        p_values = np.array([0.1, 0.2, 0.3])
        result = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)

        # None should be significant
        assert result['n_significant'] == 0


class TestSidakCorrection:
    """Tests for Šidák correction."""

    def test_sidak_basic(self):
        """Test basic Šidák correction."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = multiple_testing.sidak_correction(p_values, alpha=0.05)

        assert 'adjusted_alpha' in result
        assert 'significant' in result
        assert 'n_significant' in result

        # Adjusted alpha should be 1 - (1-α)^(1/n)
        n = len(p_values)
        expected_alpha = 1 - (1 - 0.05)**(1/n)
        assert abs(result['adjusted_alpha'] - expected_alpha) < 1e-10

    def test_sidak_more_powerful_than_bonferroni(self):
        """Test that Šidák is slightly more powerful than Bonferroni."""
        p_values = np.array([0.01, 0.011, 0.012, 0.013, 0.014])

        sidak_result = multiple_testing.sidak_correction(p_values, alpha=0.05)
        bonf_result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # Šidák should find at least as many significant
        assert sidak_result['n_significant'] >= bonf_result['n_significant']

        # Šidák adjusted alpha should be slightly larger than Bonferroni
        bonf_alpha = 0.05 / len(p_values)
        assert sidak_result['adjusted_alpha'] > bonf_alpha


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni correction."""

    def test_holm_basic(self):
        """Test basic Holm-Bonferroni correction."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.04])
        result = multiple_testing.holm_bonferroni(p_values, alpha=0.05)

        assert 'adjusted_p_values' in result
        assert 'significant' in result
        assert 'n_significant' in result

    def test_holm_more_powerful_than_bonferroni(self):
        """Test that Holm is more powerful than Bonferroni."""
        p_values = np.array([0.001, 0.01, 0.015, 0.02, 0.025])

        holm_result = multiple_testing.holm_bonferroni(p_values, alpha=0.05)
        bonf_result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # Holm should find at least as many significant
        assert holm_result['n_significant'] >= bonf_result['n_significant']

    def test_holm_sequential_rejection(self):
        """Test that Holm uses sequential rejection correctly."""
        # First very significant, rest marginal
        p_values = np.array([0.001, 0.04, 0.045, 0.048, 0.049])
        result = multiple_testing.holm_bonferroni(p_values, alpha=0.05)

        # Should reject at least the first one
        assert result['n_significant'] >= 1


class TestFalsePositiveInflation:
    """Tests for false positive inflation calculation."""

    def test_fp_inflation_basic(self):
        """Test basic FP inflation calculation."""
        inflation = multiple_testing.false_positive_inflation(n_tests=5, alpha=0.05)

        # FWER = 1 - (1-α)^n
        expected = 1 - (1 - 0.05)**5
        assert abs(inflation - expected) < 1e-10

    def test_fp_inflation_single_test(self):
        """Test that single test has no inflation."""
        inflation = multiple_testing.false_positive_inflation(n_tests=1, alpha=0.05)
        assert abs(inflation - 0.05) < 1e-10

    def test_fp_inflation_many_tests(self):
        """Test FP inflation with many tests."""
        inflation = multiple_testing.false_positive_inflation(n_tests=20, alpha=0.05)

        # Should be substantial
        assert inflation > 0.60  # More than 60% chance of at least one FP


class TestMultipleTestingSummary:
    """Tests for multiple_testing_summary function."""

    def test_summary_basic(self):
        """Test basic summary with metric names."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.04])
        metric_names = ['metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5']

        summary = multiple_testing.multiple_testing_summary(
            p_values, metric_names=metric_names, alpha=0.05
        )

        assert 'bonferroni' in summary
        assert 'benjamini_hochberg' in summary
        assert 'sidak' in summary
        assert 'holm' in summary
        assert 'uncorrected_significant' in summary
        assert 'results_table' in summary

    def test_summary_without_metric_names(self):
        """Test summary without metric names."""
        p_values = np.array([0.001, 0.01, 0.02])

        summary = multiple_testing.multiple_testing_summary(p_values, alpha=0.05)

        # Should work with auto-generated names
        assert 'results_table' in summary
        assert len(summary['results_table']) == len(p_values)

    def test_summary_recommendations(self):
        """Test that summary includes recommendations."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.04])

        summary = multiple_testing.multiple_testing_summary(p_values, alpha=0.05)

        # Should have recommendation field
        assert 'recommendation' in summary
        assert len(summary['recommendation']) > 0

    def test_summary_all_methods_consistent(self):
        """Test that all methods in summary are consistent."""
        # Very significant p-values
        p_values = np.array([0.0001, 0.0002, 0.0003])

        summary = multiple_testing.multiple_testing_summary(p_values, alpha=0.05)

        # All methods should find all significant
        assert summary['bonferroni']['n_significant'] == 3
        assert summary['benjamini_hochberg']['n_significant'] == 3
        assert summary['sidak']['n_significant'] == 3
        assert summary['holm']['n_significant'] == 3

    def test_summary_ordering(self):
        """Test that methods are ordered by power (BH > Holm > Šidák > Bonferroni)."""
        # Mixed p-values
        p_values = np.array([0.005, 0.01, 0.015, 0.02, 0.025])

        summary = multiple_testing.multiple_testing_summary(p_values, alpha=0.05)

        # BH should find most
        bh_sig = summary['benjamini_hochberg']['n_significant']
        holm_sig = summary['holm']['n_significant']
        sidak_sig = summary['sidak']['n_significant']
        bonf_sig = summary['bonferroni']['n_significant']

        # General ordering (may have ties)
        assert bh_sig >= holm_sig
        assert holm_sig >= bonf_sig
        assert sidak_sig >= bonf_sig


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_p_value(self):
        """Test all methods with single p-value."""
        p_values = np.array([0.03])

        bonf = multiple_testing.bonferroni_correction(p_values, alpha=0.05)
        bh = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)
        sidak = multiple_testing.sidak_correction(p_values, alpha=0.05)
        holm = multiple_testing.holm_bonferroni(p_values, alpha=0.05)

        # All should agree for single test
        assert bonf['n_significant'] == bh['n_significant']
        assert bonf['n_significant'] == sidak['n_significant']
        assert bonf['n_significant'] == holm['n_significant']

    def test_p_values_at_boundary(self):
        """Test with p-values exactly at significance threshold."""
        p_values = np.array([0.05, 0.05, 0.05])

        bonf = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # Bonferroni: adjusted = 0.05 * 3 = 0.15, not significant
        assert bonf['n_significant'] == 0

    def test_all_zeros(self):
        """Test with all p-values = 0 (perfectly significant)."""
        p_values = np.array([0.0, 0.0, 0.0])

        bonf = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # All should be significant
        assert bonf['n_significant'] == 3

    def test_all_ones(self):
        """Test with all p-values = 1 (not significant)."""
        p_values = np.array([1.0, 1.0, 1.0])

        bonf = multiple_testing.bonferroni_correction(p_values, alpha=0.05)

        # None should be significant
        assert bonf['n_significant'] == 0

    def test_empty_p_values(self):
        """Test that empty p-values raises error."""
        p_values = np.array([])

        with pytest.raises(ValueError):
            multiple_testing.bonferroni_correction(p_values)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        p_values = np.array([0.01, 0.02])

        with pytest.raises(ValueError):
            multiple_testing.bonferroni_correction(p_values, alpha=1.5)

        with pytest.raises(ValueError):
            multiple_testing.bonferroni_correction(p_values, alpha=-0.1)
