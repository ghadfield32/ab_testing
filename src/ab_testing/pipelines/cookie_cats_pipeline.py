"""
Cookie Cats Mobile Game A/B Test Pipeline

This module demonstrates product/growth experimentation using the Cookie Cats dataset (90K players).

Dataset: https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats
Use Case: Mobile game feature testing with retention metrics

Pipeline Steps:
1. Load player data
2. Check randomization (SRM)
3. Analyze multiple retention metrics (1-day, 7-day)
4. Multiple testing correction (Benjamini-Hochberg)
5. Ratio metrics analysis (game rounds / player)
6. Decision with multiple outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from ab_testing.data import loaders
from ab_testing.core import randomization, frequentist
from ab_testing.advanced import multiple_testing, ratio_metrics
from ab_testing.diagnostics import guardrails


def run_cookie_cats_analysis(
    sample_frac: float = 1.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete A/B test analysis on Cookie Cats mobile game dataset.

    This pipeline focuses on:
    - Multiple retention metrics (1-day, 7-day)
    - Multiple testing correction
    - Ratio metrics (engagement per player)
    - Product decision-making

    Parameters
    ----------
    sample_frac : float, default=1.0
        Fraction of data to use (0.0-1.0). Dataset has ~90K rows, so full data is manageable.
    verbose : bool, default=True
        Print detailed progress and results.

    Returns
    -------
    Dict[str, Any]
        Analysis results for retention metrics, engagement, and decisions.

    Examples
    --------
    >>> results = run_cookie_cats_analysis(verbose=True)
    >>> print(results['retention_1d']['significant'])
    True
    """

    results = {}

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    if verbose:
        print("="*70)
        print("COOKIE CATS MOBILE GAME A/B TEST PIPELINE")
        print("="*70)
        print(f"\n[1/6] Loading Cookie Cats dataset (sample_frac={sample_frac})...")

    df = loaders.load_cookie_cats(sample_frac=sample_frac)

    results['data_summary'] = {
        'total_players': len(df),
        'control_size': len(df[df['version'] == 'gate_30']),
        'treatment_size': len(df[df['version'] == 'gate_40']),
        'retention_1d_control': df[df['version'] == 'gate_30']['retention_1'].mean(),
        'retention_1d_treatment': df[df['version'] == 'gate_40']['retention_1'].mean(),
        'retention_7d_control': df[df['version'] == 'gate_30']['retention_7'].mean(),
        'retention_7d_treatment': df[df['version'] == 'gate_40']['retention_7'].mean(),
        'avg_game_rounds_control': df[df['version'] == 'gate_30']['sum_gamerounds'].mean(),
        'avg_game_rounds_treatment': df[df['version'] == 'gate_40']['sum_gamerounds'].mean(),
    }

    if verbose:
        print(f"   ✓ Loaded {results['data_summary']['total_players']:,} players")
        print(f"   ✓ Control (gate_30): {results['data_summary']['control_size']:,}")
        print(f"   ✓ Treatment (gate_40): {results['data_summary']['treatment_size']:,}")
        print(f"   ✓ 1-day retention (control): {results['data_summary']['retention_1d_control']:.2%}")
        print(f"   ✓ 1-day retention (treatment): {results['data_summary']['retention_1d_treatment']:.2%}")
        print(f"   ✓ 7-day retention (control): {results['data_summary']['retention_7d_control']:.2%}")
        print(f"   ✓ 7-day retention (treatment): {results['data_summary']['retention_7d_treatment']:.2%}")

    # ========================================================================
    # STEP 2: Check Randomization (SRM)
    # ========================================================================
    if verbose:
        print(f"\n[2/6] Checking randomization quality (Sample Ratio Mismatch)...")

    control = df[df['version'] == 'gate_30']
    treatment = df[df['version'] == 'gate_40']

    # DATASET CONFIGURATION: Cookie Cats is a properly balanced RCT
    IS_RCT = True  # This IS a randomized controlled trial
    EXPECTED_ALLOCATION = [0.5, 0.5]  # Designed 50/50 split

    # Two-stage SRM check: statistical + practical significance
    # pp_threshold=0.01 means 1 percentage point deviation is practically significant
    results['srm_check'] = randomization.srm_check(
        n_control=len(control),
        n_treatment=len(treatment),
        expected_ratio=EXPECTED_ALLOCATION,
        alpha=0.01,
        pp_threshold=0.01,  # 1pp deviation threshold (50% ± 1% = 49%-51%)
        count_threshold=None  # Large sample - use pp threshold only
    )

    # Mark as balanced RCT in results
    results['srm_check']['is_rct'] = IS_RCT
    results['srm_check']['allocation_type'] = 'rct_balanced'

    if verbose:
        if not results['srm_check']['srm_detected']:
            print(f"   ✓ SRM Check PASSED (p={results['srm_check']['p_value']:.4f})")
            print(f"   ✓ Randomization valid - safe to proceed")
        elif results['srm_check']['srm_warning']:
            # Statistically significant but not practically significant (borderline)
            print(f"   ⚠️  SRM Warning (borderline case)")
            print(f"   Statistical: p={results['srm_check']['p_value']:.6f} < 0.01 (DETECTED)")
            print(f"   Practical: {results['srm_check']['max_pp_deviation']:.4f} pp < {0.01} pp (NOT SEVERE)")
            print(f"   → Allocation: {results['srm_check']['ratio_control']:.2%} / {results['srm_check']['ratio_treatment']:.2%}")
            print(f"   → With large samples (90K+), tiny deviations become statistically detectable")
            print(f"   → Proceeding with analysis but flagging for awareness")
        else:
            # Both statistically AND practically significant (severe SRM)
            print(f"   ✗ SRM Check FAILED (SEVERE)!")
            print(f"   Statistical: p={results['srm_check']['p_value']:.6f} < 0.01")
            print(f"   Practical: {results['srm_check']['max_pp_deviation']:.4f} pp > {0.01} pp")
            print(f"   Allocation differs from expected 50/50 split")

    # ========================================================================
    # HARD GATE: Stop execution if SRM is SEVERE (statistical + practical)
    # ========================================================================
    if IS_RCT and results['srm_check']['srm_severe']:
        if verbose:
            print(f"\n{'='*70}")
            print(f"⛔ ANALYSIS HALTED: SEVERE SRM FAILURE")
            print(f"{'='*70}")
            print(f"\n❌ SRM check failed - randomization may be compromised")
            print(f"   NO statistical inference will be performed")
            print(f"   Investigate and fix before proceeding")
            print(f"\n🔍 Check for:")
            print(f"   - Randomization bugs")
            print(f"   - Logging failures")
            print(f"   - Group-specific technical issues")
            print(f"{'='*70}\n")

        results['status'] = 'INVALID'
        results['status_reason'] = 'Severe SRM failure - randomization compromised'
        return results

    # Safe to continue (either no SRM or just a warning)
    results['status'] = 'VALID'
    if results['srm_check']['srm_warning']:
        results['status_reason'] = 'SRM warning (borderline) - analysis proceeded with caution'

    # ========================================================================
    # STEP 3: Analyze Multiple Retention Metrics
    # ========================================================================
    if verbose:
        print(f"\n[3/6] Analyzing multiple retention metrics...")

    # 1-day retention
    results['retention_1d'] = frequentist.z_test_proportions(
        x_control=control['retention_1'].sum(),
        n_control=len(control),
        x_treatment=treatment['retention_1'].sum(),
        n_treatment=len(treatment),
        alpha=0.05,
        two_sided=True
    )

    # 7-day retention
    results['retention_7d'] = frequentist.z_test_proportions(
        x_control=control['retention_7'].sum(),
        n_control=len(control),
        x_treatment=treatment['retention_7'].sum(),
        n_treatment=len(treatment),
        alpha=0.05,
        two_sided=True
    )

    if verbose:
        print(f"   ✓ 1-Day Retention:")
        print(f"     - Control: {results['data_summary']['retention_1d_control']:.2%}")
        print(f"     - Treatment: {results['data_summary']['retention_1d_treatment']:.2%}")
        print(f"     - Lift: {results['retention_1d']['relative_lift']:.2%}")
        print(f"     - P-value: {results['retention_1d']['p_value']:.6f}")
        print(f"     - Significant: {results['retention_1d']['significant']}")
        print(f"")
        print(f"   ✓ 7-Day Retention:")
        print(f"     - Control: {results['data_summary']['retention_7d_control']:.2%}")
        print(f"     - Treatment: {results['data_summary']['retention_7d_treatment']:.2%}")
        print(f"     - Lift: {results['retention_7d']['relative_lift']:.2%}")
        print(f"     - P-value: {results['retention_7d']['p_value']:.6f}")
        print(f"     - Significant: {results['retention_7d']['significant']}")

    # ========================================================================
    # STEP 4: Multiple Testing Correction
    # ========================================================================
    if verbose:
        print(f"\n[4/6] Applying multiple testing correction (Benjamini-Hochberg)...")

        print("\n📚 LEARNING: Multiple Testing & False Discovery Rate (FDR)")
        print("   What is multiple testing?")
        print("   - Testing MULTIPLE metrics/hypotheses in the SAME experiment")
        print("   - Example: Testing both 1-day retention AND 7-day retention")
        print("   - Problem: More tests = more chances of false positives")
        print("   ")
        print("   Why correction is necessary:")
        print("   - With alpha=0.05, each test has 5% false positive rate")
        print("   - Test 2 metrics: ~10% chance of at least one false positive")
        print("   - Test 20 metrics: ~64% chance of at least one false positive!")
        print("   - Without correction: Appear to find effects that don't exist")
        print("   ")
        print("   Correction methods:")
        print("   - Bonferroni: Divide alpha by number of tests (very conservative)")
        print("   - Benjamini-Hochberg: Control False Discovery Rate (FDR)")
        print("   - FDR: Expected proportion of false positives among rejections")
        print("   - BH is less conservative than Bonferroni (more power)")
        print("   ")
        print("   When to use which:")
        print("   - Bonferroni: When false positives are VERY costly")
        print("   - Benjamini-Hochberg: When you want balance (power vs error control)")
        print("   - No correction: When metrics are truly independent and exploratory")
        print("   ")
        print("   For this experiment:")
        print("   - Testing 2 retention metrics (1-day, 7-day)")
        print("   - Using Benjamini-Hochberg FDR control")
        print("   - Reason: Metrics correlated (users with 1-day likely to have 7-day)")

    p_values = [
        results['retention_1d']['p_value'],
        results['retention_7d']['p_value']
    ]
    metric_names = ['retention_1d', 'retention_7d']

    results['multiple_testing'] = multiple_testing.benjamini_hochberg(
        p_values=p_values,
        alpha=0.05
    )

    if verbose:
        print(f"\n✓ Multiple Testing Correction Results:")
        print(f"   Method: Benjamini-Hochberg FDR control")
        print(f"   Original alpha: 0.05 (5% false positive rate per test)")
        print(f"   Number of tests: {len(p_values)}")
        print(f"   FDR threshold: {results['multiple_testing'].get('fdr_threshold', 0.05):.4f}")
        print(f"   ")
        print(f"   Individual test results:")
        for name, p_val, adj_p, sig in zip(
            metric_names,
            p_values,
            results['multiple_testing']['adjusted_p_values'],
            results['multiple_testing']['significant']
        ):
            print(f"     {name}:")
            print(f"       Original p-value: {p_val:.6f}")
            print(f"       Adjusted p-value: {adj_p:.6f}")
            print(f"       Significant after correction: {sig}")

        print(f"\n💡 INTERPRETATION:")
        n_sig_original = sum(p < 0.05 for p in p_values)
        n_sig_adjusted = sum(results['multiple_testing']['significant'])

        print(f"   Significant before correction: {n_sig_original}/{len(p_values)}")
        print(f"   Significant after correction: {n_sig_adjusted}/{len(p_values)}")

        if n_sig_adjusted < n_sig_original:
            print(f"   ⚠️ Correction changed conclusions for {n_sig_original - n_sig_adjusted} metric(s)")
            print(f"   ⚠️ Without correction: Would have claimed significant effect")
            print(f"   ⚠️ With correction: Not enough evidence after accounting for multiple tests")
            print(f"   ⚠️ This is why multiple testing correction matters!")
        elif n_sig_adjusted == n_sig_original and n_sig_original > 0:
            print(f"   ✓ All originally significant results remain significant after correction")
            print(f"   ✓ Strong evidence - effects robust to multiple testing")
        else:
            print(f"   ○ No significant results even before correction")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - LinkedIn: Uses Benjamini-Hochberg for metric suites (5-10 metrics)")
        print("   - Facebook: Adjusts for primary metrics but not secondary/exploratory")
        print("   - Spotify: Pre-declares primary metrics (no correction for those)")
        print("   - Netflix: Conservative approach - corrects for all concurrent metrics")
        print("   ")
        print("   Best practices:")
        print("   - Declare primary metrics upfront (test those without correction)")
        print("   - Apply correction to secondary/exploratory metrics")
        print("   - Use BH when testing 2-20 metrics (good power/error balance)")
        print("   - Use Bonferroni when testing >20 metrics or critical decisions")

    # ========================================================================
    # STEP 5: Ratio Metric Analysis (Engagement)
    # ========================================================================
    if verbose:
        print(f"\n[5/6] Analyzing engagement ratio metric (game rounds / player)...")

        print("\n📚 LEARNING: Ratio Metrics & Delta Method")
        print("   What are ratio metrics?")
        print("   - Metrics expressed as ratios: CTR = clicks/impressions, ARPU = revenue/users")
        print("   - In our case: Average game rounds per player")
        print("   - More informative than totals (accounts for user base size)")
        print("   ")
        print("   Why ratio metrics need special handling:")
        print("   - Naive approach: Test numerator and denominator separately (WRONG)")
        print("   - Problem: Ratios have complex variance structure")
        print("   - Example: Small change in denominator amplifies ratio change")
        print("   - Need: Delta method for proper confidence intervals")
        print("   ")
        print("   What is the Delta method?")
        print("   - Mathematical technique for deriving variance of ratios")
        print("   - Uses first-order Taylor approximation")
        print("   - Formula: Var(X/Y) ≈ (μx/μy)² × [Var(X)/μx² + Var(Y)/μy² - 2×Cov(X,Y)/(μx×μy)]")
        print("   - Allows construction of confidence intervals for ratio metrics")
        print("   ")
        print("   Common ratio metrics in industry:")
        print("   - CTR (Click-Through Rate): clicks / impressions")
        print("   - Conversion Rate: purchases / visitors")
        print("   - ARPU (Average Revenue Per User): revenue / active users")
        print("   - Engagement rate: interactions / sessions")
        print("   - Our metric: game rounds / players")

    # Use delta method for ratio metric
    control_rounds = control['sum_gamerounds'].values
    treatment_rounds = treatment['sum_gamerounds'].values

    results['engagement'] = ratio_metrics.ratio_metric_test(
        numerator_control=control_rounds,
        denominator_control=np.ones(len(control)),  # 1 player = 1 denominator
        numerator_treatment=treatment_rounds,
        denominator_treatment=np.ones(len(treatment)),
        alpha=0.05
    )

    if verbose:
        print(f"\n✓ Ratio Metric Results:")
        print(f"   Metric: Average game rounds per player")
        print(f"   Control mean: {control_rounds.mean():.2f} rounds/player")
        print(f"   Treatment mean: {treatment_rounds.mean():.2f} rounds/player")
        print(f"   Absolute difference: {results['engagement']['ratio_diff']:.2f} rounds")
        print(f"   Relative change: {results['engagement']['relative_lift']:.2%}")
        print(f"   Standard error: {results['engagement'].get('se', 0.0):.4f}")
        print(f"   Z-statistic: {results['engagement'].get('z_statistic', 0.0):.4f}")
        print(f"   P-value: {results['engagement']['p_value']:.6f}")
        print(f"   95% CI: [{results['engagement']['ci_lower']:.2f}, {results['engagement']['ci_upper']:.2f}]")
        print(f"   Significant: {'✓ YES' if results['engagement']['significant'] else '○ NO'}")

        print(f"\n💡 INTERPRETATION:")
        if results['engagement']['significant']:
            print(f"   ✓ SIGNIFICANT CHANGE in engagement")
            if results['engagement']['relative_lift'] > 0:
                print(f"   ✓ Treatment INCREASES engagement by {results['engagement']['relative_lift']:.2%}")
                print(f"   ✓ Players in treatment play {results['engagement']['ratio_diff']:.1f} more rounds on average")
                print(f"   ✓ This suggests treatment makes game more engaging")
            else:
                print(f"   ⚠️ Treatment DECREASES engagement by {abs(results['engagement']['relative_lift']):.2%}")
                print(f"   ⚠️ Players in treatment play {abs(results['engagement']['ratio_diff']):.1f} fewer rounds")
                print(f"   ⚠️ This could indicate frustration or reduced enjoyment")
        else:
            print(f"   ○ NO SIGNIFICANT change in engagement")
            print(f"   ○ Observed difference ({results['engagement']['relative_lift']:.2%}) could be due to chance")
            print(f"   ○ Cannot confidently say treatment affects engagement")

        print(f"\n   Business context:")
        print(f"   - Engagement metric correlates with monetization (more play = more IAP)")
        print(f"   - But: High engagement with low retention = burnout risk")
        print(f"   - Ideal: Balanced increase in both retention AND engagement")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Zynga: Tracks session length, game rounds, purchase frequency (all ratios)")
        print("   - King (Candy Crush): Monitors levels completed per session")
        print("   - Supercell: Balances engagement with player satisfaction (avoid grind)")
        print("   - General rule: Delta method for all ratio metrics with >1000 samples")
        print("   ")
        print("   When NOT to use delta method:")
        print("   - Very small samples (<100) - bootstrap instead")
        print("   - Denominator essentially constant - simple t-test ok")
        print("   - Complex ratios (e.g., ratios of ratios) - simulation methods")

    # ========================================================================
    # STEP 6: Decision Framework
    # ========================================================================
    if verbose:
        print(f"\n[6/6] Making ship/hold/abandon decision...")

        print("\n📚 LEARNING: Product Decision Frameworks")
        print("   What makes this different from marketing A/B tests?")
        print("   - Marketing: Binary outcome (click/no click, convert/don't convert)")
        print("   - Product: MULTIPLE outcomes to balance (retention, engagement, satisfaction)")
        print("   - Can't optimize one metric at expense of others")
        print("   ")
        print("   Decision framework components:")
        print("   1. PRIMARY metric: What we're trying to improve (1-day retention)")
        print("   2. GUARDRAIL metrics: What we must not harm (7-day retention, engagement)")
        print("   3. DECISION rules: When to ship/hold/abandon")
        print("   ")
        print("   For this experiment:")
        print("   - Primary: 1-day retention (early indicator of feature impact)")
        print("   - Guardrail 1: 7-day retention (long-term health)")
        print("   - Guardrail 2: Engagement (game rounds per player)")
        print("   ")
        print("   Why this structure?")
        print("   - Gate at level 40 vs 30: Delays when players hit paywall")
        print("   - Hypothesis: Later gate = better day-1 retention (less frustration)")
        print("   - Risk: Players get addicted to free game, quit when gate appears later")
        print("   - Hence: Guard 7-day retention and overall engagement")

    # Use 1-day retention as primary metric
    primary = results['retention_1d']

    # 7-day retention as guardrail (shouldn't decrease)
    guardrail_retention_7d = guardrails.non_inferiority_test(
        control=control['retention_7'].values.astype(float),
        treatment=treatment['retention_7'].values.astype(float),
        delta=-0.01,  # Allow max 1% degradation
        metric_type='relative',
        alpha=0.05
    )
    guardrail_retention_7d['metric_name'] = 'retention_7d'

    # Engagement as secondary guardrail
    guardrail_engagement = guardrails.non_inferiority_test(
        control=control_rounds,
        treatment=treatment_rounds,
        delta=-0.05,  # Allow max 5% degradation
        metric_type='relative',
        alpha=0.05
    )
    guardrail_engagement['metric_name'] = 'avg_game_rounds'

    results['guardrails'] = {
        'retention_7d': guardrail_retention_7d,
        'engagement': guardrail_engagement
    }

    results['decision'] = guardrails.evaluate_guardrails(
        primary_result=primary,
        guardrail_results=[guardrail_retention_7d, guardrail_engagement]
    )

    if verbose:
        print(f"\n✓ Decision Analysis:")
        print(f"   Primary metric: 1-day retention")
        print(f"     - Control: {results['data_summary']['retention_1d_control']:.2%}")
        print(f"     - Treatment: {results['data_summary']['retention_1d_treatment']:.2%}")
        print(f"     - Lift: {primary['relative_lift']:.2%}")
        print(f"     - Result: {'✓ SIGNIFICANT' if primary['significant'] else '○ NOT SIGNIFICANT'}")
        print(f"   ")
        # Guardrail 1: 7-day retention (relative metric)
        print(f"   Guardrail 1: 7-day retention")
        print(f"     - Tolerance: -1.0% (max allowed degradation)")
        # Calculate relative change for display (difference is absolute, need to convert to %)
        if guardrail_retention_7d.get('metric_type') == 'relative':
            retention_7d_pct_change = guardrail_retention_7d['difference'] / guardrail_retention_7d['mean_control']
            retention_7d_ci_lower_pct = guardrail_retention_7d['ci_lower'] / guardrail_retention_7d['mean_control']
            print(f"     - Actual change: {retention_7d_pct_change:.2%}")
            print(f"     - 95% CI lower bound: {retention_7d_ci_lower_pct:.2%}")
        else:
            print(f"     - Actual change: {guardrail_retention_7d.get('difference', 0.0):.4f}")
        print(f"     - Result: {'✓ PASSED' if guardrail_retention_7d['passed'] else '✗ FAILED'}")
        print(f"   ")

        # Guardrail 2: Engagement (relative metric)
        print(f"   Guardrail 2: Engagement (avg game rounds)")
        print(f"     - Tolerance: -5.0% (max allowed degradation)")
        # Calculate relative change for display (difference is absolute, need to convert to %)
        if guardrail_engagement.get('metric_type') == 'relative':
            engagement_pct_change = guardrail_engagement['difference'] / guardrail_engagement['mean_control']
            engagement_ci_lower_pct = guardrail_engagement['ci_lower'] / guardrail_engagement['mean_control']
            print(f"     - Actual change: {engagement_pct_change:.2%}")
            print(f"     - 95% CI lower bound: {engagement_ci_lower_pct:.2%}")
            # Explain CI-based decision
            if not guardrail_engagement['passed'] and engagement_pct_change > -0.05:
                print(f"     - Note: Point estimate OK, but CI extends below tolerance")
        else:
            print(f"     - Actual change: {guardrail_engagement.get('difference', 0.0):.4f}")
        print(f"     - Result: {'✓ PASSED' if guardrail_engagement['passed'] else '✗ FAILED'}")

        decision = results['decision']['decision'].upper()
        print(f"\n   >>> FINAL DECISION: {decision} <<<")
        print(f"")

        print(f"💡 INTERPRETATION:")
        if decision == 'SHIP':
            print(f"   ✓ RECOMMENDATION: Move gate from level 30 to level 40")
            print(f"   ✓ Primary metric improved: {primary['relative_lift']:.2%} lift in 1-day retention")
            print(f"   ✓ All guardrails passed: No harm to long-term metrics")
            print(f"   ")
            print(f"   Expected impact:")
            print(f"   - {abs(primary['relative_lift']):.1f}% more players return day 1")
            print(f"   - For 100K daily new players: {abs(primary['relative_lift']) * 100000:.0f} additional day-1 returns")
            print(f"   - Larger player base → more monetization opportunities")
            print(f"   ")
            print(f"   Next steps:")
            print(f"   1. Roll out to 100% of players")
            print(f"   2. Monitor 7-day and 30-day retention post-launch (validate guardrails)")
            print(f"   3. Track revenue impact (more players but less urgency to pay?)")
        elif decision == 'ABANDON':
            print(f"   ✗ RECOMMENDATION: Do NOT move gate")
            if not primary['significant'] or primary['relative_lift'] < 0:
                print(f"   ✗ Primary metric failed: No improvement or negative impact")
            if not guardrail_retention_7d['passed']:
                print(f"   ✗ Guardrail violation: 7-day retention harmed beyond tolerance")
            if not guardrail_engagement['passed']:
                print(f"   ✗ Guardrail violation: Engagement harmed beyond tolerance")
            print(f"   ")
            print(f"   Why this might have failed:")
            print(f"   - Gate too late: Players quit when they finally hit it")
            print(f"   - Inconsistent difficulty: Levels 31-40 too hard without boosts")
            print(f"   - Monetization timing: Players don't value IAP until later")
            print(f"   ")
            print(f"   Next steps:")
            print(f"   1. Analyze user feedback and qualitative data")
            print(f"   2. Consider alternative gate positions (level 35?)")
            print(f"   3. Test with refined design addressing failure root cause")
        else:  # HOLD
            print(f"   ○ RECOMMENDATION: Need more evidence")
            print(f"   ○ Results inconclusive - effect promising but not definitive")
            print(f"   ")
            print(f"   Options:")
            print(f"   1. Extend experiment duration (more players)")
            print(f"   2. Increase traffic allocation (50% → 90%)")
            print(f"   3. Analyze subgroups (new vs existing players)")

        print(f"\n🏢 INDUSTRY PRACTICE (Mobile Gaming):")
        print("   - Zynga: Always tests monetization changes with retention guardrails")
        print("   - King: Typical tolerances: -2% for revenue, -1% for retention")
        print("   - Supercell: Runs 2-4 week tests for feature changes (novelty effects)")
        print("   - Playrix: Segments by player lifetime (new vs whale vs churned)")
        print("   ")
        print("   Key lessons from mobile gaming A/B tests:")
        print("   - Short-term metrics can mislead (novelty, frustration)")
        print("   - Always monitor LTV (lifetime value) post-launch")
        print("   - Retention >>>> engagement for long-term success")
        print("   - Different player segments respond VERY differently")

    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    # Fix Windows terminal encoding for emoji support
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\n" + "🎓"*35)
    print("A/B TESTING MASTERCLASS: PRODUCT/GROWTH EXPERIMENTATION")
    print("Dataset: Cookie Cats Mobile Game (90K+ players)")
    print("🎓"*35)

    print("\nThis pipeline demonstrates product experimentation best practices:")
    print("  1. Multiple outcome metrics (1-day, 7-day retention)")
    print("  2. Multiple testing correction (Benjamini-Hochberg FDR)")
    print("  3. Ratio metrics analysis (engagement per player)")
    print("  4. Balanced decision-making (primary + guardrails)")
    print("  5. Mobile gaming-specific considerations")

    print("\n" + "="*70)
    print("Starting analysis...")
    print("="*70)

    # Run pipeline
    results = run_cookie_cats_analysis(verbose=True)

    # ========================================================================
    # COMPREHENSIVE FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 FINAL SUMMARY & KEY LEARNINGS")
    print("="*70)

    print(f"\n1. EXPERIMENT OVERVIEW")
    print(f"   Game: Cookie Cats (match-3 puzzle game)")
    print(f"   Total players: {results['data_summary']['total_players']:,}")
    print(f"   Test: Gate position (level 30 vs level 40)")
    print(f"   Context: Gate forces players to wait or pay to continue")
    print(f"   Hypothesis: Moving gate to level 40 improves retention (less early frustration)")

    print(f"\n2. DATA QUALITY")
    print(f"   Control (gate_30): {results['data_summary']['control_size']:,} players")
    print(f"   Treatment (gate_40): {results['data_summary']['treatment_size']:,} players")
    print(f"   Group balance: {results['data_summary']['treatment_size']/results['data_summary']['control_size']:.4f}")
    print(f"   SRM check: {'✓ PASSED' if not results['srm_check']['srm_detected'] else '✗ FAILED'}")

    # Check if analysis was halted due to SRM failure
    if results.get('status') == 'INVALID':
        print(f"\n⛔ ANALYSIS STATUS: INVALID")
        print(f"   Reason: {results.get('status_reason', 'Unknown')}")
        print(f"\n🔍 SRM FAILURE DETAILS:")
        print(f"   Expected allocation: 50.0% control / 50.0% treatment")
        print(f"   Actual allocation: {results['srm_check']['ratio_control']:.2%} control / {results['srm_check']['ratio_treatment']:.2%} treatment")
        print(f"   Chi-square statistic: {results['srm_check']['chi2_statistic']:.4f}")
        print(f"   P-value: {results['srm_check']['p_value']:.6f} (threshold: 0.01)")
        print(f"   Absolute difference: {abs(results['data_summary']['control_size'] - results['data_summary']['treatment_size']):,} players")
        print(f"\n❌ NO STATISTICAL INFERENCE PERFORMED")
        print(f"   When SRM fails in a true RCT, all subsequent analysis is invalid.")
        print(f"   You MUST investigate and fix the randomization before drawing conclusions.")
        print(f"\n🔍 INVESTIGATION CHECKLIST:")
        print(f"   □ Review randomization implementation code")
        print(f"   □ Check for group-specific technical issues (e.g., crashes, loading failures)")
        print(f"   □ Verify logging/tracking is working for both groups")
        print(f"   □ Look for selection bias (e.g., certain users excluded from one group)")
        print(f"   □ Check if ratio was intentionally designed (e.g., 90/10 ramp)")
        print(f"\n📚 KEY LEARNING:")
        print(f"   SRM is the FIRST check for a reason - it validates the experiment's foundation.")
        print(f"   Even small imbalances (like this 49.56% vs 50.44%) can indicate serious bugs.")
        print(f"   With large samples (90K+ here), tiny deviations become statistically detectable.")
        print(f"   Industry practice: alpha=0.01 for SRM (stricter than 0.05 for other tests).")

        # Skip rest of summary since no analysis was performed
        print(f"\n" + "="*70)
        print(f"✅ Pipeline complete. Fix SRM issue before re-running analysis.")
        print(f"="*70 + "\n")
    else:
        # If analysis completed successfully, show full results
        print(f"\n3. PRIMARY METRIC: 1-Day Retention")
        print(f"   Control: {results['data_summary']['retention_1d_control']:.2%}")
        print(f"   Treatment: {results['data_summary']['retention_1d_treatment']:.2%}")
        print(f"   Absolute lift: {results['retention_1d']['absolute_lift']:.4f} ({results['retention_1d']['absolute_lift']*100:.2f} percentage points)")
        print(f"   Relative lift: {results['retention_1d']['relative_lift']:.2%}")
        print(f"   P-value: {results['retention_1d']['p_value']:.6f}")
        print(f"   Significant: {'✓ YES (p < 0.05)' if results['retention_1d']['significant'] else '○ NO (p ≥ 0.05)'}")
        print(f"   95% CI: [{results['retention_1d'].get('ci_lower', 0.0):.4f}, {results['retention_1d'].get('ci_upper', 0.0):.4f}]")

        print(f"\n4. SECONDARY METRIC: 7-Day Retention")
        print(f"   Control: {results['data_summary']['retention_7d_control']:.2%}")
        print(f"   Treatment: {results['data_summary']['retention_7d_treatment']:.2%}")
        print(f"   Absolute lift: {results['retention_7d']['absolute_lift']:.4f}")
        print(f"   Relative lift: {results['retention_7d']['relative_lift']:.2%}")
        print(f"   P-value: {results['retention_7d']['p_value']:.6f}")
        print(f"   Significant: {'✓ YES' if results['retention_7d']['significant'] else '○ NO'}")

        print(f"\n5. MULTIPLE TESTING CORRECTION")
        print(f"   Method: Benjamini-Hochberg (FDR control)")
        print(f"   Tests: 2 (retention_1d, retention_7d)")
        print(f"   Original alpha: 0.05")
        print(f"   Adjusted results:")
        n_sig_before = sum([results['retention_1d']['p_value'] < 0.05, results['retention_7d']['p_value'] < 0.05])
        n_sig_after = sum(results['multiple_testing']['significant'])
        print(f"     - Significant before correction: {n_sig_before}/2")
        print(f"     - Significant after correction: {n_sig_after}/2")
        if n_sig_before > n_sig_after:
            print(f"     - ⚠️ Correction changed {n_sig_before - n_sig_after} conclusion(s)!")
        else:
            print(f"     - ✓ Results robust to multiple testing")

        print(f"\n6. ENGAGEMENT (Ratio Metric)")
        print(f"   Metric: Average game rounds per player")
        print(f"   Control: {results['data_summary']['avg_game_rounds_control']:.2f} rounds/player")
        print(f"   Treatment: {results['data_summary']['avg_game_rounds_treatment']:.2f} rounds/player")
        print(f"   Difference: {results['engagement']['ratio_diff']:.2f} rounds")
        print(f"   Relative change: {results['engagement']['relative_lift']:.2%}")
        print(f"   P-value: {results['engagement']['p_value']:.6f}")
        print(f"   Significant: {'✓ YES' if results['engagement']['significant'] else '○ NO'}")
        print(f"   Method: Delta method for ratio metric variance")

        print(f"\n7. GUARDRAIL METRICS")
        print(f"   Guardrail 1: 7-day retention")
        print(f"     - Tolerance: -1.0% (max allowed degradation)")
        print(f"     - Result: {'✓ PASSED' if results['guardrails']['retention_7d']['passed'] else '✗ FAILED'}")
        print(f"   Guardrail 2: Engagement (avg rounds)")
        print(f"     - Tolerance: -5.0% (max allowed degradation)")
        print(f"     - Result: {'✓ PASSED' if results['guardrails']['engagement']['passed'] else '✗ FAILED'}")

        print(f"\n8. FINAL DECISION")
        decision = results['decision']['decision'].upper()
        print(f"   >>> {decision} <<<")
        print(f"   ")
        if decision == 'SHIP':
            print(f"   ✓ Recommendation: Move gate from level 30 to level 40")
            print(f"   ✓ Expected impact: {results['retention_1d']['relative_lift']:.2%} improvement in 1-day retention")
            print(f"   ✓ No guardrail violations detected")
            print(f"   ")
            print(f"   For 100K daily new players:")
            print(f"   - Additional day-1 returns: {abs(results['retention_1d']['relative_lift']) * 100000:.0f} players")
            print(f"   - Larger engaged base → more monetization opportunities")
        elif decision == 'ABANDON':
            print(f"   ✗ Recommendation: Do NOT move gate")
            print(f"   ✗ Either primary metric negative or guardrail violations")
            print(f"   ✗ Redesign and retest with refined approach")
        else:
            print(f"   ○ Recommendation: Gather more evidence")
            print(f"   ○ Results promising but inconclusive")

        print(f"\n" + "="*70)
        print(f"📚 KEY TAKEAWAYS FOR PRODUCT A/B TESTING")
        print(f"="*70)

        print(f"\n✓ CRITICAL STEPS - ALWAYS DO THESE:")
        print(f"  1. Test MULTIPLE outcomes (retention at different time windows)")
        print(f"  2. Apply multiple testing correction (Benjamini-Hochberg for 2-20 metrics)")
        print(f"  3. Use proper methods for ratio metrics (Delta method, not naive tests)")
        print(f"  4. Set guardrails to prevent unintended harm (long-term > short-term)")
        print(f"  5. Balance statistical significance with business context")
        print(f"  6. Monitor post-launch to validate experiment conclusions")

        print(f"\n⚠️  COMMON PITFALLS TO AVOID:")
        print(f"  1. Testing only one outcome (miss delayed/long-term effects)")
        print(f"  2. No multiple testing correction (inflate false positive rate)")
        print(f"  3. Naive ratio tests (wrong variance, wrong CIs)")
        print(f"  4. Optimizing primary at expense of guardrails (short-term thinking)")
        print(f"  5. Ignoring novelty effects in product changes (test longer)")
        print(f"  6. Same thresholds for all metrics (retention ≠ clicks)")

        print(f"\n🏢 INDUSTRY BEST PRACTICES (Mobile Gaming):")
        print(f"  - Zynga: Tests all monetization changes for 2-4 weeks (novelty)")
        print(f"  - King: Multiple retention windows (1d, 3d, 7d, 30d)")
        print(f"  - Supercell: Strict guardrails on player satisfaction metrics")
        print(f"  - Playrix: Segments tests by player lifetime value (new/whale/churned)")
        print(f"  - Typical tolerances: Revenue -2%, Retention -1%, Engagement -5%")

        print(f"\n💡 WHAT TO DO NEXT:")
        print(f"  1. Explore the data: Try different sample_frac values")
        print(f"  2. Experiment with guardrail thresholds (how sensitive?)")
        print(f"  3. Compare Bonferroni vs Benjamini-Hochberg corrections")
        print(f"  4. Analyze by user segments (if demographic data available)")
        print(f"  5. Read the mobile gaming case studies cited in this pipeline")

        print(f"\n🎮 PRODUCT-SPECIFIC INSIGHTS:")
        print(f"  - Gates are monetization chokepoints (balance revenue vs retention)")
        print(f"  - Early gates: Faster monetization but higher churn")
        print(f"  - Late gates: Better retention but delayed revenue")
        print(f"  - Optimal position varies by game genre and player base")
        print(f"  - Always test with retention + engagement + revenue metrics")

        print(f"\n" + "="*70)
        print(f"✅ Pipeline complete! You've mastered product experimentation.")
        print(f"="*70 + "\n")
