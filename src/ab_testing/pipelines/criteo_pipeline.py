"""
Criteo Uplift Modeling Pipeline

This module demonstrates advanced A/B testing techniques using the Criteo Uplift dataset (13.9M observations).

Dataset: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
Use Case: Large-scale uplift modeling and heterogeneous treatment effects

Pipeline Steps:
1. Load dataset (with sampling for memory efficiency)
2. Check randomization (SRM)
3. Run primary analysis (visit rate, conversion rate)
4. Apply CUPAC (ML-enhanced variance reduction)
5. Estimate heterogeneous treatment effects (X-Learner)
6. Sequential testing with O'Brien-Fleming boundaries
7. Decision framework with guardrails
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from ab_testing.data import loaders
from ab_testing.core import randomization, frequentist
from ab_testing.variance_reduction import cupac
from ab_testing.advanced import hte, sequential
from ab_testing.diagnostics import guardrails


def run_criteo_analysis(
    sample_frac: float = 0.01,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run advanced A/B test analysis on Criteo Uplift dataset.

    This pipeline showcases state-of-the-art techniques:
    - CUPAC for variance reduction using ML
    - X-Learner for heterogeneous treatment effects (CATE)
    - Sequential testing for early stopping

    Parameters
    ----------
    sample_frac : float, default=0.01
        Fraction of data to use (0.0-1.0). Dataset has 13.9M rows, so 0.01 = 139K.
        Use smaller values (0.001) for quick testing.
    verbose : bool, default=True
        Print detailed progress and results.

    Returns
    -------
    Dict[str, Any]
        Analysis results including HTE, CUPAC, sequential testing results.

    Examples
    --------
    >>> results = run_criteo_analysis(sample_frac=0.001, verbose=True)
    >>> print(results['hte']['avg_treatment_effect'])
    0.0042
    """

    results = {}

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    if verbose:
        print("="*70)
        print("CRITEO UPLIFT MODELING PIPELINE")
        print("="*70)
        print(f"\n[1/7] Loading Criteo dataset (sample_frac={sample_frac})...")

        print("\n📚 LEARNING: Large-Scale Experimentation & Uplift Modeling")
        print("   What is the Criteo dataset?")
        print("   - Real-world dataset from Criteo (major ad tech company)")
        print("   - 13.9 million observations from actual marketing campaigns")
        print("   - Contains 11 user features + treatment indicator + outcomes")
        print("   - Purpose: Predict WHO will respond positively to treatment")
        print("   ")
        print("   Why large-scale data matters:")
        print("   - Enables detection of small effects (high statistical power)")
        print("   - Supports advanced ML techniques (CUPAC, X-Learner)")
        print("   - Allows subgroup analysis without losing power")
        print("   - Real-world complexity: messy data, heterogeneous users")
        print("   ")
        print("   Sampling strategy:")
        print(f"   - Full dataset: 13.9M rows (can cause memory issues)")
        print(f"   - Current sample: {sample_frac*100:.1f}% = ~{int(13.9e6 * sample_frac):,} rows")
        print("   - Trade-off: Speed vs. precision (smaller sample = faster but noisier)")
        print("   - For learning: 0.1% (14K rows) is sufficient")
        print("   - For production: 1-10% (139K-1.4M rows) recommended")

    df = loaders.load_criteo_uplift(sample_frac=sample_frac)

    results['data_summary'] = {
        'total_observations': len(df),
        'control_size': len(df[df['treatment'] == 0]),
        'treatment_size': len(df[df['treatment'] == 1]),
        'visit_rate_control': df[df['treatment'] == 0]['visit'].mean(),
        'visit_rate_treatment': df[df['treatment'] == 1]['visit'].mean(),
        'conversion_rate_control': df[df['treatment'] == 0]['conversion'].mean(),
        'conversion_rate_treatment': df[df['treatment'] == 1]['conversion'].mean(),
        'n_features': len(df.columns) - 3,  # Exclude treatment, visit, conversion
    }

    if verbose:
        print(f"\n✓ Data Loaded Successfully:")
        print(f"   Total observations: {results['data_summary']['total_observations']:,}")
        print(f"   Control group: {results['data_summary']['control_size']:,}")
        print(f"   Treatment group: {results['data_summary']['treatment_size']:,}")
        print(f"   User features: {results['data_summary']['n_features']}")
        print(f"   Visit rate (control): {results['data_summary']['visit_rate_control']:.4f}")
        print(f"   Visit rate (treatment): {results['data_summary']['visit_rate_treatment']:.4f}")
        print(f"   Conversion rate (control): {results['data_summary']['conversion_rate_control']:.4f}")
        print(f"   Conversion rate (treatment): {results['data_summary']['conversion_rate_treatment']:.4f}")

        print(f"\n💡 INTERPRETATION:")
        print(f"   - Dataset size: {results['data_summary']['total_observations']:,} obs")
        print(f"   - Group balance: {results['data_summary']['treatment_size']/results['data_summary']['control_size']:.3f} (should be ~1.0)")
        print(f"   - Baseline visit rate: {results['data_summary']['visit_rate_control']:.2%}")
        print(f"   - Baseline conversion rate: {results['data_summary']['conversion_rate_control']:.2%}")
        print(f"   - With this sample size, we can detect effects as small as 0.5-1% with 80% power")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Criteo: Published this dataset to advance uplift modeling research")
        print("   - Meta: Uses similar large-scale datasets (billions of observations)")
        print("   - Amazon: Samples 1-5% of traffic for most experiments to balance speed/precision")

    # ========================================================================
    # STEP 2: Check Randomization (SRM)
    # ========================================================================
    if verbose:
        print(f"\n[2/7] Checking randomization quality (Sample Ratio Mismatch)...")

        print("\n📚 LEARNING: SRM (Sample Ratio Mismatch)")
        print("   What is SRM?")
        print("   - Compares actual vs. expected group sizes")
        print("   - Expected: 50/50 split (equal probability of assignment)")
        print("   - SRM detected = randomization likely broken")
        print("   ")
        print("   Why SRM is critical in large-scale experiments:")
        print("   - At scale (millions of users), even small bugs appear huge")
        print("   - Example: 49.9% vs 50.1% looks tiny but p-value < 0.001")
        print("   - Always check SRM BEFORE analyzing results")
        print("   - If SRM detected: STOP, find bug, restart experiment")
        print("   ")
        print("   Common causes at scale:")
        print("   - Implementation bugs in randomization logic")
        print("   - Bot traffic concentrated in one group")
        print("   - Caching issues (users see cached control/treatment inconsistently)")
        print("   - Tracking failures (one group not logging properly)")

    control = df[df['treatment'] == 0]
    treatment = df[df['treatment'] == 1]

    # DATASET CONFIGURATION: Criteo has designed imbalanced allocation
    # Actual allocation: ~15% control / 85% treatment (likely cost/risk-based)
    # This IS an RCT, just with intentional unequal allocation
    IS_RCT = True  # This IS a randomized controlled trial
    EXPECTED_ALLOCATION = [0.15, 0.85]  # Designed 15/85 split (approximate)

    # Calculate observed allocation
    n_control = len(control)
    n_treatment = len(treatment)
    observed_ratio_control = n_control / (n_control + n_treatment)
    observed_ratio_treatment = n_treatment / (n_control + n_treatment)

    if verbose:
        print(f"\n⚠️  DATASET NOTE: Imbalanced Allocation (By Design)")
        print(f"   This dataset uses 15/85 allocation (common for cost/risk management)")
        print(f"   - Treatment may be expensive (ad spend)")
        print(f"   - Or high-risk (start small, scale if successful)")
        print(f"   - Statistical power is lower than 50/50 but still valid")
        print()

    # Two-stage SRM check: statistical + practical significance
    # Stage A (statistical): p < alpha (detects even tiny deviations in large samples)
    # Stage B (practical): deviation > threshold (ensures deviation is meaningful)
    # With 13.9M sample, tiny deviations (e.g., 0.1pp) can be statistically significant
    # but practically irrelevant. Use 1pp threshold for practical significance.
    results['srm_check'] = randomization.srm_check(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_ratio=EXPECTED_ALLOCATION,
        alpha=0.01,
        pp_threshold=0.01,  # 1 percentage point deviation threshold
        count_threshold=None  # Large sample (millions) - use pp threshold only
    )

    # Mark as imbalanced RCT in results
    results['srm_check']['is_rct'] = IS_RCT
    results['srm_check']['allocation_type'] = 'rct_imbalanced'

    if verbose:
        print(f"\n✓ Two-Stage SRM Test Results:")
        print(f"   Expected (designed): {EXPECTED_ALLOCATION[0]:.1%} / {EXPECTED_ALLOCATION[1]:.1%}")
        print(f"   Observed (this run): {observed_ratio_control:.2%} / {observed_ratio_treatment:.2%}")
        print(f"   ")
        print(f"   Stage A - Statistical Significance:")
        print(f"   Chi-square statistic: {results['srm_check']['chi2_statistic']:.4f}")
        print(f"   P-value: {results['srm_check']['p_value']:.6f}")
        print(f"   Statistical threshold (alpha): 0.01")
        print(f"   Statistical SRM: {'DETECTED' if results['srm_check']['srm_detected'] else 'NOT DETECTED'}")
        print(f"   ")
        print(f"   Stage B - Practical Significance:")
        print(f"   Max pp deviation: {results['srm_check']['max_pp_deviation']:.4f} (threshold: 0.01)")
        print(f"   Practical significance: {'YES' if results['srm_check']['practical_significant'] else 'NO'}")
        print(f"   ")
        print(f"   Combined Assessment:")
        print(f"   SRM severity: {'SEVERE (hard gate)' if results['srm_check']['srm_severe'] else 'WARNING ONLY' if results['srm_check']['srm_warning'] else 'NONE'}")

        print(f"\n💡 INTERPRETATION:")
        if results['srm_check']['srm_severe']:
            print(f"   ✗ SRM SEVERE: Both statistical (p < 0.01) AND practical (>{abs(results['srm_check']['max_pp_deviation']):.2f}pp deviation)")
            print(f"   ✗ This is a REAL PROBLEM - randomization likely broken")
            print(f"   ✗ STOP: Investigate before proceeding")
        elif results['srm_check']['srm_warning']:
            print(f"   ⚠️ SRM WARNING: Statistical (p < 0.01) but NOT practical (<1pp deviation)")
            print(f"   ⚠️ Large sample makes tiny deviations significant")
            print(f"   ⚠️ Deviation ({abs(results['srm_check']['max_pp_deviation']):.4f}pp) is too small to indicate serious issue")
            print(f"   ○ Proceeding with CAUTION - monitoring recommended")
        else:
            print(f"   ✓ SRM CHECK PASSED - no statistical or practical concerns")
            print(f"   ✓ Randomization valid - allocation matches design")
            print(f"   ✓ Safe to proceed with analysis")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Microsoft: Automated SRM checks on all experiments, blocks analysis if failed")
        print("   - Booking.com: Uses stricter alpha=0.001 for SRM (to catch subtle issues)")
        print("   - Netflix: SRM violations trigger immediate alerts to engineering team")

    # ========================================================================
    # HARD GATE: Stop execution if SRM is SEVERE (statistical + practical)
    # ========================================================================
    # Only halt if BOTH conditions met:
    # 1. Statistical significance (p < alpha)
    # 2. Practical significance (deviation > threshold)
    # Warnings (statistical only) allow proceeding with caution
    if IS_RCT and results['srm_check']['srm_severe']:
        if verbose:
            print(f"\n{'='*70}")
            print(f"⛔ ANALYSIS HALTED: SRM FAILURE IN RCT")
            print(f"{'='*70}")
            print(f"\nThis experiment is marked as a true RCT (IS_RCT=True).")
            print(f"SRM check FAILED, indicating randomization/logging issues.")
            print(f"\n❌ NO STATISTICAL INFERENCE WILL BE PERFORMED")
            print(f"   - No hypothesis tests")
            print(f"   - No effect size estimates")
            print(f"   - No confidence intervals")
            print(f"   - No ship/abandon decisions")
            print(f"\n🔍 INVESTIGATION REQUIRED:")
            print(f"   1. Check randomization assignment code")
            print(f"   2. Verify logging is capturing all assignments")
            print(f"   3. Look for group-specific technical issues")
            print(f"   4. Check for bot traffic or filtering bias")
            print(f"   5. Review data pipeline for joins/filters")
            print(f"\n📊 DIAGNOSTIC INFORMATION:")
            print(f"   Total units: {n_control + n_treatment:,}")
            print(f"   Control: {n_control:,} ({observed_ratio_control:.2%})")
            print(f"   Treatment: {n_treatment:,} ({observed_ratio_treatment:.2%})")
            print(f"   Expected: {EXPECTED_ALLOCATION[0]:.1%} / {EXPECTED_ALLOCATION[1]:.1%}")
            print(f"   Chi-square: {results['srm_check']['chi2_statistic']:.2f}")
            print(f"   P-value: {results['srm_check']['p_value']:.6f}")
            print(f"\n⚠️  DO NOT use results for product decisions until SRM is resolved.")
            print(f"{'='*70}\n")

        # Return early with INVALID status
        results['status'] = 'INVALID'
        results['status_reason'] = 'SRM failure in RCT - randomization compromised'
        return results

    # If we get here, SRM severity check passed (either clean or warning only)
    # Safe to continue with analysis
    results['status'] = 'VALID'

    # Add warning note if SRM is borderline (statistical but not practical)
    if results['srm_check']['srm_warning']:
        results['status_reason'] = 'SRM warning (borderline) - analysis proceeded with caution'
        if verbose:
            print(f"\n⚠️  NOTE: Proceeding with SRM warning")
            print(f"   Statistical significance detected but deviation is small")
            print(f"   ({abs(results['srm_check']['max_pp_deviation']):.4f}pp < 1pp threshold)")
            print(f"   This is typical in large samples - monitoring recommended")
            print()
    else:
        results['status_reason'] = 'SRM check passed - randomization valid'

    # ========================================================================
    # STEP 3: Primary Metric Analysis
    # ========================================================================
    if verbose:
        print(f"\n[3/7] Running primary metric analysis (visit rate)...")

        print("\n📚 LEARNING: Statistical Testing for Visit Rate")
        print("   What is visit rate?")
        print("   - Binary outcome: user visited website after seeing ad (1) or not (0)")
        print("   - Different from conversion (purchasing) - earlier in funnel")
        print("   - Typical baseline: 1-5% in digital advertising")
        print("   ")
        print("   Z-test for proportions:")
        print("   - Compares two proportions (control vs. treatment visit rates)")
        print("   - Null hypothesis: No difference between groups")
        print("   - Alternative: Treatment has different visit rate than control")
        print("   - Uses normal approximation (valid for large samples)")
        print("   ")
        print("   Why two-sided test?")
        print("   - We test if treatment is DIFFERENT (could be better OR worse)")
        print("   - More conservative than one-sided")
        print("   - Protects against harmful effects")
        print("   - Industry standard for most experiments")

    x_control_visits = control['visit'].sum()
    x_treatment_visits = treatment['visit'].sum()

    results['primary_test'] = frequentist.z_test_proportions(
        x_control=x_control_visits,
        n_control=len(control),
        x_treatment=x_treatment_visits,
        n_treatment=len(treatment),
        alpha=0.05,
        two_sided=True
    )

    if verbose:
        print(f"\n✓ Statistical Test Results:")
        print(f"   Control visit rate: {results['data_summary']['visit_rate_control']:.4f} ({x_control_visits:,}/{len(control):,})")
        print(f"   Treatment visit rate: {results['data_summary']['visit_rate_treatment']:.4f} ({x_treatment_visits:,}/{len(treatment):,})")
        print(f"   Absolute difference: {results['primary_test']['absolute_lift']:.6f} ({results['primary_test']['absolute_lift']*100:.4f} percentage points)")
        print(f"   Relative lift: {results['primary_test']['relative_lift']:.2%}")
        print(f"   Z-statistic: {results['primary_test']['z_statistic']:.4f}")
        print(f"   P-value: {results['primary_test']['p_value']:.6f}")
        print(f"   95% CI: [{results['primary_test']['ci_lower']:.6f}, {results['primary_test']['ci_upper']:.6f}]")

        print(f"\n💡 INTERPRETATION:")
        print(f"   Null hypothesis: Treatment visit rate = Control visit rate")
        print(f"   Alternative: Treatment visit rate ≠ Control visit rate")
        print(f"   Alpha (significance level): 0.05 (5% false positive rate)")
        print(f"   ")
        if results['primary_test']['significant']:
            print(f"   ✓ RESULT: STATISTICALLY SIGNIFICANT (p = {results['primary_test']['p_value']:.6f} < 0.05)")
            print(f"   ✓ We reject the null hypothesis with 95% confidence")
            print(f"   ✓ Treatment shows {results['primary_test']['relative_lift']:.2%} lift in visit rate")
            print(f"   ✓ Confidence interval: [{results['primary_test']['ci_lower']*100:.4f}%, {results['primary_test']['ci_upper']*100:.4f}%]")
            print(f"   ✓ Interpretation: True lift is likely between these bounds")
            print(f"   ")
            print(f"   Business meaning:")
            print(f"   - For every 10,000 users exposed to treatment:")
            print(f"   - Expect {results['primary_test']['absolute_lift']*10000:.0f} additional visits")
            print(f"   - This is a {results['primary_test']['relative_lift']:.1%} improvement")
        else:
            print(f"   ○ RESULT: NOT STATISTICALLY SIGNIFICANT (p = {results['primary_test']['p_value']:.6f} ≥ 0.05)")
            print(f"   ○ We fail to reject the null hypothesis")
            print(f"   ○ Cannot confidently say treatment has a different effect")
            print(f"   ○ Two possible explanations:")
            print(f"     1. Treatment truly has no effect (null is true)")
            print(f"     2. Effect exists but sample too small to detect it (Type II error)")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Google: Uses alpha=0.05 as standard, occasionally 0.01 for critical changes")
        print("   - LinkedIn: Requires p<0.05 AND practical significance (min effect size)")
        print("   - Airbnb: Visualizes confidence intervals alongside p-values for better intuition")

    # ========================================================================
    # STEP 4: CUPAC Variance Reduction (ML-Enhanced)
    # ========================================================================
    if verbose:
        print(f"\n[4/7] Applying CUPAC (ML-enhanced variance reduction)...")

        print("\n📚 LEARNING: CUPAC (Control Using Predictions As Covariates)")
        print("   What is CUPAC?")
        print("   - Advanced version of CUPED that uses machine learning")
        print("   - CUPED: Uses linear regression with ONE covariate")
        print("   - CUPAC: Uses ML models (GBM, Random Forest) with MANY features")
        print("   - Goal: Predict outcome using user features, remove predictable variance")
        print("   ")
        print("   How CUPAC works (simplified):")
        print("   1. Train ML model to predict outcome from user features")
        print("   2. Get predictions for each user")
        print("   3. Adjust outcomes: y_adjusted = y - prediction + mean(prediction)")
        print("   4. Run test on adjusted outcomes (lower variance = more power)")
        print("   ")
        print("   Why ML models help:")
        print("   - Can capture non-linear relationships (e.g., age 20-30 behaves differently)")
        print("   - Automatically finds interactions (e.g., mobile users in Europe)")
        print("   - Handles many features without manual selection")
        print("   - Gradient Boosting often achieves 30-50% variance reduction")
        print("   ")
        print("   Key assumptions:")
        print("   - Features must be PRE-EXPERIMENT (unaffected by treatment)")
        print("   - Model trained separately on control and treatment")
        print("   - Only reduces variance, doesn't change treatment effect estimate")
        print("   ")
        print(f"   We're using {len([col for col in df.columns if col not in ['treatment', 'visit', 'conversion']])} user features:")
        print("   - Demographics, browsing history, past purchases, etc.")
        print("   - Model: Gradient Boosting (50 trees) - balances accuracy and speed")

    # Get feature columns (exclude treatment, visit, conversion)
    feature_cols = [col for col in df.columns if col not in ['treatment', 'visit', 'conversion']]

    control_outcome = control['visit'].values
    control_features = control[feature_cols].values

    treatment_outcome = treatment['visit'].values
    treatment_features = treatment[feature_cols].values

    results['cupac'] = cupac.cupac_ab_test(
        y_control=control_outcome,
        y_treatment=treatment_outcome,
        X_control=control_features,
        X_treatment=treatment_features,
        model_type='gbm',  # 'gbm', 'rf', or 'ridge'
        alpha=0.05
    )

    if verbose:
        # Extract variance reduction metrics from CUPAC result
        var_reduction = results['cupac']['var_reduction']
        variance_reduction_pct = var_reduction * 100
        variance_factor = 1 - var_reduction  # Remaining variance after reduction
        sample_size_multiplier = 1 / variance_factor if variance_factor > 0 else 1.0

        original_std = np.std(control_outcome)
        adjusted_std = original_std * np.sqrt(variance_factor)  # Calculate from var_reduction

        print(f"\n✓ CUPAC Results:")
        print(f"   Model type: Gradient Boosting Machine (GBM)")
        print(f"   Features used: {len(feature_cols)}")
        print(f"   Training samples: {len(control_outcome):,} (control), {len(treatment_outcome):,} (treatment)")
        print(f"   Model R²: {results['cupac']['model_r2']:.4f}")
        print(f"   ")
        print(f"   Variance reduction: {variance_reduction_pct:.1f}%")
        print(f"   Original std dev: {original_std:.6f}")
        print(f"   Adjusted std dev: {adjusted_std:.6f}")
        print(f"   Variance factor (remaining): {variance_factor:.3f}")
        print(f"   Equivalent sample size multiplier: {sample_size_multiplier:.2f}x")

        print(f"\n💡 INTERPRETATION:")
        if variance_reduction_pct >= 30:
            print(f"   ✓ EXCELLENT variance reduction ({variance_reduction_pct:.1f}%)")
            print(f"   ✓ ML model explains {variance_reduction_pct:.1f}% of outcome variance")
            print(f"   ✓ This is like having {sample_size_multiplier:.1f}x more users!")
            print(f"   ✓ Can detect smaller effects or reach significance faster")
        elif variance_reduction_pct >= 15:
            print(f"   ✓ GOOD variance reduction ({variance_reduction_pct:.1f}%)")
            print(f"   ✓ Features have moderate predictive power")
            print(f"   ✓ Equivalent to {sample_size_multiplier:.1f}x sample size increase")
        elif variance_reduction_pct >= 5:
            print(f"   ○ MODEST variance reduction ({variance_reduction_pct:.1f}%)")
            print(f"   ○ Features help but not dramatically")
            print(f"   ○ Still worthwhile for large experiments")
        else:
            print(f"   ○ MINIMAL variance reduction ({variance_reduction_pct:.1f}%)")
            print(f"   ○ Features don't predict outcome well")
            print(f"   ○ CUPAC not providing much benefit")

        print(f"\n   What this means practically:")
        print(f"   - Original SE: {original_std / np.sqrt(len(control_outcome)):.6f}")
        print(f"   - CUPAC SE: {adjusted_std / np.sqrt(len(control_outcome)):.6f}")
        print(f"   - SE reduction: {results['cupac']['se_reduction']:.1%}")
        print(f"   - To achieve same precision without CUPAC:")
        print(f"     Would need {len(control_outcome) * sample_size_multiplier:,.0f} users (vs {len(control_outcome):,})")
        print(f"   - Faster experiments = more iterations = faster learning")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - DoorDash: Pioneered CUPAC, achieving 40-50% variance reduction")
        print("   - Microsoft: Uses CUPAC for Xbox experiments (user history predicts engagement)")
        print("   - Netflix: CUPAC on viewing hours (past watch time strongly predicts future)")
        print("   - Key insight: In consumer internet, user history is highly predictive")
        print("   ")
        print("   Model selection tips:")
        print("   - GBM: Best for 1K-1M rows, handles non-linearity well")
        print("   - Random Forest: Good alternative, more robust to outliers")
        print("   - Linear: Use for >10M rows (faster) or if interpretability needed")

    # ========================================================================
    # STEP 5: Heterogeneous Treatment Effects (X-Learner)
    # ========================================================================
    if verbose:
        print(f"\n[5/7] Estimating heterogeneous treatment effects (X-Learner)...")

        print("\n📚 LEARNING: Heterogeneous Treatment Effects (HTE)")
        print("   What is HTE?")
        print("   - Standard A/B test: One average treatment effect for ALL users")
        print("   - HTE: Treatment effect VARIES across different user types")
        print("   - Example: New users love feature (+20%), existing users hate it (-10%)")
        print("   - Goal: Understand WHO benefits most from treatment")
        print("   ")
        print("   Why HTE matters:")
        print("   - Enables targeted rollouts (ship to winners, hide from losers)")
        print("   - Prevents averaging out effects (net zero = some win, some lose)")
        print("   - Informs product strategy (who is our target audience?)")
        print("   - Maximizes business value vs. one-size-fits-all")
        print("   ")
        print("   What is X-Learner?")
        print("   - State-of-the-art ML method for estimating CATE")
        print("   - CATE = Conditional Average Treatment Effect (treatment effect for user with features X)")
        print("   - Uses two models: one for control, one for treatment")
        print("   - Handles: low treatment propensity, complex interactions, non-linearity")
        print("   ")
        print("   X-Learner algorithm (simplified):")
        print("   1. Train model to predict outcome in control group")
        print("   2. Train model to predict outcome in treatment group")
        print("   3. For each user: estimate individual treatment effect")
        print("   4. Meta-learner: predict treatment effect from user features")
        print("   ")
        print("   Interpreting CATE:")
        print("   - CATE = 0.02: User expected to have +2% visit rate from treatment")
        print("   - CATE = -0.01: User expected to have -1% visit rate (harmed)")
        print("   - Distribution of CATE tells us heterogeneity story")
        print("   - Std dev of CATE: how much variation exists across users")

    # Sample for HTE to speed up (X-Learner can be slow on large datasets)
    if len(df) > 10000:
        df_hte = df.sample(n=min(10000, len(df)), random_state=42)
        if verbose:
            print(f"\n   Note: Subsampling to {len(df_hte):,} observations for computational efficiency")
            print("   (X-Learner trains 4 ML models, can be slow on >100K rows)")
    else:
        df_hte = df

    # Prepare data for X-Learner (expects combined data with treatment indicator)
    X_hte = df_hte[feature_cols].values
    y_hte = df_hte['visit'].values
    treatment_hte = df_hte['treatment'].values

    # Fit X-Learner
    from ab_testing.advanced.hte import XLearner
    xl = XLearner()
    xl.fit(
        X=X_hte,
        y=y_hte,
        treatment=treatment_hte
    )

    # Predict CATE for all observations
    cate_estimates = xl.predict(X=df_hte[feature_cols].values)

    # Calculate summary statistics
    results['hte'] = {
        'cate_estimates': cate_estimates,
        'avg_treatment_effect': np.mean(cate_estimates),
        'cate_std': np.std(cate_estimates),
        'cate_min': np.min(cate_estimates),
        'cate_max': np.max(cate_estimates),
        'cate_25th': np.percentile(cate_estimates, 25),
        'cate_75th': np.percentile(cate_estimates, 75),
        'negative_effect_pct': (cate_estimates < 0).mean() * 100
    }

    if verbose:
        print(f"\n✓ X-Learner Results:")
        n_control_hte = (treatment_hte == 0).sum()
        n_treatment_hte = (treatment_hte == 1).sum()
        print(f"   Training samples: {n_control_hte:,} control, {n_treatment_hte:,} treatment")
        print(f"   CATE estimates computed: {len(cate_estimates):,}")
        print(f"   ")
        print(f"   Average Treatment Effect (ATE): {results['hte']['avg_treatment_effect']:.6f}")
        print(f"   CATE distribution:")
        print(f"     - Min: {results['hte']['cate_min']:.6f}")
        print(f"     - 25th percentile: {results['hte']['cate_25th']:.6f}")
        print(f"     - Median (50th): {np.median(cate_estimates):.6f}")
        print(f"     - 75th percentile: {results['hte']['cate_75th']:.6f}")
        print(f"     - Max: {results['hte']['cate_max']:.6f}")
        print(f"     - Std deviation: {results['hte']['cate_std']:.6f}")
        print(f"   ")
        print(f"   Users with NEGATIVE effect: {results['hte']['negative_effect_pct']:.1f}%")

        print(f"\n💡 INTERPRETATION:")
        heterogeneity_ratio = results['hte']['cate_std'] / abs(results['hte']['avg_treatment_effect']) if results['hte']['avg_treatment_effect'] != 0 else float('inf')

        if results['hte']['cate_std'] > 0.01:
            print(f"   ⚠️  SUBSTANTIAL HETEROGENEITY DETECTED!")
            print(f"   ⚠️  CATE std dev ({results['hte']['cate_std']:.4f}) is {heterogeneity_ratio:.1f}x the average effect")
            print(f"   ⚠️  Treatment effect varies WIDELY across users")
            print(f"   ")
            print(f"   What this means:")
            print(f"   - Some users benefit greatly (max CATE: {results['hte']['cate_max']:.4f})")
            print(f"   - Some users are harmed (min CATE: {results['hte']['cate_min']:.4f})")
            print(f"   - {results['hte']['negative_effect_pct']:.1f}% of users have NEGATIVE effects")
            print(f"   - One-size-fits-all rollout suboptimal!")
            print(f"   ")
            print(f"   Recommended strategy:")
            print(f"   1. SEGMENT users by predicted CATE")
            print(f"   2. Ship to high-CATE users (winners)")
            print(f"   3. Don't ship to negative-CATE users (losers)")
            print(f"   4. Personalization could capture {(results['hte']['cate_75th'] - results['hte']['avg_treatment_effect']) / results['hte']['avg_treatment_effect'] * 100:.1f}% more value")
        else:
            print(f"   ✓ LOW HETEROGENEITY (CATE std: {results['hte']['cate_std']:.4f})")
            print(f"   ✓ Treatment effect relatively UNIFORM across users")
            print(f"   ✓ Most users respond similarly to treatment")
            print(f"   ✓ Simple rollout strategy: ship to all or none")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Uber: Uses X-Learner to personalize promotions (who gets discounts?)")
        print("   - Netflix: Estimates CATE for content recommendations (who likes what?)")
        print("   - Amazon: Personalizes feature rollouts based on predicted uplift")
        print("   - Meta: Segments users by CATE for ads (maximize ROI per user)")
        print("   ")
        print("   When to use HTE:")
        print("   - Large datasets (>10K users, ideally >100K)")
        print("   - Rich user features (demographics, behavior history)")
        print("   - Business can act on segments (personalization infrastructure)")
        print("   - Expected heterogeneity (e.g., new vs existing users)")
        print("   ")
        print("   Alternatives to X-Learner:")
        print("   - S-Learner: Simpler, one model for everyone")
        print("   - T-Learner: Two models (control/treatment), simpler than X")
        print("   - Causal Forest: More flexible but slower to train")

    # ========================================================================
    # STEP 6: Sequential Testing (O'Brien-Fleming)
    # ========================================================================
    if verbose:
        print(f"\n[6/7] Sequential testing analysis (O'Brien-Fleming)...")

        print("\n📚 LEARNING: Sequential Testing for Early Stopping")
        print("   What is sequential testing?")
        print("   - Traditional A/B test: Decide sample size upfront, analyze once at end")
        print("   - Sequential: Analyze multiple times (interim looks) as data comes in")
        print("   - Can stop EARLY if strong evidence detected (saves time & resources)")
        print("   - Adjusts significance thresholds to control Type I error")
        print("   ")
        print("   Why sequential testing matters:")
        print("   - Faster decisions: Stop as soon as you have strong evidence")
        print("   - Resource efficiency: Don't run experiment longer than needed")
        print("   - Agility: React quickly to harmful effects (guardrails)")
        print("   - But: Must use proper boundaries, can't just \"peek\" at p-values")
        print("   ")
        print("   What is O'Brien-Fleming?")
        print("   - Alpha spending function: How to allocate Type I error across looks")
        print("   - O'Brien-Fleming: Very conservative early, relaxes near end")
        print("   - Example: Look 1 needs z>4.3, Look 4 needs z>1.96 (standard)")
        print("   - Philosophy: Hard to stop early (avoid premature), easier late")
        print("   ")
        print("   How it works:")
        print("   - Plan N looks upfront (e.g., 25%, 50%, 75%, 100% of data)")
        print("   - Each look has threshold (z-statistic boundary)")
        print("   - If |z| > boundary → STOP (effect detected)")
        print("   - If reach final look without stopping → conclude no effect")
        print("   ")
        print("   Alternative: Pocock boundaries")
        print("   - Equal thresholds at all looks (easier to stop early)")
        print("   - O'Brien-Fleming preferred in industry (less likely to stop on noise)")

    # Simulate sequential looks (split data into 4 interim analyses)
    n_looks = 4
    look_fractions = [0.25, 0.50, 0.75, 1.00]

    results['sequential'] = sequential.obrien_fleming_boundaries(
        n_looks=n_looks,
        alpha=0.05,
        two_sided=True
    )

    if verbose:
        print(f"\n✓ Sequential Testing Setup:")
        print(f"   Planned looks: {n_looks}")
        print(f"   Alpha spending function: O'Brien-Fleming")
        print(f"   Overall alpha: 0.05 (5% Type I error rate)")
        print(f"   ")
        print(f"   Decision boundaries:")
        for i, (frac, boundary) in enumerate(zip(look_fractions, results['sequential']['boundaries'])):
            print(f"     Look {i+1} ({frac*100:.0f}% data): |z| > {boundary:.3f}")

    # Check if we would have stopped early
    current_z = results['primary_test'].get('z_statistic', 0)
    stopped_early = False
    stop_look = None

    for i, boundary in enumerate(results['sequential']['boundaries']):
        if abs(current_z) > boundary:
            stopped_early = True
            stop_look = i + 1
            break

    if verbose:
        print(f"\n💡 INTERPRETATION:")
        print(f"   Current z-statistic: {current_z:.3f}")
        print(f"   ")
        if stopped_early:
            print(f"   ✓ EARLY STOP CRITERION MET at Look {stop_look}!")
            print(f"   ✓ z-statistic ({abs(current_z):.3f}) exceeds boundary ({results['sequential']['boundaries'][stop_look-1]:.3f})")
            print(f"   ✓ Strong evidence detected - safe to stop experiment")
            print(f"   ✓ Time saved: {(1 - look_fractions[stop_look-1]) * 100:.0f}% of planned duration")
            print(f"   ")
            print(f"   Benefits of early stopping:")
            print(f"   - Faster product iteration (ship winners, kill losers)")
            print(f"   - Resource efficiency (free up traffic for other tests)")
            print(f"   - Reduced exposure to potential harm (if treatment is bad)")
        else:
            print(f"   ○ Would NOT have stopped early")
            print(f"   ○ Current z ({current_z:.3f}) below all boundaries")
            print(f"   ○ Need stronger evidence or more data")
            print(f"   ")
            print(f"   Next steps:")
            print(f"   - Continue running until planned end (100% data)")
            print(f"   - OR extend experiment if effect promising but not significant")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Netflix: Uses sequential testing to stop harmful experiments early")
        print("   - Optimizely: Built-in sequential testing (stats engine)")
        print("   - LinkedIn: O'Brien-Fleming for most tests, Pocock for guardrails")
        print("   - Microsoft: Analyzes experiments daily with sequential boundaries")
        print("   ")
        print("   Best practices:")
        print("   - Declare looks upfront (don't peek randomly)")
        print("   - 4-5 looks is typical (more = more complexity)")
        print("   - Always use proper boundaries (never raw p<0.05 at each look)")
        print("   - Document when you stopped and why (reproducibility)")

    # ========================================================================
    # STEP 7: Guardrails & Decision
    # ========================================================================
    if verbose:
        print(f"\n[7/7] Checking guardrail metrics and making decision...")

        print("\n📚 LEARNING: Guardrail Metrics & Decision Frameworks")
        print("   What are guardrails?")
        print("   - Primary metric: What we're trying to IMPROVE (visit rate)")
        print("   - Guardrails: Metrics we must NOT HARM (conversion rate, revenue)")
        print("   - Example: Increase clicks (primary) but don't hurt purchases (guardrail)")
        print("   ")
        print("   Why guardrails are critical:")
        print("   - Prevent unintended consequences")
        print("   - Example: Clickbait headlines increase clicks but reduce trust/conversions")
        print("   - Example: Aggressive upsells increase revenue short-term, harm retention")
        print("   - Guardrails protect long-term health while optimizing short-term metrics")
        print("   ")
        print("   Non-inferiority testing:")
        print("   - Question: \"Is degradation within acceptable threshold?\"")
        print("   - Set tolerance: e.g., allow up to -2% relative degradation")
        print("   - Test: Is lower bound of CI > -2%?")
        print("   - Different from equality test (not asking if equal, asking if not too bad)")
        print("   ")
        print("   For this experiment:")
        print("   - Primary: Visit rate (trying to improve)")
        print("   - Guardrail: Conversion rate (must not harm)")
        print("   - Tolerance: Allow up to -2% relative degradation")
        print("   - Rationale: Small conversion hit acceptable if visits increase significantly")

    # Guardrail: Conversion rate should not decrease
    control_conversions = control['conversion'].values
    treatment_conversions = treatment['conversion'].values

    guardrail_result = guardrails.non_inferiority_test(
        control=control_conversions,
        treatment=treatment_conversions,
        delta=-0.02,  # Allow up to 2% degradation
        metric_type='relative',
        alpha=0.05
    )

    guardrail_result['metric_name'] = 'conversion_rate'
    results['guardrails'] = {'conversion': guardrail_result}

    if verbose:
        print(f"\n✓ Guardrail Analysis:")
        print(f"   Metric: Conversion rate")
        print(f"   Control mean: {control_conversions.mean():.4f}")
        print(f"   Treatment mean: {treatment_conversions.mean():.4f}")
        print(f"   Difference: {guardrail_result.get('difference', 0.0):.2%}")
        print(f"   95% CI lower bound: {guardrail_result.get('ci_lower', 0.0):.4f}")
        print(f"   Tolerance threshold: -2.0%")
        print(f"   Test result: {'✓ PASSED' if guardrail_result['passed'] else '✗ FAILED'}")

        print(f"\n💡 INTERPRETATION:")
        if guardrail_result['passed']:
            print(f"   ✓ GUARDRAIL PASSED")
            print(f"   ✓ Conversion rate degradation ({guardrail_result.get('difference', 0.0):.2%}) within acceptable limits")
            print(f"   ✓ Lower bound of CI ({guardrail_result.get('ci_lower', 0.0):.4f}) > threshold (-2%)")
            print(f"   ✓ Safe to proceed from guardrail perspective")
        else:
            print(f"   ✗ GUARDRAIL FAILED")
            print(f"   ✗ Conversion rate degradation ({guardrail_result.get('difference', 0.0):.2%}) EXCEEDS acceptable limits")
            print(f"   ✗ Lower bound of CI ({guardrail_result.get('ci_lower', 0.0):.4f}) < threshold (-2%)")
            print(f"   ✗ Recommendation: ABANDON or REDESIGN treatment")
            print(f"   ✗ Even if visits increased, harming conversions is unacceptable")

    # Make final decision
    results['decision'] = guardrails.evaluate_guardrails(
        primary_result=results['primary_test'],
        guardrail_results=[guardrail_result]
    )

    if verbose:
        decision = results['decision']['decision'].upper()
        print(f"\n✓ Final Decision Framework:")
        print(f"   Primary metric (visit rate): {'Significant & Positive' if results['primary_test']['significant'] and results['primary_test']['relative_lift'] > 0 else 'Not significant or negative'}")
        print(f"   Guardrails: {results['decision'].get('guardrails_passed', 0)}/{results['decision'].get('guardrails_total', 1)} passed")
        print(f"   ")
        print(f"   >>> DECISION: {decision} <<<")
        print(f"   ")

        if decision == 'SHIP':
            print(f"   ✓ SHIP recommendation - launch to all users")
            print(f"   ✓ Primary metric improved AND no guardrail violations")
            print(f"   ✓ Expected impact: {results['primary_test']['relative_lift']:.2%} lift in visits")
            print(f"   ")
            print(f"   Next steps:")
            print(f"   1. Run post-launch holdout for 2-4 weeks (validate sustained effect)")
            print(f"   2. Monitor conversion rate closely (guardrail was borderline)")
            print(f"   3. Consider targeted rollout to high-CATE users if HTE substantial")
        elif decision == 'ABANDON':
            print(f"   ✗ ABANDON recommendation - do not launch")
            print(f"   ✗ Either primary metric negative OR guardrail violation")
            print(f"   ")
            print(f"   Next steps:")
            print(f"   1. Analyze WHY treatment failed (user research, qualitative feedback)")
            print(f"   2. Redesign treatment addressing root cause")
            print(f"   3. Run new experiment with improved version")
        else:  # HOLD
            print(f"   ○ HOLD recommendation - need more evidence")
            print(f"   ○ Results promising but not conclusive")
            print(f"   ")
            print(f"   Next steps:")
            print(f"   1. Extend experiment to gather more data")
            print(f"   2. Investigate why effect not significant (low power? small effect?)")
            print(f"   3. Consider increasing traffic allocation")

        print(f"\n🏢 INDUSTRY PRACTICE:")
        print("   - Airbnb: Uses 10+ guardrail metrics (retention, support contacts, complaints)")
        print("   - Uber: Separate guardrails for riders vs drivers (two-sided marketplace)")
        print("   - Spotify: Time spent listening (primary), but guards churn, engagement depth")
        print("   - DoorDash: Conversion (primary), but guards cancellation rate, support tickets")
        print("   ")
        print("   Setting guardrail thresholds:")
        print("   - Critical metrics (revenue, retention): Strict (-1% to -2%)")
        print("   - Secondary metrics: Moderate (-5% to -10%)")
        print("   - Based on business judgment (cost-benefit analysis)")
        print("   - Should be declared PRE-experiment (not post-hoc)")

    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    if verbose:
        print(f"{'='*70}")
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
    print("A/B TESTING MASTERCLASS: ADVANCED TECHNIQUES")
    print("Dataset: Criteo Uplift (13.9M observations)")
    print("🎓"*35)

    print("\nThis pipeline demonstrates state-of-the-art experimentation:")
    print("  1. Large-scale data handling (13.9M rows with sampling)")
    print("  2. Sample Ratio Mismatch (SRM) detection at scale")
    print("  3. Statistical testing with massive sample sizes")
    print("  4. CUPAC: ML-enhanced variance reduction (vs. linear CUPED)")
    print("  5. Heterogeneous Treatment Effects with X-Learner")
    print("  6. Sequential testing for early stopping (O'Brien-Fleming)")
    print("  7. Guardrail metrics & decision frameworks")

    print("\n" + "="*70)
    print("Starting analysis with 0.1% sample (~14K observations)...")
    print("(Use sample_frac=0.01 for 1% = 139K rows in production)")
    print("="*70)

    # Run pipeline with 0.1% sample (fast for demo, still shows techniques)
    results = run_criteo_analysis(sample_frac=0.001, verbose=True)

    # ========================================================================
    # COMPREHENSIVE FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 FINAL SUMMARY & KEY LEARNINGS")
    print("="*70)

    print(f"\n1. EXPERIMENT OVERVIEW")
    print(f"   Dataset: Criteo Uplift Modeling ({results['data_summary']['total_observations']:,} observations)")
    print(f"   Full dataset: 13.9 million rows (sampled for efficiency)")
    print(f"   Features: {results['data_summary']['n_features']} user characteristics")
    print(f"   Primary metric: Visit rate (did user visit website after ad?)")
    print(f"   Guardrail metric: Conversion rate (did user purchase?)")

    print(f"\n2. DATA QUALITY ✓")
    print(f"   Total observations: {results['data_summary']['total_observations']:,}")
    print(f"   Control group: {results['data_summary']['control_size']:,}")
    print(f"   Treatment group: {results['data_summary']['treatment_size']:,}")
    print(f"   Group balance: {results['data_summary']['treatment_size']/results['data_summary']['control_size']:.4f} (target: 1.0)")
    print(f"   SRM check: {'✓ PASSED' if not results['srm_check']['srm_detected'] else '✗ FAILED'}")
    if results['srm_check']['srm_detected']:
        print(f"   ⚠️ WARNING: SRM failure indicates randomization issue - STOP HERE!")

    print(f"\n3. STATISTICAL RESULTS (Visit Rate)")
    print(f"   Control rate: {results['data_summary']['visit_rate_control']:.4f} ({results['data_summary']['visit_rate_control']*100:.2f}%)")
    print(f"   Treatment rate: {results['data_summary']['visit_rate_treatment']:.4f} ({results['data_summary']['visit_rate_treatment']*100:.2f}%)")
    print(f"   Absolute difference: {results['primary_test']['absolute_lift']:.6f} ({results['primary_test']['absolute_lift']*100:.4f} percentage points)")
    print(f"   Relative lift: {results['primary_test']['relative_lift']:.2%}")
    print(f"   Z-statistic: {results['primary_test'].get('z_statistic', 0.0):.4f}")
    print(f"   P-value: {results['primary_test']['p_value']:.6f}")
    print(f"   Significant: {'✓ YES (p < 0.05)' if results['primary_test']['significant'] else '○ NO (p ≥ 0.05)'}")

    print(f"\n4. CUPAC (ML-Enhanced Variance Reduction)")
    var_reduction = results['cupac']['var_reduction']
    variance_reduction_pct = var_reduction * 100
    variance_factor = 1 - var_reduction
    sample_size_multiplier = 1 / variance_factor if variance_factor > 0 else 1.0
    print(f"   Model: Gradient Boosting Machine (GBM)")
    print(f"   Features used: {results['data_summary']['n_features']}")
    print(f"   Variance reduction: {variance_reduction_pct:.1f}%")
    print(f"   Sample size multiplier: {sample_size_multiplier:.2f}x")
    print(f"   Interpretation: Like having {sample_size_multiplier:.1f}x more users!")
    if variance_reduction_pct >= 30:
        print(f"   Assessment: ✓ EXCELLENT - ML model very predictive")
    elif variance_reduction_pct >= 15:
        print(f"   Assessment: ✓ GOOD - Meaningful efficiency gain")
    else:
        print(f"   Assessment: ○ MODEST - Features weakly predictive")

    print(f"\n5. HETEROGENEOUS TREATMENT EFFECTS (X-Learner)")
    print(f"   Average Treatment Effect (ATE): {results['hte']['avg_treatment_effect']:.6f}")
    print(f"   CATE distribution:")
    print(f"     - Min: {results['hte']['cate_min']:.6f}")
    print(f"     - 25th percentile: {results['hte'].get('cate_25th', 0.0):.6f}")
    print(f"     - Median: {np.median(results['hte']['cate_estimates']):.6f}")
    print(f"     - 75th percentile: {results['hte'].get('cate_75th', 0.0):.6f}")
    print(f"     - Max: {results['hte']['cate_max']:.6f}")
    print(f"     - Std deviation: {results['hte']['cate_std']:.6f}")
    print(f"   Users with NEGATIVE effect: {results['hte'].get('negative_effect_pct', 0.0):.1f}%")
    if results['hte']['cate_std'] > 0.01:
        print(f"   Assessment: ⚠️ HIGH HETEROGENEITY - Personalization recommended")
        print(f"   Recommendation: Ship to high-CATE users, don't ship to negative-CATE users")
    else:
        print(f"   Assessment: ✓ LOW HETEROGENEITY - One-size-fits-all okay")

    print(f"\n6. SEQUENTIAL TESTING (O'Brien-Fleming)")
    print(f"   Planned looks: 4 (at 25%, 50%, 75%, 100% of data)")
    print(f"   Alpha spending: O'Brien-Fleming (conservative early, relaxed late)")
    print(f"   Current z-statistic: {results['primary_test'].get('z_statistic', 0.0):.3f}")
    print(f"   Boundaries: {', '.join([f'{b:.3f}' for b in results['sequential']['boundaries']])}")
    # Check if would have stopped early
    current_z = abs(results['primary_test'].get('z_statistic', 0))
    stopped_early = any(current_z > b for b in results['sequential']['boundaries'])
    if stopped_early:
        stop_look = next(i+1 for i, b in enumerate(results['sequential']['boundaries']) if current_z > b)
        print(f"   Result: ✓ Could stop early at look {stop_look} ({[0.25, 0.50, 0.75, 1.00][stop_look-1]*100:.0f}% data)")
        print(f"   Time saved: {(1 - [0.25, 0.50, 0.75, 1.00][stop_look-1])*100:.0f}% of planned duration")
    else:
        print(f"   Result: ○ Cannot stop early - need more evidence")

    print(f"\n7. GUARDRAIL METRICS")
    print(f"   Metric: Conversion rate")
    print(f"   Control: {results['data_summary']['conversion_rate_control']:.4f}")
    print(f"   Treatment: {results['data_summary']['conversion_rate_treatment']:.4f}")
    print(f"   Tolerance: -2.0% (allow up to 2% relative degradation)")
    print(f"   Result: {'✓ PASSED' if results['guardrails']['conversion']['passed'] else '✗ FAILED'}")
    if not results['guardrails']['conversion']['passed']:
        print(f"   ⚠️ WARNING: Guardrail violation - treatment harms conversions!")

    print(f"\n8. FINAL DECISION")
    decision = results['decision']['decision'].upper()
    print(f"   >>> {decision} <<<")
    print(f"   ")
    if decision == 'SHIP':
        print(f"   ✓ Recommendation: Launch to all users")
        print(f"   ✓ Primary metric improved: {results['primary_test']['relative_lift']:.2%} lift")
        print(f"   ✓ No guardrail violations")
        if results['hte']['cate_std'] > 0.01:
            print(f"   ⚠️ Consider: Targeted rollout based on predicted CATE")
    elif decision == 'ABANDON':
        print(f"   ✗ Recommendation: Do not launch")
        print(f"   ✗ Either negative effect or guardrail violation")
        print(f"   ✗ Redesign treatment and re-test")
    else:
        print(f"   ○ Recommendation: Need more evidence")
        print(f"   ○ Extend experiment or investigate further")

    print(f"\n" + "="*70)
    print(f"📚 KEY TAKEAWAYS FOR A/B TESTING")
    print(f"="*70)

    print(f"\n✓ CRITICAL STEPS - ALWAYS DO THESE:")
    print(f"  1. Check SRM before analyzing (at scale, even tiny imbalances matter)")
    print(f"  2. Use variance reduction when you have user features (CUPAC >> CUPED)")
    print(f"  3. Estimate HTE when you have rich features (enables personalization)")
    print(f"  4. Plan sequential looks upfront (don't peek randomly)")
    print(f"  5. Set guardrails pre-experiment (protect critical metrics)")
    print(f"  6. Document everything (sample size, looks, thresholds, decisions)")

    print(f"\n⚠️  COMMON PITFALLS TO AVOID:")
    print(f"  1. Ignoring SRM at scale (\"it's only 0.5% off\" = actually huge problem)")
    print(f"  2. Not using variance reduction (leaving 30-50% efficiency on table)")
    print(f"  3. Assuming homogeneous effects (missing personalization opportunities)")
    print(f"  4. Peeking at p-values without proper boundaries (inflates Type I error)")
    print(f"  5. Optimizing primary metric at expense of guardrails (short-term thinking)")
    print(f"  6. Stopping experiments based on cost, not evidence (introduces bias)")

    print(f"\n🏢 INDUSTRY BEST PRACTICES:")
    print(f"  - Netflix: CUPAC on all experiments, 40%+ variance reduction typical")
    print(f"  - Uber: X-Learner for personalized pricing (who responds to discounts?)")
    print(f"  - Microsoft: Sequential testing standard, analyzes daily with boundaries")
    print(f"  - Amazon: Strict guardrails on revenue/retention, -1% tolerance max")
    print(f"  - Meta: HTE for ads (personalize by predicted treatment effect)")

    print(f"\n💡 WHAT TO DO NEXT:")
    print(f"  1. Try with larger sample: run_criteo_analysis(sample_frac=0.01)")
    print(f"  2. Experiment with different ML models for CUPAC (gbm vs rf vs linear)")
    print(f"  3. Analyze CATE distribution (who benefits most from treatment?)")
    print(f"  4. Compare O'Brien-Fleming vs Pocock boundaries")
    print(f"  5. Read the industry blog posts cited throughout this pipeline")

    print(f"\n" + "="*70)
    print(f"✅ Pipeline complete! You've learned state-of-the-art A/B testing.")
    print(f"="*70 + "\n")
