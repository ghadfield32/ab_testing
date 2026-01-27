# file: src/ab_testing_framework.py
"""
Production A/B Testing Framework v7.0 - COMPLETE LEARNING CURRICULUM
=====================================================================

IMPROVEMENTS OVER v6:
- ALL techniques now demonstrated in output (not just in code)
- Uses ACTUAL experiment data (no hardcoded values)
- Full explanations with WHY, WHEN TO USE, and KEY INSIGHTS
- Comparison tables showing technique improvements
- Proper effect size interpretation throughout
- Complete thought process for learning

Author: Geoff (Analytics Engineer / Data Scientist)
Last Updated: January 2026
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mstats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class DataGenerator:
    """Generate realistic experiment data with proper correlations."""
    
    @staticmethod
    def subscription_experiment(n: int = 100000, seed: int = 42) -> pd.DataFrame:
        """
        Novel use case: Subscription product onboarding redesign.
        
        Creates data with:
        - Pre-experiment metrics correlated with outcomes (for CUPED/CUPAC)
        - Heterogeneous treatment effects by segment
        - Triggered flag for ITT vs Per-Protocol
        - Multiple metrics for multiple testing demo
        """
        np.random.seed(seed)
        
        # Latent user quality drives everything
        user_quality = np.random.beta(2, 3, size=n)
        
        # Pre-experiment data (correlated with outcomes)
        pre_engagement = 0.7 * user_quality + 0.3 * np.random.beta(2, 3, size=n)
        pre_revenue = np.where(
            np.random.random(n) < (0.3 + 0.5 * pre_engagement),
            np.random.lognormal(4 + 1.5 * user_quality, 0.5, size=n), 0)
        pre_sessions = np.random.poisson(5 + 15 * pre_engagement, size=n)
        
        # Segments for HTE
        channel = np.random.choice(['Organic', 'Paid', 'Social', 'Referral'], n, p=[0.3, 0.25, 0.25, 0.2])
        device = np.random.choice(['iOS', 'Android', 'Web'], n, p=[0.4, 0.35, 0.25])
        
        # Treatment assignment
        variant = np.random.choice(['A', 'B'], n, p=[0.5, 0.5])
        is_treatment = variant == 'B'
        
        # Triggered (90% see treatment in B, 100% in A)
        triggered = np.where(is_treatment & (np.random.random(n) < 0.9), 1,
                           np.where(~is_treatment, 1, 0))
        
        # Conversion with heterogeneous treatment effect
        base_conv = 0.15 + 0.25 * pre_engagement
        te = 0.03 * (0.7 + 0.6 * pre_engagement) * is_treatment * triggered
        te = np.where(channel == 'Referral', te * 1.3, te)
        conv_prob = np.clip(base_conv + te, 0, 1)
        converted = np.random.binomial(1, conv_prob)
        
        # Revenue (only converters)
        revenue = np.zeros(n)
        conv_mask = converted == 1
        n_conv = conv_mask.sum()
        base_rev = np.random.lognormal(
            4.5 + 0.8 * user_quality[conv_mask] + 0.4 * np.log1p(pre_revenue[conv_mask])/5, 
            0.4, n_conv)
        te_rev = 15 * (1 + 0.5 * user_quality[conv_mask]) * is_treatment[conv_mask]
        revenue[conv_mask] = np.maximum(0, base_rev + te_rev)
        
        # Additional metrics for multiple testing
        sessions_week1 = np.random.poisson(3 + 8 * pre_engagement + 2 * converted + is_treatment, size=n)
        pages_viewed = np.random.poisson(5 + 10 * pre_engagement + 3 * converted + 2 * is_treatment, size=n)
        retention_7d = np.random.binomial(1, 0.4 + 0.3 * pre_engagement + 0.05 * is_treatment)
        nps_score = np.clip(5 + 3 * user_quality + 0.5 * is_treatment + np.random.normal(0, 1.5, n), 1, 10)
        
        return pd.DataFrame({
            'user_id': np.arange(1, n + 1),
            'variant': variant,
            'channel': channel,
            'device': device,
            'triggered': triggered,
            'pre_engagement': pre_engagement,
            'pre_revenue': pre_revenue,
            'pre_sessions': pre_sessions,
            'converted': converted,
            'revenue': revenue,
            'sessions_week1': sessions_week1,
            'pages_viewed': pages_viewed,
            'retention_7d': retention_7d,
            'nps_score': nps_score,
            '_user_quality': user_quality
        })


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE DEMO WITH FULL EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    """
    Complete A/B Testing curriculum with full explanations.
    Every technique is demonstrated with actual data and explained in detail.
    """
    
    print("═" * 90)
    print("█ COMPLETE A/B TESTING CURRICULUM - WITH FULL THOUGHT PROCESS")
    print("█ Everything you need to ace experimentation interviews")
    print("═" * 90)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("📊 DATA GENERATION: Subscription Onboarding Experiment")
    print("─" * 90)
    
    df = DataGenerator.subscription_experiment(n=100000)
    df_a = df[df['variant'] == 'A']
    df_b = df[df['variant'] == 'B']
    
    conv_a, conv_b = df_a['converted'].mean(), df_b['converted'].mean()
    n_a, n_b = len(df_a), len(df_b)
    
    print(f"""
Scenario: Testing redesigned onboarding flow for subscription product
- Control (A): Existing onboarding  
- Treatment (B): New personalized onboarding

Sample: {n_a + n_b:,} users ({n_a:,} control, {n_b:,} treatment)
Conversion: A = {conv_a:.2%}, B = {conv_b:.2%}
Observed Lift: {(conv_b/conv_a - 1)*100:.1f}% relative ({(conv_b - conv_a)*100:.2f} percentage points)
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 1: FUNDAMENTALS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 90)
    print("█ LEVEL 1: FUNDAMENTALS (Must-know for any DS role)")
    print("═" * 90)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1.1 SAMPLE SIZE / POWER ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📐 1.1 SAMPLE SIZE CALCULATION")
    print("─" * 90)
    
    print("""
🎯 THE QUESTION: "How many users do we need?"
   This is the #1 most common experimentation interview question.

📚 KEY CONCEPTS:
   • Power (1-β): Probability of detecting a TRUE effect. Typically 80%.
   • Alpha (α): False positive rate. Typically 5%.
   • MDE: Minimum Detectable Effect - smallest lift worth detecting.
   
📝 FORMULA (binary metrics):
   n = 2 × (z_α + z_β)² × p̄(1-p̄) / δ²
   
   where z_α=1.96 (α=0.05), z_β=0.84 (80% power), p̄=pooled rate, δ=effect size
""")
    
    # Calculate for binary
    baseline = 0.25
    mde_rel = 0.10
    p1, p2 = baseline, baseline * (1 + mde_rel)
    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
    n_binary = int(np.ceil(zt_ind_solve_power(effect_size=h, alpha=0.05, power=0.80)))
    
    print(f"""📊 EXAMPLE - Binary Metric (Conversion):
   Baseline: {p1:.0%}
   MDE: {mde_rel:.0%} relative lift ({p1:.0%} → {p2:.1%})
   Cohen's h: {h:.4f}
   
   ✅ RESULT: Need {n_binary:,} users/group ({n_binary*2:,} total)
""")
    
    # Calculate for continuous
    baseline_rev, std_rev = 175, 80
    mde_abs = 15
    d = mde_abs / std_rev
    n_cont = int(np.ceil(tt_ind_solve_power(effect_size=d, alpha=0.05, power=0.80)))
    
    print(f"""📊 EXAMPLE - Continuous Metric (Revenue):
   Baseline: ${baseline_rev} (std=${std_rev})
   MDE: ${mde_abs} absolute lift
   Cohen's d: {d:.4f}
   
   ✅ RESULT: Need {n_cont:,} users/group ({n_cont*2:,} total)

💡 INTERVIEW TIP: Always ask:
   - One-sided or two-sided test?
   - What's the business-relevant MDE?
   - How long to reach this sample size?
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1.2 Z-TEST FOR PROPORTIONS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📊 1.2 Z-TEST FOR PROPORTIONS (Binary Metrics)")
    print("─" * 90)
    
    print("""
🎯 USE WHEN: Comparing conversion rates, CTR, or any 0/1 outcome.

📝 THE TEST:
   H₀: p_A = p_B (no difference)
   H₁: p_A ≠ p_B (two-sided)
   
   Z = (p_B - p_A) / SE_pooled
   where SE_pooled = √[p̄(1-p̄)(1/n_A + 1/n_B)]
   
⚠️ KEY INSIGHT: We use POOLED SE for the test (assumes H₀ true),
   but NON-POOLED SE for the confidence interval.
""")
    
    x_a, x_b = int(df_a['converted'].sum()), int(df_b['converted'].sum())
    p_a, p_b = x_a/n_a, x_b/n_b
    
    # Pooled for test
    p_pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    z_stat = (p_b - p_a) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Non-pooled for CI
    se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci = (p_b - p_a - 1.96*se_diff, p_b - p_a + 1.96*se_diff)
    
    # Effect size
    cohens_h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
    h_interp = "Large" if abs(cohens_h) > 0.8 else "Medium" if abs(cohens_h) > 0.5 else "Small" if abs(cohens_h) > 0.2 else "Negligible"
    
    print(f"""📊 RESULTS:
   Control:   {x_a:,} / {n_a:,} = {p_a:.4f} ({p_a:.2%})
   Treatment: {x_b:,} / {n_b:,} = {p_b:.4f} ({p_b:.2%})
   
   Absolute lift: {(p_b - p_a)*100:.2f} percentage points
   Relative lift: {(p_b/p_a - 1)*100:.1f}%
   
   Z-statistic: {z_stat:.4f}
   P-value: {p_value:.2e} {'✅ Significant!' if p_value < 0.05 else '❌ Not significant'}
   95% CI: ({ci[0]*100:.2f}pp, {ci[1]*100:.2f}pp)
   
   Effect Size (Cohen's h): {cohens_h:.4f} → {h_interp}

📖 P-VALUE INTERPRETATION:
   • P-value = {p_value:.2e} means: IF there were no real effect (H₀ true),
     we'd see a difference this large only {p_value*100:.4f}% of the time.
   • P-value is NOT the probability treatment works!
   • P-value is NOT the probability of a false positive on THIS experiment!
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1.3 WELCH'S T-TEST (Continuous)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📊 1.3 WELCH'S T-TEST (Continuous Metrics)")
    print("─" * 90)
    
    print("""
🎯 USE WHEN: Comparing means of continuous outcomes (revenue, time, sessions).

📝 WHY WELCH'S (not Student's)?
   • Does NOT assume equal variances between groups
   • More robust in practice - always use Welch's unless specific reason not to
   • scipy.stats.ttest_ind(equal_var=False) gives Welch's
""")
    
    # Revenue for converters
    df_conv = df[df['converted'] == 1]
    rev_a = df_conv[df_conv['variant'] == 'A']['revenue'].values
    rev_b = df_conv[df_conv['variant'] == 'B']['revenue'].values
    
    t_stat, t_pval = stats.ttest_ind(rev_b, rev_a, equal_var=False)
    diff = rev_b.mean() - rev_a.mean()
    se = np.sqrt(rev_a.var()/len(rev_a) + rev_b.var()/len(rev_b))
    ci_t = (diff - 1.96*se, diff + 1.96*se)
    
    pooled_std = np.sqrt(((len(rev_a)-1)*rev_a.var() + (len(rev_b)-1)*rev_b.var()) / (len(rev_a)+len(rev_b)-2))
    cohens_d = diff / pooled_std
    d_interp = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small" if abs(cohens_d) > 0.2 else "Negligible"
    
    print(f"""📊 RESULTS (Revenue, converters only):
   Control:   n={len(rev_a):,}, mean=${rev_a.mean():.2f}, std=${rev_a.std():.2f}
   Treatment: n={len(rev_b):,}, mean=${rev_b.mean():.2f}, std=${rev_b.std():.2f}
   
   Difference: ${diff:.2f} ({diff/rev_a.mean()*100:.1f}% relative)
   95% CI: (${ci_t[0]:.2f}, ${ci_t[1]:.2f})
   
   T-statistic: {t_stat:.4f}
   P-value: {t_pval:.2e} {'✅ Significant!' if t_pval < 0.05 else '❌ Not significant'}
   
   Effect Size (Cohen's d): {cohens_d:.4f} → {d_interp}
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1.4 SRM CHECK
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("🔍 1.4 SRM (Sample Ratio Mismatch) CHECK")
    print("─" * 90)
    
    print("""
🎯 WHY CHECK SRM?
   If randomization is broken, ALL your results are INVALID.
   Always check BEFORE looking at outcome metrics!

⚠️ COMMON CAUSES:
   • Bot filtering affecting groups differently
   • Page load failures in treatment
   • Browser/device compatibility issues
   • Redirect failures
""")
    
    expected = np.array([0.5, 0.5]) * (n_a + n_b)
    observed = np.array([n_a, n_b])
    chi2 = np.sum((observed - expected)**2 / expected)
    srm_p = 1 - stats.chi2.cdf(chi2, df=1)
    
    print(f"""📊 RESULTS:
   Expected: A={int(expected[0]):,}, B={int(expected[1]):,} (50/50)
   Observed: A={n_a:,}, B={n_b:,} ({n_a/(n_a+n_b):.2%}/{n_b/(n_a+n_b):.2%})
   
   Chi-square: {chi2:.4f}
   P-value: {srm_p:.4f}
   
   {"⚠️ SRM DETECTED! Investigate before trusting results." if srm_p < 0.01 else "✅ No SRM - randomization healthy"}
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 2: INTERMEDIATE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 90)
    print("█ LEVEL 2: INTERMEDIATE (Expected for DS roles)")
    print("═" * 90)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.1 BAYESIAN A/B TESTING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("🎲 2.1 BAYESIAN A/B TESTING")
    print("─" * 90)
    
    print("""
🎯 WHY BAYESIAN?
   • Gives PROBABILITY statements: "P(B > A) = 95%" (intuitive!)
   • No p-hacking concerns with continuous monitoring
   • Can compute expected loss for decision-making

📝 MODEL: Beta-Binomial
   Prior: Beta(1, 1) = Uniform
   Posterior: Beta(1 + successes, 1 + failures)
""")
    
    # Posterior
    post_a = (1 + x_a, 1 + n_a - x_a)
    post_b = (1 + x_b, 1 + n_b - x_b)
    
    np.random.seed(42)
    samples_a = np.random.beta(*post_a, 100000)
    samples_b = np.random.beta(*post_b, 100000)
    prob_b_better = (samples_b > samples_a).mean()
    lift_samples = (samples_b / samples_a - 1) * 100
    
    loss_a = np.maximum(0, samples_b - samples_a).mean()
    loss_b = np.maximum(0, samples_a - samples_b).mean()
    
    print(f"""📊 RESULTS:
   Posterior A: Beta({post_a[0]}, {post_a[1]})
   Posterior B: Beta({post_b[0]}, {post_b[1]})
   
   P(B > A) = {prob_b_better:.4f} ({prob_b_better:.2%})
   
   Expected lift: {lift_samples.mean():.2f}%
   95% Credible Interval: ({np.percentile(lift_samples, 2.5):.2f}%, {np.percentile(lift_samples, 97.5):.2f}%)
   
   Expected loss if choose A: {loss_a:.6f} (in conversion rate)
   Expected loss if choose B: {loss_b:.6f}
   → Recommendation: {'Choose B' if loss_b < loss_a else 'Choose A'}

💡 NOTE: P(B > A) ≈ 100% with large samples and real effects.
   This is CORRECT - there's essentially no posterior mass where A > B.
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.2 MULTIPLE TESTING CORRECTION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("⚖️ 2.2 MULTIPLE TESTING CORRECTION")
    print("─" * 90)
    
    print("""
🎯 THE PROBLEM:
   Testing 5 metrics at α=0.05: P(≥1 FP) = 1 - 0.95⁵ = 22.6%!

📝 SOLUTIONS:
   • Bonferroni: α_adj = α/n (strict, controls FWER)
   • Benjamini-Hochberg: Controls FDR (less strict, more power)
   
🔍 WHEN TO USE:
   • Bonferroni: Safety metrics, confirmatory analysis
   • BH: Exploratory analysis, many metrics
""")
    
    # Calculate ACTUAL p-values from real metrics
    metrics = ['converted', 'sessions_week1', 'pages_viewed', 'retention_7d', 'nps_score']
    pvalues = []
    effects = []
    
    for metric in metrics:
        if df[metric].nunique() <= 2:  # Binary
            _, pval = proportions_ztest(
                [df_a[metric].sum(), df_b[metric].sum()],
                [len(df_a), len(df_b)]
            )
            effect = df_b[metric].mean() - df_a[metric].mean()
        else:  # Continuous
            _, pval = stats.ttest_ind(df_b[metric].dropna(), df_a[metric].dropna(), equal_var=False)
            effect = df_b[metric].mean() - df_a[metric].mean()
        pvalues.append(pval)
        effects.append(effect)
    
    _, bonf_p, _, _ = multipletests(pvalues, method='bonferroni')
    _, bh_p, _, _ = multipletests(pvalues, method='fdr_bh')
    
    fp_risk = 1 - (1 - 0.05) ** len(metrics)
    
    print(f"""📊 RESULTS (Using ACTUAL experiment metrics):
   False positive risk (no correction): {fp_risk:.1%}
   
   {'Metric':<15} {'Effect':>10} {'Raw p':>12} {'Bonf p':>12} {'BH p':>12} {'Raw':>5} {'Bonf':>5} {'BH':>5}
   {'-'*82}""")
    
    for i, m in enumerate(metrics):
        raw_sig = '✅' if pvalues[i] < 0.05 else '❌'
        bonf_sig = '✅' if bonf_p[i] < 0.05 else '❌'
        bh_sig = '✅' if bh_p[i] < 0.05 else '❌'
        eff_str = f"{effects[i]:.4f}" if abs(effects[i]) < 1 else f"{effects[i]:.2f}"
        print(f"   {m:<15} {eff_str:>10} {pvalues[i]:>12.6f} {bonf_p[i]:>12.6f} {bh_p[i]:>12.6f} {raw_sig:>5} {bonf_sig:>5} {bh_sig:>5}")
    
    print(f"""
   Summary: Raw={sum(p < 0.05 for p in pvalues)}/5, Bonferroni={sum(p < 0.05 for p in bonf_p)}/5, BH={sum(p < 0.05 for p in bh_p)}/5
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.3 CUPED
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📉 2.3 CUPED (Variance Reduction)")
    print("─" * 90)
    
    print("""
🎯 THE IDEA:
   Pre-experiment data is correlated with outcome.
   By "adjusting out" this correlation, we reduce variance!

📝 FORMULA:
   Y_adj = Y - θ(X - X̄)
   where θ = Cov(Y,X) / Var(X)
   
   Variance reduction ≈ ρ² (correlation squared)
   If ρ = 0.5, we reduce variance by 25%!

🔧 PRACTICAL BENEFIT:
   20% variance reduction → ~20% fewer users needed
""")
    
    # CUPED on revenue
    y = df_conv['revenue'].values
    x = df_conv['pre_revenue'].values
    treatment = (df_conv['variant'] == 'B').astype(int).values
    
    corr = np.corrcoef(y, x)[0, 1]
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    
    # Raw
    y_c, y_t = y[treatment == 0], y[treatment == 1]
    effect_raw = y_t.mean() - y_c.mean()
    se_raw = np.sqrt(y_c.var()/len(y_c) + y_t.var()/len(y_t))
    ci_raw = (effect_raw - 1.96*se_raw, effect_raw + 1.96*se_raw)
    
    # CUPED
    y_c_adj, y_t_adj = y_adj[treatment == 0], y_adj[treatment == 1]
    effect_adj = y_t_adj.mean() - y_c_adj.mean()
    se_adj = np.sqrt(y_c_adj.var()/len(y_c_adj) + y_t_adj.var()/len(y_t_adj))
    ci_adj = (effect_adj - 1.96*se_adj, effect_adj + 1.96*se_adj)
    
    var_reduction = 1 - y_adj.var() / y.var()
    
    print(f"""📊 RESULTS (Revenue, converters):
   Correlation (pre_revenue ↔ revenue): {corr:.4f}
   θ (adjustment coefficient): {theta:.4f}
   
   Theoretical variance reduction (ρ²): {corr**2 * 100:.1f}%
   Actual variance reduction: {var_reduction * 100:.1f}%
   
   ┌─────────────┬──────────────────┬──────────────────┐
   │             │ Raw              │ CUPED            │
   ├─────────────┼──────────────────┼──────────────────┤
   │ Effect      │ ${effect_raw:>13.2f}   │ ${effect_adj:>13.2f}   │
   │ SE          │ ${se_raw:>13.2f}   │ ${se_adj:>13.2f}   │
   │ CI Width    │ ${ci_raw[1]-ci_raw[0]:>13.2f}   │ ${ci_adj[1]-ci_adj[0]:>13.2f}   │
   └─────────────┴──────────────────┴──────────────────┘
   
   SE reduction: {(1 - se_adj/se_raw)*100:.1f}%
   → We could run with ~{(1 - se_adj/se_raw)*100:.0f}% fewer users for same power!
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.4 BOOTSTRAP CI
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("🔄 2.4 BOOTSTRAP CONFIDENCE INTERVALS")
    print("─" * 90)
    
    print("""
🎯 WHY BOOTSTRAP?
   • Non-parametric: no normality assumption
   • Great for skewed data (revenue!)
   • Works for any statistic (medians, ratios, percentiles)

📝 HOW IT WORKS:
   1. Resample with replacement
   2. Compute statistic difference
   3. Repeat 10,000 times
   4. Use percentiles for CI
""")
    
    boot_diffs = []
    for _ in range(10000):
        boot_a = np.random.choice(rev_a, len(rev_a), replace=True)
        boot_b = np.random.choice(rev_b, len(rev_b), replace=True)
        boot_diffs.append(boot_b.mean() - boot_a.mean())
    boot_diffs = np.array(boot_diffs)
    
    boot_ci = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))
    
    print(f"""📊 RESULTS (Revenue, 10,000 bootstrap samples):
   Point estimate: ${diff:.2f}
   Bootstrap SE: ${boot_diffs.std():.2f}
   95% CI (percentile): (${boot_ci[0]:.2f}, ${boot_ci[1]:.2f})
   
   Significant: {'✅ Yes' if not (boot_ci[0] <= 0 <= boot_ci[1]) else '❌ No'}
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.5 MANN-WHITNEY U
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📊 2.5 MANN-WHITNEY U TEST (Non-parametric)")
    print("─" * 90)
    
    print("""
🎯 WHY USE IT?
   • Non-parametric: no normality assumption
   • Compares distributions, not just means
   • Robust to outliers

🔍 WHEN TO USE:
   • Heavily skewed data
   • Ordinal data
   • When t-test assumptions violated
""")
    
    u_stat, u_pval = stats.mannwhitneyu(rev_b, rev_a, alternative='two-sided')
    r = 1 - (2 * u_stat) / (len(rev_a) * len(rev_b))
    r_interp = "Large" if abs(r) > 0.5 else "Medium" if abs(r) > 0.3 else "Small" if abs(r) > 0.1 else "Negligible"
    
    print(f"""📊 RESULTS (Revenue):
   Control median: ${np.median(rev_a):.2f}
   Treatment median: ${np.median(rev_b):.2f}
   
   U-statistic: {u_stat:,.0f}
   P-value: {u_pval:.6f} {'✅ Significant' if u_pval < 0.05 else '❌ Not significant'}
   
   Effect Size (rank-biserial r): {r:.4f} → {r_interp}
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.6 ITT vs PER-PROTOCOL
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📋 2.6 ITT vs PER-PROTOCOL ANALYSIS")
    print("─" * 90)
    
    print("""
🎯 THE DISTINCTION:
   • ITT: Analyze EVERYONE as randomized (gold standard)
   • Per-Protocol: Only those who SAW the treatment

📚 WHY BOTH?
   • ITT preserves randomization, gives unbiased estimate
   • PP shows effect on those actually exposed
   • Big gap = implementation problems
""")
    
    # ITT
    effect_itt = df_b['converted'].mean() - df_a['converted'].mean()
    
    # Per-Protocol
    df_a_pp = df[(df['variant'] == 'A') & (df['triggered'] == 1)]
    df_b_pp = df[(df['variant'] == 'B') & (df['triggered'] == 1)]
    effect_pp = df_b_pp['converted'].mean() - df_a_pp['converted'].mean()
    
    trigger_a = df_a['triggered'].mean()
    trigger_b = df_b['triggered'].mean()
    
    print(f"""📊 RESULTS:
   Trigger rates: A = {trigger_a:.1%}, B = {trigger_b:.1%}
   → {(1-trigger_b)*100:.0f}% of B users didn't see treatment!
   
   ┌─────────────────┬─────────────────┬─────────────────┐
   │                 │ ITT             │ Per-Protocol    │
   ├─────────────────┼─────────────────┼─────────────────┤
   │ N (Control)     │ {len(df_a):>15,} │ {len(df_a_pp):>15,} │
   │ N (Treatment)   │ {len(df_b):>15,} │ {len(df_b_pp):>15,} │
   │ Effect          │ {effect_itt:>15.4f} │ {effect_pp:>15.4f} │
   └─────────────────┴─────────────────┴─────────────────┘
   
   PP effect is {(effect_pp/effect_itt - 1)*100:.0f}% higher than ITT
   → Makes sense: PP measures effect on those who actually saw treatment

💡 RECOMMENDATION: Report ITT as primary (unbiased), PP as secondary
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 3: ADVANCED
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 90)
    print("█ LEVEL 3: ADVANCED (Differentiating for Senior roles)")
    print("═" * 90)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.1 CUPAC
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("🚀 3.1 CUPAC (ML-Enhanced Variance Reduction)")
    print("─" * 90)
    
    print("""
🎯 IMPROVEMENT OVER CUPED:
   • Uses ML to predict from MULTIPLE features
   • Captures non-linear relationships
   • Often achieves 25-40%+ variance reduction

📝 METHOD (DoorDash, 2020):
   1. Train ML model to predict Y from pre-experiment features
   2. Use cross-validated predictions (avoid overfitting)
   3. Apply CUPED adjustment with predictions as covariate
""")
    
    # CUPAC
    X = df_conv[['pre_revenue', 'pre_engagement', 'pre_sessions']].values
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, min_samples_leaf=100, random_state=42)
    cv_preds = cross_val_predict(model, X, y, cv=5)
    
    r2 = 1 - np.sum((y - cv_preds)**2) / np.sum((y - y.mean())**2)
    pred_corr = np.corrcoef(y, cv_preds)[0, 1]
    
    theta_cupac = np.cov(y, cv_preds)[0, 1] / cv_preds.var()
    y_cupac = y - theta_cupac * (cv_preds - cv_preds.mean())
    
    y_c_cupac, y_t_cupac = y_cupac[treatment == 0], y_cupac[treatment == 1]
    effect_cupac = y_t_cupac.mean() - y_c_cupac.mean()
    se_cupac = np.sqrt(y_c_cupac.var()/len(y_c_cupac) + y_t_cupac.var()/len(y_t_cupac))
    var_reduction_cupac = 1 - y_cupac.var() / y.var()
    
    print(f"""📊 RESULTS:
   Model R²: {r2:.4f}
   Prediction correlation: {pred_corr:.4f}
   
   ┌─────────────────┬──────────────┬──────────────┬──────────────┐
   │                 │ Raw          │ CUPED        │ CUPAC        │
   ├─────────────────┼──────────────┼──────────────┼──────────────┤
   │ Var. Reduction  │ --           │ {var_reduction*100:>10.1f}%  │ {var_reduction_cupac*100:>10.1f}%  │
   │ SE Reduction    │ --           │ {(1-se_adj/se_raw)*100:>10.1f}%  │ {(1-se_cupac/se_raw)*100:>10.1f}%  │
   │ SE              │ ${se_raw:>10.2f}  │ ${se_adj:>10.2f}  │ ${se_cupac:>10.2f}  │
   └─────────────────┴──────────────┴──────────────┴──────────────┘
   
   CUPAC achieves {var_reduction_cupac/var_reduction:.1f}x the variance reduction of CUPED!
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.2 SEQUENTIAL TESTING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("⏱️ 3.2 SEQUENTIAL TESTING")
    print("─" * 90)
    
    print("""
🎯 THE PROBLEM:
   Looking at results multiple times inflates Type I error.
   5 looks at α=0.05 → actual error ≈ 14%!

📝 SOLUTION:
   Use adjusted boundaries at each look.
   
   Methods:
   • O'Brien-Fleming: Conservative early, full power at end
   • Pocock: Constant boundaries, easier to stop early
""")
    
    # Use actual z-statistic from experiment
    actual_z = z_stat  # From our z-test earlier
    current_look = 3
    total_looks = 5
    info_frac = current_look / total_looks
    
    # O'Brien-Fleming boundary
    boundary = stats.norm.ppf(1 - 0.05/2) / np.sqrt(info_frac)
    can_stop = abs(actual_z) > boundary
    
    print(f"""📊 RESULTS (Using actual experiment z-statistic):
   Actual z-statistic: {actual_z:.4f}
   Current look: {current_look} of {total_looks}
   Information fraction: {info_frac:.0%}
   
   O'Brien-Fleming boundary: {boundary:.4f}
   |z| > boundary? {abs(actual_z):.4f} > {boundary:.4f} → {'Yes' if can_stop else 'No'}
   
   Decision: {"✅ STOP - Significant at interim" if can_stop else f"🔄 CONTINUE - Wait for look {current_look + 1}"}

💡 WHY THIS MATTERS:
   Without correction, checking 5 times gives ~14% false positive rate.
   Sequential testing maintains valid α = 5%.
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.3 X-LEARNER (HTE)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("🎯 3.3 X-LEARNER (Heterogeneous Treatment Effects)")
    print("─" * 90)
    
    print("""
🎯 WHY X-LEARNER?
   • Estimates individual-level treatment effects (CATE)
   • Better than stratified analysis for complex heterogeneity
   • Enables personalization decisions

📝 THE 4 STEPS:
   1. Fit μ₀(x) on control, μ₁(x) on treatment
   2. Impute effects: τ₁ = Y₁ - μ₀(X₁), τ₀ = μ₁(X₀) - Y₀
   3. Fit τ models on imputed effects
   4. Combine with propensity weighting
""")
    
    # X-Learner
    T = (df['variant'] == 'B').astype(int).values
    X_full = df[['pre_engagement', 'pre_sessions']].values
    Y_full = df['converted'].values
    
    control_mask = T == 0
    treated_mask = T == 1
    
    mu_0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    mu_1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    mu_0.fit(X_full[control_mask], Y_full[control_mask])
    mu_1.fit(X_full[treated_mask], Y_full[treated_mask])
    
    tau_1 = Y_full[treated_mask] - mu_0.predict(X_full[treated_mask])
    tau_0 = mu_1.predict(X_full[control_mask]) - Y_full[control_mask]
    
    tau_model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    tau_model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    tau_model_0.fit(X_full[control_mask], tau_0)
    tau_model_1.fit(X_full[treated_mask], tau_1)
    
    g = T.mean()
    cate = g * tau_model_0.predict(X_full) + (1 - g) * tau_model_1.predict(X_full)
    ate = cate.mean()
    
    print(f"""📊 RESULTS:
   Average Treatment Effect (ATE): {ate:.4f} ({ate*100:.2f}pp)
   ATT (effect on treated): {cate[treated_mask].mean():.4f}
   ATC (effect on control): {cate[control_mask].mean():.4f}
   
   CATE Distribution:
   • Std dev: {cate.std():.4f}
   • Range: ({cate.min():.4f}, {cate.max():.4f})
   • p10: {np.percentile(cate, 10):.4f}, p50: {np.percentile(cate, 50):.4f}, p90: {np.percentile(cate, 90):.4f}
   
   Heterogeneity detected: {'Yes' if cate.std() > abs(ate) * 0.5 else 'No'}
   
💡 INTERPRETATION:
   High heterogeneity means treatment works differently for different users.
   → Consider personalization or targeting high-CATE segments.
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.4 QUANTILE TREATMENT EFFECTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📊 3.4 QUANTILE TREATMENT EFFECTS")
    print("─" * 90)
    
    print("""
🎯 WHY QUANTILES?
   • Mean can HIDE heterogeneity
   • Treatment might affect distribution differently
   • Example: Helps low spenders but not high spenders
""")
    
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    print(f"\n   {'Quantile':<10} {'Control':>12} {'Treatment':>12} {'Effect':>12}")
    print(f"   {'-'*50}")
    
    for q in quantiles:
        q_a = np.percentile(rev_a, q*100)
        q_b = np.percentile(rev_b, q*100)
        print(f"   p{int(q*100):<8} ${q_a:>10.2f}  ${q_b:>10.2f}  ${q_b-q_a:>10.2f}")
    
    effects_q = [np.percentile(rev_b, q*100) - np.percentile(rev_a, q*100) for q in quantiles]
    print(f"""
   Effect range: ${min(effects_q):.2f} to ${max(effects_q):.2f}
   Heterogeneous: {'Yes' if max(effects_q) - min(effects_q) > np.abs(np.mean(effects_q)) * 0.5 else 'No'}
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.5 DELTA METHOD
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("📊 3.5 DELTA METHOD (Ratio Metrics)")
    print("─" * 90)
    
    print("""
🎯 RATIO METRICS:
   • Revenue per user = Total Revenue / Users
   • CTR = Clicks / Impressions
   • Pages per session = Pages / Sessions

📝 THE PROBLEM:
   Ratio of means ≠ Mean of ratios
   Need proper variance estimation!
""")
    
    # Revenue per user (including non-converters)
    num_a = df_a['revenue'].values
    denom_a = np.ones(len(df_a))  # Per user
    num_b = df_b['revenue'].values
    denom_b = np.ones(len(df_b))
    
    ratio_a = num_a.sum() / len(df_a)
    ratio_b = num_b.sum() / len(df_b)
    
    def delta_variance(num, denom):
        n = len(num)
        mu_n, mu_d = num.mean(), denom.mean()
        var_n = num.var(ddof=1)
        var_d = denom.var(ddof=1)
        cov_nd = np.cov(num, denom, ddof=1)[0, 1]
        var_ratio = (1/mu_d**2) * var_n - (2*mu_n/mu_d**3) * cov_nd + (mu_n**2/mu_d**4) * var_d
        return var_ratio / n
    
    var_a = delta_variance(num_a, denom_a)
    var_b = delta_variance(num_b, denom_b)
    se_ratio = np.sqrt(var_a + var_b)
    diff_ratio = ratio_b - ratio_a
    ci_ratio = (diff_ratio - 1.96*se_ratio, diff_ratio + 1.96*se_ratio)
    z_ratio = diff_ratio / se_ratio
    p_ratio = 2 * (1 - stats.norm.cdf(abs(z_ratio)))
    
    print(f"""📊 RESULTS (Revenue per User):
   Control: ${ratio_a:.2f}/user
   Treatment: ${ratio_b:.2f}/user
   
   Difference: ${diff_ratio:.2f} ({diff_ratio/ratio_a*100:.1f}% relative)
   SE (Delta method): ${se_ratio:.2f}
   95% CI: (${ci_ratio[0]:.2f}, ${ci_ratio[1]:.2f})
   P-value: {p_ratio:.6f}
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.6 WINSORIZATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("✂️ 3.6 WINSORIZATION (Outlier Handling)")
    print("─" * 90)
    
    print("""
🎯 WHY WINSORIZE?
   • Outliers can dominate revenue metrics
   • More robust than trimming (keeps sample size)
   • Common for revenue, time-on-site
""")
    
    rev_a_wins = mstats.winsorize(rev_a, limits=[0.01, 0.01])
    rev_b_wins = mstats.winsorize(rev_b, limits=[0.01, 0.01])
    
    effect_wins = float(rev_b_wins.mean() - rev_a_wins.mean())
    se_wins = np.sqrt(float(rev_a_wins.var())/len(rev_a) + float(rev_b_wins.var())/len(rev_b))
    
    print(f"""📊 RESULTS (1%/99% Winsorization):
   ┌─────────────┬──────────────┬──────────────┐
   │             │ Raw          │ Winsorized   │
   ├─────────────┼──────────────┼──────────────┤
   │ Mean A      │ ${rev_a.mean():>10.2f}  │ ${float(rev_a_wins.mean()):>10.2f}  │
   │ Mean B      │ ${rev_b.mean():>10.2f}  │ ${float(rev_b_wins.mean()):>10.2f}  │
   │ Effect      │ ${diff:>10.2f}  │ ${effect_wins:>10.2f}  │
   │ SE          │ ${se:>10.2f}  │ ${se_wins:>10.2f}  │
   └─────────────┴──────────────┴──────────────┘
   
   SE reduction: {(1 - se_wins/se)*100:.1f}%
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 4: PRODUCTION
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 90)
    print("█ LEVEL 4: PRODUCTION (Real-world deployment)")
    print("═" * 90)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.1 DECISION FRAMEWORK
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("✅ 4.1 DECISION FRAMEWORK")
    print("─" * 90)
    
    print("""
📝 DECISION MATRIX:
   ┌─────────────────┬──────────────┬─────────────────────────────┐
   │ Significant?    │ Positive?    │ Decision                    │
   ├─────────────────┼──────────────┼─────────────────────────────┤
   │ ✅ Yes          │ ✅ Yes       │ ✅ SHIP (if guardrails OK)  │
   │ ✅ Yes          │ ❌ No        │ ❌ DO NOT SHIP              │
   │ ❌ No           │ ✅ Yes       │ 🔄 CONTINUE or ABANDON      │
   │ ❌ No           │ ❌ No        │ ❌ ABANDON                  │
   └─────────────────┴──────────────┴─────────────────────────────┘
""")
    
    is_sig = p_value < 0.05
    is_positive = (p_b - p_a) > 0
    
    print(f"""📊 THIS EXPERIMENT:
   Significant: {'✅ Yes' if is_sig else '❌ No'} (p = {p_value:.2e})
   Positive: {'✅ Yes' if is_positive else '❌ No'} (lift = {(p_b - p_a)*100:.2f}pp)
   
   → DECISION: {'✅ SHIP' if is_sig and is_positive else '❌ DO NOT SHIP'}
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.2 BUSINESS IMPACT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("💰 4.2 BUSINESS IMPACT TRANSLATION")
    print("─" * 90)
    
    effect_conv = p_b - p_a
    annual_users = 10_000_000
    ltv = 150
    additional_conversions = effect_conv * annual_users
    revenue_impact = additional_conversions * ltv
    
    print(f"""📊 AT SCALE:
   Annual users: {annual_users:,}
   Baseline conversion: {p_a:.2%}
   Treatment conversion: {p_b:.2%}
   
   Additional conversions: {additional_conversions:,.0f}/year
   Revenue impact: ${revenue_impact:,.0f}/year
   
   💼 EXECUTIVE SUMMARY:
   "The new onboarding flow is expected to generate {additional_conversions:,.0f} additional
   conversions per year, worth ${revenue_impact/1e6:.1f}M in annual revenue."
""")
    
    print("\n" + "═" * 90)
    print("█ COMPLETE CURRICULUM FINISHED!")
    print("═" * 90)
    
    print(df)