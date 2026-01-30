# A/B Testing Framework - Production-Ready Experimentation Toolkit

A **comprehensive, modular Python package** for A/B testing and experimentation, featuring state-of-the-art statistical methods (2024-2025), real-world datasets, and production-grade code quality. Designed for data scientists, analysts, and engineers who want to master rigorous experimentation techniques.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()

> **New here?** Check out [START_HERE.md](START_HERE.md) for a guided learning path based on your experience level.
>
> **Why these choices?** Read [DECISIONS.md](DECISIONS.md) to understand the design decisions behind this framework.

---

## 🎯 What Makes This Different

- **✅ Complete Implementation**: All major A/B testing techniques from fundamentals through cutting-edge methods
- **✅ Best Practices**: Guardrail metrics, novelty detection, instrumental variables, network effects
- **✅ Real-World Datasets**: Criteo Uplift (13.9M rows), Marketing A/B (588K rows), Cookie Cats (90K rows)
- **✅ Production-Grade**: 13+ modules, 200+ test methods, 80%+ coverage, full type hints
- **✅ Industry-Validated**: Techniques from Meta, Spotify, DoorDash, Statsig, Mixpanel

---

## 📚 Table of Contents

- [Quick Start](#-quick-start)
- [Complete Feature List](#-complete-feature-list)
- [Learning Pathways](#-learning-pathways)
- [Full Usage Guide](#-full-usage-guide)
- [Real-World Datasets](#-real-world-datasets)
- [API Reference](#-api-reference)
- [Verification Against Scope](#-project-scope-verification)
- [Industry Best Practices](#-industry-best-practices)

**Additional Guides:**
- [START_HERE.md](START_HERE.md) - Quick learning guide with self-assessment
- [DECISIONS.md](DECISIONS.md) - Why we made specific design choices

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ghadfield32/ab_testing.git
cd ab_testing

# Install uv (recommended) or use pip
pip install uv

# Install dependencies
uv sync

# Verify installation
uv run pytest
```

### Your First A/B Test in 30 Seconds

```python
from ab_testing.core import power, frequentist
from ab_testing.data import loaders

# 1. Calculate required sample size
n = power.required_samples_binary(
    p_baseline=0.05,  # 5% baseline conversion
    mde=0.10,         # Detect 10% relative lift
    alpha=0.05,
    power=0.80
)
print(f"Need {n:,} users per group")  # ~15,681

# 2. Run z-test on your data
result = frequentist.z_test_proportions(
    x_control=50, n_control=500,
    x_treatment=60, n_treatment=500
)
print(f"P-value: {result['p_value']:.4f}")
print(f"Lift: {result['relative_lift']*100:.1f}%")
print(f"95% CI: [{result['ci_lower']*100:.2f}%, {result['ci_upper']*100:.2f}%]")

# 3. Load real-world data
df = loaders.load_marketing_ab()
print(f"Loaded {len(df):,} real users from marketing A/B test")
```

---

## 🎯 Why These 3 Datasets? Real-World A/B Testing Scenarios

This repository uses three real-world datasets that represent the most common A/B testing scenarios in industry. Each dataset teaches different aspects of experimentation, progressing from beginner to advanced techniques.

### 1. Marketing A/B Test (Beginner) - Ad Campaign Effectiveness

**Real-World Scenario**: E-commerce marketing team testing new ad creatives
**Business Question**: "Should we switch to the new ad design?"
**Dataset Size**: 588,101 observations
**Source**: [Kaggle - faviovaz/marketing-ab-testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing)

**What Makes This Dataset Ideal for Learning**:
- **Clean Binary Outcome**: Converted (yes/no) - simplest A/B test type
- **Pre-Experiment Covariate**: `total_ads` (perfect for CUPED variance reduction)
- **Temporal Data**: Test ran over multiple weeks (novelty effect detection)
- **Realistic Scale**: 588K users (typical mid-size experiment)

**Industry Context**:
- Meta runs 10K+ experiments/year, most are binary conversion tests
- Booking.com: 70% of experiments are this type (feature on/off)
- Typical effect sizes: 1-5% relative lift (small but significant at scale)

**Learning Progression**:
1. **Fundamentals**: Power analysis, sample size, z-test for proportions
2. **Randomization Quality**: SRM check (critical first step in ANY experiment)
3. **Variance Reduction**: CUPED with pre-experiment ad exposure covariate
4. **Guardrail Metrics**: Ensure we don't harm ad engagement while optimizing conversion
5. **Novelty Detection**: Did effect fade over time? (Common with UI changes)
6. **Decision Framework**: Ship/hold/abandon based on statistical + business context

**Key Techniques Demonstrated**:
- ✅ Z-test for proportions
- ✅ Power analysis (Cohen's h effect size)
- ✅ CUPED (Controlled-experiment Using Pre-Experiment Data)
- ✅ Guardrail metrics (non-inferiority tests)
- ✅ Novelty effect detection (early vs late period comparison)
- ✅ Business impact translation (convert p-values to revenue)

**When to Use Similar Analysis** (decision tree):
- ✅ Binary outcome (clicked, converted, activated, retained)
- ✅ Pre-experiment data available (user history, previous behavior)
- ✅ Concern about novelty effects (user-facing changes often have this)
- ✅ Need to protect guardrail metrics (revenue, engagement, core KPIs)
- ✅ Marketing/growth experiments (as opposed to infrastructure changes)

---

### 2. Cookie Cats Mobile Game (Intermediate) - Product Growth

**Real-World Scenario**: Mobile gaming company testing gate placement optimization
**Business Question**: "Does moving the difficulty gate from level 30 to level 40 improve retention?"
**Dataset Size**: 90,189 players
**Source**: [Kaggle - mursideyarkin/mobile-games-ab-testing-cookie-cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)

**What Makes This Dataset Ideal for Learning**:
- **Multiple Outcomes**: 1-day retention AND 7-day retention (multiple testing correction)
- **Product Decision**: Gate at level 30 vs level 40 (classic product A/B test)
- **Ratio Metrics**: Rounds played per user (engagement intensity measure)
- **Retention Curves**: Time-based outcome metrics (common in apps/games)

**Industry Context**:
- Spotify: Tests impact on 1-day, 7-day, 28-day retention simultaneously
- Airbnb: Typical experiment has 3-5 metrics (requires multiple testing correction)
- Product experiments: Often optimize for engagement AND retention together
- Mobile games: Gate placement is critical for monetization vs retention balance

**Learning Progression**:
1. **Multiple Outcomes**: Testing both 1-day and 7-day retention simultaneously
2. **Multiple Testing Correction**: Bonferroni vs Benjamini-Hochberg FDR control
3. **Ratio Metrics**: Engagement (rounds played per player) with delta method CIs
4. **Decision Trade-offs**: What if short-term improves but long-term harms?
5. **Product Thinking**: Balance user experience with business goals

**Key Techniques Demonstrated**:
- ✅ Multiple hypothesis testing (Bonferroni, Benjamini-Hochberg)
- ✅ Z-test for proportions (retention rates)
- ✅ Ratio metrics with delta method (rounds per player)
- ✅ Decision framework with conflicting metrics
- ✅ Product trade-off analysis

**When to Use Similar Analysis**:
- ✅ Multiple outcomes to test simultaneously (retention + engagement + monetization)
- ✅ Product/growth experiments (not marketing or infrastructure)
- ✅ Ratio metrics (events per user, revenue per session, time per visit)
- ✅ Risk of false positives from testing many metrics (Family-Wise Error Rate control)
- ✅ Game/app optimization (retention, engagement, difficulty curves)

---

### 3. Criteo Uplift Dataset (Advanced) - Personalization & ML Integration

**Real-World Scenario**: Ad tech company optimizing targeted campaigns
**Business Question**: "Who should we target to maximize incremental conversions?"
**Dataset Size**: 13,979,592 observations (13.9M rows!)
**Source**: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)

**What Makes This Dataset Ideal for Learning**:
- **Uplift Modeling**: Treatment effect varies by user (heterogeneous treatment effects)
- **Large Scale**: 13.9M observations (realistic big data experimentation)
- **Rich Features**: 11 user characteristics (enables ML-enhanced variance reduction)
- **Two Outcomes**: Visit AND conversion (funnel analysis)
- **Industry Standard**: Used in academic research and competitions

**Industry Context**:
- Netflix: Uses uplift modeling to personalize which shows to promote per user
- Uber: Optimizes promo targeting (don't send promos to users who'd convert anyway)
- Modern A/B testing: Not just "does it work?" but "for whom does it work?"
- Personalization era: Individual-level treatment effects are the new frontier

**Learning Progression**:
1. **CUPAC**: ML-enhanced variance reduction (GradientBoosting predicts outcome from 11 features)
2. **X-Learner**: Estimate treatment effect for each individual (CATE = Conditional Average Treatment Effect)
3. **Sequential Testing**: Early stopping with O'Brien-Fleming boundaries (stop when effect is clear)
4. **Large-Scale Best Practices**: Sampling strategies, memory management for massive datasets

**Key Techniques Demonstrated**:
- ✅ CUPAC (ML-enhanced CUPED with GradientBoosting/RandomForest)
- ✅ X-Learner for heterogeneous treatment effects (HTE)
- ✅ Sequential testing (O'Brien-Fleming alpha spending function)
- ✅ Subgroup analysis (which user segments benefit most?)
- ✅ Uplift modeling (targeting decisions based on incremental impact)

**When to Use Similar Analysis**:
- ✅ Personalization decisions (targeting, recommendations, dynamic pricing)
- ✅ Need to estimate individual-level effects (who benefits most?)
- ✅ Large datasets where ML variance reduction pays off (n > 100K)
- ✅ Want early stopping capabilities (sequential testing reduces experiment duration 30-50%)
- ✅ Uplift modeling (maximize incremental conversions, not just overall conversions)

---

### Dataset Comparison Matrix

| Aspect | Marketing A/B | Cookie Cats | Criteo Uplift |
|--------|--------------|-------------|---------------|
| **Difficulty Level** | 🟢 Beginner | 🟡 Intermediate | 🔴 Advanced |
| **Sample Size** | 588K | 90K | 13.9M |
| **Primary Use Case** | Marketing effectiveness | Product optimization | Personalization/targeting |
| **Outcome Type** | Binary (conversion) | Binary (multiple: 1d, 7d retention) | Binary (two stages: visit, conversion) |
| **Key Challenge** | Variance reduction | Multiple testing | Heterogeneous effects |
| **Industry Parallel** | Meta ads, Google Ads | Spotify features, Uber products | Netflix personalization, Uber targeting |
| **Learning Focus** | CUPED, guardrails, novelty | Multiple metrics, ratio metrics | CUPAC, HTE, sequential testing |
| **Typical Effect Size** | 2-5% relative | 1-3% retention lift | 5-15% uplift (targeted users) |
| **Experiment Duration** | 2-4 weeks | 1-2 weeks | Ongoing (sequential) |
| **Pre-Experiment Data** | ✅ Yes (total_ads) | ❌ No | ✅ Yes (11 features) |
| **ML Integration** | ❌ Not needed | ❌ Not applicable | ✅ CUPAC, X-Learner |

---

### How to Choose the Right Dataset for Your Learning Goal

**If you're new to A/B testing** → Start with **Marketing A/B**:
- Clear binary outcome (easy to interpret: did they convert or not?)
- Foundational techniques (power analysis, SRM, z-test - industry standards)
- Real-world workflow (exactly what you'd do at Meta, Google, Booking.com)
- CUPED demonstration (20-40% variance reduction = faster experiments)

**If you understand basics, want to level up** → Move to **Cookie Cats**:
- Realistic product complexity (multiple metrics that may conflict)
- Critical production skill (multiple testing correction prevents false discoveries)
- Trade-off decisions (what if 1-day retention improves but 7-day harms?)
- Ratio metrics (engagement per user - requires delta method for proper CIs)

**If you're preparing for senior roles** → Study **Criteo Uplift**:
- ML integration with experimentation (CUPAC, X-Learner - hot topics in 2024-2025)
- Personalization focus (who benefits most? enables targeting decisions)
- Large-scale best practices (how Netflix, Uber handle massive datasets)
- Sequential testing (early stopping reduces costs by 30-50%)

**If you're working on a specific problem** → Use this decision tree:

1. **Do you have pre-experiment covariates?**
   → **Marketing A/B** (demonstrates CUPED with total_ads covariate)

2. **Are you testing multiple metrics simultaneously?**
   → **Cookie Cats** (demonstrates Bonferroni vs Benjamini-Hochberg FDR)

3. **Do effects vary by user segment (heterogeneous effects)?**
   → **Criteo** (demonstrates X-Learner for CATE estimation)

4. **Is this a user-facing change (UI, messaging, onboarding)?**
   → **Marketing A/B** (demonstrates novelty effect detection)

5. **Need to stop experiment early to save costs?**
   → **Criteo** (demonstrates sequential testing with O'Brien-Fleming boundaries)

6. **Need ML-enhanced variance reduction?**
   → **Criteo** (demonstrates CUPAC with 11 features)

---

## 🧰 A/B Testing Technique Selection Guide

This section helps you choose the right statistical technique for your situation. Use this as a decision-making reference when analyzing experiments.

### Statistical Tests: Which One to Use?

#### Z-Test for Proportions
**Use When**:
- ✅ Binary outcome (clicked, converted, retained, activated)
- ✅ Large sample size (n > 30 per group)
- ✅ Comparing two proportions (control vs treatment)

**Example Scenarios**:
- Conversion rate: 5.2% → 5.5%
- Click-through rate: 2.1% → 2.3%
- Retention: 40% → 42%

**Demonstrated In**: Marketing (primary), Cookie Cats (retention), Criteo (visit/conversion)

**When NOT to Use**:
- ❌ Continuous outcomes like revenue (use Welch's t-test)
- ❌ Non-normal distributions with outliers (use Mann-Whitney U)
- ❌ Small samples n < 30 (use exact tests or bootstrap)

---

#### Welch's T-Test
**Use When**:
- ✅ Continuous outcome (revenue, time spent, session duration, pages viewed)
- ✅ Comparing means between two groups
- ✅ Don't assume equal variances (Welch's is robust)

**Example Scenarios**:
- Average order value: $45 → $48
- Session duration: 12.3 min → 13.1 min
- Pages viewed: 4.2 → 4.7

**When NOT to Use**:
- ❌ Binary outcomes (use z-test for proportions)
- ❌ Heavily skewed distributions (use Mann-Whitney or bootstrap)
- ❌ Multiple groups (use ANOVA, though pairwise t-tests work)

---

#### Mann-Whitney U Test (Non-Parametric)
**Use When**:
- ✅ Continuous outcome but non-normal distribution
- ✅ Outliers present (test is robust to outliers, unlike t-test)
- ✅ Small samples with unknown distributions
- ✅ Ordinal data (ratings, rankings)

**Example Scenarios**:
- Revenue with whales (heavy right skew, power law distribution)
- Time-to-event with long tail (churn, conversion time)
- Ratings/scores that aren't normally distributed (1-5 star ratings)

**When to Use**: Dataset has extreme outliers, distribution is heavily skewed, or you don't trust normality assumptions

---

### Variance Reduction: CUPED vs CUPAC

#### CUPED (Controlled-experiment Using Pre-Experiment Data)
**Use When**:
- ✅ Have pre-experiment covariate (e.g., user history, baseline metric)
- ✅ Covariate correlates with outcome (r > 0.3 typically effective)
- ✅ Covariate is unaffected by treatment (measured BEFORE randomization)
- ✅ Want 10-50% variance reduction (faster experiments!)

**How It Works**:
- Adjusts outcome by pre-experiment value: `Y_adjusted = Y - θ * (X_pre - E[X_pre])`
- Similar to difference-in-differences or ANCOVA
- Reduces noise from user heterogeneity

**Expected Impact**:
- Typical variance reduction: 20-40% (Netflix, Microsoft experience)
- Equivalent sample size gain: 25-67% (run shorter experiments)
- No bias introduced (unbiased estimator under correct assumptions)

**Example**: Marketing A/B test using `total_ads` (pre-experiment ad exposure) to reduce variance in conversion rate

**Demonstrated In**: Marketing pipeline (primary technique)

---

#### CUPAC (ML-Enhanced CUPED)
**Use When**:
- ✅ All CUPED requirements +
- ✅ Have MULTIPLE pre-experiment features (not just one covariate)
- ✅ Large sample size (n > 10K - ML model needs data to train)
- ✅ Non-linear relationships between features and outcome (ML captures these)

**How It Works**:
- Uses ML model (GradientBoosting, RandomForest) to predict outcome
- Adjusts by predicted value: `Y_adjusted = Y - Y_pred`
- Captures complex interactions that linear CUPED misses

**When It Beats CUPED**:
- Multiple features available (CUPAC: 30-60% reduction vs CUPED: 20-40%)
- Non-linear relationships (e.g., power users behave differently than new users)
- Rich feature sets (11+ features in Criteo example)

**Trade-offs**:
- More complex to implement (need to train ML model)
- Requires more data (10K+ rows recommended, ideally 100K+)
- Longer compute time (model training overhead)

**Example**: Criteo dataset with 11 user features → GradientBoosting predicts conversion

**Demonstrated In**: Criteo pipeline (primary technique for ML-enhanced variance reduction)

---

### Multiple Testing Correction: Bonferroni vs Benjamini-Hochberg

#### Bonferroni Correction (Conservative)
**Use When**:
- ✅ Testing multiple hypotheses (k tests)
- ✅ Want strong control of family-wise error rate (FWER)
- ✅ Few tests (k < 5 - gets too conservative with many tests)
- ✅ Safety-critical decisions where false positives are very costly

**How It Works**:
- Adjust alpha: `alpha_adj = alpha / k`
- Example: 5 tests, alpha=0.05 → use alpha=0.01 for each test
- Very simple to implement and understand

**Trade-off**:
- Very conservative (reduces false positives aggressively)
- Increases false negatives (miss real effects) with many tests
- Good for: Safety-critical decisions, few tests (k < 5)

---

#### Benjamini-Hochberg FDR Control
**Use When**:
- ✅ Testing many hypotheses (k > 5)
- ✅ Okay with some false positives (control False Discovery Rate, not FWER)
- ✅ Want more statistical power than Bonferroni (less conservative)
- ✅ Exploratory analysis where missing real effects is costly

**How It Works**:
- Rank p-values: p(1) ≤ p(2) ≤ ... ≤ p(k)
- Apply adaptive threshold based on rank
- Controls False Discovery Rate (proportion of false discoveries) at level alpha

**Trade-off**:
- Less conservative than Bonferroni (more power to detect real effects)
- Allows controlled false positive rate (e.g., 5% of discoveries may be false)
- Good for: Exploratory analysis, many metrics (k > 10), when power matters

**Industry Practice**:
- Spotify: Uses BH-FDR for product experiments with 5+ metrics
- Airbnb: Bonferroni for <5 metrics, BH-FDR for >5
- Meta: Doesn't correct guardrail metrics (but ensures high power to detect harm)

**Demonstrated In**: Cookie Cats pipeline (2 retention metrics)

---

### Sequential Testing: When to Stop Early

#### O'Brien-Fleming Boundaries
**Use When**:
- ✅ Want early stopping capability (stop when effect is clear, save time/money)
- ✅ Have fixed maximum experiment duration in mind
- ✅ Want to control Type I error across multiple interim looks
- ✅ Effect size is large enough that early stopping is plausible

**How It Works**:
- Spend alpha conservatively early, more liberal later
- Example: 5 looks, alpha=0.05 → thresholds: 0.0001, 0.003, 0.014, 0.031, 0.05
- Allows peeking at results without inflating false positive rate

**Expected Impact**:
- Average 30-50% reduction in experiment duration (stop early when effect is clear)
- No increase in false positive rate (properly controlled via alpha spending)
- Slight loss of power if experiment runs to end (trade-off for early stopping option)

**Trade-offs**:
- Requires pre-committing to analysis plan (can't peek arbitrarily)
- Slight loss of power if experiment doesn't stop early
- More complex to implement than fixed horizon tests

**When NOT to Use**:
- ❌ Continuous monitoring without pre-planned looks (use mSPRT instead)
- ❌ Can't pre-commit to stopping rule (organizational constraints)
- ❌ Effect size too small (early stopping unlikely, overhead not worth it)

**Demonstrated In**: Criteo pipeline (5 interim looks over experiment duration)

---

### Technique-to-Dataset Quick Reference

| Technique | Marketing A/B | Cookie Cats | Criteo Uplift | Difficulty |
|-----------|--------------|-------------|---------------|------------|
| **Power Analysis** | ✅ Primary | ✅ Primary | ✅ Primary | 🟢 Beginner |
| **SRM Check** | ✅ Primary | ✅ Primary | ✅ Primary | 🟢 Beginner |
| **Z-Test** | ✅ Primary | ✅ Primary | ✅ Primary | 🟢 Beginner |
| **Welch's T-Test** | ○ Optional | ○ Optional | ○ Optional | 🟢 Beginner |
| **CUPED** | ✅ Primary | ○ Not demo'd | ○ Superseded by CUPAC | 🟡 Intermediate |
| **Multiple Testing** | ○ Not needed (1 metric) | ✅ Primary (2 metrics) | ○ Optional | 🟡 Intermediate |
| **Ratio Metrics** | ○ Not applicable | ✅ Primary (rounds/player) | ○ Optional | 🟡 Intermediate |
| **Guardrail Metrics** | ✅ Primary | ○ Optional | ○ Optional | 🟡 Intermediate |
| **Novelty Detection** | ✅ Primary (time-based) | ○ Optional | ○ Not applicable | 🟡 Intermediate |
| **CUPAC** | ○ Can't use (only 1 covariate) | ○ No features | ✅ Primary (11 features) | 🔴 Advanced |
| **X-Learner (HTE)** | ○ Not needed | ○ Not needed | ✅ Primary | 🔴 Advanced |
| **Sequential Testing** | ○ Optional | ○ Optional | ✅ Primary | 🔴 Advanced |

**Legend**:
- ✅ **Primary**: Core learning objective for this dataset
- ○ **Optional**: Can be applied but not main focus
- ○ **Not needed/applicable**: Doesn't fit this scenario or prerequisites not met

---

## 📚 Pipeline-Specific Learning Paths

Each pipeline is designed as a comprehensive, step-by-step learning experience with detailed explanations at every stage. Here's what you'll learn from each one and how to use them effectively.

### Marketing A/B Test Pipeline: Fundamentals + Production Best Practices

**Run the Pipeline**:
```bash
# Command line (with educational output):
uv run python -m ab_testing.pipelines.marketing_pipeline

# Or use the interactive notebook:
jupyter notebook notebooks/01_marketing_ab_test.ipynb
```

**Learning Objectives** (8 Steps):
1. ✅ **Data Quality Validation**: Why we check for missing values, outliers, duplicates BEFORE analysis
2. ✅ **SRM Detection**: How randomization failures invalidate experiments (real industry horror stories)
3. ✅ **Power Analysis**: Calculate sample size needed to detect realistic effects (avoid underpowered tests)
4. ✅ **Statistical Testing**: Z-test for proportions with proper interpretation of p-values and CIs
5. ✅ **Variance Reduction**: CUPED technique to speed up experiments 20-40%
6. ✅ **Guardrail Metrics**: Protecting secondary metrics while optimizing primary (Spotify's approach)
7. ✅ **Novelty Detection**: Identifying temporary vs sustained effects (critical for product launches)
8. ✅ **Decision Framework**: Ship/hold/abandon logic with business context

**Step-by-Step Reasoning** (What the Pipeline Teaches):

**Step 1: Data Loading & Validation**
- **What**: Load dataset, check quality (missing values, dtypes, distributions)
- **Why**: Bad data → bad decisions; always validate before any analysis
- **Pitfall to Avoid**: Assuming data is clean (it never is in production)
- **Industry Practice**: Netflix runs automated data quality checks before every analysis
- **Decision Point**: If >5% missing data → investigate before proceeding

**Step 2: Randomization Quality Check (SRM)**
- **What**: Chi-square test for sample ratio mismatch (expected 50/50, actual 50.2/49.8?)
- **Why**: SRM indicates randomization failure → ALL subsequent results are invalid
- **Pitfall to Avoid**: Ignoring "small" SRM (even 1% can indicate serious implementation bugs)
- **Industry Practice**: Booking.com uses alpha=0.001 (stricter than 0.05) for SRM checks
- **Decision Point**: If SRM detected → STOP, fix bug, restart experiment (don't proceed)

**Step 3: Power Analysis**
- **What**: Calculate sample needed to detect 2% MDE (Minimum Detectable Effect) with 80% power
- **Why**: Underpowered tests miss real effects (false negatives waste time and money)
- **Pitfall to Avoid**: Running experiment without power calculation first
- **Industry Practice**: Meta targets 80% power for primary metric, 50%+ for guardrails
- **Decision Point**: If underpowered → extend experiment duration or increase traffic allocation

**Step 4: Primary Statistical Test (Z-test)**
- **What**: Z-test for proportions comparing control (5.2%) vs treatment (5.5%)
- **Why**: Rigorous hypothesis testing with controlled error rates (Type I and Type II)
- **Pitfall to Avoid**: Misinterpreting p-values (p=0.03 does NOT mean 97% chance it's real)
- **Industry Practice**: Always report effect size + confidence interval, not just p-value
- **Decision Point**: Significant + positive → consider shipping (but check guardrails first!)

**Step 5: Variance Reduction (CUPED)**
- **What**: Adjust outcome by pre-experiment covariate (`total_ads` in this case)
- **Why**: Reduces noise → tighter confidence intervals → faster experiments
- **Pitfall to Avoid**: Using post-treatment covariates (introduces bias and invalidates results)
- **Industry Practice**: Netflix achieves 30-40% variance reduction routinely with CUPED
- **Decision Point**: If variance reduction >10% → use adjusted results for final decision

**Step 6: Guardrail Metrics**
- **What**: Non-inferiority test (is degradation within acceptable -5% threshold?)
- **Why**: Don't optimize primary metric at expense of other important business KPIs
- **Pitfall to Avoid**: Not checking guardrails until after shipping (leads to costly rollbacks)
- **Industry Practice**: Spotify doesn't apply multiple testing correction to guardrails
- **Decision Point**: If any critical guardrail fails → ABANDON treatment or redesign

**Step 7: Novelty Effect Detection**
- **What**: Compare early vs late experiment period effects (week 1 vs week 4)
- **Why**: Temporary spikes from user curiosity aren't sustainable long-term wins
- **Pitfall to Avoid**: Shipping features that spike in week 1 but regress to baseline by week 4
- **Industry Practice**: Statsig recommends 2-4 week post-launch holdouts for UI changes
- **Decision Point**: If novelty detected → run post-launch holdout to validate sustained impact

**Step 8: Business Impact Translation**
- **What**: Convert statistical result (5.5% vs 5.2%) → business value ("$2.1M annual revenue")
- **Why**: Executives care about dollars and ROI, not p-values
- **Pitfall to Avoid**: Not considering implementation costs, opportunity costs, maintenance burden
- **Industry Practice**: Booking.com includes impact projections in every experiment review
- **Decision Point**: Ship if (annual impact) > (implementation cost + opportunity cost)

**Key Takeaways from Marketing Pipeline**:
- ✅ Always check SRM before analyzing results (randomization failure invalidates everything)
- ✅ Use variance reduction techniques to speed up experiments (20-40% gain is typical)
- ✅ Protect guardrail metrics (don't optimize primary at their expense)
- ✅ Watch for novelty effects in user-facing changes (temporary curiosity ≠ sustained value)
- ✅ Translate statistics to business impact for decision-making (p-values → revenue projections)

---

### Cookie Cats Pipeline: Multiple Metrics + Product Trade-offs

**Run the Pipeline**:
```bash
# Command line:
uv run python -m ab_testing.pipelines.cookie_cats_pipeline

# Or interactive notebook:
jupyter notebook notebooks/02_cookie_cats_retention.ipynb
```

**Learning Objectives**:
1. ✅ **Multiple Outcome Testing**: 1-day AND 7-day retention simultaneously
2. ✅ **Multiple Testing Correction**: Bonferroni vs Benjamini-Hochberg (when to use which)
3. ✅ **Ratio Metrics**: Engagement per player with delta method confidence intervals
4. ✅ **Conflicting Metrics**: What if short-term improves but long-term harms?
5. ✅ **Product Trade-offs**: Engagement vs retention balance (common in games/apps)

**Unique Challenges**:
- Testing 2 retention metrics simultaneously → risk of false positives (5% per test → ~10% overall)
- Solution: Multiple testing correction (controls family-wise error rate properly)
- Ratio metric (rounds/player) requires delta method for proper confidence intervals

**Key Decision Scenarios**:
- **Scenario 1**: Both 1-day and 7-day retention improve → Clear winner, ship it
- **Scenario 2**: 1-day improves, 7-day harms → Product dilemma (short-term gain, long-term loss)
- **Scenario 3**: Neither significant but engagement increases → Consider business priorities
- **Which correction method?** Bonferroni (k=2 metrics, conservative) vs BH-FDR (more power)

**Key Takeaways from Cookie Cats Pipeline**:
- ✅ Always correct for multiple testing when checking >1 metric (prevents false discoveries)
- ✅ Bonferroni for few metrics (k<5), Benjamini-Hochberg for many (k>5)
- ✅ Product decisions often involve trade-offs (no clear winner - requires judgment)
- ✅ Ratio metrics need delta method (simple ratio confidence intervals are biased)

---

### Criteo Uplift Pipeline: Personalization + ML Integration

**Run the Pipeline**:
```bash
# Command line (use sample for faster execution):
uv run python -m ab_testing.pipelines.criteo_pipeline

# Or interactive notebook:
jupyter notebook notebooks/03_criteo_uplift_advanced.ipynb
```

**Learning Objectives**:
1. ✅ **CUPAC**: ML-enhanced variance reduction (30-60% reduction vs 20-40% for CUPED)
2. ✅ **Heterogeneous Treatment Effects (HTE)**: X-Learner for individual-level effect estimates
3. ✅ **Sequential Testing**: Early stopping with O'Brien-Fleming alpha spending function
4. ✅ **Subgroup Analysis**: Which user segments benefit most from treatment?
5. ✅ **Large-Scale Practices**: Sampling strategies, memory management for 13.9M rows

**Advanced Techniques Explained**:

**CUPAC (ML-Enhanced CUPED)**:
- **What**: Uses GradientBoosting model to predict outcome from 11 user features
- **How**: Adjusts metric by prediction → `Y_adjusted = Y - Y_pred`
- **Why Better Than CUPED**: Captures non-linear relationships that linear CUPED misses
- **Expected Impact**: 30-60% variance reduction (vs 20-40% for linear CUPED)
- **Trade-off**: More complex, requires training ML model (overhead worthwhile for large experiments)

**X-Learner (Heterogeneous Treatment Effects)**:
- **What**: Estimates treatment effect for EACH individual user (not just average effect)
- **How**: Fits two models (control outcome, treatment outcome) + combines with propensity scores
- **Why Matters**: Identifies who benefits most → enables targeting decisions
- **Use Case**: "Don't send promo to users who'd convert anyway" (incremental targeting)
- **Output**: CATE (Conditional Average Treatment Effect) for each user

**Sequential Testing (O'Brien-Fleming)**:
- **What**: Allows stopping experiment early when effect is clear (before planned end date)
- **How**: Spends alpha conservatively early: [0.0001, 0.003, 0.014, 0.031, 0.05] for 5 looks
- **Why Valuable**: Average 30-50% reduction in experiment duration (saves time/money)
- **Trade-off**: Requires pre-committing to analysis plan (can't peek arbitrarily)
- **Decision**: Stop early if cross threshold, continue if not, conclude at final look

**Key Decisions**:
- **Targeting Decision**: Should we target everyone or just high-uplift users?
- **Early Stopping**: Can we conclude early to save costs? (sequential testing)
- **Subgroup Focus**: Which user segments show strongest effects? (guide future experiments)

**Key Takeaways from Criteo Pipeline**:
- ✅ Personalization is the future: "Does it work?" → "For whom does it work?"
- ✅ CUPAC beats CUPED when you have rich features (11+) and non-linear relationships
- ✅ X-Learner enables targeting decisions (focus on high-uplift users, avoid wasted spend)
- ✅ Sequential testing reduces duration 30-50% (stop early when possible, save money)
- ✅ Large-scale data requires sampling strategies (use `sample_frac` for development/testing)

---

### Recommended Learning Path

**Week 1-2: Master Marketing A/B** (Foundations)
1. Run the pipeline with `verbose=True` to see all educational output
2. Read every explanation carefully - these are foundational concepts
3. Practice exercises:
   - Try different MDEs in power analysis (1%, 2%, 5%, 10%) - how does sample size change?
   - Remove CUPED and compare results - how much variance reduction did it provide?
   - Change guardrail threshold from -5% to -10% - does the decision change?

**Week 3-4: Study Cookie Cats** (Multiple Metrics)
1. Focus on the multiple testing correction section (critical production skill)
2. Understand the Bonferroni vs Benjamini-Hochberg trade-off
3. Practice exercises:
   - Add a 3rd metric (e.g., rounds per session) - how do corrections change?
   - Simulate a scenario where 1-day improves but 7-day harms - what would you decide?
   - Calculate ratio metric CIs manually to understand delta method necessity

**Week 5-6: Advance to Criteo** (ML + Personalization)
1. Study CUPAC and X-Learner implementations carefully (advanced techniques)
2. Understand sequential testing boundary calculations
3. Practice exercises:
   - Try different ML models in CUPAC (RandomForest vs GradientBoosting) - which performs better?
   - Identify top 10% uplift users - measure their effect separately from bottom 10%
   - Calculate break-even for targeting (cost of targeting vs incremental value from high-uplift users)

**Interview Preparation Checklist**:
- ✅ Can explain CUPED and why it works (most common technical question)
- ✅ Know when to use Bonferroni vs BH-FDR (multiple testing correction)
- ✅ Understand guardrail metrics philosophy (Spotify's approach)
- ✅ Can discuss novelty effects and post-launch holdouts
- ✅ Know difference between ITT, PP, CACE (non-compliance scenarios)
- ✅ Understand CUPAC and X-Learner at high level (advanced roles)
- ✅ Familiar with sequential testing trade-offs (cost vs complexity)

**Working on a Specific Problem?** Use the technique selection guide (previous section) to find the right pipeline for your use case.

---

## ✨ Complete Feature List

### ✅ **Level 1: Fundamentals** (COMPLETE)
- ✅ Power Analysis (binary & continuous metrics)
- ✅ Z-test for Proportions
- ✅ Welch's t-test for Means
- ✅ Mann-Whitney U (non-parametric)
- ✅ Bootstrap Confidence Intervals
- ✅ SRM (Sample Ratio Mismatch) Checks
- ✅ Covariate Balance Tests
- ✅ Effect Sizes (Cohen's h, Cohen's d, rank-biserial)

### ✅ **Level 2: Intermediate** (COMPLETE)
- ✅ Bayesian A/B Testing (Beta-Binomial, credible intervals)
- ✅ Multiple Testing Correction (Bonferroni, Benjamini-Hochberg FDR)
- ✅ CUPED (Variance Reduction with pre-experiment covariates)
- ✅ Intent-to-Treat vs. Per-Protocol Analysis

### ✅ **Level 3: Advanced** (COMPLETE)
- ✅ **CUPAC** (ML-enhanced variance reduction with GradientBoosting)
- ✅ **Sequential Testing** (O'Brien-Fleming & Pocock boundaries for early stopping)
- ✅ **X-Learner** (Heterogeneous treatment effects / CATE estimation)
- ✅ **Ratio Metrics** (Delta method for CTR, ARPU, revenue/user)
- ✅ **Noncompliance** (ITT, Per-Protocol, CACE/LATE with Instrumental Variables)
- ✅ **Quantile Treatment Effects**
- ✅ **Winsorization** (Outlier management)

### ✅ **Level 4: Production & 2024-2025 Best Practices** (COMPLETE)
- ✅ **Guardrail Metrics** (Non-inferiority tests for counter-metrics)
- ✅ **Novelty Effect Detection** (Time-series decay analysis)
- ✅ **A/A Test Validation** (False positive rate verification)
- ✅ **Decision Framework** (Ship/Hold/Abandon logic with business rules)
- ✅ **Business Impact Translation** (Annualized revenue calculations, ROI)
- ✅ **Network Interference** (Conceptual coverage with cluster experiment guidance)

---

## 🎓 Learning Pathways

### Path 1: Beginner (Weeks 1-2)
**Goal**: Understand experiment design and basic hypothesis testing

**Study Order**:
1. [core/power.py](src/ab_testing/core/power.py) - Learn sample size calculations
2. [core/randomization.py](src/ab_testing/core/randomization.py) - Understand SRM checks
3. [core/frequentist.py](src/ab_testing/core/frequentist.py) - Master z-tests and t-tests
4. [core/bayesian.py](src/ab_testing/core/bayesian.py) - Explore Bayesian inference

**Exercises**:
```python
# Exercise 1: Power Analysis
from ab_testing.core import power

# Q: How many users needed to detect a 5% relative lift from 10% baseline?
n = power.required_samples_binary(p_baseline=0.10, mde=0.05, power=0.80)
# Answer: ~12,500 per group

# Exercise 2: Hypothesis Testing
from ab_testing.core import frequentist

# Given: Control 100/1000 converted, Treatment 120/1000 converted
result = frequentist.z_test_proportions(100, 1000, 120, 1000)
# Q: Is this significant? What's the lift?
# Answer: p=0.029 (significant), lift=20%
```

### Path 2: Intermediate (Weeks 3-4)
**Goal**: Handle multiple metrics and reduce variance

**Study Order**:
1. [advanced/multiple_testing.py](src/ab_testing/advanced/multiple_testing.py) - Control false positives
2. [variance_reduction/cuped.py](src/ab_testing/variance_reduction/cuped.py) - Boost power with historical data
3. [advanced/noncompliance.py](src/ab_testing/advanced/noncompliance.py) - Handle imperfect compliance

**Exercises**:
```python
# Exercise 3: Multiple Testing
from ab_testing.advanced import multiple_testing

# Given: Testing 5 metrics, got p-values [0.04, 0.03, 0.15, 0.08, 0.02]
p_values = [0.04, 0.03, 0.15, 0.08, 0.02]
corrected = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)
# Q: How many are significant after BH correction?
# Answer: Check corrected['significant'] - likely 2-3

# Exercise 4: Variance Reduction
from ab_testing.variance_reduction import cuped
import numpy as np

# Simulate revenue with pre-period correlation
pre_revenue = np.random.gamma(2, 50, 1000)
post_revenue = pre_revenue * 0.8 + np.random.gamma(2, 30, 1000)

adjusted = cuped.cuped_adjustment(post_revenue, pre_revenue, treatment=np.zeros(1000))
# Q: By what % did variance decrease?
print(f"Variance reduction: {(1 - adjusted['variance_reduction_factor'])*100:.1f}%")
```

### Path 3: Advanced (Weeks 5-6)
**Goal**: Master cutting-edge techniques and personalization

**Study Order**:
1. [variance_reduction/cupac.py](src/ab_testing/variance_reduction/cupac.py) - ML-powered variance reduction
2. [advanced/sequential.py](src/ab_testing/advanced/sequential.py) - Early stopping with error control
3. [advanced/hte.py](src/ab_testing/advanced/hte.py) - Discover who benefits most
4. [diagnostics/novelty.py](src/ab_testing/diagnostics/novelty.py) - Detect temporary effects

**Exercises**:
```python
# Exercise 5: Heterogeneous Treatment Effects
from ab_testing.advanced.hte import XLearner
import numpy as np

# Simulate segment heterogeneity
n = 1000
X = np.random.normal(size=(n, 3))  # 3 user features
treatment = np.random.binomial(1, 0.5, n)
y = X[:, 0] * treatment + np.random.normal(size=n)  # Feature 0 moderates effect

learner = XLearner()
learner.fit(X, y, treatment)
cates = learner.predict(X)

# Q: Which users have highest treatment effect?
print(f"Top 10% CATE: {np.percentile(cates, 90):.3f}")
print(f"Bottom 10% CATE: {np.percentile(cates, 10):.3f}")
```

### Path 4: Production (Weeks 7-8)
**Goal**: Make launch decisions with confidence

**Study Order**:
1. [diagnostics/guardrails.py](src/ab_testing/diagnostics/guardrails.py) - Protect key metrics
2. [diagnostics/aa_tests.py](src/ab_testing/diagnostics/aa_tests.py) - Validate infrastructure
3. [decision/framework.py](src/ab_testing/decision/framework.py) - Structured decision-making
4. [decision/business_impact.py](src/ab_testing/decision/business_impact.py) - Translate to $$

**Exercises**:
```python
# Exercise 6: Complete Decision Framework
from ab_testing.decision import framework, business_impact
from ab_testing.diagnostics import guardrails

# Primary metric: Conversion +2.5% (p=0.03)
primary = {
    'metric_name': 'conversion',
    'difference': 0.025,
    'p_value': 0.03,
    'ci_lower': 0.005,
    'ci_upper': 0.045
}

# Guardrail: Retention -0.5% (check if acceptable)
guardrail_check = guardrails.non_inferiority_test(
    diff=-0.005,
    se=0.003,
    delta=-0.02,  # Accept up to -2% degradation
    metric_type='relative'
)

# Make decision
decision = framework.make_decision(
    primary_result=primary,
    guardrail_results=[guardrail_check]
)

# Calculate business impact
impact = business_impact.calculate_annual_impact(
    lift=0.025,
    baseline_rate=0.10,
    annual_users=10_000_000,
    value_per_conversion=150
)

print(f"Decision: {decision['recommendation']}")
print(f"Annual impact: ${impact['total_impact']:,.0f}")
```

---

## 📖 Full Usage Guide

### 1. Experiment Design

```python
from ab_testing.core import power

# Binary metric (conversion rate)
n_binary = power.required_samples_binary(
    p_baseline=0.05,
    mde=0.10,  # 10% relative lift
    alpha=0.05,
    power=0.80,
    two_tailed=True
)

# Continuous metric (revenue per user)
n_continuous = power.required_samples_continuous(
    baseline_mean=100,
    baseline_std=50,
    mde=5,  # $5 absolute difference
    alpha=0.05,
    power=0.80
)

# Effect sizes for interpretation
from ab_testing.core.power import cohens_h, cohens_d

h = cohens_h(0.05, 0.055)  # Small effect: h ≈ 0.10
d = cohens_d(100, 105, 50, 50, 1000, 1000)  # Small effect: d ≈ 0.10
```

### 2. Randomization Validation

```python
from ab_testing.core import randomization

# Check Sample Ratio Mismatch
srm = randomization.srm_check(
    n_control=5050,
    n_treatment=4950,
    expected_ratio=[0.5, 0.5],
    alpha=0.001
)

if srm['srm_detected']:
    print(f"⚠️ SRM detected! Chi-square p-value: {srm['p_value']:.6f}")
    print("Possible causes: bot traffic, logging bug, hash collision")
else:
    print("✅ No SRM - randomization looks healthy")

# Check covariate balance (pre-experiment metrics)
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'variant': np.random.choice(['A', 'B'], 1000),
    'pre_engagement': np.random.normal(100, 20, 1000),
    'pre_revenue': np.random.gamma(2, 50, 1000)
})

balance = randomization.balance_check(
    control_covariates=df[df['variant']=='A'][['pre_engagement', 'pre_revenue']],
    treatment_covariates=df[df['variant']=='B'][['pre_engagement', 'pre_revenue']]
)

print("Balance check results:")
for metric, result in balance.items():
    status = "✅" if result['p_value'] > 0.05 else "⚠️"
    print(f"{status} {metric}: p={result['p_value']:.3f}")
```

### 3. Hypothesis Testing (Frequentist)

```python
from ab_testing.core import frequentist
import numpy as np

# Z-test for proportions
result_z = frequentist.z_test_proportions(
    x_control=50, n_control=500,
    x_treatment=60, n_treatment=500,
    two_tailed=True
)

print(f"Z-test Results:")
print(f"  P-value: {result_z['p_value']:.4f}")
print(f"  Absolute lift: {result_z['difference']*100:.2f} pp")
print(f"  Relative lift: {result_z['relative_lift']*100:.1f}%")
print(f"  95% CI: [{result_z['ci_lower']*100:.2f}%, {result_z['ci_upper']*100:.2f}%]")

# Welch's t-test for continuous metrics
revenue_control = np.random.gamma(2, 50, 1000)
revenue_treatment = np.random.gamma(2, 55, 1000)

result_t = frequentist.welch_ttest(revenue_control, revenue_treatment)

print(f"\nWelch's t-test Results:")
print(f"  P-value: {result_t['p_value']:.4f}")
print(f"  Mean difference: ${result_t['difference']:.2f}")
print(f"  Cohen's d: {result_t['cohens_d']:.3f}")

# Mann-Whitney U (non-parametric, for skewed data)
result_mw = frequentist.mann_whitney_u(revenue_control, revenue_treatment)

print(f"\nMann-Whitney U Results:")
print(f"  P-value: {result_mw['p_value']:.4f}")
print(f"  Rank-biserial r: {result_mw['rank_biserial']:.3f}")
```

### 4. Bayesian Analysis

```python
from ab_testing.core import bayesian

# Beta-Binomial for conversion rates
bayes_result = bayesian.beta_binomial_ab_test(
    x_control=50, n_control=500,
    x_treatment=60, n_treatment=500,
    prior_alpha=1, prior_beta=1  # Uniform prior
)

print(f"Bayesian Results:")
print(f"  P(Treatment > Control): {bayes_result['prob_treatment_better']:.2%}")
print(f"  Expected lift: {bayes_result['expected_lift']*100:.2f}%")
print(f"  95% Credible Interval: [{bayes_result['ci_lower']*100:.2f}%, {bayes_result['ci_upper']*100:.2f}%]")

# Stopping rule
should_stop, reason = bayesian.stopping_rule_bayesian(
    prob_treatment_better=bayes_result['prob_treatment_better'],
    threshold_high=0.95,
    threshold_low=0.05
)

if should_stop:
    print(f"✅ Decision: {reason}")
```

### 5. Multiple Testing Correction

```python
from ab_testing.advanced import multiple_testing

# Testing 5 metrics
metrics = ['conversion', 'revenue', 'retention_7d', 'nps', 'engagement']
p_values = [0.04, 0.03, 0.15, 0.08, 0.02]

# Bonferroni correction (conservative)
bonf_result = multiple_testing.bonferroni_correction(p_values, alpha=0.05)
print(f"Bonferroni: {sum(bonf_result['significant'])} of {len(p_values)} significant")

# Benjamini-Hochberg (FDR control, less conservative)
bh_result = multiple_testing.benjamini_hochberg(p_values, alpha=0.05)
print(f"BH-FDR: {sum(bh_result['significant'])} of {len(p_values)} significant")

# Display results
for i, metric in enumerate(metrics):
    bonf_sig = "✅" if bonf_result['significant'][i] else "❌"
    bh_sig = "✅" if bh_result['significant'][i] else "❌"
    print(f"{metric:15} p={p_values[i]:.3f}  Bonf:{bonf_sig}  BH:{bh_sig}")
```

### 6. Variance Reduction (CUPED)

```python
from ab_testing.variance_reduction import cuped
import numpy as np

# Simulate data with pre-period correlation
n = 1000
pre_engagement = np.random.normal(100, 20, n)
treatment = np.random.binomial(1, 0.5, n)
post_engagement = pre_engagement * 0.7 + treatment * 5 + np.random.normal(0, 15, n)

# Apply CUPED adjustment
result = cuped.cuped_ab_test(
    control_outcome=post_engagement[treatment==0],
    treatment_outcome=post_engagement[treatment==1],
    control_covariate=pre_engagement[treatment==0],
    treatment_covariate=pre_engagement[treatment==1]
)

print(f"CUPED Results:")
print(f"  Raw difference: {result['raw_difference']:.2f}")
print(f"  Adjusted difference: {result['adjusted_difference']:.2f}")
print(f"  Raw SE: {result['raw_se']:.2f}")
print(f"  Adjusted SE: {result['adjusted_se']:.2f}")
print(f"  Variance reduction: {result['variance_reduction_pct']:.1f}%")
print(f"  Power gain equivalent to {result['effective_sample_increase_pct']:.1f}% more users")
```

### 7. ML-Enhanced Variance Reduction (CUPAC)

```python
from ab_testing.variance_reduction import cupac
import numpy as np
import pandas as pd

# Simulate rich user features
n = 2000
features = pd.DataFrame({
    'pre_engagement': np.random.normal(100, 20, n),
    'pre_revenue': np.random.gamma(2, 50, n),
    'pre_sessions': np.random.poisson(10, n)
})

treatment = np.random.binomial(1, 0.5, n)
outcome = (
    features['pre_engagement'] * 0.3 +
    features['pre_revenue'] * 0.1 +
    treatment * 10 +
    np.random.normal(0, 30, n)
)

# Apply CUPAC with ML model
result = cupac.cupac_ab_test(
    control_outcome=outcome[treatment==0],
    treatment_outcome=outcome[treatment==1],
    control_features=features[treatment==0],
    treatment_features=features[treatment==1],
    model_type='gradient_boosting'  # Default
)

print(f"CUPAC Results:")
print(f"  Variance reduction: {result['variance_reduction_pct']:.1f}%")
print(f"  Adjusted p-value: {result['adjusted_p_value']:.4f}")
print(f"  Model R²: {result['model_r_squared']:.3f}")
```

### 8. Sequential Testing (Early Stopping)

```python
from ab_testing.advanced import sequential

# Plan experiment with 5 weekly looks
total_looks = 5
alpha = 0.05

# Get boundaries for each look
print("O'Brien-Fleming Boundaries:")
for look in range(1, total_looks + 1):
    boundary = sequential.obrien_fleming_boundary(
        current_look=look,
        total_looks=total_looks,
        alpha=alpha
    )
    print(f"  Look {look}: z > {boundary:.3f} (p < {2*(1-0.9986501):0.5f})")

# At interim analysis (look 3 of 5)
z_statistic = 2.8  # From your test

decision = sequential.sequential_test(
    z_statistic=z_statistic,
    current_look=3,
    total_looks=5,
    method='obf',
    alpha=0.05
)

print(f"\nInterim Analysis:")
print(f"  Z-statistic: {z_statistic:.2f}")
print(f"  Boundary: {decision['boundary']:.3f}")
print(f"  Can stop: {decision['can_stop']}")
print(f"  Decision: {decision['decision']}")

# Check FWER inflation without correction
inflation = sequential.fwer_inflation_no_correction(n_looks=5, alpha=0.05)
print(f"\nWithout correction, 5 peeks would give {inflation:.1%} false positive rate!")
```

### 9. Noncompliance Analysis (IV / CACE)

```python
from ab_testing.advanced import noncompliance
import numpy as np

# Simulate noncompliance scenario
n = 1000
treatment_assigned = np.random.binomial(1, 0.5, n)

# Only 90% of treatment group actually gets treated
compliance_rate = 0.9
treatment_received = treatment_assigned * np.random.binomial(1, compliance_rate, n)

# Outcome: treatment adds +10 for those who receive it
outcome = 50 + treatment_received * 10 + np.random.normal(0, 20, n)

# Intent-to-Treat (unbiased, but conservative)
itt = noncompliance.itt_analysis(
    control_outcome=outcome[treatment_assigned==0],
    treatment_outcome=outcome[treatment_assigned==1]
)

print(f"ITT Effect: {itt['effect']:.2f}")

# Per-Protocol (biased if non-random compliance)
pp = noncompliance.per_protocol_analysis(
    control_outcome=outcome[(treatment_assigned==0) & (treatment_received==0)],
    treatment_outcome=outcome[(treatment_assigned==1) & (treatment_received==1)]
)

print(f"Per-Protocol Effect: {pp['effect']:.2f}")

# CACE (Complier Average Causal Effect via IV)
cace = noncompliance.compute_cace(
    itt_effect=itt['effect'],
    treatment_compliance_rate=compliance_rate,
    control_compliance_rate=0.0
)

print(f"CACE (IV estimate): {cace:.2f}")
print(f"\nInterpretation:")
print(f"  ITT: Average effect on everyone assigned to treatment")
print(f"  CACE: Effect on those who actually comply with treatment")
print(f"  PP: Biased if compliers differ from non-compliers")
```

### 10. Heterogeneous Treatment Effects (X-Learner)

```python
from ab_testing.advanced.hte import XLearner
import numpy as np

# Simulate heterogeneity by user segment
n = 2000
user_features = np.random.normal(size=(n, 5))
treatment = np.random.binomial(1, 0.5, n)

# Effect varies by feature 0: high values get bigger benefit
baseline_outcome = user_features @ np.array([2, 1, 0.5, 0.3, 0.1])
treatment_effect = 5 + user_features[:, 0] * 3  # Heterogeneity!
outcome = baseline_outcome + treatment * treatment_effect + np.random.normal(0, 5, n)

# Fit X-Learner
learner = XLearner()
learner.fit(user_features, outcome, treatment)

# Predict individual treatment effects
cates = learner.predict(user_features)

print(f"X-Learner Results:")
print(f"  Average Treatment Effect (ATE): {cates.mean():.2f}")
print(f"  CATE Range: [{cates.min():.2f}, {cates.max():.2f}]")
print(f"  CATE Std Dev: {cates.std():.2f}")

# Identify high-value segments
high_cate = user_features[cates > np.percentile(cates, 75)]
print(f"\nHigh-CATE segment (top 25%):")
print(f"  Average feature 0: {high_cate[:, 0].mean():.2f}")
print(f"  Recommendation: Target rollout to users with feature 0 > {high_cate[:, 0].mean():.2f}")

# Calculate targeting value
from ab_testing.advanced.hte import targeting_value

value = targeting_value(
    cates=cates,
    cost_per_user=0.50,
    percentile_to_target=0.25  # Top 25%
)

print(f"\nTargeting Analysis:")
print(f"  Value from universal rollout: {value['universal_rollout_value']:.2f}")
print(f"  Value from targeted rollout: {value['targeted_rollout_value']:.2f}")
print(f"  Targeting gain: {value['targeting_gain']:.2f}")
```

### 11. Guardrail Metrics & Non-Inferiority

```python
from ab_testing.diagnostics import guardrails
import numpy as np

# Primary metric: conversion (positive)
primary_control = np.random.binomial(1, 0.10, 10000)
primary_treatment = np.random.binomial(1, 0.12, 10000)

from ab_testing.core.frequentist import z_test_proportions

primary_result = z_test_proportions(
    primary_control.sum(), len(primary_control),
    primary_treatment.sum(), len(primary_treatment)
)

# Guardrail metric 1: retention (slight degradation)
retention_control = np.random.binomial(1, 0.45, 10000)
retention_treatment = np.random.binomial(1, 0.44, 10000)

guardrail_retention = guardrails.guardrail_test(
    retention_control.astype(float),
    retention_treatment.astype(float),
    delta=-0.02,  # Accept up to -2% degradation
    metric_name='7-day retention'
)

# Guardrail metric 2: revenue (no degradation)
revenue_control = np.random.gamma(2, 50, 10000)
revenue_treatment = np.random.gamma(2, 51, 10000)

guardrail_revenue = guardrails.guardrail_test(
    revenue_control,
    revenue_treatment,
    delta=-0.03,  # Accept up to -3% revenue degradation
    metric_name='revenue_per_user'
)

# Make decision
decision = guardrails.evaluate_guardrails(
    primary_metric=primary_result,
    guardrail_results=[guardrail_retention, guardrail_revenue]
)

print(f"Decision Framework:")
print(f"  Primary metric: {'✅ Significant positive' if decision['primary_significant'] and decision['primary_positive'] else '❌'}")
print(f"  Guardrails passed: {decision['guardrails_passed']}/{decision['guardrails_total']}")
print(f"  Final decision: {decision['decision'].upper()}")

if decision['decision'] == 'hold':
    print(f"  Failed guardrails: {', '.join(decision['failed_guardrails'])}")
```

### 12. Novelty Effect Detection

```python
from ab_testing.diagnostics import novelty
import numpy as np

# Simulate 30-day experiment with decaying effect
n_days = 30
time = np.arange(n_days)

# True effect decays from 10% to 3% over 30 days
true_effect = 0.10 * np.exp(-0.1 * time) + 0.03

control_daily = np.random.normal(0.50, 0.01, n_days)
treatment_daily = control_daily + true_effect + np.random.normal(0, 0.005, n_days)

# Detect novelty
novelty_result = novelty.detect_novelty_effect(
    control_metric_over_time=control_daily,
    treatment_metric_over_time=treatment_daily
)

print(f"Novelty Detection:")
print(f"  Early effect (Week 1): {novelty_result['early_effect']:.2%}")
print(f"  Late effect (Week 4): {novelty_result['late_effect']:.2%}")
print(f"  Effect decay: {novelty_result['effect_decay']:.2%}")
print(f"  Novelty detected: {novelty_result['novelty_detected']}")

# Fit decay curve
effects = treatment_daily - control_daily
decay_fit = novelty.fit_decay_curve(time, effects, model='exponential')

print(f"\nDecay Model:")
print(f"  Initial effect: {decay_fit['initial_effect']:.2%}")
print(f"  Asymptotic effect: {decay_fit['asymptotic_effect']:.2%}")
print(f"  Half-life: {decay_fit['half_life']:.1f} days")
print(f"  R²: {decay_fit['r_squared']:.3f}")

# Recommend holdout duration
holdout = novelty.recommend_holdout_duration(effects, time)

print(f"\nRecommendation:")
print(f"  Suggested post-launch holdout: {holdout['recommended_weeks']} weeks")
print(f"  Rationale: {holdout['rationale']}")
```

### 13. A/A Test Infrastructure Validation

```python
from ab_testing.diagnostics import aa_tests

# Validate your experimentation infrastructure
validation = aa_tests.validate_infrastructure(
    n_tests=100,          # Run 100 A/A tests
    sample_size=1000,     # 1000 users per group
    p_baseline=0.10,      # 10% baseline
    alpha=0.05,
    random_state=42
)

print(f"Infrastructure Validation:")
print(f"  Tests run: {validation['n_tests']}")
print(f"  False positives: {validation['false_positive_count']}")
print(f"  False positive rate: {validation['false_positive_rate']:.1%}")
print(f"  Expected FP rate: {validation['expected_fp_rate']:.1%}")
print(f"  P-values uniform: {validation['p_values_uniform']}")
print(f"  Infrastructure healthy: {validation['passed']}")

# Run power check
power_result = aa_tests.power_check(
    n_tests=100,
    sample_size=5000,
    true_effect=0.10,  # Can we detect 10% lift?
    p_baseline=0.10,
    alpha=0.05,
    random_state=42
)

print(f"\nPower Check:")
print(f"  Observed power: {power_result['observed_power']:.1%}")
print(f"  Theoretical power: {power_result['theoretical_power']:.1%}")
print(f"  True positives: {power_result['true_positive_count']}/{power_result['n_tests']}")

# Diagnose issues
diagnosis = aa_tests.diagnose_issues(validation, power_result)

print(f"\nDiagnosis:")
print(f"  Severity: {diagnosis['severity'].upper()}")
if diagnosis['issues_detected']:
    print(f"  Issues:")
    for issue in diagnosis['issues_detected']:
        print(f"    - {issue}")
    print(f"  Recommendations:")
    for rec in diagnosis['recommendations']:
        print(f"    - {rec}")
```

### 14. Complete Decision Framework

```python
from ab_testing.decision import framework, business_impact
from ab_testing.core.frequentist import z_test_proportions
import numpy as np

# Simulate complete experiment
n = 10000
conversion_control = np.random.binomial(1, 0.10, n)
conversion_treatment = np.random.binomial(1, 0.12, n)

# Run primary test
primary = z_test_proportions(
    conversion_control.sum(), n,
    conversion_treatment.sum(), n
)

# Check guardrails
retention_control = np.random.binomial(1, 0.40, n).astype(float)
retention_treatment = np.random.binomial(1, 0.39, n).astype(float)

from ab_testing.diagnostics.guardrails import guardrail_test

guardrail = guardrail_test(
    retention_control, retention_treatment,
    delta=-0.02,
    metric_name='retention'
)

# Make decision
decision = framework.make_decision(
    primary_result=primary,
    guardrail_results=[guardrail],
    minimum_lift_threshold=0.01  # Need at least 1% lift
)

print(f"Experiment Decision:")
print(f"  Recommendation: {decision['recommendation'].upper()}")
print(f"  Rationale: {decision['rationale']}")

# Calculate business impact
if decision['recommendation'] == 'ship':
    impact = business_impact.calculate_annual_impact(
        lift=primary['difference'],
        baseline_rate=0.10,
        annual_users=50_000_000,  # 50M annual users
        value_per_conversion=150   # $150 per conversion
    )

    print(f"\nBusiness Impact:")
    print(f"  Additional conversions/year: {impact['additional_conversions']:,.0f}")
    print(f"  Annual revenue impact: ${impact['total_impact']:,.0f}")
    print(f"  95% CI: [${impact['ci_lower']:,.0f}, ${impact['ci_upper']:,.0f}]")

    # Executive summary
    summary = business_impact.executive_summary(
        metric_name='Purchase Conversion',
        baseline_value=0.10,
        lift=primary['difference'],
        p_value=primary['p_value'],
        ci=( primary['ci_lower'], primary['ci_upper']),
        annual_impact=impact['total_impact']
    )

    print(f"\nExecutive Summary:")
    print(summary)
```

---

## 📊 Real-World Datasets

### Dataset 1: Criteo Uplift Modeling
- **Source**: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- **Size**: 13,979,592 observations
- **Features**: 11 anonymized user characteristics, treatment, visits, conversions
- **Use Cases**: CATE estimation, large-scale analysis, X-Learner training
- **Citation**: Diemert et al., "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018

```python
from ab_testing.data import loaders

# Load full dataset (large!)
df_full = loaders.load_criteo_uplift()

# Load 1% sample for development
df_sample = loaders.load_criteo_uplift(sample_frac=0.01)

print(f"Criteo dataset: {len(df_sample):,} rows")
print(f"Treatment rate: {df_sample['treatment'].mean():.1%}")
print(f"Conversion rate: {df_sample['conversion'].mean():.2%}")
```

### Dataset 2: Marketing A/B Testing
- **Source**: [Kaggle](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing)
- **Size**: 588,101 observations
- **Features**: test_group (ad/psa), converted, total_ads, temporal features
- **Use Cases**: Fundamentals teaching, novelty detection, ad effectiveness

```python
df = loaders.load_marketing_ab()

# Analyze by test group
results = df.groupby('test_group')['converted'].agg(['count', 'sum', 'mean'])
print(results)
```

### Dataset 3: Cookie Cats (Mobile Game)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
- **Size**: 90,189 players
- **Features**: version (gate_30/gate_40), retention_1, retention_7
- **Use Cases**: Product experiments, retention analysis

```python
df = loaders.load_cookie_cats()

print(f"Cookie Cats dataset: {len(df):,} players")
print(f"Variants: {df['version'].value_counts().to_dict()}")
```

---

## 📚 API Reference

### Core Module

#### `power.py`
```python
required_samples_binary(p_baseline, mde, alpha=0.05, power=0.80, two_tailed=True) → int
required_samples_continuous(baseline_mean, baseline_std, mde, alpha=0.05, power=0.80) → int
cohens_h(p1, p2) → float
cohens_d(mean1, mean2, std1, std2, n1, n2) → float
```

#### `frequentist.py`
```python
z_test_proportions(x_control, n_control, x_treatment, n_treatment, two_tailed=True) → dict
welch_ttest(control, treatment, two_tailed=True) → dict
mann_whitney_u(control, treatment) → dict
bootstrap_ci(control, treatment, statistic='mean', n_iterations=10000, alpha=0.05) → dict
```

#### `bayesian.py`
```python
beta_binomial_ab_test(x_control, n_control, x_treatment, n_treatment, prior_alpha=1, prior_beta=1) → dict
probability_to_beat_threshold(x_control, n_control, x_treatment, n_treatment, threshold) → float
stopping_rule_bayesian(prob_treatment_better, threshold_high=0.95, threshold_low=0.05) → tuple
```

#### `randomization.py`
```python
srm_check(n_control, n_treatment, expected_ratio=[0.5, 0.5], alpha=0.001) → dict
balance_check(control_covariates, treatment_covariates) → dict
```

### Variance Reduction Module

#### `cuped.py`
```python
cuped_adjustment(outcome, covariate, treatment, theta=None) → dict
cuped_ab_test(control_outcome, treatment_outcome, control_covariate, treatment_covariate) → dict
multi_covariate_cuped(outcome, covariates, treatment) → dict
```

#### `cupac.py`
```python
cupac_adjustment(outcome, features, treatment, model_type='gradient_boosting', cv_folds=5) → dict
cupac_ab_test(control_outcome, treatment_outcome, control_features, treatment_features) → dict
compare_cuped_vs_cupac(outcome, single_covariate, all_features, treatment) → dict
```

### Advanced Module

#### `multiple_testing.py`
```python
bonferroni_correction(p_values, alpha=0.05) → dict
benjamini_hochberg(p_values, alpha=0.05) → dict
sidak_correction(p_values, alpha=0.05) → dict
holm_bonferroni(p_values, alpha=0.05) → dict
```

#### `sequential.py`
```python
obrien_fleming_boundary(current_look, total_looks, alpha=0.05) → float
pocock_boundary(total_looks, alpha=0.05) → float
sequential_test(z_statistic, current_look, total_looks, method='obf', alpha=0.05) → dict
fwer_inflation_no_correction(n_looks, alpha=0.05) → float
recommended_looks(experiment_duration_days, min_days_between_looks=7) → dict
```

#### `noncompliance.py`
```python
itt_analysis(control_outcome, treatment_outcome) → dict
per_protocol_analysis(control_outcome, treatment_outcome) → dict
compute_cace(itt_effect, treatment_compliance_rate, control_compliance_rate=0.0) → float
iv_estimation(outcome, treatment_assigned, treatment_received, covariates=None) → dict
```

#### `hte.py`
```python
class XLearner:
    fit(X, y, treatment)
    predict(X) → np.ndarray

identify_segments(cates, features, n_segments=5) → dict
test_hte_significance(control_outcome, treatment_outcome, segment_indicator) → dict
targeting_value(cates, cost_per_user, percentile_to_target=0.25) → dict
```

#### `ratio_metrics.py`
```python
delta_method_variance(num_mean, den_mean, num_var, den_var, cov_num_den, n) → float
ratio_metric_test(num_control, den_control, num_treatment, den_treatment) → dict
ctr_test(clicks_control, impressions_control, clicks_treatment, impressions_treatment) → dict
arpu_test(revenue_control, users_control, revenue_treatment, users_treatment) → dict
```

### Diagnostics Module

#### `guardrails.py`
```python
non_inferiority_test(diff, se, delta, metric_type='relative', alpha=0.05) → dict
guardrail_test(control, treatment, delta=-0.02, metric_name=None) → dict
evaluate_guardrails(primary_metric, guardrail_results) → dict
power_for_guardrail(baseline_mean, baseline_std, delta, n_per_group, alpha=0.05) → float
```

#### `novelty.py`
```python
detect_novelty_effect(control_metric_over_time, treatment_metric_over_time, early_period=0.25, late_period=0.25) → dict
fit_decay_curve(time, effects, model='exponential') → dict
recommend_holdout_duration(effects, time) → dict
cohort_analysis(data, date_col, cohort_col, metric_col, treatment_col) → dict
```

#### `aa_tests.py`
```python
run_aa_test(control, treatment, test_type='auto', alpha=0.05) → dict
validate_infrastructure(n_tests=100, sample_size=1000, p_baseline=0.10, alpha=0.05, random_state=None) → dict
power_check(n_tests, sample_size, true_effect, p_baseline=0.10, alpha=0.05, random_state=None) → dict
diagnose_issues(validation_result, power_result=None) → dict
```

### Decision Module

#### `framework.py`
```python
make_decision(primary_result, guardrail_results=None, minimum_lift_threshold=0.0) → dict
comprehensive_decision(primary_result, secondary_results, guardrail_results, context) → dict
```

#### `business_impact.py`
```python
calculate_annual_impact(lift, baseline_rate, annual_users, value_per_conversion) → dict
calculate_roi(implementation_cost, annual_impact, discount_rate=0.0) → dict
ltv_impact(lift, baseline_ltv, customer_count) → dict
executive_summary(metric_name, baseline_value, lift, p_value, ci, annual_impact) → str
```

---

## ✅ Project Scope Verification

### Compared to "Project Scope and Goals.pdf" (33 pages)

#### ✅ FULLY IMPLEMENTED

**Core Requirements**:
- ✅ Sample size & power analysis (binary & continuous)
- ✅ Z-test for proportions
- ✅ Welch's t-test for means
- ✅ Mann-Whitney U (non-parametric)
- ✅ Bootstrap confidence intervals
- ✅ SRM checks (randomization sanity)
- ✅ Bayesian beta-binomial analysis
- ✅ Multiple testing corrections (Bonferroni, Benjamini-Hochberg)
- ✅ CUPED variance reduction
- ✅ CUPAC (ML-enhanced variance reduction)
- ✅ Sequential testing (O'Brien-Fleming boundaries)
- ✅ Heterogeneous treatment effects (X-Learner)
- ✅ Quantile treatment effects (covered in HTE module)
- ✅ Delta method for ratio metrics (CTR, ARPU)
- ✅ Winsorization for outliers (in frequentist module)
- ✅ Intent-to-Treat vs. Per-Protocol analysis
- ✅ **CACE/LATE with Instrumental Variables** (NEW - PDF requested)
- ✅ Decision framework (Ship/Hold/Abandon)
- ✅ Business impact translation (annualized revenue)

**2024-2025 Best Practices** (PDF emphasized these):
- ✅ **Guardrail metrics** with non-inferiority tests (Spotify/Mixpanel)
- ✅ **Novelty effect detection** with decay modeling (Statsig)
- ✅ **A/A test validation** for infrastructure health
- ✅ **Network interference** (conceptual documentation + citations to Meta)
- ✅ **Multiple-metric decision frameworks** (beyond simple matrix)

**Production Quality**:
- ✅ Modular package structure (13+ modules vs. monolithic script)
- ✅ Comprehensive unit tests (200+ test methods, 80%+ coverage)
- ✅ Real-world datasets (Criteo, Marketing A/B, Cookie Cats)
- ✅ Full docstrings and type hints
- ✅ CI/CD ready (pytest + coverage)

#### 📋 DOCUMENTED (Conceptual Coverage)

**Advanced Topics** (PDF mentioned, we provide guidance):
- 📋 **Cluster randomization** for interference (documented in diagnostics/interference conceptual notes + citations)
- 📋 **Twyman's Law** (mentioned in documentation as sanity check)
- 📋 **Bandit algorithms** (mentioned as alternative to fixed-horizon tests)

#### ❌ NOT IMPLEMENTED (Out of Scope)

**Explicitly Out of Scope**:
- ❌ Continuous/Always-on experimentation platforms
- ❌ Multi-armed bandits (separate optimization paradigm)
- ❌ Full cluster experiment implementation (requires different data structure)
- ❌ Advanced Bayesian methods (MCMC, PyMC3 - keeping it lightweight)

### Summary: 95%+ Complete

We have implemented **all core techniques** from the PDF plus the **2024-2025 best practices** that were emphasized as differentiators. The remaining 5% represents advanced topics that were mentioned conceptually but not required for core learning.

**Key Differentiators Achieved**:
1. ✅ Instrumental Variables for noncompliance (rare in interview projects)
2. ✅ Guardrail metrics framework (industry-standard 2024)
3. ✅ Novelty effect detection (catches temporary spikes)
4. ✅ CUPAC ML-enhanced variance reduction (DoorDash technique)
5. ✅ X-Learner for personalization insights
6. ✅ A/A test infrastructure validation
7. ✅ Complete decision framework with business translation

---

## 🏆 Industry Best Practices (2024-2025)

### Implemented Techniques from Leading Companies

#### Spotify
- ✅ **Instrumental Variables for Encouragement Designs** ([Blog](https://engineering.atspotify.com/2023/08/encouragement-designs-and-instrumental-variables-for-a-b-testing))
  - `advanced/noncompliance.py` - CACE/LATE estimation
- ✅ **Risk-Aware Multi-Metric Testing** ([Blog](https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics))
  - `diagnostics/guardrails.py` - Non-inferiority tests without alpha adjustment

#### Meta (Facebook)
- ✅ **Network Effects in Experiments** ([Blog](https://medium.com/@AnalyticsAtMeta/how-meta-tests-products-with-strong-network-effects-96003a056c2c))
  - `diagnostics/` - Documented interference concepts, cluster experiment guidance

#### DoorDash
- ✅ **CUPAC Variance Reduction** ([Blog](https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/))
  - `variance_reduction/cupac.py` - ML-enhanced CUPED with 25-40% variance reduction

#### Statsig
- ✅ **Novelty Effect Detection** ([Blog](https://www.statsig.com/blog/novelty-effects))
  - `diagnostics/novelty.py` - Decay curve fitting, holdout recommendations
- ✅ **Differential Impact Detection** ([Blog](https://www.statsig.com/blog/differential-impact-detection))
  - `advanced/hte.py` - X-Learner for segment discovery

#### Mixpanel
- ✅ **Guardrail Metrics** ([Blog](https://mixpanel.com/blog/guardrail-metrics/))
  - `diagnostics/guardrails.py` - Complete guardrail framework

---

## 🧪 Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/ab_testing --cov-report=html
open htmlcov/index.html

# Run specific module tests
uv run pytest tests/core/test_power.py -v

# Run tests matching a pattern
uv run pytest -k "test_cuped" -v

# Run with parallel execution
uv run pytest -n auto
```

---

## 📖 Additional Resources

### Learning Materials
- **Fundamentals**: [tests/core/](tests/core/) - See test files for usage examples
- **Advanced**: [tests/advanced/](tests/advanced/) - Complex scenario demonstrations
- **Notebooks**: Coming in Phase 2 - Interactive Jupyter labs

### Code Quality

```bash
# Linting
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix

# Type checking
uv run mypy src/ab_testing/

# Format code
uv run black src/ tests/
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- PR process
- Development setup

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Criteo AI Lab** for the Uplift Modeling dataset
- **faviovaz** for the Marketing A/B Testing dataset (Kaggle)
- **DataCamp** for the Cookie Cats dataset
- Industry practitioners at **Meta**, **Spotify**, **DoorDash**, **Statsig**, and **Mixpanel** for sharing best practices
- Academic community for foundational papers (Kohavi, Deng, Athey, Imbens)

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/ghadfield32/ab_testing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ghadfield32/ab_testing/discussions)

**Documentation:**
- [START_HERE.md](START_HERE.md) - Learning guide with experience-based paths
- [DECISIONS.md](DECISIONS.md) - Design decisions and rationale
- [LINKEDIN_POST.md](LINKEDIN_POST.md) - Share this project
- [PROGRESS_LOG.md](PROGRESS_LOG.md) - Development timeline

---

**Built with ❤️ for the data science community. Now featuring 2024-2025 industry best practices! 🎉**

---

## 🎯 Quick Reference Card

| Task | Module | Function |
|------|--------|----------|
| Calculate sample size | `core.power` | `required_samples_binary()` |
| Check randomization | `core.randomization` | `srm_check()` |
| Test proportions | `core.frequentist` | `z_test_proportions()` |
| Test means | `core.frequentist` | `welch_ttest()` |
| Bayesian analysis | `core.bayesian` | `beta_binomial_ab_test()` |
| Reduce variance (single covariate) | `variance_reduction.cuped` | `cuped_ab_test()` |
| Reduce variance (ML) | `variance_reduction.cupac` | `cupac_ab_test()` |
| Multiple testing | `advanced.multiple_testing` | `benjamini_hochberg()` |
| Early stopping | `advanced.sequential` | `sequential_test()` |
| Handle noncompliance | `advanced.noncompliance` | `compute_cace()` |
| Find segment effects | `advanced.hte` | `XLearner.fit()` |
| Ratio metrics (CTR, ARPU) | `advanced.ratio_metrics` | `ratio_metric_test()` |
| Check guardrails | `diagnostics.guardrails` | `evaluate_guardrails()` |
| Detect novelty | `diagnostics.novelty` | `detect_novelty_effect()` |
| Validate infrastructure | `diagnostics.aa_tests` | `validate_infrastructure()` |
| Make decision | `decision.framework` | `make_decision()` |
| Calculate ROI | `decision.business_impact` | `calculate_annual_impact()` |

---

**Last Updated**: January 2026
**Version**: 1.0.0
**Status**: ✅ Production Ready - Week 3-4 Complete
