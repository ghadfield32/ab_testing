# Design Decisions: A/B Testing Framework

This document explains the "why" behind major decisions made during the development of this A/B testing educational framework.

---

## Philosophy & Goals

### Target Audience
Data scientists and analysts who want to learn production-grade A/B testing, not just textbook statistics.

### Core Principles
1. **Real datasets, not toy examples** - Every technique demonstrated on actual industry data
2. **Production patterns, not academic exercises** - Code structured like real experimentation platforms
3. **Decision-focused, not just p-value focused** - Ship/Hold/Abandon decisions require judgment
4. **Progressive complexity** - Beginner → Intermediate → Advanced learning path

### What This Is NOT
- A statistics textbook (assumes basic stats knowledge)
- A multi-armed bandit framework (different paradigm)
- A real-time experimentation platform (batch analysis focus)

---

## Dataset Selection Decisions

### Why These 3 Datasets?

We chose datasets that cover the three most common A/B testing scenarios in industry:

| Dataset | Scenario | Why Selected |
|---------|----------|--------------|
| **Marketing A/B** | Marketing effectiveness | Clean binary outcome, pre-experiment covariate for CUPED demo |
| **Cookie Cats** | Product optimization | Multiple metrics, ratio metrics, product decision context |
| **Criteo Uplift** | Personalization/targeting | Industry standard, massive scale (13.9M), rich features for ML |

### Marketing A/B Dataset (Beginner)

**Source:** [Kaggle - faviovaz/marketing-ab-testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing)

**Why we chose it:**
- Binary outcome (conversion) - simplest metric type
- Pre-experiment covariate (`total_ads`) - enables CUPED demonstration
- Large enough (588K rows) for statistical rigor
- Temporal data allows novelty effect detection

**Key Discovery:** This dataset has a 96%/4% treatment split, not 50/50. It's observational data mislabeled as an A/B test. We kept it intentionally because:
- Perfect for teaching SRM detection on "real" bad data
- Demonstrates why randomization validation is critical
- Most publicly available "A/B test" datasets have similar issues

### Cookie Cats Dataset (Intermediate)

**Source:** [Kaggle - mursideyarkin/mobile-games-ab-testing-cookie-cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)

**Why we chose it:**
- Multiple outcomes (1-day AND 7-day retention) - requires multiple testing correction
- Ratio metric (rounds per player) - requires delta method for proper CIs
- Product decision context (gate placement) - teaches trade-off thinking
- Properly randomized (unlike Marketing dataset)

**Trade-off taught:** What if 1-day retention improves but 7-day retention harms? This reflects real product decisions.

### Criteo Uplift Dataset (Advanced)

**Source:** [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)

**Why we chose it:**
- Industry standard for uplift modeling research
- Massive scale (13.9M rows) - teaches large-scale practices
- 11 user features - enables CUPAC and X-Learner demonstrations
- Two-stage outcome (visit → conversion) - funnel analysis
- Used in academic papers and Kaggle competitions

**Alternative considered:** Uber/Netflix datasets aren't publicly available. Synthetic data would miss real-world messiness.

---

## Statistical Method Decisions

### Included Methods

| Method | Why Included | Primary Dataset |
|--------|--------------|-----------------|
| **Z-test for proportions** | Industry standard for binary outcomes | All 3 |
| **Welch's t-test** | Continuous outcomes, robust to unequal variance | Reference |
| **Mann-Whitney U** | Non-parametric for skewed data (revenue, time) | Reference |
| **Bayesian A/B** | Alternative paradigm, stopping rules | Reference |
| **CUPED** | 20-40% variance reduction, Netflix/Microsoft standard | Marketing |
| **CUPAC** | ML-enhanced, 30-60% reduction (DoorDash technique) | Criteo |
| **Bonferroni** | Conservative multiple testing (FWER) | Cookie Cats |
| **Benjamini-Hochberg** | FDR control for many metrics | Cookie Cats |
| **X-Learner** | HTE estimation for personalization | Criteo |
| **O'Brien-Fleming** | Sequential testing gold standard | Criteo |
| **Delta method** | Proper CIs for ratio metrics | Cookie Cats |

### Excluded Methods (and Why)

| Method | Why Excluded |
|--------|--------------|
| **Multi-armed bandits** | Different paradigm (explore-exploit vs. hypothesis testing). Scope creep. |
| **mSPRT / continuous monitoring** | O'Brien-Fleming covers the key concepts more simply |
| **MCMC / PyMC3 Bayesian** | Heavy dependency, keeping it lightweight |
| **Cluster randomization** | Requires different data structure; documented conceptually only |
| **Causal forests** | X-Learner covers HTE; forests add complexity without much pedagogical gain |
| **Regression discontinuity** | Different use case (quasi-experimental) |

### Why Two-Stage SRM Gating?

**Problem discovered:** With large samples (100K+), chi-square detects tiny deviations as "significant." A 50.1%/49.9% split with 1M users is statistically significant but practically meaningless.

**Solution implemented:**
- **Stage A (Statistical):** p-value < alpha (traditional SRM)
- **Stage B (Practical):** deviation > 1 percentage point

Only flag as severe SRM if BOTH stages fail. This prevents false alarms at scale while catching real randomization bugs.

---

## Architecture Decisions

### Package Structure

```
ab_testing/
├── core/           # Fundamentals (no external ML dependencies)
├── variance_reduction/  # CUPED, CUPAC
├── advanced/       # Techniques that build on core
├── diagnostics/    # Quality checks and validation
├── decision/       # Business-focused frameworks
├── data/           # Dataset loaders
└── pipelines/      # End-to-end demonstrations
```

**Why this structure:**
- **Mental model:** Mirrors how practitioners think (core → advanced → decision)
- **Importable components:** `from ab_testing.core import power`
- **Optional complexity:** Can use just `core` without ML dependencies

### Testing Approach

- **pytest** with 200+ tests
- **80%+ coverage** target for core modules
- **Integration tests** for pipelines (verify end-to-end)

**Why extensive tests:** Function contracts changed during refactoring and broke pipelines in 30+ places. Tests catch these regressions.

### Type Hints

Full type annotations throughout (`Dict`, `Optional`, `Literal`, etc.)

**Why:** IDE autocomplete, self-documenting code, catches bugs early.

---

## Key Lessons Learned

### 1. Function Contract Consistency

Refactoring changed return dict keys (e.g., `'difference'` → `'absolute_lift'`), breaking pipelines silently.

**Lesson:** Document return structures explicitly. Run integration tests after refactoring.

### 2. Dataset Taxonomy Matters

RCT (randomized) vs. observational data requires completely different handling:
- RCT: SRM failure = stop analysis
- Observational: SRM check is diagnostic only

**Lesson:** Always document whether data is from a true randomized experiment.

### 3. Two-Stage Gating for Scale

Statistical significance alone causes false alarms at large scale. Practical significance thresholds are essential.

**Lesson:** Add practical significance checks to production SRM detection.

### 4. CI-Based Non-Inferiority

Point estimate can pass threshold (-2.21% vs -5% tolerance) while CI lower bound fails.

**Lesson:** Non-inferiority tests use confidence intervals, not point estimates. This is correct behavior.

### 5. Windows Encoding

Emojis in print statements break on Windows (cp1252 encoding).

**Lesson:** Use UTF-8 wrappers or avoid emojis in cross-platform code.

---

## Intentional Omissions

| Feature | Why Omitted |
|---------|-------------|
| **CI/CD pipeline** | Deferred; pytest works locally for educational purposes |
| **Web interface / dashboard** | Out of scope; focus is on library and notebooks |
| **Real-time monitoring** | Different use case; this is batch analysis focused |
| **Database integration** | Adds complexity without pedagogical value |
| **Cluster randomization** | Documented conceptually; implementation requires different data structure |

---

## Industry Inspiration

Techniques and best practices drawn from:

| Company | Technique/Pattern |
|---------|-------------------|
| **Netflix** | CUPED variance reduction (30-40% improvement) |
| **DoorDash** | CUPAC ML-enhanced variance reduction |
| **Spotify** | Risk-aware multi-metric testing, guardrail frameworks |
| **Meta** | Network effects (conceptual), massive-scale practices |
| **Statsig** | Novelty effect detection, sequential testing |
| **Booking.com** | Strict SRM thresholds (alpha=0.001) |
| **Microsoft** | A/A testing infrastructure validation |

---

## Future Considerations

If extending this framework:

1. **Network effects / cluster randomization** - Would need graph-based data
2. **Multi-armed bandits** - Separate module with Thompson Sampling, UCB
3. **Continuous monitoring** - mSPRT, always-valid p-values
4. **Causal inference** - IV/2SLS, regression discontinuity, synthetic control

These are documented conceptually in the README but not implemented.
