# Start Here: A/B Testing Learning Guide

Welcome to the A/B Testing Framework! This guide will help you navigate the repository and find the best learning path for your experience level.

---

## Quick Self-Assessment

Answer these questions to find your recommended starting point:

**1. Have you run an A/B test before?**
- No → Start with [Beginner Path](#beginner-path-new-to-ab-testing)
- Yes → Continue to question 2

**2. Are you comfortable with multiple testing correction (Bonferroni, Benjamini-Hochberg)?**
- No → Start with [Intermediate Path](#intermediate-path-leveling-up)
- Yes → Continue to question 3

**3. Have you used ML-enhanced techniques (CUPAC, X-Learner, HTE)?**
- No → Start with [Advanced Path](#advanced-path-ml-integration)
- Yes → Jump to [Interview Prep](#interview-preparation)

**4. Preparing for data science interviews?**
- Yes → See [Interview Preparation](#interview-preparation)

---

## Learning Paths

### Beginner Path: New to A/B Testing

**Start with:** [03_marketing_complete_workflow.ipynb](notebooks/03_marketing_complete_workflow.ipynb)

**What you'll learn:**
- Complete A/B testing lifecycle from data to decision
- Power analysis and sample size calculation
- SRM (Sample Ratio Mismatch) detection
- Z-test for proportions
- CUPED variance reduction (20-40% faster experiments)
- Guardrail metrics and non-inferiority tests
- Novelty effect detection
- Ship/Hold/Abandon decision framework

**Dataset:** Marketing A/B (588K rows) - Binary conversion outcome

**Estimated time:** 2-3 hours

---

### Intermediate Path: Leveling Up

**Start with:** [01_cookie_cats_product_experimentation.ipynb](notebooks/01_cookie_cats_product_experimentation.ipynb)

**What you'll learn:**
- Testing multiple metrics simultaneously (1-day AND 7-day retention)
- Multiple testing correction (Bonferroni vs Benjamini-Hochberg FDR)
- Ratio metrics with delta method confidence intervals
- Product trade-off decisions (what if metrics conflict?)
- Engagement vs retention balance

**Dataset:** Cookie Cats Mobile Game (90K players)

**Estimated time:** 2-3 hours

**Prerequisites:** Completed beginner path or equivalent experience

---

### Advanced Path: ML Integration

**Start with:** [02_criteo_advanced_techniques.ipynb](notebooks/02_criteo_advanced_techniques.ipynb)

**What you'll learn:**
- CUPAC: ML-enhanced variance reduction (30-60% faster experiments)
- X-Learner: Heterogeneous treatment effects (CATE estimation)
- Sequential testing with O'Brien-Fleming boundaries (early stopping)
- Large-scale best practices (13.9M rows)
- Personalization and targeting decisions

**Dataset:** Criteo Uplift (13.9M rows, 11 user features)

**Estimated time:** 3-4 hours

**Prerequisites:** Completed intermediate path or equivalent experience

---

### Interview Preparation

**Reference:** [04_interview_guide.ipynb](notebooks/04_interview_guide.ipynb)

**What you'll find:**
- Code templates for common interview questions
- Decision frameworks explained
- Common pitfalls and how to avoid them
- Quick reference for function signatures
- Practice scenarios

**Best approach:** Work through all 3 notebooks first, then use this as a reference guide.

---

## Recommended Weekly Schedule

| Week | Focus | Notebook | Key Takeaway |
|------|-------|----------|--------------|
| 1-2 | Foundations | Marketing (03) | Complete lifecycle, CUPED, guardrails |
| 3-4 | Multiple Metrics | Cookie Cats (01) | Multiple testing correction, ratio metrics |
| 5-6 | ML Techniques | Criteo (02) | CUPAC, X-Learner, sequential testing |
| 7-8 | Interview Prep | Interview Guide (04) | Practice explaining, articulate trade-offs |

---

## Quick Start Commands

```bash
# 1. Clone the repository
git clone https://github.com/ghadfield32/ab_testing.git
cd ab_testing

# 2. Install dependencies (using uv package manager)
pip install uv
uv sync

# 3. Download datasets
uv run python setup_datasets.py --download

# 4. Verify datasets are ready
uv run python setup_datasets.py --verify

# 5. Start Jupyter and open your first notebook
uv run jupyter notebook notebooks/03_marketing_complete_workflow.ipynb
```

---

## Notebook Overview

| # | Notebook | Level | Dataset | Key Topics |
|---|----------|-------|---------|------------|
| 03 | [Marketing Complete Workflow](notebooks/03_marketing_complete_workflow.ipynb) | Beginner | Marketing A/B (588K) | Full lifecycle, CUPED, guardrails, novelty |
| 01 | [Cookie Cats Product Experimentation](notebooks/01_cookie_cats_product_experimentation.ipynb) | Intermediate | Cookie Cats (90K) | Multiple testing, ratio metrics, trade-offs |
| 02 | [Criteo Advanced Techniques](notebooks/02_criteo_advanced_techniques.ipynb) | Advanced | Criteo (13.9M) | CUPAC, X-Learner, sequential testing |
| 04 | [Interview Guide](notebooks/04_interview_guide.ipynb) | All Levels | Reference | Templates, pitfalls, practice |

---

## Module Reference

The Python package is organized into logical modules:

```
ab_testing/
├── core/           # Fundamentals: power, frequentist, bayesian, randomization
├── variance_reduction/  # CUPED, CUPAC
├── advanced/       # HTE, sequential, multiple_testing, noncompliance, ratio_metrics
├── diagnostics/    # aa_tests, guardrails, novelty
├── decision/       # framework, business_impact
├── data/           # Dataset loaders
└── pipelines/      # End-to-end demonstrations
```

---

## Need Help?

- **README.md** - Comprehensive documentation with API reference
- **DECISIONS.md** - Why we made specific design choices
- **PROGRESS_LOG.md** - Development timeline and known issues
- **GitHub Issues** - Report bugs or ask questions

---

## Next Steps

1. Run the Quick Start commands above
2. Open your first notebook based on your experience level
3. Work through the exercises interactively
4. Practice explaining your analysis decisions out loud (interview prep!)

Happy experimenting!
