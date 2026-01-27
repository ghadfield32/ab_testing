# A/B Testing Curriculum

A guided walk-through of the production-grade experimentation playbook that lives inside `src/ab_testing_framework.py`. Every technique is demonstrated on a realistic simulated onboarding experiment, along with the reasoning, formulas, and interpretation that would be expected in an analytics, data science, or experimentation interview.

## Concepts explored (what we learned and tried)

### Data generation & experiment context
- `DataGenerator.subscription_experiment` produces a 100k-user dataset with pre-experiment predictors (engagement, revenue, sessions), user segments, conversion and revenue outcomes, triggered vs. untriggered flags, and supplemental metrics such as sessions, pages viewed, retention, and NPS. This lets us validate variance-reduction techniques, multiple testing corrections, and heterogeneous treatment effect (HTE) tooling on data that mirrors real product launches.
- The scenario models a subscription onboarding redesign (A = control, B = personalized treatment) so every downstream calculation has business meaning when discussing conversion lift, revenue impact, or annualized impact.

### Level 1 – Fundamentals
- **Sample-size / power analysis**: Binary and continuous formulas (Cohen's h / d) establish how many users per group are needed for a target MDE and 80% power; best-practice interview talk tracks (α, β, one- vs two-sided) are included.
- **Z-test for proportions**: Implements the pooled z-test, displays lift in absolute/relative terms, and interprets p-value, lift, and confidence intervals while clarifying that the p-value is not the probability the treatment works.
- **Welch's t-test**: Applied to first-order revenue metrics (converters only) with unequal variance handling, effect size (Cohen's d), and robust confidence intervals.
- **SRM (Sample Ratio Mismatch)**: Chi-square test ensures randomization sanity before inspecting outcomes and lists common SRM causes so issues can be debugged early.

### Level 2 – Intermediate experimentation practices
- **Bayesian A/B testing**: Beta-Binomial posteriors show `P(B > A)`, credible intervals, and expected loss so decisions can be framed probabilistically without NHT misuse.
- **Multiple testing correction**: Raw vs. Bonferroni vs. Benjamini-Hochberg results briefly demonstrate false-positive inflation when five metrics are inspected.
- **CUPED (variance reduction)**: Pre-experiment revenue is used as a covariate to shrink variance, with tables showing SE/CI improvements and the practical takeaways on users saved.
- **Bootstrap confidence intervals**: Nonparametric percentile CIs for skewed revenue highlight why bootstrapping is necessary for heavy-tailed metrics.
- **Mann-Whitney U**: Rank-based test for skewed revenue data, including effect-size interpretation via the rank-biserial correlation.
- **Intent-to-Treat vs. Per-Protocol**: Triggered flags allow us to contrast ITT impact with the effect seen by only those exposed to treatment so implementation gaps can surface.

### Level 3 – Advanced differentiation
- **CUPAC (ML-enhanced CUPED)**: Gradient boosting predictions serve as covariates, showing substantially more variance reduction than CUPED and justifying the extra modeling overhead.
- **Sequential testing**: O’Brien-Fleming boundary shows how Type I error grows with repeated peeks and why corrected boundaries keep α at 5%.
- **X-Learner (HTE)**: Two-stage machine learning learner estimates individual treatment effects, reports ATE/ATT/ATC, and flags whether heterogeneity justifies targeting.
- **Quantile treatment effects**: Revenue percentiles show if the treatment benefits only certain spenders, which is important when means mask distributional shifts.
- **Delta method**: Ratio metrics (e.g., revenue per user or CTR) get correct variance estimates, confidence intervals, and hypothesis tests without naïve approximations.
- **Winsorization**: 1%/99% trims on converters explain how to tame outliers while keeping sample size intact and compare raw vs. winsorized SEs.

### Level 4 – Production mindset
- **Decision framework**: A simple decision matrix (significant/positive) forms the backbone of daily experimentation reviews—clearly communicating when to ship, hold, or abandon.
- **Business-impact translation**: Annualized lift × LTV scales conversions to revenue so executives can align on why the test mattered and what to expect if rolled out.

## Getting started
1. `git clone https://github.com/RamiKrispin/ab_testing.git && cd ab_testing`
2. Make sure you are on Python 3.11–3.13 (per `pyproject.toml`) and install [uv](https://uv.dev) if you haven't yet: `pip install uv`.
3. Run `uv sync` to install every dependency listed in `uv.lock` (this mirrors what the framework expects). You can now run `uv run python src/ab_testing_framework.py` or open `notebooks/ab_testing.ipynb` with `uv run jupyter lab` for the interactive walkthrough.
4. Re-run `uv sync` after pulling updates to keep the lockfile and virtual environment aligned.

## Repository layout
```
.
├── README.md                 # this guide
├── pyproject.toml            # metadata + dependency pins
├── uv.lock                   # uv-managed lockfile
├── src/
│   └── ab_testing_framework.py  # curriculum script (levels 1–4)
├── notebooks/
│   └── ab_testing.ipynb      # exploratory walkthrough of the same concepts
└── data/                     # placeholder/data saved from simulations (currently empty)
```

## Next steps
- Run the framework, explore the richer explanations in `src/ab_testing_framework.py`, and use the notebook for iterative experimentation.
- Add more modules under `src/` if you want to split the curriculum into re-usable pieces (e.g., `variance_reduction.py`).
- Capture real experiment data in `data/` for cross-validation against the synthetic generator.
