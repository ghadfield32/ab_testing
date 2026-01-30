# A/B Testing Project Development Log

## Overview
This log tracks development progress, issues, and decisions for the A/B Testing project.

---

## 2026-01-29: Notebook Debug and Alignment Session

### Issues Identified

| Notebook | Error | Root Cause | Status |
|----------|-------|------------|--------|
| 01_cookie_cats | `NameError: n_control` | Variable not defined before SRM check; wrong param names | Fixed |
| 02_criteo | `AttributeError: load_criteo` | Function name mismatch: should be `load_criteo_uplift` | Fixed |
| 03_marketing | `AttributeError: load_marketing` | Function name mismatch: should be `load_marketing_ab` | Fixed |
| 04_interview | SRM param names | Wrong param names in code template | Fixed |

### Root Cause Analysis

1. **Loader Function Naming**: Notebooks used shorthand names (`load_criteo`, `load_marketing`) but actual functions have suffixes (`load_criteo_uplift`, `load_marketing_ab`)

2. **SRM Check Parameters**: The `randomization.srm_check()` function signature uses `n_control`/`n_treatment` but notebooks used `observed_control`/`observed_treatment`

3. **Expected Ratio Format**: `expected_ratio` parameter should be a list `[0.5, 0.5]` not a float `0.5`

### Fixes Applied

- **Notebook 01 (cell-10)**: Fixed SRM check to use correct param names `n_control`/`n_treatment`; fixed `expected_ratio=[0.5, 0.5]`
- **Notebook 02 (cell-6)**: Changed `loaders.load_criteo()` to `loaders.load_criteo_uplift()`
- **Notebook 02 (cell-10)**: Fixed SRM check param names; fixed `expected_ratio` to use list format `[0.85, 0.15]`
- **Notebook 03 (cell-6)**: Changed `loaders.load_marketing()` to `loaders.load_marketing_ab()`
- **Notebook 03 (cell-12)**: Fixed SRM check param names; fixed `expected_ratio=[0.5, 0.5]`
- **Notebook 04 (cell-4)**: Fixed SRM code template to use correct param names and access `srm_severe`/`srm_warning` keys

### Available Loader Functions (Reference)

| Function | Dataset | Size |
|----------|---------|------|
| `loaders.load_criteo_uplift(sample_frac=0.01)` | Criteo Uplift | 13.9M rows |
| `loaders.load_marketing_ab(sample_frac=1.0)` | Marketing A/B | 588K rows |
| `loaders.load_cookie_cats(sample_frac=1.0)` | Cookie Cats | 90K rows |

### SRM Check Signature (Reference)

```python
randomization.srm_check(
    n_control=int,          # Observed control count
    n_treatment=int,        # Observed treatment count
    expected_ratio=[0.5, 0.5],  # Expected ratio as list
    alpha=0.01,             # Significance level
    pp_threshold=0.01,      # Practical threshold (1pp)
    count_threshold=None    # Optional count threshold
)
```

---

## Pipeline Structure

### Core Modules
- `ab_testing.core.power` - Power analysis and sample size
- `ab_testing.core.frequentist` - Z-tests, t-tests
- `ab_testing.core.bayesian` - Bayesian A/B tests
- `ab_testing.core.randomization` - SRM checks

### Variance Reduction
- `ab_testing.variance_reduction.cuped` - CUPED adjustment
- `ab_testing.variance_reduction.cupac` - ML-enhanced CUPAC

### Advanced
- `ab_testing.advanced.hte` - Heterogeneous treatment effects
- `ab_testing.advanced.multiple_testing` - FDR/FWER control
- `ab_testing.advanced.sequential` - O'Brien-Fleming boundaries

### Diagnostics
- `ab_testing.diagnostics.guardrails` - Non-inferiority tests
- `ab_testing.diagnostics.novelty` - Novelty effect detection

---

## 2026-01-29: Second Debug Session - Function Signatures

### Issues Identified (Round 2)

| Notebook | Error | Root Cause | Status |
|----------|-------|------------|--------|
| 01_cookie_cats | `KeyError: 'actual_ratio'` | SRM result uses `ratio_control`/`max_pp_deviation` not `actual_ratio` | Fixed |
| 02_criteo | `TypeError: z_test_proportions() got unexpected 'control'` | Function expects `x_control`/`n_control` (counts), not arrays | Fixed |
| 03_marketing | `TypeError: power_analysis_summary() got unexpected 'n_control'` | Function expects `p_baseline`/`mde`, not sample sizes | Fixed |

### Root Cause Analysis (Round 2)

1. **SRM Result Keys**: The `srm_check()` returns `ratio_control`, `max_pp_deviation`, `srm_severe`, `srm_warning` - not `actual_ratio`/`expected_ratio`

2. **z_test_proportions Signature**: Function expects counts (x_control, n_control, x_treatment, n_treatment), NOT arrays. Notebooks were passing arrays.

3. **power_analysis_summary Signature**: Function calculates required sample size given `p_baseline` and `mde`. Notebooks wanted MDE given sample size (inverse problem).

4. **Result Key Naming**: z_test_proportions returns `p_control`/`p_treatment`/`absolute_lift`, not `mean_control`/`mean_treatment`/`difference`

### Fixes Applied (Round 2)

**Notebook 01 (Cookie Cats)**:
- cell-11: Fixed to use `srm_result['max_pp_deviation']`, `srm_result['srm_severe']`, `srm_result['srm_warning']`
- cell-15: Convert arrays to counts for z_test_proportions; use `p_control`/`absolute_lift` keys
- cell-18: Same fix for 7-day retention test
- cell-27, cell-29: Updated result key references

**Notebook 02 (Criteo)**:
- cell-13: Convert arrays to counts; use correct result keys (`p_control`, `absolute_lift`)
- cell-17, cell-28: Updated references to use correct keys

**Notebook 03 (Marketing)**:
- cell-15: Replaced `power_analysis_summary` with custom `find_mde()` function using binary search
- cell-18: Convert arrays to counts; use correct result keys
- cell-19, cell-27: Updated result key references (`p_control`/`p_treatment` instead of `mean_*`)

### Function Signatures (Reference)

**z_test_proportions** (for binary outcomes):
```python
frequentist.z_test_proportions(
    x_control=int,      # Number of successes in control
    n_control=int,      # Total control sample size
    x_treatment=int,    # Number of successes in treatment
    n_treatment=int,    # Total treatment sample size
    alpha=0.05
)
# Returns: p_control, p_treatment, absolute_lift, relative_lift, ci_lower, ci_upper, p_value, significant
```

**power_analysis_summary** (calculates required N):
```python
power.power_analysis_summary(
    p_baseline=float,   # Baseline conversion rate
    mde=float,          # Relative MDE (e.g., 0.10 for 10%)
    alpha=0.05,
    power=0.80
)
# Returns: sample_per_group, sample_total, mde_absolute, cohens_h, interpretation
```

---

## 2026-01-29: Third Debug Session - More Result Key Mismatches

### Issues Identified (Round 3)

| Notebook | Error | Root Cause | Status |
|----------|-------|------------|--------|
| 01_cookie_cats | `KeyError: 'reject_null'` | `benjamini_hochberg()` returns `significant`, not `reject_null` | Fixed |
| 01_cookie_cats | `KeyError: 'n_discoveries'` | Should be `n_significant` | Fixed |
| 02_criteo | `TypeError: cupac_ab_test() got unexpected 'y'` | Function expects pre-split data, not combined arrays | Fixed |

### Root Cause Analysis (Round 3)

1. **benjamini_hochberg Result Keys**: Returns `significant` and `n_significant`, not `reject_null` and `n_discoveries`

2. **cupac_ab_test Signature**: Function expects data already split by treatment group:
   - `y_control`, `y_treatment` (outcomes per group)
   - `X_control`, `X_treatment` (features per group)
   - NOT combined `y`, `treatment`, `X` arrays

### Fixes Applied (Round 3)

**Notebook 01 (Cookie Cats)**:
- cell-19: Changed `bh_result['reject_null']` → `bh_result['significant']`
- cell-19: Changed `bh_result['n_discoveries']` → `bh_result['n_significant']`

**Notebook 02 (Criteo)**:
- cell-16: Split data by treatment group before calling `cupac_ab_test()`
- cell-16: Use correct result keys (`effect_adjusted`, `se_adjusted`, `ci_adjusted`)
- cell-17: Changed `cupac_result['se']` → `cupac_result['se_adjusted']`
- cell-31: Changed `cupac_result['ate']` → `cupac_result['effect_adjusted']`

### Function Signatures (Reference)

**benjamini_hochberg** (FDR correction):
```python
multiple_testing.benjamini_hochberg(
    p_values=list,    # List of p-values
    alpha=0.05        # FDR level
)
# Returns: adjusted_p_values, significant, n_significant, alpha, fdr_threshold
```

**cupac_ab_test** (ML-enhanced variance reduction):
```python
cupac.cupac_ab_test(
    y_control=np.ndarray,      # Outcome for control group
    y_treatment=np.ndarray,    # Outcome for treatment group
    X_control=np.ndarray,      # Features for control (n_control × k)
    X_treatment=np.ndarray,    # Features for treatment (n_treatment × k)
    model_type='gbm',          # 'gbm', 'rf', or 'ridge'
    cv=5,                      # Cross-validation folds
    alpha=0.05,
    random_state=None
)
# Returns: effect_adjusted, se_adjusted, ci_adjusted, p_value, significant,
#          var_reduction, model_r2, se_reduction, sample_size_reduction
```

---

## 2026-01-29: Fourth Debug Session - Ratio Metrics and Variable Scope

### Issues Identified (Round 4)

| Notebook | Error | Root Cause | Status |
|----------|-------|------------|--------|
| 01_cookie_cats | `TypeError: ratio_metric_test() got unexpected 'control_num'` | Wrong param names; should be `numerator_control` etc. | Fixed |
| 02_criteo | `NameError: name 'X' is not defined` | Cell-16 fix removed combined variables that X-Learner needs | Fixed |

### Root Cause Analysis (Round 4)

1. **ratio_metric_test Parameter Names**: Function expects `numerator_control`, `denominator_control`, `numerator_treatment`, `denominator_treatment` - not shorthand names.

2. **Variable Scope Issue**: When cell-16 was fixed for CUPAC, combined variables (`X`, `y`, `treatment`) were removed. X-Learner (cell-20) still needs these combined arrays.

### Fixes Applied (Round 4)

**Notebook 01 (Cookie Cats)**:
- cell-22: Fixed param names for `ratio_metric_test()`
- cell-22: Fixed result key references (`ratio_control` instead of `mean_control`, `ratio_diff` instead of `difference`, `relative_lift` instead of `relative_change`)

**Notebook 02 (Criteo)**:
- cell-16: Restored combined variables (`X`, `y`, `treatment`) in addition to split variables for CUPAC

### Function Signatures (Reference)

**ratio_metric_test** (delta method for ratio metrics):
```python
ratio_metrics.ratio_metric_test(
    numerator_control=np.ndarray,    # e.g., revenue or clicks
    denominator_control=np.ndarray,  # e.g., users or impressions
    numerator_treatment=np.ndarray,
    denominator_treatment=np.ndarray,
    alpha=0.05
)
# Returns: ratio_control, ratio_treatment, ratio_diff, relative_lift,
#          se_control, se_treatment, se_diff, z_statistic, p_value,
#          ci_lower, ci_upper, significant
```

**XLearner.fit** (heterogeneous treatment effects):
```python
xlearner.fit(
    X=np.ndarray,         # Combined feature matrix (n_samples × n_features)
    y=np.ndarray,         # Combined outcome (n_samples,)
    treatment=np.ndarray  # Treatment indicator 0/1 (n_samples,)
)
# Note: Requires COMBINED data, not pre-split by treatment group
```

---

## 2026-01-29: Fifth Debug Session - Sequential Testing Parameter

### Issues Identified (Round 5)

| Notebook | Error | Root Cause | Status |
|----------|-------|------------|--------|
| 02_criteo | `TypeError: obrien_fleming_boundaries() got unexpected 'one_sided'` | Parameter named `two_sided`, not `one_sided` | Fixed |

### Root Cause Analysis (Round 5)

1. **Parameter Naming Convention**: Function uses `two_sided=True` (positive framing) rather than `one_sided=False` (negative framing). Both express same logic, but parameter name differs.

### Fixes Applied (Round 5)

**Notebook 02 (Criteo)**:
- cell-26: Changed `one_sided=False` → `two_sided=True`

### Function Signatures (Reference)

**obrien_fleming_boundaries** (sequential testing):
```python
sequential.obrien_fleming_boundaries(
    n_looks=int,        # Number of planned interim analyses
    alpha=0.05,         # Overall significance level
    two_sided=True      # Whether test is two-sided (NOT one_sided)
)
# Returns: boundaries, n_looks, alpha, two_sided (NOT z_boundaries)
```

---

## 2026-01-29: Sixth Debug Session - Sequential Testing Return Keys

### Issues Identified (Round 6)

| Notebook | Error | Root Cause | Status |
|----------|-------|------------|--------|
| 02_criteo | `KeyError: 'z_boundaries'` | Return key is `boundaries`, not `z_boundaries` | Fixed |

### Root Cause Analysis (Round 6)

1. **Return Key Naming**: Function returns `'boundaries'` but notebook used `'z_boundaries'`. Three cells affected (26, 27, 28).

### Fixes Applied (Round 6)

**Notebook 02 (Criteo)**:
- cell-26: Changed `boundaries['z_boundaries'][i]` → `boundaries['boundaries'][i]`
- cell-27: Changed `boundaries['z_boundaries']` → `boundaries['boundaries']`
- cell-28: Changed `boundaries['z_boundaries'][look-1]` → `boundaries['boundaries'][look-1]`

---

## Future Work

- [ ] Add end-to-end notebook validation tests
- [ ] Consider adding aliases for loader functions (e.g., `load_criteo` -> `load_criteo_uplift`)
- [ ] Add parameter validation warnings in notebooks

---

## Notes

- All notebooks in root directory are the primary versions
- Notebooks in `notebooks/` folder are archived/backup copies
- Dataset files expected in `data/raw/{dataset_name}/`
