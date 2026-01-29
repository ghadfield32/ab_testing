"""
CUPAC: Control Using Predictions As Covariates
===============================================

ML-enhanced variance reduction technique that uses machine learning predictions
from pre-experiment features as covariates for CUPED adjustment.

Reference:
----------
DoorDash Engineering Blog (2020): "Improving Experimental Power through Control
Using Predictions as Covariate (CUPAC)"
https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/

Key Improvements over CUPED:
- Uses multiple features with non-linear relationships
- Typically achieves 25-40% variance reduction (vs 10-25% for CUPED)
- Requires cross-validation to avoid overfitting

Example Usage:
--------------
>>> from ab_testing.variance_reduction import cupac
>>> import numpy as np
>>>
>>> # Multiple pre-experiment features
>>> X = np.random.normal(0, 1, (1000, 5))
>>> y = X @ np.array([2, 1, 0.5, 0.3, 0.1]) + np.random.normal(0, 5, 1000)
>>> treatment = np.random.binomial(1, 0.5, 1000)
>>>
>>> # Apply CUPAC
>>> result = cupac.cupac_ab_test(
...     y[treatment==0], y[treatment==1],
...     X[treatment==0], X[treatment==1]
... )
>>> print(f"Variance reduction: {result['var_reduction']*100:.1f}%")
"""

import numpy as np
from typing import Dict, Optional, Literal
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict


def cupac_adjustment(
    y: np.ndarray,
    X: np.ndarray,
    model: Optional[object] = None,
    model_type: Optional[str] = None,
    cv: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Apply CUPAC variance reduction using ML predictions.

    Uses cross-validated predictions to avoid overfitting, then applies
    CUPED adjustment with predictions as covariate.

    Parameters
    ----------
    y : np.ndarray
        Outcome metric (shape: n,)
    X : np.ndarray
        Pre-experiment features (shape: n × k)
    model : object, optional
        sklearn model with fit/predict methods.
        If provided, overrides model_type.
    model_type : {'gbm', 'rf', 'ridge'}, optional
        Model type to use. Ignored if model is provided.
        Default: 'gbm' (GradientBoostingRegressor)
    cv : int, default=5
        Number of cross-validation folds
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        CUPAC-adjusted outcome values

    Notes
    -----
    - CRITICAL: Must use cross-validated predictions to avoid overfitting
    - Model should predict y from X (not treatment effect)
    - Common models: GradientBoosting, RandomForest, Ridge

    Example
    -------
    >>> X = np.random.normal(0, 1, (1000, 5))
    >>> y = X @ np.array([2, 1, 0.5, 0.3, 0.1]) + np.random.normal(0, 5, 1000)
    >>> y_adj = cupac_adjustment(y, X, random_state=42)
    >>> print(f"Variance reduction: {(1 - y_adj.var()/y.var())*100:.1f}%")
    """
    # Validation
    if len(y) != len(X):
        raise ValueError("y and X must have same number of rows")
    if len(y) < 10:
        raise ValueError("Need at least 10 observations")
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Handle model_type parameter (for backward compatibility)
    if model is None:
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=50,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'gbm' or model_type is None:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=100,
                random_state=random_state
            )
        else:
            raise ValueError(f"model_type must be one of ['gbm', 'rf', 'ridge'], got '{model_type}'")

    # Cross-validated predictions (CRITICAL to avoid overfitting)
    cv_predictions = cross_val_predict(model, X, y, cv=cv)

    # Fit model on full data for future use
    model.fit(X, y)

    # CUPED adjustment using CV predictions as covariate
    theta = np.cov(y, cv_predictions, ddof=1)[0, 1] / cv_predictions.var(ddof=1)
    y_adjusted = y - theta * (cv_predictions - cv_predictions.mean())

    return y_adjusted


def cupac_ab_test(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    X_control: np.ndarray,
    X_treatment: np.ndarray,
    model_type: Literal['gbm', 'rf', 'ridge'] = 'gbm',
    cv: int = 5,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run A/B test with CUPAC variance reduction.

    Parameters
    ----------
    y_control : np.ndarray
        Outcome metric for control group
    y_treatment : np.ndarray
        Outcome metric for treatment group
    X_control : np.ndarray
        Pre-experiment features for control group (n_control × k)
    X_treatment : np.ndarray
        Pre-experiment features for treatment group (n_treatment × k)
    model_type : {'gbm', 'rf', 'ridge'}, default='gbm'
        Model type: GradientBoosting, RandomForest, or Ridge
    cv : int, default=5
        Number of cross-validation folds
    alpha : float, default=0.05
        Significance level
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with keys:
        - model_r2: Model R² on control group
        - pred_correlation: Correlation between y and predictions
        - var_reduction: Variance reduction percentage
        - effect_raw: Raw treatment effect
        - effect_adjusted: CUPAC-adjusted treatment effect
        - se_raw: Raw standard error
        - se_adjusted: CUPAC-adjusted standard error
        - se_reduction: SE reduction percentage
        - ci_raw: Raw 95% CI
        - ci_adjusted: CUPAC-adjusted 95% CI
        - p_value_raw: Raw p-value
        - p_value_adjusted: CUPAC-adjusted p-value
        - sample_size_reduction: Equivalent sample size reduction

    Example
    -------
    >>> np.random.seed(42)
    >>> X_c = np.random.normal(0, 1, (500, 5))
    >>> X_t = np.random.normal(0, 1, (500, 5))
    >>> y_c = X_c @ np.array([2,1,0.5,0.3,0.1]) + np.random.normal(0, 5, 500)
    >>> y_t = X_t @ np.array([2,1,0.5,0.3,0.1]) + np.random.normal(10, 5, 500)
    >>> result = cupac_ab_test(y_c, y_t, X_c, X_t, random_state=42)
    >>> print(f"SE reduction: {result['se_reduction']*100:.1f}%")
    """
    if len(y_control) != len(X_control):
        raise ValueError("y_control and X_control must have same length")
    if len(y_treatment) != len(X_treatment):
        raise ValueError("y_treatment and X_treatment must have same length")
    if X_control.shape[1] != X_treatment.shape[1]:
        raise ValueError("X_control and X_treatment must have same number of features")

    # Choose model
    if model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=100,
            random_state=random_state
        )
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    else:
        raise ValueError("model_type must be 'gbm', 'rf', or 'ridge'")

    # Combine data for model training
    y_all = np.concatenate([y_control, y_treatment])

    # Train on control group only (unbiased)
    # Need to manually replicate cupac_adjustment logic to get model and preds
    cv_preds_control = cross_val_predict(model, X_control, y_control, cv=cv)
    model.fit(X_control, y_control)

    theta = (np.cov(y_control, cv_preds_control, ddof=1)[0, 1] /
             cv_preds_control.var(ddof=1))
    y_control_adj = (y_control -
                     theta * (cv_preds_control - cv_preds_control.mean()))

    # Predict on treatment group and adjust
    preds_treatment = model.predict(X_treatment)
    y_treatment_adj = (y_treatment -
                      theta * (preds_treatment - cv_preds_control.mean()))

    # Model performance
    r2 = 1 - np.sum((y_control - cv_preds_control)**2) / np.sum((y_control - y_control.mean())**2)
    pred_corr = np.corrcoef(y_control, cv_preds_control)[0, 1]

    # Raw analysis
    effect_raw = y_treatment.mean() - y_control.mean()
    se_raw = np.sqrt(y_control.var(ddof=1)/len(y_control) +
                     y_treatment.var(ddof=1)/len(y_treatment))
    t_stat_raw = effect_raw / se_raw
    p_value_raw = 2 * (1 - stats.t.cdf(abs(t_stat_raw),
                       df=len(y_control) + len(y_treatment) - 2))
    ci_raw = (effect_raw - 1.96*se_raw, effect_raw + 1.96*se_raw)

    # CUPAC analysis
    effect_adjusted = y_treatment_adj.mean() - y_control_adj.mean()
    se_adjusted = np.sqrt(y_control_adj.var(ddof=1)/len(y_control_adj) +
                          y_treatment_adj.var(ddof=1)/len(y_treatment_adj))
    t_stat_adj = effect_adjusted / se_adjusted
    p_value_adjusted = 2 * (1 - stats.t.cdf(abs(t_stat_adj),
                            df=len(y_control) + len(y_treatment) - 2))
    ci_adjusted = (effect_adjusted - 1.96*se_adjusted,
                   effect_adjusted + 1.96*se_adjusted)

    # Variance reduction
    var_raw = y_all.var(ddof=1)
    var_adjusted = np.concatenate([y_control_adj, y_treatment_adj]).var(ddof=1)
    var_reduction = 1 - var_adjusted / var_raw

    # SE reduction
    se_reduction = 1 - se_adjusted / se_raw

    # Sample size reduction
    sample_size_reduction = 1 - (se_adjusted / se_raw)**2

    # Determine significance (using adjusted p-value)
    significant = p_value_adjusted < alpha

    return {
        'model_r2': float(r2),
        'pred_correlation': float(pred_corr),
        'var_reduction': float(var_reduction),
        'mean_control': float(y_control.mean()),
        'mean_treatment': float(y_treatment.mean()),
        'difference': float(effect_adjusted),
        'effect_raw': float(effect_raw),
        'effect_adjusted': float(effect_adjusted),
        'se_raw': float(se_raw),
        'se_adjusted': float(se_adjusted),
        'se_adj_diff': float(se_adjusted),
        'se_reduction': float(se_reduction),
        'ci_raw': (float(ci_raw[0]), float(ci_raw[1])),
        'ci_adjusted': (float(ci_adjusted[0]), float(ci_adjusted[1])),
        'p_value': float(p_value_adjusted),
        'p_value_raw': float(p_value_raw),
        'p_value_adjusted': float(p_value_adjusted),
        'significant': bool(significant),
        'sample_size_reduction': float(sample_size_reduction),
        'model_type': model_type,
    }


def variance_reduction_cupac(
    y: np.ndarray,
    X: np.ndarray,
    model_type: str = 'gbm',
    cv: int = 5,
    random_state: Optional[int] = None,
) -> float:
    """
    Calculate variance reduction achieved by CUPAC.

    Parameters
    ----------
    y : np.ndarray
        Outcome metric
    X : np.ndarray
        Pre-experiment features
    model_type : {'gbm', 'rf', 'ridge'}, default='gbm'
        Model type to use
    cv : int, default=5
        Number of cross-validation folds
    random_state : int, optional
        Random seed

    Returns
    -------
    float
        Variance reduction percentage (0-1)

    Example
    -------
    >>> X = np.random.normal(0, 1, (1000, 5))
    >>> y = X @ np.array([2, 1, 0.5, 0.3, 0.1]) + np.random.normal(0, 5, 1000)
    >>> vr = variance_reduction_cupac(y, X, random_state=42)
    >>> print(f"Variance reduction: {vr*100:.1f}%")
    """
    y_adj = cupac_adjustment(
        y, X, model_type=model_type, cv=cv, random_state=random_state
    )
    var_reduction = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    return float(var_reduction)


def power_gain_cupac(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    X_control: np.ndarray,
    X_treatment: np.ndarray,
    model_type: str = 'gbm',
    cv: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate statistical power gain from using CUPAC.

    Parameters
    ----------
    y_control, y_treatment : np.ndarray
        Outcome metrics
    X_control, X_treatment : np.ndarray
        Pre-experiment features
    model_type : {'gbm', 'rf', 'ridge'}, default='gbm'
        Model type
    cv : int, default=5
        Cross-validation folds
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with keys:
        - var_reduction: Variance reduction percentage
        - se_reduction: Standard error reduction percentage
        - power_multiplier: Power gain factor (1/(1-var_reduction))
        - equivalent_sample_size_gain: Sample size reduction percentage

    Example
    -------
    >>> X_c = np.random.normal(0, 1, (500, 5))
    >>> X_t = np.random.normal(0, 1, (500, 5))
    >>> y_c = X_c.sum(axis=1) + np.random.normal(0, 5, 500)
    >>> y_t = X_t.sum(axis=1) + np.random.normal(10, 5, 500)
    >>> pg = power_gain_cupac(y_c, y_t, X_c, X_t, random_state=42)
    >>> print(f"Power multiplier: {pg['power_multiplier']:.2f}x")
    """
    result = cupac_ab_test(
        y_control, y_treatment, X_control, X_treatment,
        model_type=model_type, cv=cv, random_state=random_state
    )

    # Power multiplier: how much more powerful the test becomes
    var_red = result['var_reduction']
    if var_red < 1.0 and var_red > 0:
        power_multiplier = 1 / (1 - var_red)
    else:
        power_multiplier = 1.0

    # Equivalent sample size
    n_actual = len(y_control) + len(y_treatment)
    equivalent_n = n_actual * power_multiplier

    return {
        'var_reduction': result['var_reduction'],
        'se_reduction': result['se_reduction'],
        'power_multiplier': float(power_multiplier),
        'equivalent_n': float(equivalent_n),
        'equivalent_sample_size_gain': result['sample_size_reduction'],
    }


def compare_cuped_cupac(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,  # Single covariate for CUPED
    x_treatment: np.ndarray,
    X_control: np.ndarray,  # Multiple features for CUPAC
    X_treatment: np.ndarray,
    random_state: Optional[int] = None,
) -> Dict[str, any]:
    """
    Compare CUPED vs CUPAC variance reduction on same data.

    Parameters
    ----------
    y_control, y_treatment : np.ndarray
        Outcome metrics
    x_control, x_treatment : np.ndarray
        Single pre-experiment covariate for CUPED
    X_control, X_treatment : np.ndarray
        Multiple features for CUPAC (must include x as one column)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Comparison of CUPED vs CUPAC performance

    Example
    -------
    >>> # Create data where CUPAC should win
    >>> X_c = np.random.normal(0, 1, (500, 5))
    >>> y_c = (X_c**2).sum(axis=1) + np.random.normal(0, 5, 500)  # Non-linear
    >>> result = compare_cuped_cupac(y_c, y_t, X_c[:,0], X_t[:,0], X_c, X_t)
    >>> print(f"CUPAC improvement: {result['cupac_improvement_vs_cuped']:.1f}x")
    """
    from ab_testing.variance_reduction import cuped

    # Run CUPED
    cuped_result = cuped.cuped_ab_test(
        y_control, y_treatment, x_control, x_treatment
    )

    # Run CUPAC
    cupac_result = cupac_ab_test(
        y_control, y_treatment, X_control, X_treatment,
        random_state=random_state
    )

    # Compare
    cupac_improvement = (cupac_result['var_reduction'] /
                        cuped_result['var_reduction'])

    return {
        'cuped_var_reduction': cuped_result['var_reduction'],
        'cupac_var_reduction': cupac_result['var_reduction'],
        'cupac_improvement_vs_cuped': cupac_improvement,
        'cuped_se_reduction': cuped_result['se_reduction'],
        'cupac_se_reduction': cupac_result['se_reduction'],
        'cupac_r2': cupac_result['model_r2'],
    }


def compare_cuped_vs_cupac(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    X_control: np.ndarray,
    X_treatment: np.ndarray,
    random_state: Optional[int] = None,
) -> Dict[str, any]:
    """
    Compare CUPED vs CUPAC variance reduction.

    This simplified version uses the first column of X for CUPED.

    Parameters
    ----------
    y_control, y_treatment : np.ndarray
        Outcome metrics
    X_control, X_treatment : np.ndarray
        Multiple pre-experiment features (n × k)
        First column used for CUPED, all columns for CUPAC
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Comparison with keys:
        - cuped_var_reduction, cupac_var_reduction
        - cuped_se_diff, cupac_se_diff
        - cupac_better (bool)
        - improvement (ratio)

    Example
    -------
    >>> X_c = np.random.normal(0, 1, (500, 5))
    >>> X_t = np.random.normal(0, 1, (500, 5))
    >>> y_c = (X_c**2).sum(axis=1) + np.random.normal(0, 5, 500)
    >>> y_t = (X_t**2).sum(axis=1) + 5 + np.random.normal(0, 5, 500)
    >>> result = compare_cuped_vs_cupac(y_c, y_t, X_c, X_t)
    >>> print(f"CUPAC better: {result['cupac_better']}")
    """
    from ab_testing.variance_reduction import cuped

    # Use first column for CUPED
    x_control = X_control[:, 0]
    x_treatment = X_treatment[:, 0]

    # Run CUPED
    cuped_result = cuped.cuped_ab_test(
        y_control, y_treatment, x_control, x_treatment
    )

    # Run CUPAC
    cupac_result = cupac_ab_test(
        y_control, y_treatment, X_control, X_treatment,
        random_state=random_state
    )

    # Comparison
    cupac_better = cupac_result['var_reduction'] > cuped_result['var_reduction']
    improvement = (cupac_result['var_reduction'] /
                  max(cuped_result['var_reduction'], 1e-10))

    # Recommendation
    if cupac_better and improvement > 1.5:
        recommendation = "Use CUPAC - significantly better variance reduction"
    elif cupac_better:
        recommendation = "Use CUPAC - modestly better variance reduction"
    else:
        recommendation = "Use CUPED - simpler and similar performance"

    return {
        'cuped_var_reduction': cuped_result['var_reduction'],
        'cupac_var_reduction': cupac_result['var_reduction'],
        'cuped_se_diff': cuped_result['se_adjusted'],
        'cupac_se_diff': cupac_result['se_adjusted'],
        'cupac_better': bool(cupac_better),
        'improvement': float(improvement),
        'recommendation': recommendation,
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("CUPAC (ML-Enhanced Variance Reduction) Demo")
    print("=" * 80)

    # Simulate data with non-linear relationships
    np.random.seed(42)
    n_control = 500
    n_treatment = 500

    # Create multiple pre-experiment features
    X_control = np.random.normal(0, 1, (n_control, 5))
    X_treatment = np.random.normal(0, 1, (n_treatment, 5))

    # Outcome with non-linear relationship to features
    weights = np.array([2.0, 1.5, 1.0, 0.5, 0.3])
    y_control = (X_control @ weights + 0.5 * (X_control[:, 0]**2) +
                 np.random.normal(0, 5, n_control))
    y_treatment = (X_treatment @ weights + 0.5 * (X_treatment[:, 0]**2) +
                   np.random.normal(10, 5, n_treatment))  # +10 treatment effect

    # Run CUPAC analysis
    result = cupac_ab_test(
        y_control, y_treatment, X_control, X_treatment,
        model_type='gbm',
        random_state=42
    )

    print("\n📊 CUPAC ANALYSIS RESULTS")
    print("-" * 80)
    print(f"Model R²: {result['model_r2']:.4f}")
    print(f"Prediction correlation: {result['pred_correlation']:.4f}")
    print(f"Variance reduction: {result['var_reduction']*100:.1f}%")
    print(f"SE reduction: {result['se_reduction']*100:.1f}%")

    print("\n┌─────────────────┬──────────────┬──────────────┐")
    print("│                 │ Raw          │ CUPAC        │")
    print("├─────────────────┼──────────────┼──────────────┤")
    print(f"│ Effect          │ {result['effect_raw']:>12.2f} │ {result['effect_adjusted']:>12.2f} │")
    print(f"│ SE              │ {result['se_raw']:>12.2f} │ {result['se_adjusted']:>12.2f} │")
    print(f"│ CI Width        │ {result['ci_raw'][1]-result['ci_raw'][0]:>12.2f} │ {result['ci_adjusted'][1]-result['ci_adjusted'][0]:>12.2f} │")
    print(f"│ P-value         │ {result['p_value_raw']:>12.6f} │ {result['p_value_adjusted']:>12.6f} │")
    print("└─────────────────┴──────────────┴──────────────┘")

    print(f"\n💡 PRACTICAL IMPACT:")
    print(f"   • Could run with {result['sample_size_reduction']*100:.0f}% fewer users for same power")
    print(f"   • Variance reduction of {result['var_reduction']*100:.1f}% (typically 25-40% with CUPAC)")

    # Compare with CUPED
    print("\n📊 CUPED vs CUPAC COMPARISON")
    print("-" * 80)
    from ab_testing.variance_reduction import cuped
    cuped_result = cuped.cuped_ab_test(
        y_control, y_treatment, X_control[:, 0], X_treatment[:, 0]
    )
    print(f"CUPED variance reduction: {cuped_result['var_reduction']*100:.1f}%")
    print(f"CUPAC variance reduction: {result['var_reduction']*100:.1f}%")
    print(f"CUPAC is {result['var_reduction']/cuped_result['var_reduction']:.1f}x better than CUPED")
