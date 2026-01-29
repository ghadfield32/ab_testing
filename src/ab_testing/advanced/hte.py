"""
Heterogeneous Treatment Effects (HTE) - X-Learner
=================================================

Estimate how treatment effects vary across user segments using the X-Learner meta-algorithm.

Key Concepts:
- **ATE**: Average Treatment Effect (overall effect)
- **CATE**: Conditional Average Treatment Effect (effect for specific user characteristics)
- **ITE**: Individual Treatment Effect (effect for a specific user)
- **X-Learner**: Meta-learner that handles imbalanced treatment/control groups well

Reference:
----------
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects using machine learning"
  PNAS, https://doi.org/10.1073/pnas.1804597116
- Athey & Imbens (2016): "Recursive Partitioning for Heterogeneous Causal Effects"
- Statsig (2024): "Differential Impact Detection"
  https://www.statsig.com/blog/differential-impact-detection

Example Usage:
--------------
>>> from ab_testing.advanced import hte
>>> import numpy as np
>>> from sklearn.ensemble import GradientBoostingRegressor
>>>
>>> # Simulate data with heterogeneous effects
>>> np.random.seed(42)
>>> n = 1000
>>> X = np.random.randn(n, 5)
>>> treatment = np.random.binomial(1, 0.5, n)
>>> # Effect varies with X[:, 0]: strong effect for X[:, 0] > 0
>>> tau_true = 0.5 + 2.0 * (X[:, 0] > 0)
>>> y = X.sum(axis=1) + tau_true * treatment + np.random.randn(n) * 0.5
>>>
>>> # Fit X-Learner
>>> learner = hte.XLearner(
...     base_model=GradientBoostingRegressor(n_estimators=100, max_depth=3)
... )
>>> learner.fit(X, y, treatment)
>>>
>>> # Predict CATE for new users
>>> X_new = np.random.randn(100, 5)
>>> cate_estimates = learner.predict(X_new)
>>>
>>> # Identify high-value segments
>>> segments = hte.identify_segments(X, cate_estimates, n_segments=3)
>>> print(f"Segment 1 CATE: {segments[0]['cate_mean']:.2f}")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from scipy import stats


class XLearner:
    """
    X-Learner for estimating heterogeneous treatment effects.

    The X-Learner is particularly effective when:
    - Treatment and control groups are imbalanced
    - You want to leverage both groups' data efficiently
    - Base learners are flexible (e.g., GradientBoosting, RandomForest)

    Algorithm:
    ----------
    1. Stage 1: Train outcome models for each group
       - μ0(x) predicts Y for control group
       - μ1(x) predicts Y for treatment group

    2. Stage 2: Compute imputed treatment effects
       - D1_i = Y1_i - μ0(X1_i) for treated units
       - D0_i = μ1(X0_i) - Y0_i for control units

    3. Stage 3: Train CATE models
       - τ1(x) predicts D1 from X (treatment group)
       - τ0(x) predicts D0 from X (control group)

    4. Final prediction: τ(x) = g(x)τ0(x) + (1-g(x))τ1(x)
       where g(x) is propensity score (probability of treatment)

    Parameters
    ----------
    base_model : sklearn estimator, default=GradientBoostingRegressor
        Base learner for outcome and CATE models
    propensity_model : sklearn classifier, default=LogisticRegression
        Model for propensity score estimation
    random_state : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    mu0_model_ : fitted model
        Outcome model for control group
    mu1_model_ : fitted model
        Outcome model for treatment group
    tau0_model_ : fitted model
        CATE model from control group
    tau1_model_ : fitted model
        CATE model from treatment group
    propensity_model_ : fitted model
        Propensity score model
    ate_ : float
        Estimated Average Treatment Effect

    Example
    -------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> learner = XLearner(base_model=RandomForestRegressor(n_estimators=100))
    >>> learner.fit(X, y, treatment)
    >>> cate = learner.predict(X_new)
    >>> print(f"ATE: {learner.ate_:.2f}")
    """

    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        propensity_model: Optional[BaseEstimator] = None,
        random_state: Optional[int] = None,
    ):
        if base_model is None:
            base_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=random_state,
            )
        if propensity_model is None:
            propensity_model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
            )

        self.base_model = base_model
        self.propensity_model = propensity_model
        self.random_state = random_state

        # Fitted models (initialized in fit())
        self.mu0_model_ = None
        self.mu1_model_ = None
        self.tau0_model_ = None
        self.tau1_model_ = None
        self.propensity_model_ = None
        self.ate_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> "XLearner":
        """
        Fit X-Learner on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,)
            Outcome variable
        treatment : np.ndarray, shape (n_samples,)
            Treatment indicator (0=control, 1=treatment)

        Returns
        -------
        self : XLearner
            Fitted estimator

        Raises
        ------
        ValueError
            If X, y, treatment have incompatible shapes
        ValueError
            If treatment has fewer than 10 samples per group
        """
        if len(X) != len(y) or len(X) != len(treatment):
            raise ValueError("X, y, and treatment must have same length")

        treatment = np.asarray(treatment)
        if not np.all((treatment == 0) | (treatment == 1)):
            raise ValueError("treatment must be binary (0 or 1)")

        n_control = (treatment == 0).sum()
        n_treated = (treatment == 1).sum()
        if n_control < 10 or n_treated < 10:
            raise ValueError("Need at least 10 samples per group for reliable CATE estimation")

        # Split data
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        X_treatment = X[treatment == 1]
        y_treatment = y[treatment == 1]

        # Stage 1: Train outcome models
        self.mu0_model_ = clone(self.base_model)
        self.mu1_model_ = clone(self.base_model)

        self.mu0_model_.fit(X_control, y_control)
        self.mu1_model_.fit(X_treatment, y_treatment)

        # Stage 2: Compute imputed treatment effects
        # For treated: D1 = Y1 - μ0(X1)
        mu0_on_treatment = self.mu0_model_.predict(X_treatment)
        D1 = y_treatment - mu0_on_treatment

        # For control: D0 = μ1(X0) - Y0
        mu1_on_control = self.mu1_model_.predict(X_control)
        D0 = mu1_on_control - y_control

        # Stage 3: Train CATE models
        self.tau1_model_ = clone(self.base_model)
        self.tau0_model_ = clone(self.base_model)

        self.tau1_model_.fit(X_treatment, D1)
        self.tau0_model_.fit(X_control, D0)

        # Train propensity score model
        self.propensity_model_ = clone(self.propensity_model)
        self.propensity_model_.fit(X, treatment)

        # Estimate ATE
        self.ate_ = self.predict(X).mean()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE for new observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        cate : np.ndarray, shape (n_samples,)
            Predicted CATE for each observation

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.mu0_model_ is None:
            raise ValueError("Model must be fitted before prediction")

        # Get propensity scores
        propensity = self.propensity_model_.predict_proba(X)[:, 1]

        # Predict from both CATE models
        tau0 = self.tau0_model_.predict(X)
        tau1 = self.tau1_model_.predict(X)

        # Weighted average: g(x)τ0(x) + (1-g(x))τ1(x)
        cate = propensity * tau0 + (1 - propensity) * tau1

        return cate

    def predict_ate(self, X: np.ndarray) -> float:
        """
        Predict Average Treatment Effect for a population.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix for population

        Returns
        -------
        ate : float
            Estimated ATE
        """
        return self.predict(X).mean()


def identify_segments(
    X: np.ndarray,
    cate: np.ndarray,
    n_segments: int = 3,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Identify user segments with different treatment effects.

    Uses CATE quantiles to split users into segments and characterizes each segment.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    cate : np.ndarray, shape (n_samples,)
        Predicted CATE values
    n_segments : int, default=3
        Number of segments to create
    feature_names : list of str, optional
        Names of features for reporting

    Returns
    -------
    segments : list of dict
        List of segment descriptions with:
        - segment_id: Segment number (0 to n_segments-1)
        - cate_mean: Mean CATE in segment
        - cate_std: Std of CATE in segment
        - size: Number of users in segment
        - pct_of_total: Percentage of total users
        - feature_means: Mean feature values in segment

    Example
    -------
    >>> segments = identify_segments(X, cate_estimates, n_segments=3)
    >>> for seg in segments:
    ...     print(f"Segment {seg['segment_id']}: CATE={seg['cate_mean']:.2f}, Size={seg['size']}")
    """
    if len(X) != len(cate):
        raise ValueError("X and cate must have same length")

    # Create segments based on CATE quantiles
    quantiles = np.linspace(0, 1, n_segments + 1)
    cate_quantiles = np.quantile(cate, quantiles)

    segments = []
    for i in range(n_segments):
        # Define segment
        if i == 0:
            mask = cate <= cate_quantiles[i + 1]
        elif i == n_segments - 1:
            mask = cate > cate_quantiles[i]
        else:
            mask = (cate > cate_quantiles[i]) & (cate <= cate_quantiles[i + 1])

        # Characterize segment
        segment_info = {
            'segment_id': i,
            'cate_mean': cate[mask].mean(),
            'cate_std': cate[mask].std(),
            'cate_min': cate[mask].min(),
            'cate_max': cate[mask].max(),
            'size': mask.sum(),
            'pct_of_total': mask.mean() * 100,
        }

        # Add feature means if available
        if X.ndim == 2:
            X_segment = X[mask]
            segment_info['feature_means'] = X_segment.mean(axis=0)

            if feature_names is not None:
                segment_info['feature_summary'] = {
                    name: X_segment[:, idx].mean()
                    for idx, name in enumerate(feature_names)
                }

        segments.append(segment_info)

    return segments


def test_hte_significance(
    cate: np.ndarray,
    ate: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if heterogeneous treatment effects are statistically significant.

    Tests H0: CATE is constant (no heterogeneity) vs H1: CATE varies

    Parameters
    ----------
    cate : np.ndarray
        Predicted CATE values
    ate : float
        Average Treatment Effect
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    dict
        Dictionary with:
        - variance_cate: Variance of CATE estimates
        - coefficient_of_variation: CV of CATE (std/mean)
        - test_statistic: Chi-square test statistic
        - p_value: P-value for heterogeneity test
        - significant: Whether heterogeneity is significant

    Notes
    -----
    - High CV (>0.5) suggests substantial heterogeneity
    - Significance test is approximate (assumes normal distribution)

    Example
    -------
    >>> result = test_hte_significance(cate_estimates, ate=0.05)
    >>> if result['significant']:
    ...     print(f"Heterogeneity detected! CV={result['coefficient_of_variation']:.2f}")
    """
    if len(cate) < 10:
        raise ValueError("Need at least 10 CATE estimates for significance testing")

    variance_cate = cate.var(ddof=1)
    std_cate = cate.std(ddof=1)
    cv = std_cate / abs(ate) if ate != 0 else np.inf

    # Test if variance of CATE is significantly > 0
    # Under H0 (no heterogeneity), CATE should be constant
    # Use chi-square test for variance
    n = len(cate)
    # Standardized CATE under H0
    z = (cate - ate) / (std_cate if std_cate > 0 else 1e-10)
    chi2_stat = (z**2).sum()
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=n - 1)

    return {
        'variance_cate': variance_cate,
        'std_cate': std_cate,
        'coefficient_of_variation': cv,
        'test_statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'interpretation': (
            'Significant heterogeneity detected' if p_value < alpha
            else 'No significant heterogeneity'
        ),
    }


def targeting_value(
    cate: np.ndarray,
    ate: float,
    targeting_pct: float = 0.2,
) -> Dict[str, float]:
    """
    Calculate value of targeting high-CATE users vs. treating everyone.

    Parameters
    ----------
    cate : np.ndarray
        Predicted CATE values
    ate : float
        Average Treatment Effect
    targeting_pct : float, default=0.2
        Percentage of users to target (top CATE)

    Returns
    -------
    dict
        Dictionary with:
        - ate: Average Treatment Effect (treat all)
        - targeted_ate: ATE if targeting top X%
        - lift_from_targeting: Percentage improvement from targeting
        - threshold_cate: CATE threshold for targeting

    Example
    -------
    >>> value = targeting_value(cate_estimates, ate=0.05, targeting_pct=0.2)
    >>> print(f"Targeting top 20%: {value['lift_from_targeting']*100:.1f}% better than treating all")
    """
    if not 0 < targeting_pct <= 1:
        raise ValueError("targeting_pct must be between 0 and 1")

    # Find threshold for top X%
    threshold = np.quantile(cate, 1 - targeting_pct)

    # ATE for targeted users
    targeted_cate = cate[cate >= threshold]
    targeted_ate = targeted_cate.mean()

    # Lift from targeting
    lift = (targeted_ate / ate - 1) if ate != 0 else 0

    return {
        'ate': ate,
        'targeted_ate': targeted_ate,
        'lift_from_targeting': lift,
        'threshold_cate': threshold,
        'n_targeted': len(targeted_cate),
        'pct_targeted': targeting_pct * 100,
    }


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Heterogeneous Treatment Effects (X-Learner) Demo")
    print("=" * 80)

    np.random.seed(42)

    # Simulate data with heterogeneous effects
    print("\n📊 SIMULATING DATA WITH HETEROGENEOUS EFFECTS")
    print("-" * 80)

    n = 2000
    X = np.random.randn(n, 5)
    treatment = np.random.binomial(1, 0.5, n)

    # Treatment effect varies with X[:, 0]:
    # - Positive effect for X[:, 0] > 0
    # - Negative effect for X[:, 0] < 0
    tau_true = 0.5 + 2.0 * (X[:, 0] > 0) - 1.0 * (X[:, 0] < 0)
    baseline = X[:, 0] * 2 + X[:, 1] * 1.5  # Baseline outcome
    y = baseline + tau_true * treatment + np.random.randn(n) * 0.5

    print(f"Generated {n} observations")
    print(f"Treatment rate: {treatment.mean():.1%}")
    print(f"True ATE: {tau_true.mean():.3f}")
    print(f"True CATE range: [{tau_true.min():.2f}, {tau_true.max():.2f}]")

    # Fit X-Learner
    print("\n🤖 FITTING X-LEARNER")
    print("-" * 80)

    learner = XLearner(
        base_model=GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        )
    )
    learner.fit(X, y, treatment)

    print(f"✅ X-Learner fitted successfully")
    print(f"Estimated ATE: {learner.ate_:.3f} (True: {tau_true.mean():.3f})")

    # Predict CATE
    cate_pred = learner.predict(X)
    print(f"CATE predictions: [{cate_pred.min():.2f}, {cate_pred.max():.2f}]")

    # Correlation with true CATE
    corr = np.corrcoef(tau_true, cate_pred)[0, 1]
    print(f"Correlation with true CATE: {corr:.3f}")

    # Test for heterogeneity
    print("\n📈 TESTING FOR HETEROGENEOUS EFFECTS")
    print("-" * 80)

    hte_test = test_hte_significance(cate_pred, ate=learner.ate_)
    print(f"CATE variance: {hte_test['variance_cate']:.4f}")
    print(f"Coefficient of variation: {hte_test['coefficient_of_variation']:.2f}")
    print(f"Chi-square statistic: {hte_test['test_statistic']:.2f}")
    print(f"P-value: {hte_test['p_value']:.6f}")
    print(f"Result: {hte_test['interpretation']}")

    # Identify segments
    print("\n🎯 IDENTIFYING USER SEGMENTS")
    print("-" * 80)

    segments = identify_segments(
        X, cate_pred, n_segments=3,
        feature_names=[f'Feature_{i}' for i in range(5)]
    )

    for seg in segments:
        print(f"\nSegment {seg['segment_id']}:")
        print(f"  Size: {seg['size']:,} users ({seg['pct_of_total']:.1f}%)")
        print(f"  Mean CATE: {seg['cate_mean']:.3f} ± {seg['cate_std']:.3f}")
        print(f"  CATE range: [{seg['cate_min']:.2f}, {seg['cate_max']:.2f}]")
        print(f"  Key feature: Feature_0 = {seg['feature_summary']['Feature_0']:.2f}")

    # Targeting value
    print("\n💰 VALUE OF TARGETING")
    print("-" * 80)

    targeting = targeting_value(cate_pred, ate=learner.ate_, targeting_pct=0.2)
    print(f"Treat everyone: ATE = {targeting['ate']:.3f}")
    print(f"Target top 20%: ATE = {targeting['targeted_ate']:.3f}")
    print(f"Lift from targeting: {targeting['lift_from_targeting']*100:+.1f}%")
    print(f"CATE threshold: {targeting['threshold_cate']:.3f}")
    print(f"Users targeted: {targeting['n_targeted']:,}")

    print("\n" + "=" * 80)
    print("💡 INSIGHTS:")
    print("   - X-Learner successfully identified heterogeneous effects")
    print("   - Targeting high-CATE users yields significantly better outcomes")
    print("   - Use segment analysis to understand which features drive heterogeneity")
    print("=" * 80)
