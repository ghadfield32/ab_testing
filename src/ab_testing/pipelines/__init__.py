"""
Pipeline demonstrations for each dataset.

This module provides end-to-end pipeline demonstrations showing how to:
1. Load real-world datasets
2. Perform data validation and quality checks
3. Run comprehensive A/B test analysis
4. Apply variance reduction techniques
5. Check guardrail metrics
6. Make ship/hold/abandon decisions

Available pipelines:
- criteo_pipeline: Analysis of Criteo Uplift dataset (13.9M observations)
- marketing_pipeline: Analysis of Marketing A/B test dataset (588K observations)
- cookie_cats_pipeline: Analysis of Cookie Cats mobile game dataset (90K observations)
"""

# Lazy imports to avoid RuntimeWarning when running pipelines with python -m
# __getattr__ allows module-level attribute access to trigger imports on-demand
# This prevents eagerly importing all pipeline modules when package is imported

__all__ = [
    'run_criteo_analysis',
    'run_marketing_analysis',
    'run_cookie_cats_analysis',
]


def __getattr__(name: str):
    """
    Lazy import pipeline functions on first access.

    This prevents side effects from importing all pipeline modules when
    the package is first loaded. Resolves RuntimeWarning from python -m execution.

    Parameters
    ----------
    name : str
        The attribute/function name being accessed

    Returns
    -------
    Any
        The imported function or module attribute

    Raises
    ------
    AttributeError
        If the requested attribute doesn't exist
    """
    if name == 'run_criteo_analysis':
        from ab_testing.pipelines.criteo_pipeline import run_criteo_analysis
        return run_criteo_analysis
    elif name == 'run_marketing_analysis':
        from ab_testing.pipelines.marketing_pipeline import run_marketing_analysis
        return run_marketing_analysis
    elif name == 'run_cookie_cats_analysis':
        from ab_testing.pipelines.cookie_cats_pipeline import run_cookie_cats_analysis
        return run_cookie_cats_analysis
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
