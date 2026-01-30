"""
A/B Testing Framework - Production-Quality Experimentation Package
==================================================================

A comprehensive Python package for modern A/B testing, covering fundamentals
through advanced techniques used at Netflix, Meta, Spotify, and DoorDash.

Modules:
--------
- core: Fundamental statistical tests and power analysis
- variance_reduction: CUPED and CUPAC for improved sensitivity
- advanced: Sequential testing, HTE, noncompliance analysis
- diagnostics: Guardrails, novelty detection, A/A tests
- decision: Ship/hold/abandon frameworks
- data: Real-world dataset loaders

Example Usage:
--------------
>>> from ab_testing.data import loaders
>>> from ab_testing.core import power, frequentist
>>>
>>> # Load real marketing A/B test data
>>> df = loaders.load_marketing_ab()
>>>
>>> # Calculate required sample size
>>> n = power.required_samples_binary(p1=0.05, mde=0.10, power=0.80)
>>>
>>> # Run z-test for proportions
>>> result = frequentist.z_test_proportions(x_a=50, n_a=500, x_b=60, n_b=500)

Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

# Expose key functions at package level for convenience
from ab_testing.data import loaders
from ab_testing.core import power, frequentist, bayesian, randomization

__all__ = [
    "loaders",
    "power",
    "frequentist",
    "bayesian",
    "randomization",
]
