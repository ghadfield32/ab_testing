"""
Data Loading Utilities for Real-World A/B Test Datasets
=======================================================

This module provides functions to load and preprocess publicly available
A/B testing datasets for learning and analysis.

Datasets:
---------
1. Criteo Uplift Modeling (13.9M rows)
   - Source: Criteo AI Lab / scikit-uplift
   - Use: Large-scale uplift modeling, CATE, HTE

2. Marketing A/B Testing (588K rows)
   - Source: Kaggle (faviovaz)
   - Use: Ad effectiveness, novelty detection

3. Cookie Cats (90K rows) - Optional
   - Source: Kaggle
   - Use: Product/growth experiments, retention

Example Usage:
--------------
>>> from ab_testing.data import loaders
>>>
>>> # Load Criteo dataset (1% sample for speed)
>>> df_criteo = loaders.load_criteo_uplift(sample_frac=0.01)
>>>
>>> # Load Marketing A/B test data
>>> df_marketing = loaders.load_marketing_ab()
>>>
>>> # Get dataset metadata
>>> info = loaders.get_dataset_info('criteo_uplift')
>>> print(info['description'])
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

import pandas as pd
import numpy as np


# Dataset metadata registry
DATASETS = {
    "criteo_uplift": {
        "name": "Criteo Uplift Modeling Dataset",
        "source_url": "https://ailab.criteo.com/criteo-uplift-prediction-dataset/",
        "size": 13979592,
        "description": "Industry-standard benchmark for uplift modeling with real advertising data",
        "features": ["11 anonymized features", "treatment", "visit", "conversion"],
        "citation": "Diemert et al., 'A Large Scale Benchmark for Uplift Modeling', AdKDD 2018",
    },
    "marketing_ab": {
        "name": "Marketing A/B Testing Dataset",
        "source_url": "https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing",
        "size": 588101,
        "description": "Digital advertising A/B test comparing ads vs. PSA",
        "features": ["user_id", "test_group", "converted", "total_ads", "temporal"],
        "citation": "faviovaz (2022), Kaggle Marketing A/B Testing Dataset",
    },
    "cookie_cats": {
        "name": "Cookie Cats Mobile Game A/B Test",
        "source_url": "https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats",
        "size": 90189,
        "description": "Mobile game retention experiment testing gate placement",
        "features": ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"],
        "citation": "DataCamp / Kaggle Cookie Cats Dataset",
    },
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get metadata about available datasets.

    Parameters
    ----------
    dataset_name : str
        One of: 'criteo_uplift', 'marketing_ab', 'cookie_cats'

    Returns
    -------
    dict
        Dataset metadata including source URL, size, citation

    Example
    -------
    >>> info = get_dataset_info('criteo_uplift')
    >>> print(info['citation'])
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name]


def load_criteo_uplift(
    sample_frac: Optional[float] = None,
    cache_dir: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load Criteo uplift modeling dataset.

    This dataset is industry-standard for uplift modeling research, containing
    13.9M observations with 11 user features, treatment assignment, and two
    outcomes (visit and conversion).

    Parameters
    ----------
    sample_frac : float, optional
        Fraction of data to load (0.0-1.0). Recommended for development:
        - Quick testing: 0.01 (140K rows)
        - Development: 0.10 (1.4M rows)
        - Full analysis: 1.0 or None (13.9M rows)
    cache_dir : str, optional
        Directory to cache downloaded data. Defaults to ~/.cache/ab_testing/criteo
    random_state : int, default=42
        Random seed for reproducible sampling

    Returns
    -------
    pd.DataFrame
        Criteo dataset with columns:
        - f0-f10: User features (11 features)
        - treatment: Binary treatment indicator (0=control, 1=treatment)
        - visit: Binary outcome (0=no visit, 1=visit)
        - conversion: Binary outcome (0=no conversion, 1=conversion)

    Raises
    ------
    ImportError
        If scikit-uplift is not installed
    RuntimeError
        If dataset download or processing fails

    Examples
    --------
    >>> # Load 1% sample for quick testing
    >>> df = load_criteo_uplift(sample_frac=0.01)
    >>> print(f"Loaded {len(df):,} rows")
    Loaded 139,796 rows

    >>> # Load full dataset
    >>> df = load_criteo_uplift()
    >>> print(df.columns.tolist())
    ['f0', 'f1', ..., 'f10', 'treatment', 'visit', 'conversion']

    References
    ----------
    Dataset: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
    Paper: "A Large Scale Benchmark for Uplift Modeling" (2020)
    """
    try:
        from sklift.datasets import fetch_criteo
    except ImportError:
        raise ImportError(
            "scikit-uplift is not installed. Install with:\n"
            "    pip install scikit-uplift\n"
            "or:\n"
            "    uv pip install scikit-uplift"
        )

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ab_testing" / "criteo"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # fetch_criteo returns a Bunch object (sklearn-style data container)
        # We need to fetch twice: once for visit outcome, once for conversion

        # Fetch with visit as target
        bunch_visit = fetch_criteo(target_col="visit", treatment_col="treatment")

        # Reconstruct DataFrame from Bunch components
        # bunch.data is a numpy array with feature columns (f0-f10)
        df = pd.DataFrame(bunch_visit.data)

        # Add target outcome (visit) from Bunch
        df['visit'] = bunch_visit.target.values

        # Add treatment indicator from Bunch
        df['treatment'] = bunch_visit.treatment.values

        # Fetch conversion target separately
        bunch_conversion = fetch_criteo(target_col="conversion", treatment_col="treatment")
        df['conversion'] = bunch_conversion.target.values

        # Apply sampling if requested (before return, after full data load)
        if sample_frac is not None and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=random_state)

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed to load Criteo dataset: {str(e)}\n\n"
            "Troubleshooting:\n"
            "1. Check internet connection (dataset downloads from remote)\n"
            "2. Verify disk space (~500MB for cached data)\n"
            "3. Try using sample_frac=0.01 for quick testing\n"
            "4. Check scikit-uplift installation: pip show scikit-uplift"
        )


def load_marketing_ab(
    sample_frac: float = 1.0,
    random_state: int = 42,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load Marketing A/B Testing dataset (ad effectiveness).

    This dataset contains 588K observations from a digital advertising experiment
    comparing ads vs. public service announcements (PSA).

    Parameters
    ----------
    sample_frac : float, default=1.0
        Fraction of data to load (0.0-1.0). Use smaller values for faster testing.
    random_state : int, default=42
        Random seed for reproducible sampling when sample_frac < 1.0
    cache_dir : str, optional
        Directory containing the CSV file. Default: './data/raw/marketing_ab'
        Expected file: marketing_ab_testing.csv

    Returns
    -------
    pd.DataFrame
        Marketing A/B test data with columns:
        - user_id: Unique user identifier
        - test_group: 'ad' or 'psa'
        - converted: 1 if user converted, 0 otherwise
        - total_ads: Number of ads shown to user
        - most_ads_day: Day of week with most ad impressions
        - most_ads_hour: Hour of day with most ad impressions

    Notes
    -----
    - Size: ~19MB
    - Download from: https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing
    - Place in: ./data/raw/marketing_ab/marketing_ab_testing.csv

    **⚠️ DATA QUALITY WARNING**:
    This dataset has a **severe sample ratio mismatch (SRM)** in the original data:
    - Expected: 50% ad / 50% psa (50/50 randomization)
    - Actual: 96% ad / 4% psa (564,577 ad / 23,524 psa)
    - Chi-square test: χ² = 497,768.83, p < 0.0000000001

    **Interpretation**: This is likely **observational data** (users self-selected into groups)
    rather than a properly randomized A/B test. The severe imbalance indicates:
    1. NOT a randomization failure (would expect ~50/50 with small deviation)
    2. Likely convenience sampling or natural exposure patterns
    3. Results should NOT be interpreted causally (confounding present)

    **Educational Value**:
    - Perfect example of detecting SRM in real-world data
    - Demonstrates why SRM checks are critical (this would fail in production)
    - Illustrates difference between experimental and observational data

    **Recommendation**:
    - Use this dataset to practice SRM detection and power analysis
    - Do NOT interpret treatment effects as causal (confounding likely)
    - In production, an SRM this severe would trigger immediate investigation

    Example
    -------
    >>> df = load_marketing_ab()
    >>> print(df.groupby('test_group')['converted'].agg(['count', 'mean']))
    >>> # Load 10% sample for quick testing
    >>> df_sample = load_marketing_ab(sample_frac=0.1)
    """
    # Set default cache directory
    if cache_dir is None:
        cache_dir = "./data/raw/marketing_ab"

    file_path = Path(cache_dir) / "marketing_AB.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n\n"
            "Please download from Kaggle:\n"
            "  https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing\n\n"
            "And place in: ./data/raw/marketing_ab/marketing_AB.csv\n\n"
            "Alternatively, use Kaggle API:\n"
            "  kaggle datasets download -d faviovaz/marketing-ab-testing\n"
            "  unzip marketing-ab-testing.zip -d ./data/raw/marketing_ab/"
        )

    print(f"Loading Marketing A/B dataset from {file_path}...")
    df = pd.read_csv(file_path, index_col=0)

    # Standardize column names (remove spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Create binary treatment indicator (1 = ad, 0 = psa)
    df["treatment"] = (df["test_group"] == "ad").astype(int)

    # Data validation
    assert df["converted"].isin([0, 1]).all(), "Invalid values in 'converted'"
    assert df["test_group"].isin(["ad", "psa"]).all(), "Invalid test groups"

    print(f"Loaded Marketing A/B dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Conversion rate (ad): {df[df['test_group']=='ad']['converted'].mean():.2%}")
    print(f"  Conversion rate (psa): {df[df['test_group']=='psa']['converted'].mean():.2%}")

    # Apply sampling if requested
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"  Sampled to {len(df):,} rows ({sample_frac:.1%} of full dataset)")

    return df


def load_cookie_cats(
    sample_frac: float = 1.0,
    random_state: int = 42,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load Cookie Cats mobile game A/B test dataset (optional).

    This dataset contains 90K players from a mobile game experiment testing
    optimal placement of in-game "gates" to maximize retention.

    Parameters
    ----------
    sample_frac : float, default=1.0
        Fraction of data to load (0.0-1.0). Use smaller values for faster testing.
    random_state : int, default=42
        Random seed for reproducible sampling when sample_frac < 1.0
    cache_dir : str, optional
        Directory containing the CSV file. Default: './data/raw/cookie_cats'
        Expected file: cookie_cats.csv

    Returns
    -------
    pd.DataFrame
        Cookie Cats data with columns:
        - userid: Unique player ID
        - version: 'gate_30' (control) or 'gate_40' (treatment)
        - sum_gamerounds: Engagement (rounds played in first 14 days)
        - retention_1: 1-day retention (binary)
        - retention_7: 7-day retention (binary)

    Notes
    -----
    - Size: ~3MB
    - Download from: https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats
    - Result: gate_30 performed better than gate_40

    Example
    -------
    >>> df = load_cookie_cats()
    >>> print(df.groupby('version')['retention_7'].mean())
    >>> # Load 50% sample for faster testing
    >>> df_sample = load_cookie_cats(sample_frac=0.5)
    """
    # Set default cache directory
    if cache_dir is None:
        cache_dir = "./data/raw/cookie_cats"

    file_path = Path(cache_dir) / "cookie_cats.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n\n"
            "Please download from Kaggle:\n"
            "  https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats\n\n"
            "And place in: ./data/raw/cookie_cats/cookie_cats.csv"
        )

    print(f"Loading Cookie Cats dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Create binary treatment indicator (1 = gate_40, 0 = gate_30)
    df["treatment"] = (df["version"] == "gate_40").astype(int)

    # Data validation
    assert df["retention_1"].isin([0, 1, True, False]).all()
    assert df["retention_7"].isin([0, 1, True, False]).all()

    print(f"Loaded Cookie Cats dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  7-day retention (gate_30): {df[df['version']=='gate_30']['retention_7'].mean():.2%}")
    print(f"  7-day retention (gate_40): {df[df['version']=='gate_40']['retention_7'].mean():.2%}")

    # Apply sampling if requested
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"  Sampled to {len(df):,} rows ({sample_frac:.1%} of full dataset)")

    return df


def download_all_datasets(cache_dir: str = "./data/raw") -> Dict[str, bool]:
    """
    Attempt to download all available datasets.

    This is a convenience function that tries to load each dataset and
    reports which ones are available.

    Parameters
    ----------
    cache_dir : str
        Base directory for caching datasets

    Returns
    -------
    dict
        Status of each dataset: {dataset_name: successfully_loaded}

    Example
    -------
    >>> status = download_all_datasets()
    >>> print(f"Available datasets: {sum(status.values())}/{len(status)}")
    """
    status = {}

    # Try Criteo (easiest - via scikit-uplift)
    try:
        load_criteo_uplift(sample_frac=0.001, cache_dir=f"{cache_dir}/criteo_uplift")
        status["criteo_uplift"] = True
    except Exception as e:
        print(f"✗ Criteo dataset failed: {e}")
        status["criteo_uplift"] = False

    # Try Marketing A/B
    try:
        load_marketing_ab(cache_dir=f"{cache_dir}/marketing_ab")
        status["marketing_ab"] = True
    except Exception as e:
        print(f"✗ Marketing A/B dataset failed: {e}")
        status["marketing_ab"] = False

    # Try Cookie Cats
    try:
        load_cookie_cats(cache_dir=f"{cache_dir}/cookie_cats")
        status["cookie_cats"] = True
    except Exception as e:
        print(f"✗ Cookie Cats dataset failed: {e}")
        status["cookie_cats"] = False

    print(f"\n✓ Successfully loaded: {sum(status.values())}/{len(status)} datasets")
    return status


if __name__ == "__main__":
    # Demo: Try loading datasets
    print("=" * 80)
    print("A/B Testing Datasets - Loading Demo")
    print("=" * 80)

    # Show available datasets
    print("\nAvailable datasets:")
    for name, info in DATASETS.items():
        print(f"  - {info['name']} ({info['size']:,} rows)")
        print(f"    Source: {info['source_url']}")
        print()

    # Try loading Criteo (1% sample)
    try:
        print("\n" + "=" * 80)
        df_criteo = load_criteo_uplift(sample_frac=0.01)
        print(df_criteo.head())
    except Exception as e:
        print(f"Criteo loading failed: {e}")

    # Try loading Marketing A/B
    try:
        print("\n" + "=" * 80)
        df_marketing = load_marketing_ab()
        print(df_marketing.head())
    except Exception as e:
        print(f"Marketing A/B loading failed: {e}")
