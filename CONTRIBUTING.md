# Contributing

Thanks for taking the time to improve this experimentation toolkit.

## Development setup

Use Python 3.11 or newer (the CI matrix currently tests Python 3.11, 3.12, and 3.13).

Install the runtime dependencies used by the test suite:

```bash
python -m pip install numpy pandas scipy statsmodels scikit-learn pytest pytest-cov
```

Run the unit tests from the repository root:

```bash
PYTHONPATH=src python -m pytest
```

The repository's current coverage gate is 20%. The November 19, 2026 baseline was 22.51%; coverage is being expanded, especially for the end-to-end pipeline modules. Do not raise the threshold or advertise a higher coverage level without a measured run.

## Pull requests

- Include focused tests for behavior changes and bug fixes.
- Keep public API changes documented.
- Do not commit local virtual environments, notebook checkpoints, Python bytecode, dataset downloads, or coverage output.
- Clearly label simulated data and distinguish examples from validated production deployments.
