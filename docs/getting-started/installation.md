# Installation

## Requirements

- Python 3.8 or higher
- NumPy
- [rerum](https://github.com/alextowell/rerum) (symbolic rewriting engine)

## Install from PyPI

```bash
pip install symlik
```

This installs symlik and its dependencies.

## Install from Source

```bash
git clone https://github.com/alextowell/symlik.git
cd symlik
pip install -e .
```

## Verify Installation

```python
from symlik import LikelihoodModel
from symlik.distributions import exponential

# Quick test
model = exponential()
fit = model.fit(data={'x': [1, 2, 3]}, init={'lambda': 1.0})
print(f"MLE: {fit.params['lambda']:.3f}")  # Should print ~0.5
```

## Development Installation

For development with testing tools:

```bash
pip install -e ".[dev]"
```

This adds pytest, coverage, and code quality tools.
