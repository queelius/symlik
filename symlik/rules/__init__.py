"""
Rewriting rules for symbolic calculus.

Rules are loaded from .rerum files using rerum's DSL format.
"""

from pathlib import Path
from rerum import RuleEngine, ARITHMETIC_PRELUDE

# Rule file locations (relative to this directory)
_RULES_DIR = Path(__file__).parent
DERIVATIVE_RULES = _RULES_DIR / "derivative.rerum"
ALGEBRA_RULES = _RULES_DIR / "algebra.rerum"
INTEGRAL_RULES = _RULES_DIR / "integral.rerum"


def create_derivative_engine() -> RuleEngine:
    """Create an engine with derivative rules."""
    return RuleEngine.from_file(DERIVATIVE_RULES, fold_funcs=ARITHMETIC_PRELUDE)


def create_algebra_engine() -> RuleEngine:
    """Create an engine with algebra rules."""
    return RuleEngine.from_file(ALGEBRA_RULES, fold_funcs=ARITHMETIC_PRELUDE)


def create_integral_engine() -> RuleEngine:
    """Create an engine with integral rules."""
    return RuleEngine.from_file(INTEGRAL_RULES, fold_funcs=ARITHMETIC_PRELUDE)


def create_calculus_engine() -> RuleEngine:
    """Create an engine with derivative + algebra rules."""
    engine = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
    engine.load_file(DERIVATIVE_RULES)
    engine.load_file(ALGEBRA_RULES)
    return engine


def create_full_engine() -> RuleEngine:
    """Create an engine with all rules (derivative + algebra + integral)."""
    engine = RuleEngine(fold_funcs=ARITHMETIC_PRELUDE)
    engine.load_file(DERIVATIVE_RULES)
    engine.load_file(ALGEBRA_RULES)
    engine.load_file(INTEGRAL_RULES)
    return engine


# Default engine for symlik (lazy initialization)
_default_engine = None


def get_default_engine() -> RuleEngine:
    """Get the default calculus engine (lazy initialization)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = create_calculus_engine()
    return _default_engine


__all__ = [
    "get_default_engine",
    "create_derivative_engine",
    "create_algebra_engine",
    "create_integral_engine",
    "create_calculus_engine",
    "create_full_engine",
    "DERIVATIVE_RULES",
    "ALGEBRA_RULES",
    "INTEGRAL_RULES",
]
