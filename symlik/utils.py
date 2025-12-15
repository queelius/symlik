"""
Utility functions for symlik.

Provides data conversion and helper functions used across the library.
"""

from typing import Any, Dict, List, Union


def to_data_dict(data: Any) -> Dict[str, Any]:
    """
    Convert various data formats to dict-of-lists format.

    Accepts:
    - dict: returned as-is (assumes dict-of-lists format)
    - pandas DataFrame: converted via to_dict('list')
    - polars DataFrame: converted via to_dict(as_series=False)
    - Any object with .to_dict() method

    Args:
        data: Input data in any supported format

    Returns:
        Dictionary mapping column names to lists of values

    Examples:
        >>> to_data_dict({'x': [1, 2, 3]})
        {'x': [1, 2, 3]}

        >>> import pandas as pd
        >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> to_data_dict(df)
        {'x': [1, 2, 3], 'y': [4, 5, 6]}
    """
    # Already a dict - return as-is
    if isinstance(data, dict):
        return data

    # Check for pandas DataFrame (duck-typing)
    if _is_pandas_dataframe(data):
        return data.to_dict('list')

    # Check for polars DataFrame (duck-typing)
    if _is_polars_dataframe(data):
        return data.to_dict(as_series=False)

    # Generic: try .to_dict() method
    if hasattr(data, 'to_dict'):
        result = data.to_dict()
        # Handle different to_dict() return formats
        if isinstance(result, dict):
            # Check if it's {col: {idx: val}} format (pandas default)
            first_val = next(iter(result.values()), None)
            if isinstance(first_val, dict):
                # Convert {col: {idx: val}} to {col: [val, ...]}
                return {k: list(v.values()) for k, v in result.items()}
            return result

    raise TypeError(
        f"Cannot convert {type(data).__name__} to data dict. "
        "Expected dict, pandas DataFrame, polars DataFrame, or object with .to_dict() method."
    )


def _is_pandas_dataframe(obj: Any) -> bool:
    """Check if object is a pandas DataFrame using duck-typing."""
    # Check for pandas DataFrame characteristics without importing pandas
    return (
        hasattr(obj, 'to_dict') and
        hasattr(obj, 'columns') and
        hasattr(obj, 'iloc') and
        type(obj).__module__.startswith('pandas')
    )


def _is_polars_dataframe(obj: Any) -> bool:
    """Check if object is a polars DataFrame using duck-typing."""
    # Check for polars DataFrame characteristics without importing polars
    return (
        hasattr(obj, 'to_dict') and
        hasattr(obj, 'columns') and
        hasattr(obj, 'select') and
        type(obj).__module__.startswith('polars')
    )


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dicts overriding earlier ones.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result
