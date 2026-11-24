"""
trustboost.session
------------------
Global mapping module for TrustBoost.

One shared, cumulative mapping accumulates all sanitization inputs
during the current Streamlit browser session.
"""

from .global_mapping import (
    merge_into_global_mapping,
    tokenize_with_global_mapping,
    restore_with_global_mapping,
    clear_global_mapping,
)

__all__ = [
    "merge_into_global_mapping",
    "tokenize_with_global_mapping",
    "restore_with_global_mapping",
    "clear_global_mapping",
]
