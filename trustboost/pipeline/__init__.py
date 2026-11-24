"""
trustboost.pipeline
-------------------
End-to-end sanitization and restoration pipelines connecting contextual detection,
validation, and deterministic global tokenization.
"""

from .pipeline import (
    run_sanitization_pipeline,
    run_restoration_pipeline,
    sanitize_text,
    restore_text,
)

__all__ = [
    "run_sanitization_pipeline",
    "run_restoration_pipeline",
    "sanitize_text",
    "restore_text",
]
