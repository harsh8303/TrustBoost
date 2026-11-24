"""
trustboost.tokenizer
--------------------
Local, deterministic tokenization and restoration module for TrustBoost.
"""

from .tokenizer import tokenize_entities, restore_text

__all__ = ["tokenize_entities", "restore_text"]
