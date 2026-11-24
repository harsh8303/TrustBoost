"""
trustboost.detector
-------------------
Contextual PII detection package for TrustBoost.
"""

from trustboost.detector.pii_detector import detect_pii
from trustboost.detector.prompt_builder import VALID_ENTITY_TYPES, build_detection_prompt

__all__ = ["detect_pii", "build_detection_prompt", "VALID_ENTITY_TYPES"]
