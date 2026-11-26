"""
tests/test_pipeline.py
----------------------
PURPOSE:
    Integration and unit tests for the end-to-end sanitization pipeline
    (trustboost/pipeline/pipeline.py).

COVERS:
    1. Pipeline input validation (empty, whitespace, non-string).
    2. Successful detection and tokenization flow.
    3. No-PII flow (original text returned unchanged with empty mapping).
    4. Detection failure propagation (sanitization aborted, no false success).
    5. Tokenization failure propagation.
    6. Immutability of the original input string.
    7. Full round-trip integration using pipeline output and restore_text.
    8. Structural schema conformance for all return dictionaries.
"""

import pytest
from unittest.mock import patch
from trustboost.pipeline import sanitize_text, restore_text


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Package Exports and Input Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestPipelineExportsAndValidation:
    """Verifies public exports and input boundary checking."""

    def test_pipeline_exports(self):
        """sanitize_text and restore_text must be callable."""
        assert callable(sanitize_text)
        assert callable(restore_text)

    def test_empty_input_string(self):
        """Empty input text returns clean failure dictionary."""
        res = sanitize_text("")
        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert res["mapping"] == {}
        assert "empty" in res["message"].lower()

    def test_whitespace_only_input(self):
        """Whitespace-only input returns clean failure dictionary."""
        res = sanitize_text("   \n\t  ")
        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert "whitespace" in res["message"].lower()

    def test_non_string_input(self):
        """Non-string inputs (None, int, list) return clean failure dictionary."""
        assert sanitize_text(None)["success"] is False  # type: ignore
        assert sanitize_text(12345)["success"] is False  # type: ignore
        assert sanitize_text(["text"])["success"] is False  # type: ignore

    def test_result_structure_always_complete_on_error(self):
        """All expected keys must exist even when validation fails."""
        res = sanitize_text("")
        expected_keys = {
            "success", "original_text", "detected_entities",
            "sanitized_text", "mapping", "tokens", "message"
        }
        assert expected_keys.issubset(res.keys())


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: End-to-End Pipeline Execution (Mocked Detector)
# ═════════════════════════════════════════════════════════════════════════════

class TestPipelineExecution:
    """Tests the interaction between detection and tokenization steps."""

    def test_no_pii_detected_flow(self):
        """When detector returns zero entities, text is preserved with empty mapping."""
        text = "This is a completely public sentence about open science."
        mock_detection = {
            "success": True,
            "entities": [],
            "message": "No sensitive information detected.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = sanitize_text(text)

        assert res["success"] is True
        assert res["original_text"] == text
        assert res["sanitized_text"] == text
        assert res["detected_entities"] == []
        assert res["mapping"] == {}
        assert res["tokens"] == []
        assert "no sensitive information" in res["message"].lower()

    def test_successful_detection_and_tokenization(self):
        """Valid entities are seamlessly tokenized with mapping and metadata."""
        text = "Contact Harshit Sahu at harshit@example.com regarding Project Falcon."
        mock_detection = {
            "success": True,
            "entities": [
                {"text": "Harshit Sahu", "type": "PERSON", "start": 8, "end": 20, "reason": "Name"},
                {"text": "harshit@example.com", "type": "EMAIL", "start": 24, "end": 43, "reason": "Email"},
                {"text": "Project Falcon", "type": "CONFIDENTIAL_PROJECT", "start": 54, "end": 68, "reason": "Project"},
            ],
            "message": "Found 3 entities.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = sanitize_text(text)

        assert res["success"] is True
        assert res["original_text"] == text
        assert res["sanitized_text"] == "Contact [PERSON_1] at [EMAIL_1] regarding [CONFIDENTIAL_PROJECT_1]."
        assert res["mapping"] == {
            "[PERSON_1]": "Harshit Sahu",
            "[EMAIL_1]": "harshit@example.com",
            "[CONFIDENTIAL_PROJECT_1]": "Project Falcon",
        }
        assert len(res["tokens"]) == 3
        assert len(res["detected_entities"]) == 3
        assert "successfully sanitized" in res["message"].lower()

    def test_detection_failure_aborts_sanitization(self):
        """If detection fails (e.g. Ollama offline), pipeline must never report success."""
        text = "Secret text that could not be analyzed."
        mock_detection = {
            "success": False,
            "entities": [],
            "message": "Cannot connect to Ollama server.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = sanitize_text(text)

        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert res["mapping"] == {}
        assert res["tokens"] == []
        assert "detection failed" in res["message"].lower()

    def test_tokenization_failure_aborts_sanitization(self):
        """If tokenization encounters corrupted offsets, pipeline must report failure."""
        text = "John is here."
        mock_detection = {
            "success": True,
            # Deliberately corrupted offset
            "entities": [{"text": "John", "type": "PERSON", "start": 99, "end": 103, "reason": "Name"}],
            "message": "Entities found.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = sanitize_text(text)

        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert res["mapping"] == {}
        assert "tokenization failed" in res["message"].lower()

    def test_original_text_immutable(self):
        """Pipeline must never mutate the input string object."""
        text = "Hello Alice at alice@corp.com"
        mock_detection = {
            "success": True,
            "entities": [{"text": "Alice", "type": "PERSON", "start": 6, "end": 11, "reason": "Name"}],
            "message": "Entities found.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = sanitize_text(text)

        assert res["original_text"] == "Hello Alice at alice@corp.com"

    def test_pipeline_restoration_round_trip(self):
        """Sanitized text and mapping from pipeline must cleanly restore original text."""
        text = "Admin user Alice (alice@test.org) changed password to Secret@123."
        mock_detection = {
            "success": True,
            "entities": [
                {"text": "Alice", "type": "PERSON", "start": 11, "end": 16, "reason": "Name"},
                {"text": "alice@test.org", "type": "EMAIL", "start": 18, "end": 32, "reason": "Email"},
                {"text": "Secret@123", "type": "PASSWORD", "start": 54, "end": 64, "reason": "Password"},
            ],
            "message": "3 entities found.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            pipeline_res = sanitize_text(text)

        assert pipeline_res["success"] is True

        # Restore using the pipeline's returned mapping
        restoration_res = restore_text(pipeline_res["sanitized_text"], pipeline_res["mapping"])
        assert restoration_res["success"] is True
        assert restoration_res["restored_text"] == text
        assert restoration_res["unrestored_tokens"] == []

    def test_repeated_pii_sanitization_and_restoration_end_to_end(self):
        """Repeated PII with imprecise LLM offsets masks all occurrences and restores cleanly."""
        text = (
            "Alice Walker signed in with alice@example.com; "
            "Alice Walker confirmed alice@example.com and +1-555-0144 / +1-555-0144."
        )
        raw_llm = (
            '{"entities":['
            '{"text":"Alice Walker","type":"PERSON","start":0,"end":12},'
            '{"text":"alice@example.com","type":"EMAIL","start":10,"end":25},'
            '{"text":"Alice Walker","type":"PERSON","start":40,"end":52},'
            '{"text":"+1-555-0144","type":"PHONE","start":5,"end":15}'
            ']}'
        )
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": raw_llm, "message": "ok"},
        ):
            pipeline_res = sanitize_text(text)

        assert pipeline_res["success"] is True
        san = pipeline_res["sanitized_text"]
        assert "Alice Walker" not in san
        assert "alice@example.com" not in san
        assert "+1-555-0144" not in san
        assert san.count("[PERSON_1]") == 2
        assert san.count("[EMAIL_1]") == 2
        assert san.count("[PHONE_1]") == 2

        restoration_res = restore_text(san, pipeline_res["mapping"])
        assert restoration_res["success"] is True
        assert restoration_res["restored_text"] == text

    def test_ambiguous_unresolved_entity_fails_closed_in_pipeline(self):
        """Ambiguous entity that cannot be safely resolved aborts sanitization."""
        text = "May called May yesterday."
        raw_llm = '{"entities":[{"text":"May","type":"PERSON","start":99,"end":102}]}'
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": raw_llm, "message": "ok"},
        ):
            pipeline_res = sanitize_text(text)

        assert pipeline_res["success"] is False
        assert pipeline_res["sanitized_text"] is None
        assert "May" in pipeline_res["message"]
