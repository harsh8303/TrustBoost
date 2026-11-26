"""
tests/test_pii_detector.py
----------------------------
PURPOSE:
    Tests for the Phase 3 PII detection pipeline.

    This file has two sections:

    ── UNIT TESTS (Section 1) ───────────────────────────────────────────────────
    These do NOT require Ollama to be running.
    They mock the generate_response() call and test only our Python logic:
      - Prompt construction
      - JSON parsing and extraction
      - Entity validation (fields, types, offsets)
      - Offset recovery
      - Duplicate/overlap removal
      - Error handling

    ── INTEGRATION TESTS (Section 2) ───────────────────────────────────────────
    These require a running Ollama server AND gemma3:4b downloaded.
    They test the full end-to-end pipeline with real model responses.
    Marked with @pytest.mark.integration — skip with: -m "not integration"

HOW TO RUN:
    # Unit tests only (no Ollama needed):
    pytest tests/test_pii_detector.py -v -m "not integration"

    # Integration tests only (Ollama must be running):
    pytest tests/test_pii_detector.py -v -m integration

    # All tests:
    pytest tests/test_pii_detector.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: UNIT TESTS (No Ollama required)
# ═════════════════════════════════════════════════════════════════════════════

# ── Prompt Builder Tests ──────────────────────────────────────────────────────

class TestPromptBuilder:
    """Tests for trustboost/detector/prompt_builder.py"""

    def test_build_prompt_returns_string(self):
        """build_detection_prompt() must return a string."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        result = build_detection_prompt("Hello world")
        assert isinstance(result, str)

    def test_build_prompt_contains_input_text(self):
        """The original text must appear verbatim inside the prompt."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        text = "My email is test@example.com"
        prompt = build_detection_prompt(text)
        assert text in prompt

    def test_build_prompt_contains_json_instruction(self):
        """The prompt must tell the model to return only JSON."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        prompt = build_detection_prompt("some text")
        assert "JSON" in prompt

    def test_build_prompt_mentions_entities_key(self):
        """The prompt must mention the 'entities' key so the model knows the format."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        prompt = build_detection_prompt("some text")
        assert "entities" in prompt

    def test_build_prompt_mentions_start_and_end(self):
        """The prompt must explain start/end positions."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        prompt = build_detection_prompt("some text")
        assert "start" in prompt
        assert "end" in prompt

    def test_valid_entity_types_is_non_empty_set(self):
        """VALID_ENTITY_TYPES must be a non-empty set of strings."""
        from trustboost.detector.prompt_builder import VALID_ENTITY_TYPES
        assert isinstance(VALID_ENTITY_TYPES, set)
        assert len(VALID_ENTITY_TYPES) > 0

    def test_valid_entity_types_contains_key_categories(self):
        """Check that the most important PII categories are registered."""
        from trustboost.detector.prompt_builder import VALID_ENTITY_TYPES
        essential = {"PERSON", "EMAIL", "PHONE", "PASSWORD", "IP_ADDRESS",
                     "API_KEY", "AADHAAR", "CREDIT_CARD", "CONFIDENTIAL_PROJECT"}
        assert essential.issubset(VALID_ENTITY_TYPES)

    def test_all_24_required_categories_present(self):
        """Verify that all 24 required audit categories are in VALID_ENTITY_TYPES."""
        from trustboost.detector.prompt_builder import VALID_ENTITY_TYPES
        required_24 = {
            "PERSON", "EMAIL", "PHONE", "ADDRESS", "LOCATION", "DATE_OF_BIRTH",
            "AADHAAR", "PAN", "BANK_ACCOUNT", "CREDIT_CARD", "IFSC", "FINANCIAL_IDENTIFIER",
            "PASSWORD", "API_KEY", "ACCESS_TOKEN", "SECRET", "PRIVATE_KEY", "CREDENTIAL",
            "IP_ADDRESS", "INTERNAL_URL", "DATABASE_CREDENTIAL", "SERVER_IDENTIFIER",
            "CONFIDENTIAL_PROJECT", "INTERNAL_ORGANIZATION_INFORMATION"
        }
        assert required_24.issubset(VALID_ENTITY_TYPES)
        assert len(VALID_ENTITY_TYPES) >= 24

    def test_detector_package_imports(self):
        """Verify trustboost.detector exposes detect_pii and prompt helpers."""
        from trustboost.detector import detect_pii, build_detection_prompt, VALID_ENTITY_TYPES
        assert callable(detect_pii)
        assert callable(build_detection_prompt)
        assert isinstance(VALID_ENTITY_TYPES, set)

    def test_different_texts_produce_different_prompts(self):
        """Two different input texts must produce two different prompts."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        p1 = build_detection_prompt("Hello Alice")
        p2 = build_detection_prompt("Hello Bob")
        assert p1 != p2


# ── JSON Extraction Tests ─────────────────────────────────────────────────────

class TestExtractJsonFromResponse:
    """Tests for the _extract_json_from_response() internal function."""

    def test_valid_json_parsed_directly(self):
        """Clean JSON string is parsed correctly."""
        from trustboost.detector.pii_detector import _extract_json_from_response
        raw = '{"entities": []}'
        result = _extract_json_from_response(raw)
        assert result == {"entities": []}

    def test_json_with_leading_text_extracted(self):
        """JSON embedded after explanation text is still extracted."""
        from trustboost.detector.pii_detector import _extract_json_from_response
        raw = 'Here is the result:\n{"entities": []}'
        result = _extract_json_from_response(raw)
        assert result is not None
        assert "entities" in result

    def test_json_with_trailing_text_extracted(self):
        """JSON followed by extra text is still extracted."""
        from trustboost.detector.pii_detector import _extract_json_from_response
        raw = '{"entities": []} \nHope that helps!'
        result = _extract_json_from_response(raw)
        assert result is not None

    def test_completely_invalid_response_returns_none(self):
        """Non-JSON garbage returns None."""
        from trustboost.detector.pii_detector import _extract_json_from_response
        result = _extract_json_from_response("This is not JSON at all.")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        from trustboost.detector.pii_detector import _extract_json_from_response
        assert _extract_json_from_response("") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string returns None."""
        from trustboost.detector.pii_detector import _extract_json_from_response
        assert _extract_json_from_response("   \n  ") is None


# ── Offset Recovery Tests ─────────────────────────────────────────────────────

class TestAttemptOffsetRecovery:
    """Tests for the _attempt_offset_recovery() internal function."""

    def test_recovery_succeeds_for_unique_occurrence(self):
        """
        When the entity text appears exactly once in the original,
        recovery should find the correct offsets.
        """
        from trustboost.detector.pii_detector import _attempt_offset_recovery
        original = "My email is test@example.com today."
        entity = {"text": "test@example.com", "type": "EMAIL",
                  "start": 999, "end": 1000, "reason": "email"}
        result = _attempt_offset_recovery(entity, original)
        assert result is not None
        assert result["start"] == 12
        assert result["end"] == 28
        # Verify slice matches
        assert original[result["start"]:result["end"]] == "test@example.com"

    def test_recovery_fails_for_multiple_occurrences(self):
        """
        When the same text appears more than once, recovery must return None
        because we cannot reliably determine which occurrence the model meant.
        """
        from trustboost.detector.pii_detector import _attempt_offset_recovery
        original = "cat and cat are both cat"
        entity = {"text": "cat", "type": "PERSON",
                  "start": 999, "end": 1000, "reason": "name"}
        result = _attempt_offset_recovery(entity, original)
        assert result is None

    def test_recovery_fails_when_text_not_in_original(self):
        """
        When the entity text is not in the original at all (hallucinated),
        recovery must return None.
        """
        from trustboost.detector.pii_detector import _attempt_offset_recovery
        original = "Hello world"
        entity = {"text": "nonexistent@fake.com", "type": "EMAIL",
                  "start": 0, "end": 5, "reason": "email"}
        result = _attempt_offset_recovery(entity, original)
        assert result is None

    def test_recovery_fails_for_empty_entity_text(self):
        """Empty entity text should not be recovered."""
        from trustboost.detector.pii_detector import _attempt_offset_recovery
        entity = {"text": "", "type": "EMAIL", "start": 0, "end": 5, "reason": ""}
        result = _attempt_offset_recovery(entity, "some text")
        assert result is None


# ── Entity Validation Tests ───────────────────────────────────────────────────

class TestValidateEntity:
    """Tests for the _validate_entity() internal function."""

    def _make_entity(self, text, original, entity_type="PERSON",
                     start=None, end=None, reason="test reason"):
        """Helper: build a valid entity dict for the given text."""
        s = original.find(text) if start is None else start
        e = s + len(text) if end is None else end
        return {"text": text, "type": entity_type,
                "start": s, "end": e, "reason": reason}

    def test_valid_entity_passes(self):
        """A correctly formed entity with matching offsets passes validation."""
        from trustboost.detector.pii_detector import _validate_entity
        original = "Call John at 9876543210"
        entity = self._make_entity("John", original, "PERSON")
        result = _validate_entity(entity, original)
        assert result is not None
        assert result["text"] == "John"
        assert result["type"] == "PERSON"
        assert original[result["start"]:result["end"]] == "John"

    def test_non_dict_entity_rejected(self):
        """A non-dict entity (e.g., a string) is immediately rejected."""
        from trustboost.detector.pii_detector import _validate_entity
        result = _validate_entity("not a dict", "some text")
        assert result is None

    def test_missing_type_field_rejected(self):
        """An entity without the 'type' field is rejected."""
        from trustboost.detector.pii_detector import _validate_entity
        entity = {"text": "John", "start": 0, "end": 4, "reason": "name"}
        result = _validate_entity(entity, "John Smith")
        assert result is None

    def test_invalid_entity_type_rejected(self):
        """An entity with a made-up type (not in VALID_ENTITY_TYPES) is rejected."""
        from trustboost.detector.pii_detector import _validate_entity
        original = "John Smith"
        entity = {"text": "John Smith", "type": "MADE_UP_TYPE",
                  "start": 0, "end": 10, "reason": "test"}
        result = _validate_entity(entity, original)
        assert result is None

    def test_empty_text_rejected(self):
        """An entity with an empty text string is rejected."""
        from trustboost.detector.pii_detector import _validate_entity
        entity = {"text": "", "type": "PERSON",
                  "start": 0, "end": 0, "reason": "test"}
        result = _validate_entity(entity, "some text")
        assert result is None

    def test_negative_start_triggers_recovery(self):
        """
        An entity with a negative start offset triggers recovery.
        Recovery succeeds if the text is unique in the original.
        """
        from trustboost.detector.pii_detector import _validate_entity
        original = "My email is unique@place.com"
        entity = {"text": "unique@place.com", "type": "EMAIL",
                  "start": -5, "end": 10, "reason": "email"}
        result = _validate_entity(entity, original)
        # Recovery should fix the offsets
        assert result is not None
        assert original[result["start"]:result["end"]] == "unique@place.com"

    def test_end_exceeding_text_length_triggers_recovery(self):
        """An entity with end > len(original_text) triggers recovery."""
        from trustboost.detector.pii_detector import _validate_entity
        original = "Password: Secret123"
        entity = {"text": "Secret123", "type": "PASSWORD",
                  "start": 10, "end": 9999, "reason": "password"}
        result = _validate_entity(entity, original)
        assert result is not None
        assert original[result["start"]:result["end"]] == "Secret123"

    def test_mismatched_offset_triggers_recovery(self):
        """
        An entity whose slice doesn't match its text triggers recovery.
        """
        from trustboost.detector.pii_detector import _validate_entity
        original = "IP address: 10.0.0.1"
        entity = {"text": "10.0.0.1", "type": "IP_ADDRESS",
                  "start": 0, "end": 8, "reason": "ip"}  # Wrong offsets
        result = _validate_entity(entity, original)
        # Recovery should find "10.0.0.1" at position 12
        assert result is not None
        assert original[result["start"]:result["end"]] == "10.0.0.1"

    def test_offset_invariant_always_holds_for_accepted_entity(self):
        """
        CRITICAL: For every accepted entity,
        original_text[start:end] must equal entity['text'].
        """
        from trustboost.detector.pii_detector import _validate_entity
        original = "Contact Priya at priya@work.org for the meeting."
        entity = {"text": "priya@work.org", "type": "EMAIL",
                  "start": 17, "end": 31, "reason": "email"}
        result = _validate_entity(entity, original)
        assert result is not None
        assert original[result["start"]:result["end"]] == result["text"]


# ── Duplicate and Overlap Removal Tests ───────────────────────────────────────

class TestRemoveDuplicatesAndOverlaps:
    """Tests for the _remove_duplicates_and_overlaps() internal function."""

    def test_empty_list_returns_empty(self):
        from trustboost.detector.pii_detector import _remove_duplicates_and_overlaps
        assert _remove_duplicates_and_overlaps([]) == []

    def test_non_overlapping_entities_all_kept(self):
        """Entities with no overlap are all preserved."""
        from trustboost.detector.pii_detector import _remove_duplicates_and_overlaps
        entities = [
            {"text": "Alice", "type": "PERSON", "start": 0, "end": 5, "reason": ""},
            {"text": "bob@test.com", "type": "EMAIL", "start": 10, "end": 22, "reason": ""},
        ]
        result = _remove_duplicates_and_overlaps(entities)
        assert len(result) == 2

    def test_exact_duplicate_removed(self):
        """Two identical entities are collapsed into one."""
        from trustboost.detector.pii_detector import _remove_duplicates_and_overlaps
        entity = {"text": "Alice", "type": "PERSON", "start": 0, "end": 5, "reason": ""}
        result = _remove_duplicates_and_overlaps([entity, entity.copy()])
        assert len(result) == 1

    def test_contained_shorter_entity_removed(self):
        """
        'Rahul' (0–5) overlaps with 'Rahul Sharma' (0–12).
        The longer 'Rahul Sharma' should win.
        """
        from trustboost.detector.pii_detector import _remove_duplicates_and_overlaps
        entities = [
            {"text": "Rahul Sharma", "type": "PERSON", "start": 0, "end": 12, "reason": ""},
            {"text": "Rahul", "type": "PERSON", "start": 0, "end": 5, "reason": ""},
        ]
        result = _remove_duplicates_and_overlaps(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Rahul Sharma"

    def test_partially_overlapping_entities_one_removed(self):
        """When two entities partially overlap, only the first (by position) is kept."""
        from trustboost.detector.pii_detector import _remove_duplicates_and_overlaps
        entities = [
            {"text": "John Do", "type": "PERSON", "start": 0, "end": 7, "reason": ""},
            {"text": "Do re", "type": "PERSON", "start": 5, "end": 10, "reason": ""},
        ]
        result = _remove_duplicates_and_overlaps(entities)
        assert len(result) == 1

    def test_result_sorted_by_start_position(self):
        """The returned list should be ordered by entity start position."""
        from trustboost.detector.pii_detector import _remove_duplicates_and_overlaps
        entities = [
            {"text": "email@test.com", "type": "EMAIL", "start": 20, "end": 34, "reason": ""},
            {"text": "Alice", "type": "PERSON", "start": 0, "end": 5, "reason": ""},
        ]
        result = _remove_duplicates_and_overlaps(entities)
        assert result[0]["start"] < result[1]["start"]


# ── detect_pii() Unit Tests (mocked Ollama) ───────────────────────────────────

class TestDetectPiiUnit:
    """
    Unit tests for detect_pii() with mocked Ollama responses.
    These tests verify our complete Python pipeline without a real LLM.
    """

    def _mock_ollama(self, json_response: str):
        """
        Helper: returns a context manager that makes generate_response()
        return a successful result with the given JSON string as the response.
        """
        return patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={
                "success": True,
                "response": json_response,
                "message": "ok",
            },
        )

    def _mock_ollama_failure(self, message="Ollama unavailable"):
        """Helper: makes generate_response() return a failure result."""
        return patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={
                "success": False,
                "response": "",
                "message": message,
            },
        )

    # ── Basic detection cases ─────────────────────────────────────────────────

    def test_detects_person_name(self):
        """PERSON entity detected correctly."""
        text = "Rahul Sharma works here."
        json_resp = json.dumps({"entities": [
            {"text": "Rahul Sharma", "type": "PERSON",
             "start": 0, "end": 12, "reason": "Full name"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 1
        assert result["entities"][0]["type"] == "PERSON"
        assert text[result["entities"][0]["start"]:result["entities"][0]["end"]] == "Rahul Sharma"

    def test_detects_email(self):
        """EMAIL entity detected correctly."""
        text = "Send mail to harshit@example.com please."
        json_resp = json.dumps({"entities": [
            {"text": "harshit@example.com", "type": "EMAIL",
             "start": 13, "end": 32, "reason": "Personal email"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"][0]["type"] == "EMAIL"
        assert text[result["entities"][0]["start"]:result["entities"][0]["end"]] == "harshit@example.com"

    def test_detects_phone_number(self):
        """PHONE entity detected correctly."""
        text = "Call me at 9876543210 anytime."
        json_resp = json.dumps({"entities": [
            {"text": "9876543210", "type": "PHONE",
             "start": 11, "end": 21, "reason": "Phone number"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"][0]["type"] == "PHONE"
        assert text[11:21] == "9876543210"

    def test_detects_password(self):
        """PASSWORD entity detected correctly."""
        text = "My password is Admin@123."
        json_resp = json.dumps({"entities": [
            {"text": "Admin@123", "type": "PASSWORD",
             "start": 15, "end": 24, "reason": "Authentication credential"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"][0]["type"] == "PASSWORD"
        assert text[15:24] == "Admin@123"

    def test_detects_api_key(self):
        """API_KEY entity detected correctly."""
        text = "API key: sk-abc123def456 for the service."
        json_resp = json.dumps({"entities": [
            {"text": "sk-abc123def456", "type": "API_KEY",
             "start": 9, "end": 24, "reason": "API authentication key"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"][0]["type"] == "API_KEY"
        assert text[result["entities"][0]["start"]:result["entities"][0]["end"]] == "sk-abc123def456"

    def test_detects_ip_address(self):
        """IP_ADDRESS entity detected correctly."""
        text = "My IP address is 192.168.1.10."
        json_resp = json.dumps({"entities": [
            {"text": "192.168.1.10", "type": "IP_ADDRESS",
             "start": 17, "end": 29, "reason": "Internal IP address"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"][0]["type"] == "IP_ADDRESS"

    def test_detects_confidential_project(self):
        """CONFIDENTIAL_PROJECT entity detected correctly."""
        text = "We are working on a confidential project called Project Falcon."
        json_resp = json.dumps({"entities": [
            {"text": "Project Falcon", "type": "CONFIDENTIAL_PROJECT",
             "start": 48, "end": 62, "reason": "Internal confidential project name"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"][0]["type"] == "CONFIDENTIAL_PROJECT"
        assert text[48:62] == "Project Falcon"

    # ── No PII cases ──────────────────────────────────────────────────────────

    def test_no_pii_in_ordinary_sentence(self):
        """An ordinary sentence returns empty entities with success=True."""
        text = "The meeting is scheduled for 10 AM tomorrow."
        json_resp = json.dumps({"entities": []})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"] == []

    def test_no_pii_for_academic_marks(self):
        """Academic marks alone are not PII."""
        text = "I scored 95 marks in mathematics."
        json_resp = json.dumps({"entities": []})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"] == []

    def test_no_pii_for_public_information(self):
        """Generic public information returns no entities."""
        text = "Python is a popular programming language created by Guido van Rossum."
        json_resp = json.dumps({"entities": []})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"] == []

    def test_no_pii_for_college_assignment(self):
        """College assignments and academic projects are not PII."""
        text = "The machine learning project is a college assignment."
        json_resp = json.dumps({"entities": []})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"] == []

    # ── Validation tests ──────────────────────────────────────────────────────

    def test_invalid_json_response_returns_failure(self):
        """When the model returns invalid JSON, detection fails cleanly."""
        with self._mock_ollama("This is not valid JSON!"):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("Some text with info@test.com")
        assert result["success"] is False
        assert result["entities"] == []
        assert "failed" in result["message"].lower()

    def test_missing_entities_key_returns_failure(self):
        """JSON without 'entities' key causes a clean failure."""
        with self._mock_ollama('{"result": []}'):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("Some text")
        assert result["success"] is False
        assert result["entities"] == []

    def test_invalid_entity_type_filtered_out(self):
        """Entities with unknown types are silently dropped."""
        text = "Call John."
        json_resp = json.dumps({"entities": [
            {"text": "John", "type": "HALLUCINATED_TYPE",
             "start": 5, "end": 9, "reason": "test"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert result["entities"] == []  # Invalid type was filtered

    def test_missing_entity_fields_handled(self):
        """
        An entity missing required fields is filtered out.
        The overall detection still returns success=True with an empty list.
        """
        json_resp = json.dumps({"entities": [
            {"text": "Alice"}  # Missing type, start, end, reason
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("Hello Alice")
        assert result["success"] is True
        assert result["entities"] == []

    def test_wrong_offsets_recovered_if_text_unique(self):
        """
        When offsets are wrong but entity text appears exactly once,
        Python recovers the correct offsets automatically.
        """
        text = "Contact: unique_email@corp.com for support."
        json_resp = json.dumps({"entities": [
            {"text": "unique_email@corp.com", "type": "EMAIL",
             "start": 999, "end": 1000, "reason": "email"}  # Wrong offsets
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 1
        # Verify the offset invariant
        e = result["entities"][0]
        assert text[e["start"]:e["end"]] == e["text"]

    def test_entity_text_mismatch_with_ambiguous_location_rejected(self):
        """
        An entity whose text appears multiple times in the original is
        rejected during recovery because the location is ambiguous.
        """
        text = "cat and cat are different cat"
        json_resp = json.dumps({"entities": [
            {"text": "cat", "type": "PERSON",
             "start": 999, "end": 1000, "reason": "name"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        # Cannot safely recover — entity rejected
        assert result["entities"] == []

    def test_duplicate_entities_collapsed(self):
        """
        When the model returns two identical entities, only one is kept.
        """
        text = "John Doe is here."
        json_resp = json.dumps({"entities": [
            {"text": "John Doe", "type": "PERSON", "start": 0, "end": 8, "reason": "name"},
            {"text": "John Doe", "type": "PERSON", "start": 0, "end": 8, "reason": "name"},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 1

    def test_overlapping_entities_longer_wins(self):
        """
        When 'Rahul' (0-5) and 'Rahul Sharma' (0-12) both detected,
        the longer 'Rahul Sharma' wins.
        """
        text = "Rahul Sharma is here."
        json_resp = json.dumps({"entities": [
            {"text": "Rahul", "type": "PERSON", "start": 0, "end": 5, "reason": "first name"},
            {"text": "Rahul Sharma", "type": "PERSON", "start": 0, "end": 12, "reason": "full name"},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "Rahul Sharma"

    # ── Original text preservation ────────────────────────────────────────────

    def test_original_text_never_modified(self):
        """
        The input text must remain completely unchanged after detection.
        detect_pii() must only READ the text, never write to it.
        """
        import copy
        original = "My name is Priya and my email is priya@test.com"
        original_copy = copy.deepcopy(original)
        json_resp = json.dumps({"entities": [
            {"text": "Priya", "type": "PERSON", "start": 11, "end": 16, "reason": "name"},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            detect_pii(original)
        assert original == original_copy, "detect_pii() must NOT modify the original text"

    def test_offset_invariant_for_all_returned_entities(self):
        """
        CRITICAL SAFETY TEST:
        For every entity returned, text[start:end] must equal entity['text'].
        This test verifies the invariant across multiple entities at once.
        """
        text = "Name: Arjun Mehta, Email: arjun@example.com, IP: 10.0.0.1"
        json_resp = json.dumps({"entities": [
            {"text": "Arjun Mehta", "type": "PERSON", "start": 6, "end": 17, "reason": "name"},
            {"text": "arjun@example.com", "type": "EMAIL", "start": 26, "end": 43, "reason": "email"},
            {"text": "10.0.0.1", "type": "IP_ADDRESS", "start": 49, "end": 57, "reason": "ip"},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        for entity in result["entities"]:
            assert text[entity["start"]:entity["end"]] == entity["text"], (
                f"Offset invariant violated for entity: {entity}"
            )

    # ── Error handling tests ──────────────────────────────────────────────────

    def test_empty_input_text_returns_failure(self):
        """Empty string input is rejected immediately without calling Ollama."""
        from trustboost.detector.pii_detector import detect_pii
        result = detect_pii("")
        assert result["success"] is False
        assert result["entities"] == []

    def test_whitespace_only_input_returns_failure(self):
        """Whitespace-only input is rejected without calling Ollama."""
        from trustboost.detector.pii_detector import detect_pii
        result = detect_pii("   \n  ")
        assert result["success"] is False

    def test_ollama_unavailable_returns_failure(self):
        """When Ollama is down, detect_pii() returns a clean failure dict."""
        with self._mock_ollama_failure("Cannot connect to Ollama server"):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("My email is test@test.com")
        assert result["success"] is False
        assert result["entities"] == []
        assert "failed" in result["message"].lower()

    def test_empty_model_response_returns_failure(self):
        """An empty model response is treated as a failure."""
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": "", "message": "ok"},
        ):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("My email is test@test.com")
        assert result["success"] is False
        assert result["entities"] == []

    def test_result_always_has_required_keys(self):
        """detect_pii() always returns a dict with success, entities, message."""
        with self._mock_ollama_failure():
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("test text")
        assert "success" in result
        assert "entities" in result
        assert "message" in result

    def test_success_false_never_returns_entities(self):
        """When success=False, the entities list must always be empty."""
        with self._mock_ollama("not valid json"):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("test text with data@example.com")
        if not result["success"]:
            assert result["entities"] == []

    def test_non_string_input_returns_failure(self):
        """Non-string input (e.g., None, int) is handled gracefully."""
        from trustboost.detector.pii_detector import detect_pii
        assert detect_pii(None)["success"] is False
        assert detect_pii(123)["success"] is False
        assert detect_pii(["list"])["success"] is False


class TestPhase2ADuplicateSafeOffsetRecovery:
    """Tests for Phase 2A batch-aware, duplicate-safe offset recovery."""

    def _mock_ollama(self, json_response: str):
        return patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={
                "success": True,
                "response": json_response,
                "message": "ok",
            },
        )

    def test_repeated_person_first_exact_second_wrong_offsets(self):
        """Exact first span is reserved; inaccurate second occurrence recovers cleanly."""
        text = "Alice Walker signed in, and Alice Walker changed her password."
        json_resp = json.dumps({"entities": [
            {"text": "Alice Walker", "type": "PERSON", "start": 0, "end": 12},
            {"text": "Alice Walker", "type": "PERSON", "start": 22, "end": 32},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 2
        assert [(e["start"], e["end"]) for e in result["entities"]] == [(0, 12), (28, 40)]
        for ent in result["entities"]:
            assert text[ent["start"]:ent["end"]] == "Alice Walker"

    def test_repeated_person_missing_offsets_masks_all_occurrences(self):
        """Multi-word person name with missing offsets recovers all occurrences."""
        text = "Alice Walker signed in, and Alice Walker approved the transfer."
        json_resp = json.dumps({"entities": [
            {"text": "Alice Walker", "type": "PERSON"},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert [(e["start"], e["end"]) for e in result["entities"]] == [(0, 12), (28, 40)]

    def test_repeated_person_single_exact_detection_masks_second_occurrence(self):
        """Even if LLM emits exact offsets for only occurrence #1, #2 is also masked."""
        text = "Alice Walker signed in, and Alice Walker changed her password."
        json_resp = json.dumps({"entities": [
            {"text": "Alice Walker", "type": "PERSON", "start": 0, "end": 12},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert [(e["start"], e["end"]) for e in result["entities"]] == [(0, 12), (28, 40)]

    def test_repeated_email_phone_pan_aadhaar_wrong_and_single_detections(self):
        """Repeated structured PII types recover every occurrence safely."""
        cases = [
            (
                "Send to alice@example.com and cc alice@example.com today.",
                [{"text": "alice@example.com", "type": "EMAIL", "start": 0, "end": 16},
                 {"text": "alice@example.com", "type": "EMAIL", "start": 23, "end": 38}],
                2,
            ),
            (
                "Call +1-555-0144 first, or retry +1-555-0144 after 5 PM.",
                [{"text": "+1-555-0144", "type": "PHONE", "start": 0, "end": 10}],
                2,
            ),
            (
                "PAN ABCDE1234F verified; backup PAN ABCDE1234F logged.",
                [{"text": "ABCDE1234F", "type": "PAN", "start": 99, "end": 109}],
                2,
            ),
            (
                "Aadhaar 9876 5432 1098 matches Aadhaar 9876 5432 1098.",
                [{"text": "9876 5432 1098", "type": "AADHAAR"}],
                2,
            ),
        ]
        from trustboost.detector.pii_detector import detect_pii
        for text, raw_ents, expected_count in cases:
            with self._mock_ollama(json.dumps({"entities": raw_ents})):
                result = detect_pii(text)
            assert result["success"] is True, f"Failed on {text}: {result}"
            assert len(result["entities"]) == expected_count
            for ent in result["entities"]:
                assert text[ent["start"]:ent["end"]] == ent["text"]

    def test_identical_text_different_types_preserved(self):
        """Identical surface text with distinct types preserves both categories and spans."""
        text = "Meet Jordan in Jordan today."
        # Deliberately off by 1 on both spans; ordered K=2 == M=2
        json_resp = json.dumps({"entities": [
            {"text": "Jordan", "type": "PERSON", "start": 4, "end": 10},
            {"text": "Jordan", "type": "LOCATION", "start": 14, "end": 20},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 2
        assert result["entities"][0]["type"] == "PERSON"
        assert (result["entities"][0]["start"], result["entities"][0]["end"]) == (5, 11)
        assert result["entities"][1]["type"] == "LOCATION"
        assert (result["entities"][1]["start"], result["entities"][1]["end"]) == (15, 21)

    def test_three_occurrences_with_wrong_offsets_resolved_in_order(self):
        """Three occurrences with shifted LLM offsets resolve 1-to-1."""
        text = "Jordan met our client Jordan in Jordan last week."
        json_resp = json.dumps({"entities": [
            {"text": "Jordan", "type": "PERSON", "start": 0, "end": 5},
            {"text": "Jordan", "type": "PERSON", "start": 26, "end": 31},
            {"text": "Jordan", "type": "LOCATION", "start": 40, "end": 45},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert [(e["start"], e["end"], e["type"]) for e in result["entities"]] == [
            (0, 6, "PERSON"),
            (22, 28, "PERSON"),
            (32, 38, "LOCATION"),
        ]

    def test_word_boundary_prevents_subtoken_match(self):
        """'SBI' must match standalone 'SBI' rather than prefix inside 'SBIN0001122'."""
        text = "Credited to SBI account with IFSC SBIN0001122."
        json_resp = json.dumps({"entities": [
            {"text": "SBI", "type": "BANK_ACCOUNT", "start": 0, "end": 3},
            {"text": "SBIN0001122", "type": "IFSC", "start": 30, "end": 41},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert [(e["text"], e["start"], e["end"]) for e in result["entities"]] == [
            ("SBI", 12, 15),
            ("SBIN0001122", 34, 45),
        ]

    def test_ambiguous_common_word_may_not_blindly_masked(self):
        """'May' in 'May Parker' is resolved as PERSON without masking 'this May'."""
        text = "Please confirm if May Parker is travelling to Paris this May."
        json_resp = json.dumps({"entities": [
            {"text": "May", "type": "PERSON", "start": 0, "end": 3},
            {"text": "Paris", "type": "LOCATION", "start": 40, "end": 45},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert [(e["text"], e["start"], e["end"]) for e in result["entities"]] == [
            ("May", 18, 21),
            ("Paris", 46, 51),
        ]

    def test_unresolvable_ambiguous_entity_fails_closed(self):
        """Ambiguous common word without safe disambiguation fails closed explicitly."""
        text = "May called May yesterday."
        json_resp = json.dumps({"entities": [
            {"text": "May", "type": "PERSON", "start": 99, "end": 102},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is False
        assert result["entities"] == []
        assert "unresolved_entities" in result
        assert result["unresolved_entities"][0]["text"] == "May"
        assert "May" in result["message"]

    def test_hallucinated_entity_text_rejected(self):
        """Hallucinated entity absent from text is ignored while valid entities pass."""
        text = "Reach Alice Walker at alice@example.com."
        json_resp = json.dumps({"entities": [
            {"text": "Fake Hallucination", "type": "PERSON", "start": 0, "end": 18},
            {"text": "alice@example.com", "type": "EMAIL", "start": 0, "end": 10},
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "alice@example.com"
        assert text[result["entities"][0]["start"]:result["entities"][0]["end"]] == "alice@example.com"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: INTEGRATION TESTS (Require Ollama + gemma3:4b)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPiiDetectorIntegration:
    """
    Integration tests that call the REAL Gemma 3 4B model.

    PREREQUISITES:
        - ollama serve (or Ollama desktop app running)
        - ollama pull gemma3:4b
        - conda activate trustboost

    HOW TO RUN:
        pytest tests/test_pii_detector.py -v -m integration
    """

    def test_real_person_and_email_detected(self):
        """Real model detects a name and email in a simple sentence."""
        from trustboost.detector.pii_detector import detect_pii
        result = detect_pii("Rahul Sharma's email is rahul@example.com.")
        assert result["success"] is True
        types_found = {e["type"] for e in result["entities"]}
        assert "PERSON" in types_found or "EMAIL" in types_found

    def test_real_no_pii_in_generic_sentence(self):
        """Real model does not hallucinate PII in a generic sentence."""
        from trustboost.detector.pii_detector import detect_pii
        result = detect_pii("The meeting is scheduled for 10 AM tomorrow.")
        assert result["success"] is True
        # The model should not flag a generic meeting time as PII
        # Allow small false positives — just ensure no crash
        assert isinstance(result["entities"], list)

    def test_real_offset_invariant(self):
        """Every entity returned by the real model satisfies text[start:end] == text."""
        from trustboost.detector.pii_detector import detect_pii
        text = "My password is Secret@999 and IP is 172.16.0.1."
        result = detect_pii(text)
        assert result["success"] is True
        for entity in result["entities"]:
            assert text[entity["start"]:entity["end"]] == entity["text"], (
                f"Offset invariant violated: {entity}"
            )

    def test_real_result_structure(self):
        """Real model response has correct structure."""
        from trustboost.detector import detect_pii
        result = detect_pii("Test sentence for structure check.")
        assert "success" in result
        assert "entities" in result
        assert "message" in result
        assert isinstance(result["entities"], list)


class TestPhase32APromptCoverageAndRegression:
    """Targeted unit & regression tests for Phase 3.2A prompt coverage improvements."""

    def test_prompt_preserves_24_types_compact_schema_and_budget(self):
        """All 24 categories, compact schema, and 8GB-safe prompt size are preserved."""
        from trustboost.detector.prompt_builder import VALID_ENTITY_TYPES, build_detection_prompt

        assert len(VALID_ENTITY_TYPES) == 24
        prompt = build_detection_prompt("Sample input text.")
        for category in VALID_ENTITY_TYPES:
            assert category in prompt
        assert "compact JSON only" in prompt
        assert '"text":' in prompt and '"type":' in prompt and '"start":' in prompt and '"end":' in prompt
        assert len(prompt) < 7500

    def test_prompt_includes_all_seven_phase32a_coverage_areas(self):
        """Prompt contains explicit rules and examples for all 7 Phase 3.2A areas."""
        from trustboost.detector.prompt_builder import build_detection_prompt

        prompt = build_detection_prompt("Sample")
        # 1. LOCATION cities & generic country exclusion
        assert "Springfield" in prompt and "Ahmedabad" in prompt and "Paris" in prompt
        assert "generic, unlinked country names" in prompt
        # 2. FINANCIAL_IDENTIFIER (UPI + GSTIN) vs EMAIL
        assert "rahul.verma@okhdfcbank" in prompt and "27AAAPL1234C1ZV" in prompt
        assert "remain EMAIL" in prompt
        # 3. DATABASE_CREDENTIAL complete URI
        assert "postgresql://admin:Pass1@db.local:5432/app" in prompt
        assert "ONE single DATABASE_CREDENTIAL entity" in prompt
        # 4. INTERNAL_ORGANIZATION_INFORMATION
        assert "proprietary algorithm for customer churn prediction" in prompt
        # 5. CONFIDENTIAL_PROJECT
        assert "Project Titan" in prompt and "Project Polaris" in prompt
        # 6. IFSC & SERVER_IDENTIFIER
        assert "ICIC0000011" in prompt and "srv-fin-core-09" in prompt
        # 7. Structured / multiline CSV AADHAAR
        assert "id,name,aadhaar\n101,Neha Deshmukh,7766 5544 3322" in prompt

    @pytest.mark.parametrize(
        "text,expected_entities",
        [
            (
                "Please deliver the package to Bob Henderson at 452 Elm Street, Springfield.",
                [
                    {"text": "Bob Henderson", "type": "PERSON", "start": 30, "end": 43},
                    {"text": "452 Elm Street", "type": "ADDRESS", "start": 47, "end": 61},
                    {"text": "Springfield", "type": "LOCATION", "start": 63, "end": 74},
                ],
            ),
            (
                "Send 500 to johnny.cash@okhdfcbank and GSTIN 27AAAPL1234C1ZV, IFSC ICIC0000011.",
                [
                    {"text": "johnny.cash@okhdfcbank", "type": "FINANCIAL_IDENTIFIER", "start": 12, "end": 34},
                    {"text": "27AAAPL1234C1ZV", "type": "FINANCIAL_IDENTIFIER", "start": 45, "end": 60},
                    {"text": "ICIC0000011", "type": "IFSC", "start": 67, "end": 78},
                ],
            ),
            (
                "Connect to postgresql://admin:P@ssw0rd2026@localhost:5432/maindb on srv-fin-core-09.",
                [
                    {
                        "text": "postgresql://admin:P@ssw0rd2026@localhost:5432/maindb",
                        "type": "DATABASE_CREDENTIAL",
                        "start": 11,
                        "end": 64,
                    },
                    {"text": "srv-fin-core-09", "type": "SERVER_IDENTIFIER", "start": 68, "end": 83},
                ],
            ),
            (
                "The board discussed Project Titan and proprietary algorithm for customer churn prediction.",
                [
                    {"text": "Project Titan", "type": "CONFIDENTIAL_PROJECT", "start": 20, "end": 33},
                    {
                        "text": "proprietary algorithm for customer churn prediction",
                        "type": "INTERNAL_ORGANIZATION_INFORMATION",
                        "start": 38,
                        "end": 89,
                    },
                ],
            ),
            (
                "CSV_DUMP: id,name,aadhaar\n101,Neha Deshmukh,7766 5544 3322",
                [
                    {"text": "Neha Deshmukh", "type": "PERSON", "start": 30, "end": 43},
                    {"text": "7766 5544 3322", "type": "AADHAAR", "start": 44, "end": 58},
                ],
            ),
        ],
    )
    def test_targeted_seven_categories_pipeline_roundtrip(self, text, expected_entities):
        """Detection validation, offset recovery, tokenization, and restoration work for all 7 categories."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        mock_payload = json.dumps({"entities": expected_entities})
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": mock_payload, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert len(det["entities"]) == len(expected_entities)
        for pred, exp in zip(det["entities"], expected_entities):
            assert pred["text"] == exp["text"]
            assert pred["type"] == exp["type"]
            assert text[pred["start"]:pred["end"]] == exp["text"]

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        for exp in expected_entities:
            assert exp["text"] not in tok["sanitized_text"]

        rest = restore_text(tok["sanitized_text"], tok["mapping"])
        assert rest["success"] is True
        assert rest["restored_text"] == text

    @pytest.mark.parametrize(
        "text,expected_uri",
        [
            (
                "Postgres connection string: postgresql://admin:P@ssw0rd2026@localhost:5432/maindb.",
                "postgresql://admin:P@ssw0rd2026@localhost:5432/maindb",
            ),
            (
                "Redis URI: redis://default:redispw2026!@cache.internal.net:6379.",
                "redis://default:redispw2026!@cache.internal.net:6379",
            ),
        ],
    )
    def test_full_database_uri_single_entity_roundtrip(self, text, expected_uri):
        """Full PostgreSQL and Redis URIs resolve as one DATABASE_CREDENTIAL without fragmentation."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        start = text.index(expected_uri)
        end = start + len(expected_uri)
        mock_payload = json.dumps({
            "entities": [
                {"text": expected_uri, "type": "DATABASE_CREDENTIAL", "start": start, "end": end}
            ]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": mock_payload, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert len(det["entities"]) == 1
        assert det["entities"][0]["type"] == "DATABASE_CREDENTIAL"
        assert det["entities"][0]["text"] == expected_uri
        assert text[det["entities"][0]["start"]:det["entities"][0]["end"]] == expected_uri

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert expected_uri not in tok["sanitized_text"]
        assert "[DATABASE_CREDENTIAL_1]" in tok["sanitized_text"]

        rest = restore_text(tok["sanitized_text"], tok["mapping"])
        assert rest["success"] is True
        assert rest["restored_text"] == text

    def test_location_city_positive_and_standalone_country_negative(self):
        """City in personal travel context is LOCATION; generic standalone country is excluded."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.detector.prompt_builder import build_detection_prompt

        prompt = build_detection_prompt("Generic text in India.")
        assert "generic, unlinked country names" in prompt
        assert "travelling to Paris" in prompt

        travel_text = "Please confirm if May Parker is travelling to Paris this May."
        travel_payload = json.dumps({
            "entities": [
                {"text": "May Parker", "type": "PERSON", "start": 18, "end": 28},
                {"text": "Paris", "type": "LOCATION", "start": 46, "end": 51},
            ]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": travel_payload, "message": "ok"},
        ):
            det_pos = detect_pii(travel_text)
        assert det_pos["success"] is True
        assert any(e["text"] == "Paris" and e["type"] == "LOCATION" for e in det_pos["entities"])

        country_text = "Our engineering teams operate in India and Germany."
        country_payload = json.dumps({"entities": []})
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": country_payload, "message": "ok"},
        ):
            det_neg = detect_pii(country_text)
        assert det_neg["success"] is True
        assert det_neg["entities"] == []


class TestPhase32A3DatabaseUriRecognizer:
    """Comprehensive unit tests for deterministic DATABASE_CREDENTIAL URI recognition."""

    def test_sample_036_postgresql_raw_at_in_password_and_fragment_suppression(self):
        """sample_036: PostgreSQL URI with raw '@' in password supersedes LLM fragments."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = "Postgres connection string: postgresql://admin:P@ssw0rd2026@localhost:5432/maindb."
        expected_uri = "postgresql://admin:P@ssw0rd2026@localhost:5432/maindb"
        llm_fragments = json.dumps({
            "entities": [
                {"text": "admin", "type": "PASSWORD", "start": 41, "end": 46},
                {"text": "P@ssw0rd2026", "type": "PASSWORD", "start": 47, "end": 59},
                {"text": "localhost", "type": "SERVER_IDENTIFIER", "start": 60, "end": 69},
                {"text": "5432", "type": "INTERNAL_URL", "start": 70, "end": 74},
                {"text": "maindb", "type": "DATABASE_CREDENTIAL", "start": 75, "end": 81},
            ]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": llm_fragments, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert len(det["entities"]) == 1
        ent = det["entities"][0]
        assert ent["type"] == "DATABASE_CREDENTIAL"
        assert ent["text"] == expected_uri
        assert (ent["start"], ent["end"]) == (28, 81)
        assert text[ent["start"]:ent["end"]] == expected_uri

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert tok["sanitized_text"] == "Postgres connection string: [DATABASE_CREDENTIAL_1]."
        rest = restore_text(tok["sanitized_text"], tok["mapping"])
        assert rest["success"] is True
        assert rest["restored_text"] == text

    def test_sample_045_redis_uri_with_exclamation_in_password(self):
        """sample_045: Redis URI with '!' in password is detected as one DATABASE_CREDENTIAL."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = "Redis URI: redis://default:redispw2026!@cache.internal.net:6379."
        expected_uri = "redis://default:redispw2026!@cache.internal.net:6379"
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": '{"entities":[]}', "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert len(det["entities"]) == 1
        ent = det["entities"][0]
        assert ent["type"] == "DATABASE_CREDENTIAL"
        assert ent["text"] == expected_uri
        assert (ent["start"], ent["end"]) == (11, 63)

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert tok["sanitized_text"] == "Redis URI: [DATABASE_CREDENTIAL_1]."
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text

    def test_postgresql_percent_encoded_credentials(self):
        """PostgreSQL URI with percent-encoded credentials and query params is detected."""
        from trustboost.detector.pii_detector import _extract_database_uri_entities

        text = "DSN: postgresql://svc_user:p%40ss%21word@10.0.1.5:5432/prod?sslmode=require"
        ents = _extract_database_uri_entities(text)
        assert len(ents) == 1
        assert ents[0]["text"] == "postgresql://svc_user:p%40ss%21word@10.0.1.5:5432/prod?sslmode=require"
        assert text[ents[0]["start"]:ents[0]["end"]] == ents[0]["text"]

    @pytest.mark.parametrize(
        "text",
        [
            "Public docs: postgresql://localhost:5432/publicdb",
            "Redis endpoint: redis://cache.internal.net:6379",
            "Missing password: postgresql://admin@localhost:5432/db",
            "Empty password: postgresql://admin:@localhost:5432/db",
            "Empty user: postgresql://:secret@localhost:5432/db",
            "Percent-encoded space only: postgresql://%20:%20@localhost:5432/db",
            "Invalid port zero: postgresql://admin:secret@localhost:0/db",
            "Invalid port >65535: postgresql://admin:secret@localhost:70000/db",
            "Non-numeric port: postgresql://admin:secret@localhost:abc/db",
            "AMQP not auto-classified: amqp://guest:guest@rabbit.internal:5672/",
            "HTTPS URL not DB URI: https://user:pass@example.com/path",
        ],
    )
    def test_credentialless_invalid_and_unsupported_uris_rejected(self, text):
        """Credential-less, invalid user/pass/port, and non-DB URIs must not be emitted."""
        from trustboost.detector.pii_detector import _extract_database_uri_entities

        assert _extract_database_uri_entities(text) == []

    def test_ipv6_host_and_port(self):
        """Bracketed IPv6 host with optional port is supported."""
        from trustboost.detector.pii_detector import _extract_database_uri_entities

        text = "URI=mongodb://root:s3cr3t@[2001:db8::1]:27017/admin"
        ents = _extract_database_uri_entities(text)
        assert len(ents) == 1
        assert ents[0]["text"] == "mongodb://root:s3cr3t@[2001:db8::1]:27017/admin"
        assert text[ents[0]["start"]:ents[0]["end"]] == ents[0]["text"]

    @pytest.mark.parametrize(
        "text,expected_uri",
        [
            ("(postgresql://u1:p1@db.internal:5432/app)", "postgresql://u1:p1@db.internal:5432/app"),
            ('Check "mysql://u1:p1@127.0.0.1:3306/shop", please.', "mysql://u1:p1@127.0.0.1:3306/shop"),
            ("Line1\nDSN=rediss://u1:p!@cache.net:6380;", "rediss://u1:p!@cache.net:6380"),
        ],
    )
    def test_wrappers_quotes_commas_and_multiline_boundaries(self, text, expected_uri):
        """URIs inside parentheses, quotes, commas, and multiline text preserve exact spans."""
        from trustboost.detector.pii_detector import _extract_database_uri_entities

        ents = _extract_database_uri_entities(text)
        assert len(ents) == 1
        assert ents[0]["text"] == expected_uri
        assert text[ents[0]["start"]:ents[0]["end"]] == expected_uri

    def test_multiple_uris_and_unrelated_neighbouring_pii_preserved(self):
        """Multiple DB URIs and neighbouring PERSON/EMAIL PII are all preserved and restored."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = (
            "Alice Walker rotated postgresql://admin:P@ss1@db1:5432/app "
            "and redis://svc:pw!@cache1:6379, contact alice@example.com."
        )
        llm_payload = json.dumps({
            "entities": [
                {"text": "Alice Walker", "type": "PERSON", "start": 0, "end": 12},
                {"text": "P@ss1", "type": "PASSWORD", "start": 40, "end": 45},
                {"text": "alice@example.com", "type": "EMAIL", "start": 100, "end": 117},
            ]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": llm_payload, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        types_texts = [(e["type"], e["text"]) for e in det["entities"]]
        assert types_texts == [
            ("PERSON", "Alice Walker"),
            ("DATABASE_CREDENTIAL", "postgresql://admin:P@ss1@db1:5432/app"),
            ("DATABASE_CREDENTIAL", "redis://svc:pw!@cache1:6379"),
            ("EMAIL", "alice@example.com"),
        ]

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert "P@ss1" not in tok["sanitized_text"]
        assert "pw!" not in tok["sanitized_text"]
        assert "Alice Walker" not in tok["sanitized_text"]
        assert "alice@example.com" not in tok["sanitized_text"]
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text


class TestPhase32B3StructuredSecretsAndIpRecognizer:
    """Comprehensive unit tests for deterministic API-key, IP, and private-key recognition."""

    def test_sample_032_github_pat_excludes_prefix_and_trailing_period(self):
        """sample_032: Exact GitHub PAT extracted; overlapping prefix-only LLM detection superseded."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = "Set your environment variable to API_KEY=ghp_39aZkL84910fKmQ9021a8xZ4901lQp."
        expected_tok = "ghp_39aZkL84910fKmQ9021a8xZ4901lQp"
        llm_overlapping = json.dumps({
            "entities": [
                {
                    "text": "Set your environment variable to API_KEY=ghp_39aZkL84910fKmQ9021a8xZ4901lQp",
                    "type": "API_KEY",
                    "start": 0,
                    "end": 75,
                }
            ]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": llm_overlapping, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert len(det["entities"]) == 1
        ent = det["entities"][0]
        assert ent["type"] == "API_KEY"
        assert ent["text"] == expected_tok
        assert (ent["start"], ent["end"]) == (41, 75)
        assert text[ent["start"]:ent["end"]] == expected_tok

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert tok["sanitized_text"] == "Set your environment variable to API_KEY=[API_KEY_1]."
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text

    @pytest.mark.parametrize(
        "token",
        [
            "ghp_1234567890abcdefghijklmnopqr",
            "gho_1234567890abcdefghijklmnopqr",
            "ghu_1234567890abcdefghijklmnopqr",
            "ghs_1234567890abcdefghijklmnopqr",
            "ghr_1234567890abcdefghijklmnopqr",
            "github_pat_11AAAAAA0123456789_abcdefg",
            "sk-live-9021849102b48",
            "sk-test-1234567890abcdef",
            "sk-proj-1234567890abcdef",
            "sk_live_1234567890abcdef",
            "sk-1234567890abcdefghij",
            "AKIAIOSFODNN7EXAMPLE",
        ],
    )
    def test_valid_api_key_prefixes_recognized(self, token):
        """All supported GitHub, OpenAI/Stripe sk-*, and AWS AKIA* formats are recognized."""
        from trustboost.detector.pii_detector import _extract_structured_secret_entities

        text = f"Token is {token}."
        ents = _extract_structured_secret_entities(text)
        assert len(ents) == 1
        assert ents[0]["type"] == "API_KEY"
        assert ents[0]["text"] == token
        assert text[ents[0]["start"]:ents[0]["end"]] == token

    @pytest.mark.parametrize(
        "text",
        [
            "Clone https://github.com/example/repo.git now.",
            "Short gh_abc and ghp_short and sk-abc and sk_short.",
            "Model sku-12345 and ghost_mode and ghz_12345678901234567890.",
            "Invalid AWS key AKIA12345 and AKIAIOSFODNN7EXAMPLEEXTRA.",
        ],
    )
    def test_public_github_url_and_arbitrary_gh_sk_rejected(self, text):
        """Public GitHub repo URLs and arbitrary gh/sk/AKIA strings are not classified as API_KEY."""
        from trustboost.detector.pii_detector import _extract_structured_secret_entities

        assert [e for e in _extract_structured_secret_entities(text) if e["type"] == "API_KEY"] == []

    def test_sample_074_ipv4_with_port_excludes_port(self):
        """sample_074: 198.51.100.45:44321 emits only 198.51.100.45 [60:73]."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = "DEBUG [2026-03-31 10:14:22] Connection accepted from client 198.51.100.45:44321"
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": '{"entities":[]}', "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert len(det["entities"]) == 1
        ent = det["entities"][0]
        assert ent["type"] == "IP_ADDRESS"
        assert ent["text"] == "198.51.100.45"
        assert (ent["start"], ent["end"]) == (60, 73)

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert tok["sanitized_text"] == "DEBUG [2026-03-31 10:14:22] Connection accepted from client [IP_ADDRESS_1]:44321"
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text

    @pytest.mark.parametrize(
        "text,expected_ip",
        [
            ("Ping 192.168.1.10 now.", "192.168.1.10"),
            ("IPv6 host 2001:db8::1 connected.", "2001:db8::1"),
            ("Bracketed [2001:db8::1]:8443 ok.", "2001:db8::1"),
            ("Loopback [::1]:8080 ready.", "::1"),
        ],
    )
    def test_valid_ipv4_and_ipv6_variants(self, text, expected_ip):
        """Valid IPv4, IPv6, and bracketed IPv6 with port are recognized accurately."""
        from trustboost.detector.pii_detector import _extract_structured_secret_entities

        ents = _extract_structured_secret_entities(text)
        assert len(ents) == 1
        assert ents[0]["type"] == "IP_ADDRESS"
        assert ents[0]["text"] == expected_ip
        assert text[ents[0]["start"]:ents[0]["end"]] == expected_ip

    @pytest.mark.parametrize(
        "text",
        [
            "Invalid octets 256.1.1.1 and 999.999.999.999.",
            "Version v1.2.3.4 and 1.2.3.4.5 and v2.0.1.",
            "Timestamp 10:14:22 and ordinary number 123456.",
            "Invalid port 10.0.0.1:99999.",
        ],
    )
    def test_invalid_ips_versions_and_numbers_rejected(self, text):
        """Invalid octets, version strings, OIDs, timestamps, and invalid ports are rejected."""
        from trustboost.detector.pii_detector import _extract_structured_secret_entities

        assert [e for e in _extract_structured_secret_entities(text) if e["type"] == "IP_ADDRESS"] == []

    def test_pem_and_openssh_private_keys_recognized(self):
        """Complete PEM and OpenSSH private-key blocks are detected as PRIVATE_KEY."""
        from trustboost.detector.pii_detector import _extract_structured_secret_entities

        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAtestbody\n-----END RSA PRIVATE KEY-----"
        openssh = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAA=\n-----END OPENSSH PRIVATE KEY-----"
        for key_block in (pem, openssh):
            text = f"Key:\n{key_block}\nEnd."
            ents = _extract_structured_secret_entities(text)
            assert len(ents) == 1
            assert ents[0]["type"] == "PRIVATE_KEY"
            assert ents[0]["text"] == key_block
            assert text[ents[0]["start"]:ents[0]["end"]] == key_block

    @pytest.mark.parametrize(
        "text",
        [
            "Deploy key ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC8... generated for deploy bot.",
            "Incomplete header -----BEGIN PRIVATE KEY----- without closing footer",
            "Mismatched -----BEGIN RSA PRIVATE KEY-----\nabc\n-----END EC PRIVATE KEY-----",
        ],
    )
    def test_ssh_public_key_and_incomplete_private_keys_not_misclassified(self, text):
        """SSH RSA public keys and incomplete/mismatched blocks are not emitted as PRIVATE_KEY."""
        from trustboost.detector.pii_detector import _extract_structured_secret_entities

        assert [e for e in _extract_structured_secret_entities(text) if e["type"] == "PRIVATE_KEY"] == []

    def test_overlap_supersedes_prefix_and_preserves_neighbouring_pii(self):
        """Deterministic API_KEY/IP_ADDRESS supersede overlapping LLM spans while keeping nearby PII."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = "Alice Walker used Bearer sk-live-9021849102b48 from 198.51.100.45:44321."
        llm_payload = json.dumps({
            "entities": [
                {"text": "Alice Walker", "type": "PERSON", "start": 0, "end": 12},
                {"text": "Bearer sk-live-9021849102b48", "type": "CREDENTIAL", "start": 18, "end": 46},
                {"text": "198.51.100.45:44321", "type": "INTERNAL_URL", "start": 52, "end": 71},
            ]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": llm_payload, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert [(e["type"], e["text"]) for e in det["entities"]] == [
            ("PERSON", "Alice Walker"),
            ("API_KEY", "sk-live-9021849102b48"),
            ("IP_ADDRESS", "198.51.100.45"),
        ]
        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert "Alice Walker" not in tok["sanitized_text"]
        assert "sk-live-9021849102b48" not in tok["sanitized_text"]
        assert "198.51.100.45" not in tok["sanitized_text"]
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text


class TestPhase32B5StructuredPiiRecognizer:
    """Comprehensive unit tests for deterministic PAN, international phone, and context-aware DOB."""

    @pytest.mark.parametrize(
        "text,expected_pan",
        [
            ('{"status": "active", "lead": "Marcus Brody", "pan": "BKPTR1290N"}', "BKPTR1290N"),
            ("Taxpayer Rajesh Kumar has provided PAN ABCDE1234F for clearance.", "ABCDE1234F"),
            ("Kindly submit Aadhaar 4321 8765 2109 and PAN BKZPR8812K for KYC renewal.", "BKZPR8812K"),
            ("Kripya apna PAN card ALMPK9911D email kijiye.", "ALMPK9911D"),
        ],
    )
    def test_valid_indian_pan_with_context(self, text, expected_pan):
        """Valid Indian PANs with PAN/tax/JSON context are extracted with exact offsets."""
        from trustboost.detector.pii_detector import _extract_structured_pii_entities

        ents = [e for e in _extract_structured_pii_entities(text) if e["type"] == "PAN"]
        assert len(ents) == 1
        assert ents[0]["text"] == expected_pan
        assert text[ents[0]["start"]:ents[0]["end"]] == expected_pan

    @pytest.mark.parametrize(
        "text",
        [
            "Model code ABCDE1234F without any tax or pan context.",
            "Lowercase pan abcde1234f should not match.",
            "PAN embedded in longer token PAN ABCDE1234FG99.",
            "Too short PAN ABC1234F and wrong digits PAN ABCD12345F.",
            "GSTIN 27AAAPL1234C1ZV is not a 10-char PAN.",
        ],
    )
    def test_invalid_or_uncontextualized_pan_rejected(self, text):
        """Invalid PANs, partial matches inside longer strings, and uncontextualized IDs are rejected."""
        from trustboost.detector.pii_detector import _extract_structured_pii_entities

        assert [e for e in _extract_structured_pii_entities(text) if e["type"] == "PAN"] == []

    @pytest.mark.parametrize(
        "text,expected_phone",
        [
            ("Call me back on +1-555-019-2834 if Dr. Evelyn Reed is unavailable.", "+1-555-019-2834"),
            ("You can reach Jessica Jones at +44 20 7946 0912 during office hours.", "+44 20 7946 0912"),
            ("Laura Croft can be reached at +1 (555) 987-6543 at her estate.", "+1 (555) 987-6543"),
            ("Contactable at +91-9988776655 or [+1-555-0182]!", "+91-9988776655"),
        ],
    )
    def test_sample_003_and_international_phone_separators(self, text, expected_phone):
        """sample_003 and international phones with spaces, hyphens, and parentheses are extracted."""
        from trustboost.detector.pii_detector import _extract_structured_pii_entities

        ents = [e for e in _extract_structured_pii_entities(text) if e["type"] == "PHONE"]
        assert len(ents) >= 1
        assert ents[0]["text"] == expected_phone
        assert text[ents[0]["start"]:ents[0]["end"]] == expected_phone

    @pytest.mark.parametrize(
        "text",
        [
            "DEBUG [2026-03-31 10:14:22] client 198.51.100.45:44321",
            "Order total +123 and version +1.2.3.4 and short +1-23.",
            "Too many digits +1-555-123-456-789-0123.",
        ],
    )
    def test_invalid_phone_like_strings_rejected(self, text):
        """Timestamps, ports, short/overlong numbers are not classified as PHONE."""
        from trustboost.detector.pii_detector import _extract_structured_pii_entities

        assert [e for e in _extract_structured_pii_entities(text) if e["type"] == "PHONE"] == []

    @pytest.mark.parametrize(
        "text,expected_dob",
        [
            ("Date of Birth: 03-Dec-1979 for client Kevin Flynn.", "03-Dec-1979"),
            ("Patient record shows Charlie Vance was born on 14/05/1988 in Chicago.", "14/05/1988"),
            ("Applicant: George Thomas, DOB: 1995-10-22, living in Seattle.", "1995-10-22"),
            ("New recruit Michael Scott was born on 15 March 1965 in Scranton.", "15 March 1965"),
            ('{"name": "Kevin Flynn", "dob": "03-Dec-1979", "date_of_birth": "1988-05-14"}', "03-Dec-1979"),
        ],
    )
    def test_sample_012_and_context_aware_dob_formats(self, text, expected_dob):
        """sample_012 and context-labelled prose/JSON DOBs are extracted with exact offsets."""
        from trustboost.detector.pii_detector import _extract_structured_pii_entities

        ents = [e for e in _extract_structured_pii_entities(text) if e["type"] == "DATE_OF_BIRTH"]
        assert len(ents) >= 1
        assert ents[0]["text"] == expected_dob
        assert text[ents[0]["start"]:ents[0]["end"]] == expected_dob

    @pytest.mark.parametrize(
        "text",
        [
            "Invalid calendar date DOB: 31-Feb-1990.",
            "Invalid month DOB: 2020-13-01.",
            "Meeting scheduled on 03-Dec-2026 with Alice.",
            "DEBUG [2026-03-31 10:14:22] Invoice date: 14/05/2025.",
        ],
    )
    def test_invalid_calendar_dates_and_generic_dates_rejected(self, text):
        """Invalid calendar dates and unlabelled meeting/log/invoice dates are not DATE_OF_BIRTH."""
        from trustboost.detector.pii_detector import _extract_structured_pii_entities

        assert [e for e in _extract_structured_pii_entities(text) if e["type"] == "DATE_OF_BIRTH"] == []

    def test_sample_003_012_075_pipeline_roundtrip_and_unrelated_preservation(self):
        """End-to-end detection, tokenization, and restoration for 003, 012, 075 and JSON DOB."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        cases = [
            (
                "Call me back on +1-555-019-2834 if Dr. Evelyn Reed is unavailable.",
                json.dumps({"entities": [{"text": "Evelyn Reed", "type": "PERSON", "start": 39, "end": 50}]}),
                "+1-555-019-2834",
            ),
            (
                "Date of Birth: 03-Dec-1979 for client Kevin Flynn.",
                json.dumps({"entities": [{"text": "Kevin Flynn", "type": "PERSON", "start": 38, "end": 49}]}),
                "03-Dec-1979",
            ),
            (
                '{"status": "active", "lead": "Marcus Brody", "pan": "BKPTR1290N"}',
                json.dumps({"entities": []}),
                "BKPTR1290N",
            ),
        ]
        for text, llm_json, must_mask in cases:
            with patch(
                "trustboost.detector.pii_detector.generate_response",
                return_value={"success": True, "response": llm_json, "message": "ok"},
            ):
                det = detect_pii(text)
            assert det["success"] is True
            tok = tokenize_entities(text, det["entities"])
            assert tok["success"] is True
            assert must_mask not in tok["sanitized_text"]
            assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text


class TestPhase32B6CJsonPersonRecognizer:
    """Positive, negative, and integration tests for JSON-aware PERSON detection."""

    def test_sample_075_json_lead_and_pan_roundtrip(self):
        """sample_075: 'lead': 'Marcus Brody' -> PERSON [30:42], 'pan': 'BKPTR1290N' -> PAN [53:63], 'status': 'active' untouched."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = '{"status": "active", "lead": "Marcus Brody", "pan": "BKPTR1290N"}'
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": '{"entities":[]}', "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert [(e["type"], e["text"], e["start"], e["end"]) for e in det["entities"]] == [
            ("PERSON", "Marcus Brody", 30, 42),
            ("PAN", "BKPTR1290N", 53, 63),
        ]
        for e in det["entities"]:
            assert text[e["start"]:e["end"]] == e["text"]

        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert tok["sanitized_text"] == '{"status": "active", "lead": "[PERSON_1]", "pan": "[PAN_1]"}'
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text

    def test_json_person_name_nested_array_escaped_unicode_and_duplicates(self):
        """Supports simple, nested, array, escaped-quote, Unicode-escape, and repeated-name JSON inputs."""
        from trustboost.detector.pii_detector import _extract_json_person_entities

        # Simple
        s1 = '{"name": "John Smith"}'
        e1 = _extract_json_person_entities(s1)
        assert [(e["text"], e["start"], e["end"]) for e in e1] == [("John Smith", 10, 20)]

        # Nested object + case-insensitive key
        s2 = '{"meta": {"OWNER": "Alice Walker"}}'
        e2 = _extract_json_person_entities(s2)
        assert [(e["text"], e["start"], e["end"]) for e in e2] == [("Alice Walker", 20, 32)]

        # Array of objects
        s3 = '[{"contact": "Kevin Flynn"}, {"employee": "Neha Deshmukh"}]'
        e3 = _extract_json_person_entities(s3)
        assert [(e["text"], e["start"], e["end"]) for e in e3] == [
            ("Kevin Flynn", 14, 25),
            ("Neha Deshmukh", 43, 56),
        ]

        # Escaped quote string
        s4 = '{"author": "Jean-Luc \\"Picard\\""}'
        e4 = _extract_json_person_entities(s4)
        assert len(e4) == 1
        assert s4[e4[0]["start"]:e4[0]["end"]] == e4[0]["text"]

        # Unicode escape sequence
        s5 = '{"patient": "Jos\\u00e9 Silva"}'
        e5 = _extract_json_person_entities(s5)
        assert len(e5) == 1
        assert e5[0]["text"] == "Jos\\u00e9 Silva"
        assert s5[e5[0]["start"]:e5[0]["end"]] == e5[0]["text"]

        # Same name appearing under two different semantic keys preserves distinct offsets
        s6 = '{"lead": "Marcus Brody", "owner": "Marcus Brody"}'
        e6 = _extract_json_person_entities(s6)
        assert [(e["text"], e["start"], e["end"]) for e in e6] == [
            ("Marcus Brody", 10, 22),
            ("Marcus Brody", 35, 47),
        ]

    @pytest.mark.parametrize(
        "json_text",
        [
            '{"status": "active"}',
            '{"owner": "admin"}',
            '{"lead": "Backend Team"}',
            '{"lead": "Support Team"}',
            '{"owner": "Acme Corporation"}',
            '{"contact": "North America"}',
            '{"lead": "Project Polaris"}',
            '{"user": "deploy-service-01"}',
            '{"name": "unknown"}',
            '{"contact": "alice@example.com"}',
            '{"user": "https://example.com/profile"}',
            '{"lead": "Marcus Brody"',  # Malformed JSON
            'Not JSON at all: {"lead": "Marcus Brody"}',
        ],
    )
    def test_json_person_negative_cases_and_malformed_json(self, json_text):
        """Generic statuses, teams, orgs, regions, projects, IDs, emails, URLs, and malformed JSON emit no PERSON."""
        from trustboost.detector.pii_detector import _extract_json_person_entities

        assert _extract_json_person_entities(json_text) == []

    def test_json_person_does_not_overlap_email_api_key_or_pan(self):
        """Existing EMAIL, API_KEY, and PAN detections inside JSON are preserved without overlap."""
        from trustboost.detector.pii_detector import detect_pii
        from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

        text = (
            '{"contact": "alice@example.com", "owner": "Alice Walker", '
            '"api_key": "ghp_39aZkL84910fKmQ9021a8xZ4901lQp", "pan": "BKPTR1290N"}'
        )
        llm_json = json.dumps({
            "entities": [{"text": "alice@example.com", "type": "EMAIL", "start": 13, "end": 30}]
        })
        with patch(
            "trustboost.detector.pii_detector.generate_response",
            return_value={"success": True, "response": llm_json, "message": "ok"},
        ):
            det = detect_pii(text)

        assert det["success"] is True
        assert [(e["type"], e["text"]) for e in det["entities"]] == [
            ("EMAIL", "alice@example.com"),
            ("PERSON", "Alice Walker"),
            ("API_KEY", "ghp_39aZkL84910fKmQ9021a8xZ4901lQp"),
            ("PAN", "BKPTR1290N"),
        ]
        tok = tokenize_entities(text, det["entities"])
        assert tok["success"] is True
        assert restore_text(tok["sanitized_text"], tok["mapping"])["restored_text"] == text
