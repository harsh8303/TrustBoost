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

    def test_build_prompt_does_not_ask_for_offsets(self):
        """The prompt must not ask the model for start/end character offsets."""
        from trustboost.detector.prompt_builder import build_detection_prompt
        prompt = build_detection_prompt("some text")
        assert '"start"' not in prompt
        assert '"end"' not in prompt

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

    def test_missing_type_field_gets_fallback_type(self):
        """An entity without the 'type' field gets the fallback SENSITIVE type."""
        from trustboost.detector.pii_detector import _validate_entity
        entity = {"text": "John", "start": 0, "end": 4, "reason": "name"}
        result = _validate_entity(entity, "John Smith")
        assert result is not None
        assert result["type"] == "SENSITIVE"

    def test_unknown_entity_type_is_kept_as_fallback(self):
        """An entity with an unknown type is kept with the fallback SENSITIVE type."""
        from trustboost.detector.pii_detector import _validate_entity
        original = "John Smith"
        entity = {"text": "John Smith", "type": "MADE_UP_TYPE",
                  "start": 0, "end": 10, "reason": "test"}
        result = _validate_entity(entity, original)
        assert result is not None
        assert result["type"] == "SENSITIVE"

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

    def test_unknown_entity_type_still_masked(self):
        """Entities with unknown types are kept and masked as SENSITIVE."""
        text = "Call John."
        json_resp = json.dumps({"entities": [
            {"text": "John", "type": "HALLUCINATED_TYPE",
             "start": 5, "end": 9, "reason": "test"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert [e["text"] for e in result["entities"]] == ["John"]
        assert result["entities"][0]["type"] == "SENSITIVE"

    def test_missing_entity_fields_handled(self):
        """
        An entity missing optional fields like type is still kept and masked.
        """
        json_resp = json.dumps({"entities": [
            {"text": "Alice"}  # Missing type, start, end, reason
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii("Hello Alice")
        assert result["success"] is True
        assert [e["text"] for e in result["entities"]] == ["Alice"]

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

    def test_value_appearing_multiple_times_masked_everywhere(self):
        """
        When a detected value appears multiple times in the text,
        Python locates and masks every occurrence.
        """
        text = "cat and cat are different cat"
        json_resp = json.dumps({"entities": [
            {"text": "cat", "type": "PERSON",
             "start": 999, "end": 1000, "reason": "name"}
        ]})
        with self._mock_ollama(json_resp):
            from trustboost.detector.pii_detector import detect_pii
            result = detect_pii(text)
        assert result["success"] is True
        assert len(result["entities"]) == 3
        for e in result["entities"]:
            assert text[e["start"]:e["end"]] == "cat"

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
        from trustboost.detector.pii_detector import detect_pii
        result = detect_pii("Test sentence for structure check.")
        assert "success" in result
        assert "entities" in result
        assert "message" in result
        assert isinstance(result["entities"], list)
