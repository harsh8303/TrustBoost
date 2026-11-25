"""
tests/test_leakage_fixes.py
---------------------------
Regression tests for every leakage bug found in the Phase 2 audit.
Each test fakes the model's answer, runs the REAL pipeline, and checks that
the private value does not appear anywhere in the sanitized text.

Run:  pytest tests/test_leakage_fixes.py -v
"""

import json
from unittest.mock import patch

import pytest

from trustboost.pipeline.pipeline import run_sanitization_pipeline
from trustboost.session.global_mapping import clear_global_mapping


def fake_model(entities):
    """Makes the model 'return' the given entities, without calling Ollama."""
    return patch(
        "trustboost.detector.pii_detector.generate_response",
        return_value={"success": True, "response": json.dumps({"entities": entities}),
                      "message": "ok", "done_reason": "stop"},
    )


def sanitize(text, entities, patterns=False):
    with fake_model(entities), \
         patch("trustboost.detector.pii_detector.USE_PATTERN_BACKUP", patterns):
        result = run_sanitization_pipeline(text, clear_global_mapping())
    assert result["success"] is True
    return result["sanitized_text"]


def test_lowercase_or_unknown_type_is_still_masked():
    text = "My Aadhaar is 2345 6789 0123 and my name is Priya."
    out = sanitize(text, [{"text": "2345 6789 0123", "type": "aadhaar"},
                          {"text": "Priya", "type": "NAME"}])
    assert "2345 6789 0123" not in out
    assert "Priya" not in out


def test_every_occurrence_is_masked():
    text = "Rahul Sharma called. Please email Rahul Sharma back today."
    out = sanitize(text, [{"text": "Rahul Sharma", "type": "PERSON"}])
    assert "Rahul Sharma" not in out
    assert out.count("[PERSON_1]") == 2


def test_model_offsets_are_ignored():
    text = "Rahul Sharma called. Please email Rahul Sharma back today."
    out = sanitize(text, [{"text": "Rahul Sharma", "type": "PERSON", "start": 3, "end": 15}])
    assert "Rahul Sharma" not in out


def test_changed_spacing_from_model_is_still_found():
    text = "Card no:  4111  1111  1111  1111 expires soon."
    out = sanitize(text, [{"text": "4111 1111 1111 1111", "type": "CREDIT_CARD"}])
    assert "1111" not in out


def test_password_trailing_symbol_is_masked():
    text = "Login password Winter#2026!. Change it today."
    out = sanitize(text, [{"text": "Winter#2026", "type": "PASSWORD"}])
    assert "Winter" not in out and "#2026!" not in out
    assert out.endswith(". Change it today.")  # sentence full stop is kept


def test_password_inside_connection_string_masks_whole_string():
    text = "Use postgresql://admin:P@ssw0rd2026@localhost:5432/maindb now."
    out = sanitize(text, [{"text": "P@ssw0rd2026", "type": "PASSWORD"}])
    assert "admin" not in out and "localhost" not in out and "maindb" not in out


def test_short_name_does_not_mask_inside_longer_word_when_whole_word_exists():
    text = "Ram met Ramesh near the ramp."
    out = sanitize(text, [{"text": "Ram", "type": "PERSON"}])
    assert out.startswith("[PERSON_1] met Ramesh")


def test_pattern_backup_catches_what_the_model_missed():
    text = "PAN ABCDE1234F, email neha.k@example.in, card 4539 1488 0343 6467."
    out = sanitize(text, [], patterns=True)
    for value in ("ABCDE1234F", "neha.k@example.in", "4539 1488 0343 6467"):
        assert value not in out


def test_model_answer_with_extra_text_is_retried_then_fails_closed():
    bad = {"success": True, "response": "not json at all", "message": "ok"}
    with patch("trustboost.detector.pii_detector.generate_response", return_value=bad) as mock:
        result = run_sanitization_pipeline("Hi Bob", clear_global_mapping())
    assert result["success"] is False          # never pretends it's safe
    assert result["sanitized_text"] is None
    assert mock.call_count == 2                # one retry


def test_long_text_is_split_into_chunks():
    filler = "The weekly review covered staffing and purchase orders in detail. " * 60
    text = filler + "Contact Kavita Rao today."
    with fake_model([{"text": "Kavita Rao", "type": "PERSON"}]) as mock, \
         patch("trustboost.detector.pii_detector.USE_PATTERN_BACKUP", False):
        result = run_sanitization_pipeline(text, clear_global_mapping())
    assert mock.call_count > 1
    assert "Kavita Rao" not in result["sanitized_text"]


@pytest.mark.parametrize("number,expected", [("4539 1488 0343 6467", True), ("1234 5678 9012 3456", False)])
def test_luhn_check(number, expected):
    from trustboost.detector.pattern_detector import _passes_luhn
    assert _passes_luhn(number) is expected
