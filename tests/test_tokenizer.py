"""
tests/test_tokenizer.py
-----------------------
PURPOSE:
    Comprehensive unit tests for the Phase 4 tokenization and restoration
    module (trustboost/tokenizer/tokenizer.py).

COVERS:
    1. Package exports
    2. Input validation (text, entities, types, bounds)
    3. Token generation, formatting, and independent per-type numbering
    4. Repeated entity token reuse and cross-type differentiation
    5. Span replacement safety (beginning, middle, end, Unicode, Hinglish, emojis)
    6. Duplicate and overlapping span handling
    7. Preservation of non-sensitive text and immutability of input
    8. Restoration logic (single, multiple, repeated, partial match safety)
    9. Unmapped / unknown token handling
    10. Complete end-to-end round-trip fidelity tests
"""

import copy
import pytest
from trustboost.tokenizer import tokenize_entities, restore_text
from trustboost.tokenizer.tokenizer import _is_valid_token_format


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Package Exports and Helper Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTokenizerExportsAndHelpers:
    """Verifies public exports and token format validator."""

    def test_package_exports(self):
        """tokenize_entities and restore_text must be callable exports."""
        assert callable(tokenize_entities)
        assert callable(restore_text)

    def test_token_format_valid_cases(self):
        """Valid tokens must match [CATEGORY_INDEX] with positive integers."""
        assert _is_valid_token_format("[PERSON_1]") is True
        assert _is_valid_token_format("[EMAIL_2]") is True
        assert _is_valid_token_format("[CONFIDENTIAL_PROJECT_10]") is True
        assert _is_valid_token_format("[IP_ADDRESS_99]") is True
        assert _is_valid_token_format("[A_1]") is True

    def test_token_format_invalid_cases(self):
        """Malformed or non-token brackets must return False."""
        assert _is_valid_token_format("") is False
        assert _is_valid_token_format("PERSON_1") is False
        assert _is_valid_token_format("[PERSON]") is False
        assert _is_valid_token_format("[PERSON_0]") is False
        assert _is_valid_token_format("[PERSON_-1]") is False
        assert _is_valid_token_format("[person_1]") is False
        assert _is_valid_token_format("[PERSON_abc]") is False
        assert _is_valid_token_format("[123]") is False
        assert _is_valid_token_format("[see note 1]") is False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Tokenization Input Validation Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTokenizationInputValidation:
    """Validates edge cases in input types, empty values, and malformed entities."""

    def test_non_string_text(self):
        """Non-string text must return failure."""
        res = tokenize_entities(12345, [])  # type: ignore
        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert "string" in res["message"].lower()

    def test_empty_text(self):
        """Empty text must return failure."""
        res = tokenize_entities("", [])
        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert "empty" in res["message"].lower()

    def test_whitespace_only_text(self):
        """Whitespace-only text must return failure."""
        res = tokenize_entities("   \n\t  ", [])
        assert res["success"] is False
        assert res["sanitized_text"] is None
        assert "whitespace" in res["message"].lower()

    def test_non_list_entities(self):
        """Entities container must be a list."""
        res = tokenize_entities("Hello world", "not-a-list")  # type: ignore
        assert res["success"] is False
        assert "list" in res["message"].lower()

    def test_empty_entities_list(self):
        """Empty entities list returns original text unchanged."""
        text = "Nothing sensitive here."
        res = tokenize_entities(text, [])
        assert res["success"] is True
        assert res["sanitized_text"] == text
        assert res["mapping"] == {}
        assert res["tokens"] == []

    def test_non_dict_entity_item(self):
        """Entity item that is not a dictionary must fail."""
        res = tokenize_entities("Hello", ["not a dict"])  # type: ignore
        assert res["success"] is False
        assert "dictionary" in res["message"].lower()

    def test_missing_required_entity_keys(self):
        """Entity missing required fields must return failure."""
        entities = [{"text": "John", "type": "PERSON", "start": 0}]  # missing 'end'
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False
        assert "missing required fields" in res["message"].lower()

    def test_empty_entity_text(self):
        """Entity with empty text must fail."""
        entities = [{"text": "", "type": "PERSON", "start": 0, "end": 4}]
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False
        assert "non-empty string" in res["message"].lower()

    def test_empty_entity_type(self):
        """Entity with empty type must fail."""
        entities = [{"text": "John", "type": "   ", "start": 0, "end": 4}]
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False
        assert "non-empty string" in res["message"].lower()

    def test_non_integer_offsets(self):
        """Non-integer offsets (including bools or strings) must fail."""
        entities = [{"text": "John", "type": "PERSON", "start": "0", "end": 4}]
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False
        assert "integers" in res["message"].lower()

        # Booleans inherit from int in Python; ensure bool is rejected
        entities = [{"text": "John", "type": "PERSON", "start": False, "end": 4}]
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False

    def test_negative_offsets(self):
        """Negative start or end offsets must fail."""
        entities = [{"text": "John", "type": "PERSON", "start": -1, "end": 4}]
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False
        assert "out of bounds" in res["message"].lower()

    def test_start_greater_than_or_equal_to_end(self):
        """start >= end must fail."""
        entities = [{"text": "John", "type": "PERSON", "start": 4, "end": 4}]
        res = tokenize_entities("John Doe", entities)
        assert res["success"] is False
        assert "start >= end" in res["message"].lower()

    def test_out_of_bounds_offsets(self):
        """End offset exceeding text length must fail."""
        text = "John"
        entities = [{"text": "John", "type": "PERSON", "start": 0, "end": 99}]
        res = tokenize_entities(text, entities)
        assert res["success"] is False
        assert "out of bounds" in res["message"].lower()

    def test_entity_text_mismatch(self):
        """If text[start:end] != entity['text'], validation must fail."""
        text = "Hello John Doe"
        # "John" is at index 6..10, not 0..4
        entities = [{"text": "John", "type": "PERSON", "start": 0, "end": 4}]
        res = tokenize_entities(text, entities)
        assert res["success"] is False
        assert "does not match" in res["message"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Token Generation & Numbering Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTokenGenerationAndNumbering:
    """Verifies token formatting, independent counters, and reuse rules."""

    def test_single_person_entity(self):
        """Single PERSON entity generates [PERSON_1]."""
        text = "Call Harshit at home."
        entities = [{"text": "Harshit", "type": "PERSON", "start": 5, "end": 12}]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Call [PERSON_1] at home."
        assert res["mapping"] == {"[PERSON_1]": "Harshit"}
        assert res["tokens"] == [{"token": "[PERSON_1]", "type": "PERSON", "original": "Harshit"}]

    def test_single_email_entity(self):
        """Single EMAIL entity generates [EMAIL_1]."""
        text = "Send mail to user@example.com."
        entities = [{"text": "user@example.com", "type": "EMAIL", "start": 13, "end": 29}]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Send mail to [EMAIL_1]."
        assert res["mapping"] == {"[EMAIL_1]": "user@example.com"}

    def test_multiple_different_entity_types(self):
        """Multiple distinct types start their numbering independently at 1."""
        text = "Alice (alice@work.com) logged into 192.168.1.1."
        entities = [
            {"text": "Alice", "type": "PERSON", "start": 0, "end": 5},
            {"text": "alice@work.com", "type": "EMAIL", "start": 7, "end": 21},
            {"text": "192.168.1.1", "type": "IP_ADDRESS", "start": 35, "end": 46},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[PERSON_1] ([EMAIL_1]) logged into [IP_ADDRESS_1]."
        assert res["mapping"] == {
            "[PERSON_1]": "Alice",
            "[EMAIL_1]": "alice@work.com",
            "[IP_ADDRESS_1]": "192.168.1.1",
        }

    def test_multiple_entities_of_same_type_numbered_sequentially(self):
        """Different values of the same type increment sequentially: [TYPE_1], [TYPE_2]."""
        text = "Alice spoke with Bob and Charlie."
        entities = [
            {"text": "Alice", "type": "PERSON", "start": 0, "end": 5},
            {"text": "Bob", "type": "PERSON", "start": 17, "end": 20},
            {"text": "Charlie", "type": "PERSON", "start": 25, "end": 32},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[PERSON_1] spoke with [PERSON_2] and [PERSON_3]."
        assert res["mapping"] == {
            "[PERSON_1]": "Alice",
            "[PERSON_2]": "Bob",
            "[PERSON_3]": "Charlie",
        }

    def test_repeated_identical_entity_reuses_same_token(self):
        """Repeated identical values under the same type reuse the same token."""
        text = "Harshit sent an email. Later, Harshit confirmed."
        entities = [
            {"text": "Harshit", "type": "PERSON", "start": 0, "end": 7},
            {"text": "Harshit", "type": "PERSON", "start": 30, "end": 37},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[PERSON_1] sent an email. Later, [PERSON_1] confirmed."
        assert res["mapping"] == {"[PERSON_1]": "Harshit"}
        assert len(res["tokens"]) == 1

    def test_same_text_with_different_types_receives_different_tokens(self):
        """The same string value under different types must receive different tokens."""
        text = "Visit Harshit at Harshit headquarters."
        entities = [
            {"text": "Harshit", "type": "PERSON", "start": 6, "end": 13},
            {"text": "Harshit", "type": "ORGANIZATION", "start": 17, "end": 24},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Visit [PERSON_1] at [ORGANIZATION_1] headquarters."
        assert res["mapping"] == {
            "[PERSON_1]": "Harshit",
            "[ORGANIZATION_1]": "Harshit",
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Span Positioning, Unicode, Emojis, and Safety
# ═════════════════════════════════════════════════════════════════════════════

class TestSpanPositioningAndEncoding:
    """Verifies replacement behavior across boundaries, languages, and emojis."""

    def test_entity_at_the_beginning_of_text(self):
        """Entity at start index 0."""
        text = "Harshit is a student."
        entities = [{"text": "Harshit", "type": "PERSON", "start": 0, "end": 7}]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[PERSON_1] is a student."

    def test_entity_at_the_end_of_text(self):
        """Entity at the final character index."""
        text = "Student name is Harshit"
        entities = [{"text": "Harshit", "type": "PERSON", "start": 16, "end": 23}]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Student name is [PERSON_1]"

    def test_entity_in_the_middle_of_text(self):
        """Entity surrounded by prefix and suffix text."""
        text = "User Harshit logged out."
        entities = [{"text": "Harshit", "type": "PERSON", "start": 5, "end": 12}]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "User [PERSON_1] logged out."

    def test_adjacent_entities(self):
        """Entities immediately touching with no gap between them."""
        text = "AliceBob"
        entities = [
            {"text": "Alice", "type": "PERSON", "start": 0, "end": 5},
            {"text": "Bob", "type": "PERSON", "start": 5, "end": 8},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[PERSON_1][PERSON_2]"

    def test_original_input_remains_strictly_unchanged(self):
        """Tokenization must not mutate the input text object."""
        original = "Harshit has an email."
        copy_text = copy.deepcopy(original)
        entities = [{"text": "Harshit", "type": "PERSON", "start": 0, "end": 7}]
        tokenize_entities(original, entities)
        assert original == copy_text

    def test_preservation_of_all_unrelated_text(self):
        """Punctuation, spacing, newlines, and symbols outside entities are preserved."""
        text = "  === Header ===\n- Item 1: Harshit\n- Item 2: Test\n  === End ==="
        entities = [{"text": "Harshit", "type": "PERSON", "start": 27, "end": 34}]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "  === Header ===\n- Item 1: [PERSON_1]\n- Item 2: Test\n  === End ==="

    def test_unicode_and_hindi_english_text(self):
        """Handles multi-byte Unicode and Hindi-English sentences seamlessly."""
        text = "Mera naam Harshit Verma hai aur email harshit@test.in hai."
        entities = [
            {"text": "Harshit Verma", "type": "PERSON", "start": 10, "end": 23},
            {"text": "harshit@test.in", "type": "EMAIL", "start": 38, "end": 53},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Mera naam [PERSON_1] hai aur email [EMAIL_1] hai."

    def test_emojis_in_text(self):
        """Emojis are preserved and do not skew offset replacement."""
        text = "Hello 👋 John Doe 🚀 secret key: sk-secret-1234 🔒"
        entities = [
            {"text": "John Doe", "type": "PERSON", "start": 8, "end": 16},
            {"text": "sk-secret-1234", "type": "SECRET", "start": 31, "end": 45},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Hello 👋 [PERSON_1] 🚀 secret key: [SECRET_1] 🔒"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Duplicate and Overlapping Spans Handling
# ═════════════════════════════════════════════════════════════════════════════

class TestDuplicateAndOverlappingSpans:
    """Verifies that overlapping or duplicate entity spans do not corrupt text."""

    def test_exact_duplicate_entities_deduplicated(self):
        """Identical span passed twice is processed exactly once."""
        text = "Hello Harshit"
        entities = [
            {"text": "Harshit", "type": "PERSON", "start": 6, "end": 13},
            {"text": "Harshit", "type": "PERSON", "start": 6, "end": 13},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "Hello [PERSON_1]"
        assert len(res["tokens"]) == 1

    def test_overlapping_spans_longer_preferred_no_corruption(self):
        """Overlapping span candidate is rejected; longer span wins."""
        text = "Rahul Sharma is here."
        # "Rahul Sharma" (0..12) overlaps with "Rahul" (0..5)
        entities = [
            {"text": "Rahul", "type": "PERSON", "start": 0, "end": 5},
            {"text": "Rahul Sharma", "type": "PERSON", "start": 0, "end": 12},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[PERSON_1] is here."
        assert res["mapping"] == {"[PERSON_1]": "Rahul Sharma"}

    def test_partially_overlapping_spans_second_rejected(self):
        """Partially overlapping span is safely rejected without text damage."""
        text = "ABCDEFG"
        entities = [
            {"text": "ABCD", "type": "SECRET", "start": 0, "end": 4},
            {"text": "CDEF", "type": "SECRET", "start": 2, "end": 6},
        ]
        res = tokenize_entities(text, entities)
        assert res["success"] is True
        assert res["sanitized_text"] == "[SECRET_1]EFG"
        assert res["mapping"] == {"[SECRET_1]": "ABCD"}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Text Restoration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestRestoration:
    """Tests the reverse transformation from sanitized text back to original."""

    def test_single_token_restoration(self):
        """Restores a single token accurately."""
        sanitized = "Hello [PERSON_1]!"
        mapping = {"[PERSON_1]": "Harshit Sahu"}
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "Hello Harshit Sahu!"
        assert res["unrestored_tokens"] == []

    def test_multiple_tokens_restoration(self):
        """Restores multiple distinct tokens."""
        sanitized = "Contact [PERSON_1] at [EMAIL_1] regarding [CONFIDENTIAL_PROJECT_1]."
        mapping = {
            "[PERSON_1]": "Harshit",
            "[EMAIL_1]": "harshit@example.com",
            "[CONFIDENTIAL_PROJECT_1]": "Project Falcon",
        }
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "Contact Harshit at harshit@example.com regarding Project Falcon."
        assert res["unrestored_tokens"] == []

    def test_repeated_token_occurrence_restoration(self):
        """Restores repeated instances of the same token throughout the text."""
        sanitized = "[PERSON_1] called [PERSON_2]. [PERSON_1] then emailed [PERSON_2]."
        mapping = {
            "[PERSON_1]": "Alice",
            "[PERSON_2]": "Bob",
        }
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "Alice called Bob. Alice then emailed Bob."

    def test_unknown_or_unmapped_token_preserved_and_reported(self):
        """Tokens present in text but absent from mapping remain intact and reported."""
        sanitized = "Hello [PERSON_1], check [UNKNOWN_99] and [SECRET_1]."
        mapping = {"[PERSON_1]": "Harshit"}
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "Hello Harshit, check [UNKNOWN_99] and [SECRET_1]."
        assert "[UNKNOWN_99]" in res["unrestored_tokens"]
        assert "[SECRET_1]" in res["unrestored_tokens"]

    def test_empty_mapping_leaves_tokens_unchanged(self):
        """Empty mapping returns sanitized text untouched."""
        sanitized = "Hello [PERSON_1]."
        res = restore_text(sanitized, {})
        assert res["success"] is True
        assert res["restored_text"] == sanitized
        assert res["unrestored_tokens"] == ["[PERSON_1]"]

    def test_empty_sanitized_text(self):
        """Empty sanitized text returns empty restored text."""
        res = restore_text("", {"[PERSON_1]": "Harshit"})
        assert res["success"] is True
        assert res["restored_text"] == ""
        assert res["unrestored_tokens"] == []

    def test_invalid_sanitized_text_type(self):
        """Non-string sanitized text returns failure."""
        res = restore_text(None, {})  # type: ignore
        assert res["success"] is False
        assert res["restored_text"] is None

    def test_invalid_mapping_type(self):
        """Non-dict mapping returns failure."""
        res = restore_text("Text", "not a dict")  # type: ignore
        assert res["success"] is False
        assert res["restored_text"] is None

    def test_non_string_keys_or_values_in_mapping(self):
        """Mapping containing non-string keys or values returns failure."""
        res1 = restore_text("Text", {1: "value"})  # type: ignore
        assert res1["success"] is False

        res2 = restore_text("Text", {"[PERSON_1]": 123})  # type: ignore
        assert res2["success"] is False

    def test_partial_token_match_safety(self):
        """
        [PERSON_1] must NEVER partially overwrite [PERSON_10].
        """
        sanitized = "User 10 is [PERSON_10] and User 1 is [PERSON_1]."
        mapping = {
            "[PERSON_1]": "Alice",
            "[PERSON_10]": "Bob",
        }
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "User 10 is Bob and User 1 is Alice."

    def test_token_like_brackets_in_ordinary_text_preserved(self):
        """Bracketed non-tokens like [123] or [notes] are not treated as tokens."""
        sanitized = "See reference [1] and [chapter 2 notes] for [PERSON_1]."
        mapping = {"[PERSON_1]": "Harshit"}
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "See reference [1] and [chapter 2 notes] for Harshit."
        assert res["unrestored_tokens"] == []

    def test_second_order_replacement_prevention(self):
        """
        If restored value happens to contain a token string, it is NOT re-replaced.
        """
        sanitized = "Email: [EMAIL_1] | User: [PERSON_1]"
        # [EMAIL_1]'s value contains "[PERSON_1]"
        mapping = {
            "[EMAIL_1]": "alert-[PERSON_1]@corp.com",
            "[PERSON_1]": "Charlie",
        }
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == "Email: alert-[PERSON_1]@corp.com | User: Charlie"

    def test_special_characters_in_restored_values(self):
        """Values with regex/format special characters ($100, backslashes, quotes) restore cleanly."""
        sanitized = "Price is [FINANCIAL_1] and path is [PATH_1]."
        mapping = {
            "[FINANCIAL_1]": "$1,000.00 (USD)",
            "[PATH_1]": r"C:\Users\Admin\secrets.txt",
        }
        res = restore_text(sanitized, mapping)
        assert res["success"] is True
        assert res["restored_text"] == r"Price is $1,000.00 (USD) and path is C:\Users\Admin\secrets.txt."


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: End-to-End Round-Trip Fidelity Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestRoundTripFidelity:
    """Verifies that tokenize -> restore reconstructs the exact original text."""

    def test_full_round_trip_basic(self):
        """Basic round-trip test from prompt example."""
        original_text = "My name is Harshit Sahu and my email is harshit@example.com."
        entities = [
            {"text": "Harshit Sahu", "type": "PERSON", "start": 11, "end": 23},
            {"text": "harshit@example.com", "type": "EMAIL", "start": 40, "end": 59},
        ]
        tok_res = tokenize_entities(original_text, entities)
        assert tok_res["success"] is True

        rest_res = restore_text(tok_res["sanitized_text"], tok_res["mapping"])
        assert rest_res["success"] is True
        assert rest_res["restored_text"] == original_text

    def test_full_round_trip_complex(self):
        """Complex round-trip with repeated entities, emojis, punctuation, and multiple types."""
        original_text = (
            "🚀 Urgent: Harshit Sahu (harshit@example.com) contacted Priya Patel.\n"
            "Harshit confirmed secret project Operation Titan.\n"
            "Server IP 192.168.1.10 was accessed by Priya Patel.\n"
            "Call 9876543210 immediately! 🔐"
        )
        entities = [
            {"text": "Harshit Sahu", "type": "PERSON", "start": 10, "end": 22},
            {"text": "harshit@example.com", "type": "EMAIL", "start": 24, "end": 43},
            {"text": "Priya Patel", "type": "PERSON", "start": 55, "end": 66},
            {"text": "Harshit", "type": "PERSON", "start": 68, "end": 75},
            {"text": "Operation Titan", "type": "CONFIDENTIAL_PROJECT", "start": 101, "end": 116},
            {"text": "192.168.1.10", "type": "IP_ADDRESS", "start": 128, "end": 140},
            {"text": "Priya Patel", "type": "PERSON", "start": 157, "end": 168},
            {"text": "9876543210", "type": "PHONE", "start": 175, "end": 185},
        ]
        tok_res = tokenize_entities(original_text, entities)
        assert tok_res["success"] is True

        rest_res = restore_text(tok_res["sanitized_text"], tok_res["mapping"])
        assert rest_res["success"] is True
        assert rest_res["restored_text"] == original_text

    def test_round_trip_hindi_english(self):
        """Round-trip fidelity on mixed Hindi-English script."""
        original_text = "Mera naam Rahul Verma hai aur mera phone number 9876543210 hai."
        entities = [
            {"text": "Rahul Verma", "type": "PERSON", "start": 10, "end": 21},
            {"text": "9876543210", "type": "PHONE", "start": 48, "end": 58},
        ]
        tok_res = tokenize_entities(original_text, entities)
        assert tok_res["success"] is True

        rest_res = restore_text(tok_res["sanitized_text"], tok_res["mapping"])
        assert rest_res["success"] is True
        assert rest_res["restored_text"] == original_text
