"""
tests/test_global_mapping.py
----------------------------
PURPOSE:
    Comprehensive unit tests for the one-global-mapping architecture
    (trustboost/session/global_mapping.py).

COVERS:
    1. First input creates global mappings from scratch.
    2. Second input appends new entities; counter never resets.
    3. Different original values → different tokens.
    4. Same (type, value) pair → same token reused, no duplicate entry.
    5. Restoration works across all previously sanitized inputs.
    6. Unknown placeholders are preserved verbatim and reported.
    7. Failed / invalid input never corrupts the existing global mapping.
    8. Clearing mappings removes all stored state.
    9. No session selection required for restoration.
    10. tokenize_with_global_mapping maintains offset-safe, overlap-safe behavior.
"""

import pytest
from trustboost.session import (
    merge_into_global_mapping,
    tokenize_with_global_mapping,
    restore_with_global_mapping,
    clear_global_mapping,
)


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def empty_state():
    """Returns a fresh global state (simulates app startup)."""
    return clear_global_mapping()


@pytest.fixture
def entities_input1():
    """Entities detected in the first sanitization input."""
    return [
        {"text": "Rahul Sharma", "type": "PERSON", "start": 11, "end": 23},
    ]


@pytest.fixture
def entities_input2():
    """Entities detected in the second sanitization input."""
    return [
        {"text": "Aman Verma", "type": "PERSON", "start": 12, "end": 22},
        {"text": "aman@gmail.com", "type": "EMAIL", "start": 26, "end": 40},
    ]


@pytest.fixture
def text_input1():
    return "My name is Rahul Sharma."


@pytest.fixture
def text_input2():
    return "Contact me: Aman Verma at aman@gmail.com."


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: First input creates global mappings
# ═════════════════════════════════════════════════════════════════════════════

class TestFirstInput:
    """Verifies that the first sanitization correctly initialises the global mapping."""

    def test_first_input_creates_mapping(self, empty_state, text_input1, entities_input1):
        """First input produces a [PERSON_1] token and stores it in the global mapping."""
        result = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)

        assert result["success"] is True
        updated = result["updated_global_state"]
        assert "[PERSON_1]" in updated["global_mapping"]
        assert updated["global_mapping"]["[PERSON_1]"] == "Rahul Sharma"

    def test_first_input_sanitized_text(self, empty_state, text_input1, entities_input1):
        """Sanitized text replaces the entity span with the global token."""
        result = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        assert result["sanitized_text"] == "My name is [PERSON_1]."

    def test_first_input_no_entities_returns_original(self, empty_state, text_input1):
        """Input with no entities returns the original text unchanged."""
        result = tokenize_with_global_mapping(text_input1, [], empty_state)
        assert result["success"] is True
        assert result["sanitized_text"] == text_input1
        # Global state should be unchanged — no new entities
        assert "updated_global_state" not in result or \
               result.get("updated_global_state", empty_state)["global_mapping"] == {}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Second input appends; counters never reset
# ═════════════════════════════════════════════════════════════════════════════

class TestSecondInputAppends:
    """Verifies that a second input extends the global mapping with new sequential tokens."""

    def test_second_input_appends_tokens(
        self, empty_state, text_input1, entities_input1, text_input2, entities_input2
    ):
        """After two inputs, the mapping contains tokens from both."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        state_after_1 = r1["updated_global_state"]

        r2 = tokenize_with_global_mapping(text_input2, entities_input2, state_after_1)
        state_after_2 = r2["updated_global_state"]

        gm = state_after_2["global_mapping"]
        assert "[PERSON_1]" in gm  # from input 1
        assert "[PERSON_2]" in gm  # from input 2 (counter continues at 2, not 1)
        assert "[EMAIL_1]" in gm   # from input 2

    def test_counters_do_not_reset_between_inputs(
        self, empty_state, text_input1, entities_input1, text_input2, entities_input2
    ):
        """The PERSON counter after input 1 is 1, so input 2 must yield [PERSON_2]."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        r2 = tokenize_with_global_mapping(
            text_input2, entities_input2, r1["updated_global_state"]
        )
        gm = r2["updated_global_state"]["global_mapping"]
        assert gm.get("[PERSON_2]") == "Aman Verma"

    def test_second_input_sanitized_text_uses_global_numbers(
        self, empty_state, text_input1, entities_input1, text_input2, entities_input2
    ):
        """Sanitized text from the second input uses globally incremented token numbers."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        r2 = tokenize_with_global_mapping(
            text_input2, entities_input2, r1["updated_global_state"]
        )
        # Aman Verma → [PERSON_2], aman@gmail.com → [EMAIL_1]
        assert r2["sanitized_text"] == "Contact me: [PERSON_2] at [EMAIL_1]."


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Different values → different tokens
# ═════════════════════════════════════════════════════════════════════════════

class TestDifferentValuesGetDifferentTokens:
    """Verifies that distinct original values always receive distinct tokens."""

    def test_two_different_people_get_different_tokens(self, empty_state):
        entities = [
            {"text": "Alice", "type": "PERSON", "start": 0, "end": 5},
            {"text": "Bob",   "type": "PERSON", "start": 8, "end": 11},
        ]
        result = tokenize_with_global_mapping("Alice & Bob", entities, empty_state)

        gm = result["updated_global_state"]["global_mapping"]
        tokens = list(gm.keys())
        values = list(gm.values())

        assert len(gm) == 2
        assert "Alice" in values
        assert "Bob" in values
        # Confirm distinct tokens
        assert tokens[0] != tokens[1]

    def test_different_entity_types_maintain_independent_counters(self, empty_state):
        entities = [
            {"text": "Alice",           "type": "PERSON", "start": 0, "end": 5},
            {"text": "alice@test.com",  "type": "EMAIL",  "start": 6, "end": 20},
        ]
        result = tokenize_with_global_mapping("Alice alice@test.com", entities, empty_state)
        gm = result["updated_global_state"]["global_mapping"]
        assert "[PERSON_1]" in gm
        assert "[EMAIL_1]" in gm


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Same (type, value) → token reused, no duplicate
# ═════════════════════════════════════════════════════════════════════════════

class TestTokenReuse:
    """Verifies that repeated identical (type, value) pairs reuse the same token."""

    def test_repeated_entity_reuses_token(
        self, empty_state, text_input1, entities_input1
    ):
        """Rahul Sharma in input 1 → [PERSON_1]. Rahul Sharma again in input 3 → [PERSON_1]."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        state_after_1 = r1["updated_global_state"]

        # Second occurrence of "Rahul Sharma" in a new input
        entities_repeat = [
            {"text": "Rahul Sharma", "type": "PERSON", "start": 7, "end": 19},
        ]
        r2 = tokenize_with_global_mapping(
            "Hello, Rahul Sharma!", entities_repeat, state_after_1
        )
        state_after_2 = r2["updated_global_state"]

        gm = state_after_2["global_mapping"]
        # Only one PERSON token should exist (no PERSON_2 for Rahul Sharma)
        assert "[PERSON_1]" in gm
        assert "[PERSON_2]" not in gm
        assert gm["[PERSON_1]"] == "Rahul Sharma"

    def test_repeated_entity_sanitized_text_uses_same_token(
        self, empty_state, text_input1, entities_input1
    ):
        """Sanitized text for the repeated value uses the already-assigned token."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        entities_repeat = [
            {"text": "Rahul Sharma", "type": "PERSON", "start": 7, "end": 19},
        ]
        r2 = tokenize_with_global_mapping(
            "Hello, Rahul Sharma!", entities_repeat, r1["updated_global_state"]
        )
        assert r2["sanitized_text"] == "Hello, [PERSON_1]!"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Restoration works across all previous inputs
# ═════════════════════════════════════════════════════════════════════════════

class TestRestoration:
    """Verifies that restore_with_global_mapping correctly restores tokens from any input."""

    def test_restoration_works_after_two_inputs(
        self, empty_state, text_input1, entities_input1, text_input2, entities_input2
    ):
        """Restoration using the global mapping correctly restores values from both inputs."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        r2 = tokenize_with_global_mapping(
            text_input2, entities_input2, r1["updated_global_state"]
        )
        gm = r2["updated_global_state"]["global_mapping"]

        ai_response = "Hello [PERSON_1] and [PERSON_2], your email [EMAIL_1] was updated."
        result = restore_with_global_mapping(gm, ai_response)

        assert result["success"] is True
        assert "Rahul Sharma" in result["restored_text"]
        assert "Aman Verma" in result["restored_text"]
        assert "aman@gmail.com" in result["restored_text"]

    def test_restoration_does_not_require_session_selection(self, empty_state, text_input1, entities_input1):
        """Restoration directly uses the global mapping dict — no session identifier needed."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        gm = r1["updated_global_state"]["global_mapping"]

        # Restore without any session-related parameter
        result = restore_with_global_mapping(gm, "Hello [PERSON_1]!")
        assert result["success"] is True
        assert result["restored_text"] == "Hello Rahul Sharma!"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Unknown placeholders preserved and reported
# ═════════════════════════════════════════════════════════════════════════════

class TestUnknownTokenHandling:
    """Verifies that tokens absent from the global mapping are preserved verbatim."""

    def test_unknown_token_preserved(self, empty_state, text_input1, entities_input1):
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        gm = r1["updated_global_state"]["global_mapping"]

        result = restore_with_global_mapping(gm, "Hello [PERSON_1] and [UNKNOWN_99].")
        assert result["success"] is True
        assert "[UNKNOWN_99]" in result["restored_text"]
        assert "[UNKNOWN_99]" in result["unrestored_tokens"]

    def test_all_unknown_tokens_reported(self, empty_state):
        gm = {}
        result = restore_with_global_mapping(gm, "[PERSON_1] and [EMAIL_1].")
        assert result["success"] is True
        assert "[PERSON_1]" in result["unrestored_tokens"]
        assert "[EMAIL_1]" in result["unrestored_tokens"]
        # Tokens preserved verbatim in restored text
        assert "[PERSON_1]" in result["restored_text"]
        assert "[EMAIL_1]" in result["restored_text"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Failed / invalid input never corrupts mapping
# ═════════════════════════════════════════════════════════════════════════════

class TestFailureResistance:
    """Verifies that invalid inputs never corrupt the existing global mapping."""

    def test_invalid_entity_list_does_not_corrupt_mapping(
        self, empty_state, text_input1, entities_input1
    ):
        """After a valid first input, passing a non-list entity arg leaves mapping intact."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        state_after_1 = r1["updated_global_state"]
        mapping_before = dict(state_after_1["global_mapping"])

        # Pass invalid entities type — should fail gracefully
        r_bad = tokenize_with_global_mapping(text_input1, None, state_after_1)
        assert r_bad["success"] is False
        # Global state from previous successful run is unchanged
        assert state_after_1["global_mapping"] == mapping_before

    def test_offset_mismatch_does_not_corrupt_mapping(
        self, empty_state, text_input1, entities_input1
    ):
        """An entity with wrong offsets causes tokenization failure, not mapping corruption."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        state_after_1 = r1["updated_global_state"]
        mapping_before = dict(state_after_1["global_mapping"])

        bad_entities = [
            {"text": "Aman Verma", "type": "PERSON", "start": 0, "end": 5},  # wrong offsets
        ]
        r_bad = tokenize_with_global_mapping(
            "Hello world", bad_entities, state_after_1
        )
        assert r_bad["success"] is False
        # Existing mapping is untouched
        assert state_after_1["global_mapping"] == mapping_before

    def test_empty_text_does_not_corrupt_mapping(
        self, empty_state, text_input1, entities_input1
    ):
        """Empty string input fails cleanly without touching the global mapping."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        state_after_1 = r1["updated_global_state"]

        r_bad = tokenize_with_global_mapping("", [], state_after_1)
        assert r_bad["success"] is False

    def test_merge_with_invalid_entity_skips_silently(self, empty_state):
        """merge_into_global_mapping skips entries missing 'text' or 'type' gracefully."""
        entities = [
            {"text": "Alice", "type": "PERSON"},  # valid
            {"type": "EMAIL"},                     # missing 'text' — should skip
            {"text": ""},                          # empty text — should skip
        ]
        state = merge_into_global_mapping(empty_state, entities)
        assert len(state["global_mapping"]) == 1  # only Alice added


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Clearing mappings
# ═════════════════════════════════════════════════════════════════════════════

class TestClearMapping:
    """Verifies that clearing the global mapping removes all stored state."""

    def test_clear_removes_all_tokens(
        self, empty_state, text_input1, entities_input1
    ):
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        state_after_1 = r1["updated_global_state"]

        cleared = clear_global_mapping()
        assert cleared["global_mapping"] == {}
        assert cleared["type_counters"] == {}
        assert cleared["val_to_token"] == {}

    def test_clear_returns_fresh_state(self, empty_state):
        fresh = clear_global_mapping()
        assert isinstance(fresh, dict)
        assert "global_mapping" in fresh
        assert "type_counters" in fresh
        assert "val_to_token" in fresh

    def test_new_input_after_clear_starts_at_1(
        self, empty_state, text_input1, entities_input1
    ):
        """After clearing, token numbering restarts from 1."""
        r1 = tokenize_with_global_mapping(text_input1, entities_input1, empty_state)
        cleared = clear_global_mapping()

        r2 = tokenize_with_global_mapping(text_input1, entities_input1, cleared)
        gm = r2["updated_global_state"]["global_mapping"]
        assert "[PERSON_1]" in gm
        assert gm["[PERSON_1]"] == "Rahul Sharma"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: merge_into_global_mapping direct tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMergeIntoGlobalMapping:
    """Unit tests for the merge_into_global_mapping helper."""

    def test_merge_builds_mapping_from_entities(self, empty_state):
        entities = [{"text": "Alice", "type": "PERSON"}]
        state = merge_into_global_mapping(empty_state, entities)
        assert "[PERSON_1]" in state["global_mapping"]
        assert state["global_mapping"]["[PERSON_1]"] == "Alice"

    def test_merge_empty_list_returns_state_unchanged(self, empty_state):
        state = merge_into_global_mapping(empty_state, [])
        assert state["global_mapping"] == {}

    def test_merge_does_not_overwrite_existing_token(self, empty_state):
        entities1 = [{"text": "Alice", "type": "PERSON"}]
        state1 = merge_into_global_mapping(empty_state, entities1)

        # Try to merge a different value for the same type
        entities2 = [{"text": "Bob", "type": "PERSON"}]
        state2 = merge_into_global_mapping(state1, entities2)

        # Alice → [PERSON_1] must not be overwritten
        assert state2["global_mapping"]["[PERSON_1]"] == "Alice"
        # Bob → [PERSON_2]
        assert state2["global_mapping"]["[PERSON_2]"] == "Bob"

    def test_merge_reuses_existing_entry(self, empty_state):
        entities = [{"text": "Alice", "type": "PERSON"}]
        state1 = merge_into_global_mapping(empty_state, entities)
        state2 = merge_into_global_mapping(state1, entities)  # same entity again

        # Only one entry for Alice — no duplicates
        assert len(state2["global_mapping"]) == 1
        assert state2["type_counters"].get("PERSON") == 1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: restore_with_global_mapping validation
# ═════════════════════════════════════════════════════════════════════════════

class TestRestoreWithGlobalMapping:
    """Unit tests for restore_with_global_mapping input validation."""

    def test_restore_invalid_mapping_type_fails(self):
        result = restore_with_global_mapping(None, "Hello [PERSON_1]")
        assert result["success"] is False

    def test_restore_invalid_text_type_fails(self):
        result = restore_with_global_mapping({}, 12345)
        assert result["success"] is False

    def test_restore_empty_text_succeeds(self):
        result = restore_with_global_mapping({"[PERSON_1]": "Alice"}, "")
        assert result["success"] is True
        assert result["restored_text"] == ""

    def test_restore_full_round_trip(self, empty_state):
        """Full end-to-end: tokenize then restore produces the original text."""
        text = "My email is alice@test.com and name is Alice."
        entities = [
            {"text": "alice@test.com", "type": "EMAIL",  "start": 12, "end": 26},
            {"text": "Alice",          "type": "PERSON", "start": 39, "end": 44},
        ]
        r = tokenize_with_global_mapping(text, entities, empty_state)
        assert r["success"]

        sanitized = r["sanitized_text"]
        gm = r["updated_global_state"]["global_mapping"]

        restored = restore_with_global_mapping(gm, sanitized)
        assert restored["success"] is True
        assert restored["restored_text"] == text
