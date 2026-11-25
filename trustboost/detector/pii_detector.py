"""
trustboost/detector/pii_detector.py
--------------------------------------
PURPOSE:
    The main PII detection module for TrustBoost.

    This file contains the full detection pipeline:
      1. Build a detection prompt using prompt_builder.py
      2. Send the prompt to Gemma 3 4B via the existing Ollama client
      3. Parse the JSON response from the model
      4. Validate every entity field (types, offsets, text match)
      5. Attempt safe offset recovery when offsets are wrong
      6. Remove duplicates and overlapping detections
      7. Return a clean, structured result

PUBLIC API:
    detect_pii(text: str) -> dict
        The only function you need to call from outside this module.
        Takes raw text, returns a structured dict with entities.

WHAT THIS FILE DOES NOT DO:
    - Does NOT replace any text (that is Phase 4 — tokenization).
    - Does NOT use regex for detection.
    - Does NOT call Ollama directly (uses ollama_client.py).
    - Does NOT modify the user's original text.
"""

import json
import os

from trustboost.agent.ollama_client import generate_response
from trustboost.detector.pattern_detector import find_pattern_entities
from trustboost.detector.prompt_builder import (
    FALLBACK_ENTITY_TYPE,
    VALID_ENTITY_TYPES,
    build_detection_messages,
)

# Turn the fixed-format backup detector on/off (TRUSTBOOST_PATTERNS=0 to disable).
USE_PATTERN_BACKUP = os.getenv("TRUSTBOOST_PATTERNS", "1") != "0"

CHUNK_CHARS = 1500

SECRET_TYPES = {
    "PASSWORD", "API_KEY", "ACCESS_TOKEN", "SECRET", "PRIVATE_KEY",
    "CREDENTIAL", "DATABASE_CREDENTIAL",
}

# Characters that end a secret when stretching it.
_SECRET_STOP_CHARS = "\"'`()<>[]{},;"

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["text", "type"],
            },
        }
    },
    "required": ["entities"],
}

TYPE_ALIASES = {
    "NAME": "PERSON", "FULL_NAME": "PERSON", "PERSON_NAME": "PERSON",
    "EMAIL_ADDRESS": "EMAIL", "E_MAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE", "MOBILE": "PHONE", "MOBILE_NUMBER": "PHONE", "TELEPHONE": "PHONE",
    "AADHAR": "AADHAAR", "AADHAAR_NUMBER": "AADHAAR", "UIDAI": "AADHAAR",
    "PAN_CARD": "PAN", "PAN_NUMBER": "PAN",
    "CARD_NUMBER": "CREDIT_CARD", "DEBIT_CARD": "CREDIT_CARD", "CREDITCARD": "CREDIT_CARD",
    "ACCOUNT_NUMBER": "BANK_ACCOUNT", "BANK_ACCOUNT_NUMBER": "BANK_ACCOUNT",
    "IFSC_CODE": "IFSC",
    "IP": "IP_ADDRESS", "IPV4": "IP_ADDRESS", "IPV6": "IP_ADDRESS",
    "DOB": "DATE_OF_BIRTH", "BIRTH_DATE": "DATE_OF_BIRTH", "BIRTHDATE": "DATE_OF_BIRTH",
    "TOKEN": "ACCESS_TOKEN", "AUTH_TOKEN": "ACCESS_TOKEN", "BEARER_TOKEN": "ACCESS_TOKEN",
    "APIKEY": "API_KEY", "KEY": "API_KEY",
    "URL": "INTERNAL_URL", "LINK": "INTERNAL_URL",
    "CITY": "LOCATION", "PLACE": "LOCATION",
    "PROJECT": "CONFIDENTIAL_PROJECT",
    "CONNECTION_STRING": "DATABASE_CREDENTIAL", "DB_CREDENTIAL": "DATABASE_CREDENTIAL",
    "HOSTNAME": "SERVER_IDENTIFIER", "SERVER": "SERVER_IDENTIFIER",
}


def _normalise_type(raw_type) -> str:
    """
    Turns whatever label the model used into one of our categories.
      "aadhaar" -> "AADHAAR", "NAME" -> "PERSON", "phone number" -> "PHONE"
    If nothing matches, returns a safe catch-all type so the text is STILL masked.
    """
    t = str(raw_type or "").strip().upper().replace(" ", "_").replace("-", "_")
    t = TYPE_ALIASES.get(t, t)
    if t in VALID_ENTITY_TYPES:
        return t
    if any(word in t for word in ("PASS", "KEY", "TOKEN", "SECRET", "CRED")):
        return "CREDENTIAL"
    return FALLBACK_ENTITY_TYPE


def _find_all(haystack: str, needle: str) -> list[int]:
    """Start index of every occurrence of `needle` in `haystack` (plain str.find)."""
    positions = []
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos == -1:
            return positions
        positions.append(pos)
        start = pos + 1


def _find_ignoring_whitespace(text: str, value: str) -> list[tuple[int, int]]:
    """
    Finds `value` even if the model changed the spacing or capitals,
    e.g. model says "4111 1111" but the text has "4111  1111".
    """
    target = "".join(value.split()).lower()
    if not target:
        return []
    squashed_chars, index_map = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            low = ch.lower()
            squashed_chars.append(low if len(low) == 1 else ch)
            index_map.append(i)
    squashed = "".join(squashed_chars)
    return [
        (index_map[pos], index_map[pos + len(target) - 1] + 1)
        for pos in _find_all(squashed, target)
    ]


def _find_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """
    Every (start, end) position where `value` appears in `text`.
    Tries an exact match first, then ignoring capitals, then ignoring spacing.
    """
    spans = [(p, p + len(value)) for p in _find_all(text, value)]
    if not spans:
        lower_text = text.lower()
        if len(lower_text) == len(text):
            spans = [(p, p + len(value)) for p in _find_all(lower_text, value.lower())]
    if not spans:
        spans = _find_ignoring_whitespace(text, value)
    return spans


def _keep_whole_words(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Prefer matches that are whole words, so "Ram" does not mask the inside of
    "Ramesh". If there are no whole-word matches, keep all matches (safer).
    """
    whole = [
        (s, e) for s, e in spans
        if (s == 0 or not text[s - 1].isalnum()) and (e == len(text) or not text[e].isalnum())
    ]
    return whole if whole else spans


def _widen_secret_span(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Stretches a secret to the surrounding non-space characters, so
    "Passw0rd" becomes "Passw0rd!" and a password inside a connection string
    becomes the whole connection string. A final full stop is not included.
    """
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in _SECRET_STOP_CHARS:
        start -= 1
    while end < len(text) and not text[end].isspace() and text[end] not in _SECRET_STOP_CHARS:
        end += 1
    while end > start + 1 and text[end - 1] in ".:":
        end -= 1
    return start, end


def _locate_entity(raw_entity, original_text: str) -> list[dict]:
    """
    Turns ONE detection from the model ({"text", "type"}) into a list of exact
    entities, one for EVERY place that value appears in the original text.
    Offsets are always computed here in Python, never trusted from the model.
    """
    if not isinstance(raw_entity, dict):
        return []
    value = raw_entity.get("text")
    if not isinstance(value, str):
        return []
    value = value.strip()
    if len(value) < 2:  # a single character is never meaningful PII
        return []

    entity_type = _normalise_type(raw_entity.get("type"))
    reason = str(raw_entity.get("reason", "")).strip()

    located = []
    for start, end in _keep_whole_words(original_text, _find_occurrences(original_text, value)):
        if entity_type in SECRET_TYPES:
            start, end = _widen_secret_span(original_text, start, end)
        located.append({
            "text": original_text[start:end],
            "type": entity_type,
            "start": start,
            "end": end,
            "reason": reason,
        })
    return located


def _check_structure(parsed) -> str | None:
    """Returns an error message if the parsed JSON has the wrong shape, else None."""
    if parsed is None:
        return ("the model returned a response that could not be parsed as JSON. "
                "This may be a temporary model issue. Please try again.")
    if not isinstance(parsed, dict):
        return "expected a JSON object at the top level."
    if "entities" not in parsed:
        return "JSON response is missing the 'entities' key."
    if not isinstance(parsed["entities"], list):
        return "'entities' must be a JSON array."
    return None


def _ask_model(chunk: str, attempts: int = 2) -> dict:
    """
    Asks Gemma to find PII in one piece of text. Tries again once if the answer
    is broken. Returns {"success", "entities" (raw model output), "message"}.
    """
    system_prompt, user_message = build_detection_messages(chunk)
    last_error = "unknown error"
    for _ in range(attempts):
        result = generate_response(user_message, system=system_prompt, json_schema=DETECTION_SCHEMA)
        if not result.get("success"):
            last_error = result.get("message", "model call failed")
            continue
        raw = result.get("response", "")
        if not raw or not raw.strip():
            last_error = "the model returned an empty response."
            continue
        parsed = _extract_json_from_response(raw)
        error = _check_structure(parsed)
        if error is None:
            return {"success": True, "entities": parsed["entities"], "message": "ok"}
        last_error = error
    return {"success": False, "entities": [], "message": last_error}


def _split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """
    Splits long text into pieces of at most `max_chars`, cutting at a line break
    or the end of a sentence where possible. Short text is returned as one piece.
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut <= start:
                cut = text.rfind(". ", start, end)
            if cut > start:
                end = cut + 1
        chunks.append(text[start:end])
        start = end
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION 1: _extract_json_from_response
# ─────────────────────────────────────────────────────────────────────────────
def _extract_json_from_response(raw_response: str) -> dict | None:
    """
    Attempts to extract and parse a JSON object from the model's raw response.

    WHY THIS FUNCTION EXISTS:
        LLMs sometimes wrap their JSON in markdown code fences like:
            ```json
            { ... }
            ```
        or add a brief explanation before/after the JSON. This function
        strips that noise and finds the actual JSON object.

    STRATEGY (no regex used):
        1. First try to parse the entire response as JSON directly.
        2. If that fails, find the first '{' and last '}' and parse that slice.
        3. If that also fails, return None.

    PARAMETERS:
        raw_response (str): The raw text returned by the LLM.

    RETURNS:
        dict if parsing succeeds, None otherwise.
    """
    if not raw_response or not raw_response.strip():
        return None

    # Attempt 1: parse the full response as-is
    try:
        return json.loads(raw_response.strip())
    except json.JSONDecodeError:
        pass

    # Attempt 2: find the outermost { ... } block and parse that
    # This handles cases where the model adds text before or after the JSON
    start_idx = raw_response.find("{")
    end_idx = raw_response.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_slice = raw_response[start_idx : end_idx + 1]
        try:
            return json.loads(json_slice)
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION 2: _attempt_offset_recovery
# ─────────────────────────────────────────────────────────────────────────────
def _attempt_offset_recovery(entity: dict, original_text: str) -> dict | None:
    """
    Tries to find the correct start/end offsets for an entity whose offsets
    are wrong or missing, using plain Python string search (no regex).

    WHY THIS FUNCTION EXISTS:
        LLMs are not perfect at counting characters. The model may identify
        the correct entity TEXT but report wrong offsets. Rather than
        discarding a good detection because of a counting error, we try to
        locate the entity text in the original string ourselves.

    STRATEGY:
        - Use Python's str.find() to locate the entity text.
        - If found exactly once → use those offsets. This is reliable.
        - If found more than once → we cannot determine which occurrence
          the model meant, so we reject the entity to avoid guessing.
        - If not found at all → the entity text doesn't match the original,
          so we reject it.

    PARAMETERS:
        entity (dict): The entity dict from the LLM (may have wrong offsets).
        original_text (str): The unmodified original user text.

    RETURNS:
        dict with corrected offsets if recovery succeeds, None otherwise.
    """
    entity_text = entity.get("text", "")

    if not entity_text:
        return None

    # Count how many times this exact text appears in the original
    # We do this by finding all occurrences manually
    occurrences = []
    search_start = 0
    while True:
        pos = original_text.find(entity_text, search_start)
        if pos == -1:
            break
        occurrences.append(pos)
        search_start = pos + 1  # move forward to find the next one

    if len(occurrences) == 1:
        # Exactly one match — safe to recover
        corrected_start = occurrences[0]
        corrected_end = corrected_start + len(entity_text)
        return {
            **entity,  # copy all existing fields (type, reason, etc.)
            "start": corrected_start,
            "end": corrected_end,
        }

    # Zero matches → text not in original (entity is fabricated or modified)
    # Multiple matches → ambiguous, cannot safely pick one
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION 3: _validate_entity
# ─────────────────────────────────────────────────────────────────────────────
def _validate_entity(entity: object, original_text: str) -> dict | None:
    """
    Validates a single entity dict returned by the LLM.

    Performs these checks in order:
      1. entity must be a dict
      2. Required fields must be present: text, type, start, end, reason
      3. text must be a non-empty string
      4. type must be one of the allowed VALID_ENTITY_TYPES
      5. start and end must be integers
      6. start must be >= 0
      7. end must be <= len(original_text)
      8. start must be < end
      9. original_text[start:end] must equal entity["text"] exactly
         → If offsets are wrong, attempt recovery via _attempt_offset_recovery

    If all checks pass (or recovery succeeds), returns the validated entity.
    If any check fails and recovery also fails, returns None (entity rejected).

    PARAMETERS:
        entity (any): The raw entity value from the LLM JSON.
        original_text (str): The unmodified original text.

    RETURNS:
        dict (clean entity) or None (rejected).
    """
    # Check 1: must be a dict
    if not isinstance(entity, dict):
        return None

    # Check 2: required fields must exist
    entity = dict(entity)  # work on a copy, never change the caller's dict
    if "text" not in entity:
        return None
    entity.setdefault("type", FALLBACK_ENTITY_TYPE)
    entity.setdefault("reason", "")
    entity.setdefault("start", -1)
    entity.setdefault("end", -1)

    # Check 3: text must be a non-empty string
    entity_text = entity.get("text", "")
    if not isinstance(entity_text, str) or not entity_text.strip():
        return None

    # Check 4: type must be a valid category
    # Never throw a detection away because of its label: fix or fall back.
    entity_type = _normalise_type(entity.get("type"))
    entity["type"] = entity_type

    # Check 5: start and end must be integers (LLMs sometimes return strings)
    try:
        start = int(entity["start"])
        end = int(entity["end"])
    except (ValueError, TypeError):
        # Offsets are non-numeric — try recovery
        return _attempt_offset_recovery(entity, original_text)

    # Check 6 & 7: bounds must be valid
    text_len = len(original_text)
    if start < 0 or end > text_len or start >= end:
        # Bounds are wrong — try recovery
        return _attempt_offset_recovery(entity, original_text)

    # Check 8: the slice must match entity text exactly
    actual_slice = original_text[start:end]
    if actual_slice != entity_text:
        # Offsets don't match the entity text — try recovery
        return _attempt_offset_recovery(entity, original_text)

    # All checks passed — return a clean, normalized entity dict
    return {
        "text": entity_text,
        "type": entity_type,
        "start": start,
        "end": end,
        "reason": str(entity.get("reason", "")).strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION 4: _remove_duplicates_and_overlaps
# ─────────────────────────────────────────────────────────────────────────────
def _remove_duplicates_and_overlaps(entities: list[dict]) -> list[dict]:
    """
    Removes duplicate and overlapping entities from the validated entity list.

    WHY THIS FUNCTION EXISTS:
        A model might detect the same entity twice (exact duplicate), or detect
        a short span that is fully contained inside a longer span
        (e.g., "Rahul" inside "Rahul Sharma"). We prefer longer, more specific
        detections and remove contained shorter ones.

    STRATEGY:
        1. Sort entities by start position, then by span length (longest first).
        2. Walk through the sorted list. Keep an entity only if it does NOT
           overlap with any already-accepted entity.
        3. Two entities overlap if one's start is inside the other's range.

    PARAMETERS:
        entities (list[dict]): List of validated entity dicts.

    RETURNS:
        list[dict]: Deduplicated list sorted by start position.
    """
    if not entities:
        return []

    # Sort: by start position first, then by span length descending
    # This ensures longer spans are preferred when there's overlap
    sorted_entities = sorted(
        entities,
        key=lambda e: (e["start"], -(e["end"] - e["start"])),
    )

    accepted = []
    for candidate in sorted_entities:
        overlaps = False
        for accepted_entity in accepted:
            # Check for any overlap between candidate and an accepted entity
            # Overlap condition: one starts before the other ends
            if not (candidate["end"] <= accepted_entity["start"] or
                    candidate["start"] >= accepted_entity["end"]):
                overlaps = True
                break
        if not overlaps:
            accepted.append(candidate)

    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION 5: _parse_and_validate_response
# ─────────────────────────────────────────────────────────────────────────────
def _parse_and_validate_response(raw_response: str, original_text: str) -> dict:
    """
    Takes one raw model answer and turns it into exact, validated entities:
      1. Extract JSON from the answer and check its shape.
      2. For each detection, find EVERY occurrence in the original text.
      3. Remove duplicates and overlaps (longer span wins).
    """
    parsed = _extract_json_from_response(raw_response)
    error = _check_structure(parsed)
    if error is not None:
        return {"success": False, "entities": [], "message": f"PII detection failed: {error}"}

    located = []
    for raw_entity in parsed["entities"]:
        located.extend(_locate_entity(raw_entity, original_text))
    final_entities = _remove_duplicates_and_overlaps(located)
    return {
        "success": True,
        "entities": final_entities,
        "message": (
            f"PII detection completed successfully. Found {len(final_entities)} sensitive entity(ies)."
            if final_entities
            else "PII detection completed successfully. No sensitive information found."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: detect_pii
# ─────────────────────────────────────────────────────────────────────────────
def detect_pii(text: str) -> dict:
    """Detects PII in `text` using the local Gemma model."""
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False,
            "entities": [],
            "message": "PII detection failed: input text must be a non-empty string.",
        }

    original_text = text  # never modified

    raw_entities: list = []
    for chunk in _split_into_chunks(original_text):
        result = _ask_model(chunk)
        if not result["success"]:      # if ANY piece fails, the whole thing fails (safe)
            return {
                "success": False,
                "entities": [],
                "message": f"PII detection failed: {result['message']}",
            }
        raw_entities.extend(result["entities"])

    if USE_PATTERN_BACKUP:
        raw_entities.extend(find_pattern_entities(original_text))

    located = []
    for raw_entity in raw_entities:
        located.extend(_locate_entity(raw_entity, original_text))
    final_entities = _remove_duplicates_and_overlaps(located)

    return {
        "success": True,
        "entities": final_entities,
        "message": (
            f"PII detection completed successfully. Found {len(final_entities)} sensitive entity(ies)."
            if final_entities
            else "PII detection completed successfully. No sensitive information found."
        ),
    }
