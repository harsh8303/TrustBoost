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

from trustboost.agent.ollama_client import generate_response
from trustboost.detector.prompt_builder import VALID_ENTITY_TYPES, build_detection_prompt


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
def _validate_entity(entity: any, original_text: str) -> dict | None:
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
    required_fields = {"text", "type", "start", "end", "reason"}
    if not required_fields.issubset(entity.keys()):
        # Try recovery even with missing offset fields if text and type exist
        if "text" in entity and "type" in entity:
            entity.setdefault("reason", "")
            entity.setdefault("start", -1)
            entity.setdefault("end", -1)
        else:
            return None

    # Check 3: text must be a non-empty string
    entity_text = entity.get("text", "")
    if not isinstance(entity_text, str) or not entity_text.strip():
        return None

    # Check 4: type must be a valid category
    entity_type = entity.get("type", "")
    if entity_type not in VALID_ENTITY_TYPES:
        return None

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
    Takes the raw LLM output string and runs the full validation pipeline:
      1. Extract JSON from the response text.
      2. Check the top-level structure (must have "entities" key).
      3. Validate each entity individually.
      4. Remove duplicates and overlapping entities.

    PARAMETERS:
        raw_response (str): The raw string returned by the LLM.
        original_text (str): The original unmodified user text.

    RETURNS:
        dict:
            "success" (bool)
            "entities" (list of valid entity dicts)
            "message" (str)
    """
    # Step 1: Parse JSON
    parsed = _extract_json_from_response(raw_response)
    if parsed is None:
        return {
            "success": False,
            "entities": [],
            "message": (
                "PII detection failed: the model returned a response that could "
                "not be parsed as JSON. This may be a temporary model issue. "
                "Please try again."
            ),
        }

    # Step 2: Check top-level structure
    if not isinstance(parsed, dict):
        return {
            "success": False,
            "entities": [],
            "message": "PII detection failed: expected a JSON object at the top level.",
        }

    if "entities" not in parsed:
        return {
            "success": False,
            "entities": [],
            "message": "PII detection failed: JSON response is missing the 'entities' key.",
        }

    if not isinstance(parsed["entities"], list):
        return {
            "success": False,
            "entities": [],
            "message": "PII detection failed: 'entities' must be a JSON array.",
        }

    # Step 3: Validate each entity
    valid_entities = []
    for raw_entity in parsed["entities"]:
        validated = _validate_entity(raw_entity, original_text)
        if validated is not None:
            valid_entities.append(validated)
        # Silently skip invalid entities — the model sometimes hallucinates
        # spurious entries. We keep what's valid and discard the rest.

    # Step 4: Remove duplicates and overlapping spans
    final_entities = _remove_duplicates_and_overlaps(valid_entities)

    return {
        "success": True,
        "entities": final_entities,
        "message": (
            f"PII detection completed successfully. "
            f"Found {len(final_entities)} sensitive entity(ies)."
            if final_entities
            else "PII detection completed successfully. No sensitive information found."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: detect_pii
# ─────────────────────────────────────────────────────────────────────────────
def detect_pii(text: str) -> dict:
    """
    Detects Personally Identifiable Information (PII) in the given text
    using the locally running Gemma 3 4B model via Ollama.

    This is the main public function. All other functions in this file
    are internal helpers called by this one.

    HOW IT WORKS (the full pipeline):
        1. Validate the input text (not empty).
        2. Build a carefully designed detection prompt using prompt_builder.py.
        3. Send the prompt to Gemma 3 4B via the existing Ollama client.
        4. Receive the model's raw JSON response.
        5. Parse and validate the JSON structure.
        6. Validate each entity's fields, type, and offsets.
        7. Attempt offset recovery for entities with wrong positions.
        8. Remove duplicate and overlapping detections.
        9. Return the final structured result.

    IMPORTANT:
        - The original text is NEVER modified by this function.
        - Regex is NEVER used for detection.
        - If detection fails at any step, success=False is returned.
        - A success=True result with zero entities means no PII was found.

    PARAMETERS:
        text (str): The raw text to analyze. Must not be empty.

    RETURNS:
        On success:
            {
                "success": True,
                "entities": [
                    {
                        "text": "Rahul Sharma",
                        "type": "PERSON",
                        "start": 0,
                        "end": 12,
                        "reason": "Full name of an individual"
                    },
                    ...
                ],
                "message": "PII detection completed successfully. Found 1 entity."
            }

        On failure:
            {
                "success": False,
                "entities": [],
                "message": "PII detection failed: <reason>"
            }
    """
    # ── Step 1: Validate input ───────────────────────────────────────────────
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False,
            "entities": [],
            "message": "PII detection failed: input text must be a non-empty string.",
        }

    # Store the original text. We will NEVER modify this variable.
    original_text = text

    # ── Step 2: Build the prompt ─────────────────────────────────────────────
    prompt = build_detection_prompt(original_text)

    # ── Step 3 & 4: Send to Ollama and receive response ──────────────────────
    # We reuse the existing generate_response() from Phase 2.
    # It handles all network errors and returns a structured dict.
    ollama_result = generate_response(prompt)

    if not ollama_result["success"]:
        return {
            "success": False,
            "entities": [],
            "message": f"PII detection failed: {ollama_result['message']}",
        }

    raw_response = ollama_result["response"]

    if not raw_response or not raw_response.strip():
        return {
            "success": False,
            "entities": [],
            "message": "PII detection failed: the model returned an empty response.",
        }

    # ── Step 5–8: Parse, validate, deduplicate ───────────────────────────────
    result = _parse_and_validate_response(raw_response, original_text)

    return result
