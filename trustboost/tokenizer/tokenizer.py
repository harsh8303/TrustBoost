"""
trustboost/tokenizer/tokenizer.py
---------------------------------
PURPOSE:
    Local, deterministic, and reversible tokenization and restoration
    module for TrustBoost.

    This module handles the core privacy transformation:
      1. tokenize_entities(): Replaces sensitive PII spans with anonymous
         numbered tokens (e.g., [PERSON_1], [EMAIL_1]) and creates a local
         in-memory mapping.
      2. restore_text(): Reverses the tokenization process using the mapping,
         restoring the original values without external network calls or LLMs.

DESIGN PRINCIPLES:
    - 100% Local: No external API or LLM calls are ever made here.
    - Deterministic: Identical inputs produce identical tokens and mappings.
    - Zero Text Mutation: The original input string is never modified.
    - Span Integrity: Slices are reconstructed in a single pass to prevent
      offset invalidation bugs.
    - Token Reuse: Identical values sharing the same entity type reuse the
      same token.
    - Safe Restoration: Token scanning guarantees that partial token matches
      (e.g., [PERSON_1] vs [PERSON_10]) do not cause substring corruption,
      and restored values are never re-tokenized.
"""

from typing import Any


def _is_valid_token_format(token_str: str) -> bool:
    """
    Checks whether a candidate string strictly conforms to the token pattern:
    '[ENTITY_TYPE_INDEX]' where ENTITY_TYPE is non-empty uppercase letters/underscores,
    and INDEX is a positive integer (>= 1).

    Examples:
      - '[PERSON_1]' -> True
      - '[CONFIDENTIAL_PROJECT_2]' -> True
      - '[EMAIL_10]' -> True
      - '[ordinary text]' -> False
      - '[123]' -> False
      - '[PERSON_0]' -> False
      - '[person_1]' -> False
    """
    if len(token_str) < 5 or token_str[0] != "[" or token_str[-1] != "]":
        return False

    inner = token_str[1:-1]
    if "_" not in inner:
        return False

    parts = inner.rsplit("_", 1)
    category, num_str = parts[0], parts[1]

    if not num_str.isdigit() or int(num_str) <= 0:
        return False

    if not category or not all(c.isupper() or c == "_" for c in category):
        return False

    return True


def tokenize_entities(text: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Replaces detected PII entities in the text with structured anonymous tokens.

    PARAMETERS:
        text (str): The original user text to sanitize.
        entities (list[dict]): Validated PII entities, each containing:
            - "text": str (the exact sensitive substring)
            - "type": str (category, e.g. "PERSON", "EMAIL")
            - "start": int (inclusive start character index)
            - "end": int (exclusive end character index)

    RETURNS:
        dict:
            "success" (bool): True if tokenization succeeded, False on validation failure.
            "sanitized_text" (str | None): Sanitized text with tokens, or None on failure.
            "mapping" (dict[str, str]): Local map from token to original value.
            "tokens" (list[dict]): Metadata for each unique token generated.
            "message" (str): Status or error message.
    """
    # ── Step 1: Validate input text ───────────────────────────────────────────
    if not isinstance(text, str):
        return {
            "success": False,
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "message": "Input text must be a string.",
        }

    if len(text) == 0:
        return {
            "success": False,
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "message": "Input text cannot be empty.",
        }

    if not text.strip():
        return {
            "success": False,
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "message": "Input text cannot be whitespace only.",
        }

    # ── Step 2: Validate entities container ───────────────────────────────────
    if not isinstance(entities, list):
        return {
            "success": False,
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "message": "Entities must be a list.",
        }

    # Fast path: No entities detected -> text is already sanitized
    if len(entities) == 0:
        return {
            "success": True,
            "sanitized_text": text,
            "mapping": {},
            "tokens": [],
            "message": "No entities to tokenize. Sanitized text is identical to original.",
        }

    # ── Step 3: Validate individual entity structures and bounds ──────────────
    required_keys = {"text", "type", "start", "end"}
    text_length = len(text)

    for idx, entity in enumerate(entities):
        if not isinstance(entity, dict):
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Entity at index {idx} must be a dictionary.",
            }

        if not required_keys.issubset(entity.keys()):
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Entity at index {idx} is missing required fields (text, type, start, end).",
            }

        entity_text = entity["text"]
        entity_type = entity["type"]
        start = entity["start"]
        end = entity["end"]

        if not isinstance(entity_text, str) or not entity_text:
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Entity at index {idx} text must be a non-empty string.",
            }

        if not isinstance(entity_type, str) or not entity_type.strip():
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Entity at index {idx} type must be a non-empty string.",
            }

        # Check integer offset types (reject booleans which subclass int)
        if isinstance(start, bool) or not isinstance(start, int) or \
           isinstance(end, bool) or not isinstance(end, int):
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Entity at index {idx} offsets must be integers.",
            }

        # Check offset boundaries
        if start < 0 or end > text_length or start >= end:
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Invalid entity offsets: offsets are out of bounds or start >= end at index {idx}.",
            }

        # Invariant check: exact slice match
        if text[start:end] != entity_text:
            return {
                "success": False,
                "sanitized_text": None,
                "mapping": {},
                "tokens": [],
                "message": f"Entity text does not match original text at the given offsets at index {idx}.",
            }

    # ── Step 4: Handle duplicates and overlapping entity spans ────────────────
    # Sort entities by start position ascending, then length descending (longer span wins)
    sorted_entities = sorted(
        entities,
        key=lambda e: (e["start"], -(e["end"] - e["start"])),
    )

    accepted_entities: list[dict[str, Any]] = []
    rejected_overlaps = 0

    for candidate in sorted_entities:
        overlaps = False
        for accepted in accepted_entities:
            # Overlap occurs when one span begins before another ends
            if not (candidate["end"] <= accepted["start"] or candidate["start"] >= accepted["end"]):
                overlaps = True
                break
        if not overlaps:
            accepted_entities.append(candidate)
        else:
            rejected_overlaps += 1

    # Re-sort accepted entities strictly by start position for sequential replacement
    accepted_entities.sort(key=lambda e: e["start"])

    # ── Step 5: Deterministic Token Allocation ─────────────────────────────────
    # Numbering is maintained per entity type starting at 1.
    # Repeated identical (type, original_value) pairs reuse the same token.
    type_counters: dict[str, int] = {}
    type_and_val_to_token: dict[tuple[str, str], str] = {}
    mapping: dict[str, str] = {}
    tokens_metadata: list[dict[str, str]] = []

    entity_token_pairs: list[tuple[dict[str, Any], str]] = []

    for entity in accepted_entities:
        raw_type = entity["type"].strip().upper()
        # Clean type name for token formatting (e.g. replace non-alphanumeric with _)
        sanitized_type = "".join(c if c.isalnum() else "_" for c in raw_type)
        original_value = entity["text"]

        key = (sanitized_type, original_value)

        if key in type_and_val_to_token:
            token = type_and_val_to_token[key]
        else:
            type_counters[sanitized_type] = type_counters.get(sanitized_type, 0) + 1
            token = f"[{sanitized_type}_{type_counters[sanitized_type]}]"
            type_and_val_to_token[key] = token
            mapping[token] = original_value
            tokens_metadata.append({
                "token": token,
                "type": sanitized_type,
                "original": original_value,
            })

        entity_token_pairs.append((entity, token))

    # ── Step 6: Construct Sanitized Text (Offset-Safe Reconstruction) ─────────
    pieces: list[str] = []
    last_idx = 0

    for entity, token in entity_token_pairs:
        pieces.append(text[last_idx:entity["start"]])
        pieces.append(token)
        last_idx = entity["end"]

    pieces.append(text[last_idx:])
    sanitized_text = "".join(pieces)

    message = f"Tokenization completed successfully. Replaced {len(entity_token_pairs)} entity span(s)."
    if rejected_overlaps > 0:
        message += f" ({rejected_overlaps} overlapping or duplicate span(s) safely rejected)."

    return {
        "success": True,
        "sanitized_text": sanitized_text,
        "mapping": mapping,
        "tokens": tokens_metadata,
        "message": message,
    }


def restore_text(sanitized_text: str, mapping: dict[str, str]) -> dict[str, Any]:
    """
    Restores the original text from sanitized text using the local token mapping.

    Performs a single-pass token scan so that:
      - Partial token matches are never replaced accidentally.
      - Restored original values are never re-scanned for token replacement.
      - Unmapped tokens are preserved verbatim and reported.

    PARAMETERS:
        sanitized_text (str): The sanitized text containing tokens.
        mapping (dict[str, str]): Local token-to-original value map.

    RETURNS:
        dict:
            "success" (bool): True if restoration succeeded, False on input validation error.
            "restored_text" (str | None): The reconstructed original text, or None on error.
            "unrestored_tokens" (list[str]): List of valid tokens found in text but missing in mapping.
            "message" (str): Status or diagnostic message.
    """
    # ── Step 1: Validate inputs ───────────────────────────────────────────────
    if not isinstance(sanitized_text, str):
        return {
            "success": False,
            "restored_text": None,
            "unrestored_tokens": [],
            "message": "Sanitized text must be a string.",
        }

    if not isinstance(mapping, dict):
        return {
            "success": False,
            "restored_text": None,
            "unrestored_tokens": [],
            "message": "Mapping must be a dictionary.",
        }

    for k, v in mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return {
                "success": False,
                "restored_text": None,
                "unrestored_tokens": [],
                "message": "All mapping keys and values must be strings.",
            }

    # Fast path: empty sanitized text
    if len(sanitized_text) == 0:
        return {
            "success": True,
            "restored_text": "",
            "unrestored_tokens": [],
            "message": "Sanitized text is empty. Nothing to restore.",
        }

    # ── Step 2: Single-Pass Token Scanning and Restoration ────────────────────
    parts: list[str] = []
    unrestored_tokens: list[str] = []
    i = 0
    text_length = len(sanitized_text)

    while i < text_length:
        if sanitized_text[i] == "[":
            close_idx = sanitized_text.find("]", i)
            if close_idx != -1:
                candidate = sanitized_text[i:close_idx + 1]
                if _is_valid_token_format(candidate):
                    if candidate in mapping:
                        parts.append(mapping[candidate])
                    else:
                        parts.append(candidate)
                        if candidate not in unrestored_tokens:
                            unrestored_tokens.append(candidate)
                    i = close_idx + 1
                    continue

        parts.append(sanitized_text[i])
        i += 1

    restored_text = "".join(parts)

    if unrestored_tokens:
        message = (
            f"Restoration completed with {len(unrestored_tokens)} unrestored token(s). "
            "Unmapped tokens were preserved verbatim."
        )
    else:
        message = "Restoration completed successfully."

    return {
        "success": True,
        "restored_text": restored_text,
        "unrestored_tokens": unrestored_tokens,
        "message": message,
    }
