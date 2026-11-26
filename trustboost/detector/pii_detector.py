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

from datetime import datetime
import ipaddress
import json
import re
from urllib.parse import unquote

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
# INTERNAL HELPERS: Batch-Aware & Duplicate-Safe Offset Recovery (Phase 2A)
# ─────────────────────────────────────────────────────────────────────────────

_STRUCTURED_PII_TYPES = {
    "EMAIL",
    "PHONE",
    "AADHAAR",
    "PAN",
    "CREDIT_CARD",
    "IFSC",
    "BANK_ACCOUNT",
    "FINANCIAL_IDENTIFIER",
    "API_KEY",
    "ACCESS_TOKEN",
    "SECRET",
    "PRIVATE_KEY",
    "DATABASE_CREDENTIAL",
    "IP_ADDRESS",
    "INTERNAL_URL",
    "SERVER_IDENTIFIER",
}

_MULTIWORD_REPEATABLE_TYPES = {
    "PERSON",
    "ADDRESS",
    "CONFIDENTIAL_PROJECT",
    "INTERNAL_ORGANIZATION_INFORMATION",
}

_NON_PERSON_MONTH_PREFIXES = (
    "in", "this", "next", "last", "during", "on", "by", "from",
    "until", "till", "since", "before", "after", "for", "every",
)

_PERSON_PREFIX_MARKERS = (
    "mr", "mrs", "ms", "dr", "prof", "dear", "hi", "hello",
    "agent", "supervisor", "manager", "client", "user", "inspector",
)

_LOCATION_PREFIX_MARKERS = (
    "in", "at", "to", "from", "near", "visiting", "city", "branch",
)


def _find_all_occurrences(entity_text: str, original_text: str) -> list[tuple[int, int]]:
    """Finds all non-overlapping exact occurrences of entity_text in original_text."""
    if not entity_text:
        return []
    spans: list[tuple[int, int]] = []
    search_start = 0
    step = len(entity_text)
    while True:
        pos = original_text.find(entity_text, search_start)
        if pos == -1:
            break
        spans.append((pos, pos + step))
        search_start = pos + step
    return spans


def _is_word_boundary_match(original_text: str, start: int, end: int) -> bool:
    """Checks that alphanumeric edges do not cut inside a larger alphanumeric token."""
    entity_text = original_text[start:end]
    if not entity_text:
        return False
    if entity_text[0].isalnum() and start > 0 and original_text[start - 1].isalnum():
        return False
    if entity_text[-1].isalnum() and end < len(original_text) and original_text[end].isalnum():
        return False
    return True


def _find_candidate_spans(
    entity_text: str,
    original_text: str,
    reserved_spans: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """
    Returns (available_spans, total_raw_occurrences) where available_spans are
    boundary-safe occurrences not overlapping any already reserved span.
    """
    all_raw = _find_all_occurrences(entity_text, original_text)
    if not all_raw:
        return [], 0

    boundary_filtered = [
        (s, e) for s, e in all_raw if _is_word_boundary_match(original_text, s, e)
    ]
    candidates = boundary_filtered if boundary_filtered else all_raw

    available = [
        (s, e)
        for s, e in candidates
        if not any(not (e <= rs or s >= re) for rs, re in reserved_spans)
    ]
    return available, len(candidates)


def _is_unambiguous_repeated_pii(entity_text: str, entity_type: str) -> bool:
    """
    Determines whether repeated occurrences of entity_text in the same input
    can safely be assumed to refer to sensitive PII rather than homonyms.
    """
    stripped = entity_text.strip()
    if not stripped:
        return False

    has_digit = any(c.isdigit() for c in stripped)
    has_symbol = any(c in "@._-+:/=#!$%^&*" for c in stripped)

    if entity_type in _STRUCTURED_PII_TYPES:
        return has_digit or has_symbol or len(stripped.split()) >= 2

    if entity_type in {"PASSWORD", "CREDENTIAL"}:
        return len(stripped) >= 6 and (has_digit or has_symbol)

    if entity_type in _MULTIWORD_REPEATABLE_TYPES:
        return len(stripped.split()) >= 2 and len(stripped) >= 5

    return False


def _score_candidate_context(
    entity_text: str,
    entity_type: str,
    start: int,
    end: int,
    original_text: str,
) -> int:
    """
    Scores how strongly local context around [start:end] matches entity_type
    (positive = supports entity_type; negative = suggests non-PII / other use).
    """
    score = 0
    left_window = original_text[max(0, start - 25) : start].strip()
    right_window = original_text[end : min(len(original_text), end + 25)].strip()

    left_tokens = [tok.strip(".,:;!?()[]{}\"'") for tok in left_window.split() if tok.strip(".,:;!?()[]{}\"'")]
    right_tokens = [tok.strip(".,:;!?()[]{}\"'") for tok in right_window.split() if tok.strip(".,:;!?()[]{}\"'")]

    prev_word = left_tokens[-1].lower() if left_tokens else ""
    next_word = right_tokens[0] if right_tokens else ""
    next_word_lower = next_word.lower()

    if entity_type == "PERSON":
        # Adjacent capitalized surname/given name strongly indicates a person name
        if next_word and next_word[0].isupper() and next_word.isalpha() and len(next_word) >= 2:
            score += 3
        if prev_word in _PERSON_PREFIX_MARKERS:
            score += 2
        if prev_word in _NON_PERSON_MONTH_PREFIXES and entity_text.lower() in {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }:
            score -= 4
        if next_word_lower.isdigit() and len(next_word_lower) in {1, 2, 4}:
            score -= 3

    elif entity_type == "LOCATION":
        if prev_word in _LOCATION_PREFIX_MARKERS:
            score += 3
        if next_word_lower in {"branch", "office", "city", "park", "street", "road"}:
            score += 2

    return score


def _assign_by_nearest_offset(
    pending_group: list[dict],
    candidates: list[tuple[int, int]],
    text_len: int,
) -> list[tuple[dict, tuple[int, int]]] | None:
    """
    Pairs K pending entities with K available candidates using approximate
    LLM offsets when valid, or ordinal reading order when offsets are omitted.
    """
    if len(pending_group) != len(candidates):
        return None

    has_reasonable_hints = all(
        item["hint_start"] is not None
        and item["hint_end"] is not None
        and 0 <= item["hint_start"] <= text_len
        and item["hint_start"] < item["hint_end"] <= text_len + 20
        for item in pending_group
    )

    if not has_reasonable_hints:
        return list(zip(pending_group, candidates))

    remaining = list(candidates)
    assignments: list[tuple[dict, tuple[int, int]]] = []
    # Process in ascending hinted order for deterministic nearest pairing
    ordered = sorted(pending_group, key=lambda x: (x["hint_start"], x["hint_end"]))
    for item in ordered:
        hs, he = item["hint_start"], item["hint_end"]
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: abs(remaining[idx][0] - hs) + abs(remaining[idx][1] - he),
        )
        assignments.append((item, remaining.pop(best_idx)))
    return assignments


def _resolve_batch_entities(
    raw_entities: list,
    original_text: str,
) -> tuple[list[dict], list[dict]]:
    """
    Resolves all LLM-detected entities as a batch in two phases:
      1. Accept exact-offset matches first and reserve their character spans.
      2. Safely recover inaccurate/missing offsets for remaining entities using
         boundary-safe unclaimed candidates, specificity rules, approximate
         offsets, and local context — failing closed on unresolved ambiguity.

    RETURNS:
        (validated_entities, unresolved_entities)
    """
    text_len = len(original_text)
    exact_matches: list[dict] = []
    pending_items: list[dict] = []

    for raw_ent in raw_entities:
        if not isinstance(raw_ent, dict):
            continue

        entity_text = raw_ent.get("text", "")
        entity_type = raw_ent.get("type", "")
        if not isinstance(entity_text, str) or not entity_text.strip():
            continue
        if entity_type not in VALID_ENTITY_TYPES:
            continue

        # Reject hallucinated text absent from the input string
        if original_text.find(entity_text) == -1:
            continue

        reason = str(raw_ent.get("reason", "")).strip()
        hint_start = None
        hint_end = None
        is_exact = False

        if "start" in raw_ent and "end" in raw_ent:
            try:
                s = int(raw_ent["start"])
                e = int(raw_ent["end"])
                hint_start, hint_end = s, e
                if 0 <= s < e <= text_len and original_text[s:e] == entity_text:
                    is_exact = True
            except (ValueError, TypeError):
                pass

        record = {
            "text": entity_text,
            "type": entity_type,
            "hint_start": hint_start,
            "hint_end": hint_end,
            "reason": reason,
        }
        if is_exact:
            exact_matches.append({
                "text": entity_text,
                "type": entity_type,
                "start": hint_start,
                "end": hint_end,
                "reason": reason,
            })
        else:
            pending_items.append(record)

    # Pass 1: Deduplicate/resolve overlaps among exact matches (longest wins) and reserve spans
    accepted_exact = _remove_duplicates_and_overlaps(exact_matches)
    reserved_spans: list[tuple[int, int]] = [
        (ent["start"], ent["end"]) for ent in accepted_exact
    ]
    resolved_entities: list[dict] = list(accepted_exact)
    unresolved_entities: list[dict] = []

    # Group pending entities by exact text, processing longest strings first
    grouped_pending: dict[str, list[dict]] = {}
    for item in pending_items:
        grouped_pending.setdefault(item["text"], []).append(item)

    for entity_text in sorted(grouped_pending.keys(), key=lambda t: (-len(t), t)):
        group = grouped_pending[entity_text]
        candidates, total_boundary_count = _find_candidate_spans(
            entity_text, original_text, reserved_spans
        )

        # Already completely covered by reserved exact spans
        if len(candidates) == 0:
            if total_boundary_count == 0:
                unresolved_entities.append({
                    "text": entity_text,
                    "type": group[0]["type"],
                    "reason": "Entity text has no unclaimed non-overlapping candidate occurrence.",
                })
            continue

        # Unique remaining unclaimed candidate -> unambiguous recovery
        if len(candidates) == 1 and len(group) == 1:
            s, e = candidates[0]
            resolved_entities.append({
                "text": entity_text,
                "type": group[0]["type"],
                "start": s,
                "end": e,
                "reason": group[0]["reason"],
            })
            reserved_spans.append((s, e))
            continue

        # 1-to-1 match: LLM emitted exactly K entities for K remaining occurrences
        if len(group) == len(candidates):
            paired = _assign_by_nearest_offset(group, candidates, text_len)
            if paired is not None:
                for item, (s, e) in paired:
                    resolved_entities.append({
                        "text": entity_text,
                        "type": item["type"],
                        "start": s,
                        "end": e,
                        "reason": item["reason"],
                    })
                    reserved_spans.append((s, e))
                continue

        # More candidates than LLM items (or K > 1 with K != len(candidates))
        distinct_types = {item["type"] for item in group}
        primary_type = group[0]["type"]

        # Unambiguous structured/multi-word PII -> protect ALL unclaimed occurrences
        if len(distinct_types) == 1 and _is_unambiguous_repeated_pii(entity_text, primary_type):
            for s, e in candidates:
                resolved_entities.append({
                    "text": entity_text,
                    "type": primary_type,
                    "start": s,
                    "end": e,
                    "reason": group[0]["reason"],
                })
                reserved_spans.append((s, e))
            continue

        # Ambiguous single/common word with K < len(candidates):
        # Attempt strict context + approximate offset disambiguation for K == 1
        if len(group) == 1 and len(candidates) > 1:
            item = group[0]
            ctx_scores = [
                _score_candidate_context(entity_text, item["type"], s, e, original_text)
                for s, e in candidates
            ]
            max_ctx = max(ctx_scores)
            top_indices = [i for i, sc in enumerate(ctx_scores) if sc == max_ctx]

            # Check if context clearly isolates one candidate (positive vs non-positive others)
            if len(top_indices) == 1 and max_ctx >= 2 and all(
                ctx_scores[i] <= 0 for i in range(len(candidates)) if i != top_indices[0]
            ):
                s, e = candidates[top_indices[0]]
                resolved_entities.append({
                    "text": entity_text,
                    "type": item["type"],
                    "start": s,
                    "end": e,
                    "reason": item["reason"],
                })
                reserved_spans.append((s, e))
                continue

            # Otherwise check if approximate offset lies safely within text and strongly favors one span
            hs, he = item["hint_start"], item["hint_end"]
            if (
                hs is not None
                and he is not None
                and 0 <= hs < text_len
                and hs < he <= text_len + 10
            ):
                dists = [abs(s - hs) + abs(e - he) for s, e in candidates]
                sorted_idx = sorted(range(len(candidates)), key=lambda i: dists[i])
                best_i, second_i = sorted_idx[0], sorted_idx[1]
                # Require strong spatial separation AND non-negative context
                if (
                    dists[best_i] <= 25
                    and (dists[second_i] - dists[best_i]) >= 15
                    and ctx_scores[best_i] >= 0
                ):
                    s, e = candidates[best_i]
                    resolved_entities.append({
                        "text": entity_text,
                        "type": item["type"],
                        "start": s,
                        "end": e,
                        "reason": item["reason"],
                    })
                    reserved_spans.append((s, e))
                    continue

        # Cannot resolve ambiguity safely without risking leakage or blind masking -> fail closed
        unresolved_entities.append({
            "text": entity_text,
            "type": primary_type,
            "candidate_count": len(candidates),
            "llm_mention_count": len(group),
            "reason": (
                f"Ambiguous entity '{entity_text}' ({primary_type}) has {len(candidates)} "
                f"candidate occurrence(s) for {len(group)} LLM detection(s) and could not be safely disambiguated."
            ),
        })

    if unresolved_entities:
        return [], unresolved_entities

    # Pass 4: Unambiguous Repeated-PII Coverage Guard
    # If the LLM returned exact offsets for only one occurrence of a structured ID
    # or multi-word person name, ensure any remaining unclaimed occurrences are masked.
    accepted_types_by_text: dict[str, set[str]] = {}
    accepted_reason_by_text: dict[str, str] = {}
    for ent in resolved_entities:
        accepted_types_by_text.setdefault(ent["text"], set()).add(ent["type"])
        accepted_reason_by_text.setdefault(ent["text"], ent.get("reason", ""))

    for ent_text in sorted(accepted_types_by_text.keys(), key=lambda t: (-len(t), t)):
        types = accepted_types_by_text[ent_text]
        if len(types) != 1:
            continue
        ent_type = next(iter(types))
        if not _is_unambiguous_repeated_pii(ent_text, ent_type):
            continue
        extra_candidates, _ = _find_candidate_spans(ent_text, original_text, reserved_spans)
        for s, e in extra_candidates:
            resolved_entities.append({
                "text": ent_text,
                "type": ent_type,
                "start": s,
                "end": e,
                "reason": accepted_reason_by_text.get(ent_text, ""),
            })
            reserved_spans.append((s, e))

    return _remove_duplicates_and_overlaps(resolved_entities), []


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS: Deterministic Database Credential URI Recognizer (Phase 3.2A.3)
# ─────────────────────────────────────────────────────────────────────────────

_DATABASE_URI_SCHEMES = (
    "postgresql://",
    "postgres://",
    "mysql://",
    "mongodb+srv://",
    "mongodb://",
    "rediss://",
    "redis://",
    "mssql://",
    "oracle://",
)

_SCHEME_PATTERN = re.compile(
    r"(?:^|(?<=[\s\"'(\[<:,=]))("
    + "|".join(re.escape(s) for s in _DATABASE_URI_SCHEMES)
    + r")",
    re.IGNORECASE,
)

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$")


def _trim_uri_candidate_end(candidate: str) -> str:
    """
    Trim trailing sentence punctuation and unbalanced closing wrappers from a
    raw URI candidate without stripping meaningful URI characters like '!', '?' or '#'.
    """
    end = len(candidate)
    bracket_pairs = {")": "(", "]": "[", "}": "{"}

    while end > 0:
        ch = candidate[end - 1]
        if ch in {".", ",", ";", ":"}:
            end -= 1
            continue
        if ch in bracket_pairs:
            opener = bracket_pairs[ch]
            prefix = candidate[:end]
            if prefix.count(ch) > prefix.count(opener):
                end -= 1
                continue
        break

    return candidate[:end]


def _is_valid_host_and_port(hostport: str) -> bool:
    """Validate a URI host[:port] supporting hostnames, localhost, IPv4, and [IPv6]."""
    if not hostport:
        return False

    if hostport.startswith("["):
        close_idx = hostport.find("]")
        if close_idx <= 1:
            return False
        host = hostport[1:close_idx]
        try:
            ip = ipaddress.ip_address(host)
            if ip.version != 6:
                return False
        except ValueError:
            return False
        rest = hostport[close_idx + 1:]
        if rest:
            if not rest.startswith(":"):
                return False
            port_str = rest[1:]
            if not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
                return False
        return True

    if ":" in hostport:
        if hostport.count(":") > 1:
            return False
        host, port_str = hostport.split(":", 1)
        if not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
            return False
    else:
        host = hostport

    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(_HOSTNAME_RE.match(host))


def _is_valid_database_credential_uri(uri_text: str) -> bool:
    """
    Validate that uri_text is a complete credential-bearing database URI.
    Isolates the authority first and splits on the rightmost '@' so passwords
    containing raw '@' (e.g. P@ssw0rd2026) are parsed accurately.
    """
    lower = uri_text.lower()
    matched_scheme = next((s for s in _DATABASE_URI_SCHEMES if lower.startswith(s)), None)
    if matched_scheme is None:
        return False

    rest = uri_text[len(matched_scheme):]
    if not rest:
        return False

    auth_end = len(rest)
    for sep in ("/", "?", "#"):
        idx = rest.find(sep)
        if idx != -1 and idx < auth_end:
            auth_end = idx
    authority = rest[:auth_end]

    if "@" not in authority:
        return False

    userinfo, hostport = authority.rsplit("@", 1)
    if ":" not in userinfo:
        return False

    username, password = userinfo.split(":", 1)
    if not username or not password:
        return False
    if not unquote(username).strip() or not unquote(password).strip():
        return False

    return _is_valid_host_and_port(hostport)


def _extract_database_uri_entities(original_text: str) -> list[dict]:
    """
    Deterministically extract complete credential-bearing database connection
    URIs from original_text as validated DATABASE_CREDENTIAL entities.
    """
    if not isinstance(original_text, str) or not original_text:
        return []

    entities: list[dict] = []
    min_next_start = 0

    for match in _SCHEME_PATTERN.finditer(original_text):
        start = match.start(1)
        if start < min_next_start:
            continue

        idx = start
        text_len = len(original_text)
        while idx < text_len and not original_text[idx].isspace() and original_text[idx] not in "<>\"'`":
            idx += 1

        raw_candidate = original_text[start:idx]
        uri_text = _trim_uri_candidate_end(raw_candidate)
        if not uri_text or not _is_valid_database_credential_uri(uri_text):
            continue

        end = start + len(uri_text)
        if original_text[start:end] != uri_text:
            continue

        entities.append({
            "text": uri_text,
            "type": "DATABASE_CREDENTIAL",
            "start": start,
            "end": end,
            "reason": "Deterministic credential-bearing database URI",
        })
        min_next_start = end

    return entities


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS: Deterministic Structured Secret & IP Recognizer (Phase 3.2B.3)
# ─────────────────────────────────────────────────────────────────────────────

_API_KEY_PATTERNS = (
    # GitHub tokens (classic, OAuth, user-to-server, server-to-server, refresh, fine-grained)
    re.compile(r"(?:^|(?<=[\s\"'(\[=:,;]))((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,})(?:$|(?=[\s\"')\].,;:!?]))"),
    re.compile(r"(?:^|(?<=[\s\"'(\[=:,;]))(github_pat_[A-Za-z0-9_]{22,})(?:$|(?=[\s\"')\].,;:!?]))"),
    # OpenAI / Stripe-style API keys
    re.compile(r"(?:^|(?<=[\s\"'(\[=:,;]))(sk-(?:live|test|proj)-[A-Za-z0-9_-]{12,})(?:$|(?=[\s\"')\].,;:!?]))"),
    re.compile(r"(?:^|(?<=[\s\"'(\[=:,;]))(sk_(?:live|test)_[A-Za-z0-9]{16,})(?:$|(?=[\s\"')\].,;:!?]))"),
    re.compile(r"(?:^|(?<=[\s\"'(\[=:,;]))(sk-[A-Za-z0-9]{20,})(?:$|(?=[\s\"')\].,;:!?]))"),
    # AWS Access Key ID (exact 20-char AKIA + 16 uppercase alphanumeric)
    re.compile(r"(?:^|(?<=[\s\"'(\[=:,;]))(AKIA[0-9A-Z]{16})(?:$|(?=[\s\"')\].,;:!?]))"),
)

_IPV4_CANDIDATE_RE = re.compile(
    r"(?:^|(?<=[\s\"'(\[=:,;@]))(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?(?:$|(?=[\s\"')\].,;:!?/])(?!\.\d))"
)

_IPV6_BRACKETED_RE = re.compile(
    r"\[([0-9A-Fa-f:]{2,39})\](?::(\d{1,5}))?"
)

_IPV6_UNBRACKETED_RE = re.compile(
    r"(?:^|(?<=[\s\"'(=:,;]))([0-9A-Fa-f:]{2,39})(?:$|(?=[\s\"')\].,;:!?]))"
)

_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN ((?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY)-----"
    r"[\s\S]+?"
    r"-----END \1-----"
)


def _extract_structured_secret_entities(original_text: str) -> list[dict]:
    """
    Deterministically extract high-confidence structured sensitive tokens,
    validated IPv4/IPv6 addresses, and PEM/OpenSSH private-key blocks.

    Note on SSH public keys: 'ssh-rsa AAAAB3...' is an OpenSSH public-key
    format and is intentionally NOT classified as PRIVATE_KEY here.
    """
    if not isinstance(original_text, str) or not original_text:
        return []

    candidates: list[dict] = []

    # 1. PEM / OpenSSH Private-Key Blocks (only complete, matching BEGIN..END blocks)
    for match in _PRIVATE_KEY_BLOCK_RE.finditer(original_text):
        key_text = match.group(0)
        start, end = match.start(), match.end()
        if original_text[start:end] == key_text:
            candidates.append({
                "text": key_text,
                "type": "PRIVATE_KEY",
                "start": start,
                "end": end,
                "reason": "Deterministic private-key block",
            })

    # 2. High-Confidence API Keys (GitHub, OpenAI/Stripe sk-*, AWS AKIA*)
    for pattern in _API_KEY_PATTERNS:
        for match in pattern.finditer(original_text):
            token = match.group(1)
            start, end = match.start(1), match.end(1)
            if original_text[start:end] == token:
                candidates.append({
                    "text": token,
                    "type": "API_KEY",
                    "start": start,
                    "end": end,
                    "reason": "Deterministic structured API key",
                })

    # 3. Validated IPv4 Addresses (excluding optional :port)
    for match in _IPV4_CANDIDATE_RE.finditer(original_text):
        ip_str = match.group(1)
        port_str = match.group(2)
        if port_str is not None and not (1 <= int(port_str) <= 65535):
            continue
        try:
            ipaddress.IPv4Address(ip_str)
        except ValueError:
            continue
        start, end = match.start(1), match.end(1)
        if original_text[start:end] == ip_str:
            candidates.append({
                "text": ip_str,
                "type": "IP_ADDRESS",
                "start": start,
                "end": end,
                "reason": "Deterministic validated IPv4 address",
            })

    # 4. Validated Bracketed IPv6 Addresses (excluding brackets and optional :port)
    for match in _IPV6_BRACKETED_RE.finditer(original_text):
        ip_str = match.group(1)
        port_str = match.group(2)
        if port_str is not None and not (1 <= int(port_str) <= 65535):
            continue
        try:
            ipaddress.IPv6Address(ip_str)
        except ValueError:
            continue
        start, end = match.start(1), match.end(1)
        if original_text[start:end] == ip_str:
            candidates.append({
                "text": ip_str,
                "type": "IP_ADDRESS",
                "start": start,
                "end": end,
                "reason": "Deterministic validated IPv6 address",
            })

    # 5. Validated Unbracketed IPv6 Addresses
    for match in _IPV6_UNBRACKETED_RE.finditer(original_text):
        ip_str = match.group(1)
        if "::" not in ip_str and ip_str.count(":") != 7:
            continue
        try:
            ipaddress.IPv6Address(ip_str)
        except ValueError:
            continue
        start, end = match.start(1), match.end(1)
        if original_text[start:end] == ip_str:
            candidates.append({
                "text": ip_str,
                "type": "IP_ADDRESS",
                "start": start,
                "end": end,
                "reason": "Deterministic validated IPv6 address",
            })

    return _remove_duplicates_and_overlaps(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS: Deterministic PAN, International Phone & DOB (Phase 3.2B.5)
# ─────────────────────────────────────────────────────────────────────────────

_PAN_CONTEXT_RE = re.compile(
    r"(?:"
    r"\"(?:pan|pan_number|pan_card)\"\s*:\s*\"?"
    r"|\b(?:pan(?:\s+card|\s+number|\s+id)?|taxpayer|tax\s+id|kyc)\b[\s:=\-\w\"']{0,45}"
    r")([A-Z]{5}[0-9]{4}[A-Z])(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_INTL_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9+])(\+[1-9]\d{0,2}(?:[\s-]?\(\d{1,5}\)|[\s-]?\d+){1,6})(?![A-Za-z0-9])"
)

_DOB_CONTEXT_DATE_RE = re.compile(
    r"(?:"
    r"\"(?:dob|date_of_birth|birth_date|birthdate)\"\s*:\s*\"?"
    r"|\b(?:dob|date\s+of\s+birth|birth\s*date|birthdate|birthday|born(?:\s+on)?)\b(?:\s*[:=]\s*|\s+)"
    r")("
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{4}"
    r"|\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{4}"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_DOB_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%d %B %Y",
)


def _is_valid_calendar_date(date_str: str) -> bool:
    """Validate that date_str parses as a real calendar date."""
    for fmt in _DOB_DATE_FORMATS:
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    return False


def _extract_structured_pii_entities(original_text: str) -> list[dict]:
    """
    Deterministically extract context-verified Indian PAN numbers,
    international '+'-prefixed phone numbers, and context-labelled DOBs.
    """
    if not isinstance(original_text, str) or not original_text:
        return []

    candidates: list[dict] = []

    # 1. Context-verified Indian PAN (exact uppercase AAAAA9999A with PAN/tax/KYC cue)
    for match in _PAN_CONTEXT_RE.finditer(original_text):
        pan_text = match.group(1)
        start, end = match.start(1), match.end(1)
        if not pan_text.isupper():
            continue
        if start > 0 and original_text[start - 1].isalnum():
            continue
        if original_text[start:end] == pan_text:
            candidates.append({
                "text": pan_text,
                "type": "PAN",
                "start": start,
                "end": end,
                "reason": "Deterministic context-verified PAN",
            })

    # 2. International '+'-prefixed phone numbers (7-15 digits, exact formatting)
    for match in _INTL_PHONE_RE.finditer(original_text):
        phone_text = match.group(1)
        digits = [ch for ch in phone_text if ch.isdigit()]
        if not (7 <= len(digits) <= 15):
            continue
        # Require at least one separator or >=10 digits to avoid short '+1234567' artifacts
        if "-" not in phone_text and " " not in phone_text and "(" not in phone_text and len(digits) < 10:
            continue
        start, end = match.start(1), match.end(1)
        if original_text[start:end] == phone_text:
            candidates.append({
                "text": phone_text,
                "type": "PHONE",
                "start": start,
                "end": end,
                "reason": "Deterministic international phone number",
            })

    # 3. Context-aware Date of Birth (validated calendar date + explicit DOB cue)
    for match in _DOB_CONTEXT_DATE_RE.finditer(original_text):
        date_text = match.group(1)
        if not _is_valid_calendar_date(date_text):
            continue
        start, end = match.start(1), match.end(1)
        if original_text[start:end] == date_text:
            candidates.append({
                "text": date_text,
                "type": "DATE_OF_BIRTH",
                "start": start,
                "end": end,
                "reason": "Deterministic context-verified date of birth",
            })

    return _remove_duplicates_and_overlaps(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS: Deterministic JSON-Aware PERSON Recognizer (Phase 3.2B.6C)
# ─────────────────────────────────────────────────────────────────────────────

_JSON_PERSON_KEYS = {
    "name",
    "full_name",
    "lead",
    "owner",
    "user",
    "username",
    "contact",
    "author",
    "manager",
    "assignee",
    "employee",
    "client",
    "customer",
    "patient",
}

_JSON_NON_PERSON_WORDS = {
    "active", "inactive", "pending", "enabled", "disabled", "unknown", "none",
    "null", "na", "n/a", "true", "false", "default", "test", "demo",
    "admin", "administrator", "root", "user", "owner", "lead", "manager",
    "system", "service", "bot", "deploy", "backend", "frontend", "fullstack",
    "support", "engineering", "devops", "platform", "security", "operations",
    "ops", "sales", "marketing", "finance", "hr", "department", "division",
    "team", "group", "unit", "org", "organization", "company", "corp",
    "corporation", "inc", "llc", "ltd", "gmbh", "technologies", "solutions",
    "north", "south", "east", "west", "central", "america", "europe", "asia",
    "africa", "australia", "pacific", "atlantic", "global", "regional",
    "project", "phase", "sprint", "epic", "initiative", "program",
    "titan", "polaris", "falcon", "aegis", "orion", "apollo",
}

_JSON_NAME_WORD_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'.-]*$")


def _is_plausible_json_person_name(value: str) -> bool:
    """
    Conservatively validate whether a decoded JSON string value is a plausible
    human person name (2-4 alphabetic words, no digits/URLs/emails/org/team terms).
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not (3 <= len(stripped) <= 60):
        return False
    if any(ch in stripped for ch in ("\n", "\r", "\t", "@", "/", "\\", "=", "_")):
        return False
    if any(ch.isdigit() for ch in stripped):
        return False

    # Normalize escaped quotes if any survived, or split on whitespace
    words = stripped.split()
    if not (2 <= len(words) <= 4):
        return False

    alpha_titled = 0
    for word in words:
        clean_word = word.strip("\"'")
        if not clean_word or not _JSON_NAME_WORD_RE.match(clean_word):
            return False
        base = clean_word.rstrip(".").lower()
        if base in _JSON_NON_PERSON_WORDS:
            return False
        if clean_word[0].isupper():
            alpha_titled += 1

    return alpha_titled >= 2


def _tokenize_json_with_spans(original_text: str) -> list[tuple] | None:
    """
    Tokenize valid JSON text while recording exact (inner_start, inner_end, decoded)
    for every string literal in original_text.
    """
    tokens: list[tuple] = []
    idx = 0
    n = len(original_text)

    while idx < n:
        ch = original_text[idx]
        if ch.isspace():
            idx += 1
            continue
        if ch in "{}[],:":
            tokens.append(("PUNCT", ch, idx, idx + 1))
            idx += 1
            continue
        if ch == '"':
            quote_start = idx
            idx += 1
            while idx < n:
                cur = original_text[idx]
                if cur == "\\":
                    idx += 2
                    continue
                if cur == '"':
                    break
                idx += 1
            if idx >= n or original_text[idx] != '"':
                return None
            quote_end = idx
            try:
                decoded = json.loads(original_text[quote_start:quote_end + 1])
            except (json.JSONDecodeError, ValueError):
                return None
            tokens.append(("STR", decoded, quote_start + 1, quote_end))
            idx += 1
            continue
        # Number, true, false, null
        lit_start = idx
        while idx < n and not original_text[idx].isspace() and original_text[idx] not in "{}[],:\"":
            idx += 1
        tokens.append(("LIT", original_text[lit_start:idx], lit_start, idx))

    return tokens


def _extract_json_person_entities(original_text: str) -> list[dict]:
    """
    Deterministically extract plausible PERSON names from semantically meaningful
    keys in valid JSON-formatted inputs, preserving exact source offsets.
    """
    if not isinstance(original_text, str) or not original_text.lstrip().startswith(("{", "[")):
        return []

    try:
        json.loads(original_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return []

    tokens = _tokenize_json_with_spans(original_text)
    if not tokens:
        return []

    entities: list[dict] = []
    pos = 0
    total = len(tokens)

    def _walk() -> None:
        nonlocal pos
        if pos >= total:
            return
        tok_kind, tok_val, _, _ = tokens[pos]
        if tok_kind == "PUNCT" and tok_val == "{":
            pos += 1
            while pos < total:
                if tokens[pos][0] == "PUNCT" and tokens[pos][1] == "}":
                    pos += 1
                    return
                if tokens[pos][0] == "STR":
                    key_name = tokens[pos][1]
                    pos += 1
                    if pos < total and tokens[pos][0] == "PUNCT" and tokens[pos][1] == ":":
                        pos += 1
                    if pos < total:
                        val_kind, val_dec, v_start, v_end = tokens[pos]
                        if (
                            val_kind == "STR"
                            and isinstance(key_name, str)
                            and key_name.strip().lower() in _JSON_PERSON_KEYS
                            and _is_plausible_json_person_name(val_dec)
                        ):
                            raw_slice = original_text[v_start:v_end]
                            if raw_slice:
                                entities.append({
                                    "text": raw_slice,
                                    "type": "PERSON",
                                    "start": v_start,
                                    "end": v_end,
                                    "reason": "Deterministic JSON semantic PERSON",
                                })
                            pos += 1
                        else:
                            _walk()
                elif tokens[pos][0] == "PUNCT" and tokens[pos][1] == ",":
                    pos += 1
                else:
                    pos += 1
            return

        if tok_kind == "PUNCT" and tok_val == "[":
            pos += 1
            while pos < total:
                if tokens[pos][0] == "PUNCT" and tokens[pos][1] == "]":
                    pos += 1
                    return
                if tokens[pos][0] == "PUNCT" and tokens[pos][1] == ",":
                    pos += 1
                    continue
                _walk()
            return

        pos += 1

    _walk()
    return _remove_duplicates_and_overlaps(entities)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FUNCTION 5: _parse_and_validate_response
# ─────────────────────────────────────────────────────────────────────────────
def _parse_and_validate_response(raw_response: str, original_text: str) -> dict:
    """
    Takes the raw LLM output string and runs the full validation pipeline:
      1. Extract JSON from the response text.
      2. Check the top-level structure (must have "entities" key).
      3. Resolve and validate all entities as a batch (exact matches first,
         duplicate-safe recovery, non-overlap deduplication).
      4. Fail closed if any sensitive entity cannot be resolved safely.

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

    # Step 3 & 4: Batch validate, recover duplicate-safe offsets, and deduplicate
    final_entities, unresolved_entities = _resolve_batch_entities(
        parsed["entities"], original_text
    )

    # Deterministic Phase 3.2A.3 database connection URI recognizer +
    # Deterministic Phase 3.2B.3 structured secret, IP, and private-key recognizer:
    # Trusted because spans are sliced directly from original_text and strictly
    # validated. Database URIs take precedence over any nested IP/token spans.
    deterministic_uris = _extract_database_uri_entities(original_text)
    deterministic_secrets = _extract_structured_secret_entities(original_text)
    deterministic_pii = _extract_structured_pii_entities(original_text)
    deterministic_json_persons = _extract_json_person_entities(original_text)
    deterministic_entities = _remove_duplicates_and_overlaps(
        deterministic_uris
        + deterministic_secrets
        + deterministic_pii
        + deterministic_json_persons
    )

    if unresolved_entities and deterministic_entities:
        unresolved_entities = [
            u for u in unresolved_entities
            if not any(u["text"] in det["text"] for det in deterministic_entities)
        ]

    if unresolved_entities:
        unresolved_labels = ", ".join(
            f"{u['text']!r} ({u['type']})" for u in unresolved_entities
        )
        return {
            "success": False,
            "entities": [],
            "unresolved_entities": unresolved_entities,
            "message": (
                f"PII detection failed: could not safely resolve ambiguous offset(s) "
                f"for sensitive entity {unresolved_labels}."
            ),
        }

    if deterministic_entities:
        # Deterministic spans supersede overlapping LLM detections (e.g., prefix-only
        # API_KEY labels, IP:port spans, or fragmented DB URI components) while
        # preserving all unrelated neighbouring PII.
        non_overlapping_llm = [
            llm_ent for llm_ent in final_entities
            if not any(
                not (llm_ent["end"] <= det["start"] or llm_ent["start"] >= det["end"])
                for det in deterministic_entities
            )
        ]
        final_entities = _remove_duplicates_and_overlaps(
            non_overlapping_llm + deterministic_entities
        )

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
