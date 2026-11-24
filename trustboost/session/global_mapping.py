"""
trustboost/session/global_mapping.py
--------------------------------------
PURPOSE:
    One global, cumulative, in-memory token mapping for TrustBoost.

    Instead of multiple isolated sessions, ALL sanitization inputs during
    the current Streamlit browser session contribute to ONE shared mapping.

DESIGN:
    - Global counters: Token numbering never resets between inputs.
      If [PERSON_1] is "Rahul Sharma", a new input containing "Aman Verma"
      will get [PERSON_2], not [PERSON_1].
    - Value reuse: If the same (type, original_value) pair appears in a
      later input, its existing token is reused — no duplicate entries.
    - No silent overwrite: A token already in the mapping whose value
      differs from the incoming value is never overwritten. The existing
      entry is reused or a new token is created.
    - Fail-closed: If sanitization fails, the global mapping is never
      touched. Invalid results never corrupt existing mappings.
    - Zero persistence: Everything lives exclusively in Streamlit
      st.session_state — no disk, no database, no cloud.

PUBLIC INTERFACE:
    merge_into_global_mapping(global_state, pipeline_result) -> dict
    restore_with_global_mapping(global_mapping, ai_text)     -> dict
    clear_global_mapping()                                    -> dict
"""

from copy import deepcopy
from typing import Any

from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities


# ── Helpers ────────────────────────────────────────────────────────────────────

def _initial_global_state() -> dict[str, Any]:
    """
    Returns a fresh, empty global state container.

    The global state stores:
      - global_mapping (dict[str, str]):  token -> original_value
      - type_counters  (dict[str, int]):  entity_type -> highest counter used
      - val_to_token   (dict[str, str]):  (type+"||"+value) -> token
    """
    return {
        "global_mapping": {},
        "type_counters": {},
        "val_to_token": {},
    }


def _make_val_key(entity_type: str, original_value: str) -> str:
    """Composite lookup key to identify a (type, value) pair in val_to_token."""
    return f"{entity_type}||{original_value}"


# ── Core Public API ────────────────────────────────────────────────────────────

def merge_into_global_mapping(
    global_state: dict[str, Any],
    new_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merges newly detected entities into the global mapping without resetting
    any counters or overwriting any existing entries.

    PARAMETERS:
        global_state (dict): The current global state dict (from session state).
            Must have keys: 'global_mapping', 'type_counters', 'val_to_token'.
            Pass an empty dict on first call — it will be initialised automatically.
        new_entities (list[dict]): Validated PII entities from the pipeline.
            Each entity must have keys: 'text', 'type'.

    RETURNS:
        dict: Updated global state.  The original dict is NOT mutated; a new
              dict is returned so callers can assign it atomically.

    GUARANTEES:
        - Existing tokens in global_mapping are NEVER overwritten with a different value.
        - Counters only ever increment — they never reset.
        - Identical (type, value) pairs always map to the same token.
        - Different values for the same type always get different tokens.
        - An empty or invalid entity list returns the global state unchanged.
    """
    if not isinstance(global_state, dict):
        raise TypeError("global_state must be a dictionary.")

    if not isinstance(new_entities, list):
        # No valid entities to merge; return state unchanged
        return global_state

    if len(new_entities) == 0:
        return global_state

    # Work on a shallow copy so the caller can assign atomically
    state = deepcopy(global_state)

    # Ensure all required keys exist (handles the first-ever call with an empty dict)
    if "global_mapping" not in state:
        state["global_mapping"] = {}
    if "type_counters" not in state:
        state["type_counters"] = {}
    if "val_to_token" not in state:
        state["val_to_token"] = {}

    for entity in new_entities:
        if not isinstance(entity, dict):
            continue
        raw_type = entity.get("type", "").strip().upper()
        original_value = entity.get("text", "")

        if not raw_type or not original_value:
            continue

        # Normalise type name to match tokenizer format
        sanitized_type = "".join(c if c.isalnum() else "_" for c in raw_type)
        val_key = _make_val_key(sanitized_type, original_value)

        if val_key in state["val_to_token"]:
            # Exact (type, value) pair already has a token — reuse it
            continue

        # Allocate a new token for this unique (type, value) pair
        state["type_counters"][sanitized_type] = (
            state["type_counters"].get(sanitized_type, 0) + 1
        )
        new_token = f"[{sanitized_type}_{state['type_counters'][sanitized_type]}]"
        state["val_to_token"][val_key] = new_token
        state["global_mapping"][new_token] = original_value

    return state


def tokenize_with_global_mapping(
    text: str,
    entities: list[dict[str, Any]],
    global_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Tokenizes `text` using the GLOBAL counter state so that token numbers are
    consistent with all previous inputs.

    Instead of calling the tokenizer's internal counters (which always start
    from 1 for every call), we build the token assignments from the global
    state, then perform a simple string replacement.

    RETURNS:
        dict:
            "success"        (bool)
            "sanitized_text" (str | None)
            "new_entities"   (list[dict])  — entities that were newly added
            "message"        (str)
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False,
            "sanitized_text": None,
            "new_entities": [],
            "message": "Input text must be a non-empty string.",
        }

    if not isinstance(entities, list):
        return {
            "success": False,
            "sanitized_text": None,
            "new_entities": [],
            "message": "Entities must be a list.",
        }

    if len(entities) == 0:
        return {
            "success": True,
            "sanitized_text": text,
            "new_entities": [],
            "message": "No entities detected. Sanitized text is identical to original.",
        }

    if not isinstance(global_state, dict):
        return {
            "success": False,
            "sanitized_text": None,
            "new_entities": [],
            "message": "global_state must be a dictionary.",
        }

    state = deepcopy(global_state)
    if "global_mapping" not in state:
        state["global_mapping"] = {}
    if "type_counters" not in state:
        state["type_counters"] = {}
    if "val_to_token" not in state:
        state["val_to_token"] = {}

    # Validate entity offset invariant (text[start:end] == entity["text"])
    for idx, entity in enumerate(entities):
        if not isinstance(entity, dict):
            return {
                "success": False,
                "sanitized_text": None,
                "new_entities": [],
                "message": f"Entity at index {idx} must be a dict.",
            }
        for key in ("text", "type", "start", "end"):
            if key not in entity:
                return {
                    "success": False,
                    "sanitized_text": None,
                    "new_entities": [],
                    "message": f"Entity at index {idx} missing key '{key}'.",
                }
        s, e = entity["start"], entity["end"]
        if isinstance(s, bool) or isinstance(e, bool) or \
           not isinstance(s, int) or not isinstance(e, int):
            return {
                "success": False,
                "sanitized_text": None,
                "new_entities": [],
                "message": f"Entity offsets at index {idx} must be integers.",
            }
        if s < 0 or e > len(text) or s >= e:
            return {
                "success": False,
                "sanitized_text": None,
                "new_entities": [],
                "message": f"Invalid offsets at index {idx}.",
            }
        if text[s:e] != entity["text"]:
            return {
                "success": False,
                "sanitized_text": None,
                "new_entities": [],
                "message": f"Entity text mismatch at index {idx}.",
            }

    # Resolve overlapping spans (longest wins, same logic as tokenizer)
    sorted_entities = sorted(
        entities,
        key=lambda ent: (ent["start"], -(ent["end"] - ent["start"])),
    )
    accepted: list[dict[str, Any]] = []
    for cand in sorted_entities:
        if not any(
            not (cand["end"] <= acc["start"] or cand["start"] >= acc["end"])
            for acc in accepted
        ):
            accepted.append(cand)
    accepted.sort(key=lambda ent: ent["start"])

    # Build token assignments using global counters
    entity_token_pairs: list[tuple[dict[str, Any], str]] = []
    new_entities: list[dict[str, Any]] = []

    for entity in accepted:
        raw_type = entity["type"].strip().upper()
        sanitized_type = "".join(c if c.isalnum() else "_" for c in raw_type)
        original_value = entity["text"]
        val_key = _make_val_key(sanitized_type, original_value)

        if val_key in state["val_to_token"]:
            token = state["val_to_token"][val_key]
        else:
            state["type_counters"][sanitized_type] = (
                state["type_counters"].get(sanitized_type, 0) + 1
            )
            token = f"[{sanitized_type}_{state['type_counters'][sanitized_type]}]"
            state["val_to_token"][val_key] = token
            state["global_mapping"][token] = original_value
            new_entities.append(entity)

        entity_token_pairs.append((entity, token))

    # Reconstruct sanitized text in one offset-safe pass
    pieces: list[str] = []
    last_idx = 0
    for entity, token in entity_token_pairs:
        pieces.append(text[last_idx:entity["start"]])
        pieces.append(token)
        last_idx = entity["end"]
    pieces.append(text[last_idx:])
    sanitized_text = "".join(pieces)

    return {
        "success": True,
        "sanitized_text": sanitized_text,
        "new_entities": new_entities,
        "updated_global_state": state,
        "message": (
            f"Tokenization completed. "
            f"Replaced {len(entity_token_pairs)} span(s), "
            f"added {len(new_entities)} new global mapping(s)."
        ),
    }


def restore_with_global_mapping(
    global_mapping: dict[str, str],
    ai_text: str,
) -> dict[str, Any]:
    """
    Restores original values in `ai_text` using the single global mapping.

    PARAMETERS:
        global_mapping (dict[str, str]): The accumulated token -> original_value map.
        ai_text (str): AI response text containing TrustBoost tokens.

    RETURNS:
        Standard restore_text dict:
            "success"           (bool)
            "restored_text"     (str | None)
            "unrestored_tokens" (list[str])
            "message"           (str)
    """
    if not isinstance(global_mapping, dict):
        return {
            "success": False,
            "restored_text": None,
            "unrestored_tokens": [],
            "message": "global_mapping must be a dictionary.",
        }
    if not isinstance(ai_text, str):
        return {
            "success": False,
            "restored_text": None,
            "unrestored_tokens": [],
            "message": "AI response text must be a string.",
        }
    return restore_text(ai_text, global_mapping)


def clear_global_mapping() -> dict[str, Any]:
    """
    Returns a fresh, empty global state.
    Calling code should assign this back to st.session_state.
    """
    return _initial_global_state()
