"""
trustboost/pipeline/pipeline.py
-------------------------------
PURPOSE:
    Coordinated processing pipeline for TrustBoost.
    Responsible for coordinating deterministic processing operations:

    1. Sanitization Workflow:
       - Step 1: Receive input and validate boundary conditions.
       - Step 2: Build the detection prompt (prompt_builder.py).
       - Step 3: Call Gemma 3 4B through the existing Ollama client (ollama_client.py).
       - Step 4: Parse detection results (_extract_json_from_response).
       - Step 5: Validate all detected entities, types, and exact slice offsets.
       - Step 6: Reject invalid results and fail closed.
       - Step 7: Call global tokenization (tokenize_with_global_mapping).
       - Step 8: Update the existing global mapping state.
       - Step 9: Return sanitized text, detected entities, and metadata.
       No mapping update happens before successful validation.

    2. Restoration Workflow:
       - Step 1: Receive AI-generated response.
       - Step 2: Retrieve the existing global mapping.
       - Step 3: Restore all known placeholders.
       - Step 4: Preserve unknown placeholders.
       - Step 5: Report unmapped tokens.
       - Step 6: Return the restored response.

DESIGN CONSTRAINTS:
    - Never pretend sanitization succeeded if detection or validation fails.
    - Never return original unsanitized text as if it were sanitized on error.
    - Preserve the original input text immutably.
    - Never leak sensitive PII into error messages or logs.
    - Strictly local execution — zero external API calls.
"""

from typing import Any
from trustboost.detector.pii_detector import detect_pii
from trustboost.session.global_mapping import (
    clear_global_mapping,
    tokenize_with_global_mapping,
    restore_with_global_mapping,
)
from trustboost.tokenizer.tokenizer import restore_text as base_restore_text


# ── Sanitization Pipeline ─────────────────────────────────────────────────────

def run_sanitization_pipeline(
    text: str,
    global_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Executes the controlled 9-step sanitization workflow:
      1. Receive input and validate (non-empty, non-whitespace string).
      2. Build the detection prompt.
      3. Call Gemma 3 4B through the existing Ollama client.
      4. Parse detection results.
      5. Validate all detected entities and offsets.
      6. Reject invalid results.
      7. Call global tokenization.
      8. Update the existing global mapping state.
      9. Return sanitized text and detection information.

    PARAMETERS:
        text (str): Raw input text to sanitize.
        global_state (dict | None): The current in-memory global mapping state.
                                    If None, a fresh state is initialized.

    RETURNS:
        dict with keys:
            "success"               (bool)
            "original_text"         (str)
            "detected_entities"     (list[dict])
            "sanitized_text"        (str | None)
            "mapping"               (dict[str, str])
            "tokens"                (list[dict])
            "updated_global_state"  (dict)
            "message"               (str)
    """
    # ── Step 1: Receive and validate input ─────────────────────────────────────
    current_state = global_state if global_state is not None else clear_global_mapping()

    if not isinstance(text, str):
        return {
            "success": False,
            "original_text": text if text is not None else "",
            "detected_entities": [],
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "updated_global_state": current_state,
            "message": "Input text must be a string.",
        }

    if len(text) == 0:
        return {
            "success": False,
            "original_text": "",
            "detected_entities": [],
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "updated_global_state": current_state,
            "message": "Input text cannot be empty.",
        }

    if not text.strip():
        return {
            "success": False,
            "original_text": text,
            "detected_entities": [],
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "updated_global_state": current_state,
            "message": "Input text cannot be whitespace only.",
        }

    original_text = text

    # ── Steps 2–6: Prompt building, LLM inference, parsing & entity validation ─
    # detect_pii() encapsulates prompt construction (prompt_builder), Ollama call
    # (ollama_client), JSON extraction, and strict offset/span validation.
    detection_result = detect_pii(original_text)

    # Step 6: Reject invalid results
    if not detection_result.get("success", False):
        error_msg = detection_result.get("message", "Unknown detection error.")
        return {
            "success": False,
            "original_text": original_text,
            "detected_entities": [],
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "updated_global_state": current_state,
            "message": f"Sanitization aborted: PII detection failed ({error_msg}).",
        }

    detected_entities = detection_result.get("entities", [])

    # Fast path: No entities detected -> original text contains no PII
    if len(detected_entities) == 0:
        return {
            "success": True,
            "original_text": original_text,
            "detected_entities": [],
            "sanitized_text": original_text,
            "mapping": {},
            "tokens": [],
            "updated_global_state": current_state,
            "message": "Text analyzed successfully. No sensitive information detected.",
        }

    # ── Steps 7–8: Global tokenization & mapping state update ─────────────────
    tokenization_result = tokenize_with_global_mapping(
        original_text,
        detected_entities,
        current_state,
    )

    if not tokenization_result.get("success", False):
        error_msg = tokenization_result.get("message", "Unknown tokenization error.")
        return {
            "success": False,
            "original_text": original_text,
            "detected_entities": detected_entities,
            "sanitized_text": None,
            "mapping": {},
            "tokens": [],
            "updated_global_state": current_state,
            "message": f"Sanitization aborted: Tokenization failed ({error_msg}).",
        }

    updated_global_state = tokenization_result["updated_global_state"]
    sanitized_text = tokenization_result["sanitized_text"]

    # ── Step 9: Assemble structured output ─────────────────────────────────────
    # Extract local tokens allocated for this specific input
    local_mapping: dict[str, str] = {}
    tokens_metadata: list[dict[str, str]] = []

    for ent in detected_entities:
        raw_type = ent.get("type", "").strip().upper()
        sanitized_type = "".join(c if c.isalnum() else "_" for c in raw_type)
        val_key = f"{sanitized_type}||{ent.get('text', '')}"
        token = updated_global_state.get("val_to_token", {}).get(val_key)
        if token and token not in local_mapping:
            local_mapping[token] = ent["text"]
            tokens_metadata.append({
                "token": token,
                "type": sanitized_type,
                "original": ent["text"],
            })

    return {
        "success": True,
        "original_text": original_text,
        "detected_entities": detected_entities,
        "sanitized_text": sanitized_text,
        "mapping": local_mapping,
        "tokens": tokens_metadata,
        "updated_global_state": updated_global_state,
        "message": (
            f"Text successfully sanitized. "
            f"Replaced {len(detected_entities)} entity span(s) with {len(tokens_metadata)} unique token(s)."
        ),
    }


# Backwards-compatible alias for existing tests and callers
sanitize_text = run_sanitization_pipeline


# ── Restoration Pipeline ──────────────────────────────────────────────────────

def run_restoration_pipeline(
    ai_response: str,
    global_mapping: dict[str, str],
) -> dict[str, Any]:
    """
    Executes the controlled 6-step restoration workflow:
      1. Receive AI-generated response.
      2. Retrieve the existing global mapping.
      3. Restore all known placeholders.
      4. Preserve unknown placeholders.
      5. Report unmapped tokens.
      6. Return the restored response.

    PARAMETERS:
        ai_response (str): Text received from external AI containing tokens.
        global_mapping (dict): Token-to-original-value map.

    RETURNS:
        dict with keys:
            "success"           (bool)
            "restored_text"     (str | None)
            "unrestored_tokens" (list[str])
            "message"           (str)
    """
    # Step 1: Receive input & validate
    if not isinstance(ai_response, str):
        return {
            "success": False,
            "restored_text": None,
            "unrestored_tokens": [],
            "message": "AI response text must be a string.",
        }

    # Step 2: Retrieve and validate global mapping
    if not isinstance(global_mapping, dict):
        return {
            "success": False,
            "restored_text": None,
            "unrestored_tokens": [],
            "message": "global_mapping must be a dictionary.",
        }

    # Steps 3–5: Restore known placeholders, preserve unknown placeholders, report unmapped
    return restore_with_global_mapping(global_mapping, ai_response)


# Backwards-compatible alias
restore_text = run_restoration_pipeline
