"""
trustboost/agent/trustboost_agent.py
------------------------------------
PURPOSE:
    Central LangChain Agent for TrustBoost.

ARCHITECTURE:
    Streamlit UI
        │
    TrustBoost LangChain Agent
        │
    Tool Selection and Execution
        ├── PII Detection (Gemma 3 4B via Ollama)
        ├── Entity Validation (Offset recovery & duplicate removal)
        ├── Global Tokenization (Cumulative counter preservation)
        └── Restoration (Single-pass replacement with unmapped tracking)
        │
    Global Mapping State (st.session_state["global_state"])
        │
    Final Output (Structured results for Streamlit)

RESPONSIBILITIES:
    - Receive user's requested operation.
    - Select sanitization or restoration.
    - Call the appropriate LangChain tools.
    - Coordinate existing deterministic processing components in pipeline.py.
    - Return structured results to Streamlit.
    - Enforce safety: fail closed, never bypass validation, never mutate mappings directly.
"""

from typing import Any
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool

from trustboost.pipeline.pipeline import (
    run_sanitization_pipeline,
    run_restoration_pipeline,
)
from trustboost.session.global_mapping import (
    clear_global_mapping,
    tokenize_with_global_mapping,
    restore_with_global_mapping,
)


# ── LangChain Tool Definitions ────────────────────────────────────────────────

@tool("pii_detection")
def pii_detection_tool(text: str) -> dict[str, Any]:
    """
    Detect Personally Identifiable Information (PII) contextually in text
    using local Gemma 3 4B via Ollama. Returns raw detected entities.
    """
    from trustboost.detector.pii_detector import detect_pii
    return detect_pii(text)


@tool("entity_validation")
def entity_validation_tool(text: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate that detected entity boundaries, character offsets, and text
    slices strictly match the original text. Rejects invalid or out-of-bounds spans.
    """
    if not isinstance(text, str) or not isinstance(entities, list):
        return {"success": False, "entities": [], "message": "Invalid input types."}

    valid: list[dict[str, Any]] = []
    text_len = len(text)
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        s, e = ent.get("start"), ent.get("end")
        ent_text = ent.get("text", "")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= text_len:
            if text[s:e] == ent_text:
                valid.append(ent)

    return {
        "success": True,
        "entities": valid,
        "message": f"Validated {len(valid)} of {len(entities)} entity span(s).",
    }


@tool("global_tokenization")
def global_tokenization_tool(
    text: str,
    entities: list[dict[str, Any]],
    global_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Replace validated entity spans with sequential anonymous tokens ([PERSON_1], etc.)
    using cumulative counters, updating the shared global mapping in memory.
    """
    state = global_state if global_state is not None else clear_global_mapping()
    return tokenize_with_global_mapping(text, entities, state)


@tool("restoration")
def restoration_tool(
    ai_response: str,
    global_mapping: dict[str, str],
) -> dict[str, Any]:
    """
    Restore original sensitive values from an AI response using the global mapping.
    Unknown placeholders are safely preserved and reported.
    """
    mapping = global_mapping if isinstance(global_mapping, dict) else {}
    return restore_with_global_mapping(mapping, ai_response)


@tool("sanitize_text")
def sanitize_tool(
    text: str,
    global_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute the complete deterministic sanitization pipeline:
    validates input, detects PII via Gemma 3 4B, verifies offsets, applies
    global tokenization, and updates the in-memory global mapping.
    """
    return run_sanitization_pipeline(text, global_state)


@tool("restore_text")
def restore_tool(
    ai_response: str,
    global_mapping: dict[str, str],
) -> dict[str, Any]:
    """
    Execute the complete restoration pipeline:
    restores known placeholders using global mapping while preserving unmapped tokens.
    """
    return run_restoration_pipeline(ai_response, global_mapping)


ALL_AGENT_TOOLS: list[BaseTool] = [
    sanitize_tool,
    restore_tool,
    pii_detection_tool,
    entity_validation_tool,
    global_tokenization_tool,
    restoration_tool,
]


# ── Central TrustBoost LangChain Agent ─────────────────────────────────────────

class TrustBoostAgent(Runnable[dict[str, Any], dict[str, Any]]):
    """
    Central LangChain AI Agent for TrustBoost.

    Coordinates tool selection and deterministic pipeline execution:
      - Inspects incoming requests from the UI or external callers.
      - Selects between sanitization and restoration operations.
      - Dispatches execution to LangChain tools that delegate to pipeline.py.
      - Guarantees fail-closed safety and strict offset verification.
    """

    def __init__(
        self,
        llm: Any = None,
        tools: list[BaseTool] | None = None,
    ) -> None:
        """
        Initializes the TrustBoost LangChain agent.

        PARAMETERS:
            llm: Optional LangChain-compatible LLM (e.g., ChatOllama or mock).
                 If provided, can be used for intelligent request classification.
            tools: Optional custom list of tools; defaults to ALL_AGENT_TOOLS.
        """
        self.llm = llm
        self.tools: list[BaseTool] = tools if tools is not None else list(ALL_AGENT_TOOLS)
        self.tools_by_name: dict[str, BaseTool] = {t.name: t for t in self.tools}

    def select_tool(
        self,
        operation: str | None = None,
        input_text: str | None = None,
    ) -> BaseTool:
        """
        Selects the appropriate LangChain tool based on explicit operation
        or natural-language request analysis.

        PARAMETERS:
            operation (str | None): Explicit operation name ("sanitize" or "restore").
            input_text (str | None): Raw input query or prompt text.

        RETURNS:
            BaseTool: The selected LangChain tool instance.

        RAISES:
            ValueError: If the operation is unknown or cannot be classified.
        """
        # 1. Explicit operation parameter
        if operation and isinstance(operation, str):
            op = operation.strip().lower()
            if op in ("sanitize", "sanitization", "mask", "anonymize"):
                return self.tools_by_name["sanitize_text"]
            if op in ("restore", "restoration", "deanonymize", "unmask"):
                return self.tools_by_name["restore_text"]
            raise ValueError(
                f"Unknown operation: '{operation}'. Valid operations are 'sanitize' and 'restore'."
            )

        # 2. LLM-based classification if an LLM is configured
        if self.llm is not None and input_text:
            try:
                prompt = (
                    "You are a routing agent. Determine whether the user wants to 'sanitize' "
                    "text (detect PII and replace with tokens) or 'restore' text "
                    "(restore tokens back to original values).\n"
                    f"User request: {input_text}\n"
                    "Respond with exactly one word: 'sanitize' or 'restore'."
                )
                if hasattr(self.llm, "invoke"):
                    raw_out = self.llm.invoke(prompt)
                    out_text = getattr(raw_out, "content", str(raw_out)).strip().lower()
                elif callable(self.llm):
                    raw_out = self.llm(prompt)
                    out_text = str(raw_out).strip().lower()
                else:
                    out_text = ""

                if "restore" in out_text:
                    return self.tools_by_name["restore_text"]
                if "sanitize" in out_text:
                    return self.tools_by_name["sanitize_text"]
            except Exception:
                # Fall back to heuristic classification on LLM error
                pass

        # 3. Deterministic keyword classification fallback
        if input_text and isinstance(input_text, str):
            lowered = input_text.lower()
            if any(w in lowered for w in ("restore", "deanonymize", "unmask", "revert")):
                return self.tools_by_name["restore_text"]
            return self.tools_by_name["sanitize_text"]

        raise ValueError("No operation or input text provided for tool selection.")

    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """
        LangChain Runnable invocation entrypoint.

        PARAMETERS:
            input (dict): Request dictionary containing:
                - "operation" (optional): "sanitize" or "restore"
                - "text" or "input": The text payload
                - "global_state" (for sanitize): Current global state dict
                - "global_mapping" (for restore): Current global mapping dict
            config (RunnableConfig | None): LangChain execution config.

        RETURNS:
            dict: Structured result from the executed pipeline tool.
        """
        if not isinstance(input, dict):
            return {
                "success": False,
                "message": "Input to TrustBoostAgent must be a dictionary.",
                "error": "Invalid input type.",
            }

        operation = input.get("operation")
        text = input.get("text")
        if text is None:
            text = input.get("input")
        if text is None:
            text = input.get("prompt", "")

        global_state = input.get("global_state")
        global_mapping = input.get("global_mapping")

        # Select tool
        try:
            chosen_tool = self.select_tool(operation=operation, input_text=text if isinstance(text, str) else "")
        except ValueError as err:
            return {
                "success": False,
                "message": str(err),
                "error": str(err),
            }

        # Execute chosen tool with appropriate arguments
        try:
            if chosen_tool.name == "sanitize_text":
                return chosen_tool.invoke({
                    "text": text,
                    "global_state": global_state,
                })
            elif chosen_tool.name == "restore_text":
                # Fallback: extract global_mapping from global_state if needed
                if global_mapping is None and isinstance(global_state, dict):
                    global_mapping = global_state.get("global_mapping", {})
                if global_mapping is None:
                    global_mapping = {}

                return chosen_tool.invoke({
                    "ai_response": text,
                    "global_mapping": global_mapping,
                })
            else:
                # Direct tool invocation
                return chosen_tool.invoke({"text": text})

        except Exception as exc:
            # Enforce fail-closed safety
            return {
                "success": False,
                "message": f"Agent tool execution failed: {exc}",
                "error": str(exc),
            }

    # ── Convenience Methods ───────────────────────────────────────────────────

    def sanitize(
        self,
        text: str,
        global_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience method to execute the sanitization pipeline."""
        return self.invoke({
            "operation": "sanitize",
            "text": text,
            "global_state": global_state,
        })

    def restore(
        self,
        ai_response: str,
        global_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Convenience method to execute the restoration pipeline."""
        return self.invoke({
            "operation": "restore",
            "text": ai_response,
            "global_mapping": global_mapping,
        })


# ── Global Singleton / Factory Helper ─────────────────────────────────────────

_agent_instance: TrustBoostAgent | None = None


def get_agent(llm: Any = None, force_new: bool = False) -> TrustBoostAgent:
    """
    Returns the central TrustBoost LangChain agent instance.
    """
    global _agent_instance
    if _agent_instance is None or force_new or llm is not None:
        _agent_instance = TrustBoostAgent(llm=llm)
    return _agent_instance
