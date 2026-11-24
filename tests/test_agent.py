"""
tests/test_agent.py
-------------------
PURPOSE:
    Comprehensive unit tests for the central LangChain TrustBoost Agent
    (trustboost/agent/trustboost_agent.py).

COVERS:
    1. Agent initialization (default tools, LangChain Runnable compliance, custom LLM).
    2. Correct tool selection (explicit operation, synonyms, keyword routing, mock LLM routing).
    3. Complete sanitization pipeline through the agent.
    4. Complete restoration pipeline through the agent.
    5. Validation failures (fail-closed behavior, invalid offsets, boundary checks).
    6. Tool failures (Ollama offline / detection errors handled gracefully).
    7. Global mapping consistency through the agent (counters never reset, identical tokens reused).
    8. Unknown placeholders preserved and reported.
    9. Streamlit integration smoke flow via the agent.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from trustboost.agent.trustboost_agent import (
    TrustBoostAgent,
    get_agent,
    ALL_AGENT_TOOLS,
    sanitize_tool,
    restore_tool,
)
from trustboost.session import clear_global_mapping


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Agent Initialization
# ═════════════════════════════════════════════════════════════════════════════

class TestAgentInitialization:
    """Verifies agent construction, tool registration, and LangChain compliance."""

    def test_default_initialization(self):
        """Agent initializes with default tools and no LLM."""
        agent = TrustBoostAgent()
        assert agent is not None
        assert agent.llm is None
        assert len(agent.tools) == len(ALL_AGENT_TOOLS)

    def test_agent_is_runnable(self):
        """Agent must be an instance of LangChain Core Runnable."""
        agent = TrustBoostAgent()
        assert isinstance(agent, Runnable)
        assert hasattr(agent, "invoke")
        assert callable(agent.invoke)

    def test_tools_registered_properly(self):
        """All expected tools are registered as LangChain BaseTool instances."""
        agent = TrustBoostAgent()
        tool_names = set(agent.tools_by_name.keys())
        expected = {
            "sanitize_text",
            "restore_text",
            "pii_detection",
            "entity_validation",
            "global_tokenization",
            "restoration",
        }
        assert expected.issubset(tool_names)
        for t in agent.tools:
            assert isinstance(t, BaseTool)

    def test_custom_llm_initialization(self):
        """Agent can accept a custom or mocked LLM."""
        mock_llm = MagicMock()
        agent = TrustBoostAgent(llm=mock_llm)
        assert agent.llm is mock_llm

    def test_get_agent_factory(self):
        """get_agent factory helper returns a valid agent."""
        agent1 = get_agent()
        assert isinstance(agent1, TrustBoostAgent)
        agent2 = get_agent()
        assert agent1 is agent2  # singleton behavior


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Tool Selection Logic
# ═════════════════════════════════════════════════════════════════════════════

class TestToolSelection:
    """Verifies that the agent selects the right tool based on input or LLM."""

    def test_explicit_sanitize_operation(self):
        """operation='sanitize' selects sanitize_tool."""
        agent = TrustBoostAgent()
        t = agent.select_tool(operation="sanitize")
        assert t.name == "sanitize_text"

    def test_explicit_restore_operation(self):
        """operation='restore' selects restore_tool."""
        agent = TrustBoostAgent()
        t = agent.select_tool(operation="restore")
        assert t.name == "restore_text"

    def test_synonym_operations(self):
        """Synonyms like 'mask' or 'deanonymize' map to the correct tool."""
        agent = TrustBoostAgent()
        assert agent.select_tool(operation="mask").name == "sanitize_text"
        assert agent.select_tool(operation="anonymize").name == "sanitize_text"
        assert agent.select_tool(operation="deanonymize").name == "restore_text"
        assert agent.select_tool(operation="unmask").name == "restore_text"

    def test_invalid_operation_raises(self):
        """An unsupported operation raises ValueError."""
        agent = TrustBoostAgent()
        with pytest.raises(ValueError, match="Unknown operation"):
            agent.select_tool(operation="delete")

    def test_keyword_fallback_routing(self):
        """When no operation is provided, text keywords are used for tool selection."""
        agent = TrustBoostAgent()
        t_restore = agent.select_tool(input_text="Please restore this text for me")
        assert t_restore.name == "restore_text"

        t_sanitize = agent.select_tool(input_text="My email is alice@test.com")
        assert t_sanitize.name == "sanitize_text"

    def test_mock_llm_routing_restore(self):
        """When LLM is provided, its classification selects the tool."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "restore"
        agent = TrustBoostAgent(llm=mock_llm)

        t = agent.select_tool(input_text="Bring back original details")
        assert t.name == "restore_text"
        mock_llm.invoke.assert_called_once()

    def test_mock_llm_routing_sanitize(self):
        """When LLM returns 'sanitize', the agent selects sanitize_tool."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "sanitize"
        agent = TrustBoostAgent(llm=mock_llm)

        t = agent.select_tool(input_text="Protect these personal details")
        assert t.name == "sanitize_text"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Complete Sanitization Pipeline through Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestAgentSanitizationPipeline:
    """Verifies end-to-end sanitization executed through TrustBoostAgent."""

    def test_agent_sanitize_no_pii(self):
        """Text without PII returns success, original text, and empty mapping."""
        agent = TrustBoostAgent()
        mock_detection = {
            "success": True,
            "entities": [],
            "message": "No sensitive information detected.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = agent.sanitize("This is completely public documentation.")

        assert res["success"] is True
        assert res["sanitized_text"] == "This is completely public documentation."
        assert res["detected_entities"] == []
        assert res["mapping"] == {}

    def test_agent_sanitize_with_pii(self):
        """Valid detected PII is tokenized into global mapping through the agent."""
        agent = TrustBoostAgent()
        text = "Hello Rahul Sharma, reach me at rahul@example.com."
        mock_detection = {
            "success": True,
            "entities": [
                {"text": "Rahul Sharma", "type": "PERSON", "start": 6, "end": 18, "reason": "Name"},
                {"text": "rahul@example.com", "type": "EMAIL", "start": 32, "end": 49, "reason": "Email"},
            ],
            "message": "Found 2 entities.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = agent.sanitize(text, clear_global_mapping())

        assert res["success"] is True
        assert "[PERSON_1]" in res["sanitized_text"]
        assert "[EMAIL_1]" in res["sanitized_text"]
        assert "Rahul Sharma" in res["mapping"]["[PERSON_1]"]
        assert "updated_global_state" in res
        gm = res["updated_global_state"]["global_mapping"]
        assert gm["[PERSON_1]"] == "Rahul Sharma"
        assert gm["[EMAIL_1]"] == "rahul@example.com"

    def test_agent_invoke_dict_interface(self):
        """Invoking the agent with a dictionary input produces identical structured results."""
        agent = TrustBoostAgent()
        mock_detection = {
            "success": True,
            "entities": [],
            "message": "No entities.",
        }
        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            res = agent.invoke({
                "operation": "sanitize",
                "text": "Open data text",
            })
        assert res["success"] is True
        assert res["sanitized_text"] == "Open data text"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Complete Restoration Pipeline through Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestAgentRestorationPipeline:
    """Verifies end-to-end restoration executed through TrustBoostAgent."""

    def test_agent_restore_known_tokens(self):
        """Tokens present in the mapping are restored accurately."""
        agent = TrustBoostAgent()
        mapping = {
            "[PERSON_1]": "Aman Verma",
            "[EMAIL_1]": "aman@example.com",
        }
        ai_response = "Hello [PERSON_1], confirmation sent to [EMAIL_1]."
        res = agent.restore(ai_response, mapping)

        assert res["success"] is True
        assert res["restored_text"] == "Hello Aman Verma, confirmation sent to aman@example.com."
        assert res["unrestored_tokens"] == []

    def test_agent_round_trip_via_invoke(self):
        """Full round-trip: agent sanitizes text, then agent restores it back."""
        agent = TrustBoostAgent()
        original = "Contact Harshit at harshit@test.com."
        mock_detection = {
            "success": True,
            "entities": [
                {"text": "Harshit", "type": "PERSON", "start": 8, "end": 15, "reason": "Name"},
                {"text": "harshit@test.com", "type": "EMAIL", "start": 19, "end": 35, "reason": "Email"},
            ],
            "message": "Found 2 entities.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            san_res = agent.invoke({
                "operation": "sanitize",
                "text": original,
                "global_state": clear_global_mapping(),
            })

        assert san_res["success"] is True
        sanitized = san_res["sanitized_text"]
        gm = san_res["updated_global_state"]["global_mapping"]

        restore_res = agent.invoke({
            "operation": "restore",
            "text": sanitized,
            "global_mapping": gm,
        })

        assert restore_res["success"] is True
        assert restore_res["restored_text"] == original


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Validation Failures & Safety Checks
# ═════════════════════════════════════════════════════════════════════════════

class TestValidationFailures:
    """Verifies fail-closed behavior on corrupted entities or invalid boundaries."""

    def test_empty_input_fails_closed(self):
        """Empty text returns failure without crashing."""
        agent = TrustBoostAgent()
        res = agent.sanitize("")
        assert res["success"] is False
        assert res["sanitized_text"] is None

    def test_whitespace_input_fails_closed(self):
        """Whitespace text returns failure."""
        agent = TrustBoostAgent()
        res = agent.sanitize("   \t\n ")
        assert res["success"] is False

    def test_corrupted_offsets_fail_closed(self):
        """Offsets that do not match the text slice abort sanitization."""
        agent = TrustBoostAgent()
        corrupted_detection = {
            "success": True,
            "entities": [
                {"text": "Alice", "type": "PERSON", "start": 99, "end": 104, "reason": "Name"},
            ],
            "message": "Detected.",
        }

        state_before = clear_global_mapping()
        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=corrupted_detection):
            res = agent.sanitize("Hello Alice", state_before)

        assert res["success"] is False
        assert res["sanitized_text"] is None
        # State must not be corrupted
        assert state_before["global_mapping"] == {}

    def test_invalid_operation_returns_structured_failure(self):
        """Invoking agent with unknown operation returns failure dict, not an unhandled crash."""
        agent = TrustBoostAgent()
        res = agent.invoke({"operation": "unsupported_op", "text": "hello"})
        assert res["success"] is False
        assert "unknown operation" in res["message"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Tool Failures
# ═════════════════════════════════════════════════════════════════════════════

class TestToolFailures:
    """Verifies handling when underlying tools or Ollama encounters an error."""

    def test_detection_failure_propagates_gracefully(self):
        """If Ollama is down, agent returns success=False with clear diagnostic."""
        agent = TrustBoostAgent()
        mock_failure = {
            "success": False,
            "entities": [],
            "message": "Cannot connect to Ollama server at http://localhost:11434.",
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_failure):
            res = agent.sanitize("Sensitive message")

        assert res["success"] is False
        assert "detection failed" in res["message"].lower()

    def test_restore_invalid_types_handled(self):
        """Non-string AI response returns clean failure."""
        agent = TrustBoostAgent()
        res = agent.restore(12345, {})  # type: ignore
        assert res["success"] is False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Global Mapping Consistency through Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestGlobalMappingConsistencyThroughAgent:
    """Verifies sequential token numbering and value reuse across agent invocations."""

    def test_counters_increment_across_agent_invocations(self):
        """Input 1 gets PERSON_1, Input 2 gets PERSON_2."""
        agent = TrustBoostAgent()
        state = clear_global_mapping()

        # Input 1
        d1 = {
            "success": True,
            "entities": [{"text": "Alice", "type": "PERSON", "start": 0, "end": 5, "reason": "Name"}],
        }
        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=d1):
            r1 = agent.sanitize("Alice went home.", state)

        state = r1["updated_global_state"]
        assert state["global_mapping"]["[PERSON_1]"] == "Alice"

        # Input 2
        d2 = {
            "success": True,
            "entities": [{"text": "Bob", "type": "PERSON", "start": 0, "end": 3, "reason": "Name"}],
        }
        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=d2):
            r2 = agent.sanitize("Bob arrived.", state)

        state = r2["updated_global_state"]
        assert state["global_mapping"]["[PERSON_2]"] == "Bob"
        assert len(state["global_mapping"]) == 2

    def test_repeated_value_reuses_token(self):
        """Repeating Alice in a later input reuses [PERSON_1]."""
        agent = TrustBoostAgent()
        state = clear_global_mapping()

        d1 = {
            "success": True,
            "entities": [{"text": "Alice", "type": "PERSON", "start": 0, "end": 5, "reason": "Name"}],
        }
        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=d1):
            r1 = agent.sanitize("Alice went home.", state)

        state = r1["updated_global_state"]

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=d1):
            r2 = agent.sanitize("Alice returned.", state)

        state = r2["updated_global_state"]
        assert len(state["global_mapping"]) == 1
        assert state["global_mapping"]["[PERSON_1]"] == "Alice"
        assert "[PERSON_2]" not in state["global_mapping"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: Unknown Placeholders
# ═════════════════════════════════════════════════════════════════════════════

class TestUnknownPlaceholders:
    """Verifies that unknown placeholders are preserved verbatim and reported."""

    def test_unknown_placeholder_preserved_and_reported(self):
        agent = TrustBoostAgent()
        mapping = {"[PERSON_1]": "Alice"}
        ai_response = "Hello [PERSON_1] and [GHOST_99]."
        res = agent.restore(ai_response, mapping)

        assert res["success"] is True
        assert "Alice" in res["restored_text"]
        assert "[GHOST_99]" in res["restored_text"]
        assert "[GHOST_99]" in res["unrestored_tokens"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: Streamlit Integration Smoke Flow
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamlitIntegration:
    """Simulates app.py state interactions with the agent."""

    def test_app_session_state_sanitize_and_restore_cycle(self):
        """Simulates full Streamlit UI cycle through agent."""
        agent = get_agent()
        fake_session_state = {
            "global_state": clear_global_mapping(),
            "original_text": "",
            "sanitized_text": "",
            "detected_entities": [],
            "restoration_output": "",
        }

        input_text = "Doctor John Doe prescribed medication."
        mock_detection = {
            "success": True,
            "entities": [{"text": "John Doe", "type": "PERSON", "start": 7, "end": 15, "reason": "Doctor name"}],
        }

        # Tab 1: Sanitize
        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=mock_detection):
            result = agent.sanitize(input_text, fake_session_state["global_state"])

        assert result["success"] is True
        fake_session_state["global_state"] = result["updated_global_state"]
        fake_session_state["sanitized_text"] = result["sanitized_text"]

        # Tab 2: Restore
        ai_reply = "Patient visited [PERSON_1] today."
        restore_res = agent.restore(
            ai_reply,
            fake_session_state["global_state"]["global_mapping"],
        )

        assert restore_res["success"] is True
        assert restore_res["restored_text"] == "Patient visited John Doe today."
