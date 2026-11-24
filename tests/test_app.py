"""
tests/test_app.py
-----------------
PURPOSE:
    Tests for the Streamlit application module (app.py).

COVERS:
    1. Importability of app.py without executing the main UI workflow.
    2. Session state initialization logic.
    3. Cached Ollama status helper behavior.
"""

from unittest.mock import patch
import app


class TestAppModule:
    """Verifies that app.py complies with Streamlit entry-point conventions."""

    def test_app_has_main_entrypoint(self):
        """app.py must define a main() function callable by Streamlit."""
        assert hasattr(app, "main")
        assert callable(app.main)

    def test_app_has_status_helper(self):
        """app.py must define check_ollama_status."""
        assert hasattr(app, "check_ollama_status")
        assert callable(app.check_ollama_status)

    def test_app_has_session_state_init(self):
        """app.py must define init_session_state."""
        assert hasattr(app, "init_session_state")
        assert callable(app.init_session_state)

    def test_init_session_state_populates_keys(self):
        """init_session_state sets all expected keys in st.session_state."""
        fake_state = {}
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = fake_state
            app.init_session_state()

        # Core display / restore keys
        expected_keys = {
            "original_text",
            "sanitized_text",
            "detected_entities",
            "sanitization_message",
            "restoration_input",
            "restoration_output",
            "unrestored_tokens",
            # Backward-compat keys preserved for existing tests
            "token_mapping",
            "tokens_metadata",
            "session_timestamp",
        }
        assert expected_keys.issubset(fake_state.keys())

    def test_init_session_state_creates_global_state(self):
        """init_session_state must create the global_state dict."""
        fake_state = {}
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = fake_state
            app.init_session_state()

        assert "global_state" in fake_state
        gs = fake_state["global_state"]
        assert "global_mapping" in gs
        assert isinstance(gs["global_mapping"], dict)

    def test_check_ollama_status_handles_exception(self):
        """check_ollama_status must not crash if backend check raises an error."""
        with patch("app.get_full_status", side_effect=RuntimeError("Connection dropped")):
            res = app.check_ollama_status()
        assert res["ready"] is False
        assert res["server"]["connected"] is False

    def test_app_has_clear_sanitize_input_callback(self):
        """app.py must define clear_sanitize_input callback."""
        assert hasattr(app, "clear_sanitize_input")
        assert callable(app.clear_sanitize_input)

    def test_clear_sanitize_input_clears_only_input_widget(self):
        """clear_sanitize_input must clear sanitize_input_text while preserving all other state."""
        fake_state = {
            "sanitize_input_text": "Secret user text with Rahul Sharma",
            "original_text": "Secret user text with Rahul Sharma",
            "sanitized_text": "Secret user text with [PERSON_1]",
            "detected_entities": [{"text": "Rahul Sharma", "type": "PERSON"}],
            "global_state": {
                "global_mapping": {"[PERSON_1]": "Rahul Sharma"},
                "type_counters": {"PERSON": 1},
                "val_to_token": {"PERSON||Rahul Sharma": "[PERSON_1]"},
            },
            "restoration_input": "AI response containing [PERSON_1]",
            "restoration_output": "AI response containing Rahul Sharma",
            "unrestored_tokens": [],
        }

        with patch.object(app, "st") as mock_st:
            mock_st.session_state = fake_state
            app.clear_sanitize_input()

        # Input widget state is cleared
        assert fake_state["sanitize_input_text"] == ""

        # All output, entity, global mapping, and restoration states are strictly preserved
        assert fake_state["sanitized_text"] == "Secret user text with [PERSON_1]"
        assert fake_state["original_text"] == "Secret user text with Rahul Sharma"
        assert len(fake_state["detected_entities"]) == 1
        assert fake_state["global_state"]["global_mapping"] == {"[PERSON_1]": "Rahul Sharma"}
        assert fake_state["restoration_input"] == "AI response containing [PERSON_1]"
        assert fake_state["restoration_output"] == "AI response containing Rahul Sharma"

    def test_clear_input_followed_by_new_sanitization_flow(self):
        """After clearing input, entering new text and sanitizing works normally with global mapping accumulating."""
        fake_state = {
            "sanitize_input_text": "First text with Alice",
            "sanitized_text": "First text with [PERSON_1]",
            "detected_entities": [{"text": "Alice", "type": "PERSON"}],
            "global_state": {
                "global_mapping": {"[PERSON_1]": "Alice"},
                "type_counters": {"PERSON": 1},
                "val_to_token": {"PERSON||Alice": "[PERSON_1]"},
            },
            "restoration_input": "AI text",
            "restoration_output": "Restored text",
            "unrestored_tokens": [],
        }

        # User clicks Clear Input
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = fake_state
            app.clear_sanitize_input()

        assert fake_state["sanitize_input_text"] == ""
        assert fake_state["sanitized_text"] == "First text with [PERSON_1]"

        # User enters new text into the cleared text area
        fake_state["sanitize_input_text"] = "Second text with Bob"
        new_detection = {
            "success": True,
            "entities": [{"text": "Bob", "type": "PERSON", "start": 17, "end": 20, "reason": "Name"}],
        }

        with patch("trustboost.pipeline.pipeline.detect_pii", return_value=new_detection):
            agent = app.get_agent()
            res = agent.sanitize(fake_state["sanitize_input_text"], fake_state["global_state"])

        assert res["success"] is True
        assert "[PERSON_2]" in res["sanitized_text"]
        fake_state["global_state"] = res["updated_global_state"]
        fake_state["sanitized_text"] = res["sanitized_text"]

        assert fake_state["global_state"]["global_mapping"] == {
            "[PERSON_1]": "Alice",
            "[PERSON_2]": "Bob",
        }
        assert fake_state["sanitized_text"] == "Second text with [PERSON_2]"
