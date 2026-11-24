"""
tests/test_ollama_client.py
-----------------------------
PURPOSE:
    Tests for the trustboost/agent/ollama_client.py module.

    This file is split into two clear categories:

    ── UNIT TESTS ──────────────────────────────────────────────────────────────
    These tests do NOT need a running Ollama server.
    They use "mocking" — we fake the Ollama responses so the tests pass
    even when Ollama is offline. This lets you run them anytime, anywhere.

    ── INTEGRATION TESTS ───────────────────────────────────────────────────────
    These tests DO require a running Ollama server AND the gemma3:4b model.
    They test the real end-to-end connection between Python and the LLM.
    They are marked with @pytest.mark.integration so you can skip them
    when Ollama isn't available.

WHAT IS MOCKING?
    Mocking means replacing a real function with a fake one during testing.
    For example, instead of actually calling Ollama (which may be offline),
    we tell Python: "Pretend client.list() returns this fake data."
    This isolates the logic we're testing from external dependencies.

HOW TO RUN:
    # Run all unit tests (no Ollama required):
    pytest tests/test_ollama_client.py -v -m "not integration"

    # Run only integration tests (Ollama must be running):
    pytest tests/test_ollama_client.py -v -m integration

    # Run everything:
    pytest tests/test_ollama_client.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: UNIT TESTS (No Ollama server required)
# ═════════════════════════════════════════════════════════════════════════════

class TestGetConfig:
    """Tests for the get_config() function."""

    def test_config_returns_dict(self):
        """
        WHAT: Checks that get_config() returns a dictionary.
        WHY:  The Streamlit UI expects a dict. If a list or string is returned
              accidentally, the UI would break. This test catches that.
        """
        from trustboost.agent.ollama_client import get_config
        config = get_config()
        assert isinstance(config, dict), "get_config() must return a dict"

    def test_config_has_required_keys(self):
        """
        WHAT: Checks that the config dict contains 'model' and 'host' keys.
        WHY:  The app.py and ollama_client.py both depend on these exact keys.
              If a key is renamed or missing, we catch it here before it causes
              a silent runtime bug.
        """
        from trustboost.agent.ollama_client import get_config
        config = get_config()
        assert "model" in config, "Config must have a 'model' key"
        assert "host" in config, "Config must have a 'host' key"

    def test_config_values_are_strings(self):
        """
        WHAT: Checks that the config values are strings, not None or integers.
        WHY:  os.getenv() can return None if the variable is completely absent.
              We use defaults in ollama_client.py, but this test verifies they work.
        """
        from trustboost.agent.ollama_client import get_config
        config = get_config()
        assert isinstance(config["model"], str), "'model' must be a string"
        assert isinstance(config["host"], str), "'host' must be a string"

    def test_config_model_not_empty(self):
        """
        WHAT: Ensures the model name is not an empty string.
        WHY:  An empty model name would cause silent failures when sending prompts.
        """
        from trustboost.agent.ollama_client import get_config
        config = get_config()
        assert len(config["model"]) > 0, "'model' must not be empty"

    def test_config_host_starts_with_http(self):
        """
        WHAT: Verifies the host URL starts with 'http'.
        WHY:  Ollama uses an HTTP server. If the host is misconfigured (e.g., missing
              the protocol prefix), all connections will silently fail.
        """
        from trustboost.agent.ollama_client import get_config
        config = get_config()
        assert config["host"].startswith("http"), "Host must start with 'http'"


class TestCreateClient:
    """Tests for the create_client() function."""

    def test_create_client_returns_ollama_client(self):
        """
        WHAT: Verifies create_client() returns an ollama.Client object.
        WHY:  Every other function in this module uses the client object.
              If this function returns None or the wrong type, everything breaks.
        """
        import ollama
        from trustboost.agent.ollama_client import create_client
        client = create_client()
        assert isinstance(client, ollama.Client), "create_client() must return an ollama.Client"

    def test_create_client_does_not_raise(self):
        """
        WHAT: Checks that simply creating the client doesn't throw any error.
        WHY:  Client creation should not attempt a network connection. It should
              only configure the client. If it fails here, there's a config bug.
        """
        from trustboost.agent.ollama_client import create_client
        try:
            create_client()
        except Exception as e:
            pytest.fail(f"create_client() raised an unexpected exception: {e}")


class TestCheckServerConnectionMocked:
    """
    Unit tests for check_server_connection() using mocking.
    These tests do NOT require a running Ollama server.
    """

    def test_server_connected_when_list_succeeds(self):
        """
        WHAT: Mocks a successful client.list() call and checks that
              check_server_connection() reports "connected = True".
        WHY:  This tests our logic for the happy path — Ollama is running.
        HOW:  patch() replaces the real create_client with a fake one.
              The fake client's .list() method returns a dummy value.
        """
        from trustboost.agent import ollama_client

        fake_client = MagicMock()
        fake_client.list.return_value = MagicMock()  # Simulate a successful response

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.check_server_connection()

        assert result["connected"] is True
        assert "✅" in result["message"]

    def test_server_disconnected_when_list_raises(self):
        """
        WHAT: Mocks a failing client.list() call (server unreachable) and checks
              that check_server_connection() reports "connected = False".
        WHY:  This tests our error handling — the server is down. We want a
              clean False result, NOT an unhandled crash.
        """
        from trustboost.agent import ollama_client

        fake_client = MagicMock()
        fake_client.list.side_effect = ConnectionRefusedError("Connection refused")

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.check_server_connection()

        assert result["connected"] is False
        assert "❌" in result["message"]

    def test_server_status_has_message_key(self):
        """
        WHAT: Checks that check_server_connection() always returns a dict with a
              'message' key, even on failure.
        WHY:  The Streamlit UI reads result["message"] directly. If the key is
              missing, the UI crashes with a KeyError.
        """
        from trustboost.agent import ollama_client

        fake_client = MagicMock()
        fake_client.list.side_effect = Exception("Some random error")

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.check_server_connection()

        assert "message" in result
        assert "connected" in result


class TestCheckModelAvailabilityMocked:
    """
    Unit tests for check_model_availability() using mocking.
    """

    def test_model_found_when_present_in_list(self):
        """
        WHAT: Mocks a model list that contains 'gemma3:4b' and verifies
              check_model_availability() returns "available = True".
        WHY:  Confirms our model-matching logic works correctly.
        """
        from trustboost.agent import ollama_client

        # Build a fake model object that looks like what Ollama SDK returns
        fake_model = MagicMock()
        fake_model.model = "gemma3:4b"

        fake_list_response = MagicMock()
        fake_list_response.models = [fake_model]

        fake_client = MagicMock()
        fake_client.list.return_value = fake_list_response

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.check_model_availability()

        assert result["available"] is True
        assert "✅" in result["message"]

    def test_model_not_found_when_absent_from_list(self):
        """
        WHAT: Mocks a model list that does NOT contain 'gemma3:4b' and verifies
              check_model_availability() returns "available = False".
        WHY:  Confirms that a missing model is detected and reported clearly.
        """
        from trustboost.agent import ollama_client

        fake_model = MagicMock()
        fake_model.model = "llama3:8b"  # A different model, not the one we need

        fake_list_response = MagicMock()
        fake_list_response.models = [fake_model]

        fake_client = MagicMock()
        fake_client.list.return_value = fake_list_response

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.check_model_availability()

        assert result["available"] is False
        assert "⚠️" in result["message"]

    def test_model_unavailable_when_server_down(self):
        """
        WHAT: Mocks a server-down condition and checks that model availability
              correctly returns "available = False" with a helpful message.
        WHY:  When the server is down, we cannot check the model. The function
              should short-circuit and explain that Ollama must be started first.
        """
        from trustboost.agent import ollama_client

        fake_client = MagicMock()
        fake_client.list.side_effect = ConnectionRefusedError("Server is down")

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.check_model_availability()

        assert result["available"] is False


class TestGenerateResponseMocked:
    """
    Unit tests for generate_response() using mocking.
    """

    def test_empty_prompt_returns_failure(self):
        """
        WHAT: Calls generate_response() with an empty string.
        WHY:  An empty prompt would cause a useless or confusing LLM call.
              We handle it early and return a clear error.
        NOTE: This test does NOT need mocking — the check happens before Ollama is called.
        """
        from trustboost.agent.ollama_client import generate_response
        result = generate_response("")
        assert result["success"] is False
        assert "❌" in result["message"]

    def test_whitespace_only_prompt_returns_failure(self):
        """
        WHAT: Calls generate_response() with a whitespace-only string.
        WHY:  "   " is effectively empty. The .strip() check in our function
              should catch this. We verify it here.
        """
        from trustboost.agent.ollama_client import generate_response
        result = generate_response("   ")
        assert result["success"] is False

    def test_valid_prompt_returns_success_with_mocked_client(self):
        """
        WHAT: Mocks a successful Ollama response and verifies generate_response()
              returns "success = True" with the expected response text.
        WHY:  This validates our response extraction logic (response.message.content)
              without making a real network call.
        """
        from trustboost.agent import ollama_client

        # Build a fake ChatResponse that looks like what Ollama SDK returns
        fake_message = MagicMock()
        fake_message.content = "TrustBoost protects your personal data."

        fake_response = MagicMock()
        fake_response.message = fake_message

        fake_client = MagicMock()
        fake_client.chat.return_value = fake_response

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.generate_response("What is TrustBoost?")

        assert result["success"] is True
        assert result["response"] == "TrustBoost protects your personal data."
        assert "✅" in result["message"]

    def test_connection_error_returns_failure_dict(self):
        """
        WHAT: Mocks a network failure during generation and verifies that
              generate_response() does NOT crash — it returns a failure dict.
        WHY:  If Ollama goes down mid-session, the app must not crash. It must
              show a helpful error message instead.
        """
        from trustboost.agent import ollama_client

        fake_client = MagicMock()
        fake_client.chat.side_effect = ConnectionRefusedError("Server went down")

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.generate_response("Hello")

        assert result["success"] is False
        assert result["response"] == ""
        assert "❌" in result["message"]

    def test_response_dict_always_has_required_keys(self):
        """
        WHAT: Verifies generate_response() always returns a dict with
              'success', 'response', and 'message' keys — even on error.
        WHY:  The UI reads all three keys. Missing keys would cause a silent crash.
        """
        from trustboost.agent import ollama_client

        fake_client = MagicMock()
        fake_client.chat.side_effect = Exception("Random failure")

        with patch.object(ollama_client, "create_client", return_value=fake_client):
            result = ollama_client.generate_response("Hello")

        assert "success" in result
        assert "response" in result
        assert "message" in result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: INTEGRATION TESTS (Require a running Ollama server + gemma3:4b)
# ═════════════════════════════════════════════════════════════════════════════
# To run these: pytest tests/test_ollama_client.py -v -m integration

@pytest.mark.integration
class TestOllamaIntegration:
    """
    Integration tests that make REAL calls to a locally running Ollama server.

    PREREQUISITES:
        - Ollama must be running: ollama serve
        - Gemma 3 4B must be downloaded: ollama pull gemma3:4b
        - The trustboost conda environment must be active

    HOW TO RUN ONLY THESE:
        pytest tests/test_ollama_client.py -v -m integration
    """

    def test_real_server_connection(self):
        """
        WHAT: Makes a real connection check to the Ollama server.
        WHY:  Confirms end-to-end that Python can reach the Ollama server
              running on your Mac.
        """
        from trustboost.agent.ollama_client import check_server_connection
        result = check_server_connection()
        assert result["connected"] is True, (
            f"Could not connect to Ollama server. "
            f"Make sure 'ollama serve' is running.\n"
            f"Details: {result['message']}"
        )

    def test_real_model_availability(self):
        """
        WHAT: Checks that gemma3:4b is actually downloaded on this machine.
        WHY:  Confirms the full setup: server running + model downloaded.
        """
        from trustboost.agent.ollama_client import check_model_availability
        result = check_model_availability()
        assert result["available"] is True, (
            f"Model not found. Run: ollama pull gemma3:4b\n"
            f"Details: {result['message']}"
        )

    def test_real_prompt_generation(self):
        """
        WHAT: Sends a real test prompt to Gemma 3 4B and checks for a response.
        WHY:  This is the ultimate Phase 2 verification — Python sends a prompt,
              Gemma generates a response, Python receives it successfully.
        """
        from trustboost.agent.ollama_client import generate_response
        result = generate_response(
            "In one sentence, what is personally identifiable information?"
        )
        assert result["success"] is True, (
            f"Prompt generation failed.\nDetails: {result['message']}"
        )
        assert len(result["response"]) > 10, (
            "Response seems too short — the model may not have responded properly."
        )

    def test_real_full_status(self):
        """
        WHAT: Calls get_full_status() against a real Ollama server.
        WHY:  Verifies the combined status check used by the Streamlit UI works
              correctly in a real environment.
        """
        from trustboost.agent.ollama_client import get_full_status
        status = get_full_status()
        assert "server" in status
        assert "model" in status
        assert "config" in status
        assert "ready" in status
        assert status["ready"] is True, (
            "TrustBoost is not fully ready. Check server and model availability."
        )
