"""
trustboost/agent/ollama_client.py
-----------------------------------
PURPOSE:
    This module is the single point of contact between TrustBoost and the
    locally running Ollama server.

    Think of it as a "messenger" — it loads configuration, opens a connection
    to the Ollama server, and handles all send/receive communication.

RESPONSIBILITIES:
    1. Load Ollama configuration from environment variables (.env file).
    2. Create an Ollama client pointed at the configured host.
    3. Check whether the Ollama server is reachable.
    4. Check whether the required model (gemma3:4b) is downloaded.
    5. Send a text prompt to the model and return the response.
    6. Handle all errors (server down, model missing, timeout) gracefully.

WHAT THIS MODULE DOES NOT DO:
    - No PII detection logic (that is Phase 3).
    - No Streamlit UI code.
    - No tokenization or replacement logic.

AUTHOR NOTE:
    Every function in this file has a clear, single responsibility.
    This makes the code easy to test, debug, and explain during a viva.
"""

import os
import ollama
from dotenv import load_dotenv

# ── Load environment variables from the .env file ────────────────────────────
# load_dotenv() reads the .env file in your project root and makes its
# key=value pairs available via os.getenv().
# If the .env file doesn't exist, it falls back to system environment variables.
load_dotenv()


# ── Configuration Constants ───────────────────────────────────────────────────
# We read the model name and host from environment variables.
# If they are not set in .env, we use safe default values.
# This means the code never crashes just because .env is missing.

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# How long (seconds) to wait for the model before giving up, so the app never hangs forever.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: get_config
# ─────────────────────────────────────────────────────────────────────────────
def get_config() -> dict:
    """
    Returns the current Ollama configuration as a dictionary.

    WHY THIS FUNCTION EXISTS:
        Other parts of the app (like the Streamlit UI) may want to display
        the current model name or host address. Instead of importing raw
        constants, they call this function to get a clean config dictionary.

    RETURNS:
        dict with keys:
            "model"  → the model name (e.g., "gemma3:4b")
            "host"   → the Ollama server URL (e.g., "http://localhost:11434")
    """
    return {
        "model": OLLAMA_MODEL,
        "host": OLLAMA_HOST,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: create_client
# ─────────────────────────────────────────────────────────────────────────────
def create_client() -> ollama.Client:
    """
    Creates and returns an Ollama Python SDK client.

    WHY THIS FUNCTION EXISTS:
        The Ollama Python SDK can connect to Ollama in two ways:
          a) Implicitly — using default localhost settings.
          b) Explicitly — by passing a host parameter.

        We always use the EXPLICIT method because:
          - Our host comes from the .env file (configurable).
          - Explicit configuration is clearer, testable, and production-safe.

    HOW IT WORKS:
        ollama.Client(host=OLLAMA_HOST) creates a client object that knows
        where the Ollama server lives. All further calls (chat, list, etc.)
        go through this client to that specific host.

    RETURNS:
        An ollama.Client instance ready to send requests.
    """
    return ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: check_server_connection
# ─────────────────────────────────────────────────────────────────────────────
def check_server_connection() -> dict:
    """
    Checks whether the Ollama server is currently running and reachable.

    WHY THIS FUNCTION EXISTS:
        Before we do anything with Ollama (send prompts, check models),
        we need to know if the server is even ON.
        If the server is down, we show a helpful message rather than crashing.

    HOW IT WORKS:
        We call client.list() which asks the Ollama server to return a list
        of locally installed models. If that call succeeds, the server is up.
        If it raises a connection error, the server is not running.

        We specifically look for ConnectionError and similar exceptions
        (Ollama SDK raises these when the server is unreachable).

    RETURNS:
        dict with keys:
            "connected"  → True if server is reachable, False otherwise
            "message"    → Human-readable status message
    """
    try:
        client = create_client()
        # client.list() fetches the list of downloaded models.
        # If Ollama isn't running, this raises an exception immediately.
        client.list()
        return {
            "connected": True,
            "message": "✅ Ollama server is running and reachable.",
        }

    except Exception as e:
        # We catch all exceptions here because different OS/network conditions
        # produce different error types (ConnectionRefusedError, OSError, etc.)
        error_text = str(e)
        return {
            "connected": False,
            "message": (
                f"❌ Cannot connect to Ollama server at {OLLAMA_HOST}.\n"
                f"Make sure Ollama is running. Start it with: ollama serve\n"
                f"Technical detail: {error_text}"
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4: check_model_availability
# ─────────────────────────────────────────────────────────────────────────────
def check_model_availability() -> dict:
    """
    Checks whether the configured model (e.g., gemma3:4b) is downloaded
    and available on the local Ollama installation.

    WHY THIS FUNCTION EXISTS:
        Even if the Ollama server is running, the specific model we want to use
        might not be downloaded yet. These are two separate problems:
          - Server down → user needs to start Ollama.
          - Model missing → user needs to run: ollama pull gemma3:4b

        We handle them with separate messages so the user knows exactly what to do.

    HOW IT WORKS:
        1. We first check the server connection (reuses check_server_connection).
        2. If the server is up, we call client.list() to get all downloaded models.
        3. We loop through the model list and check if our configured model name
           appears in any of the model names.
        4. We use a flexible match (checking if our model name is contained in
           the full model name) because Ollama sometimes appends tags like
           "gemma3:4b" appearing as "gemma3:4b" or with extra metadata.

    RETURNS:
        dict with keys:
            "available"  → True if model is downloaded, False otherwise
            "message"    → Human-readable status message
    """
    # First, make sure the server is reachable
    server_status = check_server_connection()
    if not server_status["connected"]:
        return {
            "available": False,
            "message": (
                f"⚠️ Cannot check model availability — Ollama server is not running.\n"
                f"Start Ollama first, then try again."
            ),
        }

    try:
        client = create_client()
        # client.list() returns a ListResponse object.
        # Its .models attribute is a list of Model objects.
        # Each Model object has a .model attribute (the model name string).
        models_response = client.list()
        downloaded_models = models_response.models

        # Check whether our configured model is in the downloaded list.
        # We use a case-insensitive substring match for flexibility.
        model_found = any(
            OLLAMA_MODEL.lower() in m.model.lower()
            for m in downloaded_models
        )

        if model_found:
            return {
                "available": True,
                "message": f"✅ Model '{OLLAMA_MODEL}' is downloaded and ready to use.",
            }
        else:
            # Build a helpful list of what IS available
            available_names = [m.model for m in downloaded_models] if downloaded_models else []
            available_str = ", ".join(available_names) if available_names else "none"
            return {
                "available": False,
                "message": (
                    f"⚠️ Model '{OLLAMA_MODEL}' is NOT downloaded yet.\n"
                    f"Run this command to download it: ollama pull {OLLAMA_MODEL}\n"
                    f"Currently available models: {available_str}"
                ),
            }

    except Exception as e:
        return {
            "available": False,
            "message": f"❌ Error while checking model availability: {str(e)}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5: get_full_status
# ─────────────────────────────────────────────────────────────────────────────
def get_full_status() -> dict:
    """
    Returns a combined status report covering both the server and the model.

    WHY THIS FUNCTION EXISTS:
        The Streamlit UI needs to display two pieces of information:
          1. Is the Ollama server running?
          2. Is the required model downloaded?

        Instead of calling two functions separately in the UI code, we provide
        this single convenient function that bundles both checks together.

    RETURNS:
        dict with keys:
            "server"  → result from check_server_connection()
            "model"   → result from check_model_availability()
            "config"  → result from get_config()
            "ready"   → True only if BOTH server is up AND model is available
    """
    server_status = check_server_connection()
    model_status = check_model_availability()
    config = get_config()

    # TrustBoost is "fully ready" only when both conditions are satisfied
    fully_ready = server_status["connected"] and model_status["available"]

    return {
        "server": server_status,
        "model": model_status,
        "config": config,
        "ready": fully_ready,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 6: generate_response
# ─────────────────────────────────────────────────────────────────────────────
def generate_response(
    prompt: str,
    system: str | None = None,
    json_schema: dict | None = None,
) -> dict:
    """
    Sends a prompt to the configured Ollama model and returns the response.

    PARAMETERS:
        prompt (str): The user message to send.
        system (str | None): Optional fixed instructions sent as a separate
            "system" message (keeps our rules apart from the user's data).
        json_schema (dict | None): Optional JSON schema. When given, Ollama
            forces the model to reply with JSON matching this shape.

    SETTINGS USED (and why):
        temperature 0 -> same input gives the same answer every time
        seed 42       -> extra guarantee of repeatable results
        num_ctx       -> memory window big enough for prompt + text + answer
        num_predict   -> upper limit on answer length

    RETURNS:
        dict with keys "success", "response", "message", "done_reason"
    """
    if not prompt or not prompt.strip():
        return {
            "success": False,
            "response": "",
            "message": "❌ Prompt cannot be empty.",
            "done_reason": None,
        }

    try:
        client = create_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt.strip()})

        # Roughly 3 characters per token, plus room for the model's answer.
        total_chars = len(prompt) + len(system or "")
        num_ctx = min(16384, max(4096, total_chars // 3 + 2048))

        chat_kwargs = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": num_ctx,
                "num_predict": 2048,
            },
        }
        if json_schema is not None:
            chat_kwargs["format"] = json_schema

        response = client.chat(**chat_kwargs)
        response_text = response.message.content
        done_reason = getattr(response, "done_reason", None)

        # "length" means the model was cut off before finishing its answer.
        if done_reason == "length":
            return {
                "success": False,
                "response": response_text,
                "message": "❌ The model's answer was cut off because it was too long.",
                "done_reason": done_reason,
            }

        return {
            "success": True,
            "response": response_text,
            "message": "✅ Response received successfully.",
            "done_reason": done_reason,
        }

    except Exception as e:
        return {
            "success": False,
            "response": "",
            "message": f"❌ Failed to generate response: {str(e)}",
            "done_reason": None,
        }
