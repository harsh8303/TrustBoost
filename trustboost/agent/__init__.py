"""
trustboost.agent
----------------
LangChain Agent & Ollama client communication package for TrustBoost.
"""

from .ollama_client import (
    get_config,
    create_client,
    check_server_connection,
    check_model_availability,
    get_full_status,
    generate_response,
)

_agent_symbols = {
    "TrustBoostAgent",
    "get_agent",
    "ALL_AGENT_TOOLS",
    "sanitize_tool",
    "restore_tool",
    "pii_detection_tool",
    "entity_validation_tool",
    "global_tokenization_tool",
    "restoration_tool",
}

def __getattr__(name: str):
    if name in _agent_symbols:
        from . import trustboost_agent
        return getattr(trustboost_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_agent_symbols))

__all__ = [
    "TrustBoostAgent",
    "get_agent",
    "ALL_AGENT_TOOLS",
    "sanitize_tool",
    "restore_tool",
    "pii_detection_tool",
    "entity_validation_tool",
    "global_tokenization_tool",
    "restoration_tool",
    "get_config",
    "create_client",
    "check_server_connection",
    "check_model_availability",
    "get_full_status",
    "generate_response",
]
