"""LangChain Agent Architecture Evaluation Suite for TrustBoost.

Evaluates:
1. Operation Selection & Routing: Routing between sanitize, restore, and error responses.
2. Tool Execution: Direct execution of individual LangChain tools (detection, validation, tokenization, restoration).
3. Delegation & Coordination: End-to-end execution through TrustBoostAgent.
4. Error Handling: Handling missing parameters, empty texts, and invalid operations.
5. Unauthorized Mutation Prevention: Verification that agent enforces validation invariants and state integrity.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustboost.agent.trustboost_agent import (
    TrustBoostAgent,
    get_agent,
    ALL_AGENT_TOOLS,
    sanitize_tool,
    restore_tool,
    pii_detection_tool,
    entity_validation_tool,
    global_tokenization_tool,
    restoration_tool,
)
from trustboost.session.global_mapping import clear_global_mapping

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "agent_report.json"


def test_operation_routing() -> Dict[str, Any]:
    """Test operation selection and routing logic of TrustBoostAgent."""
    agent = get_agent()
    cases = []

    # 1. Sanitize operation
    state_1 = clear_global_mapping()
    res_1 = agent.invoke({
        "operation": "sanitize",
        "text": "Hello, my name is John Doe.",
        "global_state": state_1
    })
    is_sanitize_ok = (
        res_1.get("success") is True
        and "sanitized_text" in res_1
        and ("entities" in res_1 or "detected_entities" in res_1)
    )
    cases.append({
        "name": "Sanitize Operation Routing",
        "passed": is_sanitize_ok,
        "result_keys": list(res_1.keys())
    })

    # 2. Restore operation
    state_2 = clear_global_mapping()
    state_2["global_mapping"]["[PERSON_1]"] = "John Doe"
    state_2["val_to_token"]["PERSON||John Doe"] = "[PERSON_1]"
    res_2 = agent.invoke({
        "operation": "restore",
        "text": "Greetings [PERSON_1], welcome!",
        "global_mapping": state_2["global_mapping"]
    })
    is_restore_ok = (
        res_2.get("success") is True
        and res_2.get("restored_text") == "Greetings John Doe, welcome!"
    )
    cases.append({
        "name": "Restore Operation Routing",
        "passed": is_restore_ok,
        "restored_text": res_2.get("restored_text")
    })

    # 3. Unknown operation handling
    res_3 = agent.invoke({
        "operation": "unsupported_action",
        "text": "some text",
        "global_state": clear_global_mapping()
    })
    is_unknown_handled = res_3.get("success") is False and "unknown operation" in res_3.get("error", "").lower()
    cases.append({
        "name": "Unknown Operation Handling",
        "passed": is_unknown_handled,
        "error_message": res_3.get("error")
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def test_tool_definitions_and_execution() -> Dict[str, Any]:
    """Test that all 6 LangChain tools are defined and invoke properly."""
    cases = []

    # 1. Check tools registry
    expected_tools = {
        "sanitize_text", "restore_text", "pii_detection", "entity_validation",
        "global_tokenization", "restoration"
    }
    registered_names = {t.name for t in ALL_AGENT_TOOLS}
    all_registered = expected_tools.issubset(registered_names)
    cases.append({
        "name": "Tool Registry Completeness",
        "passed": all_registered,
        "registered_tools": list(registered_names)
    })

    # 2. entity_validation_tool execution
    text = "Contact Alice at alice@example.com."
    raw_entities = [
        {"text": "Alice", "type": "PERSON", "start": 8, "end": 13},
        {"text": "Invalid", "type": "PERSON", "start": 0, "end": 10}  # Slice mismatch
    ]
    val_res = entity_validation_tool.invoke({"text": text, "entities": raw_entities})
    val_ok = (
        val_res.get("success") is True
        and len(val_res.get("entities", [])) == 1
        and val_res["entities"][0]["text"] == "Alice"
    )
    cases.append({
        "name": "Entity Validation Tool Execution",
        "passed": val_ok,
        "validated_count": len(val_res.get("entities", []))
    })

    # 3. global_tokenization_tool execution
    state = clear_global_mapping()
    valid_ents = [{"text": "Alice", "type": "PERSON", "start": 8, "end": 13}]
    tok_res = global_tokenization_tool.invoke({
        "text": text,
        "entities": valid_ents,
        "global_state": state
    })
    tok_ok = (
        tok_res.get("success") is True
        and "[PERSON_1]" in tok_res.get("sanitized_text", "")
        and len(tok_res.get("new_entities", [])) == 1
    )
    cases.append({
        "name": "Global Tokenization Tool Execution",
        "passed": tok_ok,
        "sanitized_text": tok_res.get("sanitized_text")
    })

    # 4. restoration_tool execution
    map_dict = tok_res.get("updated_global_state", {}).get("global_mapping", {})
    rest_res = restoration_tool.invoke({
        "ai_response": "Greetings [PERSON_1]!",
        "global_mapping": map_dict
    })
    rest_ok = (
        rest_res.get("success") is True
        and rest_res.get("restored_text") == "Greetings Alice!"
        and len(rest_res.get("unrestored_tokens", [])) == 0
    )
    cases.append({
        "name": "Restoration Tool Execution",
        "passed": rest_ok,
        "restored_text": rest_res.get("restored_text")
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def test_error_handling_and_guards() -> Dict[str, Any]:
    """Test agent error handling on invalid schemas and missing keys."""
    agent = get_agent()
    cases = []

    # 1. Missing text/input parameter (None)
    res_1 = agent.invoke({"operation": "sanitize", "text": None, "global_state": clear_global_mapping()})
    passed_1 = res_1.get("success") is False
    cases.append({
        "name": "None Text Guard in Sanitize",
        "passed": passed_1,
        "message": res_1.get("message")
    })

    # 2. Non-dict input to agent
    res_2 = agent.invoke("not a dictionary")
    passed_2 = res_2.get("success") is False and "must be a dictionary" in res_2.get("message", "").lower()
    cases.append({
        "name": "Non-Dict Agent Input Guard",
        "passed": passed_2,
        "message": res_2.get("message")
    })

    # 3. Empty text sanitization
    res_3 = agent.invoke({
        "operation": "sanitize",
        "text": "",
        "global_state": clear_global_mapping()
    })
    passed_3 = res_3.get("success") is False and "empty" in res_3.get("message", "").lower()
    cases.append({
        "name": "Empty Text Sanitization Guard",
        "passed": passed_3,
        "message": res_3.get("message")
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def test_unauthorized_mutation_prevention() -> Dict[str, Any]:
    """Test that agent prevents unauthorized mutation and preserves global state integrity."""
    agent = get_agent()
    cases = []

    state = clear_global_mapping()
    state["global_mapping"]["[PERSON_1]"] = "Alice"
    state["val_to_token"]["PERSON||Alice"] = "[PERSON_1]"
    state["type_counters"]["PERSON"] = 1

    # Attempt to restore text that contains no mapped tokens
    res_restore = agent.invoke({
        "operation": "restore",
        "text": "Hello [UNKNOWN_TOKEN_99] and [FAKE_1]!",
        "global_mapping": state["global_mapping"]
    })

    # Verify that unknown tokens are preserved and global_state was not mutated or corrupted
    mutation_prevented = (
        res_restore.get("restored_text") == "Hello [UNKNOWN_TOKEN_99] and [FAKE_1]!"
        and len(state["global_mapping"]) == 1
        and state["global_mapping"]["[PERSON_1]"] == "Alice"
        and "[UNKNOWN_TOKEN_99]" not in state["global_mapping"]
        and "[UNKNOWN_TOKEN_99]" in res_restore.get("unrestored_tokens", [])
    )
    cases.append({
        "name": "Preserve State on Unmapped Tokens",
        "passed": mutation_prevented
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def run_all_agent_tests() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Running LangChain Agent Architecture Evaluation Suite...")

    routing_res = test_operation_routing()
    tools_res = test_tool_definitions_and_execution()
    error_res = test_error_handling_and_guards()
    mutation_res = test_unauthorized_mutation_prevention()

    total_tests = routing_res["total"] + tools_res["total"] + error_res["total"] + mutation_res["total"]
    total_passed = routing_res["passed"] + tools_res["passed"] + error_res["passed"] + mutation_res["passed"]

    report = {
        "metadata": {
            "title": "TrustBoost LangChain Agent Evaluation Report",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "overall_status": "PASSED" if total_passed == total_tests else "FAILED",
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": round(total_passed / total_tests, 4) if total_tests > 0 else 1.0
        },
        "suites": {
            "operation_routing": routing_res,
            "tool_definitions_and_execution": tools_res,
            "error_handling_and_guards": error_res,
            "unauthorized_mutation_prevention": mutation_res
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Agent Evaluation Complete: {total_passed}/{total_tests} tests passed ({report['metadata']['overall_pass_rate'] * 100:.1f}%)")
    print(f"Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_all_agent_tests()
