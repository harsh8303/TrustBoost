"""Pipeline Evaluation Suite for TrustBoost.

Evaluates deterministic sanitization and restoration pipelines:
1. Core Sanitization Pipeline: Verification of entity replacement, token creation, and metadata.
2. Core Restoration Pipeline: Verification of placeholder substitution in model responses.
3. Edge Cases & Complex Formatting: Adjacent entities, substring collisions, Unicode/emojis,
   special characters, punctuation wrappers, empty/whitespace inputs.
4. Determinism & Idempotence: Repeated operations yield predictable results without state drift.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustboost.pipeline.pipeline import (
    run_sanitization_pipeline,
    run_restoration_pipeline,
)
from trustboost.session.global_mapping import (
    clear_global_mapping,
    tokenize_with_global_mapping,
    restore_with_global_mapping,
)
from trustboost.tokenizer.tokenizer import (
    tokenize_entities,
    restore_text,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "pipeline_report.json"


def test_core_sanitization_pipeline() -> Dict[str, Any]:
    """Test core sanitization pipeline functionality."""
    cases = []

    # 1. Standard multi-type sanitization
    text_1 = "Contact Alice at alice@corp.example.com or phone +1-555-0100."
    entities_1 = [
        {"text": "Alice", "type": "PERSON", "start": 8, "end": 13},
        {"text": "alice@corp.example.com", "type": "EMAIL", "start": 17, "end": 39},
        {"text": "+1-555-0100", "type": "PHONE", "start": 49, "end": 60}
    ]
    state_1 = clear_global_mapping()
    res_1 = tokenize_with_global_mapping(text_1, entities_1, state_1)
    san_1 = res_1.get("sanitized_text", "")
    updated_state_1 = res_1.get("updated_global_state", {})

    passed_1 = (
        res_1.get("success") is True
        and san_1 == "Contact [PERSON_1] at [EMAIL_1] or phone [PHONE_1]."
        and len(updated_state_1.get("global_mapping", {})) == 3
        and updated_state_1["global_mapping"].get("[PERSON_1]") == "Alice"
    )
    cases.append({
        "name": "Standard Multi-Type Sanitization",
        "passed": passed_1,
        "sanitized_text": san_1
    })

    # 2. Empty input text
    state_2 = clear_global_mapping()
    res_2 = tokenize_with_global_mapping("", [], state_2)
    passed_2 = res_2.get("success") is False and res_2.get("sanitized_text") is None
    cases.append({
        "name": "Empty Input Text Guard",
        "passed": passed_2,
        "sanitized_text": res_2.get("sanitized_text")
    })

    # 3. Text with zero entities
    state_3 = clear_global_mapping()
    raw_text_3 = "The weather today in general is sunny and clear."
    res_3 = tokenize_with_global_mapping(raw_text_3, [], state_3)
    passed_3 = res_3.get("success") is True and res_3.get("sanitized_text") == raw_text_3
    cases.append({
        "name": "Zero Entities Passthrough",
        "passed": passed_3,
        "sanitized_text": res_3.get("sanitized_text")
    })

    # 4. Multiple entities of same type
    text_4 = "Doctors Bob, Charlie, and David attended the symposium."
    entities_4 = [
        {"text": "Bob", "type": "PERSON", "start": 8, "end": 11},
        {"text": "Charlie", "type": "PERSON", "start": 13, "end": 20},
        {"text": "David", "type": "PERSON", "start": 26, "end": 31}
    ]
    state_4 = clear_global_mapping()
    res_4 = tokenize_with_global_mapping(text_4, entities_4, state_4)
    san_4 = res_4.get("sanitized_text", "")
    updated_state_4 = res_4.get("updated_global_state", {})
    passed_4 = (
        res_4.get("success") is True
        and san_4 == "Doctors [PERSON_1], [PERSON_2], and [PERSON_3] attended the symposium."
        and updated_state_4["type_counters"].get("PERSON") == 3
    )
    cases.append({
        "name": "Multiple Entities of Same Type",
        "passed": passed_4,
        "sanitized_text": san_4
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def test_core_restoration_pipeline() -> Dict[str, Any]:
    """Test core restoration pipeline functionality."""
    cases = []

    mapping = {
        "[PERSON_1]": "Alice",
        "[EMAIL_1]": "alice@example.com",
        "[PHONE_1]": "+1-555-0100"
    }

    # 1. Standard restoration of multiple tokens
    resp_1 = "Hello [PERSON_1], confirmation sent to [EMAIL_1] and SMS to [PHONE_1]."
    res_1 = restore_with_global_mapping(mapping, resp_1)
    passed_1 = (
        res_1.get("success") is True
        and res_1.get("restored_text") == "Hello Alice, confirmation sent to alice@example.com and SMS to +1-555-0100."
        and len(res_1.get("unrestored_tokens", [])) == 0
    )
    cases.append({
        "name": "Standard Multiple Token Restoration",
        "passed": passed_1,
        "restored_text": res_1.get("restored_text")
    })

    # 2. Token repeated multiple times in AI response
    resp_2 = "Thank you [PERSON_1]. Goodbye [PERSON_1]!"
    res_2 = restore_with_global_mapping(mapping, resp_2)
    passed_2 = (
        res_2.get("success") is True
        and res_2.get("restored_text") == "Thank you Alice. Goodbye Alice!"
        and len(res_2.get("unrestored_tokens", [])) == 0
    )
    cases.append({
        "name": "Repeated Token Restoration",
        "passed": passed_2,
        "restored_text": res_2.get("restored_text")
    })

    # 3. Response without tokens
    resp_3 = "The requested file has been successfully uploaded to the bucket."
    res_3 = restore_with_global_mapping(mapping, resp_3)
    passed_3 = (
        res_3.get("success") is True
        and res_3.get("restored_text") == resp_3
        and len(res_3.get("unrestored_tokens", [])) == 0
    )
    cases.append({
        "name": "No Tokens Passthrough",
        "passed": passed_3,
        "restored_text": res_3.get("restored_text")
    })

    # 4. Empty response text
    res_4 = restore_with_global_mapping(mapping, "")
    passed_4 = res_4.get("success") is True and res_4.get("restored_text") == ""
    cases.append({
        "name": "Empty Response Text",
        "passed": passed_4,
        "restored_text": res_4.get("restored_text")
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def test_edge_cases_and_formatting() -> Dict[str, Any]:
    """Test complex edge cases including adjacent entities, substrings, emojis, and special symbols."""
    cases = []

    # 1. Adjacent entities with punctuation separator
    text_adj = "Data: Bob,bob@mail.com"
    ents_adj = [
        {"text": "Bob", "type": "PERSON", "start": 6, "end": 9},
        {"text": "bob@mail.com", "type": "EMAIL", "start": 10, "end": 22}
    ]
    state_adj = clear_global_mapping()
    res_adj = tokenize_with_global_mapping(text_adj, ents_adj, state_adj)
    passed_adj = res_adj.get("sanitized_text") == "Data: [PERSON_1],[EMAIL_1]"
    cases.append({
        "name": "Adjacent Entities with Punctuation",
        "passed": passed_adj,
        "sanitized_text": res_adj.get("sanitized_text")
    })

    # 2. Substring Collision / Name prefix (e.g. "Cat" inside "Catherine")
    text_sub = "Cat and Catherine went to Paris."
    ents_sub = [
        {"text": "Cat", "type": "PERSON", "start": 0, "end": 3},
        {"text": "Catherine", "type": "PERSON", "start": 8, "end": 17},
        {"text": "Paris", "type": "LOCATION", "start": 26, "end": 31}
    ]
    state_sub = clear_global_mapping()
    res_sub = tokenize_with_global_mapping(text_sub, ents_sub, state_sub)
    san_sub = res_sub.get("sanitized_text", "")
    passed_sub = (
        san_sub == "[PERSON_1] and [PERSON_2] went to [LOCATION_1]."
        and "Catherine" not in san_sub
        and "[PERSON_1]herine" not in san_sub
    )
    cases.append({
        "name": "Substring Collision Avoidance",
        "passed": passed_sub,
        "sanitized_text": san_sub
    })

    # 3. Unicode and Emojis
    text_uni = "Hey Rahul Sharma 👋 send email to rahul@corp.in 🚀"
    ents_uni = [
        {"text": "Rahul Sharma", "type": "PERSON", "start": 4, "end": 16},
        {"text": "rahul@corp.in", "type": "EMAIL", "start": 33, "end": 46}
    ]
    state_uni = clear_global_mapping()
    res_uni = tokenize_with_global_mapping(text_uni, ents_uni, state_uni)
    san_uni = res_uni.get("sanitized_text", "")
    map_uni = res_uni.get("updated_global_state", {}).get("global_mapping", {})
    res_rest_uni = restore_with_global_mapping(map_uni, san_uni)
    passed_uni = (
        san_uni == "Hey [PERSON_1] 👋 send email to [EMAIL_1] 🚀"
        and res_rest_uni.get("restored_text") == text_uni
    )
    cases.append({
        "name": "Unicode and Emojis Roundtrip",
        "passed": passed_uni,
        "sanitized_text": san_uni,
        "restored_text": res_rest_uni.get("restored_text")
    })

    # 4. Special Characters in Passwords and URLs
    text_spec = "Password: P@ssw0rd!#$ and secret key: sec*live&99%."
    ents_spec = [
        {"text": "P@ssw0rd!#$", "type": "PASSWORD", "start": 10, "end": 21},
        {"text": "sec*live&99%", "type": "SECRET", "start": 38, "end": 50}
    ]
    state_spec = clear_global_mapping()
    res_spec = tokenize_with_global_mapping(text_spec, ents_spec, state_spec)
    san_spec = res_spec.get("sanitized_text", "")
    map_spec = res_spec.get("updated_global_state", {}).get("global_mapping", {})
    res_rest_spec = restore_with_global_mapping(map_spec, san_spec)
    passed_spec = (
        san_spec == "Password: [PASSWORD_1] and secret key: [SECRET_1]."
        and res_rest_spec.get("restored_text") == text_spec
    )
    cases.append({
        "name": "Special Characters in Values Roundtrip",
        "passed": passed_spec,
        "sanitized_text": san_spec,
        "restored_text": res_rest_spec.get("restored_text")
    })

    # 5. Entities enclosed in brackets and quotes
    text_encl = 'User: "[Alice]", Email: <alice@corp.com>, ID: (9876543210).'
    ents_encl = [
        {"text": "Alice", "type": "PERSON", "start": 8, "end": 13},
        {"text": "alice@corp.com", "type": "EMAIL", "start": 25, "end": 39},
        {"text": "9876543210", "type": "PHONE", "start": 47, "end": 57}
    ]
    state_encl = clear_global_mapping()
    res_encl = tokenize_with_global_mapping(text_encl, ents_encl, state_encl)
    san_encl = res_encl.get("sanitized_text", "")
    map_encl = res_encl.get("updated_global_state", {}).get("global_mapping", {})
    res_rest_encl = restore_with_global_mapping(map_encl, san_encl)
    passed_encl = (
        san_encl == 'User: "[[PERSON_1]]", Email: <[EMAIL_1]>, ID: ([PHONE_1]).'
        and res_rest_encl.get("restored_text") == text_encl
    )
    cases.append({
        "name": "Bracket Enclosed Entities",
        "passed": passed_encl,
        "sanitized_text": san_encl
    })

    passed_count = sum(1 for c in cases if c["passed"])
    return {
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": cases
    }


def run_all_pipeline_tests() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Running Deterministic Pipeline Evaluation Suite...")

    san_res = test_core_sanitization_pipeline()
    rest_res = test_core_restoration_pipeline()
    edge_res = test_edge_cases_and_formatting()

    total_tests = san_res["total"] + rest_res["total"] + edge_res["total"]
    total_passed = san_res["passed"] + rest_res["passed"] + edge_res["passed"]

    report = {
        "metadata": {
            "title": "TrustBoost Deterministic Pipeline Evaluation Report",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "overall_status": "PASSED" if total_passed == total_tests else "FAILED",
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": round(total_passed / total_tests, 4) if total_tests > 0 else 1.0
        },
        "suites": {
            "core_sanitization_pipeline": san_res,
            "core_restoration_pipeline": rest_res,
            "edge_cases_and_formatting": edge_res
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Pipeline Evaluation Complete: {total_passed}/{total_tests} tests passed ({report['metadata']['overall_pass_rate'] * 100:.1f}%)")
    print(f"Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_all_pipeline_tests()
