"""Security and Privacy Evaluation Suite for TrustBoost.

Verifies:
1. Leakage Prevention: Sanitized text contains zero original PII values.
2. Restoration Fidelity: Original entities are restored accurately without corruption.
3. Unknown Placeholder Preservation: Unmapped placeholder tokens remain untouched.
4. Global Mapping Consistency: Identical values across multiple inputs reuse identical tokens.
5. Fail-Closed Resilience: Corrupted, adversarial, or truncated inputs fail safely without crashes or leaks.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

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
from trustboost.detector.pii_detector import (
    _extract_json_from_response,
    _validate_entity,
    _parse_and_validate_response,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "security_report.json"


def test_leakage_prevention() -> Dict[str, Any]:
    """Test that sanitized output does not leak any identified sensitive values."""
    test_cases = [
        {
            "name": "Personal identification",
            "text": "Please reach out to Alice Johnson at alice.johnson@example.org or phone +1-555-0199.",
            "entities": [
                {"text": "Alice Johnson", "type": "PERSON", "start": 20, "end": 33},
                {"text": "alice.johnson@example.org", "type": "EMAIL", "start": 37, "end": 62},
                {"text": "+1-555-0199", "type": "PHONE", "start": 72, "end": 83}
            ]
        },
        {
            "name": "Financial & Govt credentials",
            "text": "Customer PAN ABCDE1234F linked with Aadhaar 9999 8888 7777 and card 4111-2222-3333-4444.",
            "entities": [
                {"text": "ABCDE1234F", "type": "PAN", "start": 13, "end": 23},
                {"text": "9999 8888 7777", "type": "AADHAAR", "start": 44, "end": 58},
                {"text": "4111-2222-3333-4444", "type": "CREDIT_CARD", "start": 68, "end": 87}
            ]
        },
        {
            "name": "Security tokens and database secrets",
            "text": "Use password SuperSecret#2026 with token sk-live-9021840192ba for database uri postgresql://user:dbpass@host:5432.",
            "entities": [
                {"text": "SuperSecret#2026", "type": "PASSWORD", "start": 13, "end": 29},
                {"text": "sk-live-9021840192ba", "type": "SECRET", "start": 41, "end": 61},
                {"text": "postgresql://user:dbpass@host:5432", "type": "DATABASE_CREDENTIAL", "start": 79, "end": 113}
            ]
        }
    ]

    passed = 0
    details = []

    for case in test_cases:
        state = clear_global_mapping()
        res = tokenize_with_global_mapping(case["text"], case["entities"], state)
        sanitized_text = res.get("sanitized_text", "")

        leaked_entities = []
        for entity in case["entities"]:
            val = entity["text"]
            if val in sanitized_text:
                leaked_entities.append(val)

        is_safe = res.get("success") is True and len(leaked_entities) == 0
        if is_safe:
            passed += 1

        details.append({
            "test_case": case["name"],
            "original_text": case["text"],
            "sanitized_text": sanitized_text,
            "entities_checked": len(case["entities"]),
            "leaked_entities": leaked_entities,
            "passed": is_safe
        })

    return {
        "total": len(test_cases),
        "passed": passed,
        "failed": len(test_cases) - passed,
        "success_rate": round(passed / len(test_cases), 4),
        "details": details
    }


def test_restoration_fidelity() -> Dict[str, Any]:
    """Test that restoration accurately recovers original values without corruption."""
    test_cases = [
        {
            "name": "Exact Roundtrip Identity",
            "text": "Dr. Sarah Connor works in London and can be called on 555-0123.",
            "entities": [
                {"text": "Sarah Connor", "type": "PERSON", "start": 4, "end": 16},
                {"text": "London", "type": "LOCATION", "start": 26, "end": 32},
                {"text": "555-0123", "type": "PHONE", "start": 54, "end": 62}
            ]
        },
        {
            "name": "Simulated AI Response Roundtrip",
            "text": "Transfer 5000 INR from account 9876543210 to Priya Verma at IFSC HDFC0001234.",
            "entities": [
                {"text": "9876543210", "type": "BANK_ACCOUNT", "start": 31, "end": 41},
                {"text": "Priya Verma", "type": "PERSON", "start": 45, "end": 56},
                {"text": "HDFC0001234", "type": "IFSC", "start": 65, "end": 76}
            ],
            "ai_template": "Acknowledged. We have initiated the wire from [BANK_ACCOUNT_1] to recipient [PERSON_1] under IFSC [IFSC_1]."
        }
    ]

    passed = 0
    details = []

    for case in test_cases:
        state = clear_global_mapping()
        res_tok = tokenize_with_global_mapping(case["text"], case["entities"], state)
        sanitized_text = res_tok.get("sanitized_text", "")
        mapping = res_tok.get("updated_global_state", {}).get("global_mapping", {})

        if "ai_template" in case:
            text_to_restore = case["ai_template"]
        else:
            text_to_restore = sanitized_text

        res_rest = restore_with_global_mapping(mapping, text_to_restore)
        restored_text = res_rest.get("restored_text", "")

        # Check that original entities are in the restored text
        all_present = True
        missing = []
        for ent in case["entities"]:
            if ent["text"] not in restored_text:
                all_present = False
                missing.append(ent["text"])

        # If roundtrip exact identity
        if "ai_template" not in case:
            is_exact = restored_text == case["text"]
        else:
            is_exact = all_present and len(res_rest.get("unrestored_tokens", [])) == 0

        if is_exact:
            passed += 1

        details.append({
            "test_case": case["name"],
            "input_text": case["text"],
            "sanitized_text": sanitized_text,
            "restored_text": restored_text,
            "expected_all_entities_restored": all_present,
            "missing_entities": missing,
            "passed": is_exact
        })

    return {
        "total": len(test_cases),
        "passed": passed,
        "failed": len(test_cases) - passed,
        "success_rate": round(passed / len(test_cases), 4),
        "details": details
    }


def test_unknown_placeholder_preservation() -> Dict[str, Any]:
    """Test that placeholders not in the mapping are preserved without modification."""
    mapping = {"[PERSON_1]": "Alice Walker"}
    synthetic_ai_response = "Hello [PERSON_1], your email [EMAIL_99] and order token [ORDER_REF_10] are being processed."
    res = restore_with_global_mapping(mapping, synthetic_ai_response)
    restored = res.get("restored_text", "")
    unrestored = res.get("unrestored_tokens", [])

    expected = "Hello Alice Walker, your email [EMAIL_99] and order token [ORDER_REF_10] are being processed."
    preserves_unknown = (
        restored == expected
        and "[EMAIL_99]" in restored
        and "[ORDER_REF_10]" in restored
        and "[EMAIL_99]" in unrestored
        and "[ORDER_REF_10]" in unrestored
    )

    return {
        "total": 1,
        "passed": 1 if preserves_unknown else 0,
        "failed": 0 if preserves_unknown else 1,
        "success_rate": 1.0 if preserves_unknown else 0.0,
        "details": [
            {
                "input_response": synthetic_ai_response,
                "restored_output": restored,
                "expected_output": expected,
                "unrestored_tokens": unrestored,
                "passed": preserves_unknown
            }
        ]
    }


def test_global_mapping_consistency() -> Dict[str, Any]:
    """Test that identical PII values across multiple inputs receive identical tokens."""
    state = clear_global_mapping()

    input_1 = "User Alice Walker has requested a password reset."
    entities_1 = [{"text": "Alice Walker", "type": "PERSON", "start": 5, "end": 17}]

    input_2 = "Audit confirmed: Alice Walker logged in from IP 192.168.1.1."
    entities_2 = [
        {"text": "Alice Walker", "type": "PERSON", "start": 17, "end": 29},
        {"text": "192.168.1.1", "type": "IP_ADDRESS", "start": 48, "end": 59}
    ]

    input_3 = "Email sent to Alice Walker regarding project updates."
    entities_3 = [{"text": "Alice Walker", "type": "PERSON", "start": 14, "end": 26}]

    res_1 = tokenize_with_global_mapping(input_1, entities_1, state)
    san_1 = res_1.get("sanitized_text", "")
    state_1 = res_1.get("updated_global_state", {})

    res_2 = tokenize_with_global_mapping(input_2, entities_2, state_1)
    san_2 = res_2.get("sanitized_text", "")
    state_2 = res_2.get("updated_global_state", {})

    res_3 = tokenize_with_global_mapping(input_3, entities_3, state_2)
    san_3 = res_3.get("sanitized_text", "")
    state_3 = res_3.get("updated_global_state", {})

    consistent = (
        "[PERSON_1]" in san_1
        and "[PERSON_1]" in san_2
        and "[PERSON_1]" in san_3
        and "[PERSON_2]" not in san_2
        and "[PERSON_2]" not in san_3
        and state_3["type_counters"].get("PERSON") == 1
    )

    return {
        "total": 1,
        "passed": 1 if consistent else 0,
        "failed": 0 if consistent else 1,
        "success_rate": 1.0 if consistent else 0.0,
        "details": [
            {
                "shared_entity": "Alice Walker",
                "sanitized_1": san_1,
                "sanitized_2": san_2,
                "sanitized_3": san_3,
                "counter_value": state_3["type_counters"].get("PERSON"),
                "passed": consistent
            }
        ]
    }


def test_fail_closed_adversarial() -> Dict[str, Any]:
    """Test that malformed, corrupted, or hostile detector outputs fail closed safely."""
    cases = [
        {
            "name": "Corrupted non-JSON LLM output",
            "test_fn": lambda: _extract_json_from_response("I cannot fulfill this request as an AI assistant.") is None
        },
        {
            "name": "Truncated JSON",
            "test_fn": lambda: _extract_json_from_response('{"entities": [{"text": "Alice", "type": "PERSON", "start":') is None
        },
        {
            "name": "Negative offset injection",
            "test_fn": lambda: _validate_entity({"text": "Evil", "type": "PERSON", "start": -5, "end": 5}, "Sample text here") is None
        },
        {
            "name": "OutOfBounds offset injection",
            "test_fn": lambda: _validate_entity({"text": "Evil", "type": "PERSON", "start": 50, "end": 100}, "Short") is None
        },
        {
            "name": "Hallucinated Entity Type",
            "test_fn": lambda: _validate_entity({"text": "test", "type": "NON_EXISTENT_TYPE", "start": 0, "end": 4}, "test") is None
        },
        {
            "name": "Slice Text Mismatch",
            "test_fn": lambda: _validate_entity({"text": "Alice", "type": "PERSON", "start": 0, "end": 5}, "Bob Smith") is None
        }
    ]

    passed = 0
    details = []

    for c in cases:
        try:
            ok = c["test_fn"]()
        except Exception:
            ok = False

        if ok:
            passed += 1

        details.append({
            "name": c["name"],
            "passed": ok
        })

    return {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "success_rate": round(passed / len(cases), 4),
        "details": details
    }


def run_all_security_tests() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Running Security & Privacy Evaluation Suite...")

    leakage_res = test_leakage_prevention()
    restoration_res = test_restoration_fidelity()
    unknown_res = test_unknown_placeholder_preservation()
    consistency_res = test_global_mapping_consistency()
    fail_closed_res = test_fail_closed_adversarial()

    total_tests = (
        leakage_res["total"]
        + restoration_res["total"]
        + unknown_res["total"]
        + consistency_res["total"]
        + fail_closed_res["total"]
    )
    total_passed = (
        leakage_res["passed"]
        + restoration_res["passed"]
        + unknown_res["passed"]
        + consistency_res["passed"]
        + fail_closed_res["passed"]
    )

    report = {
        "metadata": {
            "title": "TrustBoost Security and Privacy Evaluation Report",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "overall_status": "PASSED" if total_passed == total_tests else "FAILED",
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": round(total_passed / total_tests, 4) if total_tests > 0 else 1.0
        },
        "suites": {
            "leakage_prevention": leakage_res,
            "restoration_fidelity": restoration_res,
            "unknown_placeholder_preservation": unknown_res,
            "global_mapping_consistency": consistency_res,
            "fail_closed_resilience": fail_closed_res
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Security Evaluation Complete: {total_passed}/{total_tests} tests passed ({report['metadata']['overall_pass_rate'] * 100:.1f}%)")
    print(f"Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_all_security_tests()
