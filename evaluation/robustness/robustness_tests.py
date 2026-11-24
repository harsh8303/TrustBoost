"""Robustness Evaluation Suite for TrustBoost.

Evaluates TrustBoost across 10 distinct text style categories:
1. Standard English: Well-formatted, grammatical English prose.
2. Hinglish / Code-switching: Colloquial Hindi-English transliterated text.
3. Informal / Slang / Typos: Social media/chat style with shorthand, missing punctuation, typos.
4. Complex Formatting: Markdown tables, JSON dumps, log lines, code blocks.
5. Dense Multi-Entity: Highly concentrated PII spans across multiple distinct categories.
6. Negative Samples / No-PII: False-positive resistance on clean non-sensitive technical/encyclopedic text.
7. Ambiguous Contexts: Words with dual meanings (names as common words, tech terms as entities).
8. Unusual Punctuation & Whitespace: Excessive whitespace, tab indents, nested brackets, punctuation wraps.
9. Mixed Casing: ALL-CAPS shout text, all-lowercase, erratic casing.
10. Long Text / Extended Paragraphs: Multi-sentence corporate narratives.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustboost.detector.pii_detector import detect_pii
from trustboost.session.global_mapping import (
    clear_global_mapping,
    tokenize_with_global_mapping,
    restore_with_global_mapping,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "robustness_report.json"

ROBUSTNESS_CASES = [
    {
        "category": "standard_english",
        "name": "Standard English Prose",
        "text": "Please dispatch the official contract to Attorney Arthur Pendelton at 742 Evergreen Terrace, Springfield, or email arthur.p@lawfirm.example.com.",
        "expected_has_pii": True
    },
    {
        "category": "hinglish_code_switching",
        "name": "Hinglish Transliterated Text",
        "text": "Bhai suno, mera phone number 9876543210 pe call mat karo, direct email bhej do rahul.sharma@startup.in par.",
        "expected_has_pii": True
    },
    {
        "category": "informal_chat_slang",
        "name": "Informal Chat with Typos",
        "text": "yo dude send 500 bucks to my upi rohit99@okhdfcbank asap thx!! admin pw is Winter2026# btw",
        "expected_has_pii": True
    },
    {
        "category": "complex_formatting",
        "name": "Markdown Table & Log Dump",
        "text": "| User | Email | Role |\n| --- | --- | --- |\n| Jane Doe | jane.doe@acme.com | Admin |\nDEBUG [2026-03-31] connection from 192.168.1.100",
        "expected_has_pii": True
    },
    {
        "category": "dense_multi_entity",
        "name": "Dense Multi-Entity Record",
        "text": "Client Vikramaditya Singh (PAN: ABCDE1234F, Aadhaar: 8877 6655 4433) transferred funds to account 50100987654321, IFSC: HDFC0000240 in New Delhi.",
        "expected_has_pii": True
    },
    {
        "category": "negative_no_pii",
        "name": "Negative Sample (No PII / Technical Docs)",
        "text": "Photosynthesis is the biological process used by plants and other organisms to convert light energy into chemical energy through cellular respiration.",
        "expected_has_pii": False
    },
    {
        "category": "ambiguous_contexts",
        "name": "Ambiguous Contexts (May, Rose, Paris)",
        "text": "May Parker loves the rose flower in Paris, but her sister April prefers the autumn weather in London.",
        "expected_has_pii": True
    },
    {
        "category": "unusual_punctuation_whitespace",
        "name": "Nested Brackets & Unusual Whitespace",
        "text": "Customer details:\t\t[[[Robert Vance]]]\t---\t<robert.vance@refrigeration.com>\t\t;;;\t((+1-555-0199))",
        "expected_has_pii": True
    },
    {
        "category": "mixed_casing",
        "name": "ALL-CAPS and Erratic Casing",
        "text": "ATTENTION: OFFICER JAMES T. KIRK REPORTED TO STARBASE SAN FRANCISCO WITH ACCESS CODE APEX_ALPHA_99182.",
        "expected_has_pii": True
    },
    {
        "category": "long_text_extended",
        "name": "Extended Multi-Sentence Narrative",
        "text": (
            "During the executive board briefing for Project Vanguard, Senior Director Eleanor Vance presented the quarterly targets. "
            "Eleanor requested all correspondence be routed to eleanor.vance@globalcorp.example.org. "
            "Her assistant Daniel Wu verified server health on k8s-worker-node-04a from IP address 10.0.15.22. "
            "All attendees agreed that unauthorized dissemination of internal financial forecasts violates corporate compliance."
        ),
        "expected_has_pii": True
    }
]


def run_robustness_evaluation() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting Robustness Evaluation across 10 text categories...")

    category_results = []
    total_passed = 0

    for idx, case in enumerate(ROBUSTNESS_CASES, 1):
        cat = case["category"]
        name = case["name"]
        text = case["text"]
        expected_pii = case["expected_has_pii"]

        print(f"[{idx:02d}/10] Testing: {name} ({cat})...")

        t0 = time.perf_counter()
        det_res = detect_pii(text)
        latency = round(time.perf_counter() - t0, 3)

        det_success = det_res.get("success", False)
        entities = det_res.get("entities", [])
        has_entities = len(entities) > 0

        # Sanitization & Restoration verification
        state = clear_global_mapping()
        res_tok = tokenize_with_global_mapping(text, entities, state)
        san_text = res_tok.get("sanitized_text", "")
        map_dict = res_tok.get("updated_global_state", {}).get("global_mapping", {})
        res_rest = restore_with_global_mapping(map_dict, san_text)
        rest_text = res_rest.get("restored_text", "")

        # Robustness Criteria:
        # 1. Detector must succeed without crash or failure.
        # 2. For Negative sample (No-PII), false positives should ideally be 0.
        # 3. For PII samples, detector should detect >= 1 entity.
        # 4. Sanitization and Restoration must complete cleanly and rest_text == text (or exact entity roundtrip).
        if not expected_pii:
            passed = det_success and (len(entities) == 0) and (san_text == text)
        else:
            passed = det_success and has_entities and (len(map_dict) > 0) and (rest_text == text)

        if passed:
            total_passed += 1

        category_results.append({
            "category": cat,
            "name": name,
            "text_length_chars": len(text),
            "expected_has_pii": expected_pii,
            "detector_success": det_success,
            "detected_entities_count": len(entities),
            "detected_entities": [{"text": e["text"], "type": e["type"]} for e in entities],
            "sanitized_text": san_text,
            "tokens_generated": len(map_dict),
            "restoration_exact_match": rest_text == text,
            "latency_seconds": latency,
            "passed": passed
        })

        print(f"       Result: {'PASS' if passed else 'FAIL'} | Ents: {len(entities)} | Latency: {latency:.2f}s")

    report = {
        "metadata": {
            "title": "TrustBoost Robustness Evaluation Report",
            "model": "gemma3:4b (local Ollama)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_categories_tested": len(ROBUSTNESS_CASES),
            "passed_categories": total_passed,
            "robustness_pass_rate": round(total_passed / len(ROBUSTNESS_CASES), 4)
        },
        "category_results": category_results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nRobustness Evaluation Complete! {total_passed}/{len(ROBUSTNESS_CASES)} passed.")
    print(f"Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_robustness_evaluation()
