"""Performance Benchmark for TrustBoost.

Measures latency and throughput across:
- Input text lengths: Small (~100 words), Medium (~500 words), Large (~1,000 words)
- Operations:
  - Detection (Gemma 3 4B local LLM inference)
  - Sanitization (deterministic tokenization & mapping state)
  - Restoration (deterministic placeholder replacement)
  - End-to-end pipeline execution
- Cold vs Warm start latency
- Statistical analysis: min, max, mean, median, standard deviation, throughput (words/sec, chars/sec)
"""

import sys
import json
import time
import statistics
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
from trustboost.pipeline.pipeline import (
    run_sanitization_pipeline,
    run_restoration_pipeline,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "performance_report.json"

# =============================================================================
# BENCHMARK TEXT FIXTURES
# =============================================================================

SMALL_TEXT = (
    "Customer Support Ticket #49102. Received an urgent request from Sarah Jenkins "
    "(email: sarah.jenkins@consulting.example.com, phone: +1-555-0144) regarding a pending refund "
    "for her corporate account. The customer reported that payment was debited from her credit card "
    "4111-2222-3333-4444 and credited to SBI account 30987654321 with IFSC code SBIN0001122. "
    "Her Aadhaar number 9876 5432 1098 and PAN ABCDE1234F have been verified by the compliance officer. "
    "Please escalate this case to supervisor David Miller at the Chicago branch immediately."
)

# Repeat and diversify to construct ~500 words
MEDIUM_PARAS = [
    SMALL_TEXT,
    (
        "Internal Technical Incident Report: Incident ID INC-88219. System administrator Alex Mercer "
        "reported unusual network traffic from IP address 192.168.10.45 accessing internal service "
        "https://auth.internal.corp.local/api/v1. The database connection string postgresql://admin:P@ssw0rd2026@localhost:5432/maindb "
        "was exposed in temporary debug logs on server prod-node-07. Developer Kevin Zhao regenerated "
        "the OAuth secret key sec_live_99f018a4bc82d0194e and invalidated token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID. "
        "All engineering personnel working on Project Aegis must review security policies before Monday."
    ),
    (
        "HR Onboarding Memo: New Senior Architect Elena Rostova will join the engineering department on 1st April 2026. "
        "Elena was born on 14/05/1988 and currently resides at 452 Elm Street, Springfield. Her official company "
        "email elena.rostova@acme.example.org and work mobile +1-555-019-2834 have been provisioned. "
        "Payroll requires her bank account number 50100234567890 under IFSC HDFC0000240 and tax ID BKZPR8812K. "
        "Please ensure her workstation access to bastion server 10.200.45.12 is approved by Department Head Michael Scott."
    ),
    (
        "Quarterly Financial Summary: Senior accountant Priya Sharma completed audit reconciliation for client Vikramaditya Singh. "
        "Transactions included wire transfer of 250,000 INR to Axis Bank account 112233445566 with branch IFSC UTIB0000150. "
        "Vendor invoice payments were authorized by Ananya Gupta under GSTIN 27AAAPL1234C1ZV. "
        "Client UPI payments sent to rahul.verma@okhdfcbank have been cleared without discrepancy."
    )
]
MEDIUM_TEXT = "\n\n".join(MEDIUM_PARAS)

# Repeat to construct ~1,000 words
LARGE_PARAS = MEDIUM_PARAS + [
    (
        "Security Audit Annex B: Infrastructure review conducted by security auditor Marcus Brody. "
        "Auditor identified unencrypted configuration files containing API key ghp_39aZkL84910fKmQ9021a8xZ4901lQp "
        "and Stripe test secret sk_test_51MzABCDEFGHIJKLMN1234567890. Cloud storage node k8s-worker-node-04a "
        "was quarantined. The root password Winter2026# on backup cluster srv-fin-core-09 was rotated. "
        "Confidential Project Titan documentation was relocated to secure storage at https://vault.internal.corp/secrets."
    ),
    (
        "Customer Migration Records: Client Kavita Krishnan (DOB: 1992-04-18) living at 404 Green Valley, Bengaluru, "
        "requested updating registered phone number to +91-9988776655. Her Aadhaar 4321 8765 2109 and PAN ALMPR3344Q "
        "were validated through UIDAI and NSDL portals. Account representative Suresh Menon approved the KYC update. "
        "Customer requested all future statements be dispatched to kavita.krishnan@fintech.example.in."
    ),
    (
        "Operations Dispatch Log: Field engineer Robert Langdon was dispatched to data center facility in Boston. "
        "Contact phone +1-202-555-0143 verified. Diagnostic telemetry from gateway host 172.16.254.1 indicated "
        "normal CPU load. Internal documentation available at http://staging-api.service.internal:8080/v1/health. "
        "Final sign-off provided by Chief Technology Officer Ian Malcolm residing at 12 Ocean Boulevard, Miami."
    ),
    (
        "Executive Summary: All sensitive financial margin targets and unannounced merger plans discussed during "
        "the executive board session remain classified under internal organization information rules. "
        "Access tokens ya29.a0AfH6SMD_exampleTokenValue123 and refresh token dckr_pat_83901aF923bCd0193E were archived. "
        "End of official system audit log."
    )
]
LARGE_TEXT = "\n\n".join(LARGE_PARAS)


def count_words(text: str) -> int:
    return len(text.split())


def calculate_stats(latencies: List[float], word_count: int, char_count: int) -> Dict[str, Any]:
    mean_val = statistics.mean(latencies)
    median_val = statistics.median(latencies)
    std_val = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    min_val = min(latencies)
    max_val = max(latencies)

    # Throughput based on mean latency
    words_per_sec = round(word_count / mean_val, 2) if mean_val > 0 else 0.0
    chars_per_sec = round(char_count / mean_val, 2) if mean_val > 0 else 0.0

    return {
        "iterations": len(latencies),
        "mean_seconds": round(mean_val, 4),
        "median_seconds": round(median_val, 4),
        "min_seconds": round(min_val, 4),
        "max_seconds": round(max_val, 4),
        "stdev_seconds": round(std_val, 4),
        "throughput": {
            "words_per_second": words_per_sec,
            "characters_per_second": chars_per_sec
        },
        "raw_latencies": [round(x, 4) for x in latencies]
    }


def run_benchmark(iterations: int = 3) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting TrustBoost Performance Benchmark...")
    print(f"Iterations per tier: {iterations}")

    tiers = [
        {"name": "Small Text (~100 words)", "text": SMALL_TEXT, "iterations": 2},
        {"name": "Medium Text (~500 words)", "text": MEDIUM_TEXT, "iterations": 1},
        {"name": "Large Text (~1,000 words)", "text": LARGE_TEXT, "iterations": 1}
    ]

    tier_results = {}

    # 1. Cold start measurement
    print("\n--- Measuring Cold Start Latency ---")
    t0_cold = time.perf_counter()
    cold_res = detect_pii("Ping test for cold start latency measurement.")
    cold_latency = time.perf_counter() - t0_cold
    print(f"Cold start inference latency: {cold_latency:.3f}s (success: {cold_res.get('success')})")

    # 2. Benchmarking across tiers
    for tier in tiers:
        name = tier["name"]
        text = tier["text"]
        tier_iters = tier.get("iterations", 1)
        w_cnt = count_words(text)
        c_cnt = len(text)
        print(f"\n--- Benchmarking {name} ({w_cnt} words, {c_cnt} chars, {tier_iters} iter(s)) ---")

        det_latencies = []
        san_latencies = []
        rest_latencies = []
        e2e_latencies = []

        for i in range(1, tier_iters + 1):
            print(f"Iteration {i}/{tier_iters}...")

            # End-to-end start
            t_e2e_0 = time.perf_counter()

            # Detection
            t_det_0 = time.perf_counter()
            det_res = detect_pii(text)
            t_det_1 = time.perf_counter()
            det_latency = t_det_1 - t_det_0
            det_latencies.append(det_latency)

            entities = det_res.get("entities", [])

            # Sanitization (deterministic)
            t_san_0 = time.perf_counter()
            state = clear_global_mapping()
            res_tok = tokenize_with_global_mapping(text, entities, state)
            t_san_1 = time.perf_counter()
            san_latency = t_san_1 - t_san_0
            san_latencies.append(san_latency)

            san_text = res_tok.get("sanitized_text", "")
            map_dict = res_tok.get("updated_global_state", {}).get("global_mapping", {})

            # Restoration (deterministic)
            t_rest_0 = time.perf_counter()
            res_rest = restore_with_global_mapping(map_dict, san_text)
            t_rest_1 = time.perf_counter()
            rest_latency = t_rest_1 - t_rest_0
            rest_latencies.append(rest_latency)

            # End-to-end total
            t_e2e_1 = time.perf_counter()
            e2e_latencies.append(t_e2e_1 - t_e2e_0)

            print(f"  Det: {det_latency:.3f}s | San: {san_latency * 1000:.2f}ms | Rest: {rest_latency * 1000:.2f}ms | E2E: {e2e_latencies[-1]:.3f}s | Ents: {len(entities)}")

        tier_results[name] = {
            "text_length": {
                "word_count": w_cnt,
                "char_count": c_cnt
            },
            "detection": calculate_stats(det_latencies, w_cnt, c_cnt),
            "sanitization": calculate_stats(san_latencies, w_cnt, c_cnt),
            "restoration": calculate_stats(rest_latencies, w_cnt, c_cnt),
            "end_to_end_pipeline": calculate_stats(e2e_latencies, w_cnt, c_cnt)
        }

    report = {
        "metadata": {
            "title": "TrustBoost Performance Benchmark Report",
            "model": "gemma3:4b (local Ollama)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "iterations_per_tier": iterations,
            "cold_start_seconds": round(cold_latency, 4)
        },
        "tiers": tier_results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPerformance Benchmark Complete! Report saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark(iterations=3)
