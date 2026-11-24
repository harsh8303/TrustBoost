"""Master Evaluation Runner for TrustBoost.

Executes all evaluation modules in sequence:
1. PII Detection Evaluation & Metrics Calculation
2. Security & Privacy Suite
3. LangChain Agent Architecture Suite
4. Deterministic Pipeline Suite
5. Performance Benchmark
6. Robustness Suite across 10 Text Styles

Generates all required report artifacts in evaluation/results/.
"""

import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.detection.evaluate import run_evaluation as run_detection_eval
from evaluation.security.security_tests import run_all_security_tests
from evaluation.agent.agent_tests import run_all_agent_tests
from evaluation.pipeline.pipeline_tests import run_all_pipeline_tests
from evaluation.performance.benchmark import run_benchmark
from evaluation.robustness.robustness_tests import run_robustness_evaluation

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def run_full_suite(detection_limit: int = None, benchmark_iterations: int = 3, use_cache: bool = True):
    start_time = time.time()
    print("=" * 70)
    print("TRUSTBOOST COMPREHENSIVE EVALUATION SUITE")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 70)

    # 1. Pipeline Evaluation (fast, deterministic)
    print("\n[STAGE 1/6] Running Pipeline Evaluation...")
    pipeline_rep = run_all_pipeline_tests()

    # 2. Security Evaluation (fast, deterministic + fail closed)
    print("\n[STAGE 2/6] Running Security & Privacy Evaluation...")
    security_rep = run_all_security_tests()

    # 3. Agent Evaluation (LangChain routing & state)
    print("\n[STAGE 3/6] Running LangChain Agent Evaluation...")
    agent_rep = run_all_agent_tests()

    # 4. Robustness Evaluation (10 categories with local LLM)
    print("\n[STAGE 4/6] Running Robustness Evaluation...")
    robustness_rep = run_robustness_evaluation()

    # 5. Performance Benchmark (multi-tier latency & throughput)
    print("\n[STAGE 5/6] Running Performance Benchmark...")
    perf_rep = run_benchmark(iterations=benchmark_iterations)

    # 6. Detection Evaluation (100 synthetic samples with local LLM)
    print("\n[STAGE 6/6] Running Contextual PII Detection Evaluation...")
    detection_rep = run_detection_eval(limit=detection_limit, use_cache=use_cache)

    total_duration = time.time() - start_time
    print("\n" + "=" * 70)
    print("ALL EVALUATIONS COMPLETED SUCCESSFULLY")
    print(f"Total Suite Elapsed Time: {total_duration:.2f} seconds")
    print("=" * 70)
    print("\nGenerated Artifacts in evaluation/results/:")
    for f in sorted(RESULTS_DIR.glob("*")):
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all TrustBoost evaluations.")
    parser.add_argument("--detection-limit", type=int, default=None, help="Limit number of detection samples")
    parser.add_argument("--benchmark-iters", type=int, default=3, help="Benchmark iterations per tier")
    parser.add_argument("--no-cache", action="store_true", help="Disable prediction caching")
    args = parser.parse_args()

    run_full_suite(
        detection_limit=args.detection_limit,
        benchmark_iterations=args.benchmark_iters,
        use_cache=not args.no_cache
    )
