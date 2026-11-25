"""Evaluation runner for TrustBoost Contextual PII Detection.

Runs the local Ollama Gemma 3 4B detector on evaluation/dataset/dataset.json,
computes strict entity-level matching, micro and macro metrics, category-level
breakdowns, and detailed error taxonomy, saving all reports to evaluation/results/.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trustboost.detector.pii_detector import detect_pii
from evaluation.detection.metrics import (
    compute_comprehensive_metrics,
    compute_privacy_metrics,
    export_category_metrics_csv,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "dataset.json"
CACHE_PATH = RESULTS_DIR / "cached_predictions.json"


def _code_fingerprint() -> str:
    """
    A short code that changes whenever the detector code, prompt, model or
    settings change. Cached predictions are only reused if this matches, so
    you can never accidentally report results from an older version.
    """
    h = hashlib.sha256()
    files = sorted((PROJECT_ROOT / "trustboost" / "detector").glob("*.py"))
    files.append(PROJECT_ROOT / "trustboost" / "agent" / "ollama_client.py")
    for path in files:
        h.update(path.read_bytes())
    h.update(os.getenv("OLLAMA_MODEL", "gemma3:4b").encode())
    h.update(os.getenv("TRUSTBOOST_PATTERNS", "1").encode())
    return h.hexdigest()[:16]


CODE_FINGERPRINT = _code_fingerprint()


def _cache_key(text: str) -> str:
    return hashlib.sha256(f"{CODE_FINGERPRINT}|{text}".encode("utf-8")).hexdigest()


def in_split(sample_id: str, split: str) -> bool:
    """
    dev  = 40% of samples: use these while improving the prompt/code.
    test = the other 60%: only look at these for the FINAL numbers you report.
    """
    if split == "all":
        return True
    number = int(sample_id.split("_")[1])
    is_dev = number % 5 in (1, 3)
    return is_dev if split == "dev" else not is_dev


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(limit: int = None, use_cache: bool = True, split: str = "all") -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = [s for s in load_dataset(DATASET_PATH) if in_split(s["id"], split)]
    print(f"Split: {split} | Code fingerprint: {CODE_FINGERPRINT}")

    if limit is not None and limit > 0:
        samples = samples[:limit]

    print(f"Starting PII Detection Evaluation on {len(samples)} samples...")
    print(f"Model: Gemma 3 4B via local Ollama")

    # Load cache if available
    cached_preds = {}
    if use_cache and CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached_preds = json.load(f)
            print(f"Loaded {len(cached_preds)} cached sample predictions from {CACHE_PATH}")
        except Exception as e:
            print(f"Notice: Could not load cache: {e}")

    evaluated_samples = []
    inference_times = []

    for idx, sample in enumerate(samples, 1):
        s_id = sample["id"]
        text = sample["text"]

        key = _cache_key(text)
        if key in cached_preds:
            pred_entities = cached_preds[key].get("predicted_entities", [])
            latency = cached_preds[key].get("latency_seconds", 0.0)
            status = "cached"
        else:
            t0 = time.perf_counter()
            succeeded = False
            try:
                res = detect_pii(text)
                latency = round(time.perf_counter() - t0, 3)
                succeeded = res.get("success", False)
                pred_entities = res.get("entities", []) if succeeded else []
                status = f"{latency:.2f}s" if succeeded else f"FAILED ({latency:.2f}s)"
            except Exception as e:
                latency = round(time.perf_counter() - t0, 3)
                print(f"Error on {s_id}: {e}")
                pred_entities = []
                status = f"err ({latency:.2f}s)"

            # Only cache successful runs, so a failure is retried next time.
            if succeeded:
                cached_preds[key] = {
                    "sample_id": s_id,
                    "predicted_entities": pred_entities,
                    "latency_seconds": latency,
                    "timestamp": time.time()
                }
            # Save incremental cache every 5 samples
            if idx % 5 == 0 or idx == len(samples):
                with open(CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cached_preds, f, indent=2, ensure_ascii=False)

        inference_times.append(latency)
        sample_copy = dict(sample)
        sample_copy["predicted_entities"] = pred_entities
        sample_copy["latency_seconds"] = latency
        evaluated_samples.append(sample_copy)

        print(f"[{idx:03d}/{len(samples):03d}] {s_id}: {len(sample.get('entities', []))} gold, {len(pred_entities)} pred ({status})")

    # Save final cache
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cached_preds, f, indent=2, ensure_ascii=False)

    print("\nComputing comprehensive metrics & error analysis...")
    metrics_result = compute_comprehensive_metrics(evaluated_samples)
    privacy = compute_privacy_metrics(evaluated_samples)
    metrics_result["summary"]["privacy_metrics"] = {k: v for k, v in privacy.items() if k != "leaked_entities"}

    # Add latency statistics
    if inference_times:
        metrics_result["summary"]["latency_stats"] = {
            "mean_seconds": round(sum(inference_times) / len(inference_times), 3),
            "min_seconds": round(min(inference_times), 3),
            "max_seconds": round(max(inference_times), 3),
            "total_inference_time_seconds": round(sum(inference_times), 2)
        }

    # 1. Output evaluation_report.json
    eval_report = {
        "metadata": {
            "evaluation_title": "TrustBoost Phase 3 & Pipeline Comprehensive Detection Evaluation",
            "model": "gemma3:4b (local Ollama)",
            "total_samples": len(evaluated_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "matching_mode": "strict_entity_level (exact type and character boundaries [start:end])"
        },
        "summary": metrics_result["summary"],
        "category_metrics": metrics_result["category_metrics"],
        "error_distribution": metrics_result["error_distribution"],
        "leaked_entities": privacy["leaked_entities"],
    }
    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation report to {report_path}")

    # 2. Output category_metrics.csv
    csv_path = RESULTS_DIR / "category_metrics.csv"
    export_category_metrics_csv(metrics_result["category_metrics"], str(csv_path))
    print(f"Saved category metrics CSV to {csv_path}")

    # 3. Output detection_error_analysis.json
    error_report = {
        "total_errors": len(metrics_result["detailed_errors"]),
        "error_distribution": metrics_result["error_distribution"],
        "detailed_errors": metrics_result["detailed_errors"]
    }
    error_path = RESULTS_DIR / "detection_error_analysis.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False)
    print(f"Saved error analysis to {error_path}")

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total Samples: {len(evaluated_samples)}")
    print(f"Gold Entities: {metrics_result['summary']['total_gold_entities']}")
    print(f"Predicted Entities: {metrics_result['summary']['total_pred_entities']}")
    print(f"Strict True Positives: {metrics_result['summary']['strict_true_positives']}")
    print(f"Strict False Positives: {metrics_result['summary']['strict_false_positives']}")
    print(f"Strict False Negatives: {metrics_result['summary']['strict_false_negatives']}")
    print(f"Micro Precision: {metrics_result['summary']['micro_metrics']['precision']}")
    print(f"Micro Recall:    {metrics_result['summary']['micro_metrics']['recall']}")
    print(f"Micro F1-Score:  {metrics_result['summary']['micro_metrics']['f1']}")
    print(f"Macro Precision: {metrics_result['summary']['macro_metrics']['precision']}")
    print(f"Macro Recall:    {metrics_result['summary']['macro_metrics']['recall']}")
    print(f"Macro F1-Score:  {metrics_result['summary']['macro_metrics']['f1']}")
    print(f"Error Distribution: {metrics_result['error_distribution']}")

    print("\n=== PRIVACY (the numbers that matter most) ===")
    print(f"PII characters masked (char recall): {privacy['char_recall']:.2%}")
    print(f"Entities with ANY character leaked:  {privacy['entities_with_any_leak']} / {privacy['total_gold_entities']}")
    print(f"Over-masking rate:                   {privacy['over_masking_rate']:.2%}")

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TrustBoost PII detection.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached predictions")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all",
                        help="dev = for improving; test = for final reported numbers")
    args = parser.parse_args()

    run_evaluation(limit=args.limit, use_cache=not args.no_cache, split=args.split)
