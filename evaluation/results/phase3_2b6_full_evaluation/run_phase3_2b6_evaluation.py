"""
evaluation/results/phase3_2b6_full_evaluation/run_phase3_2b6_evaluation.py
--------------------------------------------------------------------------
Phase 3.2B.6: Fresh Full 100-Sample Evaluation after JSON-Aware PERSON Detection
and Compact Variant B-Based Prompt Refinement.

Uses the exact same dataset and methodology as Phase 3.2B.5.
Does NOT modify production code, prompts, or historical evaluation reports.
Saves all outputs under evaluation/results/phase3_2b6_full_evaluation/.
"""

import json
import pathlib
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.detection.metrics import (
    compute_comprehensive_metrics,
    export_category_metrics_csv,
)
from trustboost.agent.ollama_client import OLLAMA_MODEL, create_client
from trustboost.detector.pii_detector import (
    _extract_json_from_response,
    _parse_and_validate_response,
)
from trustboost.detector.prompt_builder import build_detection_prompt
from trustboost.tokenizer.tokenizer import restore_text, tokenize_entities

DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "dataset.json"
OUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "phase3_2b6_full_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Verified Phase 3.2B.5 aggregate baseline (100 samples, 171 gold entities)
# Embedded directly so this runner is independently reproducible without Phase 3.2B.5 files.
PHASE3B5_BASELINE = {
    "summary": {
        "total_pred_entities": 176,
        "strict_true_positives": 145,
        "strict_false_positives": 31,
        "strict_false_negatives": 26,
        "micro_metrics": {
            "precision": 0.8239,
            "recall": 0.8480,
            "f1": 0.8357,
        },
        "macro_metrics": {
            "precision": 0.7378,
            "recall": 0.7884,
            "f1": 0.7540,
        },
        "latency_stats": {
            "mean_seconds": 3.995,
            "total_inference_time_seconds": 399.51,
        },
    },
    "leakage": {
        "unmasked_leaked_fns": 5,
        "entity_leakage_rate_percentage": 2.92,
        "affected_samples_count": 5,
        "sample_leakage_rate_percentage": 5.00,
    },
}


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_is_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True
        )
        return bool(out.strip())
    except Exception:
        return None


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    commit_sha = _git_head()
    git_dirty = _git_is_dirty()
    print("Starting Phase 3.2B.6 Full 100-Sample Evaluation (use_cache=False)...")
    print(
        f"Commit: {commit_sha} (dirty={git_dirty}) | Model: {OLLAMA_MODEL} | temperature=0 | format=omitted | num_ctx=4096"
    )

    client = create_client()
    evaluated_samples = []
    predictions_dump = {}
    inference_times = []
    prompt_tokens_list = []
    eval_tokens_list = []
    eval_durations_s = []

    detection_failures = []
    malformed_responses = []
    tok_verified = 0
    rest_verified = 0

    wall_start = time.perf_counter()

    for idx, sample in enumerate(samples, 1):
        s_id = sample["id"]
        text = sample["text"]
        prompt = build_detection_prompt(text)

        t0 = time.perf_counter()
        raw_text = ""
        raw_json_extracted = False
        parse_ok = False
        pred_entities = []
        error_msg = None
        prompt_tok = 0
        eval_tok = 0
        eval_dur_s = 0.0

        try:
            raw_resp = client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt.strip()}],
                options={"temperature": 0, "num_ctx": 4096},
            )
            raw_text = raw_resp.message.content if raw_resp.message else ""
            rd = raw_resp.__dict__ if hasattr(raw_resp, "__dict__") else {}
            prompt_tok = int(rd.get("prompt_eval_count") or 0)
            eval_tok = int(rd.get("eval_count") or 0)
            eval_dur_ns = rd.get("eval_duration") or 0
            eval_dur_s = round(eval_dur_ns / 1e9, 4)

            if not raw_text or not raw_text.strip():
                error_msg = "Empty LLM response"
                detection_failures.append({"sample_id": s_id, "reason": error_msg})
                malformed_responses.append(
                    {"sample_id": s_id, "reason": error_msg, "raw_text": raw_text}
                )
            else:
                extracted = _extract_json_from_response(raw_text)
                raw_json_extracted = extracted is not None
                if not raw_json_extracted:
                    malformed_responses.append(
                        {
                            "sample_id": s_id,
                            "reason": "JSON extraction failed",
                            "raw_text": raw_text[:300],
                        }
                    )
                parsed = _parse_and_validate_response(raw_text, text)
                parse_ok = bool(parsed.get("success", False))
                if parse_ok:
                    pred_entities = parsed.get("entities", [])
                else:
                    error_msg = parsed.get("message", "Validation failed")
                    detection_failures.append({"sample_id": s_id, "reason": error_msg})
        except Exception as exc:
            error_msg = str(exc)
            detection_failures.append({"sample_id": s_id, "reason": error_msg})

        latency = round(time.perf_counter() - t0, 3)
        inference_times.append(latency)
        prompt_tokens_list.append(prompt_tok)
        eval_tokens_list.append(eval_tok)
        eval_durations_s.append(eval_dur_s)

        sanitized_text = text
        mapping = {}
        tok_ok = False
        rest_ok = False
        if pred_entities:
            tok_res = tokenize_entities(text, pred_entities)
            tok_ok = bool(tok_res.get("success"))
            if tok_ok:
                sanitized_text = tok_res["sanitized_text"]
                mapping = tok_res["mapping"]
                rest_res = restore_text(sanitized_text, mapping)
                rest_ok = bool(rest_res.get("success")) and (
                    rest_res.get("restored_text") == text
                )
        else:
            tok_ok = True
            rest_ok = True

        if tok_ok:
            tok_verified += 1
        if rest_ok:
            rest_verified += 1

        predictions_dump[s_id] = {
            "predicted_entities": pred_entities,
            "latency_seconds": latency,
            "prompt_tokens": prompt_tok,
            "eval_tokens": eval_tok,
            "eval_duration_seconds": eval_dur_s,
            "raw_json_extracted": raw_json_extracted,
            "parse_success": parse_ok,
            "tokenization_ok": tok_ok,
            "exact_restoration_ok": rest_ok,
            "sanitized_text": sanitized_text,
            "error": error_msg,
        }

        sample_copy = dict(sample)
        sample_copy["predicted_entities"] = pred_entities
        sample_copy["latency_seconds"] = latency
        evaluated_samples.append(sample_copy)

        if idx % 10 == 0 or idx == len(samples):
            print(
                f"[{idx:03d}/{len(samples):03d}] {s_id}: gold={len(sample.get('entities', []))} "
                f"pred={len(pred_entities)} ({latency:.2f}s, tok={eval_tok})"
            )

    total_wall_s = round(time.perf_counter() - wall_start, 2)

    with open(OUT_DIR / "phase3_2b6_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_dump, f, indent=2, ensure_ascii=False)

    metrics_result = compute_comprehensive_metrics(evaluated_samples)
    summary = metrics_result["summary"]
    summary["latency_stats"] = {
        "mean_seconds": round(statistics.mean(inference_times), 3),
        "median_seconds": round(statistics.median(inference_times), 3),
        "min_seconds": round(min(inference_times), 3),
        "max_seconds": round(max(inference_times), 3),
        "total_inference_time_seconds": round(sum(inference_times), 2),
        "total_wall_seconds": total_wall_s,
    }
    summary["token_stats"] = {
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens_list), 2),
        "total_prompt_tokens": sum(prompt_tokens_list),
        "mean_eval_tokens": round(statistics.mean(eval_tokens_list), 2),
        "total_eval_tokens": sum(eval_tokens_list),
        "mean_eval_duration_seconds": round(statistics.mean(eval_durations_s), 3),
    }
    summary["reliability_stats"] = {
        "detection_failures_count": len(detection_failures),
        "detection_failures": detection_failures,
        "malformed_responses_count": len(malformed_responses),
        "malformed_responses": malformed_responses,
        "tokenization_verified": tok_verified,
        "exact_restoration_verified": rest_verified,
    }

    report_payload = {
        "metadata": {
            "evaluation_title": "TrustBoost Phase 3.2B.6 Full Evaluation (JSON-Aware PERSON + Compact Prompt Refinement)",
            "git_commit": commit_sha,
            "git_dirty": git_dirty,
            "model": f"{OLLAMA_MODEL} (local Ollama)",
            "configuration": {
                "temperature": 0,
                "format": "omitted",
                "num_ctx": 4096,
                "prompt_version": "Phase 3.2B.6D.3 compact Variant B-based prompt (trustboost/detector/prompt_builder.py)",
                "offset_recovery": "Phase 2A duplicate-safe offset recovery",
                "deterministic_uri_recognizer": "Phase 3.2A.3 _extract_database_uri_entities",
                "deterministic_secret_recognizer": "Phase 3.2B.3 _extract_structured_secret_entities",
                "deterministic_pii_recognizer": "Phase 3.2B.5 _extract_structured_pii_entities",
                "deterministic_json_person_recognizer": "Phase 3.2B.6C _extract_json_person_entities",
            },
            "total_samples": len(samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "matching_mode": "strict_entity_level (exact type and character boundaries [start:end])",
            "cache_used": False,
        },
        "summary": summary,
        "category_metrics": metrics_result["category_metrics"],
    }

    with open(OUT_DIR / "phase3_2b6_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    error_report = {
        "total_errors": len(metrics_result["detailed_errors"]),
        "error_distribution": metrics_result["error_distribution"],
        "detailed_errors": metrics_result["detailed_errors"],
    }
    with open(
        OUT_DIR / "phase3_2b6_detection_error_analysis.json", "w", encoding="utf-8"
    ) as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False)

    export_category_metrics_csv(
        metrics_result["category_metrics"],
        str(OUT_DIR / "phase3_2b6_category_metrics.csv"),
    )

    leaked_entities = []
    masked_protected_fns = 0
    unmasked_leaked_fns = 0
    affected_samples = set()
    spurious_fps = 0
    overlapping_fps = 0

    for sample in evaluated_samples:
        s_id = sample["id"]
        text = sample["text"]
        golds = sample.get("entities", [])
        preds = sample.get("predicted_entities", [])
        sanitized = predictions_dump[s_id]["sanitized_text"]

        gold_exact = {(g["type"], g["start"], g["end"]) for g in golds}
        pred_exact = {(p["type"], p["start"], p["end"]) for p in preds}
        for p in preds:
            key = (p["type"], p["start"], p["end"])
            if key not in gold_exact:
                has_overlap = any(
                    _spans_overlap(p["start"], p["end"], g["start"], g["end"])
                    for g in golds
                )
                if has_overlap:
                    overlapping_fps += 1
                else:
                    spurious_fps += 1

        for g in golds:
            key = (g["type"], g["start"], g["end"])
            if key not in pred_exact:
                if g["text"] in sanitized:
                    unmasked_leaked_fns += 1
                    affected_samples.add(s_id)
                    leaked_entities.append(
                        {
                            "sample_id": s_id,
                            "entity_text": g["text"],
                            "entity_type": g["type"],
                            "offsets": [g["start"], g["end"]],
                            "sample_text": text,
                            "sanitized_text": sanitized,
                        }
                    )
                else:
                    masked_protected_fns += 1

    total_gold = summary["total_gold_entities"]
    leakage_payload = {
        "summary": {
            "strict_tp": summary["strict_true_positives"],
            "strict_fp": summary["strict_false_positives"],
            "strict_fn": summary["strict_false_negatives"],
            "masked_protected_fns": masked_protected_fns,
            "unmasked_leaked_fns": unmasked_leaked_fns,
            "substring_leaked_entities": unmasked_leaked_fns,
            "entity_leakage_rate_percentage": round(
                (unmasked_leaked_fns / total_gold) * 100, 2
            ),
            "affected_samples": sorted(affected_samples),
            "sample_leakage_rate_percentage": round(
                (len(affected_samples) / len(samples)) * 100, 2
            ),
            "spurious_fps": spurious_fps,
            "overlapping_fps": overlapping_fps,
        },
        "leaked_entities": leaked_entities,
    }
    with open(
        OUT_DIR / "phase3_2b6_leakage_and_reconciliation.json", "w", encoding="utf-8"
    ) as f:
        json.dump(leakage_payload, f, indent=2, ensure_ascii=False)

    p5s = PHASE3B5_BASELINE["summary"]
    p5ls = PHASE3B5_BASELINE["leakage"]
    cur, cur_l = summary, leakage_payload["summary"]

    lines = [
        "# TrustBoost — Phase 3.2B.6 Full 100-Sample Evaluation & Comparison vs Phase 3.2B.5",
        "",
        f"**Commit**: `{commit_sha}` (`dirty={git_dirty}`) | **Model**: `{OLLAMA_MODEL}` (`temperature=0`, `format` omitted, `num_ctx=4096`)  ",
        f"**Samples**: {len(samples)} ({total_gold} gold entities) | **Fresh Inference**: `use_cache=False`  ",
        f"**Failures / Malformed**: {len(detection_failures)} / {len(malformed_responses)} | **Tok / Restore**: {tok_verified}/100 / {rest_verified}/100",
        "",
        "| Metric | Phase 3.2B.5 | Phase 3.2B.6 | Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| Predicted Entities | {p5s['total_pred_entities']} | **{cur['total_pred_entities']}** | {cur['total_pred_entities']-p5s['total_pred_entities']:+d} |",
        f"| Strict TP | {p5s['strict_true_positives']} | **{cur['strict_true_positives']}** | {cur['strict_true_positives']-p5s['strict_true_positives']:+d} |",
        f"| Strict FP | {p5s['strict_false_positives']} | **{cur['strict_false_positives']}** | {cur['strict_false_positives']-p5s['strict_false_positives']:+d} |",
        f"| Strict FN | {p5s['strict_false_negatives']} | **{cur['strict_false_negatives']}** | {cur['strict_false_negatives']-p5s['strict_false_negatives']:+d} |",
        f"| Micro Precision | {p5s['micro_metrics']['precision']*100:.2f}% | **{cur['micro_metrics']['precision']*100:.2f}%** | {(cur['micro_metrics']['precision']-p5s['micro_metrics']['precision'])*100:+.2f} pp |",
        f"| Micro Recall | {p5s['micro_metrics']['recall']*100:.2f}% | **{cur['micro_metrics']['recall']*100:.2f}%** | {(cur['micro_metrics']['recall']-p5s['micro_metrics']['recall'])*100:+.2f} pp |",
        f"| Micro F1 | {p5s['micro_metrics']['f1']*100:.2f}% | **{cur['micro_metrics']['f1']*100:.2f}%** | {(cur['micro_metrics']['f1']-p5s['micro_metrics']['f1'])*100:+.2f} pp |",
        f"| Macro Precision | {p5s['macro_metrics']['precision']*100:.2f}% | **{cur['macro_metrics']['precision']*100:.2f}%** | {(cur['macro_metrics']['precision']-p5s['macro_metrics']['precision'])*100:+.2f} pp |",
        f"| Macro Recall | {p5s['macro_metrics']['recall']*100:.2f}% | **{cur['macro_metrics']['recall']*100:.2f}%** | {(cur['macro_metrics']['recall']-p5s['macro_metrics']['recall'])*100:+.2f} pp |",
        f"| Macro F1 | {p5s['macro_metrics']['f1']*100:.2f}% | **{cur['macro_metrics']['f1']*100:.2f}%** | {(cur['macro_metrics']['f1']-p5s['macro_metrics']['f1'])*100:+.2f} pp |",
        f"| Entity Leakage Rate | {p5ls['entity_leakage_rate_percentage']:.2f}% ({p5ls['unmasked_leaked_fns']}/171) | **{cur_l['entity_leakage_rate_percentage']:.2f}% ({cur_l['unmasked_leaked_fns']}/171)** | {cur_l['entity_leakage_rate_percentage']-p5ls['entity_leakage_rate_percentage']:+.2f} pp |",
        f"| Sample Leakage Rate | {p5ls['sample_leakage_rate_percentage']:.2f}% ({p5ls['affected_samples_count']}/100) | **{cur_l['sample_leakage_rate_percentage']:.2f}% ({len(affected_samples)}/100)** | {cur_l['sample_leakage_rate_percentage']-p5ls['sample_leakage_rate_percentage']:+.2f} pp |",
        f"| Mean Latency | {p5s['latency_stats']['mean_seconds']:.3f}s | **{cur['latency_stats']['mean_seconds']:.3f}s** | {cur['latency_stats']['mean_seconds']-p5s['latency_stats']['mean_seconds']:+.3f}s |",
        f"| Total Inference Time | {p5s['latency_stats']['total_inference_time_seconds']:.2f}s | **{cur['latency_stats']['total_inference_time_seconds']:.2f}s** | {cur['latency_stats']['total_inference_time_seconds']-p5s['latency_stats']['total_inference_time_seconds']:+.2f}s |",
    ]
    with open(OUT_DIR / "phase3_2b6_comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== Phase 3.2B.6 Evaluation Complete ===")
    print(
        f"TP={cur['strict_true_positives']} FP={cur['strict_false_positives']} FN={cur['strict_false_negatives']}"
    )
    print(
        f"Micro P={cur['micro_metrics']['precision']*100:.2f}% R={cur['micro_metrics']['recall']*100:.2f}% F1={cur['micro_metrics']['f1']*100:.2f}%"
    )
    print(
        f"Macro P={cur['macro_metrics']['precision']*100:.2f}% R={cur['macro_metrics']['recall']*100:.2f}% F1={cur['macro_metrics']['f1']*100:.2f}%"
    )
    print(
        f"Leakage: {cur_l['unmasked_leaked_fns']}/171 ({cur_l['entity_leakage_rate_percentage']:.2f}%), Sample Leakage: {len(affected_samples)}/100 ({cur_l['sample_leakage_rate_percentage']:.2f}%)"
    )
    print(
        f"Latency: mean={cur['latency_stats']['mean_seconds']}s median={cur['latency_stats']['median_seconds']}s total={cur['latency_stats']['total_inference_time_seconds']}s"
    )


if __name__ == "__main__":
    main()
