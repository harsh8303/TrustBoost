"""Evaluation metrics and error taxonomy calculation for TrustBoost PII detection."""

import csv
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Check if two half-open intervals [s1, e1) and [s2, e2) overlap."""
    return max(s1, s2) < min(e1, e2)


def evaluate_sample_predictions(
    gold_entities: List[Dict[str, Any]],
    pred_entities: List[Dict[str, Any]],
    text: str = ""
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Evaluates predictions for a single sample using strict matching and categorizes errors.

    Returns:
        counts: dict with 'tp', 'fp', 'fn'
        errors: list of error dictionaries with category, details, and context.
    """
    # Track which gold and pred have been matched for strict TP
    matched_gold_strict = set()
    matched_pred_strict = set()

    # 1. Strict Match: exact start, end, and type
    for p_idx, pred in enumerate(pred_entities):
        for g_idx, gold in enumerate(gold_entities):
            if g_idx in matched_gold_strict:
                continue
            if (
                pred.get("start") == gold.get("start")
                and pred.get("end") == gold.get("end")
                and pred.get("type") == gold.get("type")
            ):
                matched_gold_strict.add(g_idx)
                matched_pred_strict.add(p_idx)
                break

    tp = len(matched_gold_strict)
    fp = len(pred_entities) - len(matched_pred_strict)
    fn = len(gold_entities) - len(matched_gold_strict)

    # 2. Error Categorization for non-strict matches
    errors = []
    unmatched_preds = [p_idx for p_idx in range(len(pred_entities)) if p_idx not in matched_pred_strict]
    unmatched_golds = [g_idx for g_idx in range(len(gold_entities)) if g_idx not in matched_gold_strict]

    # Map overlaps
    pred_to_gold_overlaps = defaultdict(list)
    gold_to_pred_overlaps = defaultdict(list)

    for p_idx in unmatched_preds:
        pred = pred_entities[p_idx]
        p_start, p_end = pred.get("start", -1), pred.get("end", -1)
        for g_idx in unmatched_golds:
            gold = gold_entities[g_idx]
            g_start, g_end = gold.get("start", -1), gold.get("end", -1)
            if spans_overlap(p_start, p_end, g_start, g_end):
                pred_to_gold_overlaps[p_idx].append(g_idx)
                gold_to_pred_overlaps[g_idx].append(p_idx)

    # Classify unmatched predictions
    resolved_golds = set()
    for p_idx in unmatched_preds:
        pred = pred_entities[p_idx]
        p_type = pred.get("type")
        p_text = pred.get("text")
        p_start = pred.get("start")
        p_end = pred.get("end")
        overlaps = pred_to_gold_overlaps.get(p_idx, [])

        if not overlaps:
            errors.append({
                "error_type": "spurious_detection",
                "description": "Model detected entity where no gold entity exists",
                "predicted": {"text": p_text, "type": p_type, "start": p_start, "end": p_end},
                "expected": None
            })
        else:
            # Overlaps with at least one gold entity
            g_idx = overlaps[0]
            gold = gold_entities[g_idx]
            resolved_golds.add(g_idx)
            g_type = gold.get("type")
            g_text = gold.get("text")
            g_start = gold.get("start")
            g_end = gold.get("end")

            if p_type != g_type:
                errors.append({
                    "error_type": "category_mismatch",
                    "description": f"Entity detected with incorrect category (predicted: {p_type}, expected: {g_type})",
                    "predicted": {"text": p_text, "type": p_type, "start": p_start, "end": p_end},
                    "expected": {"text": g_text, "type": g_type, "start": g_start, "end": g_end}
                })
            else:
                errors.append({
                    "error_type": "boundary_error",
                    "description": f"Entity detected with boundary difference: [{p_start}:{p_end}] vs [{g_start}:{g_end}]",
                    "predicted": {"text": p_text, "type": p_type, "start": p_start, "end": p_end},
                    "expected": {"text": g_text, "type": g_type, "start": g_start, "end": g_end}
                })

    # Classify remaining missed gold entities
    for g_idx in unmatched_golds:
        if g_idx not in resolved_golds:
            gold = gold_entities[g_idx]
            errors.append({
                "error_type": "missed_pii",
                "description": "Gold entity completely missed by the detector",
                "predicted": None,
                "expected": {"text": gold.get("text"), "type": gold.get("type"), "start": gold.get("start"), "end": gold.get("end")}
            })

    return {"tp": tp, "fp": fp, "fn": fn}, errors


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Computes precision, recall, and F1 given TP, FP, FN."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def compute_comprehensive_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes global micro, macro, and per-category metrics across all samples.
    Each sample in `samples` must contain 'entities' (gold) and 'predicted_entities'.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    category_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0, "pred_count": 0})
    error_counts = defaultdict(int)
    all_errors = []

    for sample in samples:
        gold = sample.get("entities", [])
        pred = sample.get("predicted_entities", [])
        text = sample.get("text", "")
        sample_id = sample.get("id", "unknown")

        for g in gold:
            t = g.get("type", "UNKNOWN")
            category_counts[t]["gold_count"] += 1
        for p in pred:
            t = p.get("type", "UNKNOWN")
            category_counts[t]["pred_count"] += 1

        counts, sample_errors = evaluate_sample_predictions(gold, pred, text)
        total_tp += counts["tp"]
        total_fp += counts["fp"]
        total_fn += counts["fn"]

        # Category level strict matches
        matched_gold_set = set()
        matched_pred_set = set()

        for p_idx, p in enumerate(pred):
            for g_idx, g in enumerate(gold):
                if g_idx in matched_gold_set:
                    continue
                if p.get("start") == g.get("start") and p.get("end") == g.get("end") and p.get("type") == g.get("type"):
                    matched_gold_set.add(g_idx)
                    matched_pred_set.add(p_idx)
                    category_counts[p["type"]]["tp"] += 1
                    break

        for p_idx, p in enumerate(pred):
            if p_idx not in matched_pred_set:
                category_counts[p.get("type", "UNKNOWN")]["fp"] += 1

        for g_idx, g in enumerate(gold):
            if g_idx not in matched_gold_set:
                category_counts[g.get("type", "UNKNOWN")]["fn"] += 1

        for err in sample_errors:
            err["sample_id"] = sample_id
            error_counts[err["error_type"]] += 1
            all_errors.append(err)

    # Micro average
    micro_metrics = compute_metrics(total_tp, total_fp, total_fn)

    # Category metrics & Macro average
    category_metrics = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for cat, counts in sorted(category_counts.items()):
        metrics = compute_metrics(counts["tp"], counts["fp"], counts["fn"])
        category_metrics[cat] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "gold_count": counts["gold_count"],
            "pred_count": counts["pred_count"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        }
        # Only include in macro average if category appears in gold or pred
        if counts["gold_count"] > 0 or counts["pred_count"] > 0:
            macro_precisions.append(metrics["precision"])
            macro_recalls.append(metrics["recall"])
            macro_f1s.append(metrics["f1"])

    macro_p = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_r = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    return {
        "summary": {
            "total_samples": len(samples),
            "total_gold_entities": sum(c["gold_count"] for c in category_counts.values()),
            "total_pred_entities": sum(c["pred_count"] for c in category_counts.values()),
            "strict_true_positives": total_tp,
            "strict_false_positives": total_fp,
            "strict_false_negatives": total_fn,
            "micro_metrics": micro_metrics,
            "macro_metrics": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4)
            }
        },
        "category_metrics": category_metrics,
        "error_distribution": dict(error_counts),
        "detailed_errors": all_errors
    }


def export_category_metrics_csv(category_metrics: Dict[str, Any], filepath: str):
    """Exports category metrics to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Gold Count", "Pred Count", "TP", "FP", "FN", "Precision", "Recall", "F1 Score"])
        for cat, data in sorted(category_metrics.items()):
            writer.writerow([
                cat,
                data["gold_count"],
                data["pred_count"],
                data["tp"],
                data["fp"],
                data["fn"],
                f"{data['precision']:.4f}",
                f"{data['recall']:.4f}",
                f"{data['f1']:.4f}"
            ])


def compute_privacy_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Privacy-focused metrics (what actually matters for TrustBoost):

    - char_recall: of all characters that belong to real PII, what fraction
      got masked? (1.0 = nothing leaked)
    - entities_with_any_leak: how many real PII items had at least one
      character left visible.
    - over_masking_rate: of all masked characters, what fraction were NOT
      PII (masking too much; annoying but safe).
    """
    total_pii_chars = masked_pii_chars = 0
    total_entities = leaky_entities = 0
    total_masked_chars = masked_non_pii_chars = 0
    leaks = []

    for sample in samples:
        masked = set()
        for p in sample.get("predicted_entities", []):
            masked.update(range(p["start"], p["end"]))
        gold_chars = set()
        for g in sample.get("entities", []):
            span = set(range(g["start"], g["end"]))
            gold_chars |= span
            total_pii_chars += len(span)
            masked_pii_chars += len(span & masked)
            total_entities += 1
            if span - masked:
                leaky_entities += 1
                leaks.append({"sample_id": sample.get("id"), "type": g["type"], "text": g["text"],
                              "unmasked_chars": len(span - masked), "total_chars": len(span)})
        total_masked_chars += len(masked)
        masked_non_pii_chars += len(masked - gold_chars)

    return {
        "char_recall": round(masked_pii_chars / total_pii_chars, 4) if total_pii_chars else 1.0,
        "entities_with_any_leak": leaky_entities,
        "entity_any_leak_rate": round(leaky_entities / total_entities, 4) if total_entities else 0.0,
        "over_masking_rate": round(masked_non_pii_chars / total_masked_chars, 4) if total_masked_chars else 0.0,
        "total_gold_entities": total_entities,
        "leaked_entities": leaks,
    }
