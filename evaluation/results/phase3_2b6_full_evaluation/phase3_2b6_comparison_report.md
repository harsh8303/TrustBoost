# TrustBoost — Phase 3.2B.6 Full 100-Sample Evaluation & Comparison vs Phase 3.2B.5

**Commit**: `a6e98a19c14b24c241bfc05499124e47211c8662` | **Model**: `gemma3:4b` (`temperature=0`, `format` omitted, `num_ctx=4096`)\
**Samples**: 100 (171 gold entities) | **Fresh Inference**: `use_cache=False`\
**Failures / Malformed**: 0 / 0 | **Tok / Restore**: 100/100 / 100/100

| Metric | Phase 3.2B.5 | Phase 3.2B.6 | Delta |
| :--- | :---: | :---: | :---: |
| Predicted Entities | 176 | **176** | +0 |
| Strict TP | 145 | **146** | +1 |
| Strict FP | 31 | **30** | -1 |
| Strict FN | 26 | **25** | -1 |
| Micro Precision | 82.39% | **82.95%** | +0.56 pp |
| Micro Recall | 84.80% | **85.38%** | +0.58 pp |
| Micro F1 | 83.57% | **84.15%** | +0.58 pp |
| Macro Precision | 73.78% | **74.71%** | +0.93 pp |
| Macro Recall | 78.84% | **80.00%** | +1.16 pp |
| Macro F1 | 75.40% | **76.61%** | +1.21 pp |
| Entity Leakage Rate | 2.92% (5/171) | **1.17% (2/171)** | -1.75 pp |
| Sample Leakage Rate | 5.00% (5/100) | **2.00% (2/100)** | -3.00 pp |
| Mean Latency | 3.995s | **4.460s** | +0.465s |
| Total Inference Time | 399.51s | **446.01s** | +46.50s |
