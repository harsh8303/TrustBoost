# TrustBoost Comprehensive Evaluation Framework

This directory contains the production-grade, modular evaluation framework for **TrustBoost**, an AI-powered local PII sanitization and restoration system powered by Google DeepMind's **Gemma 3 4B** via local Ollama.

The framework provides empirical, reproducible verification of TrustBoost's contextual PII detection, privacy guarantees, deterministic tokenization pipelines, LangChain agent architecture, and end-to-end performance.

---

## 1. Architecture Overview

TrustBoost operates as a privacy-preserving layer before text reaches external LLMs:

```
User Input Text
      │
      ▼
TrustBoost LangChain Agent (trustboost/agent/)
      │
      ├──> PII Detection Tool (Gemma 3 4B via Ollama)
      │       └─ Contextual inference across 24 PII categories
      │
      ├──> Entity Validation Tool
      │       └─ Invariant verification: text[start:end] == entity["text"]
      │
      ├──> Global Tokenization Tool
      │       └─ Cumulative mapping: [PERSON_1], [PAN_1] with token reuse
      │
      └──> Restoration Tool
              └─ Safe reverse substitution of model placeholders
```

All evaluations in this suite execute **100% locally** without external APIs, regex detection baselines, or cloud databases.

---

## 2. Directory Structure

```
evaluation/
├── README.md                           # Comprehensive documentation (this file)
├── run_all_evaluations.py              # Master runner orchestrating all suites
├── dataset/
│   ├── dataset.json                    # 100 synthetic ground-truth samples
│   └── generate_dataset.py             # Deterministic generator & offset validator
├── detection/
│   ├── evaluate.py                     # Detection evaluation runner with caching
│   └── metrics.py                      # Strict matching, micro/macro F1, error taxonomy
├── security/
│   └── security_tests.py               # Leakage, restoration fidelity, fail-closed tests
├── agent/
│   └── agent_tests.py                  # LangChain routing, tool dispatch, state tests
├── pipeline/
│   └── pipeline_tests.py               # Edge cases, adjacent spans, Unicode, emoji tests
├── performance/
│   └── benchmark.py                    # Multi-tier latency & throughput benchmark
├── robustness/
│   └── robustness_tests.py             # 10-style linguistic and formatting robustness
└── results/                            # Output evaluation reports & datasets
    ├── evaluation_report.json          # PII detection summary & micro/macro metrics
    ├── category_metrics.csv            # Per-category precision, recall, and F1
    ├── detection_error_analysis.json   # Detailed taxonomy of false pos/neg & boundary errors
    ├── security_report.json            # Security & privacy test outcomes
    ├── agent_report.json               # LangChain agent evaluation metrics
    ├── pipeline_report.json            # Deterministic pipeline edge-case results
    ├── performance_report.json         # Latency, memory, and throughput benchmarks
    └── robustness_report.json          # Robustness scores across 10 text categories
```

---

## 3. Dataset Specification (`evaluation/dataset/`)

The evaluation dataset contains **exactly 100 diverse, fully synthetic, fictional samples** stored in `evaluation/dataset/dataset.json`.

### Schema
```json
{
  "id": "sample_001",
  "category": "personal",
  "text_style": "standard",
  "text": "Hi, my name is Alice Walker and you can email me at alice.walker@example.com anytime.",
  "entities": [
    {
      "text": "Alice Walker",
      "type": "PERSON",
      "start": 15,
      "end": 27
    },
    {
      "text": "alice.walker@example.com",
      "type": "EMAIL",
      "start": 52,
      "end": 76
    }
  ]
}
```

### Key Invariants
1. **Zero-Based Character Indexing**: Spans follow half-open intervals `[start, end)`.
2. **Strict Slice Invariant**: `text[start:end] == entity["text"]` is validated programmatically on every single entity.
3. **Non-Overlapping Ground Truth**: Entities within a sample never share character spans.
4. **Fictional Data**: All names, phone numbers (555), emails (`example.com`), and tax/national IDs are synthetic.

### Entity Categories Covered (24 Types)
- **Personal (6)**: `PERSON`, `EMAIL`, `PHONE`, `ADDRESS`, `LOCATION`, `DATE_OF_BIRTH`
- **Financial & Government (6)**: `AADHAAR`, `PAN`, `BANK_ACCOUNT`, `CREDIT_CARD`, `IFSC`, `FINANCIAL_IDENTIFIER`
- **Security & Credentials (6)**: `PASSWORD`, `API_KEY`, `ACCESS_TOKEN`, `SECRET`, `PRIVATE_KEY`, `CREDENTIAL`
- **Technical & Organizational (6)**: `IP_ADDRESS`, `INTERNAL_URL`, `DATABASE_CREDENTIAL`, `SERVER_IDENTIFIER`, `CONFIDENTIAL_PROJECT`, `INTERNAL_ORGANIZATION_INFORMATION`

---

## 4. Evaluation Modules

### 4.1 PII Detection Evaluation (`evaluation/detection/`)
- **Matching Methodology**: Strict entity-level evaluation. A True Positive (TP) requires an exact match on `(type, start, end)`.
- **Metrics Computed**:
  - Precision: $\frac{TP}{TP + FP}$
  - Recall: $\frac{TP}{TP + FN}$
  - F1-Score: $\frac{2 \cdot P \cdot R}{P + R}$
  - Micro-average (aggregated across all entity instances)
  - Macro-average (unweighted average across all 24 entity classes)
  - Category-wise breakdown exported to `evaluation/results/category_metrics.csv`
- **Error Taxonomy**:
  - `missed_pii`: Model failed to detect a gold entity (False Negative).
  - `spurious_detection`: Model hallucinated an entity where none exists (False Positive).
  - `category_mismatch`: Span detected correctly or partially, but with wrong label.
  - `boundary_error`: Correct entity type, but imprecise start/end character offsets.

### 4.2 Security and Privacy Suite (`evaluation/security/`)
- **Leakage Prevention**: Validates that 0% of identified sensitive values remain in the sanitized string.
- **Restoration Fidelity**: Validates 100% round-trip restoration of entities in mock AI responses.
- **Unknown Placeholder Preservation**: Confirms that unmapped tokens (e.g., `[EMAIL_99]`) are preserved intact and never corrupted or deleted.
- **Global Mapping Consistency**: Confirms that repeated occurrences of identical PII values reuse identical tokens without duplicating or resetting counters.
- **Fail-Closed Resilience**: Evaluates adversarial inputs (corrupted non-JSON strings, truncated responses, negative offsets, hallucinated entity types, and out-of-bounds indices), ensuring graceful degradation without unhandled crashes.

### 4.3 LangChain Agent Architecture Suite (`evaluation/agent/`)
- **Operation Routing**: Evaluates dispatch for `sanitize`, `restore`, and `clear_mapping` operations.
- **Tool Registry**: Validates registration and execution of all 6 tools (`sanitize_text`, `restore_text`, `pii_detection`, `entity_validation`, `global_tokenization`, `restoration`).
- **Error Guards**: Asserts fail-closed error reporting on null inputs, invalid types, and empty strings.
- **State Integrity**: Confirms that external callers cannot corrupt or arbitrarily mutate the in-memory global mapping state.

### 4.4 Deterministic Pipeline Suite (`evaluation/pipeline/`)
- **Core Pipeline**: Verifies offset-ordered single-pass replacement preventing offset shift bugs.
- **Complex Edge Cases**:
  - Adjacent entities separated only by single delimiters (e.g. `Bob,bob@corp.com`).
  - Substring collision avoidance (e.g. "Cat" in "Catherine").
  - Multi-byte Unicode and emoji characters (`👋`, `🚀`).
  - Special symbols in secrets and passwords (`P@ssw0rd!#$`, `sec*live&99%`).
  - Brackets and quotation enclosing entities (`"[Alice]"`, `<alice@corp.com>`).

### 4.5 Performance Benchmark (`evaluation/performance/`)
- **Input Tiers**:
  - Small Text (~100 words)
  - Medium Text (~500 words)
  - Large Text (~1,000 words)
- **Granular Latencies**:
  - Model Cold-Start latency
  - LLM Detection latency (Gemma 3 4B inference)
  - Deterministic Sanitization latency (token generation and mapping update)
  - Deterministic Restoration latency (placeholder reversal)
  - End-to-End Pipeline latency
- **Statistics**: Mean, median, min, max, standard deviation, and throughput (words/sec and chars/sec).

### 4.6 Robustness Suite (`evaluation/robustness/`)
Evaluates model and pipeline behavior across 10 distinct text style categories:
1. Standard English Prose
2. Hinglish / Code-Switching
3. Informal Chat / Slang / Typos
4. Structured Formatting (Markdown tables, JSON, logs)
5. Dense Multi-Entity Records
6. Negative Samples (Technical docs, clean text without PII)
7. Ambiguous Contexts (Common nouns as names like "May", "Rose")
8. Unusual Punctuation and Whitespace (Tabs, nested brackets)
9. Mixed Casing (ALL-CAPS shout text, lowercase)
10. Extended Multi-Sentence Narratives

---

## 5. Running the Evaluations

### Running the Entire Suite
```bash
python evaluation/run_all_evaluations.py
```

### Running Individual Suites
```bash
# 1. Dataset Generation & Validation
python evaluation/dataset/generate_dataset.py

# 2. Pipeline Edge Cases & Determinism
python evaluation/pipeline/pipeline_tests.py

# 3. Security & Privacy Suite
python evaluation/security/security_tests.py

# 4. LangChain Agent Architecture Suite
python evaluation/agent/agent_tests.py

# 5. Robustness Suite
python evaluation/robustness/robustness_tests.py

# 6. Performance Benchmark
python evaluation/performance/benchmark.py

# 7. Contextual PII Detection Evaluation
python evaluation/detection/evaluate.py --limit 25
```
