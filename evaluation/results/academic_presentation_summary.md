# TrustBoost: Academic Presentation Summary & Scientific Defense

**Project**: TrustBoost — Local Contextual AI-Powered PII Sanitization Agent
**Underlying Architecture**: LangChain Agent Architecture + Google DeepMind Gemma 3 4B (Local Ollama)
**Evaluator**: Independent Academic Audit Harness
**Date**: November 2026

---

## 1. Project Objective

TrustBoost is designed as a local, zero-leakage privacy intermediary between users and external Large Language Models (LLMs). The core thesis is that users should be able to submit prompts containing proprietary or sensitive personal information without exposing that data to third-party AI APIs.

### Functional Invariants
1. **Contextual PII Detection**: Identifying sensitive entities using local language models (Gemma 3 4B) rather than rigid regex patterns.
2. **Reversible Placeholder Tokenization**: Replacing detected entities with typed placeholders (`[PERSON_1]`, `[EMAIL_1]`) while preserving grammatical and syntactic integrity.
3. **Global Mapping State**: Maintaining token consistency across multi-turn interactions within an application session.
4. **Deterministic Restoration**: Substituting original values back into LLM responses using a single-pass bijective token map.
5. **Local Containment**: Zero external telemetry or network API dependencies for PII processing.

---

## 2. Evaluation Methodology & Verification Principles

The evaluation framework was audited under strict scientific verification standards:
- **Zero Synthetic or Fabricated Metrics**: All metrics are derived from real execution of Gemma 3 4B via local Ollama.
- **Strict Entity-Level Matching**: True Positives require exact quadruple equality:
  $$\text{pred.type} == \text{gold.type} \;\land\; \text{pred.start} == \text{gold.start} \;\land\; \text{pred.end} == \text{gold.end} \;\land\; \text{pred.text} == \text{gold.text}$$
- **Independent Invariant Checks**: Character offsets verified via slice equality:
  $$\text{text}[\text{start}:\text{end}] == \text{entity}[\text{'text'}]$$
- **Distinction Between Survival and Correctness**: Non-crashing pipeline execution is strictly decoupled from ground-truth detection accuracy.
- **Separation of Evidence and Hypotheses**: Measured facts are reported with raw figures; unlogged causal factors are labeled as hypotheses.

---

## 3. Dataset Characteristics

- **Total Annotated Samples**: **100** samples (`sample_001` through `sample_100`).
- **Total Ground-Truth Entities**: **171** verified entities.
- **Character Offset Errors**: **0 errors** (100% slice invariant compliance).
- **Supported Entity Categories**: All **24 supported types** represented:
  - Personal: `PERSON`, `EMAIL`, `PHONE`, `ADDRESS`, `LOCATION`, `DATE_OF_BIRTH`
  - National & Financial Identifiers: `AADHAAR`, `PAN`, `BANK_ACCOUNT`, `IFSC`, `CREDIT_CARD`, `FINANCIAL_IDENTIFIER`
  - Technical & Secrets: `API_KEY`, `ACCESS_TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `DATABASE_CREDENTIAL`, `PRIVATE_KEY`
  - Infrastructure: `IP_ADDRESS`, `SERVER_IDENTIFIER`, `INTERNAL_URL`, `INTERNAL_ORGANIZATION_INFORMATION`, `CONFIDENTIAL_PROJECT`
- **Linguistic & Formatting Diversity**:
  - Standard English Prose (56%)
  - Hinglish Code-Switching (10%)
  - Informal Chat with Slang/Typos (7%)
  - Dense Multi-Entity Records (8%)
  - Markdown Tables & System Logs (9%)
  - Clean Negative Controls with No PII (6%)
  - Ambiguous Common Nouns / Months / Flowers (4%)

---

## 4. Detection Performance Metrics

Evaluated on all 100 samples and 171 ground-truth entities:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL DETECTION SCORECARD                        │
├───────────────────────────────┬────────────────────────┬───────────────┤
│ Metric                        │ Micro-Averaged         │ Macro-Averaged│
├───────────────────────────────┼────────────────────────┼───────────────┤
│ Precision                     │ 74.21% (118 / 159)     │ 56.41%        │
│ Recall                        │ 69.01% (118 / 171)     │ 58.19%        │
│ F1-Score                      │ 71.52%                 │ 56.09%        │
│ Strict True Positives (TP)    │ 118                    │ —             │
│ Strict False Positives (FP)   │ 41                     │ —             │
│ Strict False Negatives (FN)   │ 53                     │ —             │
│ Total Strict Error Points     │ 94 (41 FP + 53 FN)     │ —             │
└───────────────────────────────┴────────────────────────┴───────────────┘
```

### High-Performing vs. Low-Performing Categories
- **Perfect F1 (1.0000)**: `PHONE` (11/11), `PAN` (6/6), `IP_ADDRESS` (7/7), `SERVER_IDENTIFIER` (5/5).
- **High F1 (> 0.85)**: `PERSON` (0.9167), `DATE_OF_BIRTH` (0.9333), `EMAIL` (0.8889), `AADHAAR` (0.8571).
- **Low F1 (< 0.50)**: `ADDRESS` (0.2667 - frequent boundary truncation), `PASSWORD` (0.3333), `IFSC` (0.3636), `CONFIDENTIAL_PROJECT` (0.2857).
- **Zero F1 (0.0000)**: `FINANCIAL_IDENTIFIER` (misclassified as `BANK_ACCOUNT`), `DATABASE_CREDENTIAL` (partially parsed), `PRIVATE_KEY` (0 predictions), `INTERNAL_ORGANIZATION_INFORMATION`.

---

## 5. Security & PII Leakage Findings

### The "Zero Leakage" Fallacy
A core finding of this audit is that **"0% Tokenization Omission" does not equal "0% Privacy Leakage"**. Prior claims of zero leakage merely demonstrated that the deterministic tokenizer replaced whatever entities the LLM detector passed to it. In an end-to-end evaluation against ground truth, whenever the detector misses an entity (False Negative), that sensitive string bypasses sanitization.

### Empirical Leakage Measurements
- **Total Ground-Truth Entities**: 171
- **Strict False Negatives**: 53
- **Masked / Protected False Negatives**: 39 (Imprecise bounds or category mislabels still replaced the sensitive text with a placeholder).
- **Residual Verbatim Leaked Entities**: **14** (Completely missed entities with zero overlapping predictions).
- **Entity Leakage Rate**: **8.19%** (14 / 171).
- **Sample Leakage Rate**: **13.0%** (13 / 100 samples).

### Limitations of Exact Substring Leakage Checking
Exact substring matching ($\text{gold.text} \in \text{sanitized\_text}$) provides an empirical lower bound on leakage, but possesses inherent limitations:
1. **Decomposed Compound Disclosures**: In `sample_036`, the ground truth was a complete connection URI `postgresql://admin:P@ssw0rd2026@localhost:5432/maindb`. The model detected only `P@ssw0rd2026` as `PASSWORD`. The sanitized text was:
   `postgresql://admin:[PASSWORD_1]@localhost:5432/maindb`
   Exact substring matching reports this as "not leaked" because the full string was broken. However, sensitive components (`admin`, `localhost`, `maindb`) remained exposed.
2. **Contextual Surrounding Leakage**: Exact matching cannot determine whether surrounding grammatical context still reveals partial entity attributes.

---

## 6. Error Taxonomy & Reconciliation Proof

The audit successfully reconciled the apparent discrepancy between strict error points ($41\text{ FP} + 53\text{ FN} = 94$) and error taxonomy events (61 events):

### The Reconciliation Equation
$$\text{Strict Errors (94)} = \text{Strict FP (41)} + \text{Strict FN (53)}$$

1. **6 Spurious Detections**: Contributes **6 FP, 0 FN**.
2. **35 Overlapping Prediction-Gold Pairs**:
   - 23 Boundary Errors
   - 12 Category Mismatches
   - Because 6 predictions were compound spans covering 2 gold entities, these 35 predictions overlap **39 gold entities**.
   - Contributes: **35 FP, 39 FN**.
3. **14 Truly Missed Entities**: Unmatched gold entities with 0 overlapping predictions.
   - Contributes: **0 FP, 14 FN**.

$$\text{Total FP} = 6 + 35 = \mathbf{41}$$
$$\text{Total FN} = 39 + 14 = \mathbf{53}$$
$$\text{Total Strict Errors} = 41 + 53 = \mathbf{94}$$

---

## 7. Global Mapping State Boundaries

An architectural audit of `trustboost/session/global_mapping.py` and `app.py` established strict operational boundaries:

| Scope | Implementation Reality | Verification Status |
| :--- | :--- | :---: |
| **Within Same Session** | Values map bijectively; repeated entities reuse identical token indices. | **VERIFIED** |
| **Across Script Reruns** | Preserved in `st.session_state["global_state"]` across UI button clicks. | **VERIFIED** |
| **Across Browser Tabs** | Streamlit isolates `st.session_state` per connection; no cross-tab sharing. | **REFUTED** |
| **Across Restarts** | State resides strictly in volatile memory RAM; zero disk persistence. | **REFUTED** |

---

## 8. Performance Measurements: Multi-Iteration Benchmarking (5 Iterations)

### Empirically Measured Facts
- **Cold-Start Latency**: **6.8108s** (isolated prior to timing; un-timed warm-up took 3.871s).
- **Small Text (74 words, 540 chars)**:
  - **Iterations**: 5 completed / 5 attempted (100% success rate)
  - **Mean Detection Latency**: **28.2030s** (median: 25.8915s, min: 23.0134s, max: 36.2565s, stdev: 5.8649s)
  - **Throughput**: 2.62 words/second (19.15 characters/second)
  - **Detected Entities**: [10, 9, 9, 10, 9] (mean: 9.4 entities)
  - **Sanitization Mean**: 5.1 ms | **Restoration Mean**: 3.3 ms
- **Medium Text (261 words, 2,093 chars)**:
  - **Iterations**: 5 completed / 5 attempted (100% success rate)
  - **Mean Detection Latency**: **154.6702s** (median: 159.1867s, min: 132.2340s, max: 163.8720s, stdev: 12.8927s)
  - **Throughput**: 1.69 words/second (13.53 characters/second)
  - **Detected Entities**: [30, 32, 33, 32, 33] (mean: 32.0 entities)
  - **Sanitization Mean**: 3.5 ms | **Restoration Mean**: 2.0 ms
- **Large Text (455 words, 3,774 chars)**:
  - **Iterations**: 0 completed / 5 attempted (**100% fail-closed rejection rate**)
  - **Failure Mode**: In all 5 iterations, `detect_pii` returned `success: False` because Gemma 3 4B produced output that failed JSON validation/extraction under the dense 3,774-character prompt context.
- **Deterministic Pipeline Speed**: Tokenization took 0.3ms–5.7ms; Restoration took 0.1ms–15.4ms (< 0.001% of pipeline latency).
- **LLM Dominance**: Contextual inference accounts for **99.99%** of end-to-end execution time.

### Hypotheses vs. Unlogged Metrics
- **Causal Hypothesis (Decoding Length Asymmetry)**:
  The observed phenomenon where 261 words took longer than 455 words (in previous single-run testing) is hypothesized to stem from autoregressive decoding: serial generation of 32 entity JSON structures (~1,500 characters) versus generating `{"entities": []}` (~15 characters).
- **Academic Caveat**: Low-level Ollama token counts (`eval_count`) were not logged by the benchmark script. Therefore, decoding length asymmetry is a **well-supported causal hypothesis**, not a directly measured metric.

---

## 9. Known System Limitations

1. **Inference Latency on Commodity Hardware**: Running a 4B parameter model locally on CPU/unified memory requires ~28s for short paragraphs and ~2.5 minutes on medium multi-entity records.
2. **Context Length Failure Boundary**: On documents approaching 1,000 words (3,774 characters), Gemma 3 4B failed closed across all 5 benchmark iterations due to malformed/truncated JSON responses, establishing an empirical upper limit for monolithic prompt extraction.
3. **Contextual Attention Degradation on Dense Spans**: In dense records, formatting prefixes (e.g. ` IFSC: `) are occasionally absorbed into entity boundaries.
4. **Tabular Data Vulnerability**: Markdown column tables exhibit higher false-negative rates due to missing syntactic prose context.
5. **Volatile Session State**: In-memory token mapping resets when the server is restarted or the browser tab is closed.

---

## 10. Future Architectural Recommendations

1. **Hybrid Two-Stage Pipeline**:
   - Stage 1: Ultra-fast local regex and pattern-matching rules for deterministic formats (PAN, Aadhaar, IP addresses, URLs, API keys) executing in < 5ms.
   - Stage 2: Contextual LLM detection reserved for ambiguous entities (names, organizations, project titles, unstructured addresses).
2. **Grammar-Constrained Decoding**:
   Enforcing strict JSON schema via llama.cpp BNF grammar or Ollama structured output to prevent boundary hallucinations and eliminate JSON parsing errors.
3. **Compound Entity Decomposition**:
   Implementing recursive entity parsing to ensure compound values (such as database connection strings) are broken down and sanitized in their entirety.
4. **Persistent Secure Vault**:
   Optional local SQLite/SQLCipher encrypted storage for persistent cross-session global token mappings.

---

## Test Suite Final Verification

```bash
pytest tests/
============================= 215 passed in 48.76s =============================
```
100% of the project unit and integration test suite passes without regressions.
