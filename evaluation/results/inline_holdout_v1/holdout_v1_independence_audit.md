# TrustBoost — Phase 3.3B: Independent Holdout Dataset Integrity & Leakage Audit

**Scope**: Strict read-only audit comparing the 30-sample `inline_holdout_v1` evaluation set (`evaluation/results/inline_holdout_v1/holdout_v1_predictions.json` / `run_holdout_v1_evaluation.py`) against the 100-sample development benchmark (`evaluation/dataset/dataset.json`, 171 gold entities) and the production prompt (`trustboost/detector/prompt_builder.py`). No source code, prompts, datasets, tests, or evaluation artifacts were modified, and no model inference was executed.

---

## 1. Summary Counts & Final Assessment

| Audit Metric | Count |
| :--- | :---: |
| Original development samples inspected (`evaluation/dataset/dataset.json`) | **100** (171 gold entities) |
| Holdout samples inspected (`inline_holdout_v1`) | **30** (24 positive, 6 negative; 24 gold entities) |
| Exact duplicate samples (raw, whitespace-, case-, or punctuation-normalized) | **0** |
| Internal duplicates within `inline_holdout_v1` | **0** |
| Near-duplicate sentences vs `dataset.json` (`NEAR DUPLICATE`) | **0** |
| Reused gold PII values (`VALUE REUSE`: exact / normalized) | **2** exact / **0** additional normalized |
| Development-context exposure cases (`DEVELOPMENT-CONTEXT EXPOSURE`) | **3** (negative controls `holdout_26`, `holdout_27`, `holdout_28`; plus pattern echo in `holdout_06`) |
| Format-only similarity pairs (`FORMAT-ONLY SIMILARITY`) | **7** positive pairs (`holdout_03`, `holdout_09`, `holdout_10`, `holdout_11`, `holdout_16`, `holdout_19`, `holdout_21`) |
| **Overall Holdout Independence Assessment** | **MOSTLY INDEPENDENT WITH LIMITED EXPOSURE** |

---

## 2. Methodology

1. **Exact & Normalized Text Overlap**: Compared every holdout input against all 100 `dataset.json` inputs under four normalizations: exact string equality, whitespace-collapsed, case-folded, and punctuation-stripped + case-folded. Also checked for duplicates within `inline_holdout_v1`.
2. **Near-Duplicate & Template Similarity**: Computed word-token Jaccard, 2-gram and 3-gram word-shingle Jaccard, and `difflib.SequenceMatcher` ratio across all `30 × 100 = 3,000` holdout–development pairs, flagging pairs with `SequenceMatcher >= 0.45`, `word_jaccard >= 0.30`, or `3-gram_jaccard > 0` for manual inspection.
3. **Gold PII Value Overlap**: Compared all 24 holdout gold entity strings against all 171 development gold entity strings (exact, case-insensitive, and whitespace-normalized), representing matched values by SHA-256 prefix and character length.
4. **Development-Context Exposure**: Compared all 30 holdout inputs against `trustboost/detector/prompt_builder.py` and the documented Phase 3.2B.6D prompt-engineering negative-test suite.

---

## 3. Exact Text Overlap & Gold PII Value Overlap

- **Exact / Normalized Sample Overlap**: `0` matches across raw, whitespace-normalized, case-normalized, and punctuation-normalized comparisons (`0` internal holdout duplicates).
- **Gold PII Value Reuse (`VALUE REUSE`)**: **2** exact matches (and `0` additional case/whitespace-normalized matches):

| Holdout ID | Holdout Type | Dev Sample ID | Dev Type | Masked Value Fingerprint | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `holdout_01` | `AADHAAR` | `sample_020` | `AADHAAR` | `sha256:ce2e7b8dca` (len=14) | **VALUE REUSE** |
| `holdout_18` | `PAN` | `sample_017` | `PAN` | `sha256:6442fd73a9` (len=10) | **VALUE REUSE** |

Both are common synthetic test fixtures (`4321...` / `ABCDE...`) rather than real PII, and both categories are covered by deterministic recognizers in `pii_detector.py`; nevertheless, they represent exact gold value reuse across the development and holdout sets. No values were reused across differing entity categories.

---

## 4. Suspicious Pairs & Classification of Findings

### A. Holdout vs Development Dataset (`evaluation/dataset/dataset.json`)

| Holdout ID | Dev Sample ID | `SequenceMatcher` | Word Jaccard | 3-Gram Jaccard | Classification & Evidence |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `holdout_01` | `sample_020` | 0.483 | 0.267 | 0.071 | **VALUE REUSE** — Same synthetic `AADHAAR` value (`sha256:ce2e7b8dca`); different surrounding sentence and single-entity vs 2-entity structure. |
| `holdout_18` | `sample_017` | 0.326 | 0.167 | 0.000 | **VALUE REUSE** — Same synthetic `PAN` value (`sha256:6442fd73a9`); different sentence (`"PAN number: ..."` vs `"Taxpayer ... has provided PAN ..."`). |
| `holdout_03` | `sample_002` | 0.576 | 0.235 | 0.000 | **FORMAT-ONLY SIMILARITY** — Shared delivery verb/preposition (`"Deliver the ..."`), different street address and person/location structure. |
| `holdout_09` | `sample_036` | 0.589 | 0.176 | 0.000 | **FORMAT-ONLY SIMILARITY** — Both use `postgresql://` URI syntax, but with distinct username, password, host, and database name. |
| `holdout_10` | `sample_012` | 0.571 | 0.231 | 0.091 | **FORMAT-ONLY SIMILARITY** — Shared `"Date of birth:"` label and `DD-Mon-YYYY` date format; different date and no `PERSON` entity in `holdout_10`. |
| `holdout_11` | `sample_001` | 0.451 | 0.150 | 0.000 | **FORMAT-ONLY SIMILARITY** — Shared `@example.com` domain convention; different persona and sentence. |
| `holdout_16` | `sample_074` | 0.379 | 0.150 | 0.056 | **FORMAT-ONLY SIMILARITY** — Both use RFC 5737 `198.51.100.0/24` documentation IPv4 block with different host octets (`.23` vs `.45`) and different contexts (prose vs debug log). |
| `holdout_19` | `sample_031` | 0.464 | 0.182 | 0.000 | **FORMAT-ONLY SIMILARITY** — Shared `"Temporary ... password"` phrasing; different password value and sentence structure. |
| `holdout_21` | `sample_077` | 0.566 | 0.333 | 0.200 | **FORMAT-ONLY SIMILARITY** — Shared `+1-202-555-01xx` North American 555 documentation phone prefix (`-0148` vs `-0143`); different format (prose vs bulleted list). |

### B. Holdout vs Prompt & Development Context (`trustboost/detector/prompt_builder.py` / Phase 3.2B.6D)

| Holdout ID | Reference Artifact | Similarity / Match | Classification & Evidence |
| :--- | :--- | :---: | :--- |
| `holdout_26` | `prompt_builder.py` line 109 | Verbatim substring | **DEVELOPMENT-CONTEXT EXPOSURE** — `"Status: Active, role: Manager."` appears verbatim inside `prompt_builder.py` (`## CONTEXT RULES`) as an explicit negative example added in Phase 3.2B.6D.1–3.2B.6D.3. |
| `holdout_27` | Phase 3.2B.6D negative test set | Light paraphrase | **DEVELOPMENT-CONTEXT EXPOSURE** — `"We read Project Gutenberg books in our literature class."` is a light paraphrase of `"We read Project Gutenberg texts in class."` used repeatedly during Phase 3.2B.6D–3.2B.6D.3 negative testing. |
| `holdout_28` | `prompt_builder.py` Example 2 (`sample_099` style) | `SequenceMatcher = 0.653` | **DEVELOPMENT-CONTEXT EXPOSURE** — `"The office meeting is scheduled for next Monday."` closely mirrors `Example 2` in `prompt_builder.py` (`"The meeting is scheduled for 10 AM tomorrow."`). |
| `holdout_06` | `prompt_builder.py` line 117 | Pattern echo | **CLEAN / Pattern-Level Note** — `"Internal security notice: Project Aurora is restricted."` uses a new project name (`Project Aurora`) and wording, though it echoes the `"security notices"` context rule added in Phase 3.2B.6D. |

All remaining holdout samples (`holdout_02`, `holdout_04`, `holdout_05`, `holdout_07`, `holdout_08`, `holdout_12`, `holdout_13`, `holdout_14`, `holdout_15`, `holdout_17`, `holdout_20`, `holdout_22`, `holdout_23`, `holdout_24`, `holdout_25`, `holdout_29`, `holdout_30`) are **CLEAN** with respect to `dataset.json` and `prompt_builder.py` overlap.

---

## 5. Limitations

1. **Authorship Provenance**: `inline_holdout_v1` was constructed inline for Phase 3.3 evaluation without an external provenance log; textual and value-overlap checks cannot prove statistical independence or rule out shared author style/templates.
2. **Single-Entity Simplicity**: All 24 positive holdout samples are short single-sentence, single-entity inputs, whereas `dataset.json` contains denser multi-entity and structured inputs; thus, holdout metrics cannot be directly equated with production difficulty.
3. **Negative-Control Exposure**: Because 3 of the 6 negative controls (`holdout_26`, `holdout_27`, `holdout_28`) overlap with prompt/development-testing negative patterns, the `0/6` negative-control false-positive rate in Phase 3.3 overstates generalization on truly unseen negative phrasing.

---

## 6. Conclusion

**Assessment: `MOSTLY INDEPENDENT WITH LIMITED EXPOSURE`**

- **Against `evaluation/dataset/dataset.json`**: Strong textual independence (`0` exact duplicates, `0` near-duplicate sentences, `22/24` novel gold values), with **2** synthetic gold PII value reuses (`holdout_01`/`sample_020` `AADHAAR` and `holdout_18`/`sample_017` `PAN`).
- **Against prompt/development context**: **3** negative controls (`holdout_26`, `holdout_27`, `holdout_28`) have direct exposure to `prompt_builder.py` / Phase 3.2B.6D negative-testing examples (`holdout_26` is verbatim in the prompt). Future holdout suites (`inline_holdout_v2+`) should replace `holdout_01`/`holdout_18` gold values and author fresh negative controls that do not appear in `prompt_builder.py`.
