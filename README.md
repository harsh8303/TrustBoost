# TrustBoost 🔐

> **An AI-Powered PII Sanitization Agent**

---

## What is TrustBoost?

TrustBoost is a privacy-first application that acts as a protective layer between you and external AI chatbots (like ChatGPT, Gemini, or Claude).

Before you share any text with an external AI, TrustBoost automatically detects and replaces sensitive personal information (PII) with safe, anonymous tokens — so your private data never leaves your machine in its original form.

---

## Main Objective

To build a simple, locally-running Python application that:

1. Accepts plain text input from the user.
2. Coordinates workflows through a **central LangChain AI Agent**.
3. Uses a **locally running Ollama LLM** (Gemma 3 4B) to intelligently detect Personally Identifiable Information (PII) — without relying on regex or external APIs.
4. Validates all entity spans and offsets deterministically before tokenization.
5. Replaces detected PII with anonymized tokens (e.g., `[PERSON_1]`, `[EMAIL_1]`) stored in a persistent in-memory global mapping.
6. Displays the sanitized text that is safe to copy into any AI chatbot.
7. Allows the user to **restore the original text locally** using the global in-memory token mapping.

---

## Technology Stack

| Component         | Technology                  |
|-------------------|------------------------------|
| Language          | Python 3.11                 |
| Agent Framework   | LangChain (`langchain-core`)|
| Frontend UI       | Streamlit                   |
| Local LLM Runtime | Ollama                      |
| Local LLM Model   | Gemma 3 4B (`gemma3:4b`)   |
| Config Management | python-dotenv               |
| Testing           | pytest                      |
| Data Handling     | pandas                      |

> **No external AI APIs are used.** All LLM inference runs locally on your machine via Ollama.

---

## Overall Architecture

```
Streamlit UI (Web Interface)
    │
    ▼
TrustBoost LangChain Agent (trustboost.agent.TrustBoostAgent)
    │
    ▼
Tool Selection & Execution
    ├── PII Detection Tool (detect_pii / Gemma 3 4B via Ollama)
    ├── Entity Validation Tool (validate_entities / offset & bounds check)
    ├── Global Tokenization Tool (tokenize_with_global_mapping)
    └── Restoration Tool (restore_with_global_mapping)
    │
    ▼
Global Mapping State (st.session_state["global_state"])
    │
    ▼
Final Output (Structured results for Streamlit)
```

---

## Project Structure

```
TrustBoost/
│
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies (Streamlit, Ollama, LangChain, etc.)
├── pyproject.toml              # Project metadata & pytest configuration
├── README.md                   # Project documentation (this file)
├── .env.example                # Sample environment configuration (safe to commit)
├── .env                        # Your local config (do NOT commit this)
├── .gitignore                  # Files excluded from version control
│
├── trustboost/                 # Core Python package
│   ├── __init__.py
│   ├── agent/                  # Central LangChain Agent & Ollama client
│   │   ├── __init__.py
│   │   ├── ollama_client.py    # Ollama communication layer
│   │   └── trustboost_agent.py # Central LangChain agent & tools
│   ├── detector/               # Contextual PII detection (Gemma 3 4B)
│   │   ├── __init__.py
│   │   ├── pii_detector.py     # Prompting, parsing & offset validation
│   │   └── prompt_builder.py   # Structured few-shot prompt builder
│   ├── tokenizer/              # Deterministic tokenization & restoration
│   │   ├── __init__.py
│   │   └── tokenizer.py        # Token allocation & single-pass restoration
│   ├── pipeline/               # Modular deterministic pipeline coordinator
│   │   ├── __init__.py
│   │   └── pipeline.py         # 9-step sanitization & 6-step restoration workflows
│   ├── session/                # Global in-memory mapping module
│   │   ├── __init__.py
│   │   └── global_mapping.py   # Cumulative mapping, counter tracking & reuse
│   ├── evaluation/             # Evaluation hooks (upcoming)
│   └── utils/                  # Shared utility functions
│
├── tests/
│   ├── test_agent.py           # Central LangChain Agent & tool execution tests
│   ├── test_app.py             # Streamlit UI smoke & state tests
│   ├── test_global_mapping.py  # Global mapping accumulation & isolation tests
│   ├── test_ollama_client.py   # Ollama integration & unit tests
│   ├── test_pii_detector.py    # PII detection & offset recovery tests
│   ├── test_pipeline.py        # End-to-end pipeline tests
│   └── test_tokenizer.py       # Tokenization & restoration unit tests
├── data/                       # Sample texts for testing
└── docs/                       # Project documentation and diagrams
```

---

## Prerequisites

Before running TrustBoost, make sure you have the following installed:

- **Python 3.11+** — via Miniforge/Conda
- **Ollama** — the local LLM runtime
- **Gemma 3 4B** — the language model used for PII detection

---

## Setup Instructions

### Step 1 — Install Ollama

Download and install Ollama from the official website:

```
https://ollama.com/download
```

Verify the installation:

```bash
ollama --version
```

### Step 2 — Download Gemma 3 4B

Pull the model to your local machine (this downloads ~2.5 GB):

```bash
ollama pull gemma3:4b
```

Verify the download:

```bash
ollama list
```

You should see `gemma3:4b` in the list.

### Step 3 — Start the Ollama Server

```bash
ollama serve
```

> On macOS, Ollama may already be running in the background if you installed the desktop app. If `ollama serve` says "address already in use", Ollama is already running — that's fine.

Verify the server is reachable:

```bash
curl http://localhost:11434/api/tags
```

### Step 4 — Clone the Repository

```bash
git clone https://github.com/your-username/TrustBoost.git
cd TrustBoost
```

### Step 5 — Activate the Python Environment

```bash
conda activate trustboost
```

Verify:

```bash
python --version
# Expected: Python 3.11.x

which python
# Expected: /Users/<you>/miniforge3/envs/trustboost/bin/python
```

### Step 6 — Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 7 — Configure Environment Variables

```bash
cp .env.example .env
```

The default `.env` configuration:

```
OLLAMA_MODEL=gemma3:4b
OLLAMA_HOST=http://localhost:11434
```

No API keys are required. This project uses a fully local Ollama model.

### Step 8 — Run the Streamlit Application

```bash
streamlit run app.py
```

The application opens automatically in your browser at `http://localhost:8501`.

---

## Running Tests

### Unit Tests (no Ollama required)

```bash
pytest -v -m "not integration"
```

### Integration Tests (Ollama must be running + gemma3:4b downloaded)

```bash
pytest -v -m integration
```

### All Tests

```bash
pytest
```

---

## Development Status

| Phase | Description                                    | Status        |
|-------|------------------------------------------------|---------------|
| 1     | Project Setup & Environment                    | ✅ Complete    |
| 2     | Ollama Installation & Model Integration        | ✅ Complete    |
| 3     | LLM-Based Contextual PII Detection             | ✅ Complete    |
| 4     | Tokenization & Restoration Logic               | ✅ Complete    |
| 5     | Streamlit UI & Full Pipeline                   | ✅ Complete    |
| 5+    | Global Mapping Architecture                    | ✅ Complete    |
| 5++   | Central LangChain Agent & Modular Pipeline     | ✅ Complete    |
| 6     | Evaluation & Testing                           | 🔜 Upcoming   |

---

## Central LangChain Agent Architecture

TrustBoost coordinates all operations through a single, central LangChain agent (`TrustBoostAgent`) residing in `trustboost/agent/trustboost_agent.py`.

### Agent Responsibilities

1. **Request Ingestion**: Receives the user request or operation from Streamlit (`sanitize` or `restore`).
2. **Tool Selection**: Uses explicit operation parameters, keyword routing, or an optional LLM classifier to select the appropriate LangChain tool.
3. **Pipeline Delegation**: Calls deterministic tools that delegate actual computation, offset checking, and state mutation to `trustboost.pipeline`.
4. **Safety Enforcement**:
   - Never bypasses mandatory PII offset verification.
   - Never sends sensitive text to external cloud APIs.
   - Fails closed: if detection, parsing, or validation fails, no mapping update occurs.
   - Never allows the agent to invent or hallucinate detection spans.
5. **Structured Return**: Returns standardized result dictionaries directly to the Streamlit UI.

### LangChain Tools

| Tool Name | Tool Identifier | Description |
|---|---|---|
| **Sanitize Pipeline Tool** | `sanitize_text` | Coordinates the full 9-step sanitization workflow. |
| **Restore Pipeline Tool** | `restore_text` | Coordinates the full 6-step restoration workflow. |
| **PII Detection Tool** | `pii_detection` | Sends prompt to local Gemma 3 4B via Ollama and retrieves raw detected entities. |
| **Entity Validation Tool** | `entity_validation` | Validates character offsets, category names, and exact slice matches. |
| **Global Tokenization Tool**| `global_tokenization` | Replaces validated spans with anonymous tokens using persistent counters. |
| **Restoration Tool** | `restoration` | Restores original values using the global mapping, preserving unmapped tokens. |

---

## Controlled Processing Workflows

### 1. Sanitization Workflow (9 Steps)

```
User Input Text
      │
      ▼
1. Receive Input (Validate non-empty string)
      │
      ▼
2. Build Detection Prompt (Few-shot structured prompt via prompt_builder.py)
      │
      ▼
3. Call Gemma 3 4B (Local Ollama via ollama_client.py)
      │
      ▼
4. Parse Detection Results (JSON extraction with markdown stripping)
      │
      ▼
5. Validate All Entities & Offsets (Type checking, bounds, and text[start:end] == entity["text"])
      │
      ▼
6. Reject Invalid Results (Fail-closed on corrupted spans or detection failure)
      │
      ▼
7. Call Global Tokenization (tokenize_with_global_mapping)
      │
      ▼
8. Update Global Mapping (Merge new entries into st.session_state["global_state"])
      │
      ▼
9. Return Output (Sanitized text, detected entities, token count, and status)
```

> **Safety Invariant:** No mapping update ever occurs before successful validation of all entity spans.

### 2. Restoration Workflow (6 Steps)

```
AI-Generated Response Text
      │
      ▼
1. Receive AI Response (Validate input string)
      │
      ▼
2. Retrieve Global Mapping (Fetch from st.session_state["global_state"]["global_mapping"])
      │
      ▼
3. Restore Known Placeholders (Single-pass token scan replacing [PERSON_1] -> "Rahul Sharma")
      │
      ▼
4. Preserve Unknown Placeholders (e.g. [UNKNOWN_TOKEN] preserved verbatim)
      │
      ▼
5. Report Unmapped Tokens (Populate unrestored_tokens list for UI warning)
      │
      ▼
6. Return Restored Response (Reconstructed text and diagnostic summary)
```

---

## Global Mapping Architecture

TrustBoost maintains a single shared in-memory mapping during the user's Streamlit browser session.

### How Global Mapping Works

```
Streamlit Session State:
  st.session_state["global_state"] = {
      "global_mapping": { token → original_value },
      "type_counters":  { entity_type → highest counter used },
      "val_to_token":   { "TYPE||value" → token },
  }

Input 1: "My name is Rahul Sharma."
  Mapping after Input 1:
    { "[PERSON_1]": "Rahul Sharma" }

Input 2: "Contact Aman Verma at aman@gmail.com."
  Mapping after Input 2:
    {
      "[PERSON_1]": "Rahul Sharma",
      "[PERSON_2]": "Aman Verma",
      "[EMAIL_1]":  "aman@gmail.com"
    }

Input 3: "Rahul Sharma again."
  → [PERSON_1] is REUSED. Counter does NOT increment.

Restoration:
  "Hello [PERSON_1] and [PERSON_2]"
  → "Hello Rahul Sharma and Aman Verma"
```

### Design Guarantees

| Guarantee | Description |
|---|---|
| **Global counters** | Token numbering never resets between inputs. `[PERSON_1]` persists into later inputs. |
| **Value reuse** | Identical `(type, value)` pairs always map to the same existing token. |
| **No silent overwrite** | Existing tokens are never replaced with different values. |
| **Fail-closed** | On detection or validation errors, the global mapping is left completely untouched. |
| **Ephemeral in-memory** | Mappings live in RAM (`st.session_state`) and vanish when the tab is closed. |
| **Zero disk persistence** | No database, file writes, or cloud telemetry. |

---

## User Workflow in Streamlit

1. **🛡️ Sanitize Text Tab:**
   - Paste raw sensitive text.
   - Click **"🛡️ Analyze & Sanitize"**.
   - The **TrustBoost LangChain Agent** executes the 9-step sanitization workflow.
   - Review detection metrics and the structured entities table.
   - Copy the sanitized text for sharing with ChatGPT, Claude, or Gemini.

2. **🔄 Restore AI Response Tab:**
   - Paste the external AI's response containing tokens like `[PERSON_1]` or `[EMAIL_1]`.
   - Click **"🔄 Restore Original Text"**.
   - The **TrustBoost LangChain Agent** executes the 6-step restoration workflow against the global mapping.
   - Any unknown tokens are preserved verbatim and reported.

3. **🗂️ Global Mapping Tab:**
   - Inspect all active token-to-value pairs stored in memory.
   - View the total token count.
   - Clear all mappings from RAM using the **"🗑️ Clear All Mappings"** button (requires checkbox confirmation).

---

## Supported PII Categories (24)

| Category Group | Types |
|---|---|
| **Personal Identity** | `PERSON`, `EMAIL`, `PHONE`, `ADDRESS`, `LOCATION`, `DATE_OF_BIRTH` |
| **Government & Financial** | `AADHAAR`, `PAN`, `BANK_ACCOUNT`, `CREDIT_CARD`, `IFSC`, `FINANCIAL_IDENTIFIER` |
| **Authentication & Security** | `PASSWORD`, `API_KEY`, `ACCESS_TOKEN`, `SECRET`, `PRIVATE_KEY`, `CREDENTIAL` |
| **Technical & Organizational** | `IP_ADDRESS`, `INTERNAL_URL`, `DATABASE_CREDENTIAL`, `SERVER_IDENTIFIER`, `CONFIDENTIAL_PROJECT`, `INTERNAL_ORGANIZATION_INFORMATION` |

---

## Important Notes

- 🔒 **100% Local Inference:** PII detection runs locally via Ollama (`gemma3:4b`). No data leaves your machine.
- 🤖 **LangChain AI Agent:** Central orchestrator managing tool selection and execution.
- 🛡️ **Fail-Closed Security:** Corrupted offsets or failed model outputs are rejected before mapping mutation.
- 🔄 **Reversible Tokenization:** Deterministic single-pass scanner guarantees round-trip text reconstruction.
- 🔑 **No Cloud Accounts or API Keys Required.**

---

## Author

Developed as a college B.Tech CSE project.
