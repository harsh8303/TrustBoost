"""
app.py — TrustBoost Streamlit Application
-----------------------------------------
PURPOSE:
    Web interface for TrustBoost: An AI-Powered PII Sanitization Agent.

ARCHITECTURE:
    Streamlit UI
        │
    TrustBoost LangChain Agent (trustboost.agent.TrustBoostAgent)
        │
    Tool Selection and Execution
        ├── PII Detection (Gemma 3 4B via Ollama)
        ├── Entity Validation (Offset recovery & duplicate removal)
        ├── Global Tokenization (Cumulative counter preservation)
        └── Restoration (Single-pass replacement with unmapped tracking)
        │
    Global Mapping State (st.session_state["global_state"])
        │
    Final Output (Structured results for Streamlit)

TABS:
    1. 🛡️ Sanitize Text       — Dispatches sanitization through TrustBoost Agent.
    2. 🔄 Restore AI Response  — Dispatches restoration through TrustBoost Agent.
    3. 🗂️ Global Mapping      — Inspect all stored tokens; clear with confirmation.
    4. ℹ️ Architecture        — Supported PII categories and agent workflow.

PRIVACY:
    - 100% local: Gemma 3 4B via Ollama, zero external API calls.
    - Zero disk persistence: everything lives in Streamlit session RAM.
    - Mapping is wiped when the browser tab is closed or the page is refreshed.
"""

from typing import Any
import pandas as pd
import streamlit as st

from trustboost.agent import (
    get_agent,
    get_config,
    get_full_status,
)
from trustboost.session import clear_global_mapping


# ── Cached Ollama Status Helper ───────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def check_ollama_status() -> dict[str, Any]:
    """
    Checks Ollama server and model availability.
    Cached for 60 seconds to avoid latency on repetitive UI reruns.
    """
    try:
        return get_full_status()
    except Exception as exc:
        return {
            "server": {"connected": False, "message": f"Error: {exc}"},
            "model": {"available": False, "message": "Ollama check failed."},
            "config": get_config(),
            "ready": False,
        }


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state() -> None:
    """Ensures all session state variables exist with safe initial values."""
    # Global mapping state: one dict holds the entire accumulated mapping
    if "global_state" not in st.session_state:
        st.session_state["global_state"] = clear_global_mapping()

    # View / display state for the most recent sanitization
    defaults = {
        "sanitize_input_text": "",
        "original_text": "",
        "sanitized_text": "",
        "detected_entities": [],
        "sanitization_message": "",
        "restoration_input": "",
        "restoration_output": "",
        "unrestored_tokens": [],
        # Retained for test_app.py backward-compatibility assertions
        "token_mapping": {},
        "tokens_metadata": [],
        "session_timestamp": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar UI Component ──────────────────────────────────────────────────────
def render_sidebar(
    config: dict[str, Any],
    system_ready: bool,
    server_connected: bool,
    model_available: bool,
) -> None:
    """Renders the sidebar containing Ollama status, agent status, and privacy notes."""
    with st.sidebar:
        st.title("🔐 TrustBoost")
        st.caption("AI-Powered PII Sanitization Agent")
        st.divider()

        # ── Ollama Connection ─────────────────────────────────────────────────
        st.subheader("🔌 Ollama Connection")

        if system_ready:
            st.success(f"🟢 **Ready:** `{config['model']}`", icon="✅")
        elif server_connected and not model_available:
            st.warning(f"🟡 **Model Missing:** `{config['model']}`", icon="⚠️")
            st.caption(f"Run `ollama pull {config['model']}` in terminal.")
        else:
            st.error("🔴 **Ollama Offline**", icon="❌")
            st.caption("Start Ollama in terminal with `ollama serve`.")

        col_btn, col_info = st.columns([1, 1])
        with col_btn:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_info:
            st.caption(f"Host: `{config['host'].replace('http://', '')}`")

        st.divider()

        # ── Global Mapping Stats ──────────────────────────────────────────────
        st.subheader("🗂️ Global Mapping")
        global_mapping = st.session_state.get("global_state", {}).get(
            "global_mapping", {}
        )
        total_tokens = len(global_mapping)
        if total_tokens > 0:
            st.metric("Tokens in RAM", total_tokens)
            st.caption("Shared across all sanitizations.")
        else:
            st.caption("No mappings stored yet.")

        st.divider()

        # ── Privacy Architecture Policy ───────────────────────────────────────
        st.subheader("🛡️ Privacy Policy")
        st.markdown(
            """
            - **LangChain AI Agent:** Central orchestrator selecting sanitization/restoration tools.
            - **100% Local Inference:** PII detection runs locally via Ollama (`gemma3:4b`).
            - **Global In-Memory Mapping:** Shared token space across inputs; counters never reset.
            - **Zero Disk Persistence:** No database or logs touch your sensitive data.
            """
        )
        st.caption("TrustBoost · College B.Tech CSE Project")


# ── Tab 1: Sanitize Text ──────────────────────────────────────────────────────
def clear_sanitize_input() -> None:
    """
    Streamlit button callback to clear only the input text area widget.
    Executes before widget instantiation on rerun, preserving sanitized
    output, detected entities, and global mapping.
    """
    st.session_state["sanitize_input_text"] = ""


def render_sanitize_tab(config: dict[str, Any], system_ready: bool) -> None:
    """Renders the input, agent sanitization invocation, and results display."""
    st.markdown("### 1. Input Text")
    st.caption(
        "Paste the raw text you intend to share with an AI assistant (ChatGPT, Claude, Gemini, etc.). "
        "The TrustBoost LangChain Agent will detect sensitive PII and tokenize it into the shared global mapping."
    )

    user_input = st.text_area(
        label="Enter your text",
        height=180,
        placeholder=(
            "Paste your text here. TrustBoost will detect sensitive information and "
            "anonymize it before you share it with an AI assistant."
        ),
        help="Input text is analyzed strictly on your local machine using Gemma 3 4B.",
        key="sanitize_input_text",
    )

    col_action, col_clear = st.columns([1, 5])
    with col_action:
        analyze_clicked = st.button(
            "🛡️ Analyze & Sanitize", type="primary", use_container_width=True
        )
    with col_clear:
        st.button(
            "🧹 Clear Input",
            on_click=clear_sanitize_input,
            use_container_width=False,
        )

    # ── Process Sanitization via Agent ────────────────────────────────────────
    if analyze_clicked:
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.", icon="⚠️")
        elif not system_ready:
            st.error(
                f"❌ **Ollama is not ready.** Make sure Ollama is running (`ollama serve`) "
                f"and `{config['model']}` is downloaded (`ollama pull {config['model']}`).",
                icon="🚨",
            )
        else:
            with st.spinner("TrustBoost Agent analyzing text locally with Gemma 3 4B..."):
                current_global_state = st.session_state.get(
                    "global_state", clear_global_mapping()
                )
                agent = get_agent()
                result = agent.sanitize(user_input, current_global_state)

            if result.get("success"):
                st.session_state["global_state"] = result.get(
                    "updated_global_state", current_global_state
                )
                st.session_state["original_text"] = user_input
                st.session_state["sanitized_text"] = result.get("sanitized_text", "")
                st.session_state["detected_entities"] = result.get("detected_entities", [])
                st.session_state["token_mapping"] = result.get("mapping", {})
                st.session_state["tokens_metadata"] = result.get("tokens", [])
                st.session_state["sanitization_message"] = result.get("message", "")

                total_tokens = len(
                    st.session_state["global_state"].get("global_mapping", {})
                )
                detected_count = len(result.get("detected_entities", []))

                if detected_count == 0:
                    st.success(
                        "✅ No sensitive information detected. The original text is safe to share as-is.",
                        icon="🎉",
                    )
                else:
                    st.success(
                        f"✅ **{result['message']}** "
                        f"Global mapping now contains **{total_tokens}** token(s) total.",
                        icon="🎉",
                    )
            else:
                st.error(
                    f"❌ Sanitization failed: {result.get('message', 'Unknown error.')}",
                    icon="🚨",
                )

    # ── Display Latest Sanitized Output ───────────────────────────────────────
    if st.session_state.get("sanitized_text"):
        st.markdown("---")
        st.markdown("### 2. Detection Summary")

        detected = st.session_state["detected_entities"]
        global_mapping = st.session_state.get("global_state", {}).get(
            "global_mapping", {}
        )

        cat_counts: dict[str, int] = {}
        for ent in detected:
            cat = ent.get("type", "UNKNOWN")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Entities Detected (this input)", len(detected))
        with kpi2:
            st.metric("Global Tokens in RAM", len(global_mapping))
        with kpi3:
            st.metric("Entity Categories (this input)", len(cat_counts))

        if cat_counts:
            pill_cols = st.columns(len(cat_counts))
            for i, (cat, count) in enumerate(sorted(cat_counts.items())):
                with pill_cols[i % len(pill_cols)]:
                    st.info(f"**{cat}**: {count}")

        if detected:
            st.markdown("#### Detected Entities")
            val_to_token = {v: k for k, v in global_mapping.items()}
            table_rows = []
            for ent in detected:
                ent_text = ent.get("text", "")
                ent_type = ent.get("type", "")
                assigned_token = val_to_token.get(ent_text, "—")
                table_rows.append({
                    "Entity Type": ent_type,
                    "Detected Information": ent_text,
                    "Assigned Token": assigned_token,
                    "Reason": ent.get("reason", "Contextual PII"),
                })
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 3. Sanitized Text — Safe to Copy")
        st.caption(
            "This text is safe to share with external AI assistants. "
            "Use the copy button in the top right of the box below to copy it."
        )
        st.code(st.session_state["sanitized_text"], language=None)

        with st.expander("✏️ Edit Sanitized Text (Optional)"):
            st.text_area(
                "Editable Sanitized Text",
                value=st.session_state["sanitized_text"],
                height=150,
                key="editable_sanitized_text",
            )

        with st.expander("👁️ View Original Text (Sensitive)"):
            st.text(st.session_state["original_text"])


# ── Tab 2: Restore AI Response ────────────────────────────────────────────────
def render_restore_tab() -> None:
    """Renders the AI response restoration workflow via the TrustBoost Agent."""
    st.markdown("### Restore Original Information")
    st.markdown(
        """
        After you receive a response from ChatGPT, Claude, or Gemini containing tokens
        like `[PERSON_1]` or `[EMAIL_1]`, paste it below.
        The TrustBoost LangChain Agent restores original values using the **global mapping**
        automatically — no session selection required.
        """
    )

    global_mapping = st.session_state.get("global_state", {}).get(
        "global_mapping", {}
    )

    if not global_mapping:
        st.info(
            "💡 **No tokens in the global mapping yet.** "
            "Sanitize some text in the **'🛡️ Sanitize Text'** tab first.",
            icon="ℹ️",
        )
        return

    st.success(
        f"🔒 **Global Mapping Active:** {len(global_mapping)} token(s) available for restoration.",
        icon="✅",
    )

    restore_input = st.text_area(
        label="Paste the AI response containing TrustBoost tokens",
        value=st.session_state["restoration_input"],
        height=180,
        placeholder=(
            "Example: Hello [PERSON_1], we reviewed your project "
            "[CONFIDENTIAL_PROJECT_1] and sent confirmation to [EMAIL_1]..."
        ),
        help="Tokens will be restored using the shared global mapping.",
    )

    if st.button("🔄 Restore Original Text", type="primary"):
        if not restore_input or not restore_input.strip():
            st.warning("⚠️ Please paste an AI response to restore.", icon="⚠️")
        else:
            agent = get_agent()
            restoration_result = agent.restore(restore_input, global_mapping)

            if restoration_result.get("success"):
                st.session_state["restoration_input"] = restore_input
                st.session_state["restoration_output"] = restoration_result.get(
                    "restored_text", ""
                )
                st.session_state["unrestored_tokens"] = restoration_result.get(
                    "unrestored_tokens", []
                )

                unrestored = restoration_result.get("unrestored_tokens", [])
                if unrestored:
                    st.warning(
                        f"⚠️ Restoration completed with {len(unrestored)} unmapped token(s): "
                        f"`{', '.join(unrestored)}`. Unmapped tokens were preserved verbatim.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        "🎉 All tokens successfully restored using the global mapping!",
                        icon="✅",
                    )
            else:
                st.error(
                    f"❌ Restoration failed: {restoration_result.get('message', 'Unknown error.')}",
                    icon="🚨",
                )

    # Display Restored Output
    if st.session_state.get("restoration_output"):
        st.markdown("---")
        st.markdown("### Restored Text")
        st.caption("Personal details restored from the global mapping:")
        st.code(st.session_state["restoration_output"], language=None)


# ── Tab 3: Global Mapping Viewer ──────────────────────────────────────────────
def render_global_mapping_tab() -> None:
    """Displays all stored token-to-original-value pairs and provides a clear action."""
    st.markdown("### 🗂️ Global Mapping")
    st.caption(
        "All token-to-original-value pairs accumulated during this browser session. "
        "Shared across every sanitization performed by the TrustBoost Agent."
    )

    global_mapping = st.session_state.get("global_state", {}).get(
        "global_mapping", {}
    )

    if not global_mapping:
        st.info("No token mappings stored yet.", icon="ℹ️")
    else:
        st.metric("Total Mappings", len(global_mapping))
        with st.expander(
            f"🔍 View All {len(global_mapping)} Token Mapping(s)", expanded=False
        ):
            map_data = [
                {"Token": token, "Original Value": value}
                for token, value in sorted(global_mapping.items())
            ]
            st.dataframe(pd.DataFrame(map_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ⚠️ Clear All Mappings")
    st.caption(
        "Permanently removes all stored token mappings from RAM. "
        "After clearing, previously sanitized tokens cannot be restored."
    )

    confirm_clear = st.checkbox(
        "I understand this will permanently erase all mappings in RAM.",
        key="confirm_clear_global",
    )
    if st.button(
        "🗑️ Clear All Mappings", type="primary", disabled=not confirm_clear
    ):
        st.session_state["global_state"] = clear_global_mapping()
        st.session_state["sanitized_text"] = ""
        st.session_state["detected_entities"] = []
        st.session_state["restoration_output"] = ""
        st.session_state["restoration_input"] = ""
        st.session_state["unrestored_tokens"] = []
        st.success("✅ All mappings cleared from memory.", icon="🗑️")
        st.rerun()


# ── Tab 4: Architecture & Categories ──────────────────────────────────────────
def render_architecture_tab() -> None:
    """Displays system architecture and supported category catalog."""
    st.markdown("### TrustBoost LangChain Agent Architecture")
    st.markdown(
        """
        TrustBoost coordinates privacy workflows through one central LangChain agent:

        ```
        Streamlit UI
            │
        TrustBoost LangChain Agent (trustboost.agent.TrustBoostAgent)
            │
        Tool Selection and Execution
            ├── PII Detection (Gemma 3 4B via Ollama)
            ├── Entity Validation (Offset recovery & duplicate removal)
            ├── Global Tokenization (Cumulative counter preservation)
            └── Restoration (Single-pass replacement with unmapped tracking)
            │
        Global Mapping State (st.session_state["global_state"])
            │
        Final Output (Structured results for Streamlit)
        ```

        - **Central Agent:** Coordinates tool selection and delegates to deterministic pipeline.
        - **Global Counters:** Token numbering never resets between inputs.
        - **Value Reuse:** Identical (type, value) pairs always reuse the same token.
        - **No Overwrite:** An existing token is never silently overwritten.
        - **In-Memory Volatility:** Mapping resides only in browser session RAM and disappears on exit.
        """
    )

    st.markdown("### Supported PII Categories (24)")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        st.markdown("**Personal Identity**")
        st.markdown("- `PERSON`\n- `EMAIL`\n- `PHONE`\n- `ADDRESS`\n- `LOCATION`\n- `DATE_OF_BIRTH`")

    with col_c2:
        st.markdown("**Government & Finance**")
        st.markdown("- `AADHAAR`\n- `PAN`\n- `BANK_ACCOUNT`\n- `CREDIT_CARD`\n- `IFSC`\n- `FINANCIAL_IDENTIFIER`")

    with col_c3:
        st.markdown("**Authentication & Secrets**")
        st.markdown("- `PASSWORD`\n- `API_KEY`\n- `ACCESS_TOKEN`\n- `SECRET`\n- `PRIVATE_KEY`\n- `CREDENTIAL`")

    with col_c4:
        st.markdown("**Technical & Organization**")
        st.markdown("- `IP_ADDRESS`\n- `INTERNAL_URL`\n- `DATABASE_CREDENTIAL`\n- `SERVER_IDENTIFIER`\n- `CONFIDENTIAL_PROJECT`\n- `INTERNAL_ORGANIZATION_INFORMATION`")


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="TrustBoost — AI Privacy Layer",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Query Ollama status
    status_data = check_ollama_status()
    config = status_data.get("config", get_config())
    server_connected = status_data.get("server", {}).get("connected", False)
    model_available = status_data.get("model", {}).get("available", False)
    system_ready = status_data.get("ready", False)

    render_sidebar(config, system_ready, server_connected, model_available)

    # Header
    st.title("🔐 TrustBoost")
    st.subheader("Your privacy layer for AI conversations.")
    st.markdown(
        "Detect and anonymize sensitive information locally before sharing text with external AI tools. "
        "Powered by a central LangChain Agent with one global in-memory mapping."
    )
    st.divider()

    # Application Tabs
    tab_sanitize, tab_restore, tab_mapping, tab_architecture = st.tabs([
        "🛡️ Sanitize Text",
        "🔄 Restore AI Response",
        "🗂️ Global Mapping",
        "ℹ️ Architecture & Categories",
    ])

    with tab_sanitize:
        render_sanitize_tab(config, system_ready)

    with tab_restore:
        render_restore_tab()

    with tab_mapping:
        render_global_mapping_tab()

    with tab_architecture:
        render_architecture_tab()

    st.divider()
    st.caption(
        "TrustBoost — AI-Powered PII Sanitization Agent · Built with Python, Streamlit, LangChain & Ollama · "
        "Designed for College B.Tech CSE AI/Privacy Demonstrations."
    )


if __name__ == "__main__":
    main()
