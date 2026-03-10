"""
SpatialChat — Streamlit Frontend

A conversational interface for spatial transcriptomics analysis.
Supports multi-turn conversation with dataset persistence and plot display.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import uuid
import warnings

# Suppress Pydantic serialization warnings from LangGraph internals
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

import streamlit as st

# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="SpatialChat",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ───────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_dataset_id" not in st.session_state:
    st.session_state.active_dataset_id = None


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.title("SpatialChat")
    st.caption("Natural language interface for spatial transcriptomics")

    st.divider()

    # Dataset status
    ds = st.session_state.active_dataset_id
    if ds:
        st.success(f"Dataset loaded: **{ds}**")
    else:
        st.info("No dataset loaded yet. Ask me to find one!")

    st.divider()

    # Available datasets
    st.markdown("**Available datasets:**")
    try:
        from data.loaders import load_catalog
        catalog = load_catalog()
        for ds_id, info in catalog.items():
            name = info.get("name", ds_id)
            desc = info.get("description", "")
            st.markdown(f"- **{name}** (`{ds_id}`)")
            if desc:
                st.caption(desc)
    except Exception:
        st.caption("Could not load dataset catalog.")

    st.divider()

    # New conversation button
    if st.button("New Conversation", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.chat_history = []
        st.session_state.active_dataset_id = None
        # Reset the graph singleton so it gets a fresh checkpointer
        import graph as _g
        _g._compiled = None
        _g._agents.clear()
        st.rerun()

    st.divider()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")


# ── Chat display ─────────────────────────────────────────────

st.markdown("### Ask me about spatial transcriptomics data")

# Render chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])
        # Render plots if present
        for plot_b64 in entry.get("plots", []):
            try:
                st.image(base64.b64decode(plot_b64), use_container_width=True)
            except Exception:
                st.warning("Could not render plot.")


# ── User input ───────────────────────────────────────────────

user_input = st.chat_input("e.g. Snap25 expression in Mouse Brain")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_input,
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                from graph import chat

                result = chat(
                    message=user_input,
                    thread_id=st.session_state.thread_id,
                )

                response_text = result["response"]
                plots = result.get("plots", [])

                # Update dataset ID
                if result.get("active_dataset_id"):
                    st.session_state.active_dataset_id = result["active_dataset_id"]

                # Display response
                st.markdown(response_text)

                # Display plots
                for plot_b64 in plots:
                    try:
                        st.image(base64.b64decode(plot_b64), use_container_width=True)
                    except Exception:
                        st.warning("Could not render plot.")

                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": response_text,
                    "plots": plots,
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": error_msg,
                })

    st.rerun()
