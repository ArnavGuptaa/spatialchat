"""
Main LangGraph wiring. Connects supervisor and sub-agents.

Architecture:
  1. 'reset' entry node clears per-invocation state each turn
  2. Supervisor routes to sub-agents; sub-agents append results manually
  3. Synthesizer produces final answer with plots embedded as images
  4. Checkpointer enables multi-turn conversation in LangGraph Studio
  5. Plots stored in PlotStore by ID during processing, embedded at the end
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import (
    SpatialChatState, DATASET_FINDER, EXPLORATORY,
    SPATIAL_STATS, NEIGHBORHOOD, FINISH,
)
from agents.supervisor import create_supervisor_node, create_synthesizer_node
from agents.sub_agents import (
    create_dataset_finder_agent, create_exploratory_agent,
    create_spatial_stats_agent, create_neighborhood_agent,
)
from tools.base import get_plot_base64, clear_plot_store

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Extract plain text from message content (handles both str and multimodal list).

    LangGraph Studio sends content as list of blocks: [{"text": "...", "type": "text"}]
    Jupyter / direct invocation sends content as plain str.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content) if content else ""


def _get_last_user_text(state: SpatialChatState) -> str:
    """Extract the LAST user query from messages (for multi-turn support).

    Handles BaseMessage objects, dicts, and multimodal content.
    """
    last_text = ""
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "human":
            last_text = _extract_text(msg.content)
        elif isinstance(msg, dict) and msg.get("type") == "human":
            last_text = _extract_text(msg.get("content", ""))
    return last_text


def _sub_agent_input(state: SpatialChatState, agent_name: str) -> list:
    """Build minimal messages for a sub-agent: just the user query + dataset context."""
    query = _get_last_user_text(state)
    ds = state.get("active_dataset_id")
    if agent_name != DATASET_FINDER and ds:
        query = f"[Dataset: {ds}] {query}"
    return [HumanMessage(content=query)]


# ── Agent cache (lazy creation, reused across invocations) ────

_agents: dict[str, object] = {}


def _get_agent(name: str, factory):
    """Get or create a cached sub-agent."""
    if name not in _agents:
        _agents[name] = factory()
    return _agents[name]


# ── Reset node ───────────────────────────────────────────────

def _reset_node(state: SpatialChatState) -> dict:
    """Reset per-invocation state at the start of each user turn.

    This prevents accumulation of tool_summaries, plot_ids, visited_agents
    from previous turns when using a checkpointer (LangGraph Studio).
    """
    clear_plot_store()
    return {
        "tool_summaries": [],
        "plot_ids": [],
        "visited_agents": [],
        "supervisor_turns": 0,
        "error_count": 0,
        "next_agent": None,
    }


# ── Node builders ─────────────────────────────────────────────

def _make_node(agent_name: str, factory):
    """
    Create a graph node that runs a sub-agent and extracts compact results.

    Since per-invocation fields have NO reducer, each node must manually
    concatenate its results with what's already in state.
    """

    def node(state: SpatialChatState) -> dict:
        agent_fn = _get_agent(agent_name, factory)
        input_msgs = _sub_agent_input(state, agent_name)

        # Run sub-agent (bind_tools handles tool calls internally)
        result = agent_fn(input_msgs)

        # Manually concatenate with existing per-invocation state
        existing_summaries = state.get("tool_summaries", [])
        existing_plot_ids = state.get("plot_ids", [])
        existing_visited = state.get("visited_agents", [])

        update: dict = {
            "messages": [AIMessage(content=result["summary"][:500])],
            "tool_summaries": existing_summaries + result.get("tool_summaries", []),
            "plot_ids": existing_plot_ids + result.get("plot_ids", []),
            "visited_agents": existing_visited + [agent_name],
        }

        # Dataset detection
        did = result.get("detected_dataset_id")
        if not did and agent_name == DATASET_FINDER:
            from data.loaders import get_cache
            loaded = get_cache().loaded_ids()
            if loaded:
                did = loaded[-1]
        if did:
            update["active_dataset_id"] = did

        return update

    return node


def route_from_supervisor(state: SpatialChatState) -> str:
    """Route from supervisor to next agent or synthesizer."""
    nxt = state.get("next_agent")
    valid = {DATASET_FINDER, EXPLORATORY, SPATIAL_STATS, NEIGHBORHOOD}
    return nxt if nxt in valid else "synthesizer"


# ── Graph assembly ────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build the LangGraph StateGraph.

    Args:
        checkpointer: Optional checkpointer for multi-turn conversation.
            LangGraph Studio provides its own; Streamlit uses MemorySaver.
    """
    g = StateGraph(SpatialChatState)

    # Reset node clears per-invocation state each turn
    g.add_node("reset", _reset_node)
    g.add_node("supervisor", create_supervisor_node())
    g.add_node(DATASET_FINDER, _make_node(DATASET_FINDER, create_dataset_finder_agent))
    g.add_node(EXPLORATORY, _make_node(EXPLORATORY, create_exploratory_agent))
    g.add_node(SPATIAL_STATS, _make_node(SPATIAL_STATS, create_spatial_stats_agent))
    g.add_node(NEIGHBORHOOD, _make_node(NEIGHBORHOOD, create_neighborhood_agent))
    g.add_node("synthesizer", create_synthesizer_node())

    # Flow: reset → supervisor → (sub-agents ↔ supervisor) → synthesizer → END
    g.set_entry_point("reset")
    g.add_edge("reset", "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, {
        DATASET_FINDER: DATASET_FINDER, EXPLORATORY: EXPLORATORY,
        SPATIAL_STATS: SPATIAL_STATS, NEIGHBORHOOD: NEIGHBORHOOD,
        "synthesizer": "synthesizer",
    })
    for a in [DATASET_FINDER, EXPLORATORY, SPATIAL_STATS, NEIGHBORHOOD]:
        g.add_edge(a, "supervisor")
    g.add_edge("synthesizer", END)

    return g.compile(checkpointer=checkpointer)


# ── Singleton graph (for LangGraph Studio) ────────────────────

_compiled = None


def get_graph():
    """Get or create the compiled graph with MemorySaver for multi-turn."""
    global _compiled
    if _compiled is None:
        from config.settings import get_settings
        get_settings().setup_langsmith()
        _compiled = build_graph(checkpointer=MemorySaver())
    return _compiled


def make_graph():
    """Factory for langgraph dev / LangGraph Studio."""
    return get_graph()


# ── Public chat interface (for Streamlit) ─────────────────────

def chat(message: str, thread_id: str = "default") -> dict:
    """
    Send a message to the agent and get a response with plots.

    Args:
        message: User's natural language query.
        thread_id: Thread ID for conversation continuity.

    Returns:
        dict with keys:
          - response: str (synthesized answer)
          - plots: list[str] (base64 PNG strings for display)
          - active_dataset_id: str | None (current dataset)
    """
    graph = get_graph()

    # The 'reset' node handles clearing per-invocation state automatically
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract response text
    response = "No response generated."
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
            content = msg.content
            # Handle multimodal content (text + images)
            if isinstance(content, list):
                text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                response = " ".join(text_parts) if text_parts else str(content)
            else:
                response = content
            break

    # Collect plots from PlotStore using IDs
    plots = []
    for pid in result.get("plot_ids", []):
        b64 = get_plot_base64(pid)
        if b64:
            plots.append(b64)

    return {
        "response": response,
        "plots": plots,
        "active_dataset_id": result.get("active_dataset_id"),
    }
