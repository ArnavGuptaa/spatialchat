"""
Supervisor and synthesizer nodes.

The supervisor routes to sub-agents with loop prevention.
The synthesizer produces the final user-facing answer with plots embedded.

Key design:
  - Supervisor context is MINIMAL: dataset status, visited agents, last summaries
  - Synthesizer uses tool_summaries (compact text), never sees raw data
  - Plots are embedded as data:image/png URLs in the final AIMessage
  - Hard loop breakers prevent infinite routing
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config.settings import get_settings
from agents.state import (
    SpatialChatState, DATASET_FINDER, EXPLORATORY,
    SPATIAL_STATS, NEIGHBORHOOD, FINISH,
    MAX_RETRIES, MAX_SUPERVISOR_TURNS,
)
from tools.base import get_plot_base64


# Shared text extraction helpers

def _extract_text(content) -> str:
    """Extract plain text from message content (str or multimodal list)."""
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
    """Extract the LAST user query text from state messages."""
    last_text = ""
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "human":
            last_text = _extract_text(msg.content)
        elif isinstance(msg, dict) and msg.get("type") == "human":
            last_text = _extract_text(msg.get("content", ""))
    return last_text


# Supervisor

SUPERVISOR_PROMPT = """You route user queries to specialist sub-agents. Pick ONE at a time.

AGENTS:
- dataset_finder: Search/load datasets. Call FIRST if no dataset loaded.
- exploratory: Gene expression plots. Requires dataset.
- spatial_stats: Moran's I, co-occurrence. Requires dataset.
- neighborhood: Cell neighbors, interactions. Requires dataset.

RULES:
- Dataset loaded? Skip dataset_finder.
- Each agent: call AT MOST ONCE per turn.
- Have enough results? → FINISH.
- NEVER re-route to the same agent."""


class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Brief routing rationale")
    next_agent: Literal["dataset_finder", "exploratory", "spatial_stats", "neighborhood", "FINISH"]


def create_supervisor_node():
    llm = get_settings().get_llm().with_structured_output(SupervisorDecision)

    def supervisor_node(state: SpatialChatState) -> dict:
        errors = state.get("error_count", 0)
        turns = state.get("supervisor_turns", 0)
        visited = state.get("visited_agents", [])
        active_ds = state.get("active_dataset_id")
        visited_set = set(visited)

        # Hard loop breakers
        if errors >= MAX_RETRIES or turns >= MAX_SUPERVISOR_TURNS:
            return {"next_agent": FINISH, "supervisor_turns": turns + 1}

        # Pre-LLM fast path
        if active_ds and DATASET_FINDER in visited_set:
            if EXPLORATORY not in visited_set:
                return {"next_agent": EXPLORATORY, "supervisor_turns": turns + 1}
            return {"next_agent": FINISH, "supervisor_turns": turns + 1}

        # Build COMPACT context for LLM
        ctx = []
        ctx.append(f"Dataset: {active_ds or 'NONE'}")
        if visited:
            ctx.append(f"Already called (DO NOT re-call): {', '.join(visited)}")

        # Only include last 3 summaries, each truncated
        summaries = state.get("tool_summaries", [])
        for s in summaries[-3:]:
            ctx.append(f"Result: {str(s)[:150]}")

        ctx.append(f"Turn {turns + 1}/{MAX_SUPERVISOR_TURNS}")

        # Extract user query — convert to plain HumanMessage for LLM
        user_text = _get_last_user_text(state)

        llm_input = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            SystemMessage(content="STATE:\n" + "\n".join(ctx)),
        ]
        if user_text:
            llm_input.append(HumanMessage(content=user_text))

        decision = llm.invoke(llm_input)
        chosen = decision.next_agent

        # Post-LLM guards
        if chosen == DATASET_FINDER and active_ds:
            chosen = FINISH
        if chosen != FINISH and chosen in visited_set:
            chosen = FINISH

        return {"next_agent": chosen, "supervisor_turns": turns + 1}

    return supervisor_node


# Synthesizer

SYNTHESIS_PROMPT = """Synthesize a final answer from the analysis results below.
Be concise. Cite numbers. If plots were generated, mention them (they will be attached).

IMPORTANT: At the end of your response, add a "Try next:" section with 2-3 suggested
follow-up queries the user can try. Make them specific to the dataset and results.
Format them as quoted strings the user can copy-paste, for example:
  Try next:
  - "Show cell types spatially"
  - "What is Snap25 expression across cell types?"
  - "Run spatial autocorrelation on Snap25"

RESULTS:
{results}"""


def create_synthesizer_node():
    llm = get_settings().get_llm()

    def synthesizer_node(state: SpatialChatState) -> dict:
        # Build results text from COMPACT summaries only
        summaries = state.get("tool_summaries", [])
        plot_ids = state.get("plot_ids", [])

        results_text = ""
        for i, s in enumerate(summaries):
            results_text += f"\n--- Result {i + 1} ---\n{str(s)[:300]}\n"
        if plot_ids:
            results_text += f"\nPlots generated: {len(plot_ids)}\n"
        if not results_text.strip():
            results_text = "No analysis results available."

        user_text = _get_last_user_text(state)

        llm_input = [SystemMessage(content=SYNTHESIS_PROMPT.format(results=results_text))]
        if user_text:
            llm_input.append(HumanMessage(content=user_text))

        resp = llm.invoke(llm_input)

        # Build multimodal AI message: text + plot images
        # This ensures LangGraph Studio / LangSmith can render the plots
        content_blocks = [{"type": "text", "text": resp.content}]

        for pid in plot_ids:
            b64 = get_plot_base64(pid)
            if b64:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        # If no plots, use plain text (simpler)
        if len(content_blocks) == 1:
            return {"messages": [AIMessage(content=resp.content)]}

        return {"messages": [AIMessage(content=content_blocks)]}

    return synthesizer_node
