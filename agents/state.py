"""
Shared state schema for the LangGraph.

Key design decisions:
  - messages: uses add_messages for proper dict-to-BaseMessage conversion + history
  - tool_summaries, plot_ids, visited_agents: NO reducer — these are per-invocation
    and get reset by the 'reset' entry node each turn to avoid accumulation
  - active_dataset_id: persists across turns (no reducer = last-write-wins)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SpatialChatState(TypedDict):
    # Core conversation messages — add_messages handles:
    #   1. dict-to-BaseMessage conversion (LangGraph Studio sends dicts)
    #   2. Message history accumulation across turns
    messages: Annotated[list[BaseMessage], add_messages]

    # Dataset tracking — persists across turns (last-write-wins, no reducer)
    active_dataset_id: Optional[str]

    # ── Per-invocation fields (reset by 'reset' node each turn) ──
    # No reducer = each node REPLACES the value.
    # Sub-agent nodes manually concat: existing + new

    # Routing decision from supervisor
    next_agent: Optional[str]

    # Compact tool output summaries (short strings, NO base64 or large data)
    tool_summaries: list[str]

    # Plot references (short IDs pointing to PlotStore, NOT base64 strings)
    plot_ids: list[str]

    # Loop prevention
    supervisor_turns: int
    visited_agents: list[str]
    error_count: int


# Agent name constants
DATASET_FINDER = "dataset_finder"
EXPLORATORY = "exploratory"
SPATIAL_STATS = "spatial_stats"
NEIGHBORHOOD = "neighborhood"
FINISH = "FINISH"

# Safety limits
MAX_RETRIES = 3
MAX_SUPERVISOR_TURNS = 5
