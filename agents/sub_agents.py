"""
Sub-agent factories using bind_tools for robust multi-step tool calling.

Each sub-agent uses llm.bind_tools() for the ReAct loop with proper
ToolMessage responses (matching tool_call_id). Pydantic validation on
tool args is handled by LangChain's @tool decorator with args_schema.

Key design:
  - llm.bind_tools() provides proper tool_call_id in each AIMessage
  - After execution, ToolMessage(tool_call_id=...) is appended
  - This avoids the 400 error from dangling tool_calls
  - Tool results are compacted: no base64 plots, no large data payloads
  - Plot IDs are collected separately for the PlotStore
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)

from config.settings import get_settings
from tools.dataset_tools import DATASET_TOOLS
from tools.expression_tools import EXPRESSION_TOOLS
from tools.stats_tools import SPATIAL_STATS_TOOLS
from tools.neighbor_tools import NEIGHBORHOOD_TOOLS

logger = logging.getLogger(__name__)


# Prompts (kept short)

DATASET_FINDER_PROMPT = (
    "You find and load spatial transcriptomics datasets.\n"
    "1. search_datasets to find matches\n"
    "2. load_and_summarize_dataset to load the best match\n"
    "3. validate_gene ONLY if the user mentioned a specific gene\n"
    "Stop after loading. Be concise."
)

EXPLORATORY_PROMPT = (
    "You analyze gene expression and cell types in a loaded dataset.\n"
    "Available tools:\n"
    "- get_gene_expression_spatial: plot a gene on spatial coords\n"
    "- show_spatial_domains: plot any annotation spatially\n"
    "- compare_expression: compare gene between two groups\n"
    "- plot_celltype_spatial: plot cell types on spatial coords (auto-resolves column)\n"
    "- gene_expression_by_celltype: bar chart of gene expression per cell type\n"
    "Use dataset_id from context. Call only the tool needed.\n"
    "Keep response short — key stats and interpretation only."
)

SPATIAL_STATS_PROMPT = (
    "You run spatial statistics on a loaded dataset.\n"
    "Use dataset_id from context. Call only the test requested.\n"
    "Report statistic and p-value. Be concise."
)

NEIGHBORHOOD_PROMPT = (
    "You analyze cell neighborhoods in a loaded dataset.\n"
    "Use dataset_id from context. Call only the analysis requested.\n"
    "Be concise."
)


# Compact helpers

MAX_DATA_CHARS = 300


def _compact_data(data: Any) -> Any:
    """Shrink data dicts to prevent context bloat."""
    if data is None:
        return None
    if not isinstance(data, dict):
        s = str(data)
        return s[:MAX_DATA_CHARS] + "..." if len(s) > MAX_DATA_CHARS else s
    compact = {}
    for k, v in list(data.items())[:8]:
        s = str(v)
        compact[k] = v if len(s) < 100 else s[:100] + "..."
    return compact


def _parse_tool_output(raw: str | dict) -> dict:
    """Parse tool output JSON string into a dict."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"success": True, "message": str(raw)[:300]}


def _compact_for_context(parsed: dict) -> dict:
    """Create a compact version of tool output suitable for LLM context.

    Strips plot_id and large data — those are handled separately.
    """
    compact: dict[str, Any] = {
        "success": parsed.get("success", False),
        "message": str(parsed.get("message", ""))[:300],
    }
    if parsed.get("data"):
        compact["data"] = _compact_data(parsed["data"])
    if parsed.get("plot_id"):
        compact["plot_generated"] = True  # Flag only, no base64
    if parsed.get("error"):
        compact["error"] = str(parsed["error"])[:200]
    return compact


# Sub-agent runner

def build_sub_agent(llm, tools: list, system_prompt: str, max_steps: int = 3):
    """
    Build a sub-agent function using bind_tools for multi-step tool calling.

    The agent:
      1. Uses llm.bind_tools() so AIMessage contains tool_calls with IDs
      2. Executes each tool call and appends ToolMessage with matching tool_call_id
      3. Feeds compact results back to LLM for next step
      4. Collects plot_ids separately (never in LLM context)

    Returns a callable: run(messages) -> dict with keys:
      - summary: str (final AI text, max 500 chars)
      - tool_summaries: list[str] (compact text summaries)
      - plot_ids: list[str] (plot store references)
      - detected_dataset_id: str | None
    """

    # Bind tools to LLM — this makes the LLM produce AIMessage with tool_calls
    # that include proper tool_call_id for each call
    llm_with_tools = llm.bind_tools(tools)

    # Map: tool name -> tool function (for executing after LLM selects)
    name_to_tool: dict[str, Any] = {}
    for t in tools:
        name_to_tool[t.name] = t

    def run(input_messages: list) -> dict:
        messages = [SystemMessage(content=system_prompt)] + input_messages
        all_summaries: list[str] = []
        all_plot_ids: list[str] = []
        detected_dataset_id: str | None = None

        for step in range(max_steps):
            # Call LLM with tools bound
            try:
                ai_msg: AIMessage = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                all_summaries.append(f"LLM call failed: {str(e)[:150]}")
                break

            # Append the AI message to conversation history
            messages.append(ai_msg)

            # Check if LLM made any tool calls
            tool_calls = ai_msg.tool_calls
            if not tool_calls:
                # LLM responded with text only — it's done
                break

            # Execute each tool call and create proper ToolMessages
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = tc["id"]

                tool_fn = name_to_tool.get(tool_name)
                if tool_fn is None:
                    error_msg = f"Unknown tool: {tool_name}"
                    logger.warning(error_msg)
                    messages.append(ToolMessage(
                        content=json.dumps({"success": False, "error": error_msg}),
                        tool_call_id=tool_call_id,
                    ))
                    all_summaries.append(error_msg)
                    continue

                try:
                    raw_output = tool_fn.invoke(tool_args)
                    parsed = _parse_tool_output(raw_output)

                    # Collect plot IDs (kept out of LLM context)
                    if parsed.get("plot_id"):
                        all_plot_ids.append(parsed["plot_id"])

                    # Detect dataset_id from tool output
                    if isinstance(parsed.get("data"), dict):
                        did = parsed["data"].get("dataset_id")
                        if did:
                            detected_dataset_id = did

                    # Create compact context for LLM
                    compact = _compact_for_context(parsed)
                    all_summaries.append(compact["message"])

                    # Append ToolMessage with matching tool_call_id
                    messages.append(ToolMessage(
                        content=json.dumps(compact),
                        tool_call_id=tool_call_id,
                    ))

                except Exception as e:
                    error_msg = f"Tool '{tool_name}' error: {str(e)[:200]}"
                    all_summaries.append(error_msg)
                    messages.append(ToolMessage(
                        content=json.dumps({"success": False, "error": error_msg}),
                        tool_call_id=tool_call_id,
                    ))
                    logger.exception(f"Tool execution failed: {e}")

        # Get final AI summary
        summary = "Analysis step complete."
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                summary = str(msg.content)[:500]
                break

        return {
            "summary": summary,
            "tool_summaries": all_summaries,
            "plot_ids": all_plot_ids,
            "detected_dataset_id": detected_dataset_id,
        }

    return run


# Factory functions

def create_dataset_finder_agent():
    llm = get_settings().get_sub_agent_llm()
    return build_sub_agent(llm, DATASET_TOOLS, DATASET_FINDER_PROMPT)


def create_exploratory_agent():
    llm = get_settings().get_sub_agent_llm()
    return build_sub_agent(llm, EXPRESSION_TOOLS, EXPLORATORY_PROMPT)


def create_spatial_stats_agent():
    llm = get_settings().get_sub_agent_llm()
    return build_sub_agent(llm, SPATIAL_STATS_TOOLS, SPATIAL_STATS_PROMPT)


def create_neighborhood_agent():
    llm = get_settings().get_sub_agent_llm()
    return build_sub_agent(llm, NEIGHBORHOOD_TOOLS, NEIGHBORHOOD_PROMPT)
