"""
Tool utilities: plot store and result formatting.

Pydantic validation on tool args is handled by LangChain's @tool decorator.
Sub-agents use llm.bind_tools() with proper ToolMessage responses.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Plot Store
# Keeps base64 image data OUT of LLM context entirely.
# Tools store plots here and return only a short plot_id string.

_plot_store: dict[str, str] = {}


def fig_to_plot_id(fig: plt.Figure, dpi: int = 150) -> str:
    """Convert a matplotlib Figure to base64 PNG, store it, return a short ID."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    plot_id = uuid.uuid4().hex[:8]
    _plot_store[plot_id] = b64
    logger.debug(f"Stored plot {plot_id} ({len(b64)} chars)")
    return plot_id


def get_plot_base64(plot_id: str) -> str | None:
    """Retrieve a stored plot's base64 string by ID."""
    return _plot_store.get(plot_id)


def get_all_plot_ids() -> list[str]:
    """Return all stored plot IDs."""
    return list(_plot_store.keys())


def clear_plot_store() -> None:
    """Clear all stored plots (call between sessions)."""
    _plot_store.clear()


# Tool Result Formatting

def tool_result(
    *,
    success: bool = True,
    message: str,
    data: dict[str, Any] | None = None,
    plot_id: str | None = None,
    error: str | None = None,
) -> str:
    """
    Build a consistent JSON string for tool returns.

    Key design: plots are referenced by ID (not embedded as base64).
    This keeps tool results small enough for LLM context windows.
    """
    result: dict[str, Any] = {"success": success, "message": message}
    if data is not None:
        result["data"] = data
    if plot_id is not None:
        result["plot_id"] = plot_id
    if error is not None:
        result["error"] = error
    return json.dumps(result)
