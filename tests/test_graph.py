"""
Integration tests for the SpatialChat graph.

These tests require an LLM API key to be configured.
Skip with: pytest tests/test_graph.py -k "not integration"
"""

from __future__ import annotations

import os
import pytest


requires_api_key = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="No LLM API key configured",
)


@requires_api_key
class TestGraphIntegration:
    def test_list_datasets(self):
        """The agent should be able to list available datasets."""
        from graph import chat

        result = chat("What datasets are available?")
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 10

    def test_chat_returns_active_dataset_id(self):
        """After loading a dataset, active_dataset_id should be set."""
        from graph import chat

        result = chat("Load the mouse brain visium dataset")
        assert result.get("active_dataset_id") is not None

    def test_chat_passes_dataset_id(self):
        """Passing active_dataset_id should allow analysis without re-loading."""
        from graph import chat

        # First load the dataset
        result1 = chat("Load the mouse brain visium dataset")
        ds_id = result1.get("active_dataset_id")

        if ds_id:
            # Then analyze with dataset already loaded
            result2 = chat(
                "Show Snap25 expression",
                active_dataset_id=ds_id,
            )
            assert "response" in result2
            # Should have generated a plot
            assert len(result2.get("plots", [])) > 0

    def test_no_base64_in_response_text(self):
        """Base64 plot data should NEVER appear in response text."""
        from graph import chat

        result = chat("Load mouse brain visium and show Snap25 expression")
        response = result["response"]
        # Base64 PNG starts with 'iVBOR'
        assert "iVBOR" not in response
        # Plots should be in the plots list, not the text
        if result.get("plots"):
            assert all(isinstance(p, str) and len(p) > 100 for p in result["plots"])

    def test_graceful_not_found(self):
        """Unknown dataset should return a helpful error, not crash."""
        from graph import chat

        result = chat("Load the unicorn brain dataset")
        assert "response" in result
        assert isinstance(result["response"], str)
