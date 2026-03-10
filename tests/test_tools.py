"""
Unit tests for tools, base utilities, data cache, and catalog.

Uses a synthetic AnnData fixture to avoid downloading real data.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
import anndata as ad
from unittest.mock import patch, MagicMock

# Fixtures


@pytest.fixture
def synthetic_adata():
    """Create a small synthetic AnnData for tool tests."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 50
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    gene_names[0] = "Snap25"
    gene_names[1] = "Mbp"

    adata = ad.AnnData(
        X=rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32),
        obs={"cluster": [str(i % 3) for i in range(n_cells)]},
    )
    adata.var_names = gene_names
    adata.obsm["spatial"] = rng.uniform(0, 1000, size=(n_cells, 2))
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    return adata


@pytest.fixture
def mock_cache(synthetic_adata):
    """Mock the DatasetCache to return the synthetic data."""
    mock = MagicMock()
    mock.get.return_value = synthetic_adata
    mock.loaded_ids.return_value = ["test_dataset"]
    return mock


# Base utility tests


class TestPlotStore:
    def test_fig_to_plot_id_stores_and_retrieves(self):
        """fig_to_plot_id should store base64 and return a short ID."""
        import matplotlib.pyplot as plt
        from tools.base import fig_to_plot_id, get_plot_base64, clear_plot_store

        clear_plot_store()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        pid = fig_to_plot_id(fig)

        assert isinstance(pid, str)
        assert len(pid) == 8  # hex[:8]
        b64 = get_plot_base64(pid)
        assert b64 is not None
        assert len(b64) > 100  # Should be a real PNG base64

    def test_clear_plot_store(self):
        """clear_plot_store should remove all stored plots."""
        import matplotlib.pyplot as plt
        from tools.base import fig_to_plot_id, get_plot_base64, clear_plot_store

        fig, ax = plt.subplots()
        ax.plot([1, 2])
        pid = fig_to_plot_id(fig)
        clear_plot_store()
        assert get_plot_base64(pid) is None


class TestToolResult:
    def test_success_result(self):
        from tools.base import tool_result

        result = json.loads(tool_result(success=True, message="OK", data={"x": 1}))
        assert result["success"] is True
        assert result["message"] == "OK"
        assert result["data"]["x"] == 1

    def test_error_result(self):
        from tools.base import tool_result

        result = json.loads(tool_result(success=False, message="Fail", error="bad input"))
        assert result["success"] is False
        assert "error" in result

    def test_plot_id_not_base64(self):
        """tool_result should include plot_id, not plot_base64."""
        from tools.base import tool_result

        result = json.loads(tool_result(success=True, message="OK", plot_id="abc12345"))
        assert "plot_id" in result
        assert "plot_base64" not in result
        assert result["plot_id"] == "abc12345"


# Dataset tool tests


class TestDatasetTools:
    def test_search_datasets_finds_match(self):
        from tools.dataset_tools import search_datasets

        result = json.loads(search_datasets.invoke({"query": "mouse brain"}))
        assert result["success"] is True
        assert "mouse_brain" in str(result.get("data", {}).get("matches", []))

    def test_search_datasets_no_match(self):
        from tools.dataset_tools import search_datasets

        result = json.loads(search_datasets.invoke({"query": "nonexistent_xyz"}))
        assert result["success"] is False

    def test_validate_gene_exact(self, mock_cache):
        from tools.dataset_tools import validate_gene

        with patch("tools.dataset_tools.get_cache", return_value=mock_cache):
            result = json.loads(validate_gene.invoke({
                "dataset_id": "test", "gene_name": "Snap25"
            }))
            assert result["success"] is True
            assert result["data"]["gene"] == "Snap25"

    def test_validate_gene_case_insensitive(self, mock_cache):
        from tools.dataset_tools import validate_gene

        with patch("tools.dataset_tools.get_cache", return_value=mock_cache):
            result = json.loads(validate_gene.invoke({
                "dataset_id": "test", "gene_name": "snap25"
            }))
            assert result["success"] is True
            assert result["data"]["gene"] == "Snap25"

    def test_validate_gene_not_found(self, mock_cache):
        from tools.dataset_tools import validate_gene

        with patch("tools.dataset_tools.get_cache", return_value=mock_cache):
            result = json.loads(validate_gene.invoke({
                "dataset_id": "test", "gene_name": "NONEXISTENT"
            }))
            assert result["success"] is False


# Expression tool tests


class TestExpressionTools:
    def test_gene_expression_returns_plot_id(self, mock_cache):
        from tools.expression_tools import get_gene_expression_spatial
        from tools.base import clear_plot_store, get_plot_base64

        clear_plot_store()
        with patch("tools.expression_tools.get_cache", return_value=mock_cache):
            result = json.loads(get_gene_expression_spatial.invoke({
                "dataset_id": "test", "gene": "Snap25"
            }))

        assert result["success"] is True
        assert "plot_id" in result
        assert "plot_base64" not in result  # KEY: no base64 in result
        # Verify plot is in the store
        assert get_plot_base64(result["plot_id"]) is not None

    def test_gene_expression_stats_compact(self, mock_cache):
        from tools.expression_tools import get_gene_expression_spatial

        with patch("tools.expression_tools.get_cache", return_value=mock_cache):
            result = json.loads(get_gene_expression_spatial.invoke({
                "dataset_id": "test", "gene": "Snap25"
            }))

        # Verify data is compact stats, not raw expression
        data = result["data"]
        assert "gene" in data
        assert "mean" in data
        assert "pct_expressing" in data
        # Should NOT contain raw arrays
        assert not any(isinstance(v, list) and len(v) > 10 for v in data.values())

    def test_gene_not_in_dataset(self, mock_cache):
        from tools.expression_tools import get_gene_expression_spatial

        with patch("tools.expression_tools.get_cache", return_value=mock_cache):
            result = json.loads(get_gene_expression_spatial.invoke({
                "dataset_id": "test", "gene": "NONEXISTENT"
            }))
        assert result["success"] is False


# Data cache tests


class TestDatasetCache:
    def test_cache_put_get(self, synthetic_adata):
        from data.loaders import DatasetCache

        cache = DatasetCache(max_size=2)
        cache.put("ds1", synthetic_adata)
        assert cache.get("ds1") is synthetic_adata

    def test_cache_lru_eviction(self, synthetic_adata):
        from data.loaders import DatasetCache

        cache = DatasetCache(max_size=2)
        cache.put("ds1", synthetic_adata)
        cache.put("ds2", synthetic_adata)
        cache.put("ds3", synthetic_adata)  # Should evict ds1
        assert cache.get("ds1") is None
        assert cache.get("ds2") is not None
        assert cache.get("ds3") is not None

    def test_cache_lru_access_refreshes(self, synthetic_adata):
        from data.loaders import DatasetCache

        cache = DatasetCache(max_size=2)
        cache.put("ds1", synthetic_adata)
        cache.put("ds2", synthetic_adata)
        cache.get("ds1")  # Refresh ds1
        cache.put("ds3", synthetic_adata)  # Should evict ds2 (not ds1)
        assert cache.get("ds1") is not None
        assert cache.get("ds2") is None

    def test_loaded_ids(self, synthetic_adata):
        from data.loaders import DatasetCache

        cache = DatasetCache(max_size=3)
        cache.put("a", synthetic_adata)
        cache.put("b", synthetic_adata)
        assert set(cache.loaded_ids()) == {"a", "b"}


# Catalog tests


class TestCatalog:
    def test_load_catalog(self):
        from data.loaders import load_catalog

        catalog = load_catalog()
        assert isinstance(catalog, dict)
        assert len(catalog) >= 1
        for did, info in catalog.items():
            assert "display_name" in info
            assert "species" in info


# Routing tests


class TestRouting:
    def test_route_from_supervisor_valid(self):
        from graph import route_from_supervisor

        state = {"next_agent": "exploratory"}
        assert route_from_supervisor(state) == "exploratory"

    def test_route_from_supervisor_finish(self):
        from graph import route_from_supervisor

        state = {"next_agent": "FINISH"}
        assert route_from_supervisor(state) == "synthesizer"

    def test_route_from_supervisor_invalid(self):
        from graph import route_from_supervisor

        state = {"next_agent": "unknown_agent"}
        assert route_from_supervisor(state) == "synthesizer"
