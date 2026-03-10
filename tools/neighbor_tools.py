"""
Cell neighborhood and communication analysis tools.

Plots stored in PlotStore — only IDs in tool results.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from tools.base import fig_to_plot_id, tool_result
from data.loaders import get_cache


def _ensure_neighbors(adata):
    if "spatial_neighbors" not in adata.uns:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata, coord_type="generic")


class AnnotationArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    annotation_key: str = Field(description="Column with cell type labels")


@tool(args_schema=AnnotationArgs)
def neighborhood_enrichment(dataset_id: str, annotation_key: str) -> str:
    """Test which cell type pairs are spatially enriched as neighbors."""
    import squidpy as sq

    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool tests which cell type pairs are spatially enriched as neighbors, "
                                 "revealing which types tend to co-localize in tissue.")
    if annotation_key not in adata.obs.columns:
        avail = list(adata.obs.columns)[:10]
        return tool_result(success=False, message=f"Annotation '{annotation_key}' not found.",
                           error=f"Available columns: {avail}. "
                                 f"This tool can analyze neighborhood enrichment for any categorical annotation.")

    _ensure_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=annotation_key)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    try:
        sq.pl.nhood_enrichment(adata, cluster_key=annotation_key, ax=ax)
    except Exception:
        ax.text(0.5, 0.5, "Plot unavailable", ha="center", va="center")
    pid = fig_to_plot_id(fig)

    return tool_result(success=True,
                       message=f"Neighborhood enrichment done for '{annotation_key}'.",
                       plot_id=pid)


@tool(args_schema=AnnotationArgs)
def interaction_matrix(dataset_id: str, annotation_key: str) -> str:
    """Compute cell-cell contact frequency matrix from spatial graph."""
    from scipy import sparse

    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool computes a cell-cell contact frequency matrix "
                                 "showing how often each pair of cell types are spatial neighbors.")
    if annotation_key not in adata.obs.columns:
        avail = list(adata.obs.columns)[:10]
        return tool_result(success=False, message=f"Annotation '{annotation_key}' not found.",
                           error=f"Available columns: {avail}. "
                                 f"This tool can compute interaction matrices for any categorical annotation.")

    _ensure_neighbors(adata)
    conn = adata.obsp.get("spatial_connectivities")
    if conn is None:
        return tool_result(success=False, message="No spatial connectivity matrix.",
                           error="Spatial neighbors may not have been computed correctly. "
                                 "Try neighborhood_enrichment instead, which handles neighbor computation.")

    labels = adata.obs[annotation_key].values
    categories = list(labels.cat.categories) if hasattr(labels, "cat") else sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(categories)}
    n = len(categories)
    matrix = np.zeros((n, n))

    rows, cols = conn.nonzero() if sparse.issparse(conn) else np.nonzero(conn)
    for r, c in zip(rows, cols):
        i, j = label_to_idx.get(labels[r]), label_to_idx.get(labels[c])
        if i is not None and j is not None:
            matrix[i, j] += 1

    total = matrix.sum()
    matrix_norm = matrix / total if total > 0 else matrix

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.imshow(matrix_norm, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_title(f"Cell-cell contacts: {annotation_key}")
    plt.tight_layout()
    pid = fig_to_plot_id(fig)

    return tool_result(success=True,
                       message=f"Interaction matrix: {n} types, {int(total)} contacts.",
                       plot_id=pid)


# Registry

NEIGHBORHOOD_TOOLS = [neighborhood_enrichment, interaction_matrix]
