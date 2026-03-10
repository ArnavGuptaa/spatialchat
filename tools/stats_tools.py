"""
Spatial statistics tools: Moran's I, co-occurrence.

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
from data.metadata_store import find_similar_genes


def _ensure_neighbors(adata):
    if "spatial_neighbors" not in adata.uns:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata, coord_type="generic")


# Args schemas

class AutocorrArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(default="", description="Gene to test. If empty, returns top spatially variable genes.")


class CoOccurrenceArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    annotation_key: str = Field(description="Column with cluster labels")


# Tools

@tool(args_schema=AutocorrArgs)
def spatial_autocorrelation(dataset_id: str, gene: str = "") -> str:
    """Compute Moran's I spatial autocorrelation for a gene or find top spatially variable genes."""
    import squidpy as sq

    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool computes Moran's I spatial autocorrelation — "
                                 "it can test a specific gene or find top spatially variable genes.")

    _ensure_neighbors(adata)

    if "moranI" not in adata.uns:
        genes_subset = None
        if adata.shape[1] > 5000:
            import scanpy as sc
            if "highly_variable" not in adata.var:
                sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
            genes_subset = adata.var_names[adata.var["highly_variable"]].tolist()
        sq.gr.spatial_autocorr(adata, mode="moran", genes=genes_subset, n_perms=100)

    df = adata.uns["moranI"]

    if gene:
        if gene not in df.index:
            suggestions = find_similar_genes(dataset_id, gene, n=5)
            hint = f" Did you mean: {suggestions}?" if suggestions else ""
            return tool_result(success=False,
                               message=f"Gene '{gene}' not in results.{hint}",
                               error=f"Gene may not be in the tested subset. "
                                     f"Try without a gene argument to find top spatially variable genes, "
                                     f"or validate the gene name first.")
        row = df.loc[gene]
        I = float(row["I"])
        pval = float(row.get("pval_norm", row.get("pval_z_sim", 0)))

        from scipy import sparse as sp
        expr = adata[:, gene].X
        if sp.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        spatial = adata.obsm["spatial"]
        sc = ax.scatter(spatial[:, 0], spatial[:, 1], c=expr, cmap="viridis", s=8)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"{gene}: Moran's I={I:.4f}, p={pval:.2e}")
        ax.set_aspect("equal")
        pid = fig_to_plot_id(fig)

        return tool_result(
            success=True,
            message=f"{gene}: Moran's I={I:.4f}, p={pval:.2e}",
            data={"gene": gene, "morans_I": I, "pvalue": pval},
            plot_id=pid,
        )
    else:
        # Return only top 10 genes — compact summary
        top = df.sort_values("I", ascending=False).head(10)
        genes_list = []
        for g, row in top.iterrows():
            genes_list.append({
                "gene": g,
                "I": round(float(row["I"]), 4),
                "pvalue": float(row.get("pval_norm", 0)),
            })
        return tool_result(
            success=True,
            message=f"Top {len(genes_list)} spatially variable genes found.",
            data={"top_genes": genes_list},
        )


@tool(args_schema=CoOccurrenceArgs)
def co_occurrence(dataset_id: str, annotation_key: str) -> str:
    """Compute spatial co-occurrence of cell types / clusters."""
    import squidpy as sq

    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool computes spatial co-occurrence of cell types/clusters, "
                                 "showing which types tend to appear near each other in tissue.")
    if annotation_key not in adata.obs.columns:
        avail = list(adata.obs.columns)[:10]
        return tool_result(success=False, message=f"Annotation '{annotation_key}' not found.",
                           error=f"Available columns: {avail}. "
                                 f"This tool analyzes spatial co-occurrence for any categorical annotation.")

    sq.gr.co_occurrence(adata, cluster_key=annotation_key)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    try:
        sq.pl.co_occurrence(adata, cluster_key=annotation_key, ax=ax)
    except Exception:
        ax.text(0.5, 0.5, "Co-occurrence plot unavailable", ha="center", va="center")
    pid = fig_to_plot_id(fig)

    return tool_result(success=True, message=f"Co-occurrence analysis done for '{annotation_key}'.",
                       plot_id=pid)


# Registry

SPATIAL_STATS_TOOLS = [spatial_autocorrelation, co_occurrence]
