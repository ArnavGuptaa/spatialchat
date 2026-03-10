"""
Gene expression exploration and visualization tools.

Key design: plots are stored in PlotStore and referenced by ID.
NO base64 in tool results — keeps LLM context small.
NO full expression arrays in output — only summary stats.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from tools.base import fig_to_plot_id, tool_result
from data.loaders import get_cache, get_celltype_column
from data.metadata_store import find_similar_genes


def _get_expr(adata, gene: str) -> np.ndarray:
    """Extract expression values for a gene, handling sparse matrices."""
    expr = adata[:, gene].X
    if sparse.issparse(expr):
        expr = expr.toarray()
    return expr.flatten()


# ── Args schemas ──────────────────────────────────────────────

class SpatialExprArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(description="Gene symbol (case-sensitive — call validate_gene first)")


class SpatialDomainsArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    annotation_key: str = Field(description="Column in obs like 'cluster' or 'celltype_mapped_refined'")


class CompareExprArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(description="Gene symbol")
    annotation_key: str = Field(description="Column containing groups")
    group1: str = Field(description="First group label")
    group2: str = Field(description="Second group label")


class CelltypeSpatialArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")


class GeneExprByCelltypeArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(description="Gene symbol (case-sensitive — call validate_gene first)")


# ── Tools ─────────────────────────────────────────────────────

@tool(args_schema=SpatialExprArgs)
def get_gene_expression_spatial(dataset_id: str, gene: str) -> str:
    """Plot gene expression on spatial coordinates and return summary stats."""
    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool plots gene expression on spatial coordinates.")
    if gene not in adata.var_names:
        suggestions = find_similar_genes(dataset_id, gene, n=5)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        return tool_result(success=False,
                           message=f"Gene '{gene}' not in dataset.{hint}",
                           error=f"Call validate_gene first.{' Suggestions: ' + str(suggestions) if suggestions else ''} "
                                 f"This tool can plot any gene's spatial expression pattern.")

    expr = _get_expr(adata, gene)

    # Compute SUMMARY stats only — never return raw expression arrays
    stats = {
        "gene": gene,
        "mean": round(float(np.mean(expr)), 4),
        "median": round(float(np.median(expr)), 4),
        "pct_expressing": round(float((expr > 0).mean() * 100), 1),
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    spatial = adata.obsm["spatial"]
    sc = ax.scatter(spatial[:, 0], spatial[:, 1], c=expr, cmap="viridis", s=8, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=f"{gene} expression")
    ax.set_title(f"{gene} — spatial expression")
    ax.set_aspect("equal")
    pid = fig_to_plot_id(fig)

    return tool_result(
        success=True,
        message=f"{gene}: mean={stats['mean']}, {stats['pct_expressing']}% expressing",
        data=stats,
        plot_id=pid,
    )


@tool(args_schema=SpatialDomainsArgs)
def show_spatial_domains(dataset_id: str, annotation_key: str) -> str:
    """Visualize spatial domains / clusters on tissue coordinates."""
    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first.")
    if annotation_key not in adata.obs.columns:
        avail = list(adata.obs.columns)[:10]
        return tool_result(success=False, message=f"Annotation '{annotation_key}' not found.",
                           error=f"Available columns: {avail}. "
                                 f"This tool can visualize any categorical annotation column "
                                 f"on spatial coordinates (e.g. clusters, cell types, regions).")

    cats = adata.obs[annotation_key]
    unique_vals = cats.cat.categories if hasattr(cats, "cat") else sorted(cats.unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    spatial = adata.obsm["spatial"]
    for i, val in enumerate(unique_vals):
        mask = cats == val
        ax.scatter(spatial[mask, 0], spatial[mask, 1], c=[colors[i]], s=8, alpha=0.7, label=str(val))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(f"Spatial domains: {annotation_key}")
    ax.set_aspect("equal")
    plt.tight_layout()
    pid = fig_to_plot_id(fig)

    return tool_result(success=True, message=f"{len(unique_vals)} domains shown.",
                       plot_id=pid)


@tool(args_schema=CompareExprArgs)
def compare_expression(dataset_id: str, gene: str, annotation_key: str,
                       group1: str, group2: str) -> str:
    """Compare gene expression between two groups (Mann-Whitney U test)."""
    from scipy.stats import mannwhitneyu

    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool compares gene expression between two groups "
                                 "using a Mann-Whitney U test with violin plots.")
    if gene not in adata.var_names:
        suggestions = find_similar_genes(dataset_id, gene, n=5)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        return tool_result(success=False, message=f"Gene '{gene}' not found.{hint}",
                           error=f"Call validate_gene first. "
                                 f"This tool can compare expression of any gene between two groups.")
    if annotation_key not in adata.obs.columns:
        avail = list(adata.obs.columns)[:10]
        return tool_result(success=False, message=f"Annotation '{annotation_key}' not found.",
                           error=f"Available columns: {avail}. "
                                 f"This tool compares gene expression between any two groups in an annotation column.")

    mask1 = adata.obs[annotation_key].astype(str) == str(group1)
    mask2 = adata.obs[annotation_key].astype(str) == str(group2)
    if mask1.sum() == 0 or mask2.sum() == 0:
        avail = adata.obs[annotation_key].unique().tolist()[:10]
        return tool_result(success=False, message="One or both groups empty.",
                           error=f"Available groups: {avail}. Specify two valid group labels from this list.")

    expr = _get_expr(adata, gene)
    e1, e2 = expr[mask1], expr[mask2]
    stat, pval = mannwhitneyu(e1, e2, alternative="two-sided")
    m1, m2 = float(np.mean(e1)), float(np.mean(e2))
    lfc = float(np.log2(m1 + 1) - np.log2(m2 + 1))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.violinplot([e1, e2], positions=[0, 1], showmeans=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([str(group1), str(group2)])
    ax.set_ylabel(f"{gene} expression")
    ax.set_title(f"{gene}: {group1} vs {group2}\np={pval:.2e}, log2FC={lfc:.2f}")
    pid = fig_to_plot_id(fig)

    return tool_result(
        success=True,
        message=f"{gene}: {group1} mean={m1:.3f} vs {group2} mean={m2:.3f}, p={pval:.2e}",
        data={"gene": gene, "pvalue": float(pval), "log2fc": lfc, "significant": pval < 0.05},
        plot_id=pid,
    )


@tool(args_schema=CelltypeSpatialArgs)
def plot_celltype_spatial(dataset_id: str) -> str:
    """Plot cell types on spatial coordinates. Auto-resolves the celltype column from the catalog."""
    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool plots cell types on spatial coordinates, "
                                 "showing where each cell type is located in the tissue.")

    celltype_col = get_celltype_column(dataset_id)
    if not celltype_col:
        avail_cols = list(adata.obs.columns)[:10] if adata is not None else []
        return tool_result(success=False,
                           message="No celltype column configured for this dataset.",
                           error=f"Check catalog.json — 'celltype' key is missing. "
                                 f"Available obs columns: {avail_cols}. "
                                 f"You can use show_spatial_domains with any annotation column instead.")
    if celltype_col not in adata.obs.columns:
        avail = list(adata.obs.columns)[:10]
        return tool_result(success=False,
                           message=f"Celltype column '{celltype_col}' not found in obs.",
                           error=f"Available columns: {avail}. "
                                 f"Use show_spatial_domains to visualize any of these columns spatially.")

    cats = adata.obs[celltype_col]
    unique_vals = cats.cat.categories if hasattr(cats, "cat") else sorted(cats.unique())
    n_types = len(unique_vals)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_types, 20)))
    if n_types > 20:
        # Extend with tab20b for > 20 categories
        extra = plt.cm.tab20b(np.linspace(0, 1, n_types - 20))
        colors = np.vstack([colors, extra])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    spatial = adata.obsm["spatial"]
    for i, val in enumerate(unique_vals):
        mask = cats == val
        ax.scatter(spatial[mask, 0], spatial[mask, 1],
                   c=[colors[i]], s=6, alpha=0.7, label=str(val))

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
              ncol=1 if n_types <= 15 else 2, markerscale=2)
    ax.set_title(f"Cell types in spatial coordinates ({n_types} types)")
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    plt.tight_layout()
    pid = fig_to_plot_id(fig)

    # Compute cell type counts for summary
    counts = cats.value_counts()
    top3 = counts.head(3)
    summary_lines = [f"{ct}: {n} cells ({n/len(cats)*100:.1f}%)" for ct, n in top3.items()]

    return tool_result(
        success=True,
        message=f"{n_types} cell types plotted. Top 3: {'; '.join(summary_lines)}",
        data={"n_celltypes": n_types, "celltype_column": celltype_col,
              "top_celltypes": {str(k): int(v) for k, v in top3.items()}},
        plot_id=pid,
    )


@tool(args_schema=GeneExprByCelltypeArgs)
def gene_expression_by_celltype(dataset_id: str, gene: str) -> str:
    """Compute and plot mean gene expression across cell types (bar chart)."""
    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(success=False, message="Dataset not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "This tool creates a bar chart of mean gene expression across cell types.")
    if gene not in adata.var_names:
        suggestions = find_similar_genes(dataset_id, gene, n=5)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        return tool_result(success=False, message=f"Gene '{gene}' not in dataset.{hint}",
                           error=f"Call validate_gene first.{' Suggestions: ' + str(suggestions) if suggestions else ''} "
                                 f"This tool can show expression of any gene across cell types as a bar chart.")

    celltype_col = get_celltype_column(dataset_id)
    if not celltype_col:
        return tool_result(success=False, message="No celltype column configured for this dataset.",
                           error="Check catalog.json — 'celltype' key is missing. "
                                 "You can still use get_gene_expression_spatial to see this gene's spatial pattern.")
    if celltype_col not in adata.obs.columns:
        return tool_result(success=False,
                           message=f"Celltype column '{celltype_col}' not found in obs.",
                           error="Use get_gene_expression_spatial to visualize this gene's spatial pattern instead.")

    expr = _get_expr(adata, gene)
    cats = adata.obs[celltype_col]

    # Compute mean expression per celltype
    import pandas as pd
    df = pd.DataFrame({"expression": expr, "celltype": cats.values})
    mean_expr = df.groupby("celltype", observed=True)["expression"].mean().sort_values(ascending=False)

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(mean_expr) * 0.35)))
    y_pos = range(len(mean_expr))
    ax.barh(y_pos, mean_expr.values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mean_expr.index, fontsize=8)
    ax.set_xlabel(f"Mean {gene} expression")
    ax.set_title(f"{gene} expression across cell types")
    ax.invert_yaxis()  # Highest at top
    plt.tight_layout()
    pid = fig_to_plot_id(fig)

    # Summary stats
    top3 = mean_expr.head(3)
    summary_lines = [f"{ct}: mean={v:.3f}" for ct, v in top3.items()]
    overall_mean = float(np.mean(expr))

    return tool_result(
        success=True,
        message=f"{gene} across {len(mean_expr)} cell types. Overall mean={overall_mean:.3f}. "
                f"Top 3: {'; '.join(summary_lines)}",
        data={
            "gene": gene,
            "overall_mean": round(overall_mean, 4),
            "n_celltypes": len(mean_expr),
            "top_celltypes": {str(k): round(float(v), 4) for k, v in top3.items()},
        },
        plot_id=pid,
    )


# ── Registry ──────────────────────────────────────────────────

EXPRESSION_TOOLS = [
    get_gene_expression_spatial,
    show_spatial_domains,
    compare_expression,
    plot_celltype_spatial,
    gene_expression_by_celltype,
]
