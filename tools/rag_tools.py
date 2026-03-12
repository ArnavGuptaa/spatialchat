"""
RAG (Retrieval-Augmented Generation) tools for the LLM agents.

These tools query the ChromaDB vector store and provide agents with
contextual information about genes and cell types, enabling better
reasoning and hypothesis generation about spatial transcriptomics data.

Three capabilities:
  1. Semantic gene search — find genes matching a natural language description
  2. Expression-profile similarity — find genes with similar expression patterns
  3. Cell type search — find cell types and their marker genes
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

from tools.base import tool_result
from data.loaders import get_cache
from data.metadata_store import (
    search_genes_semantic,
    find_expression_similar_genes,
    search_celltypes_semantic,
)


# ---------------------------------------------------------------------------
# Pydantic args schemas
# ---------------------------------------------------------------------------

class RAGGeneQueryArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID like 'mouse_brain_seqfish'")
    query: str = Field(
        description=(
            "Natural language query about genes, e.g. "
            "'genes highly expressed in endothelium', "
            "'transcription factors', 'Wnt signalling genes'"
        )
    )

    @field_validator("dataset_id", "query")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


class RAGSimilarGenesArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(description="Reference gene symbol (e.g. 'Snap25')")

    @field_validator("dataset_id", "gene")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


class RAGCelltypeQueryArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    query: str = Field(
        description=(
            "Natural language query about cell types, e.g. "
            "'endothelium', 'neural crest', 'progenitor cells'"
        )
    )

    @field_validator("dataset_id", "query")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(args_schema=RAGGeneQueryArgs)
def rag_query_genes(dataset_id: str, query: str) -> str:
    """Search the vector database for genes matching a biological description.

    Uses ChromaDB RAG retrieval to find genes whose expression metadata
    matches the query. Call this to discover candidate genes before
    visualising with get_gene_expression_spatial or compare_expression.

    Examples:
      - "genes highly expressed in endothelium"
      - "Hox transcription factors"
      - "Wnt signalling pathway"
    """
    cache = get_cache()
    if cache.get(dataset_id) is None:
        return tool_result(
            success=False,
            message=f"Dataset '{dataset_id}' not loaded.",
            error="Call load_and_summarize_dataset first.",
        )

    try:
        results = search_genes_semantic(dataset_id, query, n=10)
    except Exception as e:
        return tool_result(
            success=False,
            message=f"RAG search failed: {str(e)[:200]}",
            error="ChromaDB query error. Try a simpler query.",
        )

    if not results:
        return tool_result(
            success=False,
            message=f"No genes matched query: '{query}'",
            error="Try different keywords or a broader query.",
        )

    lines = [f"RAG results for '{query}' ({len(results)} genes):"]
    gene_names = []
    for i, r in enumerate(results, 1):
        gene_names.append(r["gene_symbol"])
        lines.append(
            f"  {i}. {r['gene_symbol']}: "
            f"mean={r.get('mean_expression', 0):.4f}, "
            f"{r.get('pct_expressing', 0):.1f}% expressing"
        )

    return tool_result(
        success=True,
        message="\n".join(lines),
        data={"query": query, "matches": gene_names},
    )


@tool(args_schema=RAGSimilarGenesArgs)
def rag_find_similar_genes(dataset_id: str, gene: str) -> str:
    """Find genes with similar expression profiles using vector similarity search.

    Queries the ChromaDB vector store for genes whose expression statistics
    (mean, median, % expressing, per-celltype means) are closest to the
    reference gene in cosine distance. This is useful for discovering
    co-regulated or functionally related genes.

    Returns up to 10 similar genes ranked by distance.
    """
    cache = get_cache()
    adata = cache.get(dataset_id)
    if adata is None:
        return tool_result(
            success=False,
            message=f"Dataset '{dataset_id}' not loaded.",
            error="Call load_and_summarize_dataset first.",
        )

    # Verify the gene exists
    if gene not in adata.var_names:
        return tool_result(
            success=False,
            message=f"Gene '{gene}' not found in dataset.",
            error="Use validate_gene to check the correct gene name first.",
        )

    try:
        results = find_expression_similar_genes(dataset_id, gene, n=10)
    except Exception as e:
        return tool_result(
            success=False,
            message=f"Similarity search failed: {str(e)[:200]}",
            error="ChromaDB vector query error.",
        )

    if not results:
        return tool_result(
            success=False,
            message=f"No similar genes found for '{gene}'.",
            error="The dataset may not be indexed in the vector store yet.",
        )

    lines = [f"Genes most similar to '{gene}' (expression profile):"]
    gene_names = []
    for i, r in enumerate(results, 1):
        gene_names.append(r["gene_symbol"])
        lines.append(
            f"  {i}. {r['gene_symbol']} "
            f"(distance={r.get('distance', 0):.4f}): "
            f"mean={r.get('mean_expression', 0):.4f}, "
            f"{r.get('pct_expressing', 0):.1f}% expressing"
        )

    return tool_result(
        success=True,
        message="\n".join(lines),
        data={"reference_gene": gene, "similar_genes": gene_names},
    )


@tool(args_schema=RAGCelltypeQueryArgs)
def rag_query_celltypes(dataset_id: str, query: str) -> str:
    """Search the vector database for cell types matching a description.

    Returns matching cell types with their cell counts and top marker genes.
    Use this to discover cell populations before running expression analysis.

    Examples:
      - "neural crest"
      - "endothelium"
      - "progenitor cells"
    """
    cache = get_cache()
    if cache.get(dataset_id) is None:
        return tool_result(
            success=False,
            message=f"Dataset '{dataset_id}' not loaded.",
            error="Call load_and_summarize_dataset first.",
        )

    try:
        results = search_celltypes_semantic(dataset_id, query, n=5)
    except Exception as e:
        return tool_result(
            success=False,
            message=f"Cell type search failed: {str(e)[:200]}",
            error="ChromaDB query error.",
        )

    if not results:
        return tool_result(
            success=False,
            message=f"No cell types matched: '{query}'",
            error="Try different keywords.",
        )

    lines = [f"Cell types matching '{query}':"]
    ct_names = []
    for i, r in enumerate(results, 1):
        ct_names.append(r["celltype"])
        markers = json.loads(r.get("marker_genes", "[]"))
        marker_str = ", ".join(markers[:5]) if markers else "N/A"
        lines.append(
            f"  {i}. {r['celltype']}: "
            f"{r.get('n_cells', 0)} cells, "
            f"markers: {marker_str}"
        )

    return tool_result(
        success=True,
        message="\n".join(lines),
        data={"query": query, "celltypes": ct_names},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RAG_TOOLS = [rag_query_genes, rag_find_similar_genes, rag_query_celltypes]
