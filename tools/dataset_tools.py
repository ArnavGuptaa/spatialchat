"""
Dataset discovery, loading, and validation tools.

Lean set: search -> load -> validate_gene. That's the full workflow.
Tools return compact JSON — no large data payloads.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

from tools.base import tool_result
from data.loaders import load_catalog, load_dataset, get_cache
from data.metadata_store import find_similar_genes, get_gene_list, get_celltypes


# ── Pydantic args schemas ────────────────────────────────────

class SearchArgs(BaseModel):
    query: str = Field(description="Search text like 'mouse brain' or 'human visium'")

    @field_validator("query")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query must not be empty")
        return v.strip()


class DatasetIdArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID like 'mouse_brain_visium'")

    @field_validator("dataset_id")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("dataset_id must not be empty")
        return v.strip()


class ValidateGeneArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene_name: str = Field(description="Gene symbol like 'Snap25' or 'MYC'")

    @field_validator("dataset_id", "gene_name")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


# ── Tools ─────────────────────────────────────────────────────

@tool(args_schema=SearchArgs)
def search_datasets(query: str) -> str:
    """Search available spatial transcriptomics datasets by tissue, species, or technology."""
    catalog = load_catalog()
    tokens = query.lower().split()

    matches = []
    for did, info in catalog.items():
        text = f"{info.get('tissue','')} {info.get('species','')} {info.get('technology','')} {info.get('region','')} {did}".lower()
        score = sum(1 for t in tokens if t in text)
        if score > 0:
            matches.append({"id": did, "name": info["display_name"], "score": score})

    matches.sort(key=lambda m: m["score"], reverse=True)

    if matches:
        lines = [f"Found {len(matches)} dataset(s):"]
        for m in matches:
            lines.append(f"  - {m['id']}: {m['name']}")
        return tool_result(success=True, message="\n".join(lines),
                           data={"matches": [m["id"] for m in matches]})

    available = [f"{k}: {v['display_name']}" for k, v in catalog.items()]
    return tool_result(success=False, message=f"No matches for '{query}'.",
                       error=f"Available datasets: {'; '.join(available)}. "
                             f"I can search datasets by tissue, species, technology, or region. "
                             f"Try a broader search term.")


@tool(args_schema=DatasetIdArgs)
def load_and_summarize_dataset(dataset_id: str) -> str:
    """Load a dataset into memory. Call this before any analysis tools."""
    adata = load_dataset(dataset_id)
    # Return COMPACT summary — no full gene lists or expression data
    annotations = list(adata.obs.columns)
    return tool_result(
        success=True,
        message=(
            f"Loaded '{dataset_id}': {adata.shape[0]} cells, {adata.shape[1]} genes. "
            f"Annotations: {annotations[:10]}. Has spatial: {'spatial' in adata.obsm}."
        ),
        data={
            "dataset_id": dataset_id,
            "n_cells": adata.shape[0],
            "n_genes": adata.shape[1],
            "annotations": annotations[:10],
        },
    )


@tool(args_schema=ValidateGeneArgs)
def validate_gene(dataset_id: str, gene_name: str) -> str:
    """Check if a gene exists in a loaded dataset. Returns the correct name (case-sensitive).
    Uses fuzzy matching to suggest similar genes if not found."""
    cache = get_cache()
    adata = cache.get(dataset_id)
    if adata is None:
        return tool_result(success=False, message=f"Dataset '{dataset_id}' not loaded.",
                           error="Call load_and_summarize_dataset first. "
                                 "I can search for datasets, load them, and validate gene names.")

    if gene_name in adata.var_names:
        return tool_result(success=True, message=f"Gene '{gene_name}' found (exact match).",
                           data={"gene": gene_name})

    # Use metadata store for robust fuzzy matching
    suggestions = find_similar_genes(dataset_id, gene_name, n=5)
    if suggestions and suggestions[0].lower() == gene_name.lower():
        # Exact case-insensitive match found
        return tool_result(success=True,
                           message=f"Gene found as '{suggestions[0]}' (case match for '{gene_name}').",
                           data={"gene": suggestions[0]})

    # Fall back to direct adata search if metadata not available
    if not suggestions:
        lower = gene_name.lower()
        case_matches = [g for g in adata.var_names if g.lower() == lower]
        if case_matches:
            return tool_result(success=True,
                               message=f"Gene found as '{case_matches[0]}' (case match for '{gene_name}').",
                               data={"gene": case_matches[0]})
        partial = [g for g in adata.var_names if lower in g.lower()][:5]
        suggestions = partial

    if suggestions:
        return tool_result(
            success=False,
            message=f"Gene '{gene_name}' not found. Did you mean one of these? {suggestions}",
            data={"suggestions": suggestions},
            error=f"Similar genes: {suggestions}. Try calling validate_gene with one of these.")
    else:
        gene_count = adata.shape[1]
        return tool_result(
            success=False,
            message=f"Gene '{gene_name}' not found and no similar genes detected. "
                    f"This dataset has {gene_count} genes.",
            error=f"No similar genes found. Use load_and_summarize_dataset to see dataset info, "
                  f"or search for a different gene. I can validate gene names and find similar matches.")


# ── Registry ──────────────────────────────────────────────────

DATASET_TOOLS = [search_datasets, load_and_summarize_dataset, validate_gene]
